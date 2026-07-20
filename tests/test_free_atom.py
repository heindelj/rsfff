"""Free-atom limit: for 1-atom systems all SOAP features vanish, the constraint
pins q = Q, and the model reduces to E0 + E_elem(s, Q) + head(0, z(s, Q))."""

import torch

from conftest import make_batch, make_model


def _single_atoms_batch():
    # O, O-, H, H+ at well-separated positions (separate molecules anyway)
    return make_batch(
        [torch.zeros(1, 3), torch.ones(1, 3), 2 * torch.ones(1, 3), 3 * torch.ones(1, 3)],
        [torch.tensor([8]), torch.tensor([8]), torch.tensor([1]), torch.tensor([1])],
        [0.0, -1.0, 0.0, 1.0],
    )


def test_energy_reduces_to_embedding_functions():
    model = make_model()
    batch = _single_atoms_batch()
    out = model(batch)

    assert torch.allclose(out.charges, batch.total_charge, atol=1e-12)

    s = torch.tensor([1, 1, 0, 0])  # species idx: O=1, H=0 in sorted [1, 8]
    q = batch.total_charge
    z = model.charge_embedding(s, q)
    p0 = model.featurizer.feature_dims[0]
    expected = (
        model.energy_head(torch.zeros(4, p0), z)
        + model.reference_energies[s]
        + model.element_energy(s, q)
    )
    assert (out.energy - expected).abs().max() < 1e-12


def test_free_atom_alpha_isotropic_and_charge_dependent():
    model = make_model()
    out = model(_single_atoms_batch())
    alpha = out.alpha_head
    off = alpha - torch.diag_embed(alpha.diagonal(dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-12
    # O and O- (different q) must give different alphas through z alone
    assert (alpha[0] - alpha[1]).abs().max() > 1e-8
    # same element, same q, different position -> identical alpha
    batch = make_batch(
        [torch.zeros(1, 3), 7 * torch.ones(1, 3)],
        [torch.tensor([8]), torch.tensor([8])],
        [0.0, 0.0],
    )
    out2 = model(batch)
    assert (out2.alpha_head[0] - out2.alpha_head[1]).abs().max() < 1e-12


def test_dipole_of_isolated_ion_is_q_r():
    model = make_model()
    pos = torch.tensor([[1.0, 2.0, -3.0]])
    batch = make_batch([pos], [torch.tensor([1])], [1.0])
    out = model(batch)
    assert (out.dipole[0] - pos[0]).abs().max() < 1e-12
