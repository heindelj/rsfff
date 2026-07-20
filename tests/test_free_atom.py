"""Free-atom limit: for 1-atom systems all SOAP features vanish, the constraint
pins q = Q, and E = E0 + E_MLIP(0, s) + chi0 Q + 1/2 eta0 Q^2 with zero atomic
dipole and an isotropic atomic polarizability."""

import torch

from conftest import make_batch, make_model


def _single_atoms_batch():
    # O, O-, H, H+ as separate 1-atom molecules
    return make_batch(
        [torch.zeros(1, 3), torch.ones(1, 3), 2 * torch.ones(1, 3), 3 * torch.ones(1, 3)],
        [torch.tensor([8]), torch.tensor([8]), torch.tensor([1]), torch.tensor([1])],
        [0.0, -1.0, 0.0, 1.0],
    )


def test_energy_reduces_to_species_parabola():
    model = make_model()
    batch = _single_atoms_batch()
    out = model(batch)

    assert torch.allclose(out.charges, batch.total_charge, atol=1e-12)
    assert out.atomic_dipoles.abs().max() < 1e-12  # zero vec features -> no dipole

    s = torch.tensor([1, 1, 0, 0])  # species idx: O=1, H=0 in sorted [1, 8]
    q = batch.total_charge
    p0 = model.featurizer.feature_dims[0]
    chi0 = model.params.chi0[s] + model.params.chi_mlp(
        torch.cat((torch.zeros(4, p0), model.params.species_emb(s)), dim=-1)
    ).squeeze(-1)
    eta0 = (
        torch.nn.functional.softplus(
            model.params.eta0_raw[s]
            + model.params.eta_mlp(
                torch.cat((torch.zeros(4, p0), model.params.species_emb(s)), dim=-1)
            ).squeeze(-1)
        )
        + model.params.eta_floor
    )
    e_eem = chi0 * q + 0.5 * eta0 * q * q
    # the dipole part contributes -1/2 chivec^T alpha chivec = 0 (chivec = 0)
    e_mlip = model.energy_head(torch.zeros(4, p0), s) + model.reference_energies[s]
    assert (out.energy - (e_eem + e_mlip)).abs().max() < 1e-12


def test_free_atom_alpha_isotropic_and_charge_flow_free():
    model = make_model()
    out = model(_single_atoms_batch())
    # single atoms: no charge flow -> alpha_mol == atomic alpha, isotropic
    off = out.alpha - torch.diag_embed(out.alpha.diagonal(dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-12
    d = out.alpha.diagonal(dim1=-2, dim2=-1)
    assert (d - d[:, :1]).abs().max() < 1e-12
    # same element, same position-independence
    batch = make_batch(
        [torch.zeros(1, 3), 7 * torch.ones(1, 3)],
        [torch.tensor([8]), torch.tensor([8])],
        [0.0, 0.0],
    )
    out2 = model(batch)
    assert (out2.alpha[0] - out2.alpha[1]).abs().max() < 1e-12


def test_dipole_of_isolated_ion_is_q_r():
    model = make_model()
    pos = torch.tensor([[1.0, 2.0, -3.0]])
    batch = make_batch([pos], [torch.tensor([1])], [1.0])
    out = model(batch)
    assert (out.dipole[0] - pos[0]).abs().max() < 1e-12
