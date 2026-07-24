"""Phase-1 monomer stack: diabatic assignment, reference embeddings, and the free-atom limit.

The load-bearing claim tested here is the one the whole architecture rests on
(docs/mixture_of_diabatic_embeddings.md §2.3): a fragment in isolation is described *exactly*
by its 1-body reference, as an architectural property rather than a trained one. For a lone
atom that means the model reproduces the isolated-atom QC data at integer charge -- and with
the chi/eta biases seeded from the atomic IP/EA it does so at initialization, before any
training at all.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from rsfff.mlip import (
    AtomicStateReference,
    DiabaticStateLibrary,
    assign_from_headers,
    build_monomer_model,
)
from rsfff.train.data import load_atomic_reference_batch, load_datasets

LIBRARY = "data/diabatic_states.yaml"
STATES = "data/atomic_reference_states.json"
LABELS = ["data/labels/h2o.extxyz", "data/labels/h3o+.extxyz", "data/labels/oh-.extxyz"]


@pytest.fixture
def library():
    return DiabaticStateLibrary.from_yaml(LIBRARY)


@pytest.fixture
def states():
    return AtomicStateReference.from_json(STATES, [1, 8])


def make_model(states=None, seed=None):
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2), backend="e3nn"
    )
    monomer = types.SimpleNamespace(
        emb_dim=12, weight_channels=4, hidden=32, depth=1, equiv_channels=8
    )
    sqe = types.SimpleNamespace(
        s_init=0.5, s_floor=0.0, n_radial=6, eta_init=0.5, eta_floor=0.05, psd_floor=1e-4
    )
    model = build_monomer_model(
        features, monomer, sqe, [1, 8],
        torch.tensor([-0.5013141188705883, -75.00924018194253]), states,
    )
    if seed is not None:  # activate every pathway that zero-init leaves dormant
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


# ---------------------------------------------------------------------------
# Diabatic assignment: the channel graph comes from the state, not the geometry
# ---------------------------------------------------------------------------

def test_channel_graph_is_the_diabats_not_a_distance_rule(library):
    """Bond topology is fixed by the assignment and identical across all sampled geometries.

    The label sets are Wigner-sampled, so O-H bond lengths vary substantially. A covalent-radius
    cutoff would flicker; the diabatic assignment does not.
    """
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    expected = {3: 2, 4: 3, 2: 1}  # atoms -> channels for H2O / H3O+ / OH-

    lengths = []
    for frame in range(0, len(dataset), 37):
        batch = dataset.flat_batch(torch.tensor([frame]))
        n_atoms = batch.positions.shape[0]
        assert batch.bond_index.shape[1] == expected[n_atoms]
        # every channel is O-H, with oxygen (listed first in the registry) as the head
        assert (batch.atomic_numbers[batch.bond_index[0]] == 8).all()
        assert (batch.atomic_numbers[batch.bond_index[1]] == 1).all()
        d = batch.positions[batch.bond_index[0]] - batch.positions[batch.bond_index[1]]
        lengths.append(d.norm(dim=-1))

    lengths = torch.cat(lengths)
    assert lengths.max() - lengths.min() > 0.2, "expected real geometric spread in the labels"


def test_assignment_rejects_inconsistent_frames(library):
    """Element order, charge, and multiplicity are validated, never silently coerced."""
    with pytest.raises(ValueError, match="element order"):
        assign_from_headers(library, ["H", "O", "H"], config_type="h2o")
    with pytest.raises(ValueError, match="charge"):
        assign_from_headers(library, ["O", "H", "H"], config_type="h2o", charge=1.0)
    with pytest.raises(ValueError, match="multiplicity"):
        assign_from_headers(library, ["O", "H", "H"], config_type="h2o", multiplicity=3)
    with pytest.raises(KeyError):
        assign_from_headers(library, ["O", "H"], config_type="not_a_state")


def test_model_refuses_a_batch_without_a_diabatic_assignment(states):
    """Loading without a state library gives no channel graph; fail loudly, not silently.

    Silently treating a missing assignment as "no channels" would return baseline charges and
    a plausible-looking energy, which is the worst possible failure mode here.
    """
    dataset = load_datasets(LABELS[:1], dtype=torch.float64)  # note: no library=
    batch = dataset.flat_batch(torch.tensor([0]))
    with pytest.raises(ValueError, match="diabatic assignment"):
        make_model(states)(batch)


def test_bond_indices_are_reoffset_when_frames_are_batched(library):
    """Frame-local channel indices must become batch-global ones, pointing at the right atoms."""
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    batch = dataset.flat_batch(torch.tensor([0, 500, 1000, 3]))
    assert batch.bond_index.max() < batch.positions.shape[0]
    # a channel never spans two systems
    assert (batch.batch_idx[batch.bond_index[0]] == batch.bond_batch).all()
    assert (batch.batch_idx[batch.bond_index[1]] == batch.bond_batch).all()
    # ... nor two fragments
    assert (
        batch.fragment_idx[batch.bond_index[0]] == batch.fragment_idx[batch.bond_index[1]]
    ).all()


# ---------------------------------------------------------------------------
# Reference embeddings and baseline charges
# ---------------------------------------------------------------------------

def test_baseline_charges_sum_to_the_formal_charge_exactly(library):
    """q^(0) conserves Q_a structurally, for any weights -- it is built traceless."""
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    batch = dataset.flat_batch(torch.tensor([0, 500, 1000]))
    model = make_model(seed=3)
    out = model(batch)
    total = torch.zeros(batch.n_fragments, dtype=torch.float64).index_add_(
        0, batch.fragment_idx, out.baseline_charge
    )
    assert torch.allclose(total, batch.fragment_charge, atol=1e-13)


def test_reference_embedding_distinguishes_diabatic_states(library):
    """The same element in different fragment states must get different embeddings.

    This is what a species one-hot cannot do, and the reason the density is state-decorated:
    the hydrogens of OH- and H3O+ are not the same object.
    """
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    model = make_model(seed=3)
    h3o = model(dataset.flat_batch(torch.tensor([500])))
    oh = model(dataset.flat_batch(torch.tensor([1000])))
    # atom 0 is oxygen in both; atom 1 is a hydrogen in both
    assert not torch.allclose(h3o.embedding[0], oh.embedding[0], atol=1e-6)
    assert not torch.allclose(h3o.embedding[1], oh.embedding[1], atol=1e-6)


# ---------------------------------------------------------------------------
# The free-atom limit
# ---------------------------------------------------------------------------

def test_free_atom_energies_are_exact_at_initialization(states):
    """A lone atom reproduces its reference-state energy *before any training*.

    With zero features and no channels the model collapses to
    ``E = E0[Z] + E_1(e) + chi Q + eta Q^2 / 2``, and seeding chi/eta at the Mulliken values
    gives ``chi + eta/2 = IP`` and ``-chi + eta/2 = -EA`` identically. So q = 0, +1, -1 are
    reproduced exactly; nothing is fitted here.
    """
    model = make_model(states)
    anchor = load_atomic_reference_batch(states, [1, 8])
    out = model(anchor)
    assert torch.allclose(out.charges, anchor.total_charge, atol=1e-13)
    assert torch.allclose(out.energy, anchor.energy, atol=1e-12)


def test_free_atom_limit_survives_training_of_the_environment_heads(states):
    """Perturbing every weight must not break q = Q: no channels means nothing can flow."""
    model = make_model(states, seed=5)
    anchor = load_atomic_reference_batch(states, [1, 8])
    out = model(anchor)
    assert torch.allclose(out.charges, anchor.total_charge, atol=1e-13)
    assert out.transfers.numel() == 0
    assert torch.allclose(out.alpha, out.alpha.transpose(-1, -2), atol=1e-13)


def test_isolated_atom_has_no_environment(states):
    """Zero SOAP density for a lone atom is what makes the anchors exact -- verify it holds."""
    model = make_model(states)
    anchor = load_atomic_reference_batch(states, [1, 8])
    out = model(anchor)
    assert torch.allclose(out.feats.inv_feats, torch.zeros_like(out.feats.inv_feats))
    assert torch.allclose(out.feats.vec_feats, torch.zeros_like(out.feats.vec_feats))
    assert torch.allclose(out.feats.equiv_feats, torch.zeros_like(out.feats.equiv_feats))


def test_atomic_reference_states_reproduce_experimental_ionization(states):
    """Guard against a broken/stale reference file: IP and EA must be physically sane."""
    ha_ev = 27.211386245988
    chi, eta = states.head_bias_init()
    ip = (chi + 0.5 * eta) * ha_ev
    assert 13.0 < float(ip[0]) < 14.5, f"H IP = {float(ip[0]):.2f} eV (expected ~13.6)"
    assert 13.0 < float(ip[1]) < 15.0, f"O IP = {float(ip[1]):.2f} eV (expected ~13.6)"
    ea_o = float((-chi[1] + 0.5 * eta[1]) * -ha_ev)
    assert 1.0 < ea_o < 2.5, f"O EA = {ea_o:.2f} eV (expected ~1.46)"


# ---------------------------------------------------------------------------
# Symmetry and differentiability of the assembled model
# ---------------------------------------------------------------------------

def test_observables_transform_correctly_under_rotation(library, states):
    from e3nn import o3

    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    model = make_model(states, seed=11)
    batch = dataset.flat_batch(torch.tensor([0, 500, 1000]))
    out = model(batch)

    rot = o3.rand_matrix().to(torch.float64)
    rotated = dataset.flat_batch(torch.tensor([0, 500, 1000]))
    rotated.positions = rotated.positions @ rot.T
    out_rot = model(rotated)

    assert torch.allclose(out_rot.energy, out.energy, atol=1e-10)
    assert torch.allclose(out_rot.charges, out.charges, atol=1e-12)
    assert torch.allclose(out_rot.dipole, out.dipole @ rot.T, atol=1e-12)
    assert torch.allclose(out_rot.alpha, rot @ out.alpha @ rot.T, atol=1e-11)


def test_analytic_polarizability_matches_the_field_derivative(library, states):
    """``MonomerOutput.alpha`` must equal d mu/dF through the whole model, not just the SQE part."""
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    model = make_model(states, seed=11)
    batch = dataset.flat_batch(torch.tensor([0, 500, 1000]))
    field = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    out = model(batch, field=field)
    rows = [
        torch.autograd.grad(out.dipole[:, k].sum(), field, retain_graph=True)[0]
        for k in range(3)
    ]
    assert torch.allclose(torch.stack(rows, dim=1), out.alpha, atol=1e-10)


def test_forces_and_second_derivatives_are_finite(library, states):
    """Force training double-backwards through the linear solve; no NaN anywhere."""
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    model = make_model(states, seed=11)
    batch = dataset.flat_batch(torch.tensor([0, 500, 1000]))  # mixed sizes -> padded channels
    batch.positions.requires_grad_(True)
    out = model(batch)
    (grad,) = torch.autograd.grad(out.energy.sum(), batch.positions, create_graph=True)
    assert torch.isfinite(grad).all()
    grad.pow(2).sum().backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite gradient on {name}"


def test_forces_match_finite_difference(library, states):
    """Analytic -dE/dx against a central difference -- catches sign and chain-rule errors."""
    dataset = load_datasets(LABELS, dtype=torch.float64, library=library)
    model = make_model(states, seed=11)
    batch = dataset.flat_batch(torch.tensor([500]))
    batch.positions.requires_grad_(True)
    out = model(batch)
    (grad,) = torch.autograd.grad(out.energy.sum(), batch.positions)

    step = 1e-6
    numeric = torch.zeros_like(grad)
    base = batch.positions.detach()
    for i in range(base.shape[0]):
        for c in range(3):
            plus, minus = base.clone(), base.clone()
            plus[i, c] += step
            minus[i, c] -= step
            b_p = dataset.flat_batch(torch.tensor([500]))
            b_m = dataset.flat_batch(torch.tensor([500]))
            b_p.positions, b_m.positions = plus, minus
            numeric[i, c] = (
                model(b_p).energy.sum() - model(b_m).energy.sum()
            ) / (2 * step)
    assert torch.allclose(grad, numeric, atol=1e-6), (grad - numeric).abs().max()
