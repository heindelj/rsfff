"""The joint 1-body + electrostatics model: one shared solve, two exactness guarantees.

The point of the composite is that the multipoles which interact and the internal energy
that paid for arranging them come from the same object. These tests check that the sharing
is real (not two copies that happen to agree at init), and that neither half loses the
property it had alone: the 1-body energy stays exactly one-body, the interaction stays
exactly two-body, and ``E_internal`` appears in the former and never in the latter.
"""

import numpy as np
import pytest
import torch

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.electrostatics import SlaterElectrostatics
from rsfff.ff.many_body import mbe_decompose
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.ff.onebody import OneBodyEnergy
from rsfff.ff.onebody_elec import OneBodyElecModel
from rsfff.ff.response import (
    ElectrostaticParameterHeads,
    FragmentResponse,
    build_elec_priors,
)
from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.mlip.sqe import PairComplianceHead
from rsfff.train.data import Batch

torch.set_default_dtype(torch.float64)

NEIGHBOR_TYPES = [1, 8]
E0 = torch.tensor([-0.4941110651, -75.0780656005])   # H, O at wB97M-V/def2-TZVPD

_WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


def water_cluster(n, spacing=3.0, jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [
        _WATER + np.array([k * spacing, 0.0, 0.0]) + rng.normal(scale=jitter, size=(3, 3))
        for k in range(n)
    ]
    positions = torch.tensor(np.concatenate(blocks))
    return positions, torch.tensor([8, 1, 1] * n), torch.arange(n).repeat_interleave(3)


def make_batch(positions, numbers, fragment_idx):
    n_frag = int(fragment_idx.max()) + 1
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.long)
    return Batch(
        positions=positions.clone(),
        atomic_numbers=numbers,
        batch_idx=batch_idx,
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=fragment_idx,
        fragment_charge=torch.zeros(n_frag),
        fragment_to_batch=batch_idx.new_zeros(n_frag),
        n_fragments=n_frag,
    )


def make_model(*, randomize=False, seed=0, max_rank=2, share=True):
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 1, 2), backend="e3nn", density_channels=8,
    )
    p0 = featurizer.feature_dims[0]
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)

    def build_response():
        log_z, log_b, q0 = build_elec_priors(NEIGHBOR_TYPES)
        params = ElectrostaticParameterHeads(
            p0, p1, p2, len(NEIGHBOR_TYPES),
            log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
            irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
            irrep2_to_spherical_map=irrep2_to_spherical(
                featurizer.backend.irrep6_to_voigt()
            ),
            emb_dim=8, hidden=16, depth=1, equiv_channels=6, max_rank=max_rank,
            chi_init=torch.tensor([0.15, 0.55]), eta_init=torch.tensor([0.5, 0.5]),
        )
        return FragmentResponse(params, PairComplianceHead(p0, hidden=16, depth=1))

    response = build_response()
    bond = PairEnergyHead(
        p0, len(NEIGHBOR_TYPES), emb_dim=8, hidden=16, depth=1,
        r_on=2.5, r_off=4.0, energy_scale=0.2,
    )
    onebody = OneBodyEnergy(bond, E0, response=response)
    elec = SlaterElectrostatics(
        response if share else build_response(),
        PairEnergyHead(p0, len(NEIGHBOR_TYPES), emb_dim=8, hidden=16, depth=1),
        cutoff=12.0, r0_init=1e-6, max_rank=max_rank,
    )
    if not share:
        return featurizer, onebody, elec
    model = OneBodyElecModel(featurizer, onebody, elec)
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


# ---------------------------------------------------------------------------
# The sharing itself
# ---------------------------------------------------------------------------


def test_both_terms_read_one_solve():
    model = make_model(randomize=True)
    assert model.onebody.response is model.elec.response
    positions, numbers, frag = water_cluster(3)
    out = model(make_batch(positions, numbers, frag))
    # The composite's multipoles ARE the 1-body term's and the electrostatics term's.
    assert torch.equal(out.charges, out.elec.charges)
    assert torch.equal(out.charges, out.onebody.response.charges)
    assert torch.equal(out.onebody.energy_internal, out.elec.internal_energy)


def test_two_separate_responses_are_refused():
    """Silently holding two copies is the failure this whole module exists to prevent."""
    featurizer, onebody, elec = make_model(share=False)
    with pytest.raises(ValueError, match="free to drift apart"):
        OneBodyElecModel(featurizer, onebody, elec)


def test_a_onebody_term_without_a_response_is_refused():
    featurizer, onebody, elec = make_model(share=False)
    onebody.response = None
    with pytest.raises(ValueError, match="shared response"):
        OneBodyElecModel(featurizer, onebody, elec)


def test_internal_energy_is_in_the_onebody_half_and_not_the_interaction():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(2)
    out = model(make_batch(positions, numbers, frag))
    assert float(out.onebody.energy_internal.detach().abs().min()) > 1e-6
    # The interaction is still exactly its two pieces -- internal is not among them.
    torch.testing.assert_close(
        out.elec.energy, out.elec.energy_ff + out.elec.energy_corr, rtol=0, atol=0
    )
    torch.testing.assert_close(
        out.energy, out.onebody.energy + out.elec.energy, rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# Both exactness guarantees survive the sharing
# ---------------------------------------------------------------------------


def test_the_onebody_half_is_still_exactly_one_body():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3, spacing=3.0)
    before = model(make_batch(positions, numbers, frag))
    moved = positions.clone()
    moved[frag == 2] += torch.tensor([40.0, 0.0, 0.0])
    after = model(make_batch(moved, numbers, frag))
    # Untouched fragments bitwise; the moved one only to round-off, because its own
    # distances are recomputed from translated coordinates.
    assert torch.equal(before.fragment_energy[:2], after.fragment_energy[:2])
    assert before.fragment_energy[2].item() == pytest.approx(
        after.fragment_energy[2].item(), abs=1e-12
    )


def test_the_electrostatics_half_is_still_exactly_two_body():
    """``mbe_decompose`` on the interaction alone: every ``E^(k>=3)`` is round-off."""

    class InteractionOnly(torch.nn.Module):
        """The composite's electrostatics half, in the shape ``mbe_decompose`` wants."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, batch):
            return self.model(batch).elec

    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3, spacing=3.0)
    res = mbe_decompose(InteractionOnly(model), positions, numbers, frag)
    assert abs(float(res.total)) > 1e-6, "need a nonzero total for this to mean anything"
    assert abs(float(res.by_order[3])) * KJMOL_PER_HARTREE < 1e-10


def test_fragment_charge_is_conserved():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3)
    out = model(make_batch(positions, numbers, frag))
    q = out.charges.detach()
    per_frag = q.new_zeros(3).index_add_(0, frag, q)
    assert float(per_frag.abs().max()) < 1e-13


def test_rotation_and_translation_invariance():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3)
    base = model(make_batch(positions, numbers, frag))

    shifted = model(
        make_batch(positions + torch.tensor([2.5, -1.0, 4.0]), numbers, frag)
    )
    torch.testing.assert_close(base.energy, shifted.energy, rtol=1e-10, atol=1e-12)

    angle = torch.tensor(0.8)
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = model(make_batch(positions @ rot.T, numbers, frag))
    torch.testing.assert_close(base.energy, rotated.energy, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(base.mu @ rot.T, rotated.mu, rtol=1e-9, atol=1e-12)


def test_forces_match_central_differences_through_both_halves():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(2)
    positions = positions.clone().requires_grad_(True)
    out = model(make_batch(positions, numbers, frag))
    (grad,) = torch.autograd.grad(out.energy.sum(), positions)
    analytic = -grad

    h = 1e-5
    for atom in (0, 4):
        for axis in range(3):
            def shifted(delta):
                p = positions.detach().clone()
                p[atom, axis] += delta
                return float(model(make_batch(p, numbers, frag)).energy.detach().sum())

            numeric = -(shifted(h) - shifted(-h)) / (2 * h)
            assert analytic[atom, axis].item() == pytest.approx(numeric, abs=2e-7)
