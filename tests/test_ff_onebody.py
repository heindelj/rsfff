"""The 1-body term: additivity, per-fragment pooling, forces, and the pair list.

The headline property is that this term is **exactly one-body**. That is checked more
directly than through ``mbe_decompose``: a fragment's energy computed inside a pentamer
must be *bitwise* identical to the same fragment computed alone. That implies every
``E^(k>=2)`` is exactly zero, without going through the Mobius inversion -- and it is worth
noting why the MBE route is awkward here, since a future reader will reach for it: this
model returns a *total* energy, not an interaction energy, so ``mbe_decompose``'s internal
"the expansion sums to the total" check compares ``sum_{k>=2} E^(k) = 0`` against a total of
about -76 Hartree per monomer and fails by construction.
"""

import numpy as np
import pytest
import torch

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.ff.onebody import OneBodyEnergy, OneBodyEnvironment, OneBodyModel
from rsfff.ff.pairs import inter_fragment_pairs, intra_fragment_pairs
from rsfff.ff.response import (
    ElectrostaticParameterHeads,
    FragmentResponse,
    build_elec_priors,
)
from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.mlip.sqe import PairComplianceHead
from rsfff.train.data import Batch

torch.set_default_dtype(torch.float64)

NEIGHBOR_TYPES = [1, 8]
E0 = torch.tensor([-0.4941110651, -75.0780656005])   # H, O at wB97M-V/def2-TZVPD

_WATER = np.array(
    [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
)


def water_cluster(n, spacing=3.0, jitter=0.0, seed=0):
    """``n`` waters strung along x, optionally jittered so they are not identical."""
    rng = np.random.default_rng(seed)
    blocks = []
    for k in range(n):
        block = _WATER + np.array([k * spacing, 0.0, 0.0])
        if jitter:
            block = block + rng.normal(scale=jitter, size=block.shape)
        blocks.append(block)
    positions = torch.tensor(np.concatenate(blocks))
    numbers = torch.tensor([8, 1, 1] * n)
    fragment_idx = torch.arange(n).repeat_interleave(3)
    return positions, numbers, fragment_idx


def make_batch(positions, numbers, fragment_idx, n_systems=1, batch_idx=None):
    n_frag = int(fragment_idx.max()) + 1
    if batch_idx is None:
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.long)
    frag_to_batch = batch_idx.new_zeros(n_frag).scatter_(0, fragment_idx, batch_idx)
    return Batch(
        positions=positions.clone(),
        atomic_numbers=numbers,
        batch_idx=batch_idx,
        n_systems=n_systems,
        energy=torch.zeros(n_systems),
        fragment_idx=fragment_idx,
        fragment_to_batch=frag_to_batch,
        n_fragments=n_frag,
    )


def make_featurizer(with_response=False):
    return FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 1, 2) if with_response else (0, 2),
        backend="e3nn", density_channels=8,
    )


def make_response(featurizer, max_rank=2):
    """The shared per-fragment solve, at the priors."""
    p0 = featurizer.feature_dims[0]
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)
    log_z, log_b, q0 = build_elec_priors(NEIGHBOR_TYPES)
    params = ElectrostaticParameterHeads(
        p0, p1, p2, len(NEIGHBOR_TYPES),
        log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
        irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
        irrep2_to_spherical_map=irrep2_to_spherical(featurizer.backend.irrep6_to_voigt()),
        emb_dim=8, hidden=16, depth=1, equiv_channels=6, max_rank=max_rank,
        chi_init=torch.tensor([0.15, 0.55]), eta_init=torch.tensor([0.5, 0.5]),
    )
    return FragmentResponse(params, PairComplianceHead(p0, hidden=16, depth=1))


def make_model(randomize=False, seed=0, with_response=False, **head_kw):
    featurizer = make_featurizer(with_response)
    kw = dict(emb_dim=8, hidden=32, depth=2, r_on=2.5, r_off=4.0, energy_scale=0.2)
    kw.update(head_kw)
    head = PairEnergyHead(featurizer.feature_dims[0], len(NEIGHBOR_TYPES), **kw)
    response = make_response(featurizer) if with_response else None
    model = OneBodyModel(featurizer, OneBodyEnergy(head, E0, response=response))
    if randomize:
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in head.parameters():
                p.add_(0.3 * torch.randn_like(p))
            if response is not None:
                for p in response.parameters():
                    p.add_(0.05 * torch.randn_like(p))
    return model


# ---------------------------------------------------------------------------
# The defining property
# ---------------------------------------------------------------------------


def test_a_fragment_energy_does_not_depend_on_its_neighbours():
    """A monomer's energy alone equals its energy inside a pentamer.

    Not asserted bitwise: the two evaluations sum different numbers of terms in
    different orders, which moves the last ulp (measured 2e-14 Ha). The *sharp*
    bitwise statement is the next test, where the batch layout is held fixed.
    """
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(5, spacing=3.0, jitter=0.05)
    cluster = model(make_batch(positions, numbers, frag))

    for k in range(5):
        sel = (frag == k).nonzero(as_tuple=True)[0]
        alone = model(
            make_batch(positions[sel], numbers[sel], torch.zeros(3, dtype=torch.long))
        )
        assert alone.fragment_energy.item() == pytest.approx(
            cluster.fragment_energy[k].item(), abs=1e-12
        )


def test_moving_a_monomer_changes_nothing_at_all():
    """Dragging one monomer 40 A away leaves the others **bitwise** unchanged.

    The batch layout is identical here, so there is no summation-order excuse: any
    difference in fragments 0 and 1 would be genuine environment dependence.

    The moved fragment is checked only to 1e-12, and deliberately: every distance it
    computes goes through ``(x + 40) - (y + 40)``, which is not exactly ``x - y`` in
    floating point. That is translation round-off in its *own* internal geometry, not
    knowledge of its surroundings.
    """
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3, spacing=3.0, jitter=0.05)
    before = model(make_batch(positions, numbers, frag))

    moved = positions.clone()
    moved[frag == 2] += torch.tensor([40.0, 0.0, 0.0])
    after = model(make_batch(moved, numbers, frag))

    assert torch.equal(before.fragment_energy[:2], after.fragment_energy[:2])
    assert before.fragment_energy[2].item() == pytest.approx(
        after.fragment_energy[2].item(), abs=1e-12
    )


def test_fresh_head_gives_exactly_the_reference_sum():
    model = make_model()   # zero-initialized readout, no response attached
    positions, numbers, frag = water_cluster(3, jitter=0.05)
    out = model(make_batch(positions, numbers, frag))
    assert torch.equal(out.energy_bond, torch.zeros(3))
    assert torch.equal(out.energy_internal, torch.zeros(3))
    expected = E0[1] + 2 * E0[0]    # one O, two H, in neighbor_types order [1, 8]
    assert out.fragment_energy.tolist() == pytest.approx([expected.item()] * 3, abs=1e-12)
    assert out.energy.item() == pytest.approx(3 * expected.item(), abs=1e-12)


def test_fragment_energies_sum_to_the_frame_energy():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(4, jitter=0.05)
    out = model(make_batch(positions, numbers, frag))
    assert float(out.energy.detach()) == pytest.approx(
        float(out.fragment_energy.detach().sum()), abs=1e-12
    )
    assert torch.allclose(out.fragment_energy, out.energy_ref + out.energy_bond)


def test_batching_matches_single_frames():
    model = make_model(randomize=True)
    pos_a, num_a, frag_a = water_cluster(2, jitter=0.05, seed=1)
    pos_b, num_b, frag_b = water_cluster(3, jitter=0.05, seed=2)
    single = [
        model(make_batch(pos_a, num_a, frag_a)),
        model(make_batch(pos_b, num_b, frag_b)),
    ]
    together = model(
        make_batch(
            torch.cat((pos_a, pos_b)),
            torch.cat((num_a, num_b)),
            torch.cat((frag_a, frag_b + 2)),
            n_systems=2,
            batch_idx=torch.cat(
                (torch.zeros(6, dtype=torch.long), torch.ones(9, dtype=torch.long))
            ),
        )
    )
    assert torch.allclose(
        together.fragment_energy,
        torch.cat([s.fragment_energy for s in single]),
        atol=1e-12,
    )
    assert torch.allclose(
        together.energy, torch.stack([s.energy[0] for s in single]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# The shared response solve inside the 1-body energy
# ---------------------------------------------------------------------------


def test_internal_energy_enters_the_fragment_energy():
    """``E_1body = E0 + E_internal + E_bond``, and the internal part is not zero."""
    model = make_model(with_response=True, randomize=True)
    positions, numbers, frag = water_cluster(3, jitter=0.05)
    out = model(make_batch(positions, numbers, frag))
    assert out.response is not None
    assert float(out.energy_internal.detach().abs().min()) > 1e-6
    assert torch.allclose(
        out.fragment_energy, out.energy_ref + out.energy_internal + out.energy_bond
    )
    # It is the solve's own internal energy, not a separate computation.
    assert torch.equal(out.energy_internal, out.response.internal_energy)


def test_the_response_keeps_the_term_exactly_one_body():
    """Adding the solve must not couple fragments: charge cannot cross a boundary.

    Same split as ``test_moving_a_monomer_changes_nothing_at_all``: the untouched
    fragments are bitwise identical, the moved one only to round-off, because its own
    interatomic distances are recomputed from translated coordinates.
    """
    model = make_model(with_response=True, randomize=True)
    positions, numbers, frag = water_cluster(3, spacing=3.0, jitter=0.05)
    before = model(make_batch(positions, numbers, frag))
    moved = positions.clone()
    moved[frag == 2] += torch.tensor([40.0, 0.0, 0.0])
    after = model(make_batch(moved, numbers, frag))

    stay = frag < 2
    assert torch.equal(before.fragment_energy[:2], after.fragment_energy[:2])
    assert torch.equal(before.response.charges[stay], after.response.charges[stay])
    assert torch.equal(
        before.response.internal_energy[:2], after.response.internal_energy[:2]
    )
    torch.testing.assert_close(
        before.response.charges[~stay], after.response.charges[~stay],
        rtol=0, atol=1e-12,
    )


def test_fragment_charge_is_still_conserved():
    model = make_model(with_response=True, randomize=True)
    positions, numbers, frag = water_cluster(3, jitter=0.05)
    out = model(make_batch(positions, numbers, frag))
    q = out.response.charges.detach()
    per_frag = q.new_zeros(3).index_add_(0, frag, q)
    assert float(per_frag.abs().max()) < 1e-13


def test_forces_still_match_central_differences_through_the_solve():
    """The solve is differentiable, so the 1-body force now flows through it too."""
    model = make_model(with_response=True, randomize=True)
    positions, numbers, frag = water_cluster(2, jitter=0.05)
    positions = positions.clone().requires_grad_(True)
    out = model(make_batch(positions, numbers, frag))
    (grad,) = torch.autograd.grad(out.energy.sum(), positions)
    analytic = -grad

    h = 1e-5
    for atom in (0, 3):
        for axis in range(3):
            def shifted(delta):
                p = positions.detach().clone()
                p[atom, axis] += delta
                return float(model(make_batch(p, numbers, frag)).energy.detach().sum())

            numeric = -(shifted(h) - shifted(-h)) / (2 * h)
            assert analytic[atom, axis].item() == pytest.approx(numeric, abs=2e-7)


# ---------------------------------------------------------------------------
# The pair list
# ---------------------------------------------------------------------------


def test_intra_and_inter_pairs_partition_the_complete_graph():
    positions, _, frag = water_cluster(3, spacing=3.0, jitter=0.05)
    batch_idx = torch.zeros(9, dtype=torch.long)
    intra, _, intra_frag = intra_fragment_pairs(positions, frag)
    # A cutoff beyond the cluster diameter, so the inter list is complete too.
    inter, _ = inter_fragment_pairs(positions, batch_idx, 100.0, fragment_idx=frag)

    def as_set(index):
        return {(int(a), int(b)) for a, b in zip(index[0], index[1])}

    intra_set, inter_set = as_set(intra), as_set(inter)
    complete = {(i, j) for i in range(9) for j in range(9) if i < j}
    assert intra_set | inter_set == complete
    assert not (intra_set & inter_set)
    # Every intra pair really is inside one fragment, and pair_frag names which.
    for col in range(intra.shape[1]):
        i, j = int(intra[0, col]), int(intra[1, col])
        assert frag[i] == frag[j] == intra_frag[col]


def test_intra_pairs_have_no_cutoff():
    """Fragment membership, not distance: a stretched bond must not fall off the list."""
    positions, _, frag = water_cluster(1)
    positions[2] += torch.tensor([0.0, 0.0, -20.0])   # drag one H 20 A away
    pair_index, r, _ = intra_fragment_pairs(positions, frag)
    assert pair_index.shape[1] == 3       # still all three i<j pairs of the monomer
    assert float(r.max()) > 19.0


def test_intra_pair_distances_carry_gradients():
    positions, _, frag = water_cluster(2, jitter=0.05)
    positions.requires_grad_(True)
    _, r, _ = intra_fragment_pairs(positions, frag)
    (grad,) = torch.autograd.grad(r.sum(), positions)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0


# ---------------------------------------------------------------------------
# Symmetry and forces
# ---------------------------------------------------------------------------


def test_translation_and_rotation_invariance():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(3, jitter=0.05)
    base = model(make_batch(positions, numbers, frag)).energy

    shifted = model(
        make_batch(positions + torch.tensor([3.0, -2.0, 1.5]), numbers, frag)
    ).energy
    assert torch.allclose(base, shifted, atol=1e-10)

    angle = torch.tensor(0.9)
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = model(make_batch(positions @ rot.T, numbers, frag)).energy
    assert torch.allclose(base, rotated, atol=1e-10)


def test_forces_match_central_differences():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(2, jitter=0.05)
    positions = positions.clone().requires_grad_(True)
    out = model(make_batch(positions, numbers, frag))
    (grad,) = torch.autograd.grad(out.energy.sum(), positions)
    analytic = -grad

    h = 1e-5
    for atom in (0, 2, 4):
        for axis in range(3):
            def shifted(delta):
                p = positions.detach().clone()
                p[atom, axis] += delta
                return float(model(make_batch(p, numbers, frag)).energy.detach().sum())

            numeric = -(shifted(h) - shifted(-h)) / (2 * h)
            assert analytic[atom, axis].item() == pytest.approx(numeric, abs=2e-7)


def test_forces_are_translation_invariant_and_sum_to_zero():
    model = make_model(randomize=True)
    positions, numbers, frag = water_cluster(2, jitter=0.05)
    positions = positions.clone().requires_grad_(True)
    out = model(make_batch(positions, numbers, frag))
    (grad,) = torch.autograd.grad(out.energy.sum(), positions)
    assert (-grad).sum(0).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_missing_fragment_partition_raises():
    model = make_model()
    positions, numbers, frag = water_cluster(2)
    batch = make_batch(positions, numbers, frag)
    batch.fragment_idx = None
    with pytest.raises(ValueError, match="fragment_idx"):
        model(batch)


def test_the_environment_hook_refuses_to_be_used_silently():
    """The slot exists but is not wired up; using it must fail loudly, not be ignored."""
    model = make_model()
    positions, numbers, frag = water_cluster(2)
    feats = model.featurizer(make_batch(positions, numbers, frag), frag)
    env = OneBodyEnvironment(potential=torch.zeros(6), field=torch.zeros(6, 3))
    with pytest.raises(NotImplementedError):
        model.onebody(make_batch(positions, numbers, frag), feats, env=env)


def test_extra_dim_slot_is_inert_at_zero():
    """``PairEnergyHead(extra_dim=0)`` is bit-identical to one built without the argument."""
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 2), backend="e3nn", density_channels=8,
    )
    p0 = featurizer.feature_dims[0]
    torch.manual_seed(0)
    plain = PairEnergyHead(p0, 2)
    torch.manual_seed(0)
    slotted = PairEnergyHead(p0, 2, extra_dim=0)
    for a, b in zip(plain.parameters(), slotted.parameters()):
        assert torch.equal(a, b)
    assert sum(p.numel() for p in plain.parameters()) == sum(
        p.numel() for p in slotted.parameters()
    )


def test_a_head_with_extra_dim_demands_the_block():
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 2), backend="e3nn", density_channels=8,
    )
    head = PairEnergyHead(featurizer.feature_dims[0], 2, extra_dim=3)
    positions, numbers, frag = water_cluster(2, jitter=0.05)
    batch = make_batch(positions, numbers, frag)
    feats = featurizer(batch, frag)
    pair_index, r, _ = intra_fragment_pairs(positions, frag)
    with pytest.raises(ValueError, match="extra_dim=3"):
        head(feats.inv_feats, feats.species_idx, positions, pair_index, r)
    out = head(
        feats.inv_feats, feats.species_idx, positions, pair_index, r,
        extra_pair=torch.zeros(pair_index.shape[1], 3),
    )
    assert out.shape == (pair_index.shape[1],)
