"""Tang-Toennies damped C6 dispersion: formula, units, pair list, and forces.

The cross-repo test against pyCMM is the load-bearing one -- it pins the formula, the
geometric-mean combination, the Angstrom->bohr conversion, and the inter-fragment pair
list all at once, against an implementation that was fit to real data.
"""

import math

import numpy as np
import pytest
import torch
from ase.io import read

from rsfff.ff.dispersion import (
    DEFAULT_C6_PRIOR,
    DispersionParameterHeads,
    TTDispersion,
    build_log_priors,
    tt_damped_c6_energy,
)
from rsfff.ff.pairs import inter_fragment_pairs
from rsfff.ff.units import BOHR_ANG
from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.train.data import load_extxyz

from conftest import DATA_W2, DATA_W3, features_cfg

NEIGHBOR_TYPES = [1, 8]


# ---------------------------------------------------------------------------
# Reference implementation, transcribed from pyCMM/cmm/dispersion.py:5-21
# ---------------------------------------------------------------------------

def pycmm_dispersion(r_bohr, c6_i, c6_j, b_i, b_j):
    """pyCMM's computeDispersion, in plain numpy. Distances in bohr, energy in Hartree."""
    c6_ij = math.sqrt(c6_i * c6_j)
    b_ij = math.sqrt(b_i * b_j)
    u = b_ij * r_bohr
    damp = 1 - math.exp(-u) * (
        1 + u + u**2 / 2 + u**3 / 6 + u**4 / 24 + u**5 / 120 + u**6 / 720
    )
    return -damp * c6_ij / r_bohr**6


def pycmm_frame_energy(atoms):
    """Sum pyCMM dispersion over the inter-fragment pairs of one ASE frame."""
    pos, Z, fi = atoms.get_positions(), atoms.numbers, atoms.arrays["fragment_idx"]
    total = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if fi[i] == fi[j]:
                continue
            r = float(np.linalg.norm(pos[i] - pos[j])) / BOHR_ANG
            total += pycmm_dispersion(
                r, *(DEFAULT_C6_PRIOR[Z[i]][0], DEFAULT_C6_PRIOR[Z[j]][0]),
                *(DEFAULT_C6_PRIOR[Z[i]][1], DEFAULT_C6_PRIOR[Z[j]][1]),
            )
    return total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def w2_dataset():
    return load_extxyz(DATA_W2, dtype=torch.float64)


@pytest.fixture(scope="module")
def w2_frames():
    return read(DATA_W2, ":5")


def make_featurizer():
    cfg = features_cfg()
    return FlatLambdaSOAPFeaturizer(
        cutoff=cfg.cutoff, n_max=cfg.n_max, l_max=cfg.l_max,
        neighbor_types=NEIGHBOR_TYPES, selected_lambdas=cfg.selected_lambdas,
        backend=cfg.backend, density_channels=cfg.density_channels,
    )


def make_model(*, correction=True, r0_init=2.0, learn_r0=True, cutoff=10.0,
               b_prior=None, randomize=False, seed=1):
    """TTDispersion at the pyCMM priors. ``b_prior=None`` uses pyCMM's per-element b."""
    feat = make_featurizer()
    p0 = feat.feature_dims[0]
    if b_prior is None:      # pyCMM's fitted per-element damping exponents
        log_c6 = torch.tensor([DEFAULT_C6_PRIOR[z][0] for z in NEIGHBOR_TYPES]).log()
        log_b = torch.tensor([DEFAULT_C6_PRIOR[z][1] for z in NEIGHBOR_TYPES]).log()
    else:
        log_c6, log_b = build_log_priors(NEIGHBOR_TYPES, b_prior=b_prior)
    params = DispersionParameterHeads(
        p0, len(NEIGHBOR_TYPES), log_c6_prior=log_c6, log_b_prior=log_b,
        emb_dim=8, hidden=16, depth=1,
    )
    head = PairEnergyHead(
        p0, len(NEIGHBOR_TYPES), emb_dim=8, hidden=16, depth=1, r_on=4.0, r_off=5.0
    ) if correction else None
    model = TTDispersion(
        params, head, cutoff=cutoff, r0_init=r0_init, alpha=8.0, learn_r0=learn_r0
    )
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return feat, model


def run(feat, model, batch):
    return model(batch, feat(batch))


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------

def undamped_ff_energy(batch, cutoff=10.0):
    """Bare Tang-Toennies sum over the inter-fragment pairs, with no switch or taper.

    Isolates the formula + combination rule + Angstrom->bohr conversion + pair list from
    the range-separation machinery, so the pyCMM comparison is exact rather than
    approximate. (Through the model, ``r0`` cannot be set to exactly zero -- softplus
    keeps it positive -- so the Fermi switch is always 1 - O(1e-7).)
    """
    pair_index, r = inter_fragment_pairs(
        batch.positions, batch.batch_idx, cutoff, fragment_idx=batch.fragment_idx
    )
    i, j = pair_index
    z = batch.atomic_numbers
    c6 = torch.tensor([DEFAULT_C6_PRIOR[int(k)][0] for k in z], dtype=torch.float64)
    b = torch.tensor([DEFAULT_C6_PRIOR[int(k)][1] for k in z], dtype=torch.float64)
    return pair_index, tt_damped_c6_energy(
        r, (c6[i] * c6[j]).sqrt(), (b[i] * b[j]).sqrt()
    ).sum()


def test_matches_pycmm_water_dimer(w2_dataset, w2_frames):
    """Cross-repo regression: our formula + pair list vs a transcription of pyCMM."""
    for k in range(3):
        pair_index, energy = undamped_ff_energy(w2_dataset.flat_batch([k]))
        assert pair_index.shape[1] == 9      # 2 waters -> 3x3 inter-fragment pairs
        assert energy.item() == pytest.approx(pycmm_frame_energy(w2_frames[k]), rel=1e-12)


def test_pycmm_anchor_value(w2_dataset):
    """Hard-coded anchor, so a refactor of the inline reference cannot move the target."""
    _, energy = undamped_ff_energy(w2_dataset.flat_batch([0]))
    assert energy.item() == pytest.approx(-3.204908719236e-3, rel=1e-11)


def test_model_recovers_unswitched_limit_as_r0_shrinks(w2_dataset):
    """softplus keeps r0 > 0, so the switch approaches 1 from below rather than hitting it."""
    batch = w2_dataset.flat_batch([0])
    _, reference = undamped_ff_energy(batch)
    feat, model = make_model(correction=False, r0_init=1e-6, learn_r0=False)
    assert run(feat, model, batch).energy_ff.item() == pytest.approx(
        reference.item(), rel=1e-6
    )


def test_r6_asymptote():
    """At large r the damping saturates and the energy is exactly -C6/r^6."""
    r = torch.tensor([30.0], dtype=torch.float64)
    c6 = torch.tensor([35.8289], dtype=torch.float64)
    b = torch.tensor([1.84302], dtype=torch.float64)
    e = tt_damped_c6_energy(r, c6, b)
    assert (e * (r / BOHR_ANG) ** 6).item() == pytest.approx(-35.8289, rel=1e-10)


def test_geometric_mean_combination():
    """E depends on (C6_i, C6_j) only through sqrt(C6_i C6_j); same for b."""
    r = torch.tensor([3.0], dtype=torch.float64)
    mixed = tt_damped_c6_energy(
        r, torch.tensor([math.sqrt(4.0 * 25.0)]), torch.tensor([math.sqrt(1.0 * 4.0)])
    )
    equal = tt_damped_c6_energy(r, torch.tensor([10.0]), torch.tensor([2.0]))
    assert mixed.item() == pytest.approx(equal.item(), rel=1e-12)


def test_short_range_is_regular():
    """TT damping removes the r^-6 divergence.

    f_6(x) -> x^7/7! cancels six of the seven powers, leaving E -> -C6 b^7 r_au / 7!:
    finite at coalescence and linear in r, which is why the Fermi switch does not have to
    do the short-range regularization itself.
    """
    r = torch.tensor([1e-4, 1e-3], dtype=torch.float64)
    c6, b = 35.8289, 1.84302
    e = tt_damped_c6_energy(
        r, torch.full((2,), c6, dtype=torch.float64), torch.full((2,), b, dtype=torch.float64)
    )
    slope = e / (r / BOHR_ANG)
    limit = -c6 * b**7 / math.factorial(7)
    assert slope[0].item() == pytest.approx(limit, rel=1e-3)
    assert abs(slope[0].item() - limit) < abs(slope[1].item() - limit)   # converging


# ---------------------------------------------------------------------------
# Pair list
# ---------------------------------------------------------------------------

def test_pair_list_is_inter_fragment_and_undirected(w2_dataset):
    batch = w2_dataset.flat_batch([0, 1])
    pair_index, r = inter_fragment_pairs(
        batch.positions, batch.batch_idx, 10.0, fragment_idx=batch.fragment_idx
    )
    i, j = pair_index
    assert torch.all(i < j)                                       # each pair once
    assert torch.all(batch.fragment_idx[i] != batch.fragment_idx[j])
    assert torch.all(batch.batch_idx[i] == batch.batch_idx[j])    # no cross-frame pairs
    assert pair_index.shape[1] == 18                              # 9 per dimer, 2 frames
    assert torch.allclose(
        r, (batch.positions[i] - batch.positions[j]).norm(dim=-1)
    )


def test_pair_list_rejects_interleaved_fragments():
    positions = torch.randn(6, 3, dtype=torch.float64)
    batch_idx = torch.zeros(6, dtype=torch.long)
    bad = torch.tensor([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="non-decreasing fragment_idx"):
        inter_fragment_pairs(positions, batch_idx, 10.0, fragment_idx=bad)


def test_pair_list_covers_every_inter_fragment_pair(w2_dataset):
    """No pair silently dropped: the list must match an explicit i<j double loop."""
    batch = w2_dataset.flat_batch([0])
    pair_index, _ = inter_fragment_pairs(
        batch.positions, batch.batch_idx, 10.0, fragment_idx=batch.fragment_idx
    )
    got = {(int(a), int(b)) for a, b in pair_index.T.tolist()}
    fi = batch.fragment_idx
    want = {
        (i, j)
        for i in range(len(fi)) for j in range(i + 1, len(fi))
        if fi[i] != fi[j]
    }
    assert got == want


# ---------------------------------------------------------------------------
# Symmetries, batching, forces
# ---------------------------------------------------------------------------

def test_invariance_under_rotation_and_translation(w2_dataset):
    feat, model = make_model(randomize=True)
    batch = w2_dataset.flat_batch([0])
    e0 = run(feat, model, batch).energy.item()

    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    moved = w2_dataset.flat_batch([0])
    moved.positions = moved.positions @ R.T + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    assert run(feat, model, moved).energy.item() == pytest.approx(e0, abs=1e-10)


def test_forces_match_finite_difference(w2_dataset):
    feat, model = make_model(randomize=True)
    batch = w2_dataset.flat_batch([0])
    batch.positions = batch.positions.clone().requires_grad_(True)
    energy = run(feat, model, batch).energy.sum()
    (grad,) = torch.autograd.grad(energy, batch.positions)

    h = 1e-4
    for atom, comp in ((0, 0), (2, 1), (4, 2)):
        num = []
        for sign in (+1, -1):
            b = w2_dataset.flat_batch([0])
            b.positions = b.positions.clone()
            b.positions[atom, comp] += sign * h
            num.append(run(feat, model, b).energy.sum().item())
        assert grad[atom, comp].item() == pytest.approx((num[0] - num[1]) / (2 * h), abs=1e-8)


def test_batch_equals_separate_frames(w2_dataset):
    feat, model = make_model(randomize=True)
    together = run(feat, model, w2_dataset.flat_batch([0, 1, 2])).energy
    for k, idx in enumerate([0, 1, 2]):
        alone = run(feat, model, w2_dataset.flat_batch([idx])).energy
        assert together[k].item() == pytest.approx(alone.item(), abs=1e-12)


def test_mixed_cluster_sizes_batch(w2_dataset):
    """A w2 and a w3 frame in one batch: pooling must not leak between them."""
    w3 = load_extxyz(DATA_W3, dtype=torch.float64)
    feat, model = make_model(randomize=True)
    e2 = run(feat, model, w2_dataset.flat_batch([0])).energy.item()
    e3 = run(feat, model, w3.flat_batch([0])).energy.item()

    b2, b3 = w2_dataset.flat_batch([0]), w3.flat_batch([0])
    from rsfff.train.data import Batch
    merged = Batch(
        positions=torch.cat([b2.positions, b3.positions]),
        atomic_numbers=torch.cat([b2.atomic_numbers, b3.atomic_numbers]),
        batch_idx=torch.cat([b2.batch_idx, b3.batch_idx + 1]),
        n_systems=2,
        energy=torch.cat([b2.energy, b3.energy]),
        fragment_idx=torch.cat([b2.fragment_idx, b3.fragment_idx + b2.n_fragments]),
        n_fragments=b2.n_fragments + b3.n_fragments,
    )
    out = run(feat, model, merged).energy
    assert out[0].item() == pytest.approx(e2, abs=1e-12)
    assert out[1].item() == pytest.approx(e3, abs=1e-12)


# ---------------------------------------------------------------------------
# Range separation and tapers
# ---------------------------------------------------------------------------

def test_taper_is_continuous_through_the_cutoff():
    """Sweeping a pair through r_cut leaves no step in energy or in its derivative."""
    feat, model = make_model(correction=False, cutoff=5.0, r0_init=1e-6, learn_r0=False)
    from rsfff.train.data import Batch

    def energy_at(d):
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.9, 0.0, -0.3],
             [d, 0.0, 0.0], [d, 0.0, 0.96], [d + 0.9, 0.0, -0.3]],
            dtype=torch.float64, requires_grad=True,
        )
        b = Batch(
            positions=pos, atomic_numbers=torch.tensor([8, 1, 1, 8, 1, 1]),
            batch_idx=torch.zeros(6, dtype=torch.long), n_systems=1,
            energy=torch.zeros(1),
            fragment_idx=torch.tensor([0, 0, 0, 1, 1, 1]), n_fragments=2,
        )
        e = run(feat, model, b).energy.sum()
        (g,) = torch.autograd.grad(e, pos)
        # d moves the whole second fragment, so dE/dd is the sum over its three atoms.
        return e.item(), g[3:, 0].sum().item()

    # A step in the energy or a kink in it shows up as the analytic gradient diverging
    # from the central difference; comparing consecutive energies would only measure the
    # genuine slope. Sample straight through r_cut = 5.0.
    h = 1e-5
    for d in (4.5, 4.9, 4.99, 5.0, 5.01, 5.1):
        analytic = energy_at(d)[1]
        numeric = (energy_at(d + h)[0] - energy_at(d - h)[0]) / (2 * h)
        assert analytic == pytest.approx(numeric, abs=1e-9)
    # Exactly zero once *every* pair clears r_cut. The offset H atoms mean the closest
    # contact is ~d-0.9, so d must exceed 5.9 for that -- at d=5.5 there is still a pair
    # at 4.6 A contributing.
    assert abs(energy_at(7.0)[0]) == 0.0


def test_fermi_switch_removes_short_range(w2_dataset):
    """r0 large kills the explicit term; r0 -> 0 recovers the unswitched value."""
    batch = w2_dataset.flat_batch([0])
    feat, off = make_model(correction=False, r0_init=20.0, learn_r0=False)
    assert abs(run(feat, off, batch).energy_ff.item()) < 1e-12

    _, on = make_model(correction=False, r0_init=1e-6, learn_r0=False)
    assert run(feat, on, batch).energy_ff.item() == pytest.approx(-3.204908719236e-3, rel=1e-6)


def test_r0_default_keeps_most_of_the_dispersion(w2_dataset):
    """A guard on the r0=2.0 default: the FF must remain the backbone, not a bit player.

    Measured during planning: r0=3.0 leaves the explicit term only ~24% of the dispersion
    (the dominant water-dimer contacts, H-bonded H...O at ~1.9 A and O...O at ~2.8 A, are
    both inside 3 A). If someone raises the default, this fails and they have to decide
    deliberately.
    """
    batch = w2_dataset.flat_batch(range(20))
    feat, default = make_model(correction=False, r0_init=2.0, learn_r0=False)
    _, unswitched = make_model(correction=False, r0_init=1e-6, learn_r0=False)
    kept = (
        run(feat, default, batch).energy_ff.sum()
        / run(feat, unswitched, batch).energy_ff.sum()
    ).item()
    assert 0.6 < kept < 1.0


def test_r0_is_learnable_and_positive():
    feat, model = make_model()
    assert model.r0.item() == pytest.approx(2.0, rel=1e-6)
    assert model.r0_raw.requires_grad
    with torch.no_grad():
        model.r0_raw.fill_(-50.0)
    assert model.r0.item() >= 0.0                # softplus keeps it positive


# ---------------------------------------------------------------------------
# Parameter heads
# ---------------------------------------------------------------------------

def test_parameters_start_at_the_prior(w2_dataset):
    feat, model = make_model()
    batch = w2_dataset.flat_batch([0])
    out = run(feat, model, batch)
    expected_c6 = torch.tensor(
        [DEFAULT_C6_PRIOR[int(z)][0] for z in batch.atomic_numbers], dtype=torch.float64
    )
    assert torch.allclose(out.c6, expected_c6, rtol=1e-12)


def test_unknown_element_raises():
    with pytest.raises(KeyError, match="no C6 prior"):
        build_log_priors([1, 8, 6])


def test_c6_prior_override():
    log_c6, log_b = build_log_priors([1, 8, 6], c6_prior={6: 30.0}, b_prior=2.0)
    assert log_c6.exp()[2].item() == pytest.approx(30.0)
    assert torch.allclose(log_b.exp(), torch.full((3,), 2.0, dtype=log_b.dtype))


def test_intra_fragment_grouping_makes_parameters_local(w2_dataset):
    """The ablation hook: with group_idx=fragment_idx, C6 ignores the other molecule."""
    feat, model = make_model(randomize=True)
    near = w2_dataset.flat_batch([0])
    far = w2_dataset.flat_batch([0])
    far.positions = far.positions.clone()
    far.positions[3:] += 50.0                    # push the second water away

    c6_near = model.params(feat(near, near.fragment_idx).inv_feats,
                           feat(near, near.fragment_idx).species_idx)[0]
    c6_far = model.params(feat(far, far.fragment_idx).inv_feats,
                          feat(far, far.fragment_idx).species_idx)[0]
    assert torch.equal(c6_near[:3], c6_far[:3])

    # With the default supersystem grouping the parameters *do* move -- which is the
    # environment dependence we opted into, not a bug.
    c6_super_near = model.params(feat(near).inv_feats, feat(near).species_idx)[0]
    c6_super_far = model.params(feat(far).inv_feats, feat(far).species_idx)[0]
    assert not torch.allclose(c6_super_near[:3], c6_super_far[:3])
