"""Slater-damped multipolar Pauli: formula, units, pair list, symmetry, and forces.

The cross-repo test against pyCMM is the load-bearing one -- it pins the damping
polynomials, the sqrt(b_i b_j) combination, the Angstrom->bohr conversion, and the
inter-fragment pair list at once, against an implementation fit to real data.

The rotation test is the one that matters for the dipole head specifically: the Pauli
dipole comes from lambda=1 features, and nothing else in this file would notice if that
path lost its equivariance.
"""

import math

import numpy as np
import pytest
import torch
from ase.io import read

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.multipole import irrep2_to_spherical, spherical_to_cartesian_quadrupole
from rsfff.ff.pauli import (
    DEFAULT_PAULI_PRIOR,
    PauliModel,
    PauliMultipoleHeads,
    SlaterPauli,
    build_pauli_priors,
)
from rsfff.ff.units import BOHR_ANG, KJMOL_PER_HARTREE
from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.train.data import load_extxyz

from conftest import DATA_W2, DATA_W3, features_cfg

NEIGHBOR_TYPES = [1, 8]

#: Charges-only Slater Pauli at the pyCMM priors over the 9 inter-fragment pairs of the
#: first w2 frame, measured during planning. Frozen so a refactor that changes the number
#: has to say so out loud.
W2_FRAME0_ANCHOR = 1.9912249080e-2


# ---------------------------------------------------------------------------
# Reference implementation, transcribed from pyCMM/cmm/short_range.py:22-39
# ---------------------------------------------------------------------------

def pycmm_charge_pauli(r_bohr, q_i, q_j, b_i, b_j):
    """pyCMM's charge-charge Pauli term. Distances in bohr, energy in Hartree."""
    u = math.sqrt(b_i * b_j) * r_bohr
    f1 = (1 + 11 * u / 16 + 3 * u**2 / 16 + u**3 / 48) * math.exp(-u)
    return f1 * q_i * q_j / r_bohr


def pycmm_frame_energy(atoms, cutoff=7.0):
    """Sum pyCMM charge-only Pauli over the inter-fragment pairs of one ASE frame."""
    pos, Z, fi = atoms.get_positions(), atoms.numbers, atoms.arrays["fragment_idx"]
    total = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if fi[i] == fi[j]:
                continue
            r_ang = float(np.linalg.norm(pos[i] - pos[j]))
            if r_ang > cutoff:
                continue
            total += pycmm_charge_pauli(
                r_ang / BOHR_ANG,
                DEFAULT_PAULI_PRIOR[Z[i]][0], DEFAULT_PAULI_PRIOR[Z[j]][0],
                DEFAULT_PAULI_PRIOR[Z[i]][1], DEFAULT_PAULI_PRIOR[Z[j]][1],
            )
    return total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def w2_dataset():
    return load_extxyz(DATA_W2, dtype=torch.float64)


@pytest.fixture(scope="module")
def w3_dataset():
    return load_extxyz(DATA_W3, dtype=torch.float64)


def make_featurizer():
    cfg = features_cfg()
    return FlatLambdaSOAPFeaturizer(
        cutoff=cfg.cutoff, n_max=cfg.n_max, l_max=cfg.l_max,
        neighbor_types=NEIGHBOR_TYPES, selected_lambdas=cfg.selected_lambdas,
        backend=cfg.backend, density_channels=cfg.density_channels,
    )


def make_model(*, correction=False, max_rank=1, cutoff=7.0, taper_width=1.0,
               inter_only=True, environment_q=False, learn_dipole=True,
               randomize=False, seed=1, featurizer=None):
    """SlaterPauli at the pyCMM priors. Defaults to the bare backbone."""
    feat = featurizer or make_featurizer()
    p0, p1, p2 = feat.feature_dims[0], feat.feature_dims.get(1), feat.feature_dims.get(2)
    log_q, log_b, mu_scale, quad_scale = build_pauli_priors(NEIGHBOR_TYPES)
    params = PauliMultipoleHeads(
        p0, p1, len(NEIGHBOR_TYPES),
        log_q_prior=log_q, log_b_prior=log_b, dipole_scale=mu_scale,
        p2=p2, quad_scale=quad_scale,
        irrep2_to_spherical=irrep2_to_spherical(feat.backend.irrep6_to_voigt()),
        emb_dim=8, hidden=16, depth=1, equiv_channels=6,
        max_rank=max_rank, environment_q=environment_q, learn_dipole=learn_dipole,
    )
    head = PairEnergyHead(
        p0, len(NEIGHBOR_TYPES), emb_dim=8, hidden=16, depth=1, r_on=4.0, r_off=5.0
    ) if correction else None
    model = SlaterPauli(
        params, head, cutoff=cutoff, taper_width=taper_width,
        max_rank=max_rank, inter_only=inter_only,
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

def test_reproduces_pycmm_on_the_first_w2_frame(w2_dataset):
    """The anchor: pinned constant *and* an independent inline transcription."""
    batch = w2_dataset.flat_batch([0])
    feat, model = make_model(max_rank=0)
    with torch.no_grad():
        out = run(feat, model, batch)
    assert out.pair_index.shape[1] == 9, "w2 has 9 inter-fragment pairs inside 7 A"
    energy = float(out.energy)
    assert energy == pytest.approx(W2_FRAME0_ANCHOR, rel=1e-10)

    atoms = read(DATA_W2, index="0")
    assert energy == pytest.approx(pycmm_frame_energy(atoms), rel=1e-10)


def test_reproduces_pycmm_over_many_frames(w2_dataset):
    """Not just one lucky geometry -- the whole pair list and mask, 20 frames deep."""
    batch = w2_dataset.flat_batch(range(20))
    feat, model = make_model(max_rank=0)
    with torch.no_grad():
        out = run(feat, model, batch)
    want = torch.tensor(
        [pycmm_frame_energy(a) for a in read(DATA_W2, index=":20")], dtype=torch.float64
    )
    torch.testing.assert_close(out.energy, want, rtol=1e-10, atol=1e-14)


def test_energy_is_repulsive_at_the_priors(w2_dataset):
    """Positive q on every atom makes the monopole term repulsive by construction."""
    feat, model = make_model(max_rank=0)
    with torch.no_grad():
        out = run(feat, model, w2_dataset.flat_batch(range(50)))
    assert torch.all(out.energy > 0.0)


def test_decays_exponentially_not_as_a_power_law():
    """The undamped tail is subtracted off, so this is short-ranged by construction.

    Stated as the leading asymptotic ``f_1(u) -> (u^3/48) exp(-u)``, which is exact rather
    than a threshold: the cubic prefactor is why a naive "decay per unit u" bound, or even
    ``-log f / u``, converges to 1 far too slowly to assert on.
    """
    from rsfff.ff.multipole import slater_two_center_damp
    u = torch.tensor([50.0, 300.0])
    ratio = slater_two_center_damp(u, 0)[0] * u.exp() * 48.0 / u.pow(3)
    assert float(ratio[1]) < float(ratio[0])              # approaching from above
    assert abs(float(ratio[1]) - 1.0) < 0.05, ratio


# ---------------------------------------------------------------------------
# Dipoles and equivariance
# ---------------------------------------------------------------------------

def test_dipoles_start_at_zero_so_rank1_equals_rank0(w2_dataset):
    """AtomicVectorHead is zero-init: the dipole is a pure addition, not a perturbation.

    ``mu`` is *exactly* zero; the two energies then agree to round-off rather than bitwise,
    because the rank-1 contraction sums a 4x4 block whose accumulation order BLAS is free
    to choose.
    """
    batch = w2_dataset.flat_batch(range(5))
    feat = make_featurizer()
    _, m0 = make_model(max_rank=0, featurizer=feat)
    _, m1 = make_model(max_rank=1, featurizer=feat)
    with torch.no_grad():
        e0, out1 = run(feat, m0, batch).energy, run(feat, m1, batch)
    assert torch.equal(out1.mu, torch.zeros_like(out1.mu))
    torch.testing.assert_close(out1.energy, e0, rtol=1e-14, atol=0)


def test_perturbed_dipole_head_actually_changes_the_energy(w2_dataset):
    """Guards the test above from passing because the dipole path is dead."""
    batch = w2_dataset.flat_batch(range(5))
    feat = make_featurizer()
    _, m0 = make_model(max_rank=0, featurizer=feat)
    _, m1 = make_model(max_rank=1, featurizer=feat, randomize=True)
    with torch.no_grad():
        out1 = run(feat, m1, batch)
        e0 = run(feat, m0, batch).energy
    assert out1.mu.abs().max() > 1e-6
    assert (out1.energy - e0).abs().max() > 1e-6


@pytest.mark.parametrize("max_rank", [0, 1, 2])
def test_rotation_invariance(w2_dataset, max_rank):
    """The end-to-end equivariance check: features, dipole head, and tensor together."""
    batch = w2_dataset.flat_batch(range(4))
    feat, model = make_model(max_rank=max_rank, randomize=True)
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    rotated = w2_dataset.flat_batch(range(4))
    rotated.positions = rotated.positions @ R.T
    with torch.no_grad():
        out, out_r = run(feat, model, batch), run(feat, model, rotated)
    torch.testing.assert_close(out.energy, out_r.energy, rtol=1e-10, atol=1e-14)
    # and the multipoles themselves rotate rather than staying put
    if max_rank >= 1:
        torch.testing.assert_close(out.mu @ R.T, out_r.mu, rtol=1e-9, atol=1e-13)
    if max_rank == 2:
        q_c = spherical_to_cartesian_quadrupole(out.quad_s)
        q_c_r = spherical_to_cartesian_quadrupole(out_r.quad_s)
        # a rank-2 tensor transforms as R Q R^T, which is what would break if the
        # lambda=2 -> spherical change of basis were a hand-written relabeling
        torch.testing.assert_close(
            torch.einsum("ab,pbc,dc->pad", R, q_c, R), q_c_r, rtol=1e-9, atol=1e-13
        )


def test_translation_invariance(w2_dataset):
    batch = w2_dataset.flat_batch(range(4))
    feat, model = make_model(randomize=True)
    shifted = w2_dataset.flat_batch(range(4))
    shifted.positions = shifted.positions + torch.tensor([3.1, -2.0, 0.7])
    with torch.no_grad():
        e, e_shift = run(feat, model, batch).energy, run(feat, model, shifted).energy
    torch.testing.assert_close(e, e_shift, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# Pair list, batching, permutation
# ---------------------------------------------------------------------------

def test_no_intramolecular_pair_survives_the_mask(w2_dataset, w3_dataset):
    for ds in (w2_dataset, w3_dataset):
        batch = ds.flat_batch(range(10))
        feat, model = make_model()
        with torch.no_grad():
            out = run(feat, model, batch)
        i, j = out.pair_index
        assert torch.all(batch.fragment_idx[i] != batch.fragment_idx[j])
        assert out.pair_index.shape[1] > 0


def test_inter_only_off_admits_intramolecular_pairs(w2_dataset):
    """The mask is doing real work -- turning it off changes the pair list."""
    batch = w2_dataset.flat_batch(range(3))
    feat, masked = make_model(inter_only=True)
    _, unmasked = make_model(inter_only=False, featurizer=feat)
    with torch.no_grad():
        n_masked = run(feat, masked, batch).pair_index.shape[1]
        n_all = run(feat, unmasked, batch).pair_index.shape[1]
    assert n_all == n_masked + 3 * 2 * 3, "w2 has 3 intra pairs per monomer, 2 monomers"


@pytest.mark.parametrize("dataset_name", ["w2_dataset", "w3_dataset"])
def test_batching_matches_single_frames(request, dataset_name):
    """Frames in a batch must not see each other -- the ragged pooling's core invariant."""
    dataset = request.getfixturevalue(dataset_name)
    feat, model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        singles = torch.tensor(
            [float(run(feat, model, dataset.flat_batch([k])).energy) for k in (0, 1, 2)],
            dtype=torch.float64,
        )
        batched = run(feat, model, dataset.flat_batch([0, 1, 2])).energy
    torch.testing.assert_close(batched, singles, rtol=1e-11, atol=1e-15)


def test_permuting_atoms_within_a_frame_leaves_the_energy_unchanged(w2_dataset):
    """fragment_idx must stay grouped, so permute whole monomers rather than atoms."""
    batch = w2_dataset.flat_batch([0])
    feat, model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        e = run(feat, model, batch).energy

    perm = torch.tensor([3, 4, 5, 0, 1, 2])
    swapped = w2_dataset.flat_batch([0])
    swapped.positions = swapped.positions[perm]
    swapped.atomic_numbers = swapped.atomic_numbers[perm]
    with torch.no_grad():
        e_perm = run(feat, model, swapped).energy
    torch.testing.assert_close(e, e_perm, rtol=1e-10, atol=1e-14)


def test_separated_monomers_give_exactly_zero(w2_dataset):
    """Past the taper the term is *exactly* zero, not merely small."""
    batch = w2_dataset.flat_batch([0])
    batch.positions = batch.positions.clone()
    batch.positions[3:] += torch.tensor([20.0, 0.0, 0.0])
    feat, model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        out = run(feat, model, batch)
    assert out.pair_index.shape[1] == 0
    assert float(out.energy) == 0.0


# ---------------------------------------------------------------------------
# The correction head
# ---------------------------------------------------------------------------

def test_fresh_correction_contributes_exactly_nothing(w2_dataset):
    batch = w2_dataset.flat_batch(range(5))
    feat, model = make_model(correction=True)
    with torch.no_grad():
        out = run(feat, model, batch)
    assert torch.equal(out.energy_corr, torch.zeros_like(out.energy_corr))
    torch.testing.assert_close(out.energy, out.energy_ff, rtol=0, atol=0)


def test_correction_is_short_ranged(w3_dataset):
    """dE is exactly zero past its envelope, whatever the pair-list cutoff is.

    Uses w3: a water *dimer* has no inter-fragment pair beyond ~4.8 A, so there would be
    nothing past ``r_off = 5.0`` for this to check.
    """
    batch = w3_dataset.flat_batch(range(10))
    feat, model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        out = run(feat, model, batch)
    far = out.r >= model.correction.r_off
    assert far.any(), "need some pairs past r_off for this test to mean anything"
    assert torch.equal(out.e_pair_corr[far], torch.zeros_like(out.e_pair_corr[far]))


# ---------------------------------------------------------------------------
# Continuity and forces
# ---------------------------------------------------------------------------

def test_energy_is_continuous_through_the_cutoff(w2_dataset):
    """Sweeping a monomer out through the taper.

    Phrased as ``analytic dE/dd`` vs a central difference rather than as a bound on the
    step size: the energy is genuinely steep here (it decays exponentially), so a raw step
    is large on the near side of the sweep for honest reasons. A truncation would instead
    show up as the two disagreeing -- a step sends the finite difference to ~E/h while the
    analytic derivative stays finite.

    ``d`` is the *displacement*, not a pair distance: the monomers already sit ~3.5 A apart
    along the offset direction, so the pair list only starts crossing the 6-7 A taper near
    d = 9. Sweeping 5.5-7.5 (the obvious guess) never reaches the cutoff at all.
    """
    feat, model = make_model(correction=True, cutoff=7.0, randomize=True)

    def energy_at(d, grad=False):
        batch = w2_dataset.flat_batch([0])
        shift = torch.zeros(3, dtype=torch.float64, requires_grad=grad)
        pos = batch.positions.clone()
        batch.positions = torch.cat(
            (pos[:3], pos[3:] + torch.tensor([d, 0.0, 0.0]) + shift), dim=0
        )
        return run(feat, model, batch).energy.sum(), shift

    h = 1e-6
    for d in (8.5, 9.0, 9.5, 10.0, 10.4):
        e, shift = energy_at(d, grad=True)
        e.backward()
        analytic = float(shift.grad[0])
        with torch.no_grad():
            numeric = (energy_at(d + h)[0] - energy_at(d - h)[0]) / (2 * h)
        assert analytic == pytest.approx(float(numeric), rel=1e-5, abs=1e-12), f"d={d}"

    with torch.no_grad():
        # every pair now beyond the 7 A cutoff: exactly zero, not merely small
        assert float(energy_at(11.0)[0]) == 0.0


def test_forces_match_central_differences(w2_dataset):
    """Autograd through damping, tensor, dipole head and taper, against finite differences."""
    feat, model = make_model(correction=True, randomize=True)
    base = w2_dataset.flat_batch([0])
    pos = base.positions.clone().requires_grad_(True)
    base.positions = pos
    run(feat, model, base).energy.sum().backward()
    analytic = pos.grad.clone()

    h = 1e-5
    numeric = torch.zeros_like(analytic)
    for a in range(3):
        for atom in (0, 4):
            vals = []
            for sign in (+1, -1):
                batch = w2_dataset.flat_batch([0])
                batch.positions = batch.positions.clone()
                batch.positions[atom, a] += sign * h
                with torch.no_grad():
                    vals.append(float(run(feat, model, batch).energy))
            numeric[atom, a] = (vals[0] - vals[1]) / (2 * h)
    for atom in (0, 4):
        torch.testing.assert_close(
            analytic[atom], numeric[atom], rtol=1e-5, atol=1e-8
        )


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

def test_dipoles_without_lambda1_features_raise():
    feat = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=3, l_max=2, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 2), backend="e3nn", density_channels=4,
    )
    log_q, log_b, mu_scale, quad_scale = build_pauli_priors(NEIGHBOR_TYPES)
    with pytest.raises(ValueError, match="lambda=1"):
        PauliMultipoleHeads(
            feat.feature_dims[0], feat.feature_dims.get(1), len(NEIGHBOR_TYPES),
            log_q_prior=log_q, log_b_prior=log_b, dipole_scale=mu_scale, max_rank=1,
        )


def test_unknown_element_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="Refusing to guess"):
        build_pauli_priors([1, 6, 8])


def test_inter_only_without_fragments_raises(w2_dataset):
    batch = w2_dataset.flat_batch([0])
    batch.fragment_idx = None
    feat, model = make_model()
    with pytest.raises(ValueError, match="fragment_idx"):
        run(feat, model, batch)


def test_pair_weight_hook_scales_the_whole_pair(w2_dataset):
    """The reactivity hook: a constant weight scales FF and correction alike."""
    batch = w2_dataset.flat_batch(range(3))
    feat, model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        plain = run(feat, model, batch)
    # Same module, only the hook toggled -- two make_model calls would draw different
    # nn.Linear initializations from the global RNG and compare unrelated models.
    model.pair_weight = lambda b, idx, r: torch.full_like(r, 0.25)
    with torch.no_grad():
        weighted = run(feat, model, batch)
    torch.testing.assert_close(0.25 * plain.energy_ff, weighted.energy_ff)
    torch.testing.assert_close(0.25 * plain.energy_corr, weighted.energy_corr)
    assert weighted.energy_corr.abs().max() > 0.0


def test_kjmol_scale_is_sane(w2_dataset):
    """A units guard: mod_pauli on water dimers is tens of kJ/mol, not tens of Hartree."""
    feat, model = make_model(max_rank=0)
    with torch.no_grad():
        out = run(feat, model, w2_dataset.flat_batch(range(50)))
    mean = float(out.energy.mean()) * KJMOL_PER_HARTREE
    assert 10.0 < mean < 200.0, f"{mean} kJ/mol"


# ---------------------------------------------------------------------------
# Quadrupoles (rank 2)
# ---------------------------------------------------------------------------

def test_quadrupoles_start_at_zero_so_rank2_equals_rank1(w2_dataset):
    """AtomicQuadrupoleHead is zero-init, so rank 2 is a pure addition over rank 1."""
    batch = w2_dataset.flat_batch(range(5))
    feat = make_featurizer()
    _, m1 = make_model(max_rank=1, featurizer=feat)
    _, m2 = make_model(max_rank=2, featurizer=feat)
    with torch.no_grad():
        e1, out2 = run(feat, m1, batch).energy, run(feat, m2, batch)
    assert torch.equal(out2.quad_s, torch.zeros_like(out2.quad_s))
    torch.testing.assert_close(out2.energy, e1, rtol=1e-13, atol=0)


def test_perturbed_quadrupole_head_actually_changes_the_energy(w2_dataset):
    """Guards the test above from passing because the quadrupole path is dead."""
    batch = w2_dataset.flat_batch(range(5))
    feat = make_featurizer()
    _, m1 = make_model(max_rank=1, featurizer=feat)
    _, m2 = make_model(max_rank=2, featurizer=feat, randomize=True)
    with torch.no_grad():
        out2 = run(feat, m2, batch)
        e1 = run(feat, m1, batch).energy
    assert out2.quad_s.abs().max() > 1e-6
    assert (out2.energy - e1).abs().max() > 1e-6


def test_learned_quadrupoles_are_traceless(w2_dataset):
    """Structural, because the head emits 5 spherical components rather than 6 Cartesian."""
    feat, model = make_model(max_rank=2, randomize=True)
    with torch.no_grad():
        out = run(feat, model, w2_dataset.flat_batch(range(5)))
    q_c = spherical_to_cartesian_quadrupole(out.quad_s)
    torch.testing.assert_close(
        torch.einsum("paa->p", q_c), torch.zeros(q_c.shape[0]), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(q_c, q_c.transpose(-1, -2))


def test_rank2_forces_match_central_differences(w2_dataset):
    """Autograd through the 10x10 tensor, p7/p9 damping and the quadrupole head."""
    feat, model = make_model(correction=True, max_rank=2, randomize=True)
    base = w2_dataset.flat_batch([0])
    pos = base.positions.clone().requires_grad_(True)
    base.positions = pos
    run(feat, model, base).energy.sum().backward()
    analytic = pos.grad.clone()

    h = 1e-5
    for atom in (0, 4):
        numeric = torch.zeros(3, dtype=torch.float64)
        for a in range(3):
            vals = []
            for sign in (+1, -1):
                batch = w2_dataset.flat_batch([0])
                batch.positions = batch.positions.clone()
                batch.positions[atom, a] += sign * h
                with torch.no_grad():
                    vals.append(float(run(feat, model, batch).energy))
            numeric[a] = (vals[0] - vals[1]) / (2 * h)
        torch.testing.assert_close(analytic[atom], numeric, rtol=1e-5, atol=1e-8)


def test_quadrupoles_without_lambda2_features_raise():
    log_q, log_b, mu_scale, quad_scale = build_pauli_priors(NEIGHBOR_TYPES)
    feat = make_featurizer()
    with pytest.raises(ValueError, match="lambda=2"):
        PauliMultipoleHeads(
            feat.feature_dims[0], feat.feature_dims.get(1), len(NEIGHBOR_TYPES),
            log_q_prior=log_q, log_b_prior=log_b, dipole_scale=mu_scale,
            p2=None, quad_scale=quad_scale, max_rank=2,
        )


def test_quadrupoles_without_the_change_of_basis_raise():
    """max_rank=2 must not silently fall back to the raw lambda=2 slots."""
    log_q, log_b, mu_scale, quad_scale = build_pauli_priors(NEIGHBOR_TYPES)
    feat = make_featurizer()
    with pytest.raises(ValueError, match="irrep2_to_spherical"):
        PauliMultipoleHeads(
            feat.feature_dims[0], feat.feature_dims.get(1), len(NEIGHBOR_TYPES),
            log_q_prior=log_q, log_b_prior=log_b, dipole_scale=mu_scale,
            p2=feat.feature_dims.get(2), quad_scale=quad_scale,
            irrep2_to_spherical=None, max_rank=2,
        )
