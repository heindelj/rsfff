"""Classical electrostatics: two-body exactness, charge conservation, and the pyCMM anchor.

The headline is :func:`test_model_is_exactly_two_body`. Every other term in the repo is a
pair sum whose *parameters* see the whole cluster, so it is non-additive; this one is
constrained to be exactly two-body because that is what classical electrostatics between
frozen densities is. The property is structural (descriptors and the charge solve are both
grouped by fragment) but it is easy to break by accident, so it is measured rather than
assumed.
"""

import math

import numpy as np
import pytest
import torch
from ase.io import read

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.electrostatics import (
    FragmentResponse,
    DEFAULT_ELEC_PRIOR,
    ElectrostaticParameterHeads,
    ElectrostaticsModel,
    SlaterElectrostatics,
    build_elec_priors,
)
from rsfff.ff.many_body import mbe_decompose
from rsfff.ff.multipole import (
    build_polytensor,
    damped_interaction_tensor,
    irrep2_to_spherical,
    multipole_pair_energy,
    slater_one_center_damp,
    slater_two_center_damp,
    spherical_to_cartesian_quadrupole,
)
from rsfff.ff.pairs import inter_fragment_pairs, intra_fragment_channels
from rsfff.ff.units import BOHR_ANG, KJMOL_PER_HARTREE
from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.mlip.sqe import PairComplianceHead, atomic_quadrupole_energy, sqe_solve
from rsfff.train.data import load_extxyz

from conftest import DATA_W2, DATA_W3, features_cfg

NEIGHBOR_TYPES = [1, 8]

#: Point + penetration at rank 1 over the inter-fragment pairs of w2 frame 0, driven with
#: pyCMM's fitted water multipoles and charge-penetration parameters. Frozen so a refactor
#: that changes the number has to say so out loud.
W2_FRAME0_POINT = -5.0772973690e-3
W2_FRAME0_PEN = -3.5519048317e-3


# ---------------------------------------------------------------------------
# Reference implementation (pyCMM water.xml + cmm/force_field.py:956-995)
# ---------------------------------------------------------------------------

PYCMM_LOCAL = {
    8: dict(q=-0.390896, mu=np.array([0.0, 0.0, -0.094298])),
    1: dict(q=0.195448, mu=np.array([0.0910288, 0.0, -0.207851])),
}


def _local_frame(pos, z_atom, x_atom, i, axis):
    zv = pos[z_atom] - pos[i]
    zv = zv / np.linalg.norm(zv)
    xv = pos[x_atom] - pos[i]
    xv = xv / np.linalg.norm(xv)
    if axis == "Bisector":
        zv = zv + xv
        zv = zv / np.linalg.norm(zv)
    xv = xv - np.dot(zv, xv) * zv
    xv = xv / np.linalg.norm(xv)
    return np.stack([xv, np.cross(zv, xv), zv])


def pycmm_water_multipoles(atoms):
    """pyCMM's permanent charges and dipoles rotated into the global frame, in a.u."""
    pos = atoms.get_positions() / BOHR_ANG
    z, frag = atoms.numbers, atoms.arrays["fragment_idx"]
    q = np.zeros(len(z))
    mu = np.zeros((len(z), 3))
    for f in np.unique(frag):
        idx = np.where(frag == f)[0]
        o = idx[z[idx] == 8][0]
        h = idx[z[idx] == 1]
        for atom, (za, xa, axis, key) in (
            (o, (h[0], h[1], "Bisector", 8)),
            (h[0], (o, h[1], "ZThenX", 1)),
            (h[1], (o, h[0], "ZThenX", 1)),
        ):
            rot = _local_frame(pos, za, xa, atom, axis)
            q[atom] = PYCMM_LOCAL[key]["q"]
            mu[atom] = PYCMM_LOCAL[key]["mu"] @ rot
    return torch.tensor(q), torch.tensor(mu)


def pycmm_frame_energy(atoms, cutoff=12.0):
    """``(E_point, E_pen)`` in Hartree over the inter-fragment pairs of one frame."""
    q, mu = pycmm_water_multipoles(atoms)
    z = torch.tensor(atoms.numbers)
    zc = torch.tensor([DEFAULT_ELEC_PRIOR[int(v)][0] for v in z])
    b = torch.tensor([DEFAULT_ELEC_PRIOR[int(v)][1] for v in z])
    pos = torch.tensor(atoms.get_positions())
    frag = torch.tensor(np.asarray(atoms.arrays["fragment_idx"], dtype=np.int64))

    pair, r = inter_fragment_pairs(
        pos, torch.zeros(len(z), dtype=torch.long), cutoff, fragment_idx=frag
    )
    i, j = pair[0], pair[1]
    dr = (pos[j] - pos[i]) / BOHR_ANG
    r_au = r / BOHR_ANG
    r_inv = 1.0 / r_au

    m_real = build_polytensor(q, mu)
    m_shell = build_polytensor(q - zc, mu)
    m_nuc = build_polytensor(zc, None)

    e_point = multipole_pair_energy(
        m_real[i], m_real[j], damped_interaction_tensor(dr, None, r_inv, max_rank=1)
    ).sum()
    b_ij = (0.5 * (b[i].log() + b[j].log())).exp()
    e_pen = (
        multipole_pair_energy(
            m_shell[i], m_shell[j],
            damped_interaction_tensor(
                dr, -slater_two_center_damp(b_ij * r_au, 1), r_inv, max_rank=1),
        )
        + multipole_pair_energy(
            m_shell[i], m_nuc[j],
            damped_interaction_tensor(
                dr, -slater_one_center_damp(b[i] * r_au, 1), r_inv, max_rank=1),
        )
        + multipole_pair_energy(
            m_nuc[i], m_shell[j],
            damped_interaction_tensor(
                dr, -slater_one_center_damp(b[j] * r_au, 1), r_inv, max_rank=1),
        )
    ).sum()
    return float(e_point), float(e_pen)


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
        neighbor_types=NEIGHBOR_TYPES, selected_lambdas=(0, 1, 2),
        backend=cfg.backend, density_channels=cfg.density_channels,
    )


def make_model(*, correction=False, max_rank=1, cutoff=12.0, r0_init=1e-6,
               randomize=False, seed=1, featurizer=None, **model_kw):
    """Model at the priors. ``r0_init`` defaults to ~0 so the switch is out of the way."""
    feat = featurizer or make_featurizer()
    p0, p1, p2 = feat.feature_dims[0], feat.feature_dims.get(1), feat.feature_dims.get(2)
    log_z, log_b, q0 = build_elec_priors(NEIGHBOR_TYPES)
    params = ElectrostaticParameterHeads(
        p0, p1, p2, len(NEIGHBOR_TYPES),
        log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
        irrep6_to_voigt=feat.backend.irrep6_to_voigt(),
        irrep2_to_spherical_map=irrep2_to_spherical(feat.backend.irrep6_to_voigt()),
        emb_dim=8, hidden=16, depth=1, equiv_channels=6, max_rank=max_rank,
    )
    head = PairEnergyHead(
        p0, len(NEIGHBOR_TYPES), emb_dim=8, hidden=16, depth=1, r_on=4.0, r_off=5.0
    ) if correction else None
    compliance = PairComplianceHead(p0, hidden=16, depth=1)
    elec = SlaterElectrostatics(
        FragmentResponse(params, compliance), head,
        cutoff=cutoff, r0_init=r0_init, max_rank=max_rank,
    )
    model = ElectrostaticsModel(feat, elec, **model_kw)
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


# ---------------------------------------------------------------------------
# The headline: exactly two-body
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [1, 2])
def test_model_is_exactly_two_body(w3_dataset, max_rank):
    """Every E^(k>=3) is zero to round-off -- the property this whole design buys.

    Not "small": the descriptors and the charge solve are both grouped by fragment, so the
    higher orders are structurally absent rather than merely tiny.
    """
    model = make_model(correction=True, max_rank=max_rank, randomize=True)
    for k in range(3):
        frame = w3_dataset.flat_batch([k])
        res = mbe_decompose(
            model, frame.positions, frame.atomic_numbers, frame.fragment_idx
        )
        assert abs(float(res.total)) > 1e-6, "need a nonzero total for this to mean anything"
        assert abs(float(res.by_order[3])) * KJMOL_PER_HARTREE < 1e-10


def test_removing_a_neighbor_leaves_the_others_parameters_untouched(w3_dataset):
    """The mechanism behind the test above, checked directly on the parameters."""
    model = make_model(randomize=True)
    batch = w3_dataset.flat_batch([0])
    with torch.no_grad():
        base = model(batch)
    moved = w3_dataset.flat_batch([0])
    moved.positions = moved.positions.clone()
    moved.positions[6:] += torch.tensor([6.0, 0.0, 0.0])
    with torch.no_grad():
        out = model(moved)
    for name in ("charges", "chi", "eta", "z", "b"):
        a, b_ = getattr(base, name)[:6], getattr(out, name)[:6]
        assert torch.equal(a, b_), f"{name} moved when a distant monomer did"


# ---------------------------------------------------------------------------
# The energy function against pyCMM
# ---------------------------------------------------------------------------

def test_energy_function_reproduces_pycmm():
    """Anchor on w2 frame 0: frozen constants *and* an independent transcription.

    Drives the multipole machinery directly with pyCMM's parameters, so it pins the formula,
    the a.u. conversion and the pair list without involving the SQE solve at all.
    """
    atoms = read(DATA_W2, index="0")
    point, pen = pycmm_frame_energy(atoms)
    assert point == pytest.approx(W2_FRAME0_POINT, rel=1e-10)
    assert pen == pytest.approx(W2_FRAME0_PEN, rel=1e-10)
    # Penetration is not a rounding correction: it is ~41% of the total here.
    assert abs(pen) > 0.3 * abs(point + pen)


def test_penetration_dies_at_long_range_while_the_point_term_survives():
    """The 1/r tail must be exactly the undamped multipole interaction."""
    q = torch.tensor([1.0, -1.0])
    zc = torch.tensor([3.61565, 0.93619])
    b = torch.tensor([2.13358, 2.33322])
    for r_ang, tol in ((10.0, 1e-6), (25.0, 1e-12)):
        dr = torch.tensor([[r_ang / BOHR_ANG, 0.0, 0.0]])
        r_au = dr.norm(dim=-1)
        r_inv = 1.0 / r_au
        i = torch.tensor([0])
        j = torch.tensor([1])
        m_real = build_polytensor(q, None)
        m_shell = build_polytensor(q - zc, None)
        m_nuc = build_polytensor(zc, None)
        point = multipole_pair_energy(
            m_real[i], m_real[j], damped_interaction_tensor(dr, None, r_inv, max_rank=1)
        )
        b_ij = (b[0] * b[1]).sqrt().reshape(1)
        pen = (
            multipole_pair_energy(m_shell[i], m_shell[j], damped_interaction_tensor(
                dr, -slater_two_center_damp(b_ij * r_au, 1), r_inv, max_rank=1))
            + multipole_pair_energy(m_shell[i], m_nuc[j], damped_interaction_tensor(
                dr, -slater_one_center_damp(b[0] * r_au, 1), r_inv, max_rank=1))
            + multipole_pair_energy(m_nuc[i], m_shell[j], damped_interaction_tensor(
                dr, -slater_one_center_damp(b[1] * r_au, 1), r_inv, max_rank=1))
        )
        assert abs(float(pen) / float(point)) < tol, f"at {r_ang} A"
        # and the point term is exactly the Coulomb law
        assert float(point) == pytest.approx(float(q[0] * q[1] * r_inv), rel=1e-12)


# ---------------------------------------------------------------------------
# The local solve
# ---------------------------------------------------------------------------

def test_fragment_charge_is_conserved_exactly(w2_dataset, w3_dataset):
    """Identically, for any parameters -- a property of SQE's coordinates, not the solution."""
    for ds in (w2_dataset, w3_dataset):
        for randomize in (False, True):
            model = make_model(randomize=randomize, seed=3)
            batch = ds.flat_batch(range(6))
            with torch.no_grad():
                out = model(batch)
            per_frag = out.charges.new_zeros(int(batch.n_fragments)).index_add_(
                0, batch.fragment_idx, out.charges
            )
            assert per_frag.abs().max() < 1e-13


def test_complete_channel_graph_reduces_to_the_covalent_one(w2_dataset):
    """The identity the channel enumeration rests on: s(H-H) -> 0 recovers O-H-only charges.

    If this ever fails, ``intra_fragment_channels`` is no longer a superset of the covalent
    graph and the "let the model learn the topology" argument is void.
    """
    batch = w2_dataset.flat_batch([0, 1])
    frag, pos, z = batch.fragment_idx, batch.positions, batch.atomic_numbers
    n_frag = int(batch.n_fragments)
    g = torch.Generator().manual_seed(0)
    chi = torch.randn(len(pos), generator=g) * 0.2
    eta = torch.rand(len(pos), generator=g) + 0.4
    q0 = torch.zeros(len(pos))

    full_ij, full_bb = intra_fragment_channels(frag)
    is_hh = (z[full_ij[0]] == 1) & (z[full_ij[1]] == 1)
    cov_ij, cov_bb = full_ij[:, ~is_hh], full_bb[~is_hh]

    def solve(ij, bb, s):
        return sqe_solve(chi, eta, s, q0, pos, ij, frag, bb, n_frag,
                         with_polarizability=False).charges

    covalent = solve(cov_ij, cov_bb, torch.full((cov_ij.shape[1],), 0.5))
    s_full = torch.full((full_ij.shape[1],), 0.5)
    s_closed = s_full.clone()
    s_closed[is_hh] = 0.0
    torch.testing.assert_close(solve(full_ij, full_bb, s_closed), covalent, rtol=0, atol=1e-14)
    # ... and the extra channel is not inert when it is open
    assert (solve(full_ij, full_bb, s_full) - covalent).abs().max() > 1e-4


def test_internal_energy_is_excluded_from_the_interaction(w2_dataset):
    """The 1-body cost of polarizing a monomer must not leak into the interaction energy."""
    model = make_model(randomize=True)
    batch = w2_dataset.flat_batch(range(4))
    with torch.no_grad():
        out = model(batch)
    assert out.internal_energy.abs().max() > 1e-8, "nothing to guard against otherwise"
    torch.testing.assert_close(out.energy, out.energy_ff + out.energy_corr, rtol=0, atol=0)
    torch.testing.assert_close(out.energy_ff, out.energy_point + out.energy_pen)


# ---------------------------------------------------------------------------
# The rank-2 response sector
# ---------------------------------------------------------------------------

def test_quadrupoles_come_from_the_response_solve(w2_dataset):
    """``Theta == -c * chiquad`` exactly: the sector is a solve, not a free head."""
    model = make_model(max_rank=2, randomize=True)
    with torch.no_grad():
        out = model(w2_dataset.flat_batch(range(3)))
    assert out.chiquad is not None and out.cquad is not None
    torch.testing.assert_close(
        out.quad_s, -out.cquad.unsqueeze(-1) * out.chiquad, rtol=0, atol=0
    )
    # ...and c is strictly positive, so the sector can only lower the internal energy.
    assert float(out.cquad.min()) > 0.0


def test_quadrupole_energy_is_the_minimum_of_its_own_functional(w2_dataset):
    """Perturbing Theta off ``-C chiquad`` raises the on-site quadratic.

    ``E(Theta) = chiquad . Theta + 1/2 Theta^T C^-1 Theta`` is what
    ``atomic_quadrupole_energy`` reports the minimized value of, without ever forming
    ``C^-1``. Reconstructing it here from ``C`` is the check that the closed form is the
    minimum rather than merely some value.
    """
    model = make_model(max_rank=2, randomize=True)
    batch = w2_dataset.flat_batch(range(2))
    with torch.no_grad():
        out = model(batch)
        n_frag = int(batch.n_fragments)
        e_min = atomic_quadrupole_energy(
            out.chiquad, out.cquad, batch.fragment_idx, n_frag
        )

        def functional(theta):
            per_atom = (out.chiquad * theta).sum(-1) + 0.5 * theta.pow(2).sum(-1) / out.cquad
            return per_atom.new_zeros(n_frag).index_add_(0, batch.fragment_idx, per_atom)

        torch.testing.assert_close(functional(out.quad_s), e_min, rtol=1e-10, atol=1e-14)
        assert float(e_min.max()) <= 0.0
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            perturbed = out.quad_s + 0.1 * torch.randn(
                out.quad_s.shape, generator=g, dtype=out.quad_s.dtype
            )
            assert bool((functional(perturbed) > e_min).all())


def test_the_quadrupole_sector_contributes_to_internal_energy_only(w2_dataset):
    """Turning on max_rank 2 must move ``internal_energy`` -- and only via this sector."""
    feat = make_featurizer()
    batch = w2_dataset.flat_batch(range(3))
    with torch.no_grad():
        rank1 = make_model(max_rank=1, featurizer=feat)(batch)
        rank2 = make_model(max_rank=2, featurizer=feat)(batch)
    # The chiquad head is zero-initialized, so at init rank 2 is rank 1 plus exactly zero.
    assert torch.equal(rank2.quad_s, torch.zeros_like(rank2.quad_s))
    torch.testing.assert_close(rank2.internal_energy, rank1.internal_energy, rtol=0, atol=0)
    torch.testing.assert_close(rank2.energy, rank1.energy, rtol=1e-12, atol=1e-15)

    # With a nonzero chiquad the internal energy drops (C is PSD) while `energy` still
    # excludes it -- energy_ff + energy_corr, as always.
    model = make_model(max_rank=2, randomize=True, featurizer=feat)
    with torch.no_grad():
        out = model(batch)
    assert float(out.quad_s.abs().max()) > 0.0
    torch.testing.assert_close(out.energy, out.energy_ff + out.energy_corr, rtol=0, atol=0)


def test_mismatched_max_rank_raises(w2_dataset):
    """Heads at one rank and the interaction tensor at another is silent otherwise."""
    feat = make_featurizer()
    p0, p1, p2 = feat.feature_dims[0], feat.feature_dims.get(1), feat.feature_dims.get(2)
    log_z, log_b, q0 = build_elec_priors(NEIGHBOR_TYPES)
    params = ElectrostaticParameterHeads(
        p0, p1, p2, len(NEIGHBOR_TYPES),
        log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
        irrep6_to_voigt=feat.backend.irrep6_to_voigt(),
        irrep2_to_spherical_map=irrep2_to_spherical(feat.backend.irrep6_to_voigt()),
        emb_dim=8, hidden=16, depth=1, equiv_channels=6, max_rank=1,
    )
    with pytest.raises(ValueError, match="max_rank"):
        SlaterElectrostatics(
            FragmentResponse(params, PairComplianceHead(p0, hidden=16, depth=1)),
            None, max_rank=2,
        )


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [1, 2])
def test_rotation_invariance(w2_dataset, max_rank):
    """Proves chivec/alpha (and the quadrupole) stayed equivariant end to end."""
    model = make_model(correction=True, max_rank=max_rank, randomize=True)
    theta = 0.8
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    batch = w2_dataset.flat_batch(range(4))
    rotated = w2_dataset.flat_batch(range(4))
    rotated.positions = rotated.positions @ R.T
    with torch.no_grad():
        a, b = model(batch), model(rotated)
    torch.testing.assert_close(a.energy, b.energy, rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(a.charges, b.charges, rtol=1e-11, atol=1e-14)
    if a.mu is not None:
        torch.testing.assert_close(a.mu @ R.T, b.mu, rtol=1e-9, atol=1e-13)
    if a.quad_s is not None:
        # Checked in Cartesian form, where the transformation law is R Theta R^T and does
        # not require constructing a Wigner-D. The energy alone would not catch a
        # quadrupole that transformed wrongly but consistently on both atoms of a pair.
        qa = spherical_to_cartesian_quadrupole(a.quad_s)
        qb = spherical_to_cartesian_quadrupole(b.quad_s)
        torch.testing.assert_close(R @ qa @ R.T, qb, rtol=1e-9, atol=1e-13)


def test_translation_invariance(w2_dataset):
    model = make_model(correction=True, randomize=True)
    batch = w2_dataset.flat_batch(range(4))
    shifted = w2_dataset.flat_batch(range(4))
    shifted.positions = shifted.positions + torch.tensor([2.5, -1.0, 4.0])
    with torch.no_grad():
        a, b = model(batch), model(shifted)
    torch.testing.assert_close(a.energy, b.energy, rtol=1e-10, atol=1e-14)


def test_batching_matches_single_frames(w3_dataset):
    model = make_model(correction=True, randomize=True)
    with torch.no_grad():
        singles = torch.tensor(
            [float(model(w3_dataset.flat_batch([k])).energy) for k in (0, 1, 2)],
            dtype=torch.float64,
        )
        batched = model(w3_dataset.flat_batch([0, 1, 2])).energy
    torch.testing.assert_close(batched, singles, rtol=1e-11, atol=1e-15)


# ---------------------------------------------------------------------------
# The switch, the correction, and configuration errors
# ---------------------------------------------------------------------------

def test_fresh_correction_contributes_exactly_nothing(w2_dataset):
    model = make_model(correction=True)
    with torch.no_grad():
        out = model(w2_dataset.flat_batch(range(4)))
    assert torch.equal(out.energy_corr, torch.zeros_like(out.energy_corr))
    torch.testing.assert_close(out.energy, out.energy_ff, rtol=0, atol=0)


def test_range_separation_limits(w2_dataset):
    """r0 -> 0 leaves the backbone alone; r0 -> large switches it off."""
    batch = w2_dataset.flat_batch(range(3))
    feat = make_featurizer()
    with torch.no_grad():
        wide = make_model(r0_init=40.0, featurizer=feat)(batch)
        narrow = make_model(r0_init=1e-6, featurizer=feat)(batch)
    assert wide.energy_ff.abs().max() < 1e-12
    assert narrow.energy_ff.abs().max() > 1e-4


def test_forces_match_central_differences(w2_dataset):
    """Autograd through the SQE solve, the damping and the switch together."""
    model = make_model(correction=True, randomize=True)
    base = w2_dataset.flat_batch([0])
    pos = base.positions.clone().requires_grad_(True)
    base.positions = pos
    model(base).energy.sum().backward()
    analytic = pos.grad.clone()

    h = 1e-5
    for atom in (0, 4):
        numeric = torch.zeros(3, dtype=torch.float64)
        for a in range(3):
            vals = []
            for sign in (+1, -1):
                b = w2_dataset.flat_batch([0])
                b.positions = b.positions.clone()
                b.positions[atom, a] += sign * h
                with torch.no_grad():
                    vals.append(float(model(b).energy))
            numeric[a] = (vals[0] - vals[1]) / (2 * h)
        torch.testing.assert_close(analytic[atom], numeric, rtol=2e-5, atol=1e-8)


def test_environment_features_require_an_explicit_override():
    """Losing two-body exactness must be a decision, not a default."""
    feat = make_featurizer()
    with pytest.raises(ValueError, match="allow_environment"):
        make_model(featurizer=feat, intra_fragment=False)
    make_model(featurizer=feat, intra_fragment=False, allow_environment=True)


def test_unknown_element_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="Refusing to guess"):
        build_elec_priors([1, 6, 8])


def test_kjmol_scale_is_sane(w2_dataset):
    """A units guard: cls_elec on water dimers is tens of kJ/mol, not tens of Hartree."""
    model = make_model()
    with torch.no_grad():
        out = model(w2_dataset.flat_batch(range(30)))
    mean = float(out.energy.mean()) * KJMOL_PER_HARTREE
    assert -300.0 < mean < 100.0, f"{mean} kJ/mol"
