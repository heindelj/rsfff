"""Slater damping factors and the damped multipole interaction tensor.

The convention tests are the load-bearing ones. Charge-charge and dipole-dipole are both
symmetric under swapping i and j, so they survive a sign error in the contraction order or
in the direction of ``dr_vec``; only the charge-dipole cross terms notice. Those are
checked against their closed forms and against the swap identity explicitly.
"""

import itertools
import math

import pytest
import torch

from rsfff.features.features import _resolve_backend
from rsfff.ff.multipole import (
    build_polytensor,
    cartesian_to_spherical_quadrupole,
    damped_interaction_tensor,
    irrep2_to_spherical,
    multipole_pair_energy,
    slater_one_center_damp,
    slater_two_center_damp,
    spherical_to_cartesian_quadrupole,
)
from rsfff.mlip.response_heads import voigt_vector_to_symmetric_matrix


# ---------------------------------------------------------------------------
# Reference implementation, transcribed from pyCMM/cmm/short_range.py:6-39
# ---------------------------------------------------------------------------

def pycmm_two_center(u):
    e = math.exp(-u)
    p1 = 1 + 11 * u / 16 + 3 * u**2 / 16 + u**3 / 48
    p3 = 1 + u + u**2 / 2 + 7 * u**3 / 48 + u**4 / 48
    p5 = 1 + u + u**2 / 2 + u**3 / 6 + u**4 / 24 + u**5 / 144
    return [p * e for p in (p1, p3, p5)]


def pycmm_one_center(u):
    e = math.exp(-u)
    p1 = 1 + u / 2
    p3 = 1 + u + u**2 / 2
    p5 = p3 + u**3 / 6
    return [p * e for p in (p1, p3, p5)]


@pytest.fixture
def pairs():
    """A handful of random pairs with random charges and dipoles, in atomic units."""
    g = torch.Generator().manual_seed(0)
    dr = torch.randn(7, 3, generator=g) * 2.0
    return dict(
        dr=dr,
        r=dr.norm(dim=-1),
        q_i=torch.randn(7, generator=g),
        q_j=torch.randn(7, generator=g),
        mu_i=torch.randn(7, 3, generator=g),
        mu_j=torch.randn(7, 3, generator=g),
        b_ij=torch.rand(7, generator=g) + 1.5,
    )


# ---------------------------------------------------------------------------
# Damping factors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("damp_fn", [slater_two_center_damp, slater_one_center_damp])
@pytest.mark.parametrize("max_rank", [0, 1])
def test_damp_is_one_at_zero(damp_fn, max_rank):
    """f_n(0) == 1 exactly: no cancellation, so no small-u series branch is needed."""
    f = damp_fn(torch.zeros(3), max_rank)
    assert f.shape[0] == 2 * max_rank + 1
    assert torch.equal(f, torch.ones_like(f))


@pytest.mark.parametrize(
    "damp_fn,reference", [(slater_two_center_damp, pycmm_two_center),
                          (slater_one_center_damp, pycmm_one_center)]
)
def test_damp_matches_pycmm(damp_fn, reference):
    """Against pyCMM's polynomials transcribed inline (pyCMM is not a dependency)."""
    u = torch.tensor([0.05, 0.5, 1.0, 3.0, 7.0, 20.0])
    got = damp_fn(u, 1)
    want = torch.tensor([reference(float(x)) for x in u]).T
    torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("damp_fn", [slater_two_center_damp, slater_one_center_damp])
def test_damp_decays_monotonically(damp_fn):
    """Strictly decreasing to zero -- this is what makes the term short-ranged."""
    u = torch.linspace(0.0, 40.0, 400)
    f = damp_fn(u, 1)
    assert torch.all(f[:, 1:] < f[:, :-1])
    assert torch.all(f[:, -1] < 1e-10)
    assert torch.all(f >= 0.0)


def test_damp_rank_prefix():
    """max_rank=0 returns exactly the first factor of the max_rank=1 stack."""
    u = torch.linspace(0.1, 10.0, 50)
    torch.testing.assert_close(
        slater_two_center_damp(u, 0), slater_two_center_damp(u, 1)[:1]
    )


def test_rank_above_two_raises():
    with pytest.raises(ValueError, match="max_rank"):
        slater_two_center_damp(torch.ones(2), 3)


# ---------------------------------------------------------------------------
# The interaction tensor
# ---------------------------------------------------------------------------

def test_rank0_is_damped_coulomb(pairs):
    damp = slater_two_center_damp(pairs["b_ij"] * pairs["r"], 0)
    T = damped_interaction_tensor(pairs["dr"], damp, max_rank=0)
    e = multipole_pair_energy(
        build_polytensor(pairs["q_i"], max_rank=0),
        build_polytensor(pairs["q_j"], max_rank=0),
        T,
    )
    torch.testing.assert_close(e, damp[0] * pairs["q_i"] * pairs["q_j"] / pairs["r"])


def test_undamped_rank1_matches_closed_forms(pairs):
    """Charge-charge, charge-dipole and dipole-dipole, each written out longhand."""
    dr, r = pairs["dr"], pairs["r"]
    q_i, q_j, mu_i, mu_j = pairs["q_i"], pairs["q_j"], pairs["mu_i"], pairs["mu_j"]
    T = damped_interaction_tensor(dr, None, max_rank=1)
    e = multipole_pair_energy(
        build_polytensor(q_i, mu_i), build_polytensor(q_j, mu_j), T
    )
    rhat = dr / r.unsqueeze(-1)
    want = (
        q_i * q_j / r
        + q_j * (mu_i * dr).sum(-1) / r**3
        - q_i * (mu_j * dr).sum(-1) / r**3
        + ((mu_i * mu_j).sum(-1)
           - 3 * (mu_i * rhat).sum(-1) * (mu_j * rhat).sum(-1)) / r**3
    )
    torch.testing.assert_close(e, want)


def test_swapping_i_and_j_leaves_the_energy_unchanged(pairs):
    """dr -> -dr together with (i, j) -> (j, i). The charge-dipole sign convention test."""
    damp = slater_two_center_damp(pairs["b_ij"] * pairs["r"], 1)
    e_ij = multipole_pair_energy(
        build_polytensor(pairs["q_i"], pairs["mu_i"]),
        build_polytensor(pairs["q_j"], pairs["mu_j"]),
        damped_interaction_tensor(pairs["dr"], damp, max_rank=1),
    )
    e_ji = multipole_pair_energy(
        build_polytensor(pairs["q_j"], pairs["mu_j"]),
        build_polytensor(pairs["q_i"], pairs["mu_i"]),
        damped_interaction_tensor(-pairs["dr"], damp, max_rank=1),
    )
    torch.testing.assert_close(e_ij, e_ji)
    # And the charge-dipole part is actually present, so the test above has something to
    # constrain: zeroing the dipoles must change the answer.
    e_q = multipole_pair_energy(
        build_polytensor(pairs["q_i"]), build_polytensor(pairs["q_j"]),
        damped_interaction_tensor(pairs["dr"], damp, max_rank=1),
    )
    assert (e_ij - e_q).abs().max() > 1e-3


def test_zero_dipoles_reduce_rank1_to_rank0(pairs):
    """A charges-only model is the rank-1 model with its dipole block zeroed, exactly."""
    u = pairs["b_ij"] * pairs["r"]
    e0 = multipole_pair_energy(
        build_polytensor(pairs["q_i"], max_rank=0),
        build_polytensor(pairs["q_j"], max_rank=0),
        damped_interaction_tensor(pairs["dr"], slater_two_center_damp(u, 0), max_rank=0),
    )
    e1 = multipole_pair_energy(
        build_polytensor(pairs["q_i"]), build_polytensor(pairs["q_j"]),
        damped_interaction_tensor(pairs["dr"], slater_two_center_damp(u, 1), max_rank=1),
    )
    torch.testing.assert_close(e0, e1)


def test_rotation_invariance(pairs):
    """Rotating dr and both dipoles together leaves the pair energy alone."""
    theta = 0.9
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    damp = slater_two_center_damp(pairs["b_ij"] * pairs["r"], 1)

    def energy(dr, mu_i, mu_j):
        return multipole_pair_energy(
            build_polytensor(pairs["q_i"], mu_i), build_polytensor(pairs["q_j"], mu_j),
            damped_interaction_tensor(dr, damp, max_rank=1),
        )

    torch.testing.assert_close(
        energy(pairs["dr"], pairs["mu_i"], pairs["mu_j"]),
        energy(pairs["dr"] @ R.T, pairs["mu_i"] @ R.T, pairs["mu_j"] @ R.T),
    )


def test_wrong_damping_width_raises():
    dr = torch.randn(4, 3)
    with pytest.raises(ValueError, match="damping factors"):
        damped_interaction_tensor(dr, slater_two_center_damp(torch.ones(4), 0), max_rank=1)


def test_gradient_matches_central_differences(pairs):
    """Autograd through damping and tensor together, against finite differences."""
    dr = pairs["dr"].clone().requires_grad_(True)
    b, q_i, q_j = pairs["b_ij"], pairs["q_i"], pairs["q_j"]
    mu_i, mu_j = pairs["mu_i"], pairs["mu_j"]

    def total(v):
        damp = slater_two_center_damp(b * v.norm(dim=-1), 1)
        return multipole_pair_energy(
            build_polytensor(q_i, mu_i), build_polytensor(q_j, mu_j),
            damped_interaction_tensor(v, damp, max_rank=1),
        ).sum()

    total(dr).backward()
    analytic = dr.grad.clone()

    h = 1e-6
    numeric = torch.zeros_like(analytic)
    with torch.no_grad():
        for p in range(dr.shape[0]):
            for a in range(3):
                plus, minus = pairs["dr"].clone(), pairs["dr"].clone()
                plus[p, a] += h
                minus[p, a] -= h
                numeric[p, a] = (total(plus) - total(minus)) / (2 * h)
    torch.testing.assert_close(analytic, numeric, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Rank 2: quadrupoles
# ---------------------------------------------------------------------------

def pycmm_two_center_rank2(u):
    """p7 and p9, transcribed from pyCMM/cmm/short_range.py:36-37."""
    e = math.exp(-u)
    tmp = 1 + u + u**2 / 2 + u**3 / 6 + u**4 / 24
    p7 = tmp + u**5 / 120 + u**6 / 720
    p9 = p7 + u**7 / 5040
    return [p * e for p in (p7, p9)]


def pycmm_one_center_rank2(u):
    """p7 and p9, transcribed from pyCMM/cmm/short_range.py:16-18.

    Note p9 is built from p5, not p7 -- transcribed as written, not "fixed".
    """
    e = math.exp(-u)
    p5 = 1 + u + u**2 / 2 + u**3 / 6
    p7 = p5 + u**4 / 30
    p9 = p5 + u**4 * 4 / 105 + u**5 / 210
    return [p * e for p in (p7, p9)]


@pytest.mark.parametrize(
    "damp_fn,reference", [(slater_two_center_damp, pycmm_two_center_rank2),
                          (slater_one_center_damp, pycmm_one_center_rank2)]
)
def test_rank2_damp_matches_pycmm(damp_fn, reference):
    u = torch.tensor([0.05, 0.5, 1.0, 3.0, 7.0, 20.0])
    got = damp_fn(u, 2)
    assert got.shape[0] == 5
    want = torch.tensor([reference(float(x)) for x in u]).T
    torch.testing.assert_close(got[3:], want, rtol=1e-13, atol=1e-15)
    # the first three must be untouched by the rank bump
    torch.testing.assert_close(got[:3], damp_fn(u, 1))


def test_rank2_tensor_equals_autograd_derivatives_of_one_over_r():
    """The check that makes a 100-entry transcription safe.

    ``T[a][b] = (-1)^rank(b) d^(rank(a)+rank(b)) (1/r)``: the sign lives on the *column*
    because ``dr = pos[j] - pos[i]``, so derivatives with respect to site i pick up a minus.
    """
    slots = [(), (0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def nth_derivatives(v, order):
        out = {}
        for idx in itertools.product(range(3), repeat=order):
            w = v.clone().requires_grad_(True)
            cur = 1.0 / w.norm()
            for a in idx:
                cur = torch.autograd.grad(cur, w, create_graph=True)[0][a]
            out[idx] = cur.item()
        return out

    for seed in (3, 11, 42):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(3, generator=g) * 1.7
        T = damped_interaction_tensor(v.unsqueeze(0), None, max_rank=2)[0]
        d = {0: {(): (1.0 / v.norm()).item()}}
        for n in (1, 2, 3, 4):
            d[n] = nth_derivatives(v, n)
        for a, sa in enumerate(slots):
            for b, sb in enumerate(slots):
                idx = tuple(sorted(sa + sb))
                want = ((-1) ** len(sb)) * d[len(idx)][idx]
                assert T[a, b].item() == pytest.approx(want, rel=1e-9, abs=1e-12), (
                    f"seed {seed}, entry ({a}, {b})"
                )


def test_zero_quadrupoles_reduce_rank2_to_rank1(pairs):
    u = pairs["b_ij"] * pairs["r"]
    e1 = multipole_pair_energy(
        build_polytensor(pairs["q_i"], pairs["mu_i"]),
        build_polytensor(pairs["q_j"], pairs["mu_j"]),
        damped_interaction_tensor(pairs["dr"], slater_two_center_damp(u, 1), max_rank=1),
    )
    e2 = multipole_pair_energy(
        build_polytensor(pairs["q_i"], pairs["mu_i"], None, max_rank=2),
        build_polytensor(pairs["q_j"], pairs["mu_j"], None, max_rank=2),
        damped_interaction_tensor(pairs["dr"], slater_two_center_damp(u, 2), max_rank=2),
    )
    torch.testing.assert_close(e1, e2)


def test_charge_quadrupole_matches_the_closed_form(pairs):
    """Pins the 1/3, 2/3 polytensor weights against ``q_j Tr(Q d2phi) / 3``."""
    dr, r = pairs["dr"], pairs["r"]
    q_i, q_j = pairs["q_i"], pairs["q_j"]
    g = torch.Generator().manual_seed(7)
    quad = spherical_to_cartesian_quadrupole(torch.randn(7, 5, generator=g))

    T = damped_interaction_tensor(dr, None, max_rank=2)
    e = multipole_pair_energy(
        build_polytensor(q_i, None, quad, max_rank=2),
        build_polytensor(q_j, None, None, max_rank=2),
        T,
    )
    # d^2(1/r)/da db = 3 a b / r^5 - delta_ab / r^3; traceless Q kills the delta term.
    # Both polytensors carry charges, so the charge-charge term is present too.
    d2 = (3.0 * torch.einsum("pa,pb->pab", dr, dr) / r[:, None, None] ** 5
          - torch.eye(3) / r[:, None, None] ** 3)
    want = q_i * q_j / r + q_j * torch.einsum("pab,pab->p", quad, d2) / 3.0
    torch.testing.assert_close(e, want)
    # ... and the quadrupole part is actually doing something.
    assert (e - q_i * q_j / r).abs().max() > 1e-3


def test_quadrupole_quadrupole_is_symmetric_under_swap(pairs):
    g = torch.Generator().manual_seed(9)
    qi = spherical_to_cartesian_quadrupole(torch.randn(7, 5, generator=g))
    qj = spherical_to_cartesian_quadrupole(torch.randn(7, 5, generator=g))
    damp = slater_two_center_damp(pairs["b_ij"] * pairs["r"], 2)
    e_ij = multipole_pair_energy(
        build_polytensor(pairs["q_i"], pairs["mu_i"], qi, max_rank=2),
        build_polytensor(pairs["q_j"], pairs["mu_j"], qj, max_rank=2),
        damped_interaction_tensor(pairs["dr"], damp, max_rank=2),
    )
    e_ji = multipole_pair_energy(
        build_polytensor(pairs["q_j"], pairs["mu_j"], qj, max_rank=2),
        build_polytensor(pairs["q_i"], pairs["mu_i"], qi, max_rank=2),
        damped_interaction_tensor(-pairs["dr"], damp, max_rank=2),
    )
    torch.testing.assert_close(e_ij, e_ji)


def test_rank2_rotation_invariance(pairs):
    theta = 1.3
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    g = torch.Generator().manual_seed(5)
    qi = spherical_to_cartesian_quadrupole(torch.randn(7, 5, generator=g))
    qj = spherical_to_cartesian_quadrupole(torch.randn(7, 5, generator=g))
    damp = slater_two_center_damp(pairs["b_ij"] * pairs["r"], 2)

    def energy(dr, mi, mj, Qi, Qj):
        return multipole_pair_energy(
            build_polytensor(pairs["q_i"], mi, Qi, max_rank=2),
            build_polytensor(pairs["q_j"], mj, Qj, max_rank=2),
            damped_interaction_tensor(dr, damp, max_rank=2),
        )

    rot_q = torch.einsum("ab,pbc,dc->pad", R, qi, R)     # R Q R^T
    torch.testing.assert_close(
        energy(pairs["dr"], pairs["mu_i"], pairs["mu_j"], qi, qj),
        energy(pairs["dr"] @ R.T, pairs["mu_i"] @ R.T, pairs["mu_j"] @ R.T,
               rot_q, torch.einsum("ab,pbc,dc->pad", R, qj, R)),
    )


# ---------------------------------------------------------------------------
# Spherical <-> Cartesian quadrupoles
# ---------------------------------------------------------------------------

def test_spherical_to_cartesian_is_symmetric_and_traceless():
    g = torch.Generator().manual_seed(1)
    q_s = torch.randn(20, 5, generator=g)
    q_c = spherical_to_cartesian_quadrupole(q_s)
    torch.testing.assert_close(q_c, q_c.transpose(-1, -2))
    torch.testing.assert_close(
        torch.einsum("paa->p", q_c), torch.zeros(20), atol=1e-14, rtol=0
    )


def test_cartesian_to_spherical_round_trips():
    g = torch.Generator().manual_seed(2)
    q_s = torch.randn(20, 5, generator=g)
    back = cartesian_to_spherical_quadrupole(spherical_to_cartesian_quadrupole(q_s))
    torch.testing.assert_close(back, q_s)


def test_spherical_to_cartesian_wrong_width_raises():
    with pytest.raises(ValueError, match="5 spherical"):
        spherical_to_cartesian_quadrupole(torch.zeros(3, 6))


def test_irrep2_to_spherical_agrees_with_the_backend_basis():
    """The map must reproduce the backend's own lambda=2 -> Cartesian change of basis.

    If this drifts, the quadrupole head is emitting a rotated mixture of components and
    equivariance is silently gone -- everything else about the model still looks fine.
    """
    backend = _resolve_backend("e3nn")
    m = backend.irrep6_to_voigt()
    C = irrep2_to_spherical(m)
    g = torch.Generator().manual_seed(4)
    a2 = torch.randn(12, 5, generator=g)

    got = spherical_to_cartesian_quadrupole(a2 @ C)
    irrep6 = torch.cat((torch.zeros(12, 1), a2), dim=-1)
    want = voigt_vector_to_symmetric_matrix(irrep6 @ m.t())
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)


def test_irrep2_to_spherical_is_not_a_permutation():
    """Documents *why* the map is derived rather than hand-written.

    e3nn's real spherical harmonics use a permuted axis convention, so two of the lambda=2
    slots mix q20 with q22c. A hand-written relabeling would pass every shape check and
    quietly break rotational equivariance.
    """
    C = irrep2_to_spherical(_resolve_backend("e3nn").irrep6_to_voigt())
    nonzero_per_slot = (C.abs() > 1e-9).sum(dim=1)
    assert int((nonzero_per_slot > 1).sum()) == 2, C
