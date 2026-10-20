"""Unequal-exponent two-center Slater damping (rsfff.ff.multipole.slater_two_center_damp_unequal).

Validated independently of the transcription from HIPPO (Rackers et al. 2021, Appendix A):
  * f1 against direct quadrature of the Coulomb integral of two normalized exp(-b r) densities;
  * f3..f9 against the recursion that defines them from f1, f_{n+2} = f_n - r f_n'/n (autograd);
  * the b_i -> b_j limit against the equal-exponent polynomials, symmetry in i <-> j;
  * series and closed form agree at the switch, float32 against float64, finite gradients at b_i == b_j.
"""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from rsfff.ff.multipole import (
    UNEQUAL_SWITCH,
    slater_two_center_damp,
    slater_two_center_damp_unequal,
)

D = torch.float64


def _coulomb_quadrature(a, b, R):
    """int rho_b(r) phi_a(|r - R|) d^3r for normalized 1s Slater densities (exponents a, b)."""
    x, w = leggauss(240)
    rmax = 45.0 / b
    r, wr = (x + 1) / 2 * rmax, w * rmax / 2
    c, wc = leggauss(96)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    s = np.sqrt(rr ** 2 + R ** 2 - 2 * rr * R * cc)
    phi = (1 - (1 + a * s / 2) * np.exp(-a * s)) / s
    rho = b ** 3 / (8 * np.pi) * np.exp(-b * rr)
    return float(np.sum(np.outer(wr, wc) * 2 * np.pi * rr ** 2 * rho * phi))


@pytest.mark.parametrize("a,b", [(3.78, 1.68), (2.83, 1.68), (2.4, 1.87), (2.0, 2.05), (1.5, 4.0)])
def test_f1_matches_quadrature(a, b):
    R = np.array([1.5, 2.5, 4.0, 6.0, 9.0])
    f1 = slater_two_center_damp_unequal(torch.full((5,), a, dtype=D), torch.full((5,), b, dtype=D),
                                        torch.tensor(R, dtype=D), max_rank=0)[0].numpy()
    # the overlap complement: J(R) = (1 - f1) / R
    J = np.array([_coulomb_quadrature(a, b, x) for x in R])
    assert np.allclose(1 - R * J, f1, atol=1e-8)


@pytest.mark.parametrize("ratio", [1.0005, 1.01, 1.03, 1.3, 2.5])   # both branches
def test_higher_orders_follow_from_f1(ratio):
    r = torch.linspace(0.3, 14.0, 60, dtype=D).requires_grad_(True)
    bi = torch.full_like(r, 2.2)
    f = slater_two_center_damp_unequal(bi, bi * ratio, r, max_rank=2)
    for k, n in enumerate((1, 3, 5, 7)):
        (df,) = torch.autograd.grad(f[k].sum(), r, retain_graph=True)
        assert torch.allclose(f[k + 1], f[k] - r * df / n, atol=1e-10), n


def test_equal_limit_and_symmetry():
    r = torch.linspace(0.1, 20.0, 80, dtype=D)
    b = torch.full_like(r, 1.9)
    assert torch.allclose(slater_two_center_damp_unequal(b, b, r, 2), slater_two_center_damp(1.9 * r, 2), atol=1e-14)
    assert torch.allclose(slater_two_center_damp_unequal(b, 1.6 * b, r, 2),
                          slater_two_center_damp_unequal(1.6 * b, b, r, 2), atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_branches_agree_at_the_switch(dtype):
    """Series and closed form evaluated at the same eps = switch (by moving the switch either side)."""
    sw = UNEQUAL_SWITCH[dtype]
    r = torch.linspace(0.2, 15.0, 50, dtype=dtype)
    z = torch.full_like(r, 2.3)
    bi, bj = z * (1 + sw), z * (1 - sw)
    series = slater_two_center_damp_unequal(bi, bj, r, 2, switch=sw * 1.01)
    closed = slater_two_center_damp_unequal(bi, bj, r, 2, switch=sw * 0.99)
    assert (series - closed).abs().max() < (1e-11 if dtype == torch.float64 else 2e-5)


def test_float32_against_float64():
    rng = np.random.default_rng(0)
    bi = rng.uniform(1.4, 4.0, 400)
    bj = bi * np.exp(rng.normal(scale=0.3, size=400))
    r = rng.uniform(0.2, 15.0, 400)
    f64 = slater_two_center_damp_unequal(*(torch.tensor(x, dtype=torch.float64) for x in (bi, bj, r)), max_rank=2)
    f32 = slater_two_center_damp_unequal(*(torch.tensor(x, dtype=torch.float32) for x in (bi, bj, r)), max_rank=2)
    assert (f64 - f32.double()).abs().max() < 2e-5


def test_finite_gradients_at_equal_exponents():
    b = torch.tensor([2.0, 2.0, 3.0], dtype=D, requires_grad=True)
    c = torch.tensor([2.0, 2.0 + 1e-9, 1.5], dtype=D, requires_grad=True)
    r = torch.tensor([1.0, 3.0, 2.0], dtype=D, requires_grad=True)
    slater_two_center_damp_unequal(b, c, r, 2).sum().backward()
    for t in (b, c, r):
        assert torch.isfinite(t.grad).all()


def test_multi_term_shapes_are_sums_of_terms():
    """A shape split into two half-weight terms with the same exponent equals the single term (bilinearity), and
    a multi-term shape on one side only also matches the explicit sum."""
    import torch
    from rsfff.ff.multipole import shaped_one_center_multi, shaped_two_center_damp
    torch.manual_seed(1)
    P = 7
    r = torch.linspace(1.0, 9.0, P, dtype=torch.float64)
    bi = torch.full((P,), 1.3, dtype=torch.float64)
    bj = torch.full((P,), 2.1, dtype=torch.float64)
    wi = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64).expand(P, 4)
    wj = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(P, 4)
    single = shaped_two_center_damp(bi, bj, wi, wj, r, 2)
    bm = torch.stack([bi, bi], 1)
    wm = torch.stack([0.5 * wi, 0.5 * wi], 1)
    assert torch.allclose(shaped_two_center_damp(bm, bj, wm, wj, r, 2), single, atol=1e-12)
    # two different exponents on side i: the sum of the two single-term results
    b2 = torch.stack([bi, bi * 1.7], 1)
    w2 = torch.stack([0.6 * wi, 0.4 * wi], 1)
    ref = 0.6 * shaped_two_center_damp(bi, bj, wi, wj, r, 2) + 0.4 * shaped_two_center_damp(bi * 1.7, bj, wi, wj, r, 2)
    assert torch.allclose(shaped_two_center_damp(b2, bj, w2, wj, r, 2), ref, atol=1e-12)
    from rsfff.ff.multipole import shaped_one_center_damp
    ref1 = 0.6 * shaped_one_center_damp(bi * r, wi, 2) + 0.4 * shaped_one_center_damp(bi * 1.7 * r, wi, 2)
    assert torch.allclose(shaped_one_center_multi(b2, w2, r, 2), ref1, atol=1e-12)
