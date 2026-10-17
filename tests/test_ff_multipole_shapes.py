"""Shaped Slater densities (rsfff.ff.multipole: shaped_one_center_damp, shaped_two_center_damp).

The densities are positive mixtures of normalized r^k e^{-b r}, k = 0..3. Checked independently of the
generated code (scripts/derive/slater_shapes):
  * one-center f1 against radial quadrature of each basis density's potential;
  * two-center f1 against quadrature of the Coulomb integral of two basis densities (all k, l, both orders);
  * f3..f9 against the defining recursion f_{n+2} = f_n - r f_n'/n (autograd), in both branches;
  * reduction to the 1s forms, symmetry under swapping the two sites, branch agreement at the switch,
    finite gradients through b_i == b_j, float32 inputs.
"""

from math import factorial, pi

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from rsfff.ff.multipole import (
    SHAPE_SWITCH,
    shaped_one_center_damp,
    shaped_two_center_damp,
    slater_one_center_damp,
    slater_two_center_damp_unequal,
)

D = torch.float64


def _onehot(k, n=1, K=3):
    w = torch.zeros(n, K + 1, dtype=D)
    w[:, k] = 1.0
    return w


def _basis_potential(k, b, s):
    """Potential at distance s of the normalized density b^{k+3}/(4 pi (k+2)!) r^k e^{-b r} (numerical)."""
    N = b ** (k + 3) / (4 * pi * factorial(k + 2))
    inner = quad(lambda x: N * x ** k * np.exp(-b * x) * 4 * pi * x * x, 0, s)[0] / s
    outer = quad(lambda x: N * x ** k * np.exp(-b * x) * 4 * pi * x, s, np.inf)[0]
    return inner + outer


@pytest.mark.parametrize("k", range(4))
def test_one_center_potential(k):
    b, rs = 1.8, (0.4, 1.3, 3.0, 7.0)
    for r in rs:
        f1 = shaped_one_center_damp(torch.tensor([b * r], dtype=D), _onehot(k), 0)[0, 0].item()
        assert abs((1 - f1) / r - _basis_potential(k, b, r)) < 1e-10


def _J_quad(k, l, a, b, R):
    x, wq = leggauss(200)
    rmax = 60.0 / b
    r, wr = (x + 1) / 2 * rmax, wq * rmax / 2
    c, wc = leggauss(96)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    s = np.sqrt(rr ** 2 + R ** 2 - 2 * rr * R * cc)
    f1 = shaped_one_center_damp(torch.tensor(a * s.ravel(), dtype=D), _onehot(k, s.size), 0)[0].numpy().reshape(s.shape)
    phi = (1 - f1) / s
    rho = b ** (l + 3) / (4 * pi * factorial(l + 2)) * rr ** l * np.exp(-b * rr)
    return float(np.sum(np.outer(wr, wc) * 2 * pi * rr ** 2 * rho * phi))


@pytest.mark.parametrize("k,l", [(k, l) for k in range(4) for l in range(4)])
def test_two_center_coulomb_matches_quadrature(k, l):
    for a, b in ((2.4, 1.7), (1.9, 2.6), (3.8, 1.6), (2.0, 2.1)):
        R = torch.tensor([1.5, 3.0, 6.0], dtype=D)
        f1 = shaped_two_center_damp(torch.full_like(R, a), torch.full_like(R, b), _onehot(k, 3), _onehot(l, 3), R, 0)[0]
        for i, Rv in enumerate(R.tolist()):
            assert abs((1 - f1[i].item()) / Rv - _J_quad(k, l, a, b, Rv)) < 1e-8, (k, l, a, b, Rv)


@pytest.mark.parametrize("ratio", [1.0, 1.05, 1.2, 1.6, 2.4])
def test_two_center_recursion(ratio):
    rng = np.random.default_rng(1)
    P = 40
    r = torch.linspace(0.5, 13.0, P, dtype=D).requires_grad_(True)
    wi = torch.softmax(torch.tensor(rng.normal(size=(P, 4)), dtype=D), 1)
    wj = torch.softmax(torch.tensor(rng.normal(size=(P, 4)), dtype=D), 1)
    f = shaped_two_center_damp(torch.full_like(r, 2.2), torch.full_like(r, 2.2 * ratio), wi, wj, r, 2)
    for idx, n in enumerate((1, 3, 5, 7)):
        (df,) = torch.autograd.grad(f[idx].sum(), r, retain_graph=True)
        assert torch.allclose(f[idx + 1], f[idx] - r * df / n, atol=1e-9), (ratio, n)


@pytest.mark.parametrize("k", range(4))
def test_one_center_recursion(k):
    u = torch.linspace(0.1, 25.0, 60, dtype=D).requires_grad_(True)
    f = shaped_one_center_damp(u, _onehot(k, 60), 2)
    for idx, n in enumerate((1, 3, 5, 7)):
        (df,) = torch.autograd.grad(f[idx].sum(), u, retain_graph=True)
        assert torch.allclose(f[idx + 1], f[idx] - u * df / n, atol=1e-12)


def test_reductions_and_symmetry():
    r = torch.linspace(0.3, 15.0, 50, dtype=D)
    bi, bj = torch.full_like(r, 2.6), torch.full_like(r, 1.8)
    w1s = _onehot(0, 50)
    assert torch.allclose(shaped_two_center_damp(bi, bj, w1s, w1s, r, 2), slater_two_center_damp_unequal(bi, bj, r, 2), atol=1e-13)
    assert torch.allclose(shaped_one_center_damp(bi * r, w1s, 2), slater_one_center_damp(bi * r, 2), atol=1e-15)
    rng = np.random.default_rng(2)
    wi = torch.softmax(torch.tensor(rng.normal(size=(50, 4)), dtype=D), 1)
    wj = torch.softmax(torch.tensor(rng.normal(size=(50, 4)), dtype=D), 1)
    assert torch.allclose(shaped_two_center_damp(bi, bj, wi, wj, r, 2), shaped_two_center_damp(bj, bi, wj, wi, r, 2), atol=1e-12)


def test_branches_agree_at_switch():
    r = torch.linspace(0.3, 15.0, 50, dtype=D)
    z = torch.full_like(r, 2.3)
    bi, bj = z * (1 + SHAPE_SWITCH), z * (1 - SHAPE_SWITCH)
    for k in range(4):
        for l in range(4):
            wi, wj = _onehot(k, 50), _onehot(l, 50)
            ser = shaped_two_center_damp(bi, bj, wi, wj, r, 2, switch=SHAPE_SWITCH * 1.01)
            clo = shaped_two_center_damp(bi, bj, wi, wj, r, 2, switch=SHAPE_SWITCH * 0.99)
            assert (ser - clo).abs().max() < 1e-9, (k, l)


def test_gradients_finite_and_float32():
    b = torch.tensor([2.0, 2.0, 3.0], dtype=D, requires_grad=True)
    c = torch.tensor([2.0, 2.0 + 1e-9, 1.5], dtype=D, requires_grad=True)
    r = torch.tensor([1.0, 3.0, 2.0], dtype=D, requires_grad=True)
    wl = torch.zeros(3, 4, dtype=D, requires_grad=True)
    w = torch.softmax(wl, 1)
    shaped_two_center_damp(b, c, w, w, r, 2).sum().backward()
    for t in (b, c, r, wl):
        assert torch.isfinite(t.grad).all()
    f32 = shaped_two_center_damp(b.detach().float(), c.detach().float(), w.detach().float(), w.detach().float(), r.detach().float(), 2)
    assert f32.dtype == torch.float32
    f64 = shaped_two_center_damp(b.detach(), c.detach(), w.detach(), w.detach(), r.detach(), 2)
    assert (f32.double() - f64).abs().max() < 1e-6
