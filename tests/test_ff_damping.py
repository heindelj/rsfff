"""Tang-Toennies damping and the Fermi range-separation switch.

The small-x tests are the point of this file: ``f_n(x)`` is a difference of near-equal
O(1) numbers whose true value is ~1e-18, so the naive expression loses every significant
digit there. Several of these tests fail outright against a single-branch implementation.
"""

import math

import pytest
import torch

from rsfff.ff.damping import (
    DEFAULT_SERIES_BELOW,
    _direct,
    _series,
    fermi_switch,
    tang_toennies,
)

ORDERS = [6, 8, 10]


def pycmm_f6(u):
    """pyCMM's hard-coded 6-term series, transcribed from cmm/dispersion.py:30-31.

    Inlined because pyCMM is not a dependency of this repo; this is the cross-repo
    reference the dispersion energy must agree with.
    """
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    return 1 - torch.exp(-u) * (
        1 + u + u2 / 2 + u3 / 6 + u4 / 24 + u5 / 120 + u6 / 720
    )


@pytest.mark.parametrize("order", ORDERS)
def test_endpoints_and_monotonicity(order):
    assert tang_toennies(torch.zeros(1), order).item() == 0.0
    # The approach to 1 is exp(-x) x^n/n!, which for n=10 is still 5e-12 at x=50.
    assert tang_toennies(torch.tensor([80.0]), order).item() == pytest.approx(1.0, abs=1e-12)
    # dense grid spanning both branches
    x = torch.linspace(1e-6, 40.0, 4001)
    f = tang_toennies(x, order)
    assert torch.all(f.diff() > 0)
    assert torch.all((f >= 0) & (f <= 1))


@pytest.mark.parametrize("order", ORDERS)
def test_small_x_asymptote(order):
    """f_n(x) -> x^(n+1)/(n+1)!  as x -> 0.

    This is the test that fails without the series branch: the direct form returns
    literal 0.0 at x = 1e-3 in float64, so the ratio below is 0 instead of 1.

    The next term in the expansion is -x^(n+2)/(n+2)!, so the relative deviation from
    the leading term is x(n+1)/(n+2) < x -- which is the bound asserted here, rather
    than a fixed tolerance that would silently pass a wrong prefactor at small x.
    """
    x = torch.tensor([1e-4, 1e-3, 1e-2])
    leading = x ** (order + 1) / math.factorial(order + 1)
    ratio = tang_toennies(x, order) / leading
    assert torch.all((ratio - 1.0).abs() <= x)


def test_matches_pycmm_series():
    """Agreement with pyCMM's expression wherever that expression is trustworthy."""
    u = torch.linspace(2.0, 30.0, 501)
    assert torch.allclose(tang_toennies(u, 6), pycmm_f6(u), atol=1e-12, rtol=0)


@pytest.mark.parametrize("order", ORDERS)
def test_branch_continuity(order):
    """The two branches agree in value and derivative *at the same point*.

    Comparing f(seam - eps) with f(seam + eps) instead would measure the genuine slope
    (~0.012 for order 6), not a discontinuity -- so both branches are evaluated here at
    exactly ``series_below``.
    """
    x = torch.tensor(DEFAULT_SERIES_BELOW, requires_grad=True)
    below = _series(x, order)
    above = _direct(x, order)
    assert below.item() == pytest.approx(above.item(), rel=1e-14)
    (g_below,) = torch.autograd.grad(below, x, retain_graph=True)
    (g_above,) = torch.autograd.grad(above, x)
    assert g_below.item() == pytest.approx(g_above.item(), rel=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_derivative_closed_form(order):
    """d/dx f_n(x) = exp(-x) x^n / n!, exactly -- including on the series branch."""
    x = torch.tensor([0.05, 0.5, 1.0, 1.9, 2.1, 5.0, 12.0], requires_grad=True)
    (g,) = torch.autograd.grad(tang_toennies(x, order).sum(), x)
    expected = torch.exp(-x.detach()) * x.detach() ** order / math.factorial(order)
    assert torch.allclose(g, expected, rtol=1e-10, atol=1e-300)


def test_no_nan_from_unselected_branch():
    """torch.where evaluates both branches; the series one overflows without clamping."""
    x = torch.tensor([0.0, 1e-8, 1.0, 100.0, 700.0], requires_grad=True)
    f = tang_toennies(x, 6)
    assert torch.isfinite(f).all()
    (g,) = torch.autograd.grad(f.sum(), x)
    assert torch.isfinite(g).all()


def test_float32_stability():
    x64 = torch.logspace(-2, math.log10(5.0), 200, dtype=torch.float64)
    ref = tang_toennies(x64, 6)
    got = tang_toennies(x64.to(torch.float32), 6).to(torch.float64)
    assert torch.all((got - ref).abs() / ref < 1e-4)


def test_negative_order_raises():
    with pytest.raises(ValueError):
        tang_toennies(torch.ones(1), -1)


# ---------------------------------------------------------------------------
# Fermi switch
# ---------------------------------------------------------------------------

def test_fermi_midpoint_and_limits():
    r = torch.tensor([0.0, 2.0, 10.0])
    s = fermi_switch(r, 2.0, 8.0)
    assert s[1].item() == pytest.approx(0.5)
    assert s[0].item() < 1e-6          # off at short range
    assert s[2].item() == pytest.approx(1.0, abs=1e-12)   # on at long range
    # Non-decreasing everywhere; strictly increasing before the logistic saturates
    # to exactly 1.0 in floating point (which it does past r ~ 6.5 here).
    r = torch.linspace(0.0, 12.0, 1001)
    assert torch.all(fermi_switch(r, 2.0, 8.0).diff() >= 0)
    r = torch.linspace(0.5, 5.0, 1001)
    assert torch.all(fermi_switch(r, 2.0, 8.0).diff() > 0)


def test_fermi_saturation_is_finite():
    """Large |alpha (r - r0)|: value and gradient must stay finite, not inf/inf."""
    r = torch.tensor([-1e3, 1e3], requires_grad=True)
    s = fermi_switch(r, 0.0, 10.0)
    assert torch.isfinite(s).all()
    (g,) = torch.autograd.grad(s.sum(), r)
    assert torch.isfinite(g).all()


def test_fermi_learnable_r0():
    """r0 may be a tensor and carries gradient, so the crossover can be learned."""
    r0 = torch.tensor(2.0, requires_grad=True)
    s = fermi_switch(torch.tensor([1.5, 2.5, 4.0]), r0, 8.0)
    (g,) = torch.autograd.grad(s.sum(), r0)
    assert torch.isfinite(g) and g.item() < 0     # raising r0 switches more off


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_fermi_rejects_nonpositive_alpha(alpha):
    with pytest.raises(ValueError):
        fermi_switch(torch.ones(1), 2.0, alpha)
