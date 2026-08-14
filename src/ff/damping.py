"""Damping and range-separation functions for the force-field terms.

Two independent pieces, both pure functions of their arguments:

- :func:`tang_toennies` -- the incomplete-gamma damping ``f_n(x)`` that regularizes a
  ``1/r^n`` multipole-expansion term at short range, where the expansion diverges but
  the true interaction does not. Used by the dispersion term
  (:mod:`rsfff.ff.dispersion`) and, eventually, by the damped Coulomb of the induction
  solve (``docs/range_separated_mlip.md`` §4.5).
- :func:`fermi_switch` -- the range-separation switch. Multiplies an explicit
  force-field term so it is **off at short range** and full-strength at long range,
  handing the near-field to the short-range network under the specified fragmentation.

This module deliberately knows nothing about C6, multipoles, or any particular energy
term, so that a future ``ff/electrostatics.py`` can reuse it without importing
dispersion-flavored code.

Units: :func:`tang_toennies` takes a **dimensionless** argument ``x = b * r``.
:func:`fermi_switch` takes distances in **Angstrom** (the model's position units), so
that ``r0`` reads as the number one actually reasons about.
"""

from __future__ import annotations

import math

import torch

#: Below this value of ``x`` the direct form of ``f_n`` is catastrophically cancelling
#: and the series form is used instead. 2.0 is safe for float32 *and* float64 -- see
#: :func:`tang_toennies` for the measured errors.
DEFAULT_SERIES_BELOW = 2.0

#: Number of terms kept in the small-``x`` tail series. 20 terms give a relative
#: truncation error below 1e-17 at ``x = 2`` (the largest ``x`` the branch ever sees).
_SERIES_TERMS = 20


def _direct(x: torch.Tensor, order: int) -> torch.Tensor:
    """``1 - exp(-x) * sum_{k=0}^{order} x^k/k!`` evaluated as written (Horner)."""
    # Horner from the top: (((1/n! * x + 1/(n-1)!) * x + ...) * x + 1)
    poly = torch.full_like(x, 1.0 / math.factorial(order))
    for k in range(order - 1, -1, -1):
        poly = poly * x + 1.0 / math.factorial(k)
    return 1.0 - torch.exp(-x) * poly


def _series(x: torch.Tensor, order: int) -> torch.Tensor:
    """``exp(-x) * sum_{k=order+1}^{order+_SERIES_TERMS} x^k/k!`` -- same function, no cancellation.

    Follows from ``e^x = sum_{k>=0} x^k/k!``: the ``1 -`` and the first ``order+1``
    terms cancel *algebraically*, leaving only the tail. Every term is positive, so
    nothing is subtracted and the small-``x`` limit ``x^(order+1)/(order+1)!`` comes out
    exactly.
    """
    lo = order + 1
    poly = torch.full_like(x, 1.0 / math.factorial(lo + _SERIES_TERMS - 1))
    for k in range(lo + _SERIES_TERMS - 2, lo - 1, -1):
        poly = poly * x + 1.0 / math.factorial(k)
    return torch.exp(-x) * poly * x.pow(lo)


def tang_toennies(
    x: torch.Tensor,
    order: int = 6,
    *,
    series_below: float = DEFAULT_SERIES_BELOW,
) -> torch.Tensor:
    """Tang-Toennies damping ``f_n(x) = 1 - exp(-x) * sum_{k=0}^{n} x^k/k!``.

    ``x = b_ij * r`` is dimensionless (``b`` in bohr^-1, ``r`` in bohr). ``order = n``
    matches the inverse power being damped: ``n = 6`` for a C6 term, 8 for C8, 10 for
    C10. ``f_n`` rises monotonically from 0 at ``x = 0`` to 1 as ``x -> inf``, with the
    short-range limit ``f_n(x) -> x^(n+1)/(n+1)!``.

    Two algebraically identical branches, selected at ``series_below``:

    - ``x >= series_below``: the direct expression above.
    - ``x < series_below``: the cancellation-free tail ``exp(-x) * sum_{k>n} x^k/k!``.

    The second branch is **not optional**. The direct form subtracts two near-equal
    O(1) numbers to produce a tiny result; measured relative error at ``order=6`` *in
    float64* is 1.1e+02 at ``x=0.01`` (true value 2e-18) and 1.6e-05 at ``x=0.1``, and
    float32 is far worse. Multiplied by the ``1/r^6`` prefactor this is real energy
    noise, and autograd through the direct form inherits the same cancellation, so it
    corrupts forces too. The series branch differentiates cleanly.

    For the current water-cluster data the small-``x`` branch is never reached (the
    shortest inter-fragment contact is ~1.9 Ang, i.e. ``x ~ 5.5``); it is insurance for
    the reactive regime this repo targets, and costs ~20 fused multiply-adds on the edge
    array -- negligible against the featurizer. Please do not "optimize" it away.
    """
    if order < 0:
        raise ValueError(f"tang_toennies needs order >= 0, got {order}")
    # Both branches of torch.where are always evaluated, so each must be fed inputs it
    # can handle: unclamped, the series branch computes x**26 and overflows to inf,
    # and inf * exp(-x) = NaN then poisons the gradient of the *unselected* branch.
    xs = x.clamp(max=series_below)
    xl = x.clamp(min=series_below)
    return torch.where(x < series_below, _series(xs, order), _direct(xl, order))


def fermi_switch(
    r: torch.Tensor,
    r0: torch.Tensor | float,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Range-separation switch ``S(r) = 1 / (1 + exp(-alpha (r - r0)))``: 0 short, 1 long.

    Multiplies an explicit force-field term to switch it **off** below ``r0``, so the
    short-range network carries that region instead. ``r`` and ``r0`` are in Angstrom;
    ``alpha`` (Angstrom^-1) sets the width of the crossover. Both ``r0`` and ``alpha``
    may be tensors so the crossover can be learned (``docs/range_separated_mlip.md``
    §4.2: "the crossover should be learned, not hard-coded at a fixed cutoff") -- with a
    penalty biasing ``r0`` small, so the model must justify handing energy to the network.

    A tensor ``alpha`` is **not** checked for positivity: the check below is a
    construction-time guard on a Python float, and a learned ``alpha`` is kept positive by
    its own parameterization (``softplus`` in
    :class:`rsfff.ff.unified.RangeSeparationHeads`) rather than by a runtime assertion on
    every forward pass. A negative ``alpha`` would invert the switch, turning the term off
    at long range instead of short -- so whatever produces it owes that guarantee.

    Implemented with :func:`torch.sigmoid` rather than a hand-rolled ``1/(1+exp(...))``:
    sigmoid is the branch-stable formulation at large ``|alpha (r - r0)|``, where the
    naive version overflows and its gradient becomes ``inf/inf``.

    The logistic has exponential tails, so ``S`` never reaches *exactly* 0 or 1. Neither
    matters here: at short range :func:`tang_toennies` has already sent the damped term
    to zero linearly in ``r``, and at long range the exact zero is supplied by the
    neighbor-list taper, not by ``S``.
    """
    if not isinstance(alpha, torch.Tensor) and not alpha > 0:
        raise ValueError(f"fermi_switch needs alpha > 0, got {alpha}")
    return torch.sigmoid(alpha * (r - r0))
