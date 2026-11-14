"""C²-continuous switching and validity-envelope functions for the diabatic mixture.

Two geometric weights, both built from the quintic smoothstep
``S₅(t) = t³(6t² − 15t + 10)`` whose value **and first two derivatives** vanish at ``t = 0, 1``.
That C² edge is what makes the force continuous where a channel switches off or a diabatic state
is pruned (docs/mixture_of_diabatic_embeddings.md §2.4.4, §4.4.4): a merely C⁰/C¹ ramp would
leave a kink or a spike in ``-∇E`` at the boundary.

- :func:`pairwise_switch` multiplies inter-fragment channel compliances so charge transfer shuts
  off smoothly as a bond is pulled past the short-range cutoff (§2.4.3).
- :func:`validity_bump` is the per-diabat validity envelope ``Ω_K`` (§2.2.3): a compact-support
  bump that turns a diabatic state on and off over a geometric order parameter, so the local
  gate mixes only the states that are physically relevant at a given geometry.

Everything is plain differentiable torch, so gradients flow for the force/continuity checks.
"""

from __future__ import annotations

import torch


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep on ``[0, 1]``: 0 below, 1 above, C² at both ends.

    ``S₅(t) = 6t⁵ − 15t⁴ + 10t³`` with ``S₅' = S₅'' = 0`` at ``t = 0`` and ``t = 1``. Input is
    clamped, so values outside ``[0, 1]`` saturate flat (zero derivative) rather than extrapolate.
    """
    t = t.clamp(0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def pairwise_switch(r: torch.Tensor, r_on: float, r_off: float) -> torch.Tensor:
    """Inter-fragment channel switch ``S(r) ∈ [0, 1]``: 1 for ``r ≤ r_on``, 0 for ``r ≥ r_off``.

    Multiplies the compliance of an inter-fragment channel (§2.4.3), so charge transfer along a
    breaking bond decays to exactly zero — and asymptotic charges become exactly integer — before
    the channel leaves the neighbor list. ``r_on`` should sit **above the training bond-length
    range** so ``S ≡ 1`` wherever the frozen model was actually fit; ``r_off`` is the short-range
    cutoff. Decreasing on ``[r_on, r_off]``.
    """
    if not r_off > r_on:
        raise ValueError(f"pairwise_switch needs r_off > r_on, got r_on={r_on}, r_off={r_off}")
    return 1.0 - smoothstep((r - r_on) / (r_off - r_on))


def validity_bump(
    x: torch.Tensor,
    *,
    lo0: float = float("-inf"),
    lo1: float = float("-inf"),
    hi1: float = float("inf"),
    hi0: float = float("inf"),
) -> torch.Tensor:
    """Validity envelope ``Ω(x) ∈ [0, 1]`` over a scalar order parameter (§2.2.3).

    A C² compact-support bump: rises 0→1 on ``[lo0, lo1]``, is flat 1 on ``[lo1, hi1]``, and falls
    1→0 on ``[hi1, hi0]``. Leaving a side at its infinite default makes that side open — e.g. a
    bound state closing only at large stretch uses ``hi1, hi0`` and leaves ``lo0, lo1 = -inf``; a
    dissociated state opening only past some onset uses ``lo0, lo1`` and leaves ``hi1, hi0 = +inf``.

    Breakpoints must satisfy ``lo0 ≤ lo1 ≤ hi1 ≤ hi0``. The rising and falling edges are quintic
    smoothsteps, so ``Ω`` is C² everywhere including where it meets 0 — the property that keeps the
    gated mixture (and its forces) continuous as a state is pruned.
    """
    if not (lo0 <= lo1 <= hi1 <= hi0):
        raise ValueError(
            f"validity_bump needs lo0 <= lo1 <= hi1 <= hi0, got {(lo0, lo1, hi1, hi0)}"
        )
    rise = (
        torch.ones_like(x) if lo1 == float("-inf")
        else smoothstep((x - lo0) / (lo1 - lo0)) if lo1 > lo0
        else (x >= lo1).to(x.dtype)          # degenerate step (lo0 == lo1): still 0/1, no NaN
    )
    fall = (
        torch.ones_like(x) if hi1 == float("inf")
        else 1.0 - smoothstep((x - hi1) / (hi0 - hi1)) if hi0 > hi1
        else (x <= hi1).to(x.dtype)
    )
    return rise * fall
