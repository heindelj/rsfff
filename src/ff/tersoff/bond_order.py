"""Explicit (closed-form) bond orders: the pairing functional evaluated at a formula.

``docs/fff_tersoff.md`` §2. :mod:`rsfff.ff.pairing.bond_order` *minimizes*
``sum_ij [-J p + kappa p^2 / 2]`` under the valence marginals ``sum_j p_ij + u_i = v_i``. Here
``p`` is written down instead of solved for, in two steps:

1. **Raw bond order** -- the unconstrained stationary point of the barrier-free functional,
   ``kappa p = J`` clipped to ``[0, 1]``::

       b_ij = clip(J_ij / kappa_ij, 0, 1)        (smoothed over w = T / kappa, see below)

   ``J`` is the Slater-overlap coupling of the pairing model, so the radial shape, the
   exponential decay and the anisotropy (bond angles from the valence dipole / quadrupole)
   are inherited; nothing is switched by distance. ``b = 1/2`` at ``J = kappa / 2``, where
   the pairing model's attractive branch turns over.

2. **Saturation** -- what makes ``sum_j p_ij <= v_i``:

   ``rebo``
       one factor per atom, Tersoff / REBO style::

           N_i = sum_j b_ij,   s_i = v_i / smoothmax(v_i, N_i),   p_ij = b_ij s_i s_j

       ``s_i = 1`` below capacity and ``v_i / N_i`` above it, so the bound holds by
       construction; the sharing between an atom's partners is *linear* in ``b``. Tersoff's
       own ``(1 + zeta^n)^(-1/2n)`` has the same limits but is ``2^(-1/2n)`` *at* capacity;
       here ``p = 1`` on a bond has a meaning (it is the co-membership), so the factor is one
       up to the capacity and only bends above it, over ``smoothmax(a, b) = a + eps
       softplus((b - a) / eps)`` with ``eps`` = ``saturation_width`` electrons.

   ``waterfill`` (default)
       the pairing functional restricted to one atom has the water-filling solution
       ``p_i(j) = clip((J_ij - lambda_i) / kappa_ij, 0, 1)`` with ``lambda_i >= 0`` the one
       scalar that fills the capacity (``lambda_i = 0`` when the raw orders fit). The root
       is monotone in ``lambda``; it is taken as ``k_steps`` safeguarded Newton steps
       (bisection fallback inside the bracket ``[0, max_j J_ij + kappa]``) without the
       graph, then two differentiable Newton steps from the root (exact first and second
       derivatives of the implicit function, as in the pairing solve). The two ends are
       reconciled by ``p_ij = min(p_i(j), p_j(i))`` -- the smaller grant wins, which is the
       dual solution whenever one end is the binding one, and the even split of a shared
       proton when both are -- smoothed as ``(x + y)/2 - sqrt(((x - y)/2)^2 + eps^2) + eps``
       so that it is exact on ``x = y`` (every bond of an intact molecule sits there) and
       exceeds the true minimum by at most ``eps`` (``reconcile_width``) elsewhere. Hence
       ``sum_j p_ij <= v_i + g_i + deg_i eps`` with ``g_i`` the filling residual, which the
       Newton steps drive below ``10 T`` (``converged``). No multiplicative factor is applied
       on top: any smooth ``min(1, v/N)`` bends *at* capacity. Competition is energetic: a partner whose
       ``J`` falls below ``lambda_i`` gets (smoothly) zero, two equal partners split ``v_i``
       evenly with a transition width set by ``kappa``.

The clip is ``w [softplus(y / w) - softplus((y - 1) / w)]`` with ``w = T / kappa``: the
pairing model's barrier temperature ``T`` plays the same smoothing role here, so ``p`` is C-inf
and ``1 - p ~ exp(-(J - kappa) / T)`` on a bond, as there.

Nothing here is stationary in ``p``, so forces carry ``dp/dR`` -- ordinary autograd through an
explicit graph; ``gradgradcheck`` holds for the force loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..pairing.electronic_state import ElectronicState, capacity, e0_terms

__all__ = [
    "SATURATIONS",
    "explicit_state",
    "overbinding_penalty",
    "raw_bond_order",
    "saturate_rebo",
    "saturate_waterfill",
    "tersoff_state_energy",
]

SATURATIONS = ("rebo", "waterfill")
_V_FLOOR = 1.0e-6


def _soft_clip(y: torch.Tensor, w: torch.Tensor, complement: bool = False):
    """``clip(y, 0, 1)`` smoothed over ``w``, and its derivative; with ``complement`` also
    ``1 - clip`` evaluated cancellation-free (``w [softplus((1 - y)/w) - softplus(-y/w)]``),
    which is what a saturated bond's tail needs."""
    a = y / w
    c = (y - 1.0) / w
    value = w * (F.softplus(a) - F.softplus(c))
    deriv = torch.sigmoid(a) - torch.sigmoid(c)
    if not complement:
        return value, deriv
    comp = w * (F.softplus(-c) - F.softplus(-a))
    return value, deriv, comp


def raw_bond_order(J: torch.Tensor, kappa: torch.Tensor, temperature) -> torch.Tensor:
    """``clip(J / kappa, 0, 1)`` smoothed over ``T / kappa``: the unconstrained pairing order."""
    temperature = torch.as_tensor(temperature, dtype=J.dtype, device=J.device)
    return _soft_clip(J / kappa, temperature / kappa)[0]


def _coordination(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``N_a = sum over pairs touching a`` of a per-pair quantity."""
    return x.new_zeros(n_atoms).index_add_(0, i, x).index_add_(0, j, x)


def _coordination_two(x_i, x_j, i, j, n_atoms):
    return x_i.new_zeros(n_atoms).index_add_(0, i, x_i).index_add_(0, j, x_j)


def _smoothmax(a, b, eps):
    """``max(a, b)`` smoothed over ``eps``; equals ``a`` to ``eps softplus(-(a - b)/eps)``."""
    return a + eps * F.softplus((b - a) / eps)


def saturate_rebo(b, valence, i, j, n_atoms, width: float = 0.02):
    """``(p, s)``: ``p_ij = b_ij s_i s_j`` with ``s_a = v_a / smoothmax(v_a, N_a)``."""
    n = _coordination(b, i, j, n_atoms)
    v = valence.clamp(min=_V_FLOOR)
    s = v / _smoothmax(v, n, width)
    return b * s[i] * s[j], s


def _smoothmin(x, y, eps):
    """``min(x, y)`` smoothed over ``eps``, exact on ``x = y`` and at most ``eps`` above the true
    minimum elsewhere."""
    h = 0.5 * (x - y)
    return 0.5 * (x + y) - torch.sqrt(h * h + eps * eps) + eps


def saturate_waterfill(J, kappa, b, valence, i, j, n_atoms, temperature, k_steps: int = 8,
                       reconcile_width: float = 2.0e-3, dual_sweeps: int = 0,
                       polish_steps: int = 4):
    """``(p, lam, residual)``: per-atom water filling reconciled over the two ends.

    ``dual_sweeps > 0`` adds that many Jacobi sweeps of the coupled dual: each atom refills
    against the couplings its partners have already claimed, ``J_ij - lam_j``, so a pair's
    two ends move toward the pairing solve's ``p_ij = clip((J - lam_i - lam_j) / kappa)``
    (damped by one half per sweep). **Experimental**: on the Zundel scans one or two sweeps
    shrink the under-filling of the shared proton, but the damped Jacobi iteration is not
    monotone and a sweep can leave an atom on the wrong side of a breakpoint (residual ~1);
    ``0`` (the pure per-atom rule) is the default and the only tested setting. The proper
    rung above it is the pairing solve itself.

    ``polish_steps`` differentiable Newton steps on the smooth filling follow the hard
    root: each removes a factor ~3 of the clip tails' residual (the hard root puts an
    empty partner exactly on its breakpoint, where the smooth clip is ``w ln 2``).

    ``residual`` is ``|sum_j p_a(j) - v_a|`` on the atoms whose filling is active (zero on the
    others), in the last sweep.
    """
    temperature = torch.as_tensor(temperature, dtype=J.dtype, device=J.device)
    v = valence.clamp(min=0.0)
    w = temperature / kappa                                   # clip width in p units
    zero = J.new_zeros(n_atoms)

    def fill(lam, other, split: bool = False):
        """``sum_j p_a(j) - v_a`` per atom, its derivative in ``lam_a``, and the two grants;
        ``other`` is the partners' multiplier from the previous sweep. With ``split``, the
        pieces of the log form: ``(fill, dfill, room, droom)`` with ``fill`` what the
        unsaturated partners hold and ``room = v - (what the saturated ones hold)``."""
        y_i, dy_i, c_i = _soft_clip((J - lam[i] - other[j]) / kappa, w, complement=True)
        y_j, dy_j, c_j = _soft_clip((J - lam[j] - other[i]) / kappa, w, complement=True)
        d_i, d_j = -dy_i / kappa, -dy_j / kappa                   # d y / d lam
        g = _coordination_two(y_i, y_j, i, j, n_atoms) - v
        dg = _coordination_two(d_i, d_j, i, j, n_atoms)
        if not split:
            return g, dg, y_i, y_j
        # cancellation-free: room = (v - n_sat) + sum_sat (1 - y), the pairing solve's form
        s_i, s_j = y_i > 0.5, y_j > 0.5
        one = torch.ones_like(J)
        n_sat = _coordination_two(torch.where(s_i, one, 0.0), torch.where(s_j, one, 0.0), i, j, n_atoms)
        fill_ = _coordination_two(torch.where(s_i, 0.0, y_i), torch.where(s_j, 0.0, y_j), i, j, n_atoms)
        dfill = _coordination_two(torch.where(s_i, 0.0, d_i), torch.where(s_j, 0.0, d_j), i, j, n_atoms)
        room = (v - n_sat) + _coordination_two(torch.where(s_i, c_i, 0.0), torch.where(s_j, c_j, 0.0), i, j, n_atoms)
        droom = -_coordination_two(torch.where(s_i, d_i, 0.0), torch.where(s_j, d_j, 0.0), i, j, n_atoms)
        return g, dg, fill_, dfill, room, droom

    def hard_step(lam, other):
        """One Newton step on the *unsmoothed* filling ``sum_k clip((J'_k - lam)/kappa_k)``,
        which is piecewise linear in ``lam``: exact on a linear piece, and on a flat piece
        (every partner saturated or empty) a jump to the nearest breakpoint in the direction
        of the root -- the saturated partner that unsaturates first going up, the empty one
        that fills first going down. Returns ``(g, lam_new)``."""
        inf = J.new_full((n_atoms,), float("inf"))
        y_lin_i = (J - lam[i] - other[j]) / kappa
        y_lin_j = (J - lam[j] - other[i]) / kappa
        g = _coordination_two(y_lin_i.clamp(0.0, 1.0), y_lin_j.clamp(0.0, 1.0), i, j, n_atoms) - v
        # one-sided slopes: a partner sitting exactly on a breakpoint counts toward the
        # derivative in the direction the root lies (up where g > 0, down where g < 0)
        def slope_of(mask_i, mask_j):
            return _coordination_two(
                torch.where(mask_i, 1.0 / kappa, 0.0), torch.where(mask_j, 1.0 / kappa, 0.0),
                i, j, n_atoms,
            )
        slope_up = slope_of((y_lin_i > 0.0) & (y_lin_i <= 1.0), (y_lin_j > 0.0) & (y_lin_j <= 1.0))
        slope_down = slope_of((y_lin_i >= 0.0) & (y_lin_i < 1.0), (y_lin_j >= 0.0) & (y_lin_j < 1.0))
        slope = torch.where(g > 0.0, slope_up, slope_down)
        up = inf.index_reduce(0, i, torch.where(y_lin_i > 1.0, J - other[j] - kappa, inf[:1].expand_as(J)), "amin", include_self=True)
        up = up.index_reduce(0, j, torch.where(y_lin_j > 1.0, J - other[i] - kappa, inf[:1].expand_as(J)), "amin", include_self=True)
        down = (-inf).index_reduce(0, i, torch.where(y_lin_i < 0.0, J - other[j], -inf[:1].expand_as(J)), "amax", include_self=True)
        down = down.index_reduce(0, j, torch.where(y_lin_j < 0.0, J - other[i], -inf[:1].expand_as(J)), "amax", include_self=True)
        flat = slope <= 0.0
        newton = lam + g / slope.clamp(min=1.0e-300)
        jump = torch.where(g > 0.0, up, down)
        new = torch.where(flat, jump, newton)
        # the hard filling has a whole interval of roots when the capacity is met exactly by
        # saturated partners alone (g = 0 on a flat piece); take its bottom -- the smallest
        # multiplier that does the job, which is where the smooth root sits too
        at_root = g.abs() <= 1.0e-9
        new = torch.where(at_root & (slope_down > 0.0), lam, new)
        new = torch.where(at_root & (slope_down <= 0.0) & torch.isfinite(down), down, new)
        return g, new

    def root(other, start):
        """The per-atom root against ``other`` from ``start``: the piecewise-linear Newton
        with a bracket, without the graph, then two differentiable Newton steps on the
        smoothed filling (exact first and second derivatives of the implicit function,
        since the Newton map has zero derivative in ``lam`` at a root)."""
        with torch.no_grad():
            g0, _, _, _ = fill(zero, other)
            active = g0 > 0.0                                  # more raw order than capacity
            hi = torch.zeros_like(zero).index_reduce_(0, i, J, "amax", include_self=True)
            hi = hi.index_reduce_(0, j, J, "amax", include_self=True) + kappa.max()
            lo = zero.clone()
            lam = torch.where(active, torch.minimum(start.detach(), hi), zero)
            for _ in range(int(k_steps)):
                g, newton = hard_step(lam, other)
                # g is monotone decreasing in lam: the root is above lam where g > 0
                lo = torch.where(g > 0.0, lam, lo)
                hi = torch.where(g < 0.0, lam, hi)
                inside = torch.isfinite(newton) & (newton >= lo) & (newton <= hi)
                lam = torch.where(inside, newton, 0.5 * (lo + hi))
                lam = torch.where(active, lam, zero)
        # Newton on the smooth filling from the hard root, in the pairing solve's log form
        # (``ln fill - ln room``): the hard root leaves every squeezed partner exactly on its
        # breakpoint, and from there the smooth root is where its clip *tail* balances the
        # bonds' tails -- an exponential in lam, on which the plain step moves one e-fold
        # (``T``) per iteration and the log form is exact. Differentiable: at a root the
        # Newton map has zero derivative in lam, so these steps carry the implicit first and
        # second derivatives. Kept inside ``[0, hi + kappa]`` (an unconverged atom is not
        # thrown off; where the clamp binds the derivative is zero, which is what an
        # unconverged root deserves).
        tiny = torch.finfo(J.dtype).tiny
        cap = hi + kappa.max()
        for _ in range(int(polish_steps)):
            g, dg, fill_, dfill, room, droom = fill(lam, other, split=True)
            plain = lam - g / dg.clamp(max=-1.0e-12)
            ok = (fill_ > tiny) & (room > tiny)
            h = torch.log(fill_.clamp(min=tiny)) - torch.log(room.clamp(min=tiny))
            dh = dfill / fill_.clamp(min=tiny) - droom / room.clamp(min=tiny)
            logform = lam - h / dh.clamp(max=-1.0e-12)
            logform = torch.where(ok, logform, plain)
            # the log form is exact once both tails are asymptotic and overshoots before
            # that (the squeezed partner sits on its breakpoint, where softplus has half
            # its asymptotic slope); try it and its two bisections toward the current
            # point, and keep whichever candidate has the smallest plain residual -- the
            # current point if none improves (comparisons only, no gradient of their own)
            candidates = (logform, 0.5 * (lam + logform), 0.25 * (3.0 * lam + logform), plain)
            with torch.no_grad():
                best_g = g.abs()
                best_k = torch.full_like(lam, -1.0)
                for k, cand in enumerate(candidates):
                    g_k = fill(torch.maximum(torch.minimum(cand, cap), zero), other)[0].abs()
                    better = g_k < best_g
                    best_g = torch.where(better, g_k, best_g)
                    best_k = torch.where(better, torch.full_like(lam, float(k)), best_k)
            newton = lam
            for k, cand in enumerate(candidates):
                newton = torch.where(best_k == float(k), cand, newton)
            lam = torch.where(active, torch.maximum(torch.minimum(newton, cap), zero), zero)
        return lam, active

    other = zero
    lam, active = root(other, zero)
    for _ in range(int(dual_sweeps)):
        other = 0.5 * (other + lam)
        lam, active = root(other, lam)                         # warm start: the root moves little
    g, _, y_i, y_j = fill(lam, other)
    residual = torch.where(active, g.abs(), zero)
    # the smaller of the two ends' grants (each is <= b: the multipliers only lower the clip)
    p = _smoothmin(y_i, y_j, reconcile_width)
    return p, lam, residual


def overbinding_penalty(b, valence, i, j, n_atoms, kappa_atom):
    """ReaxFF-style ``sum_a kappa_a softplus(N_a - v_a)^2`` on the *raw* coordination."""
    n = _coordination(b, i, j, n_atoms)
    over = F.softplus(n - valence)
    return kappa_atom * over * over


def explicit_state(
    J: torch.Tensor,             # (Pb,)
    kappa: torch.Tensor,         # (Pb,)
    tables: torch.Tensor,        # (N, 5) capacity rows (n0, v0, a, b, shell)
    q: torch.Tensor,             # (N,) formal charges
    sub_pairs: torch.Tensor,     # (2, Pb)
    batch_idx: torch.Tensor,     # (N,)
    n_systems: int,
    temperature,
    *,
    saturation: str = "waterfill",
    width: float = 0.02,
    k_steps: int = 8,
    reconcile_width: float = 2.0e-3,
    dual_sweeps: int = 0,
    polish_steps: int = 4,
) -> ElectronicState:
    """The explicit electronic state of every frame: bond orders, unpaired counts, capacities.

    Returned in the pairing model's :class:`ElectronicState` so everything downstream (the
    co-membership, the conditioning, the diagnostics and plots) reads it unchanged. ``lam``
    is the water-filling multiplier (zeros under ``rebo``); ``mu`` / ``nu`` are zeros;
    ``residual`` is the per-frame maximum filling residual; ``converged`` is ``residual``
    below ``10 T``.
    """
    if saturation not in SATURATIONS:
        raise ValueError(f"saturation must be one of {SATURATIONS}, got {saturation!r}")
    n_atoms = int(tables.shape[0])
    i, j = sub_pairs[0], sub_pairs[1]
    n0, v0, a, bq = tables[:, 0], tables[:, 1], tables[:, 2], tables[:, 3]
    n = n0 - q
    valence, _ = capacity(n, n0, v0, a, bq)
    b = raw_bond_order(J, kappa, temperature)
    if saturation == "rebo":
        p, _ = saturate_rebo(b, valence, i, j, n_atoms, width=width)
        lam = J.new_zeros(n_atoms)
        residual_atom = J.new_zeros(n_atoms)
    else:
        p, lam, residual_atom = saturate_waterfill(
            J, kappa, b, valence, i, j, n_atoms, temperature, k_steps=k_steps,
            reconcile_width=reconcile_width, dual_sweeps=dual_sweeps,
            polish_steps=polish_steps,
        )
    u = (valence - _coordination(p, i, j, n_atoms)).clamp(min=0.0)
    residual = (
        J.new_zeros(n_systems)
        .index_reduce_(0, batch_idx, residual_atom.detach(), "amax", include_self=True)
    )
    tol = 10.0 * float(torch.as_tensor(temperature))
    return ElectronicState(
        p=p, u=u, n=n, q=q, valence=valence, lam=lam,
        mu=J.new_zeros(n_systems), nu=J.new_zeros(n_systems),
        n_iter=int(k_steps) + int(polish_steps) if saturation == "waterfill" else 0,
        converged=residual <= tol, residual=residual,
    )


def tersoff_state_energy(J, kappa, chi, eta, st: ElectronicState):
    """``(per-pair, per-atom)`` terms, Hartree: ``-J p + kappa p^2 / 2`` and ``E0(q) - E0(0)``.

    No entropic terms (they exist to smooth a *solve*; the explicit ``p`` is smooth by
    construction), so a free neutral atom contributes exactly zero and a saturated bond is
    ``-J + kappa / 2`` as in the pairing model.
    """
    e_pair = -J * st.p + 0.5 * kappa * st.p * st.p
    e_atom = e0_terms(st.q, chi, eta)[0]
    return e_pair, e_atom
