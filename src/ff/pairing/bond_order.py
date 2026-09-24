"""The bond order as a variational state: pairing energy, the valence-constrained solve, and
the co-membership it induces.

The functional (``docs/fff_pairing.md`` §2), in Hartree, over a candidate pair list ``(i, j)``
with ``i < j`` and the atoms it touches::

    F(p) = sum_ij [ -J_ij p_ij + 1/2 kappa_ij p_ij^2 + T ( p ln p + (1 - p) ln(1 - p) ) ]
         + T sum_i [ u_i ln u_i - v_i ln v_i ]

    subject to   sum_j p_ij + u_i = v_i   for every atom i.

``J_ij`` is the bare pairing energy (a Slater-damped multipolar overlap the caller computes --
the same operator as the Pauli repulsion), ``kappa_ij > 0`` the pairing hardness, ``v_i`` the
valence capacity (unpaired valence electrons available for pairing) and ``u_i`` the residual
unpaired count. The two entropic terms are barriers: the pair one keeps ``0 < p < 1``, the
atom one keeps ``u > 0``, and both vanish as ``T -> 0``, where the problem is the quadratic
program "pair up as much as the couplings reward, never more than the valence allows". ``F`` is
strictly convex on a convex set, so the minimizer is unique and a smooth function of the
parameters; that is what makes the resulting potential well-posed.

**Solver.** The equality constraints are handled in the dual. With multipliers ``lambda_i`` the
stationarity conditions decouple per pair and per atom::

    kappa_ij p_ij + T logit(p_ij) = J_ij - lambda_i - lambda_j        (a scalar equation)
    u_i = exp(-1 - lambda_i / T)

and the multipliers are fixed by the ``N`` marginal equations
``R_i(lambda) = sum_j p_ij + u_i - v_i = 0``, whose Jacobian is ``-(D + A)`` with
``A_ij = dp_ij/da`` and ``D_ii = sum_j A_ij + u_i / T`` -- symmetric, diagonally dominant,
positive definite. That system is solved by a damped Newton iteration with a dense batched
factorization per frame (a frame is a few hundred atoms at most; revisit with CG if that ever
changes).

**Derivatives.** Nothing is unrolled. The Newton iteration runs under ``no_grad`` to
convergence, then *two* Newton steps are re-applied with the graph on, from the converged
(detached) point. One differentiable Newton step from the solution reproduces the exact first
derivative of the implicit function (the residual is zero there, so the derivative of the
inverse Jacobian drops out); two steps reproduce the exact second derivative as well, which is
what a force loss needs (``d/dtheta [dE/dR]`` runs through ``dp/dR``). The same trick is used
inside the scalar per-pair equation. See ``docs/fff_pairing.md`` §2.3 for the argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "BondOrderSolution",
    "DEFAULT_VALENCE",
    "comembership_from_bond_order",
    "pairing_energy",
    "solve_bond_order",
    "valence_table",
]

#: Valence capacity per element: the number of unpaired valence electrons of the neutral
#: ground-state atom that are available for covalent pairing. Charge dependence enters through
#: the model (``docs/fff_pairing.md`` §3), not here.
DEFAULT_VALENCE: dict[int, float] = {
    1: 1.0, 2: 0.0,
    3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0, 7: 3.0, 8: 2.0, 9: 1.0, 10: 0.0,
    11: 1.0, 12: 2.0, 13: 3.0, 14: 4.0, 15: 3.0, 16: 2.0, 17: 1.0, 18: 0.0,
    19: 1.0, 20: 2.0, 35: 1.0, 53: 1.0,
}


def valence_table(neighbor_types, overrides: dict[int, float] | None = None) -> torch.Tensor:
    """``(n_species,)`` valence capacities ordered like ``neighbor_types``. Refuses to guess."""
    table = dict(DEFAULT_VALENCE)
    if overrides:
        table.update({int(z): float(v) for z, v in overrides.items()})
    missing = [int(z) for z in neighbor_types if int(z) not in table]
    if missing:
        raise KeyError(
            f"no valence capacity for atomic number(s) {missing}; extend DEFAULT_VALENCE or "
            f"pass valence={{Z: value}}"
        )
    return torch.tensor([float(table[int(z)]) for z in neighbor_types])


def _xlogx(x: torch.Tensor) -> torch.Tensor:
    """``x ln x`` with a finite gradient at ``x = 0`` (a saturated ``p`` or ``1 - p`` underflows
    to an exact zero, and ``xlogy``'s ``-inf`` there times a zero ``dp/dR`` is a NaN force)."""
    return x * torch.log(x.clamp(min=torch.finfo(x.dtype).tiny))


# ---------------------------------------------------------------------------
# the scalar per-pair equation


def _pair_residual(x, a, kappa, temperature):
    """``phi(x) = kappa sigma(x) + T x - a`` and ``phi'(x) > 0`` for ``x = logit(p)``."""
    s = torch.sigmoid(x)
    return kappa * s + temperature * x - a, kappa * s * (1.0 - s) + temperature


def _pair_newton_step(x, a, kappa, temperature):
    phi, dphi = _pair_residual(x, a, kappa, temperature)
    return x - phi / dphi


def _solve_pair_logit(
    a: torch.Tensor, kappa: torch.Tensor, temperature: torch.Tensor, *, maxiter: int = 40
) -> torch.Tensor:
    """``x = logit(p)`` solving ``kappa p + T logit(p) = a`` elementwise, differentiable to
    second order.

    ``phi`` is strictly increasing with ``phi'' = kappa s(1-s)(1-2s)`` bounded, so Newton
    with a backtracking guard on ``|phi|`` converges from the piecewise-linear start below
    (the two asymptotic branches, and the linearization about ``p = 1/2`` between them).
    The logit is what the caller wants: ``1 - p = sigmoid(-x)`` is then exact where ``p``
    itself rounds to one.
    """
    from . import fused

    if fused.fused_enabled(a):
        x = fused.pair_logit(a, kappa, float(temperature), maxiter=maxiter)
        x = _pair_newton_step(x, a, kappa, temperature)
        return _pair_newton_step(x, a, kappa, temperature)
    with torch.no_grad():
        x = torch.where(
            a > kappa, (a - kappa) / temperature,
            torch.where(a < 0.0, a / temperature, (a - 0.5 * kappa) / (temperature + 0.25 * kappa)),
        )
        phi, _ = _pair_residual(x, a, kappa, temperature)
        for _ in range(maxiter):
            x_new = _pair_newton_step(x, a, kappa, temperature)
            phi_new, _ = _pair_residual(x_new, a, kappa, temperature)
            worse = phi_new.abs() > phi.abs()
            for _ in range(6):
                if not bool(worse.any()):
                    break
                x_half = torch.where(worse, 0.5 * (x + x_new), x_new)
                phi_half, _ = _pair_residual(x_half, a, kappa, temperature)
                x_new = torch.where(worse, x_half, x_new)
                phi_new = torch.where(worse, phi_half, phi_new)
                worse = worse & (phi_new.abs() > phi.abs())
            x, phi = x_new, phi_new
            if bool((phi.abs() <= 1.0e-13 * (1.0 + a.abs())).all()):
                break
    # two differentiable Newton steps from the converged point: exact to second order
    x = x.detach()
    x = _pair_newton_step(x, a, kappa, temperature)
    x = _pair_newton_step(x, a, kappa, temperature)
    return x


def _solve_pair(a, kappa, temperature):
    """``p`` solving ``kappa p + T logit(p) = a`` elementwise (see :func:`_solve_pair_logit`)."""
    return torch.sigmoid(_solve_pair_logit(a, kappa, temperature))


def _dp_da(p: torch.Tensor, pc: torch.Tensor, kappa: torch.Tensor, temperature: torch.Tensor):
    """``dp/da = p(1-p) / (kappa p(1-p) + T)``, with ``1 - p`` supplied exactly."""
    s = p * pc
    return s / (kappa * s + temperature)


# ---------------------------------------------------------------------------
# the marginal (dual) system


@dataclass
class BondOrderSolution:
    """The converged bond order and everything a diagnostic wants to know about the solve."""

    p: torch.Tensor            # (Pb,) bond orders on the candidate pairs
    u: torch.Tensor            # (N,) residual unpaired valence
    lam: torch.Tensor          # (N,) dual variables
    n_iter: int
    converged: torch.Tensor    # (B,) bool per frame
    residual: torch.Tensor     # (B,) final max |R| per frame


def _frame_layout(batch_idx: torch.Tensor, n_systems: int):
    """``(local (N,), sizes (B,))``: index of each atom within its frame, and frame sizes."""
    sizes = torch.bincount(batch_idx, minlength=int(n_systems))
    starts = torch.cumsum(sizes, 0) - sizes
    local = torch.arange(batch_idx.shape[0], device=batch_idx.device) - starts[batch_idx]
    return local, sizes


def _marginal(lam, J, kappa, temperature, valence, i, j, n_atoms):
    """Everything one Newton iteration needs at the multipliers ``lam``.

    Returns ``(R, p, w, dpda, dpda, diag, |R|, g_pair, g_atom)``:

    ``R = fill - room`` is the marginal residual, evaluated as a balance of two positive
    sides so it is cancellation-free on a saturated atom (its bonds sit at ``p = 1 - e^{-big}``
    and what they leave must equal the slack plus the ``e^{-big}`` bond orders of the pairs it
    does not form)::

        room_i = (v_i - n_sat_i) + sum_{sat j} (1 - p_ij)      [saturated pairs: x > 0,
        fill_i = u_i + sum_{unsat j} p_ij                        1 - p = sigmoid(-x) exactly]

    ``R`` is also the gradient of the dual function ``g(lam) = min_p L`` whose Hessian is
    ``-K`` with ``K = diag(u/T + sum_j dp/da) + offdiag(dp/da)``: symmetric positive definite,
    so the Newton step ``K^{-1} R`` is an ascent direction of ``g``, which is what the line
    search reads. ``w = v - sum_j p`` is the primal slack. ``g_pair`` / ``g_atom`` are the
    dual function's terms (the atom entropy collapses to ``-T u`` at the optimum).
    """
    a = J - lam[i] - lam[j]
    x = _solve_pair_logit(a, kappa, temperature)
    p = torch.sigmoid(x)
    pc = torch.sigmoid(-x)
    is_sat = x > 0.0
    sat = is_sat.to(p.dtype)
    zeros = torch.zeros_like(p)
    room_pair = torch.where(is_sat, pc, zeros)
    fill_pair = torch.where(is_sat, zeros, p)
    n_sat = torch.zeros_like(valence).index_add(0, i, sat).index_add(0, j, sat)
    room = (valence - n_sat).index_add(0, i, room_pair).index_add(0, j, room_pair)
    log_u = (-1.0 - lam / temperature).clamp(-700.0, 60.0)
    u = torch.exp(log_u)
    fill = u.index_add(0, i, fill_pair).index_add(0, j, fill_pair)
    R = fill - room
    dpda = _dp_da(p, pc, kappa, temperature)
    diag = (u / temperature).index_add(0, i, dpda).index_add(0, j, dpda)
    w = room - (fill - u)
    ent = _xlogx(p) + _xlogx(pc)
    g_pair = -J * p + 0.5 * kappa * p * p + temperature * ent + (lam[i] + lam[j]) * p
    g_atom = -temperature * u - lam * valence
    return R, p, w, dpda, dpda, diag, R.abs(), g_pair, g_atom


def _newton_direction(rhs, coef_i, coef_j, diag, i, j, batch_idx, local, n_systems, n_max):
    """``delta = (K + rho I)^{-1} rhs`` per frame, batched dense; ``lam + delta`` is the step.

    ``K`` has ``diag`` on the diagonal, ``coef_i`` at ``(i, j)`` and ``coef_j`` at ``(j, i)``:
    symmetric and diagonally dominant at the solution. ``rho = sqrt(eps)`` is a
    Levenberg-Marquardt floor for the rows of atoms whose every term has underflowed (a
    bond's ``1 - p`` at ``e^-200``): nothing there is determined, and the floor keeps the
    step finite instead of roundoff divided by nothing.
    """
    dtype, device = rhs.dtype, rhs.device
    rho = float(torch.finfo(dtype).eps) ** 0.5
    K = torch.eye(int(n_max), dtype=dtype, device=device).unsqueeze(0).repeat(int(n_systems), 1, 1)
    b_atom, l_atom = batch_idx, local
    K = K.index_put((b_atom, l_atom, l_atom), diag + rho - 1.0, accumulate=True)
    b_pair = batch_idx[i]
    K = K.index_put((b_pair, local[i], local[j]), coef_i, accumulate=True)
    K = K.index_put((b_pair, local[j], local[i]), coef_j, accumulate=True)
    b = torch.zeros(int(n_systems), int(n_max), dtype=dtype, device=device)
    b = b.index_put((b_atom, l_atom), rhs)
    delta = torch.linalg.solve(K, b.unsqueeze(-1)).squeeze(-1)
    return delta[b_atom, l_atom]


def solve_bond_order(
    J: torch.Tensor,             # (Pb,) bare pairing energies, Hartree
    kappa: torch.Tensor,         # (Pb,) pairing hardness per pair, Hartree
    valence: torch.Tensor,       # (N,) valence capacity per atom
    temperature,                 # scalar tensor or float, Hartree
    pair_index: torch.Tensor,    # (2, Pb) candidate pairs, i < j
    batch_idx: torch.Tensor,     # (N,) frame per atom, non-decreasing
    n_systems: int,
    *,
    tol: float = 1.0e-10,
    maxiter: int = 100,
    step_max: float | None = None,
) -> BondOrderSolution:
    """Minimize the pairing functional under the valence marginals. Differentiable to 2nd order.

    The forward iteration is a damped Newton method on the multipliers (trust region on the
    step, frame-wise backtracking on the residual); the graph is attached by two Newton steps
    re-applied from the converged point (module docstring). Convergence is judged on what the
    next step would do to ``p`` and ``w``, which is well conditioned, rather than on the
    multipliers, which sit on a roundoff floor on every saturated atom. Frames still moving
    after ``maxiter`` steps are reported in ``converged`` rather than raised: the training
    loop logs them like the coupled solve's ``cg_fail``.

    The returned ``u`` is the primal slack ``w = v - sum_j p`` (cancellation-free), so the
    marginal holds exactly by construction and the energy needs no exponential.
    """
    n_atoms = int(valence.shape[0])
    i, j = pair_index[0], pair_index[1]
    temperature = torch.as_tensor(temperature, dtype=J.dtype, device=J.device)
    if step_max is None:
        # a multiplier moves by at most 20 T per iteration: the dual slack then changes by
        # at most e^20 per step, which the line search can still undo
        step_max = 20.0 * float(temperature)
    local, sizes = _frame_layout(batch_idx, n_systems)
    n_max = int(sizes.max()) if sizes.numel() else 0
    tol = max(float(tol), 8.0 * float(torch.finfo(J.dtype).eps))

    def frame_max(x, index=None):
        index = batch_idx if index is None else index
        return x.abs().new_zeros(int(n_systems)).scatter_reduce(
            0, index, x.abs(), "amax", include_self=True
        )

    def state(lam):
        return _marginal(lam, J, kappa, temperature, valence, i, j, n_atoms)

    def dual(g_pair, g_atom):
        return g_atom.new_zeros(int(n_systems)).index_add_(0, batch_idx, g_atom).index_add_(
            0, batch_idx[i], g_pair
        )

    with torch.no_grad():
        # Start: an atom without partners has u = v exactly; an atom with a pair that will
        # saturate (J > kappa) starts at half that pair's excess, which puts the pair at
        # p = 1/2 and its slack already at e^{-(J - kappa)/2T}. Newton on the exponential
        # slack converges by one e-fold per step from the over-large side, so starting the
        # slack small is what keeps a bonded atom from costing ~50 iterations.
        lam = -temperature * (1.0 + valence.clamp(min=1.0e-12).log())
        excess = 0.5 * (J - kappa)
        half = torch.zeros_like(lam).scatter_reduce(0, i, excess, "amax", include_self=True)
        half = half.scatter_reduce(0, j, excess, "amax", include_self=True)
        lam = torch.maximum(lam, half)
        st = state(lam)
        rhs, p, w, ci, cj, diag, merit = st[:7]
        err = frame_max(merit)
        g = dual(*st[7:])
        n_iter = 0
        done = err <= tol
        for it in range(maxiter):
            if bool(done.all()):
                break
            n_iter = it + 1
            delta = _newton_direction(rhs, ci, cj, diag, i, j, batch_idx, local, n_systems, n_max)
            move = frame_max(delta)
            delta = delta * (step_max / move.clamp(min=step_max))[batch_idx]
            # the Newton step is an ascent direction of the concave dual g (its Hessian is
            # -K); accept by Armijo on g, or by a decrease of the primal imbalance, whichever
            # holds -- g is what carries a step across a saturated plateau, the imbalance
            # what settles the last digits where g's gain is below roundoff
            slope = (rhs * delta).new_zeros(int(n_systems)).index_add_(0, batch_idx, rhs * delta)
            active = ~done
            step = active.to(J.dtype)
            best = None
            for _ in range(30):
                lam_try = lam + step[batch_idx] * delta
                trial = state(lam_try)
                err_try = frame_max(trial[6])
                g_try = dual(*trial[7:])
                armijo = g_try >= g + 1.0e-4 * step * slope
                improved = ~active | armijo | (err_try <= err)
                if best is None:
                    best = (lam_try, *trial[:7], err_try, g_try, improved)
                else:
                    # per frame: keep the first accepted step; a frame that has not
                    # accepted yet takes the newest (smallest) trial
                    take = ~best[-1]
                    take_a = take[batch_idx]
                    take_p = take[batch_idx[i]]
                    best = (
                        torch.where(take_a, lam_try, best[0]),
                        torch.where(take_a, trial[0], best[1]),
                        torch.where(take_p, trial[1], best[2]),
                        torch.where(take_a, trial[2], best[3]),
                        torch.where(take_p, trial[3], best[4]),
                        torch.where(take_p, trial[4], best[5]),
                        torch.where(take_a, trial[5], best[6]),
                        torch.where(take_a, trial[6], best[7]),
                        torch.where(take, err_try, best[8]),
                        torch.where(take, g_try, best[9]),
                        best[-1] | improved,
                    )
                if bool(best[-1].all()):
                    break
                step = torch.where(improved, step, 0.5 * step)
            lam, rhs, p, w, ci, cj, diag, merit, err, g = best[:10]
            done = done | (err <= tol)
        converged = done

    # the graph: two Newton steps from the converged multipliers (the trust-region clamp is
    # inactive there and differentiates as the identity)
    lam = lam.detach()
    for _ in range(2):
        rhs, p, w, ci, cj, diag, merit = state(lam)[:7]
        delta = _newton_direction(rhs, ci, cj, diag, i, j, batch_idx, local, n_systems, n_max)
        lam = lam + delta * (step_max / frame_max(delta).clamp(min=step_max))[batch_idx]
    rhs, p, w, ci, cj, diag, merit = state(lam)[:7]
    return BondOrderSolution(
        p=p, u=w.clamp(min=0.0), lam=lam, n_iter=n_iter, converged=converged,
        residual=frame_max(merit.detach())
    )


# ---------------------------------------------------------------------------
# energy and co-membership


def pairing_energy(
    J: torch.Tensor,
    kappa: torch.Tensor,
    temperature,
    p: torch.Tensor,
    u: torch.Tensor,
    valence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(per-pair (Pb,), per-atom (N,))`` terms of the minimized functional, Hartree.

    The per-atom part is referenced to the free atom (``u = v``), so an atom without partners
    contributes exactly zero and the isolated-atom reference energies stay what they are.
    """
    temperature = torch.as_tensor(temperature, dtype=J.dtype, device=J.device)
    e_pair = -J * p + 0.5 * kappa * p * p + temperature * (_xlogx(p) + _xlogx(1.0 - p))
    e_atom = temperature * (_xlogx(u) - _xlogx(valence))
    return e_pair, e_atom


def _two_paths(pair_index: torch.Tensor, n_atoms: int):
    """``(e1, e2, i, k)``: every pair of candidate edges sharing an apex, as edge indices.

    Same padded-triangle enumeration as :func:`rsfff.ff.film.bonded._angles_from_bonds`, but
    it returns the *edge* indices (so the caller can multiply their bond orders) and the two
    end atoms.
    """
    device = pair_index.device
    n_edges = pair_index.shape[1]
    if n_edges == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, z, z
    edges = torch.arange(n_edges, device=device)
    atoms = torch.cat((pair_index[0], pair_index[1]))
    others = torch.cat((pair_index[1], pair_index[0]))
    edge_of = torch.cat((edges, edges))
    order = torch.argsort(atoms, stable=True)
    others_s, edge_s = others[order], edge_of[order]
    counts = torch.bincount(atoms[order], minlength=n_atoms)
    deg_max = int(counts.max())
    if deg_max < 2:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, z, z
    offsets = torch.cumsum(counts, 0) - counts
    a, b = torch.triu_indices(deg_max, deg_max, offset=1, device=device)
    keep = b.unsqueeze(0) < counts.unsqueeze(1)
    first = (offsets.unsqueeze(1) + a.unsqueeze(0))[keep]
    second = (offsets.unsqueeze(1) + b.unsqueeze(0))[keep]
    return edge_s[first], edge_s[second], others_s[first], others_s[second]


def comembership_from_bond_order(
    p: torch.Tensor,             # (Pb,) bond orders on the candidate pairs
    sub_index: torch.Tensor,     # (Pb,) position of each candidate pair in the full list
    pair_index: torch.Tensor,    # (2, P) the full pair list, sorted by (i, j), i < j
    n_atoms: int,
    *,
    include_13: bool = True,
) -> torch.Tensor:
    """``(P,)`` co-membership ``c_ij`` on the full pair list, from the bond orders.

    ``c = 1 - (1 - c12)(1 - c13)`` with ``c12 = p_ij`` on a candidate pair (0 elsewhere) and
    ``c13 = 1 - prod_k (1 - p_ik p_kj)`` over the two-step paths of the candidate graph. This
    is the fractional-bond-order version of the classical 1-2 / 1-3 exclusions: exactly the
    film model's ``P_ij`` on a saturated graph, and it decays smoothly with the bond orders as
    a bond breaks. 1-4 paths are not enumerated (nothing in the water data needs them; add a
    third product when torsional fragments appear).
    """
    n_pairs = pair_index.shape[1]
    c12 = p.new_zeros(n_pairs).index_put((sub_index,), p)
    log_keep = torch.zeros_like(c12)
    if include_13:
        sub_pairs = pair_index[:, sub_index]
        e1, e2, i, k = _two_paths(sub_pairs, n_atoms)
        if e1.numel():
            lo = torch.minimum(i, k)
            hi = torch.maximum(i, k)
            keys = pair_index[0] * n_atoms + pair_index[1]
            want = lo * n_atoms + hi
            pos = torch.searchsorted(keys, want).clamp(max=max(n_pairs - 1, 0))
            found = keys[pos] == want
            w = (p[e1] * p[e2])[found]
            log_keep = log_keep.index_add(
                0, pos[found], torch.log1p(-w.clamp(max=1.0 - 1.0e-12))
            )
    return 1.0 - (1.0 - c12) * torch.exp(log_keep)
