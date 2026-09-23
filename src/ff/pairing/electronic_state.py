"""The joint bond-order / formal-count solve: the electronic state of a frame from its geometry.

``docs/fff_pairing.md`` §7. :mod:`bond_order` solves the pairing functional at a **given**
capacity per atom. Here the capacity is a function of the atom's *formal* electron count
``n_i``, and ``n_i`` is a variable of the same minimization::

    min_{p, u, n}   F(p, u)  +  sum_i E0_i(q_i)                 q_i = n0_i - n_i

    s.t.   sum_j p_ij + u_i = v_i(n_i)         (per atom)
           sum_{i in frame} n_i = N_frame      (electron count; fixed by the total charge)
           sum_{i in frame} u_i = 2S_frame     (multiplicity; only when 2S > 0)

with ``F`` the pairing functional of :mod:`bond_order`, ``E0(q)`` the charged atomic reference
(piecewise linear through ``IP`` and ``-EA``, smoothed at the integer -- see :func:`e0_terms`:
the cost of moving a formal electron, and the reason formal charges stay integer unless the
bonding pays), and
the **capacity polynomial** ``v(n) = v0 + a (n - n0) + b (n - n0)^2`` per element: linear
``c - n`` on the electron-rich side of the shell (O: 2 at n = 6, 3 at 5, 1 at 7), and a
smooth peak for a half-filled one (H: ``n(2 - n)``, 1 at n = 1 and 0 at 0 or 2).

Two charges, deliberately distinct
----------------------------------
``n_i`` is the Lewis-structure electron count -- integer-like, it moves only when a bond is
made or broken, and it is what sets how many bonds an atom can form. The SQE / partial
charge is bond *polarity* and lives downstream; feeding it into the capacity would give
water's oxygen a capacity near 1.6 and is exactly the mistake this separation avoids. What
drives ``n_i`` here is the pairing pressure ``lambda_i`` against the atomic ``E0``: a
saturated oxygen with a third proton pressing on it pays ``IP(H) - EA(O)``-ish for a formal
electron and buys a third bond with it (hydronium), a Zundel midpoint settles at ``n = 5.5``
on both oxygens, and an intact water stays within ~0.01 e of the integers, because moving
formal charge onto a hydrogen only lowers its capacity below one.

What this solve cannot do on its own is *create* ions in solution: the electrostatic
stabilization of a separated ion pair lives in the induction solve, not in ``E0``. That is
the one place a perturbative feedback (the environment's potential ``phi_i q_i`` added to
``E0``) will be needed; it is not here yet.

Solver
------
The dual in ``(lambda, mu, nu)`` -- the multipliers of the three constraints -- is concave
and its inner minimizations are scalar: ``p`` from the pair equation of :mod:`bond_order`,
``u_i = exp(-1 - (lambda_i + nu)/T)``, and ``n_i`` from a monotone stationarity condition
solved by Newton per atom. The residuals are
the constraints, their Jacobian is minus the symmetric dual Hessian, and the bordered
``(N + 2) x (N + 2)`` system per frame is factorized densely, as in :mod:`bond_order`, with
the same Armijo line search on the dual and the same two differentiable Newton steps from
the converged point for exact first and second derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bond_order import (
    DEFAULT_VALENCE,
    _dp_da,
    _frame_layout,
    _solve_pair_logit,
    _xlogx,
)

__all__ = [
    "CAPACITY_SHELL",
    "DEFAULT_CHI_ETA",
    "chi_eta_table",
    "ElectronicState",
    "capacity_table",
    "solve_electronic_state",
    "state_energy",
]

#: Valence-shell capacity per element (electrons): 2 for the first row, 8 for the others.
CAPACITY_SHELL: dict[int, int] = {1: 2, 2: 2}

#: ``(chi, eta)`` = ((IP + EA)/2, IP - EA) in Hartree at wB97M-V/def2-TZVPD
#: (``data/atomic_reference_states_wb97mv_tzvpd.json``), the fallback when a model is built
#: without the states file. Only what the water data needs; other elements come from the file.
DEFAULT_CHI_ETA: dict[int, tuple[float, float]] = {
    1: (0.2480122495, 0.4921976311),
    8: (0.2808177263, 0.4523349594),
}


def chi_eta_table(neighbor_types, atomic_states=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(chi (n_species,), eta (n_species,))`` from an ``AtomicStateReference`` or the defaults."""
    if atomic_states is not None:
        chi = atomic_states.chi_mulliken.clone().to(torch.get_default_dtype())
        eta = atomic_states.hardness.clone().to(torch.get_default_dtype())
        bad = torch.isnan(chi) | torch.isnan(eta)
        if bool(bad.any()):
            fill = torch.tensor([DEFAULT_CHI_ETA.get(int(z), (0.25, 0.5)) for z in neighbor_types])
            chi = torch.where(bad, fill[:, 0].to(chi.dtype), chi)
            eta = torch.where(bad, fill[:, 1].to(eta.dtype), eta)
        return chi, eta
    rows = [DEFAULT_CHI_ETA.get(int(z), (0.25, 0.5)) for z in neighbor_types]
    t = torch.tensor(rows)
    return t[:, 0].clone(), t[:, 1].clone()


def _shell(z: int) -> int:
    return CAPACITY_SHELL.get(int(z), 8)


def _n_valence(z: int) -> int:
    """Neutral valence-electron count: H 1, He 2, Li..Ne 1..8, Na..Ar 1..8, ..."""
    z = int(z)
    if z <= 2:
        return z
    if z <= 10:
        return z - 2
    if z <= 18:
        return z - 10
    return {19: 1, 20: 2, 35: 7, 53: 7}.get(z, 8)


def capacity_table(neighbor_types, valence_overrides=None) -> torch.Tensor:
    """``(n_species, 5)`` rows ``(n0, v0, a, b, shell)`` of ``v(n) = v0 + a (n - n0) + b (n - n0)^2``
    and the shell capacity ``shell`` that bounds ``0 < n < shell``.

    ``v0`` is the neutral valence (:data:`DEFAULT_VALENCE`). Electron-rich elements
    (``n0 > shell/2``) gain capacity by losing electrons, ``a = -1``; electron-poor ones
    (``n0 < shell/2``) by gaining them, ``a = +1``; a half-filled shell is at its peak,
    ``a = 0, b = -1``.
    """
    table = dict(DEFAULT_VALENCE)
    if valence_overrides:
        table.update({int(z): float(v) for z, v in valence_overrides.items()})
    rows = []
    for z in neighbor_types:
        z = int(z)
        if z not in table:
            raise KeyError(f"no valence capacity for Z={z}; extend DEFAULT_VALENCE")
        n0, shell = _n_valence(z), _shell(z)
        v0 = float(table[z])
        if 2 * n0 == shell:
            a, b = 0.0, -1.0
        elif 2 * n0 > shell:
            a, b = -1.0, 0.0
        else:
            a, b = 1.0, 0.0
        rows.append([float(n0), v0, a, b, float(shell)])
    return torch.tensor(rows)


def capacity(n, n0, v0, a, b):
    """``v(n)`` and ``v'(n)``; the capacity is clamped at zero from below (smoothly enough:
    the clamp is only reached for a proton or a hydride, which pair nothing anyway)."""
    d = n - n0
    v = v0 + a * d + b * d * d
    dv = a + 2.0 * b * d
    return v.clamp(min=0.0), torch.where(v > 0.0, dv, torch.zeros_like(dv))


@dataclass
class ElectronicState:
    """The converged state of every frame in a batch."""

    p: torch.Tensor          # (Pb,) bond orders
    u: torch.Tensor          # (N,) residual unpaired valence (primal slack)
    n: torch.Tensor          # (N,) formal electron counts
    q: torch.Tensor          # (N,) formal charges n0 - n
    valence: torch.Tensor    # (N,) capacity v(n)
    lam: torch.Tensor        # (N,)
    mu: torch.Tensor         # (B,)
    nu: torch.Tensor         # (B,)
    n_iter: int
    converged: torch.Tensor  # (B,) bool
    residual: torch.Tensor   # (B,)


EPS_Q = 0.01
#: Beyond one formal electron the linear extrapolation is unphysical (O2- is unbound, O2+
#: costs a second IP); a quadratic wall of this many times the hardness takes over there.
WALL_FACTOR = 4.0


def e0_terms(q, chi, eta, eps=EPS_Q):
    """``E0(q) - E0(0)``, ``E0'``, ``E0''`` for the charged atomic reference.

    ``E0 = chi q + eta s(q) / 2 + WALL_FACTOR eta w(q)^2 / 2`` with ``s = sqrt(q^2 + eps^2) - eps``
    (a smoothed ``|q|``) and ``w = eps softplus((s - 1) / eps)`` (a smoothed ``max(|q| - 1, 0)``).
    Piecewise linear between the integer states -- the energy of a fractionally charged
    atom is the ensemble average, with the derivative discontinuity at the integer --
    through ``E0(+1) = IP`` and ``E0(-1) = -EA`` up to O(eps), and a quadratic wall past one
    electron either way. A quadratic in ``q`` would make every fractional split of a formal
    charge cheaper than the integer one and hydronium's charge would smear over its
    hydrogens; this form keeps formal charges at the integers unless the bonding pays.
    """
    r = torch.sqrt(q * q + eps * eps)
    s_abs = r - eps
    ds = q / r
    d2s = eps * eps / (r * r * r)
    z = (s_abs - 1.0) / eps
    sig = torch.sigmoid(z)
    w = eps * torch.nn.functional.softplus(z)
    dw = sig * ds
    d2w = sig * (1.0 - sig) / eps * ds * ds + sig * d2s
    k = WALL_FACTOR * eta
    e = chi * q + 0.5 * eta * s_abs + 0.5 * k * w * w
    de = chi + 0.5 * eta * ds + k * w * dw
    d2e = 0.5 * eta * d2s + k * (dw * dw + w * d2w)
    return e, de, d2e


def _count_residual(n, lam, mu_atom, chi, eta, n0, a, b, shell, temperature):
    """``h(n) = -E0'(q) + B'(n) - lam v'(n) + mu`` and ``h'(n) > 0``, ``q = n0 - n``, with the
    shell barrier ``B(n) = T [n ln n + (shell - n) ln(shell - n)]`` keeping ``0 < n < shell``."""
    q = n0 - n
    _, de0, d2e0 = e0_terms(q, chi, eta)
    dv = a + 2.0 * b * (n - n0)
    tiny = torch.finfo(n.dtype).tiny
    n_c = n.clamp(min=tiny)
    m_c = (shell - n).clamp(min=tiny)
    db = temperature * (torch.log(n_c) - torch.log(m_c))
    d2b = temperature * (1.0 / n_c + 1.0 / m_c)
    return -de0 + db - lam * dv + mu_atom, d2e0 + d2b - 2.0 * lam * b


def barrier(n, n0, shell, temperature):
    """``B(n) - B(n0)``: zero at the neutral count."""
    return temperature * (_xlogx(n) + _xlogx(shell - n) - _xlogx(n0) - _xlogx(shell - n0))


def _formal_count(lam, mu_atom, chi, eta, n0, a, b, shell, temperature, *, n_bisect: int = 60):
    """``n`` solving the stationarity condition, plus ``dn/dlam = v'/h'`` and ``dn/dmu = -1/h'``.

    ``h`` is increasing in ``n`` and runs from ``-inf`` at ``n = 0`` to ``+inf`` at
    ``n = shell`` (the barrier), so a root always exists in the shell and bisection finds it
    without a single data-dependent branch; two differentiable Newton steps from there make
    the result exact to second order in every parameter.
    """
    args = (lam, mu_atom, chi, eta, n0, a, b, shell, temperature)
    with torch.no_grad():
        lo = torch.full_like(n0, 0.0)
        hi = shell.clone()
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            h, _ = _count_residual(mid, *args)
            lo = torch.where(h < 0.0, mid, lo)
            hi = torch.where(h < 0.0, hi, mid)
        n = 0.5 * (lo + hi)
    for _ in range(2):
        h, dh = _count_residual(n, *args)
        n = n - h / dh
    h, dh = _count_residual(n, *args)
    dv = a + 2.0 * b * (n - n0)
    return n, dv / dh, -1.0 / dh


def _state(x, J, kappa, temperature, tables, chi, eta, i, j, batch_idx, n_electrons, two_s):
    """Residuals, Jacobian pieces and the dual value at multipliers ``x = (lam, mu, nu)``."""
    lam, mu, nu = x
    n0, v0, a, b, shell = tables.unbind(-1)
    mu_atom = mu[batch_idx]
    nu_atom = nu[batch_idx]
    n, dn_dlam, dn_dmu = _formal_count(lam, mu_atom, chi, eta, n0, a, b, shell, temperature)
    v, dv = capacity(n, n0, v0, a, b)
    q = n0 - n

    aa = J - lam[i] - lam[j]
    xx = _solve_pair_logit(aa, kappa, temperature)
    p = torch.sigmoid(xx)
    pc = torch.sigmoid(-xx)
    is_sat = xx > 0.0
    sat = is_sat.to(p.dtype)
    zeros = torch.zeros_like(p)
    room_pair = torch.where(is_sat, pc, zeros)
    fill_pair = torch.where(is_sat, zeros, p)
    n_sat = torch.zeros_like(v).index_add(0, i, sat).index_add(0, j, sat)
    room = (v - n_sat).index_add(0, i, room_pair).index_add(0, j, room_pair)
    log_u = (-1.0 - (lam + nu_atom) / temperature).clamp(-700.0, 60.0)
    u = torch.exp(log_u)
    fill = u.index_add(0, i, fill_pair).index_add(0, j, fill_pair)

    n_sys = int(mu.shape[0])
    R = fill - room                                                    # (N,)
    S = n.new_zeros(n_sys).index_add_(0, batch_idx, n) - n_electrons   # (B,)
    has_spin = two_s > 0.0
    M = torch.where(has_spin, u.new_zeros(n_sys).index_add_(0, batch_idx, u) - two_s, torch.zeros_like(two_s))

    dpda = _dp_da(p, pc, kappa, temperature)
    # -Hessian of the dual: symmetric, with the bordered mu / nu rows
    diag = (u / temperature + dv * dn_dlam).index_add(0, i, dpda).index_add(0, j, dpda)
    K_lam_mu = dv * dn_dmu                      # (N,) = dn/dlam-row symmetric partner
    K_lam_nu = u / temperature                  # (N,)
    K_mu_mu = -dn_dmu.new_zeros(n_sys).index_add_(0, batch_idx, dn_dmu)
    K_nu_nu = (u / temperature).new_zeros(n_sys).index_add_(0, batch_idx, u / temperature)

    ent = _xlogx(p) + _xlogx(pc)
    g_pair = -J * p + 0.5 * kappa * p * p + temperature * ent + (lam[i] + lam[j]) * p
    e0, _, _ = e0_terms(q, chi, eta)
    g_atom = -temperature * u + e0 + barrier(n, n0, shell, temperature) - lam * v + mu_atom * n
    g = g_atom.new_zeros(n_sys).index_add_(0, batch_idx, g_atom).index_add_(0, batch_idx[i], g_pair)
    g = g - mu * n_electrons - nu * two_s
    w = room - (fill - u)
    pieces = dict(p=p, u=u, w=w, n=n, q=q, v=v, dpda=dpda, diag=diag, K_lam_mu=K_lam_mu,
                  K_lam_nu=K_lam_nu, K_mu_mu=K_mu_mu, K_nu_nu=K_nu_nu, R=R, S=S, M=M, g=g,
                  has_spin=has_spin)
    return pieces


PER_PAIR = ("p", "dpda")
PER_FRAME = ("S", "M", "g", "K_mu_mu", "K_nu_nu", "has_spin")


def _direction(pc, i, j, batch_idx, local, n_systems, n_max):
    """``delta = (K + rho I)^{-1} grad`` for the bordered system, per frame."""
    dtype, device = pc["R"].dtype, pc["R"].device
    rho = float(torch.finfo(dtype).eps) ** 0.5
    m = int(n_max) + 2
    K = torch.eye(m, dtype=dtype, device=device).unsqueeze(0).repeat(int(n_systems), 1, 1)
    b_atom, l_atom = batch_idx, local
    K = K.index_put((b_atom, l_atom, l_atom), pc["diag"] + rho - 1.0, accumulate=True)
    b_pair = batch_idx[i]
    K = K.index_put((b_pair, local[i], local[j]), pc["dpda"], accumulate=True)
    K = K.index_put((b_pair, local[j], local[i]), pc["dpda"], accumulate=True)
    mu_col = torch.full_like(l_atom, n_max)
    nu_col = torch.full_like(l_atom, n_max + 1)
    K = K.index_put((b_atom, l_atom, mu_col), pc["K_lam_mu"], accumulate=True)
    K = K.index_put((b_atom, mu_col, l_atom), pc["K_lam_mu"], accumulate=True)
    K = K.index_put((b_atom, l_atom, nu_col), pc["K_lam_nu"], accumulate=True)
    K = K.index_put((b_atom, nu_col, l_atom), pc["K_lam_nu"], accumulate=True)
    frames = torch.arange(int(n_systems), device=device)
    K[frames, n_max, n_max] = pc["K_mu_mu"] + rho
    # frames without a multiplicity constraint keep nu = 0: unit row, zero residual
    K[frames, n_max + 1, n_max + 1] = torch.where(pc["has_spin"], pc["K_nu_nu"] + rho, torch.ones_like(pc["K_nu_nu"]))
    K[frames[~pc["has_spin"]], n_max + 1, :] = 0.0
    K[frames[~pc["has_spin"]], :, n_max + 1] = 0.0
    K[frames[~pc["has_spin"]], n_max + 1, n_max + 1] = 1.0
    rhs = torch.zeros(int(n_systems), m, dtype=dtype, device=device)
    rhs = rhs.index_put((b_atom, l_atom), pc["R"])
    rhs[frames, n_max] = pc["S"]
    rhs[frames, n_max + 1] = pc["M"]
    delta = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)
    return delta[b_atom, l_atom], delta[:, n_max], delta[:, n_max + 1]


def solve_electronic_state(
    J: torch.Tensor,             # (Pb,) bare pairing energies
    kappa: torch.Tensor,         # (Pb,) pairing hardness per pair
    tables: torch.Tensor,        # (N, 5) per-atom (n0, v0, a, b, shell)
    chi: torch.Tensor,           # (N,) (IP + EA)/2
    eta: torch.Tensor,           # (N,) IP - EA
    temperature,
    pair_index: torch.Tensor,    # (2, Pb)
    batch_idx: torch.Tensor,     # (N,)
    n_systems: int,
    total_charge: torch.Tensor,  # (B,)
    two_s: torch.Tensor,         # (B,)
    *,
    tol: float = 1.0e-8,
    maxiter: int = 100,
    step_max: float | None = None,
) -> ElectronicState:
    """Minimize the joint functional; differentiable to second order (module docstring)."""
    i, j = pair_index[0], pair_index[1]
    dtype, device = J.dtype, J.device
    temperature = torch.as_tensor(temperature, dtype=dtype, device=device)
    if step_max is None:
        step_max = 20.0 * float(temperature)
    local, sizes = _frame_layout(batch_idx, n_systems)
    n_max = int(sizes.max()) if sizes.numel() else 0
    tol = max(float(tol), 8.0 * float(torch.finfo(dtype).eps))
    n0 = tables[:, 0]
    n_electrons = n0.new_zeros(int(n_systems)).index_add_(0, batch_idx, n0) - total_charge.to(dtype)
    two_s = two_s.to(dtype)

    def frame_max(x, index=None):
        index = batch_idx if index is None else index
        return x.abs().new_zeros(int(n_systems)).scatter_reduce(0, index, x.abs(), "amax", include_self=True)

    def merit(pc):
        return torch.maximum(torch.maximum(frame_max(pc["R"]), pc["S"].abs()), pc["M"].abs())

    def state(x):
        return _state(x, J, kappa, temperature, tables, chi, eta, i, j, batch_idx, n_electrons, two_s)

    def scaled(dl, dm, dn):
        move = torch.maximum(torch.maximum(frame_max(dl), dm.abs()), dn.abs())
        s = step_max / move.clamp(min=step_max)
        return dl * s[batch_idx], dm * s, dn * s

    with torch.no_grad():
        v0 = tables[:, 1]
        lam = -temperature * (1.0 + v0.clamp(min=1.0e-12).log())
        excess = 0.5 * (J - kappa)
        half = torch.zeros_like(lam).scatter_reduce(0, i, excess, "amax", include_self=True)
        half = half.scatter_reduce(0, j, excess, "amax", include_self=True)
        lam = torch.maximum(lam, half)
        # mu such that an atom at rest sits at its neutral count: -chi - lam a + mu = 0
        # averaged over the frame (a crude start; Newton does the rest)
        mu = (chi + lam * tables[:, 2]).new_zeros(int(n_systems)).index_add_(0, batch_idx, chi + lam * tables[:, 2]) / sizes.clamp(min=1).to(dtype)
        nu = torch.zeros(int(n_systems), dtype=dtype, device=device)
        x = (lam, mu, nu)
        pc = state(x)
        err = merit(pc)
        g = pc["g"]
        n_iter = 0
        done = err <= tol
        for it in range(maxiter):
            if bool(done.all()):
                break
            n_iter = it + 1
            dl, dm, dn = scaled(*_direction(pc, i, j, batch_idx, local, n_systems, n_max))
            g_round = 16.0 * float(torch.finfo(dtype).eps) * (1.0 + g.abs())   # roundoff in g
            slope = (pc["R"] * dl).new_zeros(int(n_systems)).index_add_(0, batch_idx, pc["R"] * dl) + pc["S"] * dm + pc["M"] * dn
            active = ~done
            step = active.to(dtype)
            best = None
            for _ in range(30):
                x_try = (x[0] + step[batch_idx] * dl, x[1] + step * dm, x[2] + step * dn)
                pc_try = state(x_try)
                err_try = merit(pc_try)
                armijo = pc_try["g"] >= g + 1.0e-4 * step * slope - g_round
                # a residual decrease alone is not enough: on the staircase n(mu) it lets a
                # descent step through and the count multiplier cycles across the kink
                improved = ~active | armijo | ((err_try <= err) & (pc_try["g"] >= g - g_round))
                if best is None:
                    best = (x_try, pc_try, err_try, improved)
                else:
                    take = ~best[3]
                    ta = take[batch_idx]
                    tp = take[batch_idx[i]]
                    x_b, pc_b, err_b, _ = best
                    x_new = (torch.where(ta, x_try[0], x_b[0]), torch.where(take, x_try[1], x_b[1]),
                             torch.where(take, x_try[2], x_b[2]))
                    pc_new = {}
                    for k, val in pc_b.items():
                        if k in PER_PAIR:
                            pc_new[k] = torch.where(tp, pc_try[k], val)
                        elif k in PER_FRAME:
                            pc_new[k] = torch.where(take, pc_try[k], val)
                        else:
                            pc_new[k] = torch.where(ta, pc_try[k], val)
                    best = (x_new, pc_new, torch.where(take, err_try, err_b), best[3] | improved)
                if bool(best[3].all()):
                    break
                step = torch.where(improved, step, 0.5 * step)
            x, pc, err, _ = best
            g = pc["g"]
            done = done | (err <= tol)

    # two differentiable Newton steps from the converged point: exact first and second
    # derivatives, and they polish the residual from ``tol`` to roundoff
    x = tuple(t.detach() for t in x)
    for _ in range(2):
        pc = state(x)
        dl, dm, dn = scaled(*_direction(pc, i, j, batch_idx, local, n_systems, n_max))
        x = (x[0] + dl, x[1] + dm, x[2] + dn)
    pc = state(x)
    residual = merit(pc).detach()
    return ElectronicState(
        p=pc["p"], u=pc["w"].clamp(min=0.0), n=pc["n"], q=pc["q"], valence=pc["v"],
        lam=x[0], mu=x[1], nu=x[2], n_iter=n_iter, converged=residual <= tol, residual=residual,
    )


def state_energy(J, kappa, temperature, chi, eta, tables, st: ElectronicState):
    """``(per-pair, per-atom)`` terms of the minimized functional, Hartree.

    The per-atom part is ``T u ln u - T v0 ln v0 + E0(q)`` -- referenced so that a free
    neutral atom contributes exactly zero, and the charged atomic reference ``E0`` is *part
    of the functional* (it is what the formal count was minimized against).
    """
    temperature = torch.as_tensor(temperature, dtype=J.dtype, device=J.device)
    n0, v0, shell = tables[:, 0], tables[:, 1], tables[:, 4]
    e_pair = -J * st.p + 0.5 * kappa * st.p * st.p + temperature * (_xlogx(st.p) + _xlogx(1.0 - st.p))
    e_atom = (
        temperature * (_xlogx(st.u) - _xlogx(v0)) + e0_terms(st.q, chi, eta)[0]
        + barrier(st.n, n0, shell, temperature)
    )
    return e_pair, e_atom
