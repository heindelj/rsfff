"""Two-state channel CT (``CompleteModel(ct_model="twostate")``; design: project doc claude/ct_twostate.md).

Every direct channel c = i -> j is a two-state system (locally excited / charge transferred) whose CT-state population
q_c in [0, 1] is a probability, not a smeared charge:

    F_c(q) = Delta_c q - 2 W_c sqrt(q (1 - q))

with W_c fixed by the direct CT energy e_c <= 0 (the environment heads times the pair kernel) as the isolated
minimum, min F_c = e_c:

    W_c^2 = e_c (e_c - Delta_c),   q0_c = -e_c / (Delta_c - 2 e_c),   k_c = F''(q0) = (Delta_c - 2 e_c)^3 / (2 W_c^2)

(weak coupling: q0 ~ -e / Delta, the EDA-linear charge, alpha_par = R^2 / k ~ 2 |e| R^2 / Delta^2; degenerate gap,
H2+: q0 = 1/2 exactly). The gap is the CT diabatic state's energy at unit charge, additive per species,

    Delta_c = I_i - A_j + phi_perm(i) - phi_perm(j) - H_ij,

phi_perm the full permanent potential at the charge slots (partner included) and -H_ij the electron-hole attraction.
The joint energy with the induced moments z, exact for independent two-state channels (<n_c n_c'> = q_c q_c' for
c != c', <n_c^2> = q_c: a channel's own charge enters linearly, at unit charge, through Delta_c),

    E(z, q) = E_xpol(z) + sum_c [F_c(q_c) - 1/2 h_c q_c^2] + Q^T (C^T z + phi_ext) + 1/2 Q^T H Q,
    Q = G q,   h_c = (G^T H G)_cc,   H = the slots' full Coulomb matrix (shape self-energies on the diagonal),

is minimised by Newton steps that each reuse the flux CT's joint linear solve. Expanding F_c about a_c gives the
per-channel quadratic 1/2 b_c q^2 - V_c q + c0_c with

    b_c = F''(a) - h_c,   V_c = F''(a) a - F'(a),   c0_c = F(a) - F'(a) a + 1/2 F''(a) a^2,

and the channels are eliminated exactly into atomic charges (w_c = 1 / b_c, K = G diag(w) G^T, S = H - C^T W):

    (1 + K S) Q = u - K psi0,   u = G w V,   psi0 = phi_ext + C^T z_pol,   q = w (V - G^T psi),   z = z_pol - W Q,
    E - E_pol = sum_c [1/2 b~_c q_c + c0_c],   b~_c = -V_c + psi0_i - psi0_j.

One step from a = q0 (F' = 0, F'' = k) is the linear-response model (``n_outer=1``); more steps converge to the
exact minimum of E (``n_outer > 1``: the expansion point is the previous solution, damped to stay inside (0, 1), and
detached -- the energy is stationary in it at convergence). The xpol solve W = A^-1 C is shared by every step: an
outer step costs one N x N solve.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["floor_gap", "isolated", "F", "dF", "d2F", "expansion", "solve"]

#: smooth floor of the gap (Hartree): Delta_eff >= GAP_MIN (> 0 >= e, so W^2 = -e (Delta - e) >= 0 and q0 <= 1/2).
#: The diabatic gap from tabulated I, A and the permanent potential goes negative only where it is not trustworthy:
#: long-range Cl- -> Na+ channels in small clusters (e ~ 0 there) and some Cl- -> Cl- channels between sites of very
#: different Madelung potential (the unbound A(Cl-) has no environment / confinement dependence yet).
GAP_MIN = 0.03
GAP_TAU = 0.01
#: a channel with W^2 below this (Hartree^2) is off: no charge, no energy (the kernel's taper reaches exactly 0)
W2_OFF = 1e-28
Q_EPS = 1e-12


def floor_gap(Delta, gmin=GAP_MIN, tau=GAP_TAU):
    """Delta_eff = gmin + tau softplus((Delta - gmin) / tau): Delta where Delta - gmin >> tau, always > gmin."""
    return gmin + tau * nn.functional.softplus((Delta - gmin) / tau)


def isolated(e, Delta):
    """Per channel (e <= 0, Delta > e): W^2, the isolated minimum q0, k q0 (= V of the linear-response step, finite as
    e -> 0) and 1 / k (-> 0 as e -> 0)."""
    s = Delta - 2 * e                                  # > Delta - e > 0
    W2 = e * (e - Delta)
    q0 = -e / s
    kq0 = s * s / (2 * (Delta - e))
    kinv = 2 * W2 / s ** 3
    return W2, q0, kq0, kinv


def F(q, Delta, W):
    return Delta * q - 2 * W * torch.sqrt(q * (1 - q))


def dF(q, Delta, W):
    return Delta - W * (1 - 2 * q) / torch.sqrt(q * (1 - q))


def d2F(q, Delta, W):
    return W / (2 * (q * (1 - q)) ** 1.5)


def expansion(a, e, Delta, h, first):
    """The per-channel quadratic 1/2 b q^2 - V q + c0 of F_c(q) - 1/2 h_c q^2 + 1/2 rho_c (q - a)^2 about a (first:
    a = q0, closed forms), as (w = 1 / b, V, c0); channels that are off get w = V = c0 = 0.

    rho_c = h_c exp(-F''/(2 h_c)) is a proximal term centred on the expansion point: it keeps b_c = F'' - h_c + rho_c
    = h_c (x - 1 + exp(-x/2)) > 0 (x = F''/h_c), so the per-channel diagonal of the atomic elimination stays positive
    and the joint Hessian diag(b) + G^T S G is positive definite. It vanishes with its gradient at a fixed point of the
    outer steps (converged: the exact minimum) and leaves the zero-field linear-response charge at q0; it only
    stiffens the linear response of strong channels (F'' ~ h; exp(-x/2) < 1e-3 for F'' > 14 h)."""
    W2, q0, kq0, kinv = isolated(e, Delta)
    on = W2 > W2_OFF
    hp = h.clamp(min=1e-12)
    if first:
        # with k = 1 / kinv (-> infinity as the channel switches off): rho = h exp(-k / 2h), and
        # 1 / (k - h + rho) = kinv / (1 - (h - rho) kinv), finite (-> 0)
        rho = hp * torch.exp(-1.0 / (2 * hp * kinv.clamp(min=1e-300)))
        w = kinv / (1 - (h - rho) * kinv)
        V = kq0 + rho * q0
        c0 = e + 0.5 * kq0 * q0 + 0.5 * rho * q0 * q0
    else:
        W = torch.sqrt(torch.where(on, W2, torch.ones_like(W2)))
        a = a.clamp(Q_EPS, 1 - Q_EPS)
        f2, f1, f0 = d2F(a, Delta, W), dF(a, Delta, W), F(a, Delta, W)
        rho = hp * torch.exp(-f2 / (2 * hp))
        w = 1.0 / (f2 - h + rho)
        V = (f2 + rho) * a - f1
        c0 = f0 - f1 * a + 0.5 * (f2 + rho) * a * a
    z = torch.zeros_like(e)
    return torch.where(on, w, z), torch.where(on, V, z), torch.where(on, c0, z)


def _atomic(Ha, S, psi0, ch, w, V):
    """One joint linear solve with the channels eliminated (module docstring). Returns Q (B, N), q (C,), psi (B, N)."""
    cf, li, lj = ch
    B, N = Ha.shape[0], Ha.shape[1]
    Kl = Ha.new_zeros(B, N, N)
    Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
    Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
    wV = w * V
    u = Ha.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
    M = torch.eye(N, dtype=Ha.dtype, device=Ha.device) + Kl @ S
    Q = torch.linalg.solve(M, (u - (Kl @ psi0[..., None])[..., 0])[..., None])[..., 0]
    psi = psi0 + (S @ Q[..., None])[..., 0]
    q = w * (V - (psi[cf, li] - psi[cf, lj]))
    return Q, q, psi


def solve(Ca, Ha, W, z_pol, phi_slots, ch, e, Delta, n_outer=1, tol=1e-10, info=None):
    """The joint minimum. Ca (B, ND, N): the slot charges' coupling to the site variables, Ha (B, N, N) the slots'
    full Coulomb (self-energies on the diagonal), W = A^-1 Ca (B, ND, N), z_pol (B, ND) the CT-free xpol solution,
    phi_slots (B, N) the external potential on the slots, ch = (frame, donor slot, acceptor slot) of the C channels,
    e (C,) <= 0 their direct energies and Delta (C,) their (floored) gaps.

    Returns (E_CT per frame (B,), per channel energy (C,), populations q (C,), joint z (B, ND), atomic charges
    Q (B, N)); ``info`` (dict) gets the number of outer steps and the last step size."""
    cf, li, lj = ch
    h = Ha[cf, li, li] + Ha[cf, lj, lj] - Ha[cf, li, lj] - Ha[cf, lj, li]
    S = Ha - Ca.transpose(1, 2) @ W
    psi0 = phi_slots + torch.einsum("bnk,bn->bk", Ca, z_pol)
    a = None
    step = torch.zeros((), dtype=e.dtype)
    it = 0
    for it in range(max(int(n_outer), 1)):
        w, V, c0 = expansion(a, e, Delta, h, first=(it == 0))
        Q, q, _ = _atomic(Ha, S, psi0, ch, w, V)
        if it == n_outer - 1:
            break
        a0 = isolated(e, Delta)[1] if a is None else a
        a0, d = a0.detach(), (q - a0).detach()
        step = d.abs().max() if d.numel() else torch.zeros((), dtype=e.dtype)
        if float(step) < tol:
            break
        # damp each channel's step to stay inside (0, 1)
        room = torch.where(d < 0, (a0 - Q_EPS) / (-d).clamp(min=1e-300), (1 - Q_EPS - a0) / d.clamp(min=1e-300))
        a = a0 + torch.clamp(0.9 * room, max=1.0) * d
    if info is not None:
        info["outer"] = it + 1
        info["step"] = float(step)
    bt_ = -V + psi0[cf, li] - psi0[cf, lj]
    e_ch = 0.5 * bt_ * q + c0
    z = z_pol - (W @ Q[..., None])[..., 0]
    return Ha.new_zeros(Ha.shape[0]).index_add(0, cf, e_ch), e_ch, q, z, Q
