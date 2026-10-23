"""Electrostatics and induction of monatomic ions with shaped densities (ion ct ff plan, R2). Torch, differentiable
(forces by autograd, double backward through the dense solve).

Densities (``IonTable``):

* permanent: point nucleus Z plus Slater 1s shells (n_t electrons, exponent b_t); frozen, pairwise;
* induced: a dipole carried by -mu . grad g_d and a quadrupole by the second derivatives of g_q, each parent
  g = sum_k w_k N_k r^k e^{-b r} (``rsfff.qcgen.pol_fit``'s shaped response), with the free-ion polarizabilities
  scaled per atom by optional confinement factors;
* frontier: the donor / acceptor densities of the CT channels (normalized r^k e^{-br}).

Every pair interaction is a damped multipole tensor (``rsfff.ff.multipole.damped_interaction_tensor``) whose damping
factors are the overlap complements of the two densities involved: one-centre (``shaped_one_center_damp``) when one
side is a point nucleus, two-centre (``shaped_two_center_damp``) when both are spread.

    all_pairs(batch_idx)                     unordered pairs within each frame
    permanent_pair_energy(...)               frozen electrostatics between ions (P,)
    induction(...)                           mutual induced dipoles + quadrupoles: one dense linear solve per batch;
                                             energy per frame, moments, the drive
    frontier_potential_matrix(...)           phibar[i, j]: potential of ion j (permanent + induced) averaged over
                                             the frontier density of i in a role -- the CT gaps' input
    external_potential(...)                  -F . x for a uniform field per frame

The quadrupole convention is Buckingham's, Theta = sum_k theta_k B_k (spherical theta, pol_fit), U = (1/3) Theta :
grad grad phi; a site's quadrupole polarizability C gives theta = C g, g_k = (1/3) B_k : G.

Units: positions in Angstrom at the interface, atomic units inside and out (Hartree, bohr, e).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..multipole import (
    build_polytensor,
    damped_interaction_tensor,
    shaped_one_center_damp,
    shaped_one_center_multi,
    shaped_two_center_damp,
    slater_one_center_damp,
    slater_two_center_damp,
    slater_two_center_damp_unequal,
    spherical_to_cartesian_quadrupole,
)
from ..units import BOHR_ANG
from .params import IonTable

__all__ = ["all_pairs", "permanent_pair_energy", "induction", "induction_operator", "solve_induction",
           "InductionResult", "frontier_operator", "frontier_potential_matrix", "external_potential", "x_to_poly",
           "pauli_kernel", "breathing_kernel"]

D_SITE = 8          # mu (3) + theta (5)


def all_pairs(batch_idx: torch.Tensor) -> torch.Tensor:
    """Unordered pairs (i < j) within each frame: (2, P)."""
    n = batch_idx.shape[0]
    i, j = torch.triu_indices(n, n, offset=1, device=batch_idx.device)
    keep = batch_idx[i] == batch_idx[j]
    return torch.stack([i[keep], j[keep]])


def _onehot0(n: int, like: torch.Tensor) -> torch.Tensor:
    w = like.new_zeros(n, 4)
    w[:, 0] = 1.0
    return w


def _poly_map(dtype, device) -> torch.Tensor:
    """(10, 8): site variables [mu, theta] -> the rank-2 polytensor [q, mu, Theta unique (weighted)]."""
    eye = torch.eye(D_SITE, dtype=dtype, device=device)
    mu, th = eye[:, :3], eye[:, 3:]
    Th = spherical_to_cartesian_quadrupole(th)
    return build_polytensor(eye.new_zeros(D_SITE), mu, Th, max_rank=2).T


def x_to_poly(x: torch.Tensor) -> torch.Tensor:
    """(N, 8) [mu, theta] -> (N, 10) polytensors."""
    return x @ _poly_map(x.dtype, x.device).T


def _tensor(dr, lam):
    """Damped rank-2 tensor with factors lam = 1 - f (stacked (5, P))."""
    return damped_interaction_tensor(dr, lam, max_rank=2)


def _r(dr):
    return dr.norm(dim=-1)


# ---- permanent ---------------------------------------------------------------------------------------

def permanent_pair_energy(pos_ang: torch.Tensor, species: torch.Tensor, pairs: torch.Tensor, tab: IonTable) -> torch.Tensor:
    """Frozen electrostatics of each pair (Hartree): nuclei and shells, penetration included."""
    i, j = pairs
    X = pos_ang / BOHR_ANG
    r = _r(X[j] - X[i])
    si, sj = species[i], species[j]
    Zi, Zj = tab.Z[si], tab.Z[sj]
    e = Zi * Zj / r
    M = tab.shell_n.shape[1]
    for t in range(M):
        nj, bj = tab.shell_n[sj, t], tab.shell_b[sj, t]
        ni, bi = tab.shell_n[si, t], tab.shell_b[si, t]
        e = e - Zi * nj * (1 - slater_one_center_damp(bj * r, 0)[0]) / r
        e = e - Zj * ni * (1 - slater_one_center_damp(bi * r, 0)[0]) / r
        for u in range(M):
            nu, bu = tab.shell_n[sj, u], tab.shell_b[sj, u]
            e = e + ni * nu * (1 - slater_two_center_damp_unequal(bi, bu, r, 0)[0]) / r
    return e


def external_potential(pos_ang, batch_idx, field):
    """phi_ext = -F . x (bohr) at each atom for a uniform field per frame (B, 3), or zeros."""
    if field is None:
        return pos_ang.new_zeros(pos_ang.shape[0])
    return -(field[batch_idx] * (pos_ang / BOHR_ANG)).sum(-1)


# ---- induction -------------------------------------------------------------------------------------

def _drive_from_permanent(X, species, dpairs, tab, b_par, w_par):
    """Coupling of the induced parent (b_par, w_par per atom) of site i to the permanent components of site j,
    for directed pairs (i <- j): (P, 10) row 0 of T, i.e. d E / d m_i per unit polytensor."""
    i, j = dpairs
    dr = X[j] - X[i]
    r = _r(dr)
    si, sj = species[i], species[j]
    bp, wp = b_par[i], w_par[i]
    lam = 1 - shaped_one_center_multi(bp, wp, r, 2)
    out = tab.Z[sj, None] * _tensor(dr, lam)[:, 0, :]
    one = _onehot0(r.shape[0], r)
    for t in range(tab.shell_n.shape[1]):
        bt = tab.shell_b[sj, t]
        lam = 1 - shaped_two_center_damp(bp, bt, wp, one, r, 2)
        out = out - tab.shell_n[sj, t, None] * _tensor(dr, lam)[:, 0, :]
    return out


def pseudo_drive_basis(pos_ang, species, batch_idx, tab: IonTable, cutoff: float = 6.0, shells: bool = False):
    """The induction drive on ion i of a unit Pauli pseudo-potential on its neighbour j, for directed pairs (i <- j)
    within ``cutoff`` (A): (dpairs (2, P), r (P,) A, basis (P, 8)); with ``shells`` also the per-shell basis
    (P, T, 8), one column per permanent shell of j (zero where j has no shell t).

    The pseudo-potential is the penetration potential of j's outermost permanent shell, phi_P = -A [1/r - (1 -
    f(b r))/r] = -A f(b r)/r (a +A smeared charge on top of a -A point charge: no net charge, exponentially short
    ranged, so a uniform field or a distant ion never sees it). For A > 0 it repels electrons near j -- the exclusion
    that keeps i's response density from flowing toward its neighbour. Its field and field gradient are averaged over
    i's induced-dipole and -quadrupole densities exactly like the permanent drive, so ``A * basis`` adds to L."""
    X = pos_ang / BOHR_ANG
    pairs = all_pairs(batch_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)
    rr = _r(X[dp[1]] - X[dp[0]])
    dp = dp[:, rr * BOHR_ANG < cutoff]
    i, j = dp
    dr = X[j] - X[i]
    r = _r(dr)
    s = species
    M = tab.shell_n.shape[1]
    # outermost occupied shell of each species
    last = torch.tensor([max(t for t in range(M) if tab.shell_n[k, t] > 0) for k in range(len(tab.symbols))])
    one = _onehot0(r.shape[0], r)
    P = _poly_map(X.dtype, X.device)

    def basis_for(bP):
        out = []
        for bpar, wpar in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
            bp, wp = bpar[s[i]], wpar[s[i]]
            t_pt = _tensor(dr, 1 - shaped_one_center_multi(bp, wp, r, 2))[:, 0, :]
            t_sh = _tensor(dr, 1 - shaped_two_center_damp(bp, bP, wp, one, r, 2))[:, 0, :]
            out.append((t_sh - t_pt) @ P)
        return torch.cat([out[0][:, :3], out[1][:, 3:]], dim=1)

    basis = basis_for(tab.shell_b[s[j], last[s[j]]])
    if not shells:
        return dp, r * BOHR_ANG, basis
    cols = [basis_for(tab.shell_b[s[j], t]) * (tab.shell_n[s[j], t] > 0).to(X.dtype)[:, None] for t in range(M)]
    return dp, r * BOHR_ANG, basis, torch.stack(cols, 1)


def pauli_kernel(pos_ang, batch_idx, qP, bP, kappa, stiff):
    """CMM's Slater Pauli kernel acting on the permanent *and* induced moments of the ions, to second order in the
    induced moments x (N, 8) [mu, theta]:

        E_P(x) = sum_{i<j} p_j^T T_P(b_ij; r_ij) p_i  +  1/2 sum_i x_i^T H_i x_i
        p_i    = [qP_i, -kappa_i * x_i]        (polytensor; the Pauli density moves with the electrons, against mu)
        T_P    = the damped interaction tensor with the overlap factors f_n(b_ij r) themselves (CMM's short-range
                 kernel: exponentially decaying, no long-range tail), b_ij = sqrt(b_i b_j)
        H_i    = sum_j qP_j f_1(b_ij r)/r [ s0d I_3 + s2d (e e^T - I/3) (+) s0q I_5 ]

    The bilinear part gives the frozen Pauli energy (qP qP, the x = 0 term), a linear drive g = dE/dx (exclusion /
    relief: induced electron density pushed out of the neighbours' Pauli density) and an inter-site coupling of the
    induced moments. H is the second-order density of i's deformation (|d psi|^2: a monopole and a quadrupole
    quadratic in mu) seen by j's Pauli monopole through the same radial kernel: the Pauli stiffening of the response,
    stiffer along the bond for s2d > 0.

    qP, bP (N,) per ion; kappa (N, 8) >= 0 per slot; stiff (N, 3) [s0d, s2d, s0q]. Returns e0 (P,) frozen pair
    energies, g (N, 8), Bij (P, 8, 8) [j-slot, i-slot] (the pair term x_j^T Bij x_i) and Hon (N, 8, 8), with the
    pairs (2, P). Atomic units out."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    P = _poly_map(X.dtype, X.device)                        # (10, 8)
    pairs = all_pairs(batch_idx)
    i, j = pairs
    dr = X[j] - X[i]
    r = _r(dr)
    f = slater_two_center_damp(torch.sqrt(bP[i] * bP[j]) * r, 2)     # (5, P)
    T = _tensor(dr, f)                                      # (P, 10, 10): [j-component, i-component]
    e0 = qP[i] * qP[j] * T[:, 0, 0]
    gi = -kappa[i] * qP[j, None] * (T[:, 0, :] @ P)
    gj = -kappa[j] * qP[i, None] * (T[:, :, 0] @ P)
    g = X.new_zeros(N, D_SITE).index_add(0, i, gi).index_add(0, j, gj)
    Bij = kappa[j][:, :, None] * (P.T @ T @ P) * kappa[i][:, None, :]
    F = T[:, 0, 0]
    e = dr / r[:, None]
    I3 = torch.eye(3, dtype=X.dtype, device=X.device)
    I5 = torch.eye(5, dtype=X.dtype, device=X.device)
    ee = e[:, :, None] * e[:, None, :] - I3 / 3

    def onsite(a, other):
        w = qP[other] * F
        Hd = w[:, None, None] * (stiff[a, 0, None, None] * I3 + stiff[a, 1, None, None] * ee)
        Hq = (w * stiff[a, 2])[:, None, None] * I5
        return torch.cat([torch.cat([Hd, X.new_zeros(Hd.shape[0], 3, 5)], 2),
                          torch.cat([X.new_zeros(Hd.shape[0], 5, 3), Hq], 2)], 1)

    Hon = X.new_zeros(N, D_SITE, D_SITE).index_add(0, i, onsite(i, j)).index_add(0, j, onsite(j, i))
    return pairs, e0, g, Bij, Hon


def _scaled_permanent_pair(r, si, sj, tab: IonTable, a, c):
    """Frozen electrostatics of pairs (P,) with every shell exponent of i scaled by e^a and of j by e^c."""
    ea, ec = torch.exp(a), torch.exp(c)
    Zi, Zj = tab.Z[si], tab.Z[sj]
    e = Zi * Zj / r
    M = tab.shell_n.shape[1]
    for t in range(M):
        nj, bj = tab.shell_n[sj, t], tab.shell_b[sj, t] * ec
        ni, bi = tab.shell_n[si, t], tab.shell_b[si, t] * ea
        e = e - Zi * nj * (1 - slater_one_center_damp(bj * r, 0)[0]) / r
        e = e - Zj * ni * (1 - slater_one_center_damp(bi * r, 0)[0]) / r
        for u in range(M):
            nu, bu = tab.shell_n[sj, u], tab.shell_b[sj, u] * ec
            e = e + ni * nu * (1 - slater_two_center_damp_unequal(bi, bu, r, 0)[0]) / r
    return e


def breathing_kernel(pos_ang, batch_idx, species, tab: IonTable, qP, bP, kappa_br):
    """Second-order expansion of the pair energies in a breathing (dilation) amplitude a_i of each ion's density,
    b -> b e^{a}: the permanent shells (frozen electrostatics, penetration) and the Pauli density (exponent
    b_P e^{kappa_br a}). A dilation keeps the charge and has no dipole, so it couples only through overlap
    (exponentially short-ranged) and never to a uniform field.

        E(a) = sum_{i<j} e_ij(a_i, a_j) ~ E(0) + D.a + 1/2 a^T H a

    Returns pairs (2, P), D (N,), the on-site curvature Hd (N,) (summed over partners) and the pair curvature
    Ho (P,) (d2 e_ij / da_i da_j). Atomic units; differentiable in qP, bP, kappa_br and the positions."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    pairs = all_pairs(batch_idx)
    i, j = pairs
    r = _r(X[j] - X[i])
    si, sj = species[i], species[j]
    with torch.enable_grad():
        a = r.new_zeros(r.shape).requires_grad_(True)
        c = r.new_zeros(r.shape).requires_grad_(True)
        u = torch.sqrt(bP[i] * bP[j]) * r * torch.exp(0.5 * (kappa_br[i] * a + kappa_br[j] * c))
        e = _scaled_permanent_pair(r, si, sj, tab, a, c) + qP[i] * qP[j] * slater_two_center_damp(u, 0)[0] / r
        ga, gc = torch.autograd.grad(e.sum(), (a, c), create_graph=True)
        haa, hac = torch.autograd.grad(ga.sum(), (a, c), create_graph=True)
        hcc = torch.autograd.grad(gc.sum(), c, create_graph=True)[0]
    D = X.new_zeros(N).index_add(0, i, ga).index_add(0, j, gc)
    Hd = X.new_zeros(N).index_add(0, i, haa).index_add(0, j, hcc)
    return pairs, D, Hd, hac


@dataclass
class InductionResult:
    energy: torch.Tensor         # (B,) Hartree, the minimized functional -1/2 L^T A^-1 L
    x: torch.Tensor              # (N, 8) [mu, theta]
    mu: torch.Tensor             # (N, 3)
    theta: torch.Tensor          # (N, 5)
    drive: torch.Tensor          # (N, 8) L


def induction_operator(pos_ang, species, batch_idx, tab: IonTable, *, field=None):
    """The position-dependent parts of the induction functional: the drive L (N, 8) and the off-diagonal coupling
    A_off (8N, 8N) between induced sites (the diagonal, alpha^-1 and C^-1, is added by :func:`solve_induction`).
    Everything here depends on the geometry and the fixed shapes only, so it can be cached per frame."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    dt, dev = X.dtype, X.device
    P = _poly_map(dt, dev)                                 # (10, 8)
    pairs = all_pairs(batch_idx)
    dpairs = torch.cat([pairs, pairs.flip(0)], dim=1)      # directed (i <- j)
    s = species
    bd, wd = tab.b_dip[s], tab.w_dip[s]
    bq, wq = tab.b_quad[s], tab.w_quad[s]

    # drive: dipole slots from the dipole parent, quadrupole slots from the quadrupole parent
    Ld = _drive_from_permanent(X, s, dpairs, tab, bd, wd) @ P   # (P, 8)
    Lq = _drive_from_permanent(X, s, dpairs, tab, bq, wq) @ P
    Lpair = torch.cat([Ld[:, :3], Lq[:, 3:]], dim=1)
    L = X.new_zeros(N, D_SITE).index_add(0, dpairs[0], Lpair)
    if field is not None:
        L = L - torch.cat([field[batch_idx], X.new_zeros(N, 5)], dim=1)

    # induced-induced coupling, per sector pair (damping between the two parents involved)
    i, j = pairs
    dr = X[j] - X[i]
    r = _r(dr)
    blocks = {}
    for a, (bi_, wi_) in (("d", (bd, wd)), ("q", (bq, wq))):
        for c, (bj_, wj_) in (("d", (bd, wd)), ("q", (bq, wq))):
            lam = 1 - shaped_two_center_damp(bi_[i], bj_[j], wi_[i], wj_[j], r, 2)
            blocks[a + c] = P.T @ _tensor(dr, lam) @ P            # (P, 8, 8): [j-slot, i-slot]
    Bij = X.new_zeros(r.shape[0], D_SITE, D_SITE)
    sl = {"d": slice(0, 3), "q": slice(3, 8)}
    for a in "dq":            # i's sector (columns)
        for c in "dq":        # j's sector (rows)
            Bij[:, sl[c], sl[a]] = blocks[a + c][:, sl[c], sl[a]]
    ar = torch.arange(D_SITE, device=dev)
    A = X.new_zeros(N, D_SITE, N, D_SITE)
    A = A.index_put((j[:, None, None], ar[None, :, None], i[:, None, None], ar[None, None, :]), Bij, accumulate=True)
    A = A.index_put((i[:, None, None], ar[None, :, None], j[:, None, None], ar[None, None, :]),
                    Bij.transpose(1, 2), accumulate=True)
    return L, A.reshape(N * D_SITE, N * D_SITE)


def pauli_induced_operator(pos_ang, species, batch_idx, tab: IonTable):
    """The penetration (overlap-complement) part of the induced-induced coupling, (8N, 8N): the same blocks as
    :func:`induction_operator`'s A_off with the damping complements f_n in place of 1 - f_n -- the kernel of the
    Pauli energy between two induced densities on the parents, before the per-atom coupling constants."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    dt, dev = X.dtype, X.device
    P = _poly_map(dt, dev)
    pairs = all_pairs(batch_idx)
    s = species
    bd, wd = tab.b_dip[s], tab.w_dip[s]
    bq, wq = tab.b_quad[s], tab.w_quad[s]
    i, j = pairs
    dr = X[j] - X[i]
    r = _r(dr)
    blocks = {}
    for a, (bi_, wi_) in (("d", (bd, wd)), ("q", (bq, wq))):
        for c, (bj_, wj_) in (("d", (bd, wd)), ("q", (bq, wq))):
            f = shaped_two_center_damp(bi_[i], bj_[j], wi_[i], wj_[j], r, 2)
            blocks[a + c] = P.T @ _tensor(dr, f) @ P
    Bij = X.new_zeros(r.shape[0], D_SITE, D_SITE)
    sl = {"d": slice(0, 3), "q": slice(3, 8)}
    for a in "dq":
        for c in "dq":
            Bij[:, sl[c], sl[a]] = blocks[a + c][:, sl[c], sl[a]]
    ar = torch.arange(D_SITE, device=dev)
    A = X.new_zeros(N, D_SITE, N, D_SITE)
    A = A.index_put((j[:, None, None], ar[None, :, None], i[:, None, None], ar[None, None, :]), Bij, accumulate=True)
    A = A.index_put((i[:, None, None], ar[None, :, None], j[:, None, None], ar[None, None, :]),
                    Bij.transpose(1, 2), accumulate=True)
    return A.reshape(N * D_SITE, N * D_SITE)


def induction_matrix(A_off, alpha, C):
    """The full induction Hessian A = A_off + diag(alpha^-1 (x3), C^-1 (x5)) for a padded batch (B, 8N, 8N)."""
    B, N = alpha.shape
    diag = torch.cat([(1.0 / alpha)[..., None].expand(B, N, 3), (1.0 / C)[..., None].expand(B, N, 5)], dim=-1)
    return A_off + torch.diag_embed(diag.reshape(B, -1))


def solve_induction(L, A_off, alpha, C, quadrupoles=True):
    """x = -(A_off + diag(alpha^-1, C^-1))^-1 L for one frame (L (N, 8)) or a padded batch (L (B, N, 8),
    A_off (B, 8N, 8N), alpha/C (B, N); padded sites need alpha = C = 1 and L = 0). Returns x like L."""
    batched = L.dim() == 3
    if not batched:
        L, A_off, alpha, C = L[None], A_off[None], alpha[None], C[None]
    B, N, _ = L.shape
    cinv = (1.0 / C) if quadrupoles else torch.full_like(C, 1e12)
    diag = torch.cat([(1.0 / alpha)[..., None].expand(B, N, 3), cinv[..., None].expand(B, N, 5)], dim=-1)
    A = A_off + torch.diag_embed(diag.reshape(B, -1))
    if quadrupoles:
        x = -torch.linalg.solve(A, L.reshape(B, -1, 1)).reshape(B, N, D_SITE)
    else:
        keep = torch.zeros(N, D_SITE, dtype=torch.bool, device=L.device)
        keep[:, :3] = True
        keep = keep.reshape(-1).nonzero().squeeze(1)
        xs = -torch.linalg.solve(A[:, keep][:, :, keep], L.reshape(B, -1)[:, keep, None])[..., 0]
        x = L.new_zeros(B, N * D_SITE).index_copy(1, keep, xs).reshape(B, N, D_SITE)
    return x if batched else x[0]


def induction(pos_ang, species, batch_idx, tab: IonTable, *, field=None, alpha_scale=None, c_scale=None,
              quadrupoles=True, n_systems=None, drive_add=None) -> InductionResult:
    """Mutual induced dipoles (+ quadrupoles) of the ions: E(x) = 1/2 sum x_i^T K_i x_i + sum L_i . x_i +
    sum_{i<j} x_j^T B_ij x_i, minimized by one dense solve over the batch (frames are independent blocks).
    ``drive_add`` (N, 8) is added to the drive L.
    ``alpha_scale``/``c_scale`` (N,) multiply the free-ion values (confinement); ``field`` (B, 3) a uniform field."""
    B = int(batch_idx.max()) + 1 if n_systems is None else int(n_systems)
    L, A_off = induction_operator(pos_ang, species, batch_idx, tab, field=field)
    if drive_add is not None:                            # e.g. the Pauli pseudo-potential drive (N, 8)
        L = L + drive_add
    al = tab.alpha[species] * (1.0 if alpha_scale is None else alpha_scale)
    cq = tab.C[species] * (1.0 if c_scale is None else c_scale)
    x = solve_induction(L, A_off, al, cq, quadrupoles)
    e_atom = 0.5 * (L * x).sum(-1)                       # -1/2 L^T A^-1 L, split by atom
    energy = L.new_zeros(B).index_add(0, batch_idx, e_atom)
    return InductionResult(energy=energy, x=x, mu=x[:, :3], theta=x[:, 3:], drive=L)


# ---- frontier-averaged potentials -------------------------------------------------------------------

def frontier_operator(pos_ang, species, batch_idx, tab: IonTable, role: str):
    """phibar[i, j] = V[i, j] + K[i, j] . x_j: the permanent part V (N, N) and the induced kernel K (N, N, 8) of the
    potential of ion j averaged over the frontier density of ion i in ``role``. Geometry and fixed shapes only."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    pairs = all_pairs(batch_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)       # (i, j): frontier at i, source j
    i, j = dp
    dr = X[j] - X[i]
    r = _r(dr)
    s = species
    bf, wf = tab.front_b[role][s[i]], tab.front_w[role][s[i]]
    # permanent: nucleus (one-centre, the frontier density spread) and shells (two-centre)
    v = tab.Z[s[j]] * (1 - shaped_one_center_damp(bf * r, wf, 0)[0]) / r
    one = _onehot0(r.shape[0], r)
    for t in range(tab.shell_n.shape[1]):
        bt = tab.shell_b[s[j], t]
        v = v - tab.shell_n[s[j], t] * (1 - shaped_two_center_damp(bf, bt, wf, one, r, 0)[0]) / r
    P = _poly_map(X.dtype, X.device)
    k = []
    for bp, wp in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
        lam = 1 - shaped_two_center_damp(bf, bp[s[j]], wf, wp[s[j]], r, 2)
        k.append(_tensor(dr, lam)[:, :, 0] @ P)            # (P, 8): contraction m_j^T T m_i, m_i = unit charge
    kv = torch.cat([k[0][:, :3], k[1][:, 3:]], dim=1)
    V = X.new_zeros(N, N).index_put((i, j), v)
    K = X.new_zeros(N, N, D_SITE).index_put((i, j), kv)
    return V, K


def _frontier_pair_terms(dr, si, sj, tab: IonTable, role: str):
    """Per directed pair (frontier at i, source j; dr = X_j - X_i, bohr): the potential v (P,) of j's permanent
    density averaged over i's frontier density in ``role``, and the kernel kv (P, 8) of j's induced moments."""
    r = _r(dr)
    bf, wf = tab.front_b[role][si], tab.front_w[role][si]
    v = tab.Z[sj] * (1 - shaped_one_center_damp(bf * r, wf, 0)[0]) / r
    one = _onehot0(r.shape[0], r)
    for t in range(tab.shell_n.shape[1]):
        v = v - tab.shell_n[sj, t] * (1 - shaped_two_center_damp(bf, tab.shell_b[sj, t], wf, one, r, 0)[0]) / r
    P = _poly_map(dr.dtype, dr.device)
    k = []
    for bp, wp in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
        lam = 1 - shaped_two_center_damp(bf, bp[sj], wf, wp[sj], r, 2)
        k.append(_tensor(dr, lam)[:, :, 0] @ P)
    return v, torch.cat([k[0][:, :3], k[1][:, 3:]], dim=1)


def frontier_field_operator(pos_ang, species, batch_idx, tab: IonTable, role: str):
    """The field (au) of ion j averaged over the frontier density of ion i in ``role``: E[i, j] = EV[i, j] +
    EK[i, j] @ x_j with the permanent part EV (N, N, 3) and the induced kernel EK (N, N, 3, 8), i.e. -grad_{R_i} of
    frontier_operator's potentials. Closed form: the dipole row of the damped interaction tensor between i's frontier
    shape and j's permanent density / induced parents (the frontier shape playing the role of an induced parent)."""
    X = pos_ang / BOHR_ANG
    N = X.shape[0]
    pairs = all_pairs(batch_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)
    i, j = dp
    s = species
    bf, wf = tab.front_b[role][s], tab.front_w[role][s]
    ev = -_drive_from_permanent(X, s, dp, tab, bf, wf)[:, 1:4]
    dr = X[j] - X[i]
    r = _r(dr)
    P = _poly_map(X.dtype, X.device)
    ks = []
    for bp, wp in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
        lam = 1 - shaped_two_center_damp(bf[i], bp[s[j]], wf[i], wp[s[j]], r, 2)
        ks.append(-(_tensor(dr, lam)[:, :, 1:4].transpose(1, 2) @ P))      # (P, 3, 8)
    ek = torch.cat([ks[0][:, :, :3], ks[1][:, :, 3:]], dim=2)
    EV = X.new_zeros(N, N, 3).index_put((i, j), ev)
    EK = X.new_zeros(N, N, 3, D_SITE).index_put((i, j), ek)
    return EV, EK


def frontier_hole_field(pos_ang, species, tab: IonTable, idx):
    """For channels idx (2, C) (donor i -> acceptor j): the field (C, 3) at the acceptor, averaged over its acceptor
    frontier density, of a unit positive charge spread as the donor's frontier density -- the hole a CT leaves.
    By the symmetry of the two-centre overlap it is also the field at the donor, averaged over its donor frontier,
    of the transferred electron (a unit negative charge spread as the acceptor frontier)."""
    X = pos_ang / BOHR_ANG
    i, j = idx
    if i.numel() == 0:
        return X.new_zeros(0, 3)
    with torch.enable_grad():
        dr = (X[i] - X[j]).detach().requires_grad_(True)       # source i, field point j
        r = _r(dr)
        ba, wa = tab.front_b["acceptor"][species[j]], tab.front_w["acceptor"][species[j]]
        bd, wd = tab.front_b["donor"][species[i]], tab.front_w["donor"][species[i]]
        u = (1 - shaped_two_center_damp(ba, bd, wa, wd, r, 0)[0]) / r
        e = torch.autograd.grad(u.sum(), dr)[0]
    return e.detach()


#: k-grid (1/bohr) of the overlap quadrature: |d ln O| < 1e-3 for every frontier pair up to 8 A (1s-1s, the slowest
#: k-decay, sets kmax; R up to 15 bohr sets dk)
_OVL_K = None


def _ovl_grid(dtype, device, kmax: float = 40.0, n: int = 1000):
    global _OVL_K
    if _OVL_K is None or _OVL_K[0].dtype != dtype or _OVL_K[0].device != device:
        k = torch.linspace(kmax / n, kmax, n, dtype=dtype, device=device)
        w = torch.full_like(k, kmax / n)
        w[-1] *= 0.5
        _OVL_K = (k, w)
    return _OVL_K


def shaped_density_ft(b, w, k):
    """Fourier transform rho(k) (C, K) of the normalized shaped density sum_n w_n N_n r^n e^{-b r} (n = 0..3, each
    component normalized), b (C,), w (C, 4): rho_n(k) = b^{n+3} sin((n+2) theta) / ((n+2) k (b^2 + k^2)^{(n+2)/2}),
    theta = atan2(k, b); rho(0) = 1."""
    if b.dim() == 2:                                # multi-term shape: sum of its terms
        out = 0.0
        for t in range(b.shape[1]):
            if t == 0 or bool((w[:, t] != 0).any()):
                out = out + shaped_density_ft(b[:, t], w[:, t], k)
        return out
    bb, kk = b[:, None], k[None, :]
    th = torch.atan2(kk, bb)
    rr = torch.sqrt(bb * bb + kk * kk)
    out = 0.0
    for n in range(w.shape[1]):
        if not bool((w[:, n] != 0).any()):          # shape powers no density uses (one-hot frontier shapes)
            continue
        m = n + 2
        out = out + w[:, n:n + 1] * bb ** (n + 3) * torch.sin(m * th) / (m * kk * rr ** m)
    return out


def shaped_overlap(bD, wD, bA, wA, R):
    """Overlap integral int rho_D(r) rho_A(r - R) d^3r (bohr^-3) of two normalized shaped densities at distance R
    (bohr), (C,): (1 / 2 pi^2) int rho_D(k) rho_A(k) sin(kR)/(kR) k^2 dk by quadrature on a fixed k grid.
    Differentiable in the exponents (the coupling's environment response goes through b)."""
    k, wk = _ovl_grid(R.dtype, R.device)
    fD = shaped_density_ft(bD, wD, k)
    fA = shaped_density_ft(bA, wA, k)
    kr = k[None, :] * R[:, None]
    sinc = torch.where(kr > 1e-8, torch.sin(kr) / kr.clamp(min=1e-30), torch.ones_like(kr))
    return (fD * fA * sinc * (k * k)[None, :] * wk[None, :]).sum(1) / (2 * math.pi ** 2)


#: exponent (1/bohr) of the 1s probe density over which the barrier potential is averaged: the tunnelling electron
#: samples a region of ~1 bohr, and a point probe would diverge when an ion sits at a bond midpoint (linear chains)
PROBE_B = 2.0


def point_potential_operator(pos_ang, species, tab: IonTable, points_ang, exclude, probe_b: float = PROBE_B):
    """Potential averaged over a normalized 1s probe density (exponent probe_b) centred at each point (M, 3) A, from
    every ion of one frame: permanent part V (M, N) (nucleus + Slater shells, penetration included) and the induced
    kernel K (M, N, 8) (each ion's induced dipole/quadrupole, damped between its induced-density shape and the
    probe). Ion exclude[m] (or -1) is left out of row m. Finite when an ion sits on the point. Hartree / e, bohr."""
    X = pos_ang / BOHR_ANG
    Pm = points_ang / BOHR_ANG
    M, N = Pm.shape[0], X.shape[0]
    mm, kk = torch.meshgrid(torch.arange(M), torch.arange(N), indexing="ij")
    mm, kk = mm.reshape(-1), kk.reshape(-1)
    dr = X[kk] - Pm[mm]                                       # source - point
    r = torch.sqrt((dr * dr).sum(-1) + 1e-12)
    sk = species[kk]
    bp_ = torch.full_like(r, probe_b)
    one = _onehot0(r.shape[0], r)
    v = tab.Z[sk] * (1 - slater_one_center_damp(probe_b * r, 0)[0]) / r
    for t in range(tab.shell_n.shape[1]):
        v = v - tab.shell_n[sk, t] * (1 - slater_two_center_damp_unequal(bp_, tab.shell_b[sk, t] * torch.ones_like(r),
                                                                         r, 0)[0]) / r
    P = _poly_map(X.dtype, X.device)
    # an ion sitting on the probe centre: its induced moments average to zero over the spherical probe
    on = r < 1e-3
    drs = torch.where(on[:, None], torch.ones_like(dr), dr)
    rs = torch.where(on, _r(drs), r)
    k = []
    for bp, wp in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
        lam = 1 - shaped_two_center_damp(bp_, bp[sk], one, wp[sk], rs, 2)
        k.append(_tensor(drs, lam)[:, :, 0] @ P)
    kv = torch.cat([k[0][:, :3], k[1][:, 3:]], dim=1) * (~on).to(X.dtype)[:, None]
    keep = (kk != exclude[mm]).to(X.dtype)
    V = (v * keep).reshape(M, N)
    Kp = (kv * keep[:, None]).reshape(M, N, D_SITE)
    return V, Kp


def frontier_pair_coulomb(pos_ang, species, tab: IonTable, idx):
    """J (C,) Hartree: Coulomb interaction of the donor frontier density of i with the acceptor frontier density of j
    (both normalized) for channels idx (2, C) -- the electron-hole attraction of a transferred electron."""
    X = pos_ang / BOHR_ANG
    i, j = idx
    if i.numel() == 0:
        return X.new_zeros(0)
    r = _r(X[i] - X[j])
    ba, wa = tab.front_b["acceptor"][species[j]], tab.front_w["acceptor"][species[j]]
    bd, wd = tab.front_b["donor"][species[i]], tab.front_w["donor"][species[i]]
    return (1 - shaped_two_center_damp(ba, bd, wa, wd, r, 0)[0]) / r


def frontier_potential_matrix(pos_ang, species, batch_idx, tab: IonTable, role: str, x=None) -> torch.Tensor:
    """phibar[i, j] (N, N): potential of ion j -- its permanent density, and its induced moments x (N, 8) if given --
    averaged over the normalized frontier density of ion i in ``role`` ('donor' or 'acceptor'). Zero on the
    diagonal and across frames. A uniform field adds external_potential(...) to every row."""
    V, K = frontier_operator(pos_ang, species, batch_idx, tab, role)
    return V if x is None else V + (K * x[None, :, :]).sum(-1)
