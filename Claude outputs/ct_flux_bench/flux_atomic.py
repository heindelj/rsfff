"""Prototype: the flux-CT joint minimum solved in atomic-charge space instead of channel space.

Every term of the flux functional that couples channels -- the charge Coulomb/hardness Q^T H Q, the charges' coupling to
the induced moments Q^T C^T z, the external potential -- sees the channel charges only through the atomic charge changes
Q = G q (N per frame). The only per-channel pieces are the diagonal compliance b_c = 1/s_c^2 and the drive V_c. So the
channels can be eliminated exactly:

    stationarity in q:   b_c q_c - V_c + (G^T psi)_c = 0,      psi = H Q + C^T z + phi_ext   (the atomic potential)
    =>                   q = w (V - G^T psi),   w_c = 1/b_c = s_c^2,   Q = G q = u - K psi
    with                 K = G diag(w) G^T = sum_c w_c g_c g_c^T   (N x N weighted graph Laplacian, built by scatter)
                         u = G diag(w) V

With z = z_pol - W Q, W = (A + H_x)^-1 C (N right-hand sides of the xpol solve, not one per channel) and
S = H - C^T W (N x N), the joint minimum is the N x N system

    (1 + K S) Q = u - K (phi_ext + C^T z_pol)

and E_CT = 1/2 sum_c btilde_c q_c with btilde_c = -V_c + (G^T (phi_ext + C^T z_pol))_c, exactly as before.
1 + K S is nonsingular (K PSD, S PD). Vanishing channels have w_c -> 0 and simply drop out of K and u.

Cost per frame: xpol LU (D N)^3 + N (D N)^2 + C^T W N^2 (D N) + N^3, plus O(C) scatters -- independent of the channel
count except for the O(C) gathers/scatters. The channel-space version is (D N)^2 C + C^2 (D N) + C^3.
"""
import torch
from torch import nn

from rsfff.ff.complete.tabulate import FluxTable, flux_pair_terms
from rsfff.ff.complete.model import slater_two_center_damp_unequal
from rsfff.ff.ionct import kernels as K


def flux_operator_atomic(self, bt, sp, q, D, quadrupoles=True):
    """Same inputs as CompleteModel.flux_operator. Returns (C_atom (B, nmax D, nmax), H_atom (B, nmax, nmax),
    s (C,), V (C,), None, (cf, li, lj), e_c (C,))."""
    B, nmax = bt["B"], bt["nmax"]
    fr, loc = self._local(bt)
    if self._flux_table is None:
        self._flux_table = FluxTable(self.tab)
    tbl = self._flux_table
    if tbl.device != bt["pos"].device:
        tbl.to(bt["pos"].device)
    X = bt["pos"] / K.BOHR_ANG
    pairs = K.all_pairs(bt["frame"])
    dp = torch.cat([pairs, pairs.flip(0)], 1)
    a_, b_ = dp
    Kp, Jp = flux_pair_terms(tbl, sp, dp, X[b_] - X[a_])
    Kd = X.new_zeros(B, nmax, nmax, 8).index_put((fr[a_], loc[a_], loc[b_]), Kp)    # [l, k]: on l of a charge at k
    eta = self.flux_jself[sp] + nn.functional.softplus(self.flux_eta[sp])
    Jd = X.new_zeros(B, nmax, nmax).index_put((fr[a_], loc[a_], loc[b_]), Jp).index_put((fr, loc, loc), eta)
    Ca = Kd.permute(0, 1, 3, 2)                                                     # (B, nmax, 8, nmax): [l, a, k]
    if not quadrupoles:
        Ca = torch.cat([Ca[:, :, :3], torch.zeros_like(Ca[:, :, 3:])], 2)
    if D > 8:
        Ca = torch.cat([Ca, Ca.new_zeros(B, nmax, D - 8, nmax)], 2)
    i, j = bt["ch_index"]
    cf = bt["ch_frame"]
    li, lj = loc[i], loc[j]
    # h_c = (G^T H G)_cc by gather (no Cmax x Cmax matrix)
    h = Jd[cf, li, li] + Jd[cf, lj, lj] - Jd[cf, li, lj] - Jd[cf, lj, li]
    bD, bA = self.ct_exponents()
    rb = bt["ch_r"] / K.BOHR_ANG
    f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
    e_c = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
    ae = (-e_c).clamp(min=0.0)
    v0 = torch.exp(self.flux_lnv)
    V0 = v0[sp[i]] + v0[sp[j]]
    Vh2 = V0 * V0 + 2 * ae * h
    cc = 2 * ae / Vh2
    rev = self.flux_reverse(i, j, sp.shape[0])
    has = rev >= 0
    rv = rev.clamp(min=0)
    ae2 = torch.where(has, ae[rv], torch.zeros_like(ae))
    cc2 = torch.where(has, cc[rv], torch.zeros_like(cc))
    S_ = ae + ae2
    t2 = S_ * (1 - h * h * cc * cc2) / (S_ + 2 * h * torch.sqrt(ae * cc * ae2 * cc2 + 1e-300) + 1e-300)
    t2 = torch.where(has, t2, torch.ones_like(t2))
    Vc = torch.sqrt(t2 * Vh2)
    sc = torch.sqrt(2 * ae + 1e-30) / V0
    return Ca.reshape(B, nmax * D, nmax), Jd, sc, Vc, None, (cf, li, lj), e_c


def flux_solve_atomic(self, bt, Ca, Ha, s, V, _G, ch, z_pol, W, phi_ext):
    """Drop-in for CompleteModel.flux_solve with the atomic operator: per frame E_CT (B,), per channel 1/2 btilde_c q_c
    (C,), q_c (C,), the joint moments z (B, nmax D) and the atomic charge changes Q (B, nmax)."""
    cf, li, lj = ch
    B, N = Ha.shape[0], Ha.shape[1]
    fr, loc = self._local(bt)
    w = s * s                                                      # compliance 1 / b_c
    Kl = Ha.new_zeros(B, N, N)
    Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
    Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
    wV = w * V
    u = Ha.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
    phi = Ha.new_zeros(B, N).index_put((fr, loc), phi_ext)
    psi0 = phi + torch.einsum("bnk,bn->bk", Ca, z_pol)             # potential on the charges at z_pol
    S = Ha - Ca.transpose(1, 2) @ W                                # (B, N, N)
    M = torch.eye(N, dtype=Ha.dtype, device=Ha.device) + Kl @ S
    Q = torch.linalg.solve(M, (u - (Kl @ psi0[..., None])[..., 0])[..., None])[..., 0]
    psi = psi0 + (S @ Q[..., None])[..., 0]
    qc = w * (V - (psi[cf, li] - psi[cf, lj]))
    btil = -V + psi0[cf, li] - psi0[cf, lj]
    e_ch = 0.5 * btil * qc
    e = Ha.new_zeros(B).index_add(0, cf, e_ch)
    z = z_pol - (W @ Q[..., None])[..., 0]
    return e, e_ch, qc, z, Q


def patch(model_cls):
    """Swap the channel-space flux solve for the atomic one on a CompleteModel class (returns the originals)."""
    orig = (model_cls.flux_operator, model_cls.flux_solve)
    model_cls.flux_operator = flux_operator_atomic
    model_cls.flux_solve = flux_solve_atomic
    return orig


def unpatch(model_cls, orig):
    model_cls.flux_operator, model_cls.flux_solve = orig
