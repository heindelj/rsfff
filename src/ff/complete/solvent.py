"""The continuum (C-PCM, :mod:`rsfff.ff.cpcm`) inside the complete model, mirroring the ALMO-EDA-in-PCM definitions
of :mod:`rsfff.qcgen.almo` state by state:

* fragments: each ion alone in its own sphere, E_A + G_own[V_A]  (a spherical ion has no induced response to its
  own reaction field, so the force-field fragment reference is the permanent density alone);
* frozen: permanent densities in the complex cavity, E_elst + G_cplx[V0], V0 = sum_A V_A the smeared surface
  potential of the nuclei and Slater shells;
* polarized: the induced moments and the surface charges minimized jointly. Eliminating the charges,
  q = -f S^-1 (V0 + J x), turns the joint SPD system into the usual induction solve with a screened operator,

      A -> A - f J^T S^-1 J,   L -> L - f J^T S^-1 V0,   E(x) = 1/2 x^T A' x + L' . x + G_cplx[V0],

  so ``solve_induction_blocks`` is reused unchanged and E_pol = 1/2 L' . x is the joint minimum relative to the
  frozen state in the same cavity -- the QM "pol" (E_scfmi - E_frz, both with their own equilibrated charges);
* CT: the reaction potential of the frozen + induced state at the atoms, phi_rf(x) = J_a^T q(x), is added to the
  external potential in the CT gap (an electron moved from the solvated anion to the solvated cation loses the
  reaction field's stabilization, which is what suppresses CT in the continuum). The CT charges themselves do not
  re-polarize the continuum (second order; the self-consistent CT form keeps that out too).

The per-frame cavity, S, potentials and solves are differentiable in the positions (forces by autograd). The frozen
decomposition reported alongside: ``e_scr = G_cplx[V0] - sum_A G_cplx[V_A]`` (the screened classical cross term,
part of cls_elec) and ``e_cav = sum_A (G_cplx[V_A] - G_own[V_A])`` (desolvation), exactly the QM bookkeeping.

J (the induced moments' surface potential) uses the one-centre damped potentials of the shaped parents (unsmeared);
the permanent potentials are smeared by the points' Gaussians exactly as pyscf does."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import cpcm
from ..ionct import kernels as K
from ..ionct.params import IonTable
from ..multipole import shaped_one_center_multi

__all__ = ["SolventTerms", "solvent_terms", "permanent_surface_potential", "induced_surface_kernel"]

BOHR_ANG = K.BOHR_ANG


@dataclass
class SolventTerms:
    spec: cpcm.CPCMSpec
    dL: torch.Tensor          # (B, nmax, 8)   -f J^T S^-1 V0
    dA: torch.Tensor          # (B, 8 nmax, 8 nmax)   -f J^T S^-1 J
    phi0: torch.Tensor        # (n,)  reaction potential of the frozen state at the atoms
    Phi: torch.Tensor         # (B, nmax, 8 nmax)  d phi_rf / d x
    G_frz: torch.Tensor       # (B,)  G_cplx[V0]
    G_frag_cplx: torch.Tensor # (n,)  G_cplx[V_A]
    G_own: torch.Tensor       # (n,)  G_own[V_A]
    e_scr: torch.Tensor       # (B,)
    e_cav: torch.Tensor       # (B,)
    npoints: list             # per frame
    cavities: list            # per frame (Cavity), for export / tests

    def phi_rf(self, bt, x):
        """(n,) reaction potential at the atoms for induced moments x (B, nmax, 8)."""
        B, nmax = x.shape[0], x.shape[1]
        phi = torch.einsum("bik,bk->bi", self.Phi, x.reshape(B, -1))
        return self.phi0 + phi[bt["mask"]]


def permanent_surface_potential(cav: cpcm.Cavity, X_bohr, species, tab: IonTable):
    """(P, n): the smeared surface potential of each ion's nucleus + Slater shells."""
    z = tab.Z[species]
    d = (cav.coords[:, None, :] - X_bohr[None, :, :]).norm(dim=-1)                 # (P, n)
    V = z[None, :] * torch.erf(cav.zeta[:, None] * d) / d
    for t in range(tab.shell_n.shape[1]):
        n_t, b_t = tab.shell_n[species, t], tab.shell_b[species, t]
        if bool((n_t > 0).any()):
            V = V + _shell_columns(cav, X_bohr, n_t, b_t)
    return V


def _shell_columns(cav, X_bohr, n, b):
    """(P, n) smeared potential columns of one Slater shell per atom (electron sign)."""
    R = (cav.coords[:, None, :] - X_bohr[None, :, :]).norm(dim=-1)
    z = cav.zeta[:, None]
    s = z * z * R * R
    xp, xm = b[None, :] / (2 * z) - z * R, b[None, :] / (2 * z) + z * R
    I0p = cpcm.SQRT_PI / (2 * z) * cpcm._exp_erfcx(xp, s)
    I0m = cpcm.SQRT_PI / (2 * z) * cpcm._exp_erfcx(xm, s)
    e = torch.exp(-s) / (2 * z * z)
    I1p = (R - b[None, :] / (2 * z * z)) * I0p + e
    I1m = (-R - b[None, :] / (2 * z * z)) * I0m + e
    phi = torch.erf(z * R) / R - z / (cpcm.SQRT_PI * R) * (I0p - I0m + 0.5 * b[None, :] * (I1p - I1m))
    return -n[None, :] * phi


def induced_surface_kernel(cav: cpcm.Cavity, X_bohr, species, tab: IonTable):
    """(P, n, 8): potential at the surface points of unit induced dipole / quadrupole on each ion's shaped parents
    (one-centre damping between the parent and the point)."""
    P_, n = cav.npoints, X_bohr.shape[0]
    pp, kk = torch.meshgrid(torch.arange(P_, device=X_bohr.device), torch.arange(n, device=X_bohr.device), indexing="ij")
    pp, kk = pp.reshape(-1), kk.reshape(-1)
    dr = X_bohr[kk] - cav.coords[pp]                                   # source - point
    r = dr.norm(dim=-1)
    sk = species[kk]
    Pm = K._poly_map(X_bohr.dtype, X_bohr.device)
    k = []
    for bp, wp in ((tab.b_dip, tab.w_dip), (tab.b_quad, tab.w_quad)):
        lam = 1 - shaped_one_center_multi(bp[sk], wp[sk], r, 2)
        k.append(K._tensor(dr, lam)[:, :, 0] @ Pm)
    kv = torch.cat([k[0][:, :3], k[1][:, 3:]], dim=1)
    return kv.reshape(P_, n, K.D_SITE)


def solvent_terms(bt: dict, tab: IonTable, spec: cpcm.CPCMSpec, eps=None) -> SolventTerms | None:
    """The continuum's contributions for a batch (``bt`` from ``collate``: pos (n, 3) A, species, frame, mask).
    ``eps``: per-frame dielectric constants (B,) overriding spec.eps (1 = gas phase for that frame; None = spec.eps
    for every frame). Returns None when no frame has a continuum."""
    B, nmax = bt["B"], bt["nmax"]
    pos, sp, fr = bt["pos"], bt["species"], bt["frame"]
    dt, dev = pos.dtype, pos.device
    n = sp.shape[0]
    eps_b = [spec.eps] * B if eps is None else [float(e) for e in eps]
    f_b = [cpcm.f_epsilon(e, spec.method) for e in eps_b]
    if not any(f_b):
        return None
    dL = pos.new_zeros(B, nmax, 8)
    dA = pos.new_zeros(B, 8 * nmax, 8 * nmax)
    Phi = pos.new_zeros(B, nmax, 8 * nmax)
    phi0 = pos.new_zeros(n)
    G_frz, e_scr, e_cav = pos.new_zeros(B), pos.new_zeros(B), pos.new_zeros(B)
    G_frag, G_own = pos.new_zeros(n), pos.new_zeros(n)
    symbols = tab.symbols
    npts, cavs = [], []
    for b in range(B):
        f = f_b[b]
        idx = (fr == b).nonzero().squeeze(1)
        nb = idx.shape[0]
        if f == 0.0 or nb == 0:
            npts.append(0)
            cavs.append(None)
            continue
        X = pos[idx] / BOHR_ANG
        s_b = sp[idx]
        radii = cpcm.sphere_radii([symbols[int(k)] for k in s_b], spec, dtype=dt, device=dev)
        cav = cpcm.build_cavity(X, radii, spec)
        S = cpcm.s_matrix(cav)
        chol = torch.linalg.cholesky(S)
        Va = permanent_surface_potential(cav, X, s_b, tab)                      # (P, nb)
        J = induced_surface_kernel(cav, X, s_b, tab).reshape(cav.npoints, 8 * nb)
        d = (cav.coords[:, None, :] - X[None, :, :]).norm(dim=-1)
        Ja = torch.erf(cav.zeta[:, None] * d) / d                               # (P, nb)
        rhs = torch.cat([Va, J, Ja], 1)
        Y = torch.cholesky_solve(rhs, chol)                                     # S^-1 [Va | J | Ja]
        Yv, Yj, Ya = Y[:, :nb], Y[:, nb:nb + 8 * nb], Y[:, nb + 8 * nb:]
        V0 = Va.sum(1)
        SV0 = Yv.sum(1)
        g_frz = -0.5 * f * (V0 @ SV0)
        g_frag = -0.5 * f * (Va * Yv).sum(0)                                    # (nb,)
        # own cavities: one sphere per ion
        g_own = []
        for k in range(nb):
            cav1 = cpcm.build_cavity(X[k:k + 1], radii[k:k + 1], spec)
            S1 = cpcm.s_matrix(cav1)
            V1 = permanent_surface_potential(cav1, X[k:k + 1], s_b[k:k + 1], tab)[:, 0]
            g_own.append(cpcm.energy(S1, V1, f))
        g_own = torch.stack(g_own)
        dL[b, :nb] = (-f * (J.T @ SV0)).reshape(nb, 8)
        dA[b, :8 * nb, :8 * nb] = -f * (J.T @ Yj)
        Phi[b, :nb, :8 * nb] = -f * (Ja.T @ Yj)
        phi0[idx] = -f * (Ja.T @ SV0)
        G_frz[b] = g_frz
        G_frag[idx] = g_frag
        G_own[idx] = g_own
        e_scr[b] = g_frz - g_frag.sum()
        e_cav[b] = (g_frag - g_own).sum()
        npts.append(cav.npoints)
        cavs.append(cav)
    return SolventTerms(spec, dL, dA, phi0, Phi, G_frz, G_frag, G_own, e_scr, e_cav, npts, cavs)
