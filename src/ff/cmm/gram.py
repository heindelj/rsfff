"""The Gram term of parameter-free exchange polarization (theory/parameter_free_xpol.tex).

Each ion's response is a perturbation of one von Weizsaecker orbital phi_i = sqrt(rho_i) (rho_i its electron density,
the permanent shells): the induced electron density delta n_i = 2 phi_i delta phi_i, so delta phi_i = delta n_i / (2 phi_i).
The overlap form of the Pauli energy that xpol anchors on the frozen pair energy, E_P,ij = A_ij <rho~_i | rho~_j> with
A_ij = E0_ij / O_ij, evaluated with the perturbed density rho~_i = (phi_i + delta phi_i)^2, has the linear (drive) term
A_ij <delta n_i | rho_j> -- xpol's drive with kappa = 1/2 -- and the second-order, on-site, positive semi-definite term

    E_G(i <- j) = A_ij int delta phi_i^2 rho_j = A_ij sum_ab z_a z_b Gt_ab(R_ij),
    Gt_ab(R) = int d^3r  delta n_a(r) delta n_b(r) rho_j(|r - R|) / (4 rho_i(r))

over i's response components z = [dipole (3, Cartesian), quadrupole (5, normalized spherical), breathing (1)] with the
model's own induced-density conventions (checked against the xpol overlap maps of the precompute):

    dipole a:      delta n = d_a g_dip            = g_dip'(r) r^_a
    quadrupole c:  delta n = M_c : dd g_quad,     M_c = -(1/3) Theta_cart(e_c)  (traceless)
                           = (g_quad'' - g_quad' / r) r^T M_c r^
    breathing:     delta n = g_breath(r) - rho_i(r) / N_i    (zero monopole: the breathing shape's electrons come
                                                           out of the ion's own normalized density)

with g the normalized parent shapes. With rho_j(|r - R|) = sum_L (2L+1)/(4 pi) rho_j^(L)(r; R) P_L(r^ . R^),

    Gt_ab(R) = sum_L (2L+1)/(4 pi) F^L_{t(a) t(b)}(R) Ang_abL(R^),
    F^L_tu(R) = int dr r^2 u_t(r) u_u(r) rho_j^(L)(r; R) / (4 rho_i(r)),     Ang_abL = int dOmega p_a p_b P_L(r^ . R^)

ten radial functions per ordered species pair (dd0 dd2 qq0 qq2 qq4 bb0 db1 qb2 dq1 dq3: the ten numbers of the
cylindrical pair form of theory/response_symmetry.tex) tabulated in R, times fixed polynomials in R^.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import kernels as K
from ..multipole import spherical_to_cartesian_quadrupole
from .tabulate import RadialTable, _monomials

__all__ = ["GramTable", "gram_blocks", "gram_blocks_from_rows", "site_drive_from_rows", "GRAM_TERMS", "gram_layout"]

BOHR = K.BOHR_ANG
#: (type a, type b, L) of the ten radial functions; types d (dipole), q (quadrupole), b (breathing)
GRAM_TERMS = (("d", "d", 0), ("d", "d", 2), ("q", "q", 0), ("q", "q", 2), ("q", "q", 4), ("b", "b", 0),
              ("d", "b", 1), ("q", "b", 2), ("d", "q", 1), ("d", "q", 3))
_TYPES = ("d",) * 3 + ("q",) * 5 + ("b",)


def gram_layout(breath="shape"):
    """(component types, terms) of the response vector: [dipole 3, quadrupole 5, breathing 1]."""
    if breath != "shape":
        raise ValueError(f"breath {breath!r}: only the breathing shape ('shape')")
    return _TYPES, GRAM_TERMS


GRAM_R_STEP = 0.01          # bohr: radial quadrature step
GRAM_R_END = 30.0           # bohr
GRAM_NS = 64                # Gauss-Legendre points in the partner distance s


def _legendre(L, x):
    return [lambda x: torch.ones_like(x), lambda x: x, lambda x: 0.5 * (3 * x * x - 1),
            lambda x: 0.5 * (5 * x ** 3 - 3 * x), lambda x: (35 * x ** 4 - 30 * x * x + 3) / 8][L](x)


def _angular_polys(rh, nb=1):
    """(G, 8 + nb) the angular factors p_a(r^) of the response components at unit vectors rh (G, 3)."""
    eye = torch.eye(5, dtype=rh.dtype)
    M = -spherical_to_cartesian_quadrupole(eye) / 3.0                        # (5, 3, 3)
    q = torch.einsum("ga,cab,gb->gc", rh, M, rh)
    return torch.cat([rh, q, torch.ones_like(rh[:, :1]).expand(-1, nb)], 1)


_COEF: dict = {}


def gram_angular_coef(dtype=torch.float64, device="cpu", breath="shape"):
    """C (T, D, D, 35): Gt_ab(R) = sum_t F_t(R) sum_m C[t, a, b, m] mono_m(R^), with (2L+1)/(4 pi) folded in and
    only the blocks of the term's type pair (both orders) filled (layout: :func:`gram_layout`)."""
    key = (dtype, str(device), breath)
    if key in _COEF:
        return _COEF[key]
    f64 = torch.float64
    xg, wg = np.polynomial.legendre.leggauss(24)
    x = torch.tensor(xg, dtype=f64)
    wx = torch.tensor(wg, dtype=f64)
    nph = 48
    ph = torch.arange(nph, dtype=f64) * (2 * math.pi / nph)
    st = torch.sqrt(1 - x * x)
    rh = torch.stack([(st[:, None] * torch.cos(ph)[None]).reshape(-1), (st[:, None] * torch.sin(ph)[None]).reshape(-1),
                      x[:, None].expand(-1, nph).reshape(-1)], 1)                      # (G, 3)
    w = (wx[:, None] * (2 * math.pi / nph)).expand(-1, nph).reshape(-1)              # dOmega
    types, terms = gram_layout(breath)
    D = len(types)
    p = _angular_polys(rh, D - 8)                                                     # (G, D)
    gen = torch.Generator().manual_seed(1)
    Rh = torch.randn(400, 3, generator=gen, dtype=f64)
    Rh = Rh / Rh.norm(dim=1, keepdim=True)
    mono = _monomials(Rh)                                                             # (400, 35)
    C = torch.zeros(len(terms), D, D, 35, dtype=f64)
    for t, (ta, tb, L) in enumerate(terms):
        PL = _legendre(L, Rh @ rh.T)                                                  # (400, G)
        ang = torch.einsum("g,ga,gb,ng->nab", w, p, p, PL)                            # (400, D, D)
        sel = torch.zeros(D, D, dtype=f64)
        for a in range(D):
            for b in range(D):
                if {types[a], types[b]} == {ta, tb} and (ta == tb or types[a] != types[b]):
                    sel[a, b] = 1.0
        Y = (ang * sel).reshape(400, D * D) * (2 * L + 1) / (4 * math.pi)
        sol = torch.linalg.lstsq(mono, Y).solution                                    # (35, 81)
        res = (mono @ sol - Y).abs().max()
        if res > 1e-10:
            raise RuntimeError(f"gram angular fit residual {float(res):.2e} for term {(ta, tb, L)}")
        C[t] = sol.T.reshape(D, D, 35)
    _COEF[key] = C.to(dtype=dtype, device=device)
    return _COEF[key]


def _shape_radial(b, w, r):
    """g, g', g'' (each (G,)) of a normalized shape sum_t sum_n w[t, n] N_n r^n e^{-b_t r} at r (bohr)."""
    b = b.reshape(-1)
    w = w.reshape(b.shape[0], -1)
    g, g1, g2 = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
    for t in range(b.shape[0]):
        bt = float(b[t])
        if bt <= 0:
            continue
        e = torch.exp(-bt * r)
        for n in range(w.shape[1]):
            wn = float(w[t, n])
            if wn == 0.0:
                continue
            N = bt ** (n + 3) / (4 * math.pi * math.factorial(n + 2))
            rn = r ** n
            rn1 = n * r ** (n - 1) if n >= 1 else torch.zeros_like(r)
            rn2 = n * (n - 1) * r ** (n - 2) if n >= 2 else torch.zeros_like(r)
            g = g + wn * N * rn * e
            g1 = g1 + wn * N * (rn1 - bt * rn) * e
            g2 = g2 + wn * N * (rn2 - 2 * bt * rn1 + bt * bt * rn) * e
    return g, g1, g2


def _shell_density(tab, s, r):
    out = torch.zeros_like(r)
    for t in range(tab.shell_n.shape[1]):
        n, b = float(tab.shell_n[s, t]), float(tab.shell_b[s, t])
        if n > 0:
            out = out + n * b ** 3 / (8 * math.pi) * torch.exp(-b * r)
    return out


def response_radials(tab, s, r, breath="shape"):
    """{d, q, b}: the radial factors u_t(r) of species s's induced electron densities (zero if s has no such
    shape)."""
    _, gd1, _ = _shape_radial(tab.b_dip[s], tab.w_dip[s], r)
    _, gq1, gq2 = _shape_radial(tab.b_quad[s], tab.w_quad[s], r)
    if getattr(tab, "a0", None) is not None and float(tab.a0[s]) > 0:
        # zero monopole: the breathing shape's electrons come out of the ion's own (normalized) density
        gb, _, _ = _shape_radial(tab.b_breath[s], tab.w_breath[s], r)
        gb = gb - _shell_density(tab, s, r) / float(tab.shell_n[s].sum())
    else:
        gb = torch.zeros_like(r)
    return {"d": gd1, "q": gq2 - gq1 / r, "b": gb}


class GramTable(RadialTable):
    """Per ordered species pair (i <- j), the ten radial functions ``gram:{ab}{L}`` (:data:`GRAM_TERMS`) of
    Gt(i <- j)(R), R the distance to j (bohr^-3 per unit response squared, like the xpol overlaps), with first and second
    R-derivatives (Angstrom) by fourth-order finite differences on the grid."""

    def __init__(self, tab, r_max=10.0, breath="shape", **kw):
        self.breath = breath
        self.types, self.terms = gram_layout(breath)
        self._rhoL = {}
        f64 = torch.float64
        self._rq = torch.arange(GRAM_R_STEP / 2, GRAM_R_END, GRAM_R_STEP, dtype=f64)          # bohr
        xs, ws = np.polynomial.legendre.leggauss(GRAM_NS)
        self._xs, self._ws = torch.tensor(xs, dtype=f64), torch.tensor(ws, dtype=f64)
        super().__init__(tab, **{"probes": (), "r_max": r_max, **kw})

    def _key_extra(self):
        return f"{super()._key_extra()}|{self.breath}|{GRAM_R_STEP}|{GRAM_R_END}|{GRAM_NS}"

    def _gram_names(self):
        return [f"gram:{a}{b}{L}" for a, b, L in self.terms]

    def _names(self):
        return self._gram_names()

    def _rho_partial(self, sj, partner=None):
        """(5, G + 4, Nr): rho_j^(L)(r; R) for L = 0..4 on the padded R grid (two extra points each side).
        ``partner = ("site", beta)``: the unit site potential e^{-beta s} (1 / s + beta / 2) in place of j's density
        (zero for beta <= 0; its 1 / s is integrable in the partial waves)."""
        ck = sj if partner is None else partner
        if ck in self._rhoL:
            return self._rhoL[ck]
        h = self.h
        Rg = torch.cat([self.r[0] - h * torch.arange(2, 0, -1, dtype=self.r.dtype), self.r,
                        self.r[-1] + h * torch.arange(1, 3, dtype=self.r.dtype)]) / BOHR           # bohr
        r = self._rq
        out = torch.zeros(5, Rg.shape[0], r.shape[0], dtype=r.dtype)
        for c0 in range(0, Rg.shape[0], 16):
            R = Rg[c0:c0 + 16][:, None, None]                                                    # (c, 1, 1)
            lo, hi = (r[None, :, None] - R).abs(), r[None, :, None] + R
            s = 0.5 * (hi + lo) + 0.5 * (hi - lo) * self._xs[None, None, :]                    # (c, Nr, Ns)
            ws = 0.5 * (hi - lo) * self._ws[None, None, :]
            x = ((r[None, :, None] ** 2 + R * R - s * s) / (2 * r[None, :, None] * R)).clamp(-1.0, 1.0)
            if partner is None:
                rho = _shell_density(self.tab, sj, s)
            else:
                beta = float(partner[1])
                rho = torch.exp(-beta * s) * (1.0 / s + 0.5 * beta) if beta > 0 else torch.zeros_like(s)
            base = ws * rho * s * (2 * math.pi) / (r[None, :, None] * R)
            for L in range(5):
                out[L, c0:c0 + 16] = (base * _legendre(L, x)).sum(-1)
        self._rhoL[ck] = out
        return out

    def _functions(self, si, sj):
        return self._gram_rows(si, sj)

    @torch.no_grad()
    def _drive_rows(self, si, sj, partner):
        """{d1, q2, b0, ovl}: int dr r^2 u_t(r) v^(L)(r; R) of i's response radials (rank L their own) and of i's
        density (ovl, L = 0) with the partner function, on the Gram rows' own quadrature: with them the pair's
        second-order energy is a complete square on one discrete measure."""
        r = self._rq
        u = response_radials(self.tab, si, r, self.breath)
        wr = GRAM_R_STEP * r * r
        RL = self._rho_partial(sj, partner)
        h = self.h
        res = {}
        for name, f_r, L in (("d1", u["d"], 1), ("q2", u["q"], 2), ("b0", u["b"], 0),
                             ("ovl", _shell_density(self.tab, si, r), 0)):
            f = RL[L] @ (wr * f_r)
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res[name] = torch.stack([f[2:-2], d1, d2], -1)
        return res

    @torch.no_grad()
    def _gram_rows(self, si, sj, partner=None):
        r = self._rq
        rho_i = _shell_density(self.tab, si, r)
        u = response_radials(self.tab, si, r, self.breath)
        wr = GRAM_R_STEP * r * r / (4 * rho_i.clamp(min=1e-300))
        RL = self._rho_partial(sj, partner)
        h = self.h
        res = {}
        for a, b, L in self.terms:
            f = RL[L] @ (wr * u[a] * u[b])                                                       # (G + 4,)
            f0 = f[2:-2]
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res[f"gram:{a}{b}{L}"] = torch.stack([f0, d1, d2], -1)
        return res


def site_drive_from_rows(yd, dr):
    """(P, 9) int dn_a v from the pairs' drive rows ``yd`` (P, 3) = [d1, q2, b0] (:meth:`GramTable._drive_rows`) and
    dr = X_j - X_i (bohr): the radial integrals times the response components' own angular factors at R^."""
    p = _angular_polys(dr / dr.norm(dim=-1)[:, None], 1)
    return torch.cat([yd[:, :1] * p[:, :3], yd[:, 1:2] * p[:, 3:8], yd[:, 2:3] * p[:, 8:]], 1)


def gram_blocks(table: GramTable, s, dp, dr):
    """(P, D, D) Gt(i <- j) for directed pairs dp = (i, j) with dr = X_j - X_i (bohr, autograd flows); D = 9 or 10
    (the table's breathing layout)."""
    i, j = dp
    r = dr.norm(dim=-1)
    y = table(s[i], s[j], r * BOHR)
    breath = getattr(table, "breath", "shape")
    cols = table.cols(tuple(f"gram:{a}{b}{L}" for a, b, L in gram_layout(breath)[1]), y.device)
    return gram_blocks_from_rows(y[:, cols], dr, breath)


def gram_blocks_from_rows(yg, dr, breath="shape"):
    """(P, D, D) Gt(i <- j) from the pairs' Gram rows ``yg`` (P, terms) (a table lookup at |dr|) and dr = X_j - X_i
    (bohr)."""
    C = gram_angular_coef(dr.dtype, dr.device, breath)
    mono = _monomials(dr / dr.norm(dim=-1)[:, None])
    return torch.einsum("pt,tabm,pm->pab", yg, C, mono)
