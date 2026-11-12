"""The Gram term of parameter-free exchange polarization (theory/parameter_free_xpol.tex).

Each ion's response is a perturbation of one von Weizsaecker orbital phi_i = sqrt(rho_i) (rho_i its electron density,
the permanent shells): the induced electron density delta n_i = 2 phi_i delta phi_i, so delta phi_i = delta n_i / (2 phi_i).
The overlap form of the Pauli energy that xpol anchors on the frozen pair energy, E_P,ij = A_ij <rho~_i | rho~_j> with
A_ij = E0_ij / O_ij, evaluated with the perturbed density rho~_i = (phi_i + delta phi_i)^2, has the linear (drive) term
A_ij <delta n_i | rho_j> -- xpol's drive with kappa = 1/2 -- and the second-order, on-site, positive semi-definite term

    E_G(i <- j) = A_ij int delta phi_i^2 rho_j = A_ij sum_ab z_a z_b Gt_ab(R_ij),
    Gt_ab(R) = int d^3r  delta n_a(r) delta n_b(r) rho_j(|r - R|) / (4 rho_i(r))

over i's response components z = [dipole (3, Cartesian), quadrupole (5, normalized spherical), breathing (1)] with the
model's own induced-density conventions (checked against :func:`xpol_overlap_map`):

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

from ..ionct import kernels as K
from ..multipole import spherical_to_cartesian_quadrupole
from .tabulate import RadialTable, _monomials

__all__ = ["GramTable", "gram_blocks", "GRAM_TERMS", "gram_layout"]

BOHR = K.BOHR_ANG
#: (type a, type b, L) of the ten radial functions; types d (dipole), q (quadrupole), b (breathing)
GRAM_TERMS = (("d", "d", 0), ("d", "d", 2), ("q", "q", 0), ("q", "q", 2), ("q", "q", 4), ("b", "b", 0),
              ("d", "b", 1), ("q", "b", 2), ("d", "q", 1), ("d", "q", 3))
_TYPES = ("d",) * 3 + ("q",) * 5 + ("b",)


def gram_layout(breath="shape"):
    """(component types, terms) of the response vector: [dipole 3, quadrupole 5] plus the l = 0 slots -- one "b" (the
    breathing shape) or two "b0", "b1" (``breath="contract"``: the shell contraction modes, contract.py)."""
    if breath == "shape":
        return _TYPES, GRAM_TERMS
    bs = [f"b{t}" for t in range(2)]
    terms = [t for t in GRAM_TERMS if "b" not in t[:2]]
    terms += [(a, b, 0) for k, a in enumerate(bs) for b in bs[k:]]
    terms += [("d", a, 1) for a in bs] + [("q", a, 2) for a in bs]
    return ("d",) * 3 + ("q",) * 5 + tuple(bs), tuple(terms)
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
    """{d, q, b} (or {d, q, b0, b1} with ``breath="contract"``): the radial factors u_t(r) of species s's induced
    electron densities (zero if s has no such shape)."""
    _, gd1, _ = _shape_radial(tab.b_dip[s], tab.w_dip[s], r)
    _, gq1, gq2 = _shape_radial(tab.b_quad[s], tab.w_quad[s], r)
    if getattr(tab, "a0", None) is not None and float(tab.a0[s]) > 0:
        # zero monopole: the breathing shape's electrons come out of the ion's own (normalized) density
        gb, _, _ = _shape_radial(tab.b_breath[s], tab.w_breath[s], r)
        gb = gb - _shell_density(tab, s, r) / float(tab.shell_n[s].sum())
    else:
        gb = torch.zeros_like(r)
    if breath == "contract":
        from .contract import contraction_modes
        m = contraction_modes(tab, s, r.detach().numpy())
        return {"d": gd1, "q": gq2 - gq1 / r, "b0": torch.tensor(m[0], dtype=r.dtype),
                "b1": torch.tensor(m[1], dtype=r.dtype)}
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
        with torch.no_grad():
            super().__init__(tab, probes=(), r_max=r_max, **kw)
        del self._rhoL

    def _rho_partial(self, sj):
        """(5, G + 4, Nr): rho_j^(L)(r; R) for L = 0..4 on the padded R grid (two extra points each side)."""
        if sj in self._rhoL:
            return self._rhoL[sj]
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
            rho = _shell_density(self.tab, sj, s)
            base = ws * rho * s * (2 * math.pi) / (r[None, :, None] * R)
            for L in range(5):
                out[L, c0:c0 + 16] = (base * _legendre(L, x)).sum(-1)
        self._rhoL[sj] = out
        return out

    def _functions(self, si, sj):
        r = self._rq
        rho_i = _shell_density(self.tab, si, r)
        u = response_radials(self.tab, si, r, self.breath)
        wr = GRAM_R_STEP * r * r / (4 * rho_i.clamp(min=1e-300))
        RL = self._rho_partial(sj)
        h = self.h
        if not self.names:
            self.names = [f"gram:{a}{b}{L}" for a, b, L in self.terms]
            self.index = {k: m for m, k in enumerate(self.names)}
        res = []
        for a, b, L in self.terms:
            f = RL[L] @ (wr * u[a] * u[b])                                                       # (G + 4,)
            f0 = f[2:-2]
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res.append(torch.stack([f0, d1, d2], -1))
        return res


def gram_blocks(table: GramTable, s, dp, dr):
    """(P, D, D) Gt(i <- j) for directed pairs dp = (i, j) with dr = X_j - X_i (bohr, autograd flows); D = 9 or 10
    (the table's breathing layout)."""
    i, j = dp
    r = dr.norm(dim=-1)
    y = table(s[i], s[j], r * BOHR)                                                             # (P, 10)
    C = gram_angular_coef(dr.dtype, dr.device, getattr(table, "breath", "shape"))
    mono = _monomials(dr / r[:, None])
    return torch.einsum("pt,tabm,pm->pab", y, C, mono)


# ------------------------------------------------------------------------------------------- quadrupole: shell strain
_STRAIN_ANG = {}


def strain_angular():
    """a (3, 3): int dOmega (r^T S r^)^2 P_L(r^.R^) = sum_k a[L/2, k] I_k(S, R^) for L = 0, 2, 4, with the invariants
    of a traceless symmetric S about R^: I1 = (R^T S R^)^2, I2 = R^T S^2 R^, I3 = tr S^2 (fitted once, exact)."""
    if "a" in _STRAIN_ANG:
        return _STRAIN_ANG["a"]
    xg, wg = np.polynomial.legendre.leggauss(24)
    nph = 48
    ph = np.arange(nph) * 2 * np.pi / nph
    st = np.sqrt(1 - xg ** 2)
    rh = np.stack([(st[:, None] * np.cos(ph)).ravel(), (st[:, None] * np.sin(ph)).ravel(), np.repeat(xg, nph)], 1)
    w = np.repeat(wg, nph) * 2 * np.pi / nph
    rng = np.random.default_rng(0)
    rows, ys = [], {L: [] for L in (0, 2, 4)}
    for _ in range(60):
        M = rng.normal(size=(3, 3)); S = M + M.T; S -= np.trace(S) / 3 * np.eye(3)
        Rh = rng.normal(size=3); Rh /= np.linalg.norm(Rh)
        rows.append([(Rh @ S @ Rh) ** 2, Rh @ S @ S @ Rh, np.trace(S @ S)])
        q = np.einsum("ga,ab,gb->g", rh, S, rh) ** 2
        for L in (0, 2, 4):
            ys[L].append(np.sum(w * q * _legendre(L, torch.tensor(rh @ Rh)).numpy()))
    X = np.array(rows)
    a = np.stack([np.linalg.lstsq(X, np.array(ys[L]), rcond=None)[0] for L in (0, 2, 4)])
    if np.abs(X @ a.T - np.stack([ys[L] for L in (0, 2, 4)], 1)).max() > 1e-10:
        raise RuntimeError("strain angular fit is not exact")
    _STRAIN_ANG["a"] = a
    return a


class StrainTable(GramTable):
    """The volume-preserving strain r -> (1 + S) r of each permanent shell (the l = 2 deformation;
    theory/parameter_free_xpol.tex): per ordered species pair (i <- j) and shell t of i (zeros where absent), the overlap
    of the strained normalized shell, det(1 + S) s_t((1 + S) r), with j's electron density to second order in S,

        O_t(S) = O_t + st:d:{t} (R^T S R^) + st:q1:{t} (R^T S R^)^2 + st:q2:{t} R^T S^2 R^ + st:q3:{t} tr S^2

    from the first- and second-order strained densities r s' (r^T S r^) and r^2 s'' (r^T S r^)^2 + r s' (r^T S^2 r^ -
    (r^T S r^)^2) - tr S^2 s, projected on the Legendre partial waves of rho_j (the Gram tables' machinery)."""

    def __init__(self, tab, r_max=10.0, **kw):
        super().__init__(tab, r_max=r_max, breath="shape", **kw)

    def _functions(self, si, sj):
        r = self._rq
        RL = self._rho_partial(sj)
        a = strain_angular()
        sh = [(float(self.tab.shell_n[si, t]), float(self.tab.shell_b[si, t])) for t in range(self.tab.shell_n.shape[1])
              if float(self.tab.shell_n[si, t]) > 0]
        out = {}
        for t in range(2):
            if t < len(sh):
                b = sh[t][1]
                s = b ** 3 / (8 * math.pi) * torch.exp(-b * r)
                F = lambda f, L: RL[L] @ (GRAM_R_STEP * r * r * f)
                f1, f2, f0 = r * (-b * s), r * r * (b * b * s), s           # r s', r^2 s'', s
                c = {L: (2 * L + 1) / (4 * math.pi) * (F(f2, L) - F(f1, L)) for L in (0, 2, 4)}
                q = [0.5 * sum(c[L] * a[L // 2, k] for L in (0, 2, 4)) for k in range(3)]
                q[1] = q[1] + 0.5 * F(f1, 2)
                q[2] = q[2] + 0.5 * (F(f1, 0) / 3 - F(f1, 2) / 3 - F(f0, 0))
                d = F(f1, 2)
            else:
                d = q = None
            z = torch.zeros(self.G + 4, dtype=r.dtype)
            out[f"st:d:{t}"] = z if d is None else d
            for k in range(3):
                out[f"st:q{k + 1}:{t}"] = z if q is None else q[k]
        if not self.names:
            self.names = list(out)
            self.index = {k: m for m, k in enumerate(self.names)}
        h = self.h
        res = []
        for k in self.names:
            f = out[k]
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res.append(torch.stack([f[2:-2], d1, d2], -1))
        return res
