"""Tabulated radial kernels and the batched, differentiable precompute (complete model plan: force training).

Everything the per-frame precompute builds is a sum over atom pairs of a *radial* function of r for the ordered
species pair (s_i, s_j) times an angular tensor in r_hat. The radial functions are the shaped-density damping
complements and overlaps of the fixed parameter table (closed forms and k-space quadrature, the whole cost); the
angular parts are polynomials. So the radial functions are tabulated once per species pair on a fine grid, with
their first two derivatives, and evaluated by quintic Hermite interpolation -- exact in the sense that the force is
the derivative of the interpolated energy, and smooth (C^2). The learnable exponents (direct CT, Pauli, dispersion)
are not in the table: those kernels are closed-form Slater 1s and stay analytic in the model.

:class:`RadialTable` holds, per ordered species pair, the functions

    drive:{dip,quad}:{n}      c(r) = Z_j f^nuc_n - sum_t n_t f^t_n  (complement of the permanent drive; lam = q_j - c)
    aoff:{dd,dq,qd,qq}:{n}    f_n between parent a of i and parent c of j (lam = 1 - f)
    front:{role}:v            same complement as drive, for the frontier density of i (rank 0)
    front:{role}:{dip,quad}:{n}   f_n between frontier of i and parent of j
    envV:{probe}:{n}          drive-type complement for each environment probe of i
    envS:{probe}:{0,1,2}      overlap of probe i with j's shells and its first two radial derivatives
    ps:{dip,quad}:{n}         f^pt_n - f^sh_n, the pseudo-potential drive basis (parent of i, outer shell of j)
    ps{t}:{dip,quad}:{n}      the same for each permanent shell t of j (zero where j has no shell t)

with n = 0..4 the five damping orders of the rank-2 tensor. All of them vanish beyond the overlap range, so the
table ends at ``r_max`` and returns zero past it.

:func:`batched_precompute` assembles from positions (differentiable) the same batch dict ``collate`` makes from
per-frame caches (lean + frontier + environment blocks), for :class:`rsfff.ff.complete.CompleteModel`.
"""
from __future__ import annotations

import math

import torch

from ..ionct import ct as CT
from ..ionct import kernels as K
from ..ionct.model import PSEUDO_CUTOFF, channel_types, pseudo_type_names
from ..ionct.params import IonTable
from ..multipole import shaped_one_center_multi, shaped_two_center_damp
from .environment import INVARIANT_NAMES, PROBE_NAMES, invariants, shaped_overlap_derivs
from .periodic import (dispersion_tail_integral, erfc_screening, ewald_alpha, make_ewald_frames, periodic_pairs,
                       recip_apply, smooth_switch)

__all__ = ["FluxTable", "ProbeOverlapTable", "RadialTable", "XpolTable", "cached_table", "batched_precompute", "flux_pair_terms",
           "shape_self_coulomb", "xpol_overlap_map"]

NORD = 5          # damping orders of the rank-2 tensor
BOHR = K.BOHR_ANG

# Quintic Hermite basis on [0, 1] as a matrix on [1, t, t^2, t^3, t^4, t^5]: columns (h0, h2, h4, h1, h3, h5) --
# the left node's value / slope / curvature weights, then the right node's -- matching coef[..., (y, h y', h^2 y'')].
_HERMITE = torch.tensor([
    # h0     h2    h4     h1     h3    h5
    [1.0,   0.0,  0.0,   0.0,   0.0,  0.0],     # 1
    [0.0,   1.0,  0.0,   0.0,   0.0,  0.0],     # t
    [0.0,   0.0,  0.5,   0.0,   0.0,  0.0],     # t^2
    [-10.0, -6.0, -1.5,  10.0,  -4.0, 0.5],     # t^3
    [15.0,  8.0,  1.5,   -15.0, 7.0,  -1.0],    # t^4
    [-6.0,  -3.0, -0.5,  6.0,   -3.0, 0.5],     # t^5
], dtype=torch.float64)
_HERMITE_CACHE: dict = {}


def _hermite_matrix(dtype, device):
    key = (dtype, device)
    m = _HERMITE_CACHE.get(key)
    if m is None:
        m = _HERMITE_CACHE[key] = _HERMITE.to(dtype=dtype, device=device)
    return m


def _expand(b, w, G):
    """Species-level shape (T,), (T, 4) or (), (4,) -> per-point (G, T), (G, T, 4) / (G,), (G, 4)."""
    if b.dim() == 0:
        return b.expand(G), w[None].expand(G, -1)
    return b[None].expand(G, -1), w[None].expand(G, -1, -1)


def _tab_fingerprint(tab) -> bytes:
    """Bytes that change whenever the ion table's content does (every tensor and string field, by name)."""
    import dataclasses
    import hashlib
    h = hashlib.sha1()

    def feed(name, v):
        if isinstance(v, torch.Tensor):
            h.update(name.encode() + str(tuple(v.shape)).encode() + v.detach().cpu().double().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                feed(f"{name}.{k}", v[k])
        elif isinstance(v, (str, int, float, tuple, list)):
            h.update(name.encode() + repr(v).encode())

    for f in dataclasses.fields(tab):
        feed(f.name, getattr(tab, f.name))
    return h.digest()


def cached_table(cls, tab, **kw):
    """``cls(tab, **kw)``, kept on disk when the environment names a directory (``RSFFF_TABLE_CACHE``; the trainer
    sets it to its --cache-dir): the pair tables the model builds lazily (overlap, Gram, flux, probe) take minutes
    and depend only on the ion table. The key is the class, its arguments, the ion table's content and the source of
    the module that defines the class, so an edit of the table code or of the ion table makes a new entry."""
    import hashlib
    import inspect
    import os
    from pathlib import Path
    d = os.environ.get("RSFFF_TABLE_CACHE")
    if not d:
        return cls(tab, **kw)
    try:
        src = Path(inspect.getsourcefile(cls)).read_bytes() + Path(__file__).read_bytes()
    except (OSError, TypeError):
        return cls(tab, **kw)
    key = hashlib.sha1(cls.__name__.encode() + repr(sorted(kw.items())).encode() + _tab_fingerprint(tab) + src).hexdigest()[:16]
    path = Path(d) / f"pair_{cls.__name__}_{key}.pt"
    if path.exists():
        try:
            return torch.load(path, weights_only=False, map_location="cpu")
        except Exception:  # noqa: BLE001 -- a truncated file: rebuild
            pass
    cpu = tab.to("cpu") if hasattr(tab, "to") else tab
    table = cls(cpu, **kw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(table, tmp)
        tmp.replace(path)
    except OSError:
        pass
    return table


class RadialTable:
    def __init__(self, tab: IonTable, probes=PROBE_NAMES, r_min=0.3, r_max=12.0, h=0.01, pseudo_cutoff=PSEUDO_CUTOFF,
                 dtype=torch.float64):
        self.tab, self.probes = tab, tuple(probes)
        self.r_min, self.r_max, self.h = float(r_min), float(r_max), float(h)
        self.pseudo_cutoff = pseudo_cutoff
        G = int(round((r_max - r_min) / h)) + 1
        self.G = G
        self.r = torch.linspace(r_min, r_max, G, dtype=dtype)           # Angstrom
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        S = len(tab.symbols)
        rows = []
        for si in range(S):
            for sj in range(S):
                rows.append(self._functions(si, sj))
        F = len(self.names)
        vals = torch.stack([torch.stack([r[k] for k in range(F)]) for r in rows]).reshape(S, S, F, G, 3)
        # Hermite coefficients in grid units: y, h y', h^2 y''
        self.coef = torch.stack([vals[..., 0], vals[..., 1] * h, vals[..., 2] * h * h], -1).contiguous()
        self.n_funcs = F

    # ---- building -------------------------------------------------------------------------------------------
    def _functions(self, si, sj):
        tab = self.tab
        r = self.r.clone().requires_grad_(True)
        rb = r / BOHR
        G = self.G
        one = K._onehot0(G, rb)
        out: dict[str, torch.Tensor] = {}
        shells = [(tab.shell_b[sj, t], tab.shell_n[sj, t]) for t in range(tab.shell_n.shape[1]) if tab.shell_n[sj, t] > 0]
        b_outer = min((b for b, _ in shells), default=tab.shell_b[sj, 0])

        def complement(bp, wp):
            """Z_j f^nuc - sum_t n_t f^t: (NORD, G)."""
            c = tab.Z[sj] * shaped_one_center_multi(bp, wp, rb, 2)
            for bt, nt in shells:
                c = c - nt * shaped_two_center_damp(bp, bt.expand(G), wp, one, rb, 2)
            return c

        parents = {"dip": (tab.b_dip[si], tab.w_dip[si]), "quad": (tab.b_quad[si], tab.w_quad[si])}
        parents_j = {"dip": (tab.b_dip[sj], tab.w_dip[sj]), "quad": (tab.b_quad[sj], tab.w_quad[sj])}
        for p, (b, w) in parents.items():
            bp, wp = _expand(b, w, G)
            c = complement(bp, wp)
            for n in range(NORD):
                out[f"drive:{p}:{n}"] = c[n]
            bo = b_outer.expand(G)
            d = shaped_one_center_multi(bp, wp, rb, 2) - shaped_two_center_damp(bp, bo, wp, one, rb, 2)
            for n in range(NORD):
                out[f"ps:{p}:{n}"] = d[n]
            d_outer = d
            for t in range(tab.shell_n.shape[1]):
                if tab.shell_n[sj, t] > 0:
                    bt_ = tab.shell_b[sj, t].expand(G)
                    d = shaped_one_center_multi(bp, wp, rb, 2) - shaped_two_center_damp(bp, bt_, wp, one, rb, 2)
                else:
                    d = 0.0 * d_outer                          # keeps the graph to r (zero function, zero derivatives)
                for n in range(NORD):
                    out[f"ps{t}:{p}:{n}"] = d[n]
        for a, (ba, wa) in parents.items():
            bpa, wpa = _expand(ba, wa, G)
            for c_, (bc, wc) in parents_j.items():
                bpc, wpc = _expand(bc, wc, G)
                f = shaped_two_center_damp(bpa, bpc, wpa, wpc, rb, 2)
                for n in range(NORD):
                    out[f"aoff:{a[0]}{c_[0]}:{n}"] = f[n]
        for role in ("donor", "acceptor"):
            bf, wf = _expand(tab.front_b[role][si], tab.front_w[role][si], G)
            out[f"front:{role}:v"] = complement(bf, wf)[0]
            for p, (bc, wc) in parents_j.items():
                bpc, wpc = _expand(bc, wc, G)
                f = shaped_two_center_damp(bf, bpc, wf, wpc, rb, 2)
                for n in range(NORD):
                    out[f"front:{role}:{p}:{n}"] = f[n]
        probe_src = {"shell": None, "dip": parents["dip"], "quad": parents["quad"],
                     "donor": (tab.front_b["donor"][si], tab.front_w["donor"][si]),
                     "acceptor": (tab.front_b["acceptor"][si], tab.front_w["acceptor"][si]),
                     "breath": (tab.b_breath[si], tab.w_breath[si])}
        shells_i = [tab.shell_b[si, t] for t in range(tab.shell_n.shape[1]) if tab.shell_n[si, t] > 0]
        b_outer_i = min(shells_i, default=tab.shell_b[si, 0])
        one_s = torch.zeros(4, dtype=r.dtype)
        one_s[0] = 1.0
        for name in self.probes:
            b, w = probe_src[name] if probe_src[name] is not None else (b_outer_i, one_s)
            bp, wp = _expand(b, w, G)
            c = complement(bp, wp)
            for n in range(NORD):
                out[f"envV:{name}:{n}"] = c[n]
            S0 = S1 = S2 = torch.zeros_like(rb)
            for bt, nt in shells:
                a_, da, d2a = shaped_overlap_derivs(bp, wp, bt.expand(G), one, rb)
                S0, S1, S2 = S0 + nt * a_, S1 + nt * da, S2 + nt * d2a
            out[f"envS:{name}:0"], out[f"envS:{name}:1"], out[f"envS:{name}:2"] = S0, S1, S2
        if not self.names:
            self.names = list(out)
            self.index = {k: n for n, k in enumerate(self.names)}
        res = []
        for k in self.names:
            f = out[k]
            d1 = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
            d2 = torch.autograd.grad(d1.sum(), r, retain_graph=True)[0]
            res.append(torch.stack([f, d1, d2], -1).detach())
        return res

    # ---- evaluation ------------------------------------------------------------------------------------------
    def to(self, device):
        self.coef = self.coef.to(device)
        self.r = self.r.to(device)
        return self

    @property
    def device(self):
        return self.coef.device

    def __call__(self, si, sj, r_ang):
        """All functions at r (P,) Angstrom for species pairs (si, sj) (P,): (P, F). Quintic Hermite, zero past r_max,
        clamped below r_min.

        Vectorized: both grid nodes' coefficients in one gather, the six Hermite basis functions as
        [1, t, .., t^5] @ M, and one contraction -- ~15 kernels; equal to the transcription ``_call_ref``."""
        x = (r_ang - self.r_min) / self.h
        G = self.G
        inside = (x < G - 1) & (x >= 0)
        xc = x.clamp(0.0, G - 1 - 1e-9)
        k = xc.floor().long()
        t = xc - k.to(xc.dtype)
        kk = torch.stack([k, k + 1], -1)                                    # (P, 2): the interval's two nodes
        c = self.coef[si[:, None], sj[:, None], :, kk]                      # (P, 2, F, 3): node, function, (y, hy', h^2y'')
        t2 = t * t
        t3 = t2 * t
        tp = torch.stack([torch.ones_like(t), t, t2, t3, t2 * t2, t3 * t2], -1)     # products, not pow: finite at t = 0
        H = (tp @ _hermite_matrix(tp.dtype, tp.device)).reshape(-1, 2, 3)   # (P, node, derivative order)
        y = torch.einsum("pnfd,pnd->pf", c, H.to(c.dtype))
        return y * inside.to(y.dtype)[:, None]

    def _call_ref(self, si, sj, r_ang):
        """The per-term transcription of :meth:`__call__`, kept for the tests."""
        x = (r_ang - self.r_min) / self.h
        G = self.G
        inside = (x < G - 1) & (x >= 0)
        xc = x.clamp(0.0, G - 1 - 1e-9)
        k = xc.floor().long()
        t = xc - k.to(xc.dtype)
        c0 = self.coef[si, sj, :, k]                 # (P, F, 3)   -- advanced indexing puts P first
        c1 = self.coef[si, sj, :, k + 1]
        t2, t3 = t * t, t * t * t
        t4, t5 = t3 * t, t3 * t * t
        h0 = 1 - 10 * t3 + 15 * t4 - 6 * t5
        h1 = 10 * t3 - 15 * t4 + 6 * t5
        h2 = t - 6 * t3 + 8 * t4 - 3 * t5
        h3 = -4 * t3 + 7 * t4 - 3 * t5
        h4 = 0.5 * (t2 - 3 * t3 + 3 * t4 - t5)
        h5 = 0.5 * (t3 - 2 * t4 + t5)
        y = (c0[..., 0] * h0[:, None] + c1[..., 0] * h1[:, None] + c0[..., 1] * h2[:, None] + c1[..., 1] * h3[:, None]
             + c0[..., 2] * h4[:, None] + c1[..., 2] * h5[:, None])
        return y * inside.to(y.dtype)[:, None]

    def rows(self, y, prefixes):
        """(K, NORD, P): the damping rows ``prefix:0..4`` of an evaluation (P, F) for several prefixes at once."""
        return y[:, self.cols(tuple(f"{p}:{n}" for p in prefixes for n in range(NORD)), y.device).reshape(
            len(prefixes), NORD)].permute(1, 2, 0)

    def cols(self, names, device):
        """The column indices of ``names`` (a tuple) as a tensor on ``device``, made once (no copy per call)."""
        memo = self.__dict__.setdefault("_cols", {})
        key = (names, str(device))
        if key not in memo:
            memo[key] = torch.tensor([self.index[n] for n in names], device=device)
        return memo[key]


class FluxTable(RadialTable):
    """The rows the charge-flux CT needs (``CompleteModel(ct_model="flux")``): per ordered species pair (s_i, s_j),

        aoff:dd:{n}   f_n between i's dipole parent and j's dipole parent (the charge slot's shape)
        aoff:qd:{n}   f_n between i's quadrupole parent and j's dipole parent

    the same functions as the :class:`RadialTable` rows of those names, without the probes and frontier rows."""

    def __init__(self, tab: IonTable, r_max=12.0, **kw):
        with torch.enable_grad():           # built lazily by the model, possibly under no_grad (the r-derivatives)
            super().__init__(tab, probes=(), r_max=r_max, **kw)

    def _functions(self, si, sj):
        tab = self.tab
        r = self.r.clone().requires_grad_(True)
        rb = r / BOHR
        G = self.G
        bd, wd = _expand(tab.b_dip[sj], tab.w_dip[sj], G)
        out = {}
        for a, (ba, wa) in (("d", (tab.b_dip[si], tab.w_dip[si])), ("q", (tab.b_quad[si], tab.w_quad[si]))):
            bpa, wpa = _expand(ba, wa, G)
            f = shaped_two_center_damp(bpa, bd, wpa, wd, rb, 2)
            for n in range(NORD):
                out[f"aoff:{a}d:{n}"] = f[n]
        if not self.names:
            self.names = list(out)
            self.index = {k: n for n, k in enumerate(self.names)}
        res = []
        for k in self.names:
            f = out[k]
            d1 = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
            d2 = torch.autograd.grad(d1.sum(), r, retain_graph=True)[0]
            res.append(torch.stack([f, d1, d2], -1).detach())
        return res


class ProbeOverlapTable(RadialTable):
    """The gate of the overlap-switched environment fields (``CompleteModel(env_ct_gated=True)``): per ordered species
    pair (s_i, s_j)

        ov:{probe}    the overlap of i's probe density with j's frozen electron density, sum_t n_t <g_i | 1s_t(j)>

    (the ``envS:{probe}:0`` row of :class:`RadialTable`) for the diffuse probes donor, acceptor and breath."""

    PROBES = ("donor", "acceptor", "breath")

    def __init__(self, tab: IonTable, r_max=12.0, h=0.02, **kw):
        with torch.enable_grad():           # built lazily by the model, possibly under no_grad (the r-derivatives)
            super().__init__(tab, probes=(), r_max=r_max, h=h, **kw)

    def _functions(self, si, sj):
        tab = self.tab
        r = self.r.clone().requires_grad_(True)
        rb = r / BOHR
        G = self.G
        one = K._onehot0(G, rb)
        shells = [(tab.shell_b[sj, t], tab.shell_n[sj, t]) for t in range(tab.shell_n.shape[1]) if tab.shell_n[sj, t] > 0]
        src = {"donor": (tab.front_b["donor"][si], tab.front_w["donor"][si]),
               "acceptor": (tab.front_b["acceptor"][si], tab.front_w["acceptor"][si]),
               "breath": (tab.b_breath[si], tab.w_breath[si])}
        out = {}
        for name in self.PROBES:
            bp, wp = _expand(*src[name], G)
            S0 = 0.0 * rb
            for bt, nt in shells:
                S0 = S0 + nt * shaped_overlap_derivs(bp, wp, bt.expand(G), one, rb)[0]
            out[f"ov:{name}"] = S0
        if not self.names:
            self.names = list(out)
            self.index = {k: n for n, k in enumerate(self.names)}
        res = []
        for k in self.names:
            f = out[k]
            d1 = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
            d2 = torch.autograd.grad(d1.sum(), r, retain_graph=True)[0]
            res.append(torch.stack([f, d1, d2], -1).detach())
        return res


def shape_self_coulomb(tab: IonTable, r0: float = 1e-3) -> torch.Tensor:
    """(S,) the Coulomb self-energy (Hartree) of each species' dipole-parent shape (unit charge): the r -> 0 limit
    of (1 - f_0(r)) / r between two copies of it (converged to ~1e-9 at r0 = 1e-3 bohr)."""
    out = []
    rb = torch.tensor([r0], dtype=torch.float64)
    for s in range(len(tab.symbols)):
        b, w = _expand(tab.b_dip[s], tab.w_dip[s], 1)
        f = shaped_two_center_damp(b, b, w, w, rb, 0)[0]
        out.append(((1 - f) / rb)[0])
    return torch.stack(out).detach()


def flux_pair_terms(table: FluxTable, s, dp, dr):
    """For directed pairs dp = (i <- j) (2, P), species s (N,) and dr = r_j - r_i (P, 3) bohr: the drive on i's
    site variables (P, 8) of a unit charge in j's dipole-parent shape (dipole on i's dipole parent, quadrupole on its
    quadrupole parent: the energy is Q_j K . z_i, as the permanent drive L) and the charge-charge Coulomb (P,) of the
    two dipole-parent shapes."""
    i, j = dp
    r = dr.norm(dim=-1)
    y = table(s[i], s[j], r * BOHR)
    lam = 1 - table.rows(y, ["aoff:dd", "aoff:qd"])                     # (2, NORD, P)
    T0 = _basis_tensors(dr)[:, :, 0, :]                                # (NORD, P, 10): row 0, j as a charge
    R = torch.einsum("knp,npb->kpb", lam, T0)                          # (2, P, 10)
    Rp = R @ K._poly_map(dr.dtype, dr.device)                          # (2, P, 8)
    return torch.cat([Rp[0, :, :3], Rp[1, :, 3:]], 1), R[0, :, 0]


def _sph_y(n, x):
    """y_n(x) = j_n(x) / x^n, the spherical Bessel function over x^n (n <= 6), smooth at x -> 0: the series below
    x = 4, the upward recurrence from j0, j1 above (unstable only for x < n)."""
    small = x < 4.0
    xs = torch.where(small, torch.full_like(x, 4.0), x)
    s, c = torch.sin(xs), torch.cos(xs)
    jm, j = s / xs, (s / xs - c) / xs
    for m in range(1, n):
        jm, j = j, (2 * m + 1) / xs * j - jm
    big = (jm if n == 0 else j) / xs ** n
    xz = torch.where(small, x, torch.zeros_like(x))
    t = -0.5 * xz * xz
    term = torch.ones_like(x) / math.prod(range(1, 2 * n + 2, 2))
    ser = term
    for m in range(1, 40):
        term = term * t / (m * (2 * n + 2 * m + 1))
        ser = ser + term
    return torch.where(small, ser, big)


class XpolTable(RadialTable):
    """Radial table of the quadratic exchange-polarization functional (``CompleteModel(xpol=True)``). Per ordered
    species pair (i <- j), with rho_i i's electron density (its permanent shells, sum_t n_t 1s_t):

        xp:{dip,quad}:{n}   lam_n(r) = r^(2n+1) / (2n-1)!! (1 / 2 pi^2) int p(k) g_q(k) k^(2n+2) j_n(kr) / (kr)^n dk
                            -- the damping rows of O_q(r) = <rho_i | g_q,j(. - r)>, so that sum_n lam_n T_n is its
                            derivative tensor; contracted with j's induced moments (row 0) it is the overlap of rho_i
                            with j's induced density (:func:`xpol_overlap_map`)
        xp:breath           <rho_i | g_breath,j>: the overlap per unit l = 0 (breathing) amplitude of j
        xp:ovl              <rho_i | rho_j>: the frozen density-density overlap of the two ions

    All on the k grid and density transforms of the environment overlaps, so the ratios are on one scale."""

    SHAPES = ("dip", "quad")

    def __init__(self, tab, r_max=10.0, **kw):
        super().__init__(tab, probes=("shell",), r_max=r_max, **kw)

    @torch.no_grad()
    def _functions(self, si, sj):
        tab = self.tab
        rb = self.r / BOHR
        k, wk = K._ovl_grid(rb.dtype, rb.device)
        one_s = torch.zeros(4, dtype=rb.dtype)
        one_s[0] = 1.0

        def ft(b, w):                       # (K,); a shape with no weight is zero
            f = K.shaped_density_ft(b.reshape(1, *b.shape), w.reshape(1, *w.shape), k)
            return f[0] if torch.is_tensor(f) else torch.zeros_like(k)

        def rho(s):                         # the ion's electron density, sum_t n_t (normalized 1s of shell t)
            out = torch.zeros_like(k)
            for t in range(tab.shell_n.shape[1]):
                if tab.shell_n[s, t] > 0:
                    out = out + tab.shell_n[s, t] * ft(tab.shell_b[s, t], one_s)
            return out

        fp = rho(si)
        x = k[None, :] * rb[:, None]
        ys = torch.stack([_sph_y(n, x) for n in range(NORD + 2)])                 # (NORD + 2, G, K)
        kp = torch.stack([k ** (2 * n + 2) for n in range(NORD + 2)])            # (NORD + 2, K)
        pre = torch.stack([rb ** (2 * n + 1) / math.prod(range(1, 2 * n, 2)) for n in range(NORD + 2)])
        n_ = torch.arange(NORD + 1, dtype=rb.dtype)[:, None]
        out = {}
        for q, (bq, wq) in {"dip": (tab.b_dip[sj], tab.w_dip[sj]), "quad": (tab.b_quad[sj], tab.w_quad[sj])}.items():
            F = fp * ft(bq, wq) * wk / (2 * math.pi ** 2)
            lam = pre * torch.einsum("ngk,nk->ng", ys, kp * F[None])                # (NORD + 2, G)
            # lam_n' = (2n+1)(lam_n - lam_n+1) / r (from g_n+1 = g_n' / r), lam_n'' from the same twice
            d1 = (2 * n_ + 1) * (lam[:-1] - lam[1:]) / rb
            d2 = (2 * n_[:-1] + 1) * ((d1[:-1] - d1[1:]) / rb - (lam[:-2] - lam[1:-1]) / rb ** 2)
            for n in range(NORD):           # derivatives in r (Angstrom), like the rest of the table
                out[f"xp:{q}:{n}"] = torch.stack([lam[n], d1[n] / BOHR, d2[n] / BOHR ** 2], -1)

        def scalar(F):                      # O(r) = int F k^2 j0(kr) and its r-derivatives (Angstrom)
            O = (ys[0] * (k * k * F)[None]).sum(-1)
            O1 = -rb * (ys[1] * (k ** 4 * F)[None]).sum(-1)
            O2 = ((2 * ys[1] - ys[0]) * (k ** 4 * F)[None]).sum(-1)
            return torch.stack([O, O1 / BOHR, O2 / BOHR ** 2], -1)

        out["xp:breath"] = scalar(fp * ft(tab.b_breath[sj], tab.w_breath[sj]) * wk / (2 * math.pi ** 2))
        out["xp:ovl"] = scalar(fp * rho(sj) * wk / (2 * math.pi ** 2))
        if not self.names:
            self.names = list(out)
            self.index = {kk: m for m, kk in enumerate(self.names)}
        return [out[kk] for kk in self.names]


def xpol_overlap_map(table: XpolTable, s, dp, dr):
    """For directed pairs (i <- j), dr = X_j - X_i (bohr, autograd flows): the frozen overlap O_ij = <rho_i | rho_j>
    (P,), the linear map d_ij (P, 8) from j's induced moments [dipole, spherical quadrupole] to the change of that
    overlap (of the electron density: x is a charge-density moment, so the sign flips), and e_ij (P,) the change
    per unit breathing amplitude of j (an electron-density amplitude)."""
    i, j = dp
    r = dr.norm(dim=-1)
    y = table(s[i], s[j], r * BOHR)
    Tb = _basis_tensors(dr)                                                   # (NORD, P, 10, 10) [j, i]
    P = K._poly_map(dr.dtype, dr.device)
    lam = table.rows(y, ["xp:dip", "xp:quad"])                                # (2, NORD, P)
    Td = torch.einsum("np,npa->pa", lam[0], Tb[:, :, 1:4, 0])                 # (P, 3)
    Tq = torch.einsum("np,npa->pa", lam[1], Tb[:, :, 4:10, 0]) @ P[4:10, 3:8]  # (P, 5)
    cols = table.cols(("xp:ovl", "xp:breath"), y.device)
    return y[:, cols[0]], -torch.cat([Td, Tq], -1), y[:, cols[1]]


def _basis_tensors_ref(dr):
    """The five pure-order damped tensors T_n(dr) (NORD, P, 10, 10): T(dr, lam) = sum_n lam_n T_n (the damped tensor
    is linear in its damping rows), built in one call of the transcription on NORD replicas of the pairs."""
    P_ = dr.shape[0]
    rep = dr.repeat(NORD, 1)
    damp = dr.new_zeros(NORD, NORD * P_)
    for n in range(NORD):
        damp[n, n * P_:(n + 1) * P_] = 1.0
    return K._tensor(rep, damp).reshape(NORD, P_, 10, 10)


# exponents (a, b, c) of the 35 monomials x^a y^b z^c of degree <= 4
_MONO = torch.tensor([(a, b, d - a - b) for d in range(5) for a in range(d, -1, -1) for b in range(d - a, -1, -1)])
_COEF: dict = {}


def _monomials(dr):
    """(P, 35): the monomials of degree <= 4 of the components of dr."""
    key = ("mono", str(dr.device))
    if key not in _COEF:
        _COEF[key] = _MONO.to(dr.device)
    e = _COEF[key]
    pw = torch.cat([torch.ones_like(dr)[None], dr[None].expand(4, -1, -1)]).cumprod(0)      # (5, P, 3): dr^0..dr^4
    return (pw[e[:, 0], :, 0] * pw[e[:, 1], :, 1] * pw[e[:, 2], :, 2]).T


def _basis_coef(dtype, device):
    """C (NORD, 35, 100): T_n(dr) = r^-(2n+1) mono(dr) @ C[n]. The coefficients are the transcription's integers,
    recovered once by an exact fit of it at random points (checked to round to integers)."""
    key = (dtype, str(device))
    if key not in _COEF:
        dr = torch.randn(256, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        ex = 2 * torch.arange(NORD, dtype=torch.float64) + 1
        y = _basis_tensors_ref(dr).reshape(NORD, -1, 100) * dr.norm(dim=-1)[None, :, None] ** ex[:, None, None]
        C = torch.linalg.lstsq(_monomials(dr)[None].expand(NORD, -1, -1), y).solution
        Cr = C.round()
        if (C - Cr).abs().max() > 1e-6:
            raise RuntimeError("damped tensor coefficients are not integers: the transcription changed")
        _COEF[key] = Cr.to(dtype=dtype, device=device)
    return _COEF[key]


def _basis_tensors(dr):
    """The pure-order basis tensors: the fused contraction on the GPU, where the cost is the op count; the
    transcription on the CPU, which skips the zero entries the dense contraction computes and is faster for large
    batches there."""
    return _basis_tensors_ref(dr) if dr.device.type == "cpu" else _basis_tensors_fused(dr)


def _basis_tensors_fused(dr):
    """:func:`_basis_tensors_ref` as one contraction: the monomials of dr against the integer coefficients of each
    pure order, times r^-(2n+1) -- a handful of ops instead of the transcription's few hundred (and their double
    backward under force training)."""
    C = _basis_coef(dr.dtype, dr.device)
    ex = 2 * torch.arange(NORD, dtype=dr.dtype, device=dr.device) + 1
    rp = dr.norm(dim=-1)[None] ** -ex[:, None]                       # (NORD, P)
    T = (_monomials(dr) @ C) * rp[:, :, None]                        # (NORD, P, 100)
    return T.reshape(NORD, dr.shape[0], 10, 10)


def _luts(tab, chan):
    """Species-pair lookup tables: pseudo type index (S, S) and channel type index (S, S; -1 where the pair has no
    channel), to replace per-pair name building."""
    S = len(tab.symbols)
    ctypes, ptypes = channel_types(tab.symbols), pseudo_type_names(tab.symbols)
    ps = torch.full((S, S), -1, dtype=torch.long)
    ch = torch.full((S, S), -1, dtype=torch.long)
    for a in range(S):
        for b in range(S):
            nm = f"{tab.symbols[a]}<{tab.symbols[b]}"
            if nm in ptypes:
                ps[a, b] = ptypes.index(nm)
            nm = f"{tab.symbols[a]}>{tab.symbols[b]}"
            if nm in chan.types and nm in ctypes:
                ch[a, b] = ctypes.index(nm)
    return ps, ch


def batched_precompute(pos_ang, atomic_numbers, batch_idx, tab: IonTable, table: RadialTable, chan: CT.ChannelTable,
                       labels: dict | None = None, cell=None, cutoff: float = 9.0, switch_width: float = 1.0,
                       ewald_tol: float = 1e-10, dispersion_tail: bool = False, inv_form: str = "sq") -> dict:
    """The collated batch of the complete model straight from positions (N, 3) Angstrom (autograd flows through),
    atomic numbers (N,), frame index (N,) sorted by frame. ``labels``: a dict of per-frame tensors (B,) to attach.
    Runs on the device of ``pos_ang``; ``tab`` and ``table`` must live there too (``IonTable.to``, ``RadialTable.to``).

    Produces the keys ``collate`` makes that :class:`CompleteModel` reads: L, A_off, VD/KD/VA/KA (padded), mask,
    species, feats, frame, pos, channels (index, type, r, frame, ltaper), pseudo drive (index, type, r, basis), env.
    The damped tensors of every pair are built once as the five pure-order basis tensors and every term is a single
    contraction of table rows against them.

    ``cell`` (3, 3) or (B, 3, 3) Angstrom, rows the lattice vectors, makes the frames periodic (autograd flows
    through it too: the stress is the derivative under a strain of positions and cell). The pairs are then every
    pair and lattice image within ``cutoff`` (Angstrom), the table's short-range complements are switched off over
    the last ``switch_width``, and the bare Coulomb part of every long-range quantity is the Ewald sum
    (:mod:`rsfff.ff.complete.periodic`, splitting from ``cutoff`` and ``ewald_tol``): the net charge q_j of each
    pair term is screened by g_n(r), and the reciprocal part is added per atom. The induction operator is not
    assembled (``A_off`` and ``A_pp`` are None): its real-space pair blocks and the Ewald frames go into
    ``bt["pbc"]`` for the matrix-free solve, and the frontier potentials come as per-atom sums plus per-pair
    couplings instead of the dense (B, n, n) blocks. Dispersion is switched off over the same last
    ``switch_width``; ``dispersion_tail`` adds the uniform-density correction for the pairs beyond.

    ``inv_form``: the environment invariants' form ("sq" squared norms, "norm" magnitudes) -- the model's own
    (``CompleteModel.inv_form``), which callers pass through."""
    X = pos_ang / BOHR
    dev, dt = X.device, X.dtype
    N = X.shape[0]
    b_idx = torch.as_tensor(batch_idx, device=dev)
    B = int(b_idx.max()) + 1 if N else 0
    s = tab.species_index(torch.as_tensor(atomic_numbers, device=dev))
    counts = torch.bincount(b_idx, minlength=B)
    loc = torch.arange(N, device=dev) - (torch.cumsum(counts, 0) - counts)[b_idx]
    nmax = int(counts.max()) if N else 0
    P = K._poly_map(dt, dev)
    periodic = cell is not None
    if periodic:
        cell_ang = torch.as_tensor(cell, dtype=dt, device=dev)
        cell_ang = cell_ang.expand(B, 3, 3) if cell_ang.dim() == 2 else cell_ang
        rc = float(cutoff)
        if rc - switch_width < max(chan.cutoff, table.pseudo_cutoff):
            raise ValueError(f"cutoff - switch_width = {rc - switch_width} A is inside the CT / pseudo-drive range")
        pairs, shifts = periodic_pairs(pos_ang, cell_ang, b_idx, rc)
        Cb = cell_ang / BOHR
        dr_u = X[pairs[1]] - X[pairs[0]] + torch.einsum("pk,pkl->pl", shifts, Cb[b_idx[pairs[0]]])
    else:
        pairs = K.all_pairs(b_idx)
        dr_u = X[pairs[1]] - X[pairs[0]]
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)                  # directed (i <- j)
    i, j = dp
    dr = torch.cat([dr_u, -dr_u])
    r = dr.norm(dim=-1)
    r_ang = r * BOHR
    rhat = dr / r[:, None]
    y = table(s[i], s[j], r_ang)                                     # (Pd, F)
    qj = tab.charge[s[j]]
    if periodic:
        y = y * smooth_switch(r_ang, rc - switch_width, rc)[:, None]
        alpha = ewald_alpha(rc / BOHR, ewald_tol)
        g = erfc_screening(r, alpha)                                 # (NORD, Pd): the real-space Ewald screening
        frames = make_ewald_frames(X, Cb, b_idx, alpha, 2.2 * alpha * math.sqrt(-math.log(ewald_tol)))
        Mq = X.new_zeros(N, 10)
        Mq[:, 0] = tab.charge[s]
        out_q = recip_apply(frames, Mq)                              # (N, 10): reciprocal + self + background
        rq = out_q @ P                                               # its drive on the site variables
    else:
        g = torch.ones(1, r.shape[0], dtype=dt, device=dev)          # bare tensors
    qg = qj[None] * g                                                # (NORD or 1, Pd)
    Tb = _basis_tensors(dr)                                          # (NORD, Pd, 10, 10)
    T0 = Tb[:, :, 0, :]                                              # (NORD, Pd, 10): row 0, the source as a charge
    Tc = Tb[:, :, :, 0]                                              # column 0: the probe as a unit charge
    probes = list(table.probes)
    # all drive-type rows at once: 2 parents + 6 probes + 2 pseudo (complements c; lam = q_j - c, or d for ps)
    T_sh = tab.shell_n.shape[1]
    drive_keys = (["drive:dip", "drive:quad"] + [f"envV:{p}" for p in probes] + ["ps:dip", "ps:quad"]
                  + [f"ps{t}:{p}" for t in range(T_sh) for p in ("dip", "quad")])
    rows = table.rows(y, drive_keys)                                 # (K, NORD, Pd)
    nq = 2 + len(probes)
    lam = torch.cat([qg[None] - rows[:nq], rows[nq:]], 0)
    R0 = torch.einsum("knp,npb->kpb", lam, T0)                       # (K, Pd, 10)
    Rp = R0 @ P                                                      # (K, Pd, 8)
    # -- induction drive ------------------------------------------------------------------------------------------
    Lpair = torch.cat([Rp[0, :, :3], Rp[1, :, 3:]], 1)
    Lflat = X.new_zeros(N, 8).index_add(0, i, Lpair)
    if periodic:
        Lflat = Lflat + rq
    L = X.new_zeros(B, nmax, 8).index_put((b_idx, loc), Lflat)
    # -- induced-induced coupling (undirected pairs: the first half of dp) --------------------------------------
    nP = pairs.shape[0] and pairs.shape[1]
    ii, jj = pairs
    PTP = torch.einsum("ab,npbc,cd->npad", P.T, Tb[:, :nP], P)       # (NORD, nP, 8, 8) basis blocks
    arows = table.rows(y[:nP], ["aoff:dd", "aoff:dq", "aoff:qd", "aoff:qq"])     # (4, NORD, nP)
    blocks = torch.einsum("knp,npab->kpab", g[:, :nP] - arows, PTP)  # (4, nP, 8, 8): [a c] = dd, dq, qd, qq
    sl = {"d": slice(0, 3), "q": slice(3, 8)}
    Bij = X.new_zeros(nP, 8, 8)
    for k_, (a, c) in enumerate((("d", "d"), ("d", "q"), ("q", "d"), ("q", "q"))):
        Bij[:, sl[c], sl[a]] = blocks[k_][:, sl[c], sl[a]]
    # the penetration part of the same coupling (f rows): the induced-induced Pauli kernel before its constants
    blocks_f = torch.einsum("knp,npab->kpab", arows, PTP)
    Bf = X.new_zeros(nP, 8, 8)
    for k_, (a, c) in enumerate((("d", "d"), ("d", "q"), ("q", "d"), ("q", "q"))):
        Bf[:, sl[c], sl[a]] = blocks_f[k_][:, sl[c], sl[a]]
    if periodic:                      # not assembled: the pair blocks go to the matrix-free solve
        A_off = A_pp = None
    else:
        ar = torch.arange(8, device=dev)
        A = X.new_zeros(B, nmax, 8, nmax, 8)
        fb = b_idx[ii][:, None, None]
        A = A.index_put((fb, loc[jj][:, None, None], ar[None, :, None], loc[ii][:, None, None], ar[None, None, :]), Bij, accumulate=True)
        A = A.index_put((fb, loc[ii][:, None, None], ar[None, :, None], loc[jj][:, None, None], ar[None, None, :]),
                        Bij.transpose(1, 2), accumulate=True)
        A_off = A.reshape(B, 8 * nmax, 8 * nmax)
        Ap = X.new_zeros(B, nmax, 8, nmax, 8)
        Ap = Ap.index_put((fb, loc[jj][:, None, None], ar[None, :, None], loc[ii][:, None, None], ar[None, None, :]), Bf, accumulate=True)
        Ap = Ap.index_put((fb, loc[ii][:, None, None], ar[None, :, None], loc[jj][:, None, None], ar[None, None, :]),
                          Bf.transpose(1, 2), accumulate=True)
        A_pp = Ap.reshape(B, 8 * nmax, 8 * nmax)
    # -- frontier potentials --------------------------------------------------------------------------------------
    fkeys = ["front:donor:dip", "front:donor:quad", "front:acceptor:dip", "front:acceptor:quad"]
    frows = table.rows(y, fkeys)                                      # (4, NORD, Pd)
    KC = torch.einsum("knp,npa->kpa", g - frows, Tc) @ P             # (4, Pd, 8)
    front = {}
    for q_, (role, key) in enumerate((("donor", "D"), ("acceptor", "A"))):
        v = (qg[0] - y[:, table.index[f"front:{role}:v"]]) / r
        kv = torch.cat([KC[2 * q_][:, :3], KC[2 * q_ + 1][:, 3:]], 1)
        if periodic:                  # per-atom permanent part (real + reciprocal) and the per-pair induced coupling
            front[f"phi{key}0"] = X.new_zeros(N).index_add(0, i, v) + out_q[:, 0]
            front[f"k{key}"] = kv
        else:
            front[f"V{key}"] = X.new_zeros(B, nmax, nmax).index_put((b_idx[i], loc[i], loc[j]), v)
            front[f"K{key}"] = X.new_zeros(B, nmax, nmax, 8).index_put((b_idx[i], loc[i], loc[j]), kv)
    # -- environment features (all probes at once) ---------------------------------------------------------------
    nprobe = len(probes)
    D0 = R0[2:2 + nprobe]                                             # (nprobe, Pd, 10)
    Dp = Rp[2:2 + nprobe]
    v0 = (D0[:, :, 0] - (qg[0] / r)[None]).T                          # (Pd, nprobe)
    V0 = X.new_zeros(N, nprobe).index_add(0, i, v0)
    V1 = X.new_zeros(N, nprobe, 3).index_add(0, i, -Dp[:, :, :3].permute(1, 0, 2))
    V2 = X.new_zeros(N, nprobe, 5).index_add(0, i, Dp[:, :, 3:].permute(1, 0, 2))
    # the penetration part alone (the complements, no point charges): short-ranged, the same in clusters and under PBC
    Dpen = torch.einsum("knp,npb->kpb", -rows[2:2 + nprobe], T0) @ P                    # (nprobe, Pd, 8)
    V1pen = X.new_zeros(N, nprobe, 3).index_add(0, i, -Dpen[:, :, :3].permute(1, 0, 2))
    V2pen = X.new_zeros(N, nprobe, 5).index_add(0, i, Dpen[:, :, 3:].permute(1, 0, 2))
    if periodic:                                                      # the reciprocal part of the point-charge field
        V1 = V1 - rq[:, None, :3]
        V2 = V2 + rq[:, None, 3:]
    sidx = table.cols(tuple(f"envS:{p_}:{k}" for p_ in probes for k in range(3)), dev).reshape(nprobe, 3)
    Sy = y[:, sidx]                                                 # (Pd, nprobe, 3)
    S0p, S1p, S2p = Sy[..., 0], Sy[..., 1], Sy[..., 2]
    outer = rhat[:, :, None] * rhat[:, None, :] - torch.eye(3, dtype=dt, device=dev)[None] / 3
    S0 = X.new_zeros(N, nprobe).index_add(0, i, S0p)
    S1 = X.new_zeros(N, nprobe, 3).index_add(0, i, -S1p[:, :, None] * rhat[:, None, :])
    S2 = X.new_zeros(N, nprobe, 3, 3).index_add(0, i, (S2p - S1p / r[:, None])[:, :, None, None] * outer[:, None])
    if "breath" in probes:
        kb = probes.index("breath")
        has = (tab.a0[s] > 0).to(dt)
        V0[:, kb], S0[:, kb] = V0[:, kb] * has, S0[:, kb] * has
        V1[:, kb], V2[:, kb], S1[:, kb] = V1[:, kb] * has[:, None], V2[:, kb] * has[:, None], S1[:, kb] * has[:, None]
        V1pen[:, kb], V2pen[:, kb] = V1pen[:, kb] * has[:, None], V2pen[:, kb] * has[:, None]
        S2[:, kb] = S2[:, kb] * has[:, None, None]
    inv = invariants(V0, V1, V2, S0, S1, S2, inv_form)               # (N, nprobe, 7)
    feats = inv.reshape(N, -1)                                        # probe-major, like DensityEnvironment
    env = {}
    for k_, name in enumerate(probes):
        for key, val in (("V0", V0), ("V1", V1), ("V2", V2), ("S0", S0), ("S1", S1), ("S2", S2), ("V1pen", V1pen),
                         ("V2pen", V2pen)):
            env[f"{name}:{key}"] = val[:, k_]
    # -- pseudo-potential drive basis -----------------------------------------------------------------------------
    keep = r_ang < table.pseudo_cutoff
    memo = table.__dict__.setdefault("_luts", {})
    if str(dev) not in memo:
        memo[str(dev)] = tuple(t.to(dev) for t in _luts(tab, chan))
    ps_lut, ch_lut = memo[str(dev)]
    ps_basis = torch.cat([Rp[nq][:, :3], Rp[nq + 1][:, 3:]], 1)[keep]
    ps_basis_sh = torch.stack([torch.cat([Rp[nq + 2 + 2 * t][:, :3], Rp[nq + 3 + 2 * t][:, 3:]], 1) for t in range(T_sh)], 1)[keep]
    ps_type = ps_lut[s[i], s[j]][keep]
    # -- channels (direct CT) ----------------------------------------------------------------------------------------
    ct = ch_lut[s[i], s[j]]
    ck = (ct >= 0) & (r_ang < chan.cutoff)
    ch_index = dp[:, ck]
    ch_r = r_ang[ck]
    ch_ltaper = torch.log(CT._taper(ch_r, chan.cutoff, chan.taper).clamp(min=1e-300))
    pbc = None
    if periodic:
        # the frontier potentials of each channel's own partner (that image only, bare), removed by dphi_pairs
        pD = ck.nonzero().squeeze(1)
        pA = torch.where(pD < nP, pD + nP, pD - nP)                   # the reverse pair (j <- i)
        cb = {}
        for q_, (role, key, pp) in enumerate((("donor", "D", pD), ("acceptor", "A", pA))):
            kb = torch.einsum("knp,npa->kpa", 1 - frows[2 * q_:2 * q_ + 2][:, :, pp], Tc[:, pp]) @ P
            cb[f"ch_v{key}"] = (qj[pp] - y[pp, table.index[f"front:{role}:v"]]) / r[pp]
            cb[f"ch_k{key}"] = torch.cat([kb[0][:, :3], kb[1][:, 3:]], 1)
        pbc = {"cell": cell_ang, "volume": torch.linalg.det(Cb).abs(), "cutoff": rc, "switch_width": switch_width,
               "alpha": alpha, "frames": frames, "A_pairs": pairs, "A_blocks": Bij, "App_blocks": Bf, "dp": dp,
               "e_elst_recip": 0.5 * X.new_zeros(B).index_add(0, b_idx, Mq[:, 0] * out_q[:, 0]),
               "g0_pairs": g[0, :nP],
               "disp_tail": dispersion_tail_integral(rc, switch_width) if dispersion_tail else None, **front, **cb}
    mask = torch.zeros(B, nmax, dtype=torch.bool, device=dev).index_put((b_idx, loc), torch.ones(N, dtype=torch.bool, device=dev))
    lab = dict(labels or {})
    lab.setdefault("n", counts.to(dt))
    lab.setdefault("ct_mask", torch.ones(B, dtype=torch.bool, device=dev))
    nT = len(channel_types(tab.symbols))
    lab.setdefault("chan", torch.full((B, nT), float("nan"), dtype=dt, device=dev))
    for key in ("pol", "ct", "pauli", "disp", "elec", "int"):
        lab.setdefault(key, torch.full((B,), float("nan"), dtype=dt, device=dev))
    return {"B": B, "nmax": nmax, "L": L, "A_off": A_off, "A_pp": A_pp, "VD": front.get("VD"), "KD": front.get("KD"),
            "VA": front.get("VA"), "KA": front.get("KA"), "pbc": pbc, "pairs": pairs, "pair_dr": dr_u * BOHR, "mask": mask, "species": s, "feats": feats, "frame": b_idx, "pos": pos_ang,
            "ch_index": ch_index, "ch_type": ct[ck], "ch_r": ch_r, "ch_frame": b_idx[ch_index[0]],
            "ch_ltaper": ch_ltaper, "ps_index": dp[:, keep], "ps_type": ps_type, "ps_r": r_ang[keep],
            "ps_basis": ps_basis, "ps_basis_sh": ps_basis_sh, "env": env, "labels": lab}
