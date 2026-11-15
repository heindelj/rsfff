"""Tabulated radial kernels and the batched, differentiable precompute (complete model plan: force training).

Everything the per-frame precompute builds is a sum over atom pairs of a *radial* function of r for the ordered
species pair (s_i, s_j) times an angular tensor in r_hat. The radial functions are the shaped-density damping
complements and overlaps of the fixed parameter table (closed forms and k-space quadrature, the whole cost); the
angular parts are polynomials. So the radial functions are tabulated once per species pair on a fine grid, with
their first two derivatives, and evaluated by quintic Hermite interpolation -- exact in the sense that the force is
the derivative of the interpolated energy, and smooth (C^2). A species pair is built the first time it is needed and
kept under a key of the two ions' parameter values (:meth:`RadialTable.pair_key`), in memory, on disk and in a
model's checkpoint. The learnable exponents (direct CT, Pauli, dispersion)
are not in the table: those kernels are closed-form Slater 1s and stay analytic in the model.

:class:`RadialTable` holds, per ordered species pair, the functions

    drive:{dip,quad}:{n}      c(r) = Z_j f^nuc_n - sum_t n_t f^t_n  (complement of the permanent drive; lam = q_j - c)
    aoff:{dd,dq,qd,qq}:{n}    f_n between parent a of i and parent c of j (lam = 1 - f; dd, qd are also the CT
                              charge slots' couplings)
    envV:{probe}:{n}          drive-type complement for each diffuse probe of i (donor, acceptor, breath)
    envS:{probe}:{0,1}        overlap of probe i with j's shells and its first radial derivative

with n = 0..4 the five damping orders of the rank-2 tensor; :class:`CmmTable` adds the exchange-polarization rows of
:class:`XpolTable` and the Gram rows of :class:`rsfff.ff.cmm.gram.GramTable` on the same grid, so one lookup per
pair serves the whole precompute. All of them vanish beyond the overlap range, so the table ends at ``r_max`` and
returns zero past it.

:func:`batched_precompute` assembles from positions (differentiable) everything of a batch that depends on the
geometry alone -- the induction drive and operator, the encoder's channels and gated probe invariants, the exchange
pairs' overlap derivatives and Gram blocks, the CT charge slots' couplings, the channel list, the frozen
electrostatics -- for :class:`rsfff.ff.cmm.CmmModel`, whose forward then only does what depends on its parameters.
"""
from __future__ import annotations

import math

import torch

from . import kernels as K
from .params import IonTable
from ..multipole import shaped_one_center_multi, shaped_two_center_damp
from .environment import shaped_overlap_derivs

__all__ = ["RadialTable", "XpolTable", "shape_self_coulomb", "species_params", "table_cache_dir", "PROBES", "GATE",
           "R_XPOL", "NORD", "BOHR", "TABLE_FIELDS"]

#: the diffuse probes whose environment the encoder reads (gated invariants) and whose l = 0 shape is the breathing
#: response (breath)
PROBES = ("donor", "acceptor", "breath")
#: the probe overlap at which a neighbour's point-charge field starts to count and counts fully (smoothstep in log10)
GATE = (1e-7, 1e-5)
#: exchange pairs and environment channels end here (the frozen overlaps are < 1e-9 beyond)
R_XPOL = 10.0
#: the overlap window over which a pair's exchange ratios and channel weights switch on (above the table's noise)
O_ON, O_FULL = 1e-7, 1e-6

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


#: the ion-table fields the tabulated integrals read (per species); a pair's table is a function of these alone
TABLE_FIELDS = ("Z", "shell_n", "shell_b", "b_dip", "w_dip", "b_quad", "w_quad", "b_breath", "w_breath", "a0")


def species_params(tab: IonTable, s: int) -> bytes:
    """The values the tables depend on for species s (not its name): :data:`TABLE_FIELDS` and the frontier shapes,
    shells with zero occupation dropped (so the padding of a table with more shells does not matter)."""
    occ = tab.shell_n[s] > 0
    vals = []
    for f in TABLE_FIELDS:
        v = getattr(tab, f, None)
        if v is None:
            continue
        v = v[s]
        if f in ("shell_n", "shell_b"):
            v = v[occ]
        vals.append((f, v))
    for role in sorted(tab.front_b):
        vals += [(f"front_b.{role}", tab.front_b[role][s]), (f"front_w.{role}", tab.front_w[role][s])]
    out = b""
    for name, v in vals:
        v = torch.as_tensor(v).detach().to("cpu", torch.float64).reshape(-1)
        out += name.encode() + b":" + str(v.numel()).encode() + b":" + v.numpy().tobytes()
    return out


def table_cache_dir():
    """The directory of the per-pair disk cache: ``RSFFF_TABLE_CACHE`` (empty or "off": none), else
    ~/.cache/rsfff/tables."""
    import os
    from pathlib import Path
    d = os.environ.get("RSFFF_TABLE_CACHE")
    if d is None:
        return Path.home() / ".cache" / "rsfff" / "tables"
    return None if d in ("", "off") else Path(d)


class RadialTable:
    """The precompute's radial rows (module docstring) on the grid r_min..r_max (Angstrom, step h), quintic Hermite
    in r. Subclasses override :meth:`_names` and :meth:`_functions` (the rows of one ordered species pair) and reuse
    the grid, the storage and the evaluation.

    Pairs are built when first needed (a lookup or :meth:`ensure_species`) and identified by :meth:`pair_key`: a
    hash of the class, its :attr:`VERSION`, the grid and options and the two species' parameter values (not their
    names). A built pair is taken, in order, from ``store`` (a dict key -> coefficients, e.g. shipped with a model),
    from the disk cache (:func:`table_cache_dir`), or computed and then written to both. A key encodes everything
    the integrals depend on, so a stored pair can never be applied to a different pair of ions; bump
    :attr:`VERSION` whenever the integrals themselves change."""

    VERSION = 1

    def __init__(self, tab: IonTable, probes=PROBES, r_min=0.3, r_max=12.0, h=0.01, dtype=torch.float64, store=None):
        self.tab, self.probes = tab.to("cpu"), tuple(probes)
        self.r_min, self.r_max, self.h = float(r_min), float(r_max), float(h)
        G = int(round((r_max - r_min) / h)) + 1
        self.G = G
        self.r = torch.linspace(r_min, r_max, G, dtype=dtype)           # Angstrom
        self.names = list(self._names())
        self.index = {k: n for n, k in enumerate(self.names)}
        self.n_funcs = F = len(self.names)
        S = len(tab.symbols)
        self.coef = torch.zeros(S, S, F, G, 3, dtype=dtype)               # Hermite coefficients: y, h y', h^2 y''
        self.built = torch.zeros(S, S, dtype=torch.bool)
        self.store = store if store is not None else {}
        self._spkey = {}

    # ---- identity, storage -----------------------------------------------------------------------------------
    def _key_extra(self):
        return repr(self.probes)

    def pair_key(self, si: int, sj: int) -> str:
        import hashlib
        for s in (si, sj):
            if s not in self._spkey:
                self._spkey[s] = species_params(self.tab, s)
        h = hashlib.sha256()
        h.update(f"{type(self).__name__}|v{self.VERSION}|{self.r_min!r}|{self.r_max!r}|{self.h!r}|{self.G}|"
                 f"{self.coef.dtype}|{self._key_extra()}|".encode())
        h.update(self._spkey[si] + b"||" + self._spkey[sj])
        return h.hexdigest()[:32]

    def ensure_species(self, species):
        """Build (or fetch) every ordered pair among the species indices ``species``."""
        sp = sorted({int(s) for s in torch.as_tensor(species).reshape(-1).tolist()})
        for a in sp:
            for b in sp:
                if not bool(self.built[a, b]):
                    self._fill(a, b)

    def ensure(self, si, sj):
        """Build (or fetch) the ordered pairs (si, sj) of a lookup."""
        if bool(self.built.all()):
            return
        S = self.built.shape[0]
        codes = torch.unique(si.reshape(-1) * S + sj.reshape(-1)).cpu().tolist()
        for c in codes:
            a, b = divmod(int(c), S)
            if not bool(self.built[a, b]):
                self._fill(a, b)

    def build_all(self):
        self.ensure_species(range(self.built.shape[0]))
        return self

    def built_entries(self) -> dict:
        """{key: coefficients (F, G, 3)} of the pairs built so far (what a model ships)."""
        S = self.built.shape[0]
        return {self.pair_key(a, b): self.coef[a, b].detach().cpu().clone()
                for a in range(S) for b in range(S) if bool(self.built[a, b])}

    def _fill(self, a, b):
        key = self.pair_key(a, b)
        v = self.store.get(key)
        d = table_cache_dir()
        path = d / type(self).__name__ / f"{key}.pt" if d is not None else None
        if v is None and path is not None and path.exists():
            try:
                v = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:  # noqa: BLE001 -- a truncated file: rebuild
                v = None
        if v is None or tuple(v.shape) != (self.n_funcs, self.G, 3):
            v = self._compute(a, b)
            if path is not None:
                import os
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".tmp{os.getpid()}")
                    torch.save(v, tmp)
                    tmp.replace(path)
                except OSError:
                    pass
        self.store[key] = v
        self.coef[a, b] = v.to(self.coef.device, self.coef.dtype)
        self.built[a, b] = True

    def _compute(self, a, b):
        out = self._functions(a, b)
        if set(out) != set(self.names):
            raise RuntimeError(f"{type(self).__name__}: built rows {sorted(set(out) ^ set(self.names))} mismatch")
        v = torch.stack([out[k] for k in self.names]).to(torch.float64)          # (F, G, 3): y, y', y''
        return torch.stack([v[..., 0], v[..., 1] * self.h, v[..., 2] * self.h ** 2], -1).contiguous()

    # ---- building -------------------------------------------------------------------------------------------
    def _names(self):
        out = [f"drive:{p}:{n}" for p in ("dip", "quad") for n in range(NORD)]
        out += [f"aoff:{a}{c}:{n}" for a in "dq" for c in "dq" for n in range(NORD)]
        for name in self.probes:
            out += [f"envV:{name}:{n}" for n in range(NORD)] + [f"envS:{name}:0", f"envS:{name}:1"]
        return out

    #: the step (Angstrom) of the five-point derivative stencil of the closed-form rows
    FD_STEP = 2e-3

    def _functions(self, si, sj):
        """{name: (G, 3)} the rows of the ordered species pair (si, sj) with their first two r-derivatives: the closed
        forms evaluated once on the grid shifted by 0, +-d, +-2d and differentiated by the five-point stencil (errors
        ~1e-10 in y', ~1e-8 in y'' relative to the row's scale; y'' only shapes the interpolation inside a cell)."""
        d = self.FD_STEP
        with torch.no_grad():
            o = self._radial_values(si, sj, torch.cat([self.r + k * d for k in (-2, -1, 0, 1, 2)]))
        res = {}
        for k, f in o.items():
            f = f.reshape(5, -1)
            d1 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * d)
            d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * d * d)
            res[k] = torch.stack([f[2], d1, d2], -1)
        return res

    def _radial_values(self, si, sj, r):
        tab = self.tab
        rb = r / BOHR
        G = r.shape[0]
        one = K._onehot0(G, rb)
        out: dict[str, torch.Tensor] = {}
        shells = [(tab.shell_b[sj, t], tab.shell_n[sj, t]) for t in range(tab.shell_n.shape[1]) if tab.shell_n[sj, t] > 0]

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
        for a, (ba, wa) in parents.items():
            bpa, wpa = _expand(ba, wa, G)
            for c_, (bc, wc) in parents_j.items():
                bpc, wpc = _expand(bc, wc, G)
                f = shaped_two_center_damp(bpa, bpc, wpa, wpc, rb, 2)
                for n in range(NORD):
                    out[f"aoff:{a[0]}{c_[0]}:{n}"] = f[n]
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
            S0 = S1 = 0.0 * rb
            for bt, nt in shells:
                a_, da, _ = shaped_overlap_derivs(bp, wp, bt.expand(G), one, rb)
                S0, S1 = S0 + nt * a_, S1 + nt * da
            out[f"envS:{name}:0"], out[f"envS:{name}:1"] = S0, S1
        return out

    # ---- evaluation ------------------------------------------------------------------------------------------
    def to(self, device):
        self.coef = self.coef.to(device)
        return self

    @property
    def device(self):
        return self.coef.device

    def __call__(self, si, sj, r_ang):
        """All functions at r (P,) Angstrom for species pairs (si, sj) (P,): (P, F). Quintic Hermite, zero past r_max,
        clamped below r_min.

        Vectorized: both grid nodes' coefficients in one gather, the six Hermite basis functions as
        [1, t, .., t^5] @ M, and one contraction -- ~15 kernels; equal to the transcription ``_call_ref``."""
        self.ensure(si, sj)
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
        self.ensure(si, sj)
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
        super().__init__(tab, **{"probes": (), "r_max": r_max, **kw})

    @staticmethod
    def _xpol_names():
        return [f"xp:{q}:{n}" for q in ("dip", "quad") for n in range(NORD)] + ["xp:breath", "xp:ovl"]

    def _names(self):
        return self._xpol_names()

    def _functions(self, si, sj):
        return self._xpol_rows(si, sj)

    @torch.no_grad()
    def _xpol_rows(self, si, sj):
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
        return out


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
