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
from .environment import INVARIANT_NAMES, PROBE_NAMES, shaped_overlap_derivs

__all__ = ["RadialTable", "batched_precompute"]

NORD = 5          # damping orders of the rank-2 tensor
BOHR = K.BOHR_ANG


def _expand(b, w, G):
    """Species-level shape (T,), (T, 4) or (), (4,) -> per-point (G, T), (G, T, 4) / (G,), (G, 4)."""
    if b.dim() == 0:
        return b.expand(G), w[None].expand(G, -1)
    return b[None].expand(G, -1), w[None].expand(G, -1, -1)


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
        clamped below r_min."""
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
                       labels: dict | None = None) -> dict:
    """The collated batch of the complete model straight from positions (N, 3) Angstrom (autograd flows through),
    atomic numbers (N,), frame index (N,) sorted by frame. ``labels``: a dict of per-frame tensors (B,) to attach.
    Runs on the device of ``pos_ang``; ``tab`` and ``table`` must live there too (``IonTable.to``, ``RadialTable.to``).

    Produces the keys ``collate`` makes that :class:`CompleteModel` reads: L, A_off, VD/KD/VA/KA (padded), mask,
    species, feats, frame, pos, channels (index, type, r, frame, ltaper), pseudo drive (index, type, r, basis), env.
    The damped tensors of every pair are built once as the five pure-order basis tensors and every term is a single
    contraction of table rows against them."""
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
    pairs = K.all_pairs(b_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)                  # directed (i <- j)
    i, j = dp
    dr = X[j] - X[i]
    r = dr.norm(dim=-1)
    r_ang = r * BOHR
    rhat = dr / r[:, None]
    y = table(s[i], s[j], r_ang)                                     # (Pd, F)
    qj = tab.charge[s[j]]
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
    lam = torch.cat([qj[None, None] - rows[:nq], rows[nq:]], 0)
    R0 = torch.einsum("knp,npb->kpb", lam, T0)                       # (K, Pd, 10)
    Rp = R0 @ P                                                      # (K, Pd, 8)
    # -- induction drive ------------------------------------------------------------------------------------------
    Lpair = torch.cat([Rp[0, :, :3], Rp[1, :, 3:]], 1)
    Lflat = X.new_zeros(N, 8).index_add(0, i, Lpair)
    L = X.new_zeros(B, nmax, 8).index_put((b_idx, loc), Lflat)
    # -- induced-induced coupling (undirected pairs: the first half of dp) --------------------------------------
    nP = pairs.shape[0] and pairs.shape[1]
    ii, jj = pairs
    PTP = torch.einsum("ab,npbc,cd->npad", P.T, Tb[:, :nP], P)       # (NORD, nP, 8, 8) basis blocks
    arows = table.rows(y[:nP], ["aoff:dd", "aoff:dq", "aoff:qd", "aoff:qq"])     # (4, NORD, nP)
    blocks = torch.einsum("knp,npab->kpab", 1 - arows, PTP)          # (4, nP, 8, 8): [a c] = dd, dq, qd, qq
    sl = {"d": slice(0, 3), "q": slice(3, 8)}
    Bij = X.new_zeros(nP, 8, 8)
    for k_, (a, c) in enumerate((("d", "d"), ("d", "q"), ("q", "d"), ("q", "q"))):
        Bij[:, sl[c], sl[a]] = blocks[k_][:, sl[c], sl[a]]
    ar = torch.arange(8, device=dev)
    A = X.new_zeros(B, nmax, 8, nmax, 8)
    fb = b_idx[ii][:, None, None]
    A = A.index_put((fb, loc[jj][:, None, None], ar[None, :, None], loc[ii][:, None, None], ar[None, None, :]), Bij, accumulate=True)
    A = A.index_put((fb, loc[ii][:, None, None], ar[None, :, None], loc[jj][:, None, None], ar[None, None, :]),
                    Bij.transpose(1, 2), accumulate=True)
    A_off = A.reshape(B, 8 * nmax, 8 * nmax)
    # -- frontier potentials --------------------------------------------------------------------------------------
    fkeys = ["front:donor:dip", "front:donor:quad", "front:acceptor:dip", "front:acceptor:quad"]
    frows = table.rows(y, fkeys)                                      # (4, NORD, Pd)
    KC = torch.einsum("knp,npa->kpa", 1 - frows, Tc) @ P             # (4, Pd, 8)
    front = {}
    for q_, (role, key) in enumerate((("donor", "D"), ("acceptor", "A"))):
        v = (qj - y[:, table.index[f"front:{role}:v"]]) / r
        kv = torch.cat([KC[2 * q_][:, :3], KC[2 * q_ + 1][:, 3:]], 1)
        front[f"V{key}"] = X.new_zeros(B, nmax, nmax).index_put((b_idx[i], loc[i], loc[j]), v)
        front[f"K{key}"] = X.new_zeros(B, nmax, nmax, 8).index_put((b_idx[i], loc[i], loc[j]), kv)
    # -- environment features (all probes at once) ---------------------------------------------------------------
    nprobe = len(probes)
    D0 = R0[2:2 + nprobe]                                             # (nprobe, Pd, 10)
    Dp = Rp[2:2 + nprobe]
    v0 = (D0[:, :, 0] - (qj / r)[None]).T                             # (Pd, nprobe)
    V0 = X.new_zeros(N, nprobe).index_add(0, i, v0)
    V1 = X.new_zeros(N, nprobe, 3).index_add(0, i, -Dp[:, :, :3].permute(1, 0, 2))
    V2 = X.new_zeros(N, nprobe, 5).index_add(0, i, Dp[:, :, 3:].permute(1, 0, 2))
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
        S2[:, kb] = S2[:, kb] * has[:, None, None]
    inv = torch.stack([V0, V1.norm(dim=-1), V2.norm(dim=-1), S0, S1.norm(dim=-1), S2.flatten(2).norm(dim=-1),
                       (V1 * S1).sum(-1)], 2)                         # (N, nprobe, 7)
    feats = inv.reshape(N, -1)                                        # probe-major, like DensityEnvironment
    env = {}
    for k_, name in enumerate(probes):
        for key, val in (("V0", V0), ("V1", V1), ("V2", V2), ("S0", S0), ("S1", S1), ("S2", S2)):
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
    mask = torch.zeros(B, nmax, dtype=torch.bool, device=dev).index_put((b_idx, loc), torch.ones(N, dtype=torch.bool, device=dev))
    lab = dict(labels or {})
    lab.setdefault("n", counts.to(dt))
    lab.setdefault("ct_mask", torch.ones(B, dtype=torch.bool, device=dev))
    nT = len(channel_types(tab.symbols))
    lab.setdefault("chan", torch.full((B, nT), float("nan"), dtype=dt, device=dev))
    for key in ("pol", "ct", "pauli", "disp", "elec", "int"):
        lab.setdefault(key, torch.full((B,), float("nan"), dtype=dt, device=dev))
    return {"B": B, "nmax": nmax, "L": L, "A_off": A_off, "VD": front["VD"], "KD": front["KD"], "VA": front["VA"],
            "KA": front["KA"], "mask": mask, "species": s, "feats": feats, "frame": b_idx, "pos": pos_ang,
            "ch_index": ch_index, "ch_type": ct[ck], "ch_r": ch_r, "ch_frame": b_idx[ch_index[0]],
            "ch_ltaper": ch_ltaper, "ps_index": dp[:, keep], "ps_type": ps_type, "ps_r": r_ang[keep],
            "ps_basis": ps_basis, "ps_basis_sh": ps_basis_sh, "env": env, "labels": lab}
