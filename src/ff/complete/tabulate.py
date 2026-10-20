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
from ..ionct.model import CHANNEL_TYPES, PSEUDO_CUTOFF, PSEUDO_TYPES
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

    def pick(self, y, prefix):
        """(P, NORD) rows ``prefix:0..4`` of an evaluation (P, F)."""
        return torch.stack([y[:, self.index[f"{prefix}:{n}"]] for n in range(NORD)], 0)       # (NORD, P)


def _local_index(batch_idx):
    counts = torch.bincount(batch_idx)
    return batch_idx.shape[0], counts, torch.arange(batch_idx.shape[0]) - (torch.cumsum(counts, 0) - counts)[batch_idx]


def batched_precompute(pos_ang, atomic_numbers, batch_idx, tab: IonTable, table: RadialTable, chan: CT.ChannelTable,
                       labels: dict | None = None) -> dict:
    """The collated batch of the complete model straight from positions (N, 3) Angstrom (autograd flows through),
    atomic numbers (N,), frame index (N,) sorted by frame. ``labels``: a dict of per-frame tensors (B,) to attach.

    Produces the keys ``collate`` makes that :class:`CompleteModel` reads: L, A_off, VD/KD/VA/KA (padded), mask,
    species, feats, frame, pos, channels (index, type, r, frame, ltaper), pseudo drive (index, type, r, basis), env."""
    X = pos_ang / BOHR
    N = X.shape[0]
    b_idx = torch.as_tensor(batch_idx)
    B = int(b_idx.max()) + 1 if N else 0
    s = tab.species_index(torch.as_tensor(atomic_numbers))
    dt = X.dtype
    _, counts, loc = _local_index(b_idx)
    nmax = int(counts.max()) if N else 0
    P = K._poly_map(dt, X.device)
    pairs = K.all_pairs(b_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)                  # directed (i <- j)
    i, j = dp
    dr = X[j] - X[i]
    r = dr.norm(dim=-1)
    r_ang = r * BOHR
    rhat = dr / r[:, None]
    y = table(s[i], s[j], r_ang)                                     # (Pd, F)
    qj = tab.charge[s[j]]
    # -- induction drive ------------------------------------------------------------------------------------------
    Ld = K._tensor(dr, qj[None] - table.pick(y, "drive:dip"))[:, 0, :] @ P
    Lq = K._tensor(dr, qj[None] - table.pick(y, "drive:quad"))[:, 0, :] @ P
    Lflat = X.new_zeros(N, 8).index_add(0, i, torch.cat([Ld[:, :3], Lq[:, 3:]], 1))
    L = X.new_zeros(B, nmax, 8).index_put((b_idx, loc), Lflat)
    # -- induced-induced coupling (undirected pairs: the first half of dp) --------------------------------------
    nP = pairs.shape[1]
    ii, jj = pairs
    dr_u, y_u = dr[:nP], y[:nP]
    sl = {"d": slice(0, 3), "q": slice(3, 8)}
    Bij = X.new_zeros(nP, 8, 8)
    for a in "dq":
        for c in "dq":
            blk = P.T @ K._tensor(dr_u, 1 - table.pick(y_u, f"aoff:{a}{c}")) @ P
            Bij[:, sl[c], sl[a]] = blk[:, sl[c], sl[a]]
    ar = torch.arange(8)
    A = X.new_zeros(B, nmax, 8, nmax, 8)
    fb = b_idx[ii][:, None, None]
    A = A.index_put((fb, loc[jj][:, None, None], ar[None, :, None], loc[ii][:, None, None], ar[None, None, :]), Bij, accumulate=True)
    A = A.index_put((fb, loc[ii][:, None, None], ar[None, :, None], loc[jj][:, None, None], ar[None, None, :]),
                    Bij.transpose(1, 2), accumulate=True)
    A_off = A.reshape(B, 8 * nmax, 8 * nmax)
    # -- frontier potentials --------------------------------------------------------------------------------------
    front = {}
    for role, key in (("donor", "D"), ("acceptor", "A")):
        v = (qj - y[:, table.index[f"front:{role}:v"]]) / r
        ks = []
        for p in ("dip", "quad"):
            lam = 1 - table.pick(y, f"front:{role}:{p}")
            ks.append(K._tensor(dr, lam)[:, :, 0] @ P)
        kv = torch.cat([ks[0][:, :3], ks[1][:, 3:]], 1)
        V = X.new_zeros(B, nmax, nmax).index_put((b_idx[i], loc[i], loc[j]), v)
        Km = X.new_zeros(B, nmax, nmax, 8).index_put((b_idx[i], loc[i], loc[j]), kv)
        front[f"V{key}"], front[f"K{key}"] = V, Km
    # -- environment features -------------------------------------------------------------------------------------
    inv, env = [], {}
    outer = rhat[:, :, None] * rhat[:, None, :] - torch.eye(3, dtype=dt)[None] / 3
    for name in table.probes:
        D = K._tensor(dr, qj[None] - table.pick(y, f"envV:{name}"))[:, 0, :]
        Dp = D @ P
        v0 = D[:, 0] - qj / r
        V0 = X.new_zeros(N).index_add(0, i, v0)
        V1 = X.new_zeros(N, 3).index_add(0, i, -Dp[:, :3])
        V2 = X.new_zeros(N, 5).index_add(0, i, Dp[:, 3:])
        S0p, S1p, S2p = (y[:, table.index[f"envS:{name}:{k}"]] for k in range(3))
        S0 = X.new_zeros(N).index_add(0, i, S0p)
        S1 = X.new_zeros(N, 3).index_add(0, i, -S1p[:, None] * rhat)
        S2 = X.new_zeros(N, 3, 3).index_add(0, i, (S2p - S1p / r)[:, None, None] * outer)
        if name == "breath":
            has = (tab.a0[s] > 0).to(dt)
            V0, V1, V2, S0, S1, S2 = (V0 * has, V1 * has[:, None], V2 * has[:, None], S0 * has, S1 * has[:, None],
                                      S2 * has[:, None, None])
        inv.append(torch.stack([V0, V1.norm(dim=-1), V2.norm(dim=-1), S0, S1.norm(dim=-1), S2.flatten(1).norm(dim=-1),
                                (V1 * S1).sum(-1)], 1))
        for k_, v_ in (("V0", V0), ("V1", V1), ("V2", V2), ("S0", S0), ("S1", S1), ("S2", S2)):
            env[f"{name}:{k_}"] = v_
    feats = torch.cat(inv, 1)
    # -- pseudo-potential drive basis -----------------------------------------------------------------------------
    keep = r_ang < table.pseudo_cutoff
    ps_i, ps_j, dr_p, y_p = i[keep], j[keep], dr[keep], y[keep]
    bd = K._tensor(dr_p, table.pick(y_p, "ps:dip"))[:, 0, :] @ P
    bq = K._tensor(dr_p, table.pick(y_p, "ps:quad"))[:, 0, :] @ P
    ps_basis = torch.cat([bd[:, :3], bq[:, 3:]], 1)
    sym = tab.symbols
    ps_type = torch.tensor([PSEUDO_TYPES.index(f"{sym[a]}<{sym[b]}") for a, b in zip(s[ps_i].tolist(), s[ps_j].tolist())],
                           dtype=torch.long)
    # -- channels (direct CT) ----------------------------------------------------------------------------------------
    names = [f"{sym[a]}>{sym[b]}" for a, b in zip(s[i].tolist(), s[j].tolist())]
    ck = torch.tensor([n in chan.types for n in names], dtype=torch.bool) & (r_ang < chan.cutoff)
    ch_index = dp[:, ck]
    ch_r = r_ang[ck]
    ch_type = torch.tensor([CHANNEL_TYPES.index(n) for n, k in zip(names, ck.tolist()) if k], dtype=torch.long)
    ch_ltaper = torch.log(CT._taper(ch_r, chan.cutoff, chan.taper).clamp(min=1e-300))
    mask = torch.zeros(B, nmax, dtype=torch.bool).index_put((b_idx, loc), torch.ones(N, dtype=torch.bool))
    lab = dict(labels or {})
    lab.setdefault("n", counts.to(dt))
    lab.setdefault("ct_mask", torch.ones(B, dtype=torch.bool))
    nT = len(CHANNEL_TYPES)
    lab.setdefault("chan", torch.full((B, nT), float("nan"), dtype=dt))
    for key in ("pol", "ct", "pauli", "disp", "elec", "int"):
        lab.setdefault(key, torch.full((B,), float("nan"), dtype=dt))
    return {"B": B, "nmax": nmax, "L": L, "A_off": A_off, "VD": front["VD"], "KD": front["KD"], "VA": front["VA"],
            "KA": front["KA"], "mask": mask, "species": s, "feats": feats, "frame": b_idx, "pos": pos_ang,
            "ch_index": ch_index, "ch_type": ch_type, "ch_r": ch_r, "ch_frame": b_idx[ch_index[0]],
            "ch_ltaper": ch_ltaper, "ps_index": torch.stack([ps_i, ps_j]), "ps_type": ps_type, "ps_r": r_ang[keep],
            "ps_basis": ps_basis, "env": env, "labels": lab}
