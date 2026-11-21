"""Environment channels of the m33 complete model: two- and three-body invariants of two weighted neighbour densities.

Each directed pair (i <- j) carries two scalar weights,

    w_ij = O_ij                     the frozen density-density overlap <rho_i | rho_j> (``XpolTable`` row xp:ovl),
    s_ij = E^elst_ij O_ij           the same, signed by the frozen pairwise electrostatic energy of the two partners
                                    (nuclei + shells, the cls_elec pair term; + for like, - for unlike charges),

both switched on smoothly where the overlap leaves the table's quadrature noise (as the xpol ratios), so they vanish
smoothly with distance and the cutoff is the overlap threshold. With unit-normalised real harmonics y_L ("norm":
sum_M y_LM(a) y_LM(b) = P_L(a.b)) and the atom-summed tensors

    S^a_i = sum_j a_ij,      C^a_{i,LM} = sum_j a_ij y_LM(r_ij^),      D^ab_i = sum_j a_ij b_ij      (a, b in {w, s})

the channels are

    two-body      S^w, S^s
    three-body    T^ab_0 = 1/2 (S^a S^b - D^ab)                 = sum_{j<k} 1/2 (a_j b_k + a_k b_j)
                  T^ab_L = 1/2 (S^a S^b - C^a_L . C^b_L)       = sum_{j<k} 1/2 (a_j b_k + a_k b_j) [1 - P_L(cos theta_jk)]

for ab in (ww, ws, ss) and L = 1..LMAX, each three-body channel divided by S^w (the energy-scaling rule: a log-factor on
a parameter that multiplies an overlap-like pair energy). Every three-body channel is exactly zero for a single
neighbour at any distance; all of them are polynomials in the harmonics (smooth at symmetric sites) and go to zero
smoothly as neighbours leave. Linear cost in the number of pairs: no triple loop.

Theory: dev/rsfff theory/response_symmetry.tex (sections 6-7); plan: project doc claude/m33_plan.md.
"""
from __future__ import annotations

import math

import torch

from ..multipole import slater_one_center_damp, slater_two_center_damp_unequal

__all__ = ["LMAX", "CHANNEL_NAMES", "THREE_BODY_COLS", "TWO_BODY_COLS", "pair_electrostatics", "real_harmonics",
           "channels_from_pairs", "overlap_switch", "stacked_harmonics", "raw_channels", "R_MAX"]

#: the channels' pairs end here (the frozen overlaps are < 1e-9 beyond)
R_MAX = 10.0

LMAX = 4
_PAIRS = ("ww", "ws", "ss")
CHANNEL_NAMES = ["S^w", "S^s"] + [f"T^{ab}_{L}/S" for ab in _PAIRS for L in range(LMAX + 1)]
TWO_BODY_COLS = (0, 1)
THREE_BODY_COLS = tuple(range(2, len(CHANNEL_NAMES)))

#: the overlap window over which a pair's weights switch on (the xpol ratios' window: above the table's ~1e-8 noise)
O_ON, O_FULL = 1e-7, 1e-6


#: the step functions of the overlap windows (``CmmConfig.env_switch``)
SWITCHES = ("cubic", "quintic", "smooth")


#: the "smooth" step is exactly 0 for t <= SMOOTH_EDGE and 1 for t >= 1 - SMOOTH_EDGE (|s| < 1e-100 cut off)
SMOOTH_EDGE = 1.0 / 230.0


def switch_step(t, kind="cubic"):
    """The step from 0 at t <= 0 to 1 at t >= 1.

    ``"cubic"``    t^2 (3 - 2 t): C^1. Its second derivative jumps by 6 at both ends, a jump in the curvature of
                   everything it switches (the force constants).
    ``"quintic"``  t^3 (10 - 15 t + 6 t^2): C^2, the third derivative jumps by 60 at the ends (the form of the
                   channel taper and the periodic switch); largest |s''| 5.8.
    ``"smooth"``   1 / (1 + exp((1 - 2 t) / (t (1 - t)))): every derivative zero at both ends (C^infinity), at the
                   price of a steeper middle: slope 2 at t = 1/2 (cubic 1.5, quintic 1.9), largest |s''| 9.8."""
    if kind not in SWITCHES:
        raise ValueError(f"switch {kind!r}: one of {SWITCHES}")
    t = t.clamp(0.0, 1.0)
    if kind == "cubic":
        return t * t * (3 - 2 * t)
    if kind == "quintic":
        return t * t * t * (10 - 15 * t + 6 * t * t)
    # exact ends where the step is below 1e-100 (t <= 1/230: exp(-1/t) < 1e-100): its far tail would otherwise run
    # through 1e-300 .. 1e-160, where anything dividing by a sum of switched weights has a reciprocal whose square
    # underflows, and its gradient is 0 / 0 (a non-finite force at the edge of an overlap window). The step changes
    # by less than 1e-100 there.
    eps = SMOOTH_EDGE
    mid = (t > eps) & (t < 1 - eps)
    ts = torch.where(mid, t, torch.full_like(t, 0.5))      # the unselected branch stays finite (and so its gradient)
    s = torch.sigmoid((2 * ts - 1) / (ts * (1 - ts)))
    return torch.where(mid, s, (t >= 0.5).to(t.dtype))


def overlap_switch(O, o_on=O_ON, o_full=O_FULL, kind="cubic"):
    """:func:`switch_step` in log10 O from 0 at o_on to 1 at o_full (a negative O, table noise, counts as 0)."""
    lo, hi = math.log10(o_on), math.log10(o_full)
    return switch_step((torch.log10(O.clamp(min=1e-300)) - lo) / (hi - lo), kind)


def pair_electrostatics(si, sj, r_bohr, tab):
    """(P,) frozen electrostatic energy (Hartree) of each pair of ions (point nuclei + Slater 1s shells, penetration
    included): the pair term of :func:`rsfff.ff.cmm.elst.permanent_electrostatics`, for given separations."""
    ni, bi, nj, bj = tab.shell_n[si], tab.shell_b[si], tab.shell_n[sj], tab.shell_b[sj]
    P_, M = ni.shape
    r = r_bohr
    f1 = slater_one_center_damp((torch.cat([bj, bi], 1) * r[:, None]).reshape(-1), 0)[0].reshape(P_, 2 * M)
    e = tab.Z[si] * tab.Z[sj] - tab.Z[si] * (nj * (1 - f1[:, :M])).sum(1) - tab.Z[sj] * (ni * (1 - f1[:, M:])).sum(1)
    f2 = slater_two_center_damp_unequal(bi[:, :, None].expand(-1, -1, M).reshape(-1),
                                        bj[:, None, :].expand(-1, M, -1).reshape(-1),
                                        r[:, None].expand(-1, M * M).reshape(-1), 0)[0].reshape(P_, M, M)
    e = e + (ni[:, :, None] * nj[:, None, :] * (1 - f2)).sum((1, 2))
    return e / r


def real_harmonics(rhat, L):
    """(P, 2L+1) real spherical harmonics of the unit vectors rhat, normalised so that sum_M y_LM^2 = 1 (and
    sum_M y_LM(a) y_LM(b) = P_L(a . b))."""
    from e3nn import o3
    return o3.spherical_harmonics(L, rhat, normalize=False, normalization="norm")


_HARM_PLAN: dict = {}


def _snap(C):
    """Fitted harmonic coefficients to their exact values: each is +-sqrt(rational) with a small denominator (e3nn's
    "norm" real harmonics), so c^2 is snapped to the nearest such fraction; entries below 1e-12 are zeroed."""
    from fractions import Fraction
    out = []
    for c in C.flatten().tolist():
        if abs(c) < 1e-12:
            out.append(0.0)
            continue
        q = Fraction(c * c).limit_denominator(100000)
        if abs(float(q) - c * c) > 1e-12 * c * c:
            raise RuntimeError(f"harmonic coefficient {c!r} is not sqrt(rational)")
        out.append(math.copysign(math.sqrt(q.numerator / q.denominator), c))
    return torch.tensor(out, dtype=C.dtype).reshape(C.shape)


def _harmonics_plan(lmax, dtype, device):
    """Cached constants for :func:`stacked_harmonics` up to ``lmax``: ``idx (F, 3)`` the exponents of the homogeneous
    monomials x^a y^b z^c of degrees 1..lmax, ``C (F, sum_L (2L+1))`` their coefficients in e3nn's real harmonics
    ("norm", unnormalized input: homogeneous polynomials of degree L) -- fitted once by least squares on random unit
    vectors (exact to rounding; entries below 1e-12 zeroed) -- and ``B (sum_L (2L+1), lmax)`` the 0/1 block-sum
    matrix that adds up each degree's 2L+1 components."""
    key = (lmax, dtype, device)
    plan = _HARM_PLAN.get(key)
    if plan is None:
        import itertools
        g = torch.Generator().manual_seed(0)
        v = torch.randn(64 * lmax * lmax + 64, 3, generator=g, dtype=torch.float64)
        v = v / v.norm(dim=1, keepdim=True)
        idx, blocks, rows = [], [], 0
        for L in range(1, lmax + 1):
            ex = [e for e in itertools.product(range(L + 1), repeat=3) if sum(e) == L]
            Mo = torch.stack([v[:, 0] ** a * v[:, 1] ** b * v[:, 2] ** c for a, b, c in ex], 1)
            Cl = torch.linalg.lstsq(Mo, real_harmonics(v, L)).solution
            Cl = _snap(Cl)
            blocks.append((rows, Cl))
            idx += ex
            rows += len(ex)
        ncol = sum(2 * L + 1 for L in range(1, lmax + 1))
        C = torch.zeros(rows, ncol, dtype=torch.float64)
        B = torch.zeros(ncol, lmax, dtype=torch.float64)
        col = 0
        for L, (r0, Cl) in zip(range(1, lmax + 1), blocks):
            C[r0:r0 + Cl.shape[0], col:col + 2 * L + 1] = Cl
            B[col:col + 2 * L + 1, L - 1] = 1.0
            col += 2 * L + 1
        f = dict(dtype=dtype, device=device)
        plan = (torch.tensor(idx, dtype=torch.long, device=device), C.to(**f), B.to(**f))
        _HARM_PLAN[key] = plan
    return plan


def stacked_harmonics(rhat, lmax=LMAX):
    """(P, sum_{L=1..lmax} (2L+1)) the real harmonics of degrees 1..lmax side by side (each block equal to
    :func:`real_harmonics` (rhat, L)): one gather of coordinate powers, a product and one matmul."""
    idx, C, _ = _harmonics_plan(lmax, rhat.dtype, rhat.device)
    # powers 0..lmax of each coordinate by a running product (finite gradients at 0, unlike pow): (P, 3, lmax+1)
    pw = torch.cumprod(torch.cat([torch.ones_like(rhat)[..., None], rhat[..., None].expand(*rhat.shape, lmax)], -1), -1)
    m = pw[:, 0, idx[:, 0]] * pw[:, 1, idx[:, 1]] * pw[:, 2, idx[:, 2]]                    # (P, F) monomials
    return m @ C


def channels_from_pairs(n, i, dr, w, s, lmax=LMAX, eps=1e-100):
    """(n, 2 + 3 (lmax + 1)) channels for atoms 0..n-1 from directed pairs (i <- j): centre index i (P,), bond vectors
    dr = X_j - X_i (P, 3, any length unit), weights w, s (P,). Atoms without pairs get zeros.

    Vectorized: every per-pair quantity (w, s, the products D^ab and the weighted harmonics a y_LM of all degrees) goes
    through one index_add; the C^a_L . C^b_L contractions are one block-sum matmul. :func:`_channels_from_pairs_ref`
    is the per-L loop it replaces.

    An atom whose total weight is at most ``eps`` gets zero T (its T is O(weight) there): above ~1e-154 the reciprocal's
    gradient -1 / Sw^2 is representable; below it, it underflows to 0 / 0."""
    if lmax < 1:
        return _channels_from_pairs_ref(n, i, dr, w, s, lmax, eps)
    _, _, Bsum = _harmonics_plan(lmax, dr.dtype, dr.device)
    rhat = dr / dr.norm(dim=-1, keepdim=True)
    Y = stacked_harmonics(rhat, lmax)                                                       # (P, K)
    a = torch.stack([w, s], 1)                                                              # (P, 2)
    D = torch.stack([w * w, w * s, s * s], 1)                                               # (P, 3)  ww, ws, ss
    per = torch.cat([a, D, (a[:, :, None] * Y[:, None, :]).flatten(1)], 1)                  # (P, 5 + 2K)
    acc = w.new_zeros(n, per.shape[1]).index_add(0, i, per)
    S, Dn, Cn = acc[:, :2], acc[:, 2:5], acc[:, 5:].view(n, 2, -1)                          # (n,2) (n,3) (n,2,K)
    def pairs(t):           # t[:, (0, 0, 1)] * t[:, (0, 1, 1)] -- ww, ws, ss -- by slices (no index-tensor upload)
        return torch.cat([t[:, :1], t[:, :2]], 1) * torch.cat([t[:, :2], t[:, 1:]], 1)
    SS = pairs(S)                                                                           # (n, 3)
    CC = pairs(Cn) @ Bsum                                                                   # (n, 3, lmax)
    Sw = S[:, 0]
    ok = Sw > eps
    inv = torch.where(ok, 1.0 / torch.where(ok, Sw, torch.ones_like(Sw)), torch.zeros_like(Sw))
    T = 0.5 * (SS[:, :, None] - torch.cat([Dn[:, :, None], CC], 2)) * inv[:, None, None]   # (n, 3, lmax+1)
    return torch.cat([S, T.flatten(1)], 1)


def _channels_from_pairs_ref(n, i, dr, w, s, lmax=LMAX, eps=1e-100):
    """Reference (per-L, per-weight loop) form of :func:`channels_from_pairs`, kept for tests.

    (n, 2 + 3 (lmax + 1)) channels for atoms 0..n-1 from directed pairs (i <- j): centre index i (P,), bond vectors
    dr = X_j - X_i (P, 3, any length unit), weights w, s (P,). Atoms without pairs get zeros."""
    rhat = dr / dr.norm(dim=-1, keepdim=True)
    z = w.new_zeros(n)
    S = {"w": z.index_add(0, i, w), "s": z.index_add(0, i, s)}
    a_ = {"w": w, "s": s}
    D = {ab: z.index_add(0, i, a_[ab[0]] * a_[ab[1]]) for ab in _PAIRS}
    C = {}
    for L in range(1, lmax + 1):
        Y = real_harmonics(rhat, L)                                           # (P, 2L+1)
        for a in ("w", "s"):
            C[a, L] = w.new_zeros(n, 2 * L + 1).index_add(0, i, a_[a][:, None] * Y)
    Sw = S["w"]
    ok = Sw > eps
    inv = torch.where(ok, 1.0 / torch.where(ok, Sw, torch.ones_like(Sw)), torch.zeros_like(Sw))
    cols = [S["w"], S["s"]]
    for ab in _PAIRS:
        a, b = ab
        SS = S[a] * S[b]
        cols.append(0.5 * (SS - D[ab]) * inv)
        for L in range(1, lmax + 1):
            cols.append(0.5 * (SS - (C[a, L] * C[b, L]).sum(1)) * inv)
    return torch.stack(cols, 1)


def raw_channels(tab, xpol_table, pos_ang, species, frame, pairs=None, pair_dr=None):
    """(N, 17) the raw channels of every atom: overlap weights w_ij from ``xpol_table`` (the frozen density overlap,
    switched on above the table's noise), signed weights s_ij = E^elst_ij w_ij, from the frame's all-pairs list or
    ``pairs`` (2, P) with separations ``pair_dr`` (P, 3) Angstrom (periodic images included)."""
    from .kernels import BOHR_ANG, all_pairs
    if pairs is None:
        pairs = all_pairs(frame)
        pair_dr = pos_ang[pairs[1]] - pos_ang[pairs[0]]
    i, j = pairs
    r = pair_dr.norm(dim=-1)
    keep = r < R_MAX
    i, j, dr, r = i[keep], j[keep], pair_dr[keep], r[keep]
    a, b = torch.cat([i, j]), torch.cat([j, i])
    rr = torch.cat([r, r])
    y = xpol_table(species[a], species[b], rr)
    O = y[:, xpol_table.cols(("xp:ovl",), y.device)[0]]
    w = overlap_switch(O) * O.clamp(min=0.0)
    E = pair_electrostatics(species[a], species[b], rr / BOHR_ANG, tab)
    return channels_from_pairs(species.shape[0], a, torch.cat([dr, -dr]), w, E * w)
