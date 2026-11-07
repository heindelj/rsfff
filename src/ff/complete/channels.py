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
           "channels_from_pairs", "overlap_switch"]

LMAX = 4
_PAIRS = ("ww", "ws", "ss")
CHANNEL_NAMES = ["S^w", "S^s"] + [f"T^{ab}_{L}/S" for ab in _PAIRS for L in range(LMAX + 1)]
TWO_BODY_COLS = (0, 1)
THREE_BODY_COLS = tuple(range(2, len(CHANNEL_NAMES)))

#: the overlap window over which a pair's weights switch on (the xpol ratios' window: above the table's ~1e-8 noise)
O_ON, O_FULL = 1e-7, 1e-6


def overlap_switch(O, o_on=O_ON, o_full=O_FULL):
    """Smoothstep in log10 O from 0 at o_on to 1 at o_full (a negative O, table noise, counts as 0)."""
    lo, hi = math.log10(o_on), math.log10(o_full)
    t = ((torch.log10(O.clamp(min=1e-300)) - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


def pair_electrostatics(si, sj, r_bohr, tab):
    """(P,) frozen electrostatic energy (Hartree) of each pair of ions (point nuclei + Slater 1s shells, penetration
    included): the pair term of :func:`rsfff.ff.complete.elst.permanent_electrostatics`, for given separations."""
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


def channels_from_pairs(n, i, dr, w, s, lmax=LMAX, eps=1e-300):
    """(n, 2 + 3 (lmax + 1)) channels for atoms 0..n-1 from directed pairs (i <- j): centre index i (P,), bond vectors
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
