"""Donor-centred charge-transfer channels between ions (ion ct ff plan, R3; theory: ct_model/theory, ct_diabatic_plan).

Every directed pair i -> j of ions closer than the CT cutoff is a channel with a gap and a coupling,

    Delta_ij = g_T + s [phibar_D(i) - phibar_A(j)] + Delta^ov_T(R) * ov_scale_ij + ov_add_ij
    W_ij     = W_T(R) * exp(-R delta_ij / kappa_i) * taper(R)                     (option A: no learned W)
    delta_ij = s [phibar_D(i) - phibar_A(j)] - s dphi_T(R)                        (shift from the isolated pair)

T is the channel type (donor species, acceptor species). phibar_D(i) is the potential of every other ion -- its
permanent density and its induced moments -- and of a uniform field, averaged over the donor's frontier density
(``kernels.frontier_potential_matrix``); phibar_A(j) the same over the acceptor's. g_T, W_T(R), Delta^ov_T(R) and
dphi_T(R) (the isolated pair's own potential difference) are fixed functions fitted once to the pair scans
(``scripts/ion_ct_pairs.py``). The environment's part of the overlap gap is what the network emits: ``ov_scale``
(multiplying the pair's own overlap gap, 1 for an isolated pair) and/or ``ov_add`` (an additive overlap gap from
the donor's and acceptor's *other* neighbours, 0 for an isolated pair). The pair's own overlap gap is small at the
distances clusters sample (Cl -> Na: 0.07 Eh at 2.8 A, < 0.01 beyond 3.5 A) while the cluster gaps sit ~0.5 Eh
above the electrostatic estimate (scripts/ionct_ct_check.py: a uniform ov_scale would have to be ~4-15), so the
additive form is the one that can carry the environment.
kappa_i = sqrt(2 I_i) with I_i = I_donor(i) + s phibar_D(i), the donor bound by its whole environment (WKB: the
tail of the donor orbital decays faster in a cationic environment, so the coupling falls when the gap rises).

Per donor the energy is the lowest root of the arrowhead (one donor orbital coupled to its acceptors),

    eps_i = sum_j W_ij / (eps_i - Delta_ij),        E_CT = sum_i eps_i,

which is -sum_j W_ij / Delta_ij to second order; the channel populations are p_ij = W_ij / (eps_i - Delta_ij)^2
normalized with the donor's own weight. No feedback of the transferred charge into the electrostatics yet.

Units: Angstrom at the interface, atomic units otherwise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..units import BOHR_ANG
from . import kernels as K
from .params import DEFAULT_PARAMS, IonTable

__all__ = ["PairFunction", "ChannelTable", "channel_geometry", "channel_gaps", "ct_channels", "arrowhead_roots",
           "arrowhead", "ct_energy", "CTResult"]


# ---- fixed pair functions --------------------------------------------------------------------------------

class PairFunction:
    """Natural cubic spline through (R, y) knots, with linear extrapolation (value and slope matched) outside.
    Differentiable in R. ``R`` in Angstrom."""

    def __init__(self, R, y, dtype=torch.float64):
        R = torch.as_tensor(R, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)
        n = R.shape[0]
        h = R[1:] - R[:-1]
        A = torch.zeros(n, n, dtype=dtype)
        rhs = torch.zeros(n, dtype=dtype)
        A[0, 0] = A[-1, -1] = 1.0
        for k in range(1, n - 1):
            A[k, k - 1], A[k, k], A[k, k + 1] = h[k - 1], 2 * (h[k - 1] + h[k]), h[k]
            rhs[k] = 6 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
        self.R, self.y, self.h = R, y, h
        self.M = torch.linalg.solve(A, rhs)            # second derivatives at the knots
        self.slope_lo = (y[1] - y[0]) / h[0] - h[0] * (2 * self.M[0] + self.M[1]) / 6
        self.slope_hi = (y[-1] - y[-2]) / h[-1] + h[-1] * (self.M[-2] + 2 * self.M[-1]) / 6

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        R, y, M, h = (t.to(r.device) for t in (self.R, self.y, self.M, self.h))
        rc = r.clamp(R[0], R[-1])
        k = torch.searchsorted(R, rc.detach().contiguous(), right=True).clamp(1, R.shape[0] - 1) - 1
        a = (R[k + 1] - rc) / h[k]
        b = (rc - R[k]) / h[k]
        val = a * y[k] + b * y[k + 1] + ((a ** 3 - a) * M[k] + (b ** 3 - b) * M[k + 1]) * h[k] ** 2 / 6
        lo, hi = r < R[0], r > R[-1]
        val = torch.where(lo, y[0] + self.slope_lo.to(r) * (r - R[0]), val)
        return torch.where(hi, y[-1] + self.slope_hi.to(r) * (r - R[-1]), val)


@dataclass
class ChannelTable:
    """Per channel type 'D>A': intrinsic gap g, ln W(R), the overlap gap Delta^ov(R) >= 0 and the isolated pair's
    potential difference dphi(R) (all Hartree, R Angstrom)."""

    types: dict = field(default_factory=dict)      # 'Cl>Na' -> dict(g=..., lnW=PairFunction, dov=..., dphi=...)
    s: float = 1.0
    cutoff: float = 8.0
    taper: float = 1.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=DEFAULT_PARAMS, dtype=torch.float64) -> "ChannelTable":
        d = json.load(open(path))
        ch = d.get("channels")
        if not ch:          # no calibrated pair functions (a table built by scripts/ion_table.py): the direct CT form only
            return cls.direct(tuple(d["ions"]))
        types = {}
        for name, v in ch["types"].items():
            types[name] = {"g": float(v["g"]), "lnW": PairFunction(v["R"], v["lnW"], dtype),
                           "dov": PairFunction(v["R"], v["dov"], dtype), "dphi": PairFunction(v["R"], v["dphi"], dtype),
                           "dov_tail": float(v.get("dov_tail", 2.0)), "R_max": float(v["R"][-1])}
        return cls(types=types, s=float(ch.get("s", 1.0)), cutoff=float(ch.get("cutoff", 8.0)),
                   taper=float(ch.get("taper", 1.0)), meta={k: v for k, v in ch.items() if k != "types"})

    @classmethod
    def direct(cls, symbols, cutoff=8.0, taper=1.0):
        """Every ordered pair as a channel type with no pair functions: what the direct CT form needs (the channel
        model's gap / coupling functions are absent, so only names, cutoff and taper are used)."""
        types = {f"{a}>{b}": None for a in symbols for b in symbols}
        return cls(types=types, cutoff=cutoff, taper=taper, meta={"form": "direct"})

    def overlap_gap(self, name, r):
        """Delta^ov_T(R), clamped >= 0, decaying as exp(-dov_tail (R - R_max)) beyond the fitted range."""
        t = self.types[name]
        rc = r.clamp(max=t["R_max"])
        v = t["dov"](rc).clamp(min=0.0)
        return v * torch.exp(-t["dov_tail"] * (r - rc))


def _taper(r, r_off, width):
    x = ((r - (r_off - width)) / width).clamp(0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


# ---- channels ---------------------------------------------------------------------------------------------

@dataclass
class CTResult:
    energy: torch.Tensor          # (B,) E_CT per frame, Hartree
    eps: torch.Tensor             # (N,) per-donor arrowhead root (0 for ions without channels)
    index: torch.Tensor           # (2, C) donor, acceptor
    type: list                    # (C,) channel type names
    W: torch.Tensor               # (C,)
    Delta: torch.Tensor           # (C,)
    delta: torch.Tensor           # (C,) environment shift of the gap from the isolated pair
    p: torch.Tensor               # (C,) channel populations (electrons moved)
    dq: torch.Tensor              # (N,) CT charges (acceptors +, donors -; electrons -> negative charge)


def channel_index(pos_ang, species, batch_idx, tab: IonTable, chan: ChannelTable):
    """Directed channels (i -> j) within each frame inside the CT cutoff whose type has fitted pair functions."""
    pairs = K.all_pairs(batch_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)
    r = (pos_ang[dp[1]] - pos_ang[dp[0]]).norm(dim=-1)
    names = [f"{tab.symbols[a]}>{tab.symbols[b]}" for a, b in zip(species[dp[0]].tolist(), species[dp[1]].tolist())]
    keep = torch.tensor([n in chan.types for n in names], dtype=torch.bool) & (r < chan.cutoff).cpu()
    keep = keep.to(dp.device)
    idx = dp[:, keep]
    return idx, [n for n, k in zip(names, keep.tolist()) if k]


def channel_geometry(pos_ang, species, batch_idx, tab: IonTable, chan: ChannelTable) -> dict:
    """Everything about the channels that depends on the geometry and the fixed pair functions only (cacheable):
    index (2, C), names, r (C,) A, g, lnW (taper included), dov_pair (the pair's own overlap gap), dphi_pair."""
    idx, names = channel_index(pos_ang, species, batch_idx, tab, chan)
    i, j = idx
    r = (pos_ang[j] - pos_ang[i]).norm(dim=-1)
    g = torch.zeros_like(r)
    lnW = torch.zeros_like(r)
    dov = torch.zeros_like(r)
    dphi_pair = torch.zeros_like(r)
    for name in set(names):
        m = torch.tensor([n == name for n in names], dtype=torch.bool, device=r.device)
        t = chan.types[name]
        if t is None:               # direct-form table: no pair functions
            continue
        g = torch.where(m, torch.full_like(r, t["g"]), g)
        lnW = torch.where(m, t["lnW"](r), lnW)
        dov = torch.where(m, chan.overlap_gap(name, r), dov)
        dphi_pair = torch.where(m, chan.s * t["dphi"](r), dphi_pair)
    lnW = lnW + torch.log(_taper(r, chan.cutoff, chan.taper).clamp(min=1e-300))
    return {"index": idx, "names": names, "r": r, "g": g, "lnW": lnW, "dov_pair": dov, "dphi_pair": dphi_pair}


def channel_gaps(geom: dict, phiD, phiA, I_donor, s: float = 1.0, *, ov_scale=None, ov_add=None,
                 pair_overlap: bool = True, wkb: str = "two_sided"):
    """W, Delta, delta of every channel from the frontier potentials phiD, phiA (N,) (all sources, field included)
    and the donors' free electron-removal energies I_donor (N,). ``pair_overlap=False`` drops the fixed pair
    overlap gap, so the whole overlap gap is ``ov_add`` (the fully learned form)."""
    i, j = geom["index"]
    rb = geom["r"] / BOHR_ANG
    dphi = s * (phiD[i] - phiA[j])
    dov = geom["dov_pair"] if pair_overlap else torch.zeros_like(dphi)
    if ov_scale is not None:
        dov = dov * ov_scale
    if ov_add is not None:
        dov = dov + ov_add
    Delta = geom["g"] + dphi + dov
    delta = dphi - geom["dphi_pair"]
    lnW = geom["lnW"]
    if wkb != "none":
        shift = s * phiD[i]
        if wkb.endswith("+floor"):
            # the donor level's binding never drops below the free ion's: a destabilizing environment (an anion
            # next to an anion, phi_D < 0) would otherwise push I to the clamp, kappa -> small and the WKB
            # exponent (and the field response of W, ~ (R / kappa)^2) up without bound
            shift = KAPPA_FLOOR_W * nn_softplus(shift / KAPPA_FLOOR_W)
        I = (I_donor[i] + shift).clamp(min=0.05)
        x = -rb * delta / torch.sqrt(2 * I)
        lnW = lnW + wkb_response(x, wkb.removesuffix("+floor"))
    return torch.exp(lnW), Delta, delta


#: width (Hartree) of the smooth floor on the donor-level shift in kappa ('...+floor' WKB forms)
KAPPA_FLOOR_W = 0.02


def nn_softplus(x):
    return torch.nn.functional.softplus(x)


#: saturation scales (a_-, a_+) of the smooth WKB response: ln W shifts by a tanh(x / a), a = a_- for x < 0 (gap
#: raised, coupling suppressed) and a_+ for x > 0 (gap lowered, coupling enhanced: bounded so W can grow at most
#: e^{a_+}-fold)
WKB_CAPS = (4.0, 1.0)


def wkb_response(x, wkb: str):
    """The shift of ln W from the WKB exponent x = -R delta / kappa.

    'two_sided': x; 'smooth' or 'smooth:a_-,a_+': a tanh(x / a), analytic, slope 1 and zero
    curvature at x = 0 (the two-sided response to first and second order), saturating at -a_- and +a_+. Any form
    may carry the suffix '+floor' (handled in channel_gaps: kappa from I_donor + softplus(s phi_D)). (A one-sided
    min(x, 0) was removed: its kink at delta = 0 put a cusp in E(F) at every reference pair and symmetric geometry.)"""
    if wkb == "two_sided":
        return x
    if wkb.startswith("smooth"):
        lo, hi = WKB_CAPS if ":" not in wkb else map(float, wkb.split(":")[1].split(","))
        a = torch.where(x > 0, torch.full_like(x, hi), torch.full_like(x, lo))
        return a * torch.tanh(x / a)
    raise ValueError(f"unknown wkb form {wkb!r}")


def ct_channels(pos_ang, species, batch_idx, tab: IonTable, chan: ChannelTable, *, x_ind=None, field=None,
                ov_scale=None, ov_add=None, pair_overlap: bool = True, wkb: str = "two_sided"):
    """Gaps and couplings of every channel: (index, names, W, Delta, delta)."""
    geom = channel_geometry(pos_ang, species, batch_idx, tab, chan)
    PD = K.frontier_potential_matrix(pos_ang, species, batch_idx, tab, "donor", x=x_ind)
    PA = K.frontier_potential_matrix(pos_ang, species, batch_idx, tab, "acceptor", x=x_ind)
    phi_ext = K.external_potential(pos_ang, batch_idx, field)
    phiD = PD.sum(1) + phi_ext                          # (N,) donor frontier of i, all sources
    phiA = PA.sum(1) + phi_ext
    W, Delta, delta = channel_gaps(geom, phiD, phiA, tab.I_donor[species], chan.s, ov_scale=ov_scale, ov_add=ov_add,
                                   pair_overlap=pair_overlap, wkb=wkb)
    return geom["index"], geom["names"], W, Delta, delta


# ---- arrowhead --------------------------------------------------------------------------------------------

def _f(eps, W, D, donor, n):
    t = W / (eps[donor] - D)
    return eps - eps.new_zeros(n).index_add(0, donor, t), 1 + eps.new_zeros(n).index_add(0, donor, t / (eps[donor] - D))


def arrowhead_roots(W, Delta, donor, n_atoms, floor: float = 0.02, n_bisect: int = 60, n_newton: int = 2):
    """Lowest root of eps = sum_j W_j / (eps - Delta_j) per donor (below every Delta_j; 0 for donors without
    channels). Bisection under no_grad, then ``n_newton`` tracked Newton steps: exact first derivatives and
    accurate second derivatives (force training). Gaps are floored smoothly at ``floor``."""
    D = floor + torch.nn.functional.softplus(Delta - floor, beta=200.0)
    has = torch.zeros(n_atoms, dtype=torch.bool, device=W.device)
    has[donor] = True
    with torch.no_grad():
        Dmin = torch.full((n_atoms,), 1e6, dtype=W.dtype, device=W.device).scatter_reduce(0, donor, D, "amin")
        Wsum = W.new_zeros(n_atoms).index_add(0, donor, W.detach())
        hi = torch.where(has, Dmin - 1e-12, torch.zeros_like(Dmin))
        lo = torch.where(has, Dmin.clamp(max=0.0) - Wsum.sqrt() - 1.0, torch.zeros_like(Dmin))   # f(lo) < 0
        Dd, Wd = D.detach(), W.detach()
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f, _ = _f(mid, Wd, Dd, donor, n_atoms)
            pos = f > 0
            hi = torch.where(pos, mid, hi)
            lo = torch.where(pos, lo, mid)
        eps0 = torch.where(has, 0.5 * (lo + hi), torch.zeros_like(lo))
    eps = eps0
    for _ in range(n_newton):
        f, fp = _f(eps, W, D, donor, n_atoms)
        eps = torch.where(has, eps - f / fp, eps)
    return eps, D


def arrowhead(W, Delta, idx, n_atoms):
    """Per-donor roots, channel populations and CT charges (donor +p, acceptor -p) for channels idx (2, C)."""
    i, j = idx
    eps, D = arrowhead_roots(W, Delta, i, n_atoms)
    c2 = W / (eps[i] - D) ** 2
    norm = 1 + W.new_zeros(n_atoms).index_add(0, i, c2)
    p = c2 / norm[i]
    dq = W.new_zeros(n_atoms).index_add(0, i, p).index_add(0, j, -p)
    return eps, p, dq


def ct_energy(pos_ang, species, batch_idx, tab: IonTable, chan: ChannelTable, *, x_ind=None, field=None,
              ov_scale=None, ov_add=None, pair_overlap: bool = True, wkb: str = "two_sided", n_systems=None) -> CTResult:
    """E_CT per frame and the channel diagnostics."""
    n = pos_ang.shape[0]
    B = int(batch_idx.max()) + 1 if n_systems is None else int(n_systems)
    idx, names, W, Delta, delta = ct_channels(pos_ang, species, batch_idx, tab, chan, x_ind=x_ind, field=field,
                                              ov_scale=ov_scale, ov_add=ov_add, pair_overlap=pair_overlap, wkb=wkb)
    i, j = idx
    if idx.shape[1] == 0:
        z = pos_ang.new_zeros(n)
        return CTResult(pos_ang.new_zeros(B), z, idx, names, W, Delta, delta, W, z)
    eps, p, dq = arrowhead(W, Delta, idx, n)
    energy = pos_ang.new_zeros(B).index_add(0, batch_idx, eps)
    return CTResult(energy, eps, idx, names, W, Delta, delta, p, dq)
