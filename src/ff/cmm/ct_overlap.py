"""The frontier-overlap kernel of the CT coupling (``CmmConfig.ct_kernel = "frontier"``).

    W^2_ij = qD_i qA_j [S_ab(R) g_ab(R)]^2          a, b the species of the donor i and the acceptor j

S_ab(R) is the overlap of the free ions' frontier orbitals, the donor orbital of a (its highest occupied shell, p sigma
towards the partner) with the acceptor orbital of b (its lowest s virtual), tabulated on a uniform grid by
``scripts/cmm_frontier_shapes.py --table`` straight from the orbitals; here a natural cubic spline of it (C^2). The
coupling has the radial shape of the orbitals, a maximum near contact and a decay that steepens slowly, which one
exponential per site cannot have (claude/ct_frontier_shapes.md).

g_ab is the pair's own cutoff, the same rule for every pair with no distance in it: the C-infinity step from 1 at
R_on to 0 at R_off, the outermost distances at which |S_ab| is ``s_on`` and ``s_off`` (``ct_overlap_window``, 0.06
and 0.02). Beyond R_off the channel does not exist; the eight ions' pairs end between 3.5 and 8.2 A. A pair whose
overlap never reaches s_off has no channel; one that never reaches s_on is switched from its maximum.

The table (JSON): ``symbols``, ``r0``, ``h``, ``n`` (the grid r0 + h k in Angstrom), ``S`` {"D>A": [n values]},
``level``, and the orbitals' ``donor`` / ``acceptor`` records.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .channels import switch_step

__all__ = ["load_ct_overlap", "overlap_kernel", "switch_radii", "load_ct_coupling", "coupling_kernel"]


def _spline(y, h):
    """(n - 1, 4) coefficients a + b t + c t^2 + d t^3 (t from the left node) of the natural cubic spline of y."""
    n = len(y)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for k in range(1, n - 1):
        A[k, k - 1:k + 2] = (h, 4 * h, h)
        rhs[k] = 6 * (y[k + 1] - 2 * y[k] + y[k - 1]) / h
    m = np.linalg.solve(A, rhs)                                      # the second derivatives at the nodes
    return np.stack([y[:-1], (y[1:] - y[:-1]) / h - h * (2 * m[:-1] + m[1:]) / 6, m[:-1] / 2,
                     (m[1:] - m[:-1]) / (6 * h)], 1)


def switch_radii(S, r, s_on, s_off):
    """(R_on, R_off): the outermost radii at which |S| is s_on and s_off (linear between grid points). (0, 0) if |S|
    never reaches s_off; R_on = the radius of the largest |S| if it never reaches s_on."""
    a = np.abs(np.asarray(S, dtype=float))

    def outer(v):
        k = np.nonzero(a >= v)[0]
        if not len(k):
            return None
        k = int(k[-1])
        if k == len(a) - 1:
            raise ValueError(f"the overlap is still {a[-1]:.3g} >= {v} at the end of the grid ({r[-1]:.2f} A)")
        return float(r[k] + (r[k + 1] - r[k]) * (a[k] - v) / (a[k] - a[k + 1]))
    off = outer(s_off)
    if off is None:
        return 0.0, 0.0
    on = outer(s_on)
    return (float(r[int(np.argmax(a))]) if on is None else on), off


def load_ct_overlap(path, symbols, window=(0.06, 0.02), dtype=torch.float64) -> dict:
    """The table for a species list: {"coef": (S, S, n - 1, 4) spline coefficients [donor, acceptor], "grid": (2,)
    r0 and h in Angstrom, "r_on", "r_off": (S, S) the pairs' switch radii in Angstrom, "level"}."""
    d = json.load(open(path))
    s_on, s_off = (float(v) for v in window)
    if not 0 < s_off < s_on:
        raise ValueError(f"overlap window {tuple(window)}: needs s_on > s_off > 0")
    r0, h, n = float(d["r0"]), float(d["h"]), int(d["n"])
    r = r0 + h * np.arange(n)
    S = len(symbols)
    coef, on, off = np.zeros((S, S, n - 1, 4)), np.zeros((S, S)), np.zeros((S, S))
    for a, sa in enumerate(symbols):
        for b, sb in enumerate(symbols):
            if f"{sa}>{sb}" not in d["S"]:
                raise KeyError(f"{path}: no overlap for the channel {sa}>{sb} (the table has {d.get('symbols')})")
            y = np.asarray(d["S"][f"{sa}>{sb}"], dtype=float)
            if len(y) != n:
                raise ValueError(f"{path}: {sa}>{sb} has {len(y)} values, the grid {n}")
            coef[a, b] = _spline(y, h)
            on[a, b], off[a, b] = switch_radii(y, r, s_on, s_off)
    t = lambda x: torch.as_tensor(x, dtype=dtype)                    # noqa: E731
    return {"coef": t(coef), "grid": t([r0, h]), "r_on": t(on), "r_off": t(off), "level": d.get("level", "")}


def load_ct_coupling(path, symbols, dtype=torch.float64) -> dict:
    """The coupling table (``scripts/cmm_ct_coupling.py``) for a species list: {"coef": (S, S, n - 1, 4) spline
    coefficients of W_ab(R) in Hartree [donor, acceptor], "grid": (2,) r0 and h in Angstrom, "has": (S, S) bool, the
    ordered pairs the table holds (the others are zero), "level"}."""
    d = json.load(open(path))
    r0, h, n = float(d["r0"]), float(d["h"]), int(d["n"])
    S = len(symbols)
    coef, has = np.zeros((S, S, n - 1, 4)), np.zeros((S, S), dtype=bool)
    for a, sa in enumerate(symbols):
        for b, sb in enumerate(symbols):
            y = d["W"].get(f"{sa}>{sb}")
            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            if len(y) != n:
                raise ValueError(f"{path}: {sa}>{sb} has {len(y)} values, the grid {n}")
            coef[a, b], has[a, b] = _spline(y, h), True
    out = {"coef": torch.as_tensor(coef, dtype=dtype), "grid": torch.as_tensor([r0, h], dtype=dtype),
           "has": torch.as_tensor(has), "level": d.get("level", "")}
    if "dW_donor" in d:                 # the breathing derivatives, on every breath_every-th grid point
        kb = int(d.get("breath_every", 1))
        nb = len(range(0, n, kb))
        dco = np.zeros((2, S, S, nb - 1, 4))
        hasd = np.zeros((S, S), dtype=bool)
        for a, sa in enumerate(symbols):
            for b, sb in enumerate(symbols):
                key = f"{sa}>{sb}"
                if key in d["dW_donor"]:
                    for q, name in enumerate(("dW_donor", "dW_acceptor")):
                        y = np.asarray(d[name][key], dtype=float)
                        if len(y) != nb:
                            raise ValueError(f"{path}: {name} {key} has {len(y)} values, its grid {nb}")
                        dco[q, a, b] = _spline(y, h * kb)
                    hasd[a, b] = True
        out.update(dcoef=torch.as_tensor(dco, dtype=dtype), dgrid=torch.as_tensor([r0, h * kb], dtype=dtype),
                   has_d=torch.as_tensor(hasd), breath_params=d.get("breath_params", ""))
    return out


def _eval(coef, grid, r_ang):
    r0, h = grid[0], grid[1]
    n = coef.shape[1]
    k = ((r_ang - r0) / h).detach().floor().clamp(0, n - 1).long()
    t = r_ang - (r0 + h * k)
    c = coef[torch.arange(coef.shape[0], device=coef.device), k]
    return c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))


def coupling_kernel(coef, grid, r_on, r_off, si, sj, r_ang, dcoef=None, dgrid=None, dlam_i=None, dlam_j=None):
    """2 [W_ab g]^2 (Hartree^2) of the channels donor species ``si`` -> acceptor species ``sj`` at ``r_ang``
    (Angstrom): the spline of the tabulated frontier coupling times the pair's C-infinity switch, with the factor 2 of
    a closed-shell donor pair (the two-state channel's W^2). Exactly 0 at and beyond R_off, with every derivative.
    With ``dcoef`` (2, S, S, m - 1, 4), ``dgrid`` and the breathing amplitudes ``dlam_i`` of the donors and ``dlam_j``
    of the acceptors (per channel; electrons, > 0 an expansion): W = W_ab + dW_donor dlam_i + dW_acceptor dlam_j."""
    W = _eval(coef[si, sj], grid, r_ang)
    if dcoef is not None:
        W = W + _eval(dcoef[0, si, sj], dgrid, r_ang) * dlam_i + _eval(dcoef[1, si, sj], dgrid, r_ang) * dlam_j
    return 2.0 * (W * pair_switch(r_on, r_off, si, sj, r_ang)) ** 2


def overlap_kernel(coef, grid, r_on, r_off, si, sj, r_ang):
    """[S g]^2 of the channels donor species ``si`` -> acceptor species ``sj`` at the distances ``r_ang`` (Angstrom):
    the spline of the tabulated overlap times the pair's C-infinity switch. Exactly 0 at and beyond R_off, with every
    derivative; differentiable in ``r_ang``."""
    r0, h = grid[0], grid[1]
    n = coef.shape[2]
    x = (r_ang - r0) / h
    k = x.detach().floor().clamp(0, n - 1).long()
    t = r_ang - (r0 + h * k)                                         # extrapolates the end intervals
    c = coef[si, sj, k]
    S = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
    return (S * pair_switch(r_on, r_off, si, sj, r_ang)) ** 2


def pair_switch(r_on, r_off, si, sj, r_ang):
    """g of the channels ``si`` -> ``sj`` at ``r_ang`` (Angstrom): 1 inside the pair's R_on, the C-infinity step to
    exactly 0 at R_off (with every derivative), 0 beyond and for a pair without a channel (R_off = 0)."""
    on, off = r_on[si, sj], r_off[si, sj]
    live = off > on
    g = 1.0 - switch_step((r_ang - on) / torch.where(live, off - on, torch.ones_like(off)), "smooth")
    return torch.where(live, g, torch.zeros_like(g))
