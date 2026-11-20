"""The free-ion dispersion table of the response dispersion (``CmmConfig(disp_model="response")``).

The table is built from the ions' multipole response spectra (linear-response TDDFT on the isolated ions: for each
rank l the poles W_k and strengths f_k of alpha_l(i w) = sum_k f_k / (W_k^2 + w^2); ``scripts/cmm_ion_spectra.py``
writes them) by the Casimir-Polder integrals

    C6_ab        = (3 / pi)  int alpha1_a(i w) alpha1_b(i w) dw
    C8_ab        = C8dq_ab + C8dq_ba,     C8dq_ab = (15 / 2 pi) int alpha1_a(i w) alpha2_b(i w) dw

(dipole of a with quadrupole of b), exact for every pair of species: no combination rule. With them the one-pole
frequencies w_l = (4 / pi) int alpha_l(i w)^2 dw / alpha_l(0)^2 of each ion and rank, which the model uses only for
the *ratio* by which confinement changes a coefficient (:meth:`rsfff.ff.cmm.CmmModel.response_dispersion`).

    python -m rsfff.ff.cmm.disp_table SPECTRA.json OUT.json

``OUT.json``: {"level", "symbols", "alpha1", "alpha2", "w1", "w2" (per species), "C6", "C8dq" (species x species)},
atomic units. :func:`load_disp_table` reads it for a species list.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

__all__ = ["build_disp_table", "load_disp_table", "pair_scale"]


def _spectrum(W, f):
    W, f = torch.as_tensor(W, dtype=torch.float64), torch.as_tensor(f, dtype=torch.float64)
    keep = f > 1e-14 * f.max()
    return W[keep], f[keep]


def _static(s):
    W, f = s
    return float((f / (W * W)).sum())


def _cp(a, b):
    """int_0^inf alpha_a(i w) alpha_b(i w) dw of two pole spectra."""
    (Wa, fa), (Wb, fb) = a, b
    x, y = Wa[:, None], Wb[None, :]
    return float((fa[:, None] * fb[None, :] * math.pi / (2 * x * y * (x + y))).sum())


def build_disp_table(spectra: dict, symbols=None) -> dict:
    """The table (plain lists) from ``spectra`` = {symbol: {"W": [per rank], "f": [per rank], "xc", "basis"}}
    (ranks dipole, quadrupole, ...). An ion without quadrupole response in its basis (alpha2 = 0) has C8dq = 0 on
    its quadrupole side and w2 = w1."""
    symbols = list(symbols or spectra)
    sp = {x: [_spectrum(W, f) for W, f in zip(spectra[x]["W"][:2], spectra[x]["f"][:2])] for x in symbols}
    a1 = [_static(sp[x][0]) for x in symbols]
    a2 = [_static(sp[x][1]) for x in symbols]
    has2 = [v > 1e-8 for v in a2]
    w1 = [4 / math.pi * _cp(sp[x][0], sp[x][0]) / a ** 2 for x, a in zip(symbols, a1)]
    w2 = [4 / math.pi * _cp(sp[x][1], sp[x][1]) / a ** 2 if h else w for x, a, h, w in zip(symbols, a2, has2, w1)]
    c6 = [[3 / math.pi * _cp(sp[x][0], sp[y][0]) for y in symbols] for x in symbols]
    c8 = [[15 / (2 * math.pi) * _cp(sp[x][0], sp[y][1]) if hy else 0.0 for y, hy in zip(symbols, has2)] for x in symbols]
    v0 = spectra[symbols[0]]
    return {"level": f"{v0.get('xc', '?')}/{v0.get('basis', '?')} linear response of the isolated ions",
            "units": "atomic units; C8dq[a][b]: dipole of a with quadrupole of b, C8_ab = C8dq[a][b] + C8dq[b][a]",
            "symbols": symbols, "alpha1": a1, "alpha2": [v if h else 0.0 for v, h in zip(a2, has2)], "w1": w1, "w2": w2,
            "C6": c6, "C8dq": c8}


def load_disp_table(path, symbols, dtype=torch.float64) -> dict:
    """{"w": (S, 2) one-pole frequencies of the dipole and quadrupole response, "c6": (S, S), "c8": (S, S) (dipole of
    the row species with quadrupole of the column species), "alpha": (S, 2)} for ``symbols`` from the table file."""
    d = json.load(open(path))
    miss = [x for x in symbols if x not in d["symbols"]]
    if miss:
        raise KeyError(f"dispersion table {Path(path).name} has no {miss} (it has {d['symbols']})")
    ix = [d["symbols"].index(x) for x in symbols]
    t = lambda v: torch.tensor(v, dtype=dtype)                                           # noqa: E731
    sub = lambda M: t(M)[ix][:, ix]                                                      # noqa: E731
    return {"w": torch.stack([t(d["w1"])[ix], t(d["w2"])[ix]], 1), "c6": sub(d["C6"]), "c8": sub(d["C8dq"]),
            "alpha": torch.stack([t(d["alpha1"])[ix], t(d["alpha2"])[ix]], 1)}


def pair_scale(ra, rb, wa, wb):
    """The factor by which a one-pole Casimir-Polder coefficient alpha_a alpha_b w_a w_b / (w_a + w_b) changes when
    the two polarizabilities are scaled by ``ra``, ``rb`` at fixed oscillator mass (w -> w / sqrt(r)):

        s = ra rb (wa + wb) / (wa sqrt(rb) + wb sqrt(ra)),

    r^(3/2) for a like pair, 1 at ra = rb = 1, 0 (smoothly) when either response is fully confined."""
    return ra * rb * (wa + wb) / (wa * torch.sqrt(rb + 1e-30) + wb * torch.sqrt(ra + 1e-30))


if __name__ == "__main__":
    import sys
    tab = build_disp_table(json.load(open(sys.argv[1])))
    json.dump(tab, open(sys.argv[2], "w"), indent=1)
    S = tab["symbols"]
    print(tab["level"])
    print(f"{'':3s} {'alpha1':>8s} {'alpha2':>9s} {'w1':>6s} {'w2':>6s} {'C6':>8s} {'C8':>9s}")
    for k, x in enumerate(S):
        print(f"{x:3s} {tab['alpha1'][k]:8.3f} {tab['alpha2'][k]:9.2f} {tab['w1'][k]:6.3f} {tab['w2'][k]:6.3f} "
              f"{tab['C6'][k][k]:8.2f} {2 * tab['C8dq'][k][k]:9.1f}")
