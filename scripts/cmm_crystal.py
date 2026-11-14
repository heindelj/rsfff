"""Ideal ionic crystals on a cmm model: the energy per formula unit against the lattice constant, its minimum (the
lattice energy relative to the free ions), the lattice constant and the bulk modulus (Birch-Murnaghan fit).

    python scripts/cmm_crystal.py MODEL [--params TABLE.json] [--salts NaCl KBr LiF CsI ...] [--device cpu]

MODEL: a cmm checkpoint, a run tarball or a legacy complete-model checkpoint (rsfff.ff.cmm.load_cmm). Rock salt for
everything but the cesium halides (CsCl structure). Prints a table and the per-salt scans as JSON (--json PATH)."""
from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, load_cmm

KJ = 2625.4996
HARTREE_A3_TO_GPA = 4359.744722                # Hartree / A^3 -> GPa
# experimental reference: lattice constant (A, room temperature), lattice energy (kJ/mol, Born-Haber), bulk modulus (GPa)
EXPT = {"LiF": (4.027, 1036, 69.8), "NaCl": (5.640, 786, 24.0), "KBr": (6.600, 682, 14.8), "CsI": (4.567, 604, 12.7),
        "NaF": (4.634, 923, 46.5), "KCl": (6.293, 715, 17.4), "LiCl": (5.130, 853, 31.7), "CsCl": (4.123, 668, 18.2)}


def split_salt(name):
    a, b = re.findall(r"[A-Z][a-z]?", name)
    return a, b


def cell_of(name, a):
    """(positions (n, 3) A, symbols, cell (3, 3) A, formula units) of the conventional cell."""
    m, x = split_salt(name)
    if m == "Cs":                                   # CsCl structure (B2)
        return torch.tensor([[0.0, 0, 0], [a / 2, a / 2, a / 2]], dtype=torch.float64), [m, x], \
            torch.eye(3, dtype=torch.float64) * a, 1
    fcc = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=torch.float64) * a
    X = torch.cat([fcc, fcc + torch.tensor([a / 2, 0, 0], dtype=torch.float64)])
    return X, [m] * 4 + [x] * 4, torch.eye(3, dtype=torch.float64) * a, 4


def energy(model, name, a, dev, tail=True):
    X, sym, cell, nfu = cell_of(name, a)
    Z = torch.tensor([ATOMIC_NUMBER[s] for s in sym], device=dev)
    fr = torch.zeros(len(sym), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model(model.precompute(X.to(dev), Z, fr, cell=cell.to(dev), dispersion_tail=tail))
    return {k: float(out[k][0]) / nfu for k in ("e_int", "e_elst", "e_pauli", "e_disp", "e_pol", "e_ct")}


def birch_murnaghan(V, E):
    """(V0, E0, B0) of the third-order Birch-Murnaghan fit, E a cubic in x = V^(-2/3): at the minimum
    d2E/dV2 = (4/9) V^(-10/3) d2E/dx2, so B0 = V0 E_VV = (4/9) V0^(-7/3) E_xx (E per formula unit, V A^3)."""
    x = V ** (-2.0 / 3.0)
    c = np.polyfit(x, E, 3)
    cand = [r.real for r in np.roots(np.polyder(c)) if abs(r.imag) < 1e-12 and np.polyval(np.polyder(c, 2), r.real) > 0]
    x0 = min(cand, key=lambda r: abs(r - x.mean()))
    V0 = x0 ** -1.5
    return V0, float(np.polyval(c, x0)), 4.0 / 9.0 * V0 ** (-7.0 / 3.0) * float(np.polyval(np.polyder(c, 2), x0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--params", default=None)
    ap.add_argument("--salts", nargs="+", default=["LiF", "NaCl", "KBr", "CsI"])
    ap.add_argument("--span", type=float, default=0.08, help="lattice constants a0 (1 +- span) around experiment")
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = load_cmm(a.model, params=a.params)
    model.table()
    model.tab = model.tab.to(dev)
    model.to(dev).eval()
    rows, scans = [], {}
    for name in a.salts:
        if any(x not in model.tab.symbols for x in split_salt(name)):
            print(f"{name}: not in the model's species table")
            continue
        a_ex, U_ex, B_ex = EXPT.get(name, (None, None, None))
        grid = np.linspace(1 - a.span, 1 + a.span, a.points) * (a_ex or 5.0)
        es = [energy(model, name, float(x), dev) for x in grid]
        E = np.array([e["e_int"] for e in es])
        V = np.array([x ** 3 if split_salt(name)[0] == "Cs" else x ** 3 / 4 for x in grid])
        V0, E0, B0 = birch_murnaghan(V, E)
        a0 = V0 ** (1 / 3) if split_salt(name)[0] == "Cs" else (4 * V0) ** (1 / 3)
        k = int(np.argmin(E))
        rows.append((name, a0, -E0 * KJ, B0 * HARTREE_A3_TO_GPA, a_ex, U_ex, B_ex, es[k]))
        scans[name] = {"a": grid.tolist(), "terms": es}
    print(f"{'salt':6s} {'a0 (A)':>8s} {'expt':>7s} {'U (kJ/mol)':>11s} {'expt':>6s} {'B0 (GPa)':>9s} {'expt':>6s}   "
          f"terms at the scan minimum (kJ/mol per formula unit)")
    for name, a0, U, B, a_ex, U_ex, B_ex, t in rows:
        terms = " ".join(f"{k[2:]} {v * KJ:8.1f}" for k, v in t.items() if k != "e_int")
        print(f"{name:6s} {a0:8.3f} {a_ex or math.nan:7.3f} {U:11.1f} {U_ex or math.nan:6.0f} {B:9.1f} "
              f"{B_ex or math.nan:6.1f}   {terms}")
    if a.json:
        json.dump(scans, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
