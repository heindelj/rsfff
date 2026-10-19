"""Fixed pair functions of the ion CT channels (ion ct ff plan, R3) from the ct-scan pair scans.

    python scripts/ion_ct_pairs.py CT_SETS [--params params/ions_wb97mv_tzvpd.json]

CT_SETS is ct_model's data/ct_sets. For every pair-scan point and both directed channels D -> A (ALMO-CTA
Roothaan-step channel energy E and charge Q, the orthogonal-complement block counted in), the target is the channel's
share of the full ALMO E_CT, eps' = E * (E_CT / sum_PCT), so that an isolated pair reproduces E_CT exactly (the
higher-order relaxation the channel inversion leaves out is ~16-35 % here). With W = E^2 / Q the arrowhead root of
one channel is eps' when

    Delta_eff = eps' - W / eps',

and the gap is split as Delta_eff = g + s [phibar_D - phibar_A]_pair + Delta^ov(R), the potentials being the force
field's own (shaped densities + induced moments, rsfff.ff.ionct.kernels) with alpha and C scaled by the confinement
that reproduces the pair's ALMO pol. g is the minimum of Delta_eff - s dphi over the fitted range (for Cl -> Na at its
largest R), so Delta^ov >= 0 and vanishes there; for Cl -> Na it is compared with EA(Cl) - IP(Na) in the output.

Na-Cl beyond ~3 A mixes in the covalent (neutral) state: E_CT / sum_PCT rises to 2 at 4.4 A. The ratio is frozen at
its value at --ratio-cap (3.0 A) beyond that: a pair that far apart is not a CT channel the model is meant to describe.

Writes the 'channels' section of the parameter file: per type R (A), lnW, dov, dphi (Hartree) knots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import kernels as K

KJ = 2625.4996
Z = {"Na": 11, "Cl": 17}
#: fitted range (A) and the smallest channel charge (me-) used per type
RANGES = {"Cl>Na": (1.9, 4.6, 0.01), "Na>Cl": (1.9, 3.2, 0.009), "Cl>Cl": (2.8, 8.0, 0.01), "Na>Na": (2.8, 6.0, 0.002)}


def confinement_match(X, s, b, tab, pol_kj, steps=16):
    lo, hi = 0.0, 1.5
    for _ in range(steps):
        c = 0.5 * (lo + hi)
        sc = torch.full((2,), c, dtype=torch.float64)
        v = K.induction(X, s, b, tab, alpha_scale=sc, c_scale=sc).energy.item() * KJ
        lo, hi = (c, hi) if v > pol_kj else (lo, c)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct_sets")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--ratio-cap", type=float, default=3.0)
    ap.add_argument("--s", type=float, default=1.0)
    ap.add_argument("--cutoff", type=float, default=8.0)
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    rows = [json.loads(l) for l in open(Path(a.ct_sets) / "ct-scan.jsonl")]
    rows = [r for r in rows if r["stage"] == "label" and (r.get("tags") or {}).get("scan") == "pair"
            and r.get("cta") and "eda" in r]
    pts = {}
    for r in rows:
        sym = r["symbols"]
        X = torch.tensor(r["coords"], dtype=torch.float64)
        s = tab.species_index(torch.tensor([Z[x] for x in sym]))
        b = torch.zeros(2, dtype=torch.long)
        E = np.array(r["cta"]["E"], float)[:2, :2]
        Q = np.array(r["cta"]["Q"], float)[:2, :2]
        ce = r["cta"].get("E_to_complement") or [0.0, 0.0]
        cq = r["cta"].get("Q_to_complement") or [0.0, 0.0]
        c = confinement_match(X, s, b, tab, r["eda"]["pol"])
        sc = torch.full((2,), c, dtype=torch.float64)
        x = K.induction(X, s, b, tab, alpha_scale=sc, c_scale=sc).x
        PD = K.frontier_potential_matrix(X, s, b, tab, "donor", x=x)
        PA = K.frontier_potential_matrix(X, s, b, tab, "acceptor", x=x)
        R = float(r["tags"]["scan_r"])
        ratio = r["eda"]["ct"] / (E[0, 1] + E[1, 0] + (ce[0] or 0) + (ce[1] or 0))
        for i, j in ((0, 1), (1, 0)):
            name = f"{sym[i]}>{sym[j]}"
            pts.setdefault(name, {})[R] = dict(E=E[i, j] + (ce[i] or 0), Q=Q[i, j] + (cq[i] or 0), ratio=ratio, c=c,
                                               dphi=a.s * (PD[i, j] - PA[j, i]).item(), pair=r["tags"]["scan_pair"])
    types, report = {}, {}
    for name, (r0, r1, qmin) in RANGES.items():
        d = pts.get(name, {})
        Rs = sorted(R for R in d if r0 - 1e-9 <= R <= r1 + 1e-9 and d[R]["Q"] > qmin and d[R]["E"] < 0)
        if len(Rs) < 4:
            continue
        rat = np.array([d[R]["ratio"] for R in Rs])
        if d[Rs[0]]["pair"] == "Na-Cl":
            cap = np.interp(a.ratio_cap, Rs, rat)
            rat = np.where(np.array(Rs) > a.ratio_cap, cap, rat)
        e = np.array([d[R]["E"] for R in Rs]) / KJ
        q = np.array([d[R]["Q"] for R in Rs]) * 1e-3
        W = e * e / q
        ep = e * rat
        D_eff = ep - W / ep
        dphi = np.array([d[R]["dphi"] for R in Rs])
        rest = D_eff - dphi
        g = float(rest.min())          # Delta^ov >= 0 everywhere (= 0 at the minimum, usually the largest R)
        dov = rest - g
        tail = max(1.0, float(-np.log(max(dov[-2], 1e-6) / max(dov[-3], 1e-6)) / (Rs[-2] - Rs[-3]))) if len(Rs) > 3 else 2.0
        types[name] = {"R": Rs, "lnW": np.log(W).tolist(), "dov": dov.tolist(), "dphi": dphi.tolist(), "g": g,
                       "dov_tail": tail, "Q_min": qmin, "ratio": rat.tolist(), "confinement": [d[R]["c"] for R in Rs]}
        report[name] = (g, len(Rs), Rs[0], Rs[-1], float(dov.min()), float(dov.max()))
    p = json.load(open(a.params))
    p["channels"] = {"s": a.s, "cutoff": a.cutoff, "taper": 1.0, "ratio_cap": a.ratio_cap,
                     "source": "ct_model ct-scan pair scans (wB97M-V/def2-TZVPD ALMO-EDA2 + CTA)", "types": types}
    json.dump(p, open(a.params, "w"), indent=1)
    ions = p["ions"]
    for name, (g, n, r0, r1, lo, hi) in report.items():
        D, A = name.split(">")
        phys = ions[D]["I_donor"] - ions[A]["A_acceptor"] if ions[A]["A_acceptor"] is not None else float("nan")
        print(f"{name:6} {n:3d} points {r0:.1f}-{r1:.1f} A  g {g:+.4f} Eh (I_D - A_A {phys:+.4f})  "
              f"Delta^ov {lo:+.3f} .. {hi:+.3f}")


if __name__ == "__main__":
    main()
