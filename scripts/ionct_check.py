"""Check the fixed ion electrostatics and the free-ion induction against ALMO-EDA (ion ct ff plan, R2).

    python scripts/ionct_check.py CT_SETS [--params params/ions_wb97mv_tzvpd.json] [--no-c]

CT_SETS is ct_model's data/ct_sets (the folded ct-* sets, one JSON row per EDA). For the ct-scan pair scans and
the full ct-sat / ct-mbe / ct-walk clusters it prints

    elst      frozen shaped-density electrostatics (rsfff.ff.ionct.kernels.permanent_pair_energy) vs EDA cls_elec
    ind       mutual induced dipoles + quadrupoles with the free-ion alpha and C vs EDA pol
    c_match   the uniform scale on alpha and C that makes ind equal to pol: what confinement has to supply

Energies kJ/mol.
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


def load(root, wf):
    p = Path(root) / f"{wf}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def evaluate(r, tab, match=True):
    sym = r["symbols"]
    X = torch.tensor(r["coords"], dtype=torch.float64)
    s = tab.species_index(torch.tensor([Z[a] for a in sym]))
    b = torch.zeros(len(sym), dtype=torch.long)
    out = {"n": len(sym), "cls_elec": r["eda"].get("cls_elec"), "pol": r["eda"]["pol"],
           "elst": K.permanent_pair_energy(X, s, K.all_pairs(b), tab).sum().item() * KJ,
           "ind": K.induction(X, s, b, tab).energy.item() * KJ, "c_match": float("nan")}
    if match:
        lo, hi = 0.0, 1.5
        for _ in range(14):
            c = 0.5 * (lo + hi)
            sc = torch.full((len(sym),), c, dtype=torch.float64)
            v = K.induction(X, s, b, tab, alpha_scale=sc, c_scale=sc).energy.item() * KJ
            lo, hi = (c, hi) if v > out["pol"] else (lo, c)
        out["c_match"] = 0.5 * (lo + hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct_sets")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--no-c", action="store_true", help="skip the confinement-scale match (slow)")
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    print("pair    R/A   cls_elec     elst |     pol   ind(free) | c_match")
    for r in sorted((r for r in load(a.ct_sets, "ct-scan") if r["stage"] == "label" and "eda" in r
                     and (r.get("tags") or {}).get("scan") == "pair"),
                    key=lambda r: (r["tags"]["scan_pair"], r["tags"]["scan_r"])):
        d = evaluate(r, tab, not a.no_c)
        print(f"{r['tags']['scan_pair']:6} {r['tags']['scan_r']:5.2f} {d['cls_elec']:9.2f} {d['elst']:9.2f} | "
              f"{d['pol']:8.2f} {d['ind']:9.2f} | {d['c_match']:.3f}")
    res = []
    for wf in ("ct-sat", "ct-mbe", "ct-walk"):
        seen = set()
        for r in load(a.ct_sets, wf):
            t = r.get("tags") or {}
            if r["stage"] != "label" or "eda" not in r or "mbe_set" not in t or t.get("mbe_order") != t.get("mbe_n") \
                    or t["mbe_set"] in seen:
                continue
            seen.add(t["mbe_set"])
            res.append({"wf": wf, **evaluate(r, tab, not a.no_c)})
    el = np.array([[d["cls_elec"], d["elst"], d["n"]] for d in res])
    po = np.array([[d["pol"], d["ind"], d["c_match"]] for d in res])
    print(f"\n{len(res)} clusters: elst - cls_elec per ion: mean {np.mean((el[:, 1] - el[:, 0]) / el[:, 2]):+.2f}, "
          f"rms {np.sqrt(np.mean(((el[:, 1] - el[:, 0]) / el[:, 2]) ** 2)):.2f} kJ/mol")
    msg = f"pol: ALMO mean {po[:, 0].mean():.1f}, free-ion induction mean {po[:, 1].mean():.1f} kJ/mol"
    if not a.no_c:
        msg += f"; c_match median {np.median(po[:, 2]):.2f} (range {po[:, 2].min():.2f}-{po[:, 2].max():.2f})"
    print(msg)


if __name__ == "__main__":
    main()
