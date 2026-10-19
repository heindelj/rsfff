"""The dimer-only CT channel model inside the force field's electrostatics, against ALMO E_CT (ion ct ff plan, R3).

    python scripts/ionct_ct_check.py CT_SETS [--params ...] [--out ionct_ct_check.json]

For every full cluster of ct-sat / ct-mbe / ct-walk: the induced moments with alpha and C scaled uniformly so the
cluster's induction equals its ALMO pol (a stand-in for the confinement the network will emit), then E_CT from
rsfff.ff.ionct.ct with the isolated-pair overlap gap (ov_scale = 1) for WKB 'two_sided' | 'smooth' | 'none',
and the uniform ov_scale (two-sided WKB) or ov_add (smooth) that would reproduce ALMO E_CT -- what the network has
to supply on average. kJ/mol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K

KJ = 2625.4996
Z = {"Na": 11, "Cl": 17}


def rows(root, wf):
    p = Path(root) / f"{wf}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def bisect(f, lo, hi, target, steps=16, increasing=True):
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if (f(m) > target) == increasing:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct_sets")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tab, chan = IonTable.load(a.params), CT.ChannelTable.load(a.params)
    out = []
    for wf in ("ct-sat", "ct-mbe", "ct-walk"):
        seen = set()
        for r in rows(a.ct_sets, wf):
            t = r.get("tags") or {}
            if r["stage"] != "label" or "eda" not in r or "mbe_set" not in t or t.get("mbe_order") != t.get("mbe_n") \
                    or t["mbe_set"] in seen or int(t["mbe_n"]) < 3:
                continue
            seen.add(t["mbe_set"])
            sym = r["symbols"]
            X = torch.tensor(r["coords"], dtype=torch.float64)
            s = tab.species_index(torch.tensor([Z[x] for x in sym]))
            b = torch.zeros(len(sym), dtype=torch.long)
            n = len(sym)
            ind = lambda c: K.induction(X, s, b, tab, alpha_scale=torch.full((n,), c, dtype=torch.float64),
                                        c_scale=torch.full((n,), c, dtype=torch.float64))
            c = bisect(lambda c: ind(c).energy.item() * KJ, 0.0, 1.5, r["eda"]["pol"], increasing=False)
            x = ind(c).x
            d = {"wf": wf, "set": t["mbe_set"], "n": n, "ct": r["eda"]["ct"], "pol": r["eda"]["pol"], "c": c}
            for wkb in ("two_sided", "smooth", "none"):
                d[wkb] = CT.ct_energy(X, s, b, tab, chan, x_ind=x, wkb=wkb).energy.item() * KJ
            e_ct = lambda o: CT.ct_energy(X, s, b, tab, chan, x_ind=x, ov_scale=torch.tensor(o, dtype=torch.float64),
                                          wkb="two_sided").energy.item() * KJ
            d["ov_match"] = bisect(e_ct, 0.0, 20.0, r["eda"]["ct"], increasing=True)
            e_add = lambda o: CT.ct_energy(X, s, b, tab, chan, x_ind=x, ov_add=torch.tensor(o, dtype=torch.float64),
                                           wkb="smooth").energy.item() * KJ
            d["add_match"] = bisect(e_add, -0.3, 2.0, r["eda"]["ct"], increasing=True)
            out.append(d)
            print(f"{wf:8} {t['mbe_set']:24} n {n:2d} c {c:.2f} | ALMO {d['ct']:7.2f} | two {d['two_sided']:7.2f} "
                  f"smooth {d['smooth']:7.2f} none {d['none']:7.2f} | ov_match {d['ov_match']:.2f} add_match {d['add_match']:+.3f}", flush=True)
    A = np.array([[d["ct"], d["two_sided"], d["smooth"], d["none"], d["ov_match"], d["add_match"]] for d in out])
    for k, name in enumerate(("two_sided", "smooth", "none"), start=1):
        print(f"{name:9} rms {np.sqrt(np.mean((A[:, k] - A[:, 0]) ** 2)):6.2f}  median ratio {np.median(A[:, k] / A[:, 0]):.2f}")
    print(f"ov_scale to match: median {np.median(A[:, 4]):.2f}, range {A[:, 4].min():.2f}-{A[:, 4].max():.2f}")
    print(f"uniform ov_add to match (smooth WKB): median {np.median(A[:, 5]):+.3f} Eh, "
          f"range {A[:, 5].min():+.3f} to {A[:, 5].max():+.3f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
