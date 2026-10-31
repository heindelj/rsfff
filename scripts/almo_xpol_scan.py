"""Electrical vs exchange polarization along ion-pair scans and in the Cl- cages: the Hartree-product split of the
ALMO polarization (rsfff.qcgen.almo, ``hartree=True``; docs/almo_pcm.md, "Exchange polarization").

    python scripts/almo_xpol_scan.py run OUT.json --pairs Na-Cl Cl-Cl Na-Na [--R ...] [--cages oct6:2.6 ...]
    python scripts/almo_xpol_scan.py plot OUT.json         # -> OUT.png

Per geometry: pol, pol_elec (I_H), pol_exch (X_POL = dP_onH + R_AS), the endpoint convention (pol_elec_act,
pol_exch_act, H_cost), the induced dipoles of both states, and optionally (--disp) the change of the dispersion
estimate at AS(x_H) and x_P. Cages: Cl- at the origin with Na+ on the ion_cage_response shapes (pair, a90, a109,
a120, a180, tri3, sq4, tet4, oct6). ``plot`` is a function a notebook can call."""
import argparse
import json
import sys
import time

import numpy as np

from rsfff.qcgen.almo import Fragment, almo_eda, BOHR

Q = {"Li": 1, "Na": 1, "K": 1, "F": -1, "Cl": -1, "Br": -1}
DEFAULT_R = {"unlike": [1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0, 8.0],
             "like": [2.8, 3.0, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0, 8.0]}
SHAPES = {
    "pair": [[1, 0, 0]], "a90": [[1, 0, 0], [0, 1, 0]], "a109": [[1, 1, 1], [1, -1, -1]],
    "a120": [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0]], "a180": [[1, 0, 0], [-1, 0, 0]],
    "tri3": [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]],
    "sq4": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
    "tet4": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    "oct6": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
}
TERMS = ["pol", "pol_elec", "pol_exch", "dP_onH", "R_AS", "pol_elec_act", "pol_exch_act"]


def shell(shape, R):
    v = np.asarray(SHAPES[shape], float)
    return R * v / np.linalg.norm(v, axis=1, keepdims=True)


def geometries(args):
    for pair in args.pairs:
        a, b = pair.split("-")
        grid = args.R or DEFAULT_R["like" if Q[a] == Q[b] else "unlike"]
        for R in grid:
            yield {"kind": "pair", "pair": pair, "R": R}, [Fragment([a], np.zeros((1, 3)), charge=Q[a]),
                                                         Fragment([b], np.array([[0.0, 0.0, R]]), charge=Q[b])]
    for c in args.cages:
        shape, R = c.split(":")[0], float(c.split(":")[1])
        frags = [Fragment(["Cl"], np.zeros((1, 3)), charge=-1)]
        frags += [Fragment(["Na"], s.reshape(1, 3), charge=1) for s in shell(shape, R)]
        yield {"kind": "cage", "shape": shape, "R": R}, frags


def run(args):
    try:
        recs = json.load(open(args.out))
    except (FileNotFoundError, json.JSONDecodeError):
        recs = []
    done = {json.dumps({k: v for k, v in r.items() if k in ("kind", "pair", "shape", "R")}, sort_keys=True) for r in recs}
    for key, frags in geometries(args):
        if json.dumps(key, sort_keys=True) in done:
            continue
        t = time.time()
        out = almo_eda(frags, xc=args.xc, basis=args.basis, frozen_decomp="classical", with_ct=not args.no_ct,
                       hartree=True, hartree_disp=args.disp, verbose=0)
        st = out["states"]
        rec = {**key, "terms": {k: float(v) for k, v in out["terms"].items()},
               "dmu_H": [[float(x) for x in d] for d in st["dmu_H"]], "dmu_P": [[float(x) for x in d] for d in st["dmu_P"]],
               "hartree_iters": out["info"]["hartree"]["iters"], "scfmi_iters": out["info"]["scfmi_iters"],
               "time": out["info"]["time"]}
        recs.append(rec)
        json.dump(recs, open(args.out, "w"), indent=1)
        tm = rec["terms"]
        tag = f"{key['pair']} R {key['R']:4.2f}" if key["kind"] == "pair" else f"{key['shape']:5s} R {key['R']:4.2f}"
        print(f"{tag}: pol {tm['pol']:8.2f} = elec {tm['pol_elec']:8.2f} + exch {tm['pol_exch']:7.2f} "
              f"(dP {tm['dP_onH']:7.2f} R_AS {tm['R_AS']:7.2f}) | endpoint elec {tm['pol_elec_act']:8.2f} exch {tm['pol_exch_act']:7.2f}"
              + (f" | ct {tm['ct']:7.2f}" if "ct" in tm else "") + f"  ({time.time() - t:.0f}s)", flush=True)


def plot(path, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = json.load(open(path))
    pairs = sorted({r["pair"] for r in recs if r["kind"] == "pair"})
    cages = [r for r in recs if r["kind"] == "cage"]
    ncol = len(pairs) + (1 if cages else 0)
    fig, axes = plt.subplots(2, max(ncol, 1), figsize=(5 * max(ncol, 1), 7.5), constrained_layout=True, squeeze=False)
    for k, pair in enumerate(pairs):
        rs = sorted((r for r in recs if r.get("pair") == pair), key=lambda r: r["R"])
        R = [r["R"] for r in rs]
        ax = axes[0, k]
        for term, style in (("pol", "k-o"), ("pol_elec", "b-s"), ("pol_exch", "r-^")):
            ax.plot(R, [r["terms"][term] for r in rs], style, ms=3, label={"pol": "POL (ALMO)", "pol_elec": "I_H (Hartree product)",
                                                                            "pol_exch": "X_POL = POL - I_H"}[term])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(pair)
        ax.set_ylabel("kJ/mol")
        ax.legend(fontsize=8)
        ax = axes[1, k]
        for term, style in (("dP_onH", "m-v"), ("R_AS", "c-d"), ("pol_exch_act", "r--")):
            ax.plot(R, [r["terms"][term] for r in rs], style, ms=3, label={"dP_onH": "dP on H: P(x_H) - P(x_0)",
                                                                            "R_AS": "R_AS: A(x_P) - A(x_H)",
                                                                            "pol_exch_act": "X_act (endpoint convention)"}[term])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("R / A")
        ax.set_ylabel("kJ/mol")
        ax.legend(fontsize=8)
    if cages:
        ax = axes[0, -1]
        names = sorted({r["shape"] for r in cages}, key=lambda s: list(SHAPES).index(s))
        Rs = sorted({r["R"] for r in cages})
        x = np.arange(len(names))
        w = 0.8 / max(len(Rs), 1)
        for j, R in enumerate(Rs):
            vals = {r["shape"]: r["terms"] for r in cages if r["R"] == R}
            ax.bar(x + j * w, [vals.get(n, {}).get("pol_exch", np.nan) for n in names], w, label=f"X_POL, R {R:g}")
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(names)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Cl- in Na+ cages")
        ax.legend(fontsize=8)
        ax = axes[1, -1]
        for j, R in enumerate(Rs):
            vals = {r["shape"]: r["terms"] for r in cages if r["R"] == R}
            ax.bar(x + j * w, [vals.get(n, {}).get("pol_elec", np.nan) for n in names], w, label=f"I_H, R {R:g}")
            ax.plot(x + j * w, [vals.get(n, {}).get("pol", np.nan) for n in names], "k_", ms=10)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(names)
        ax.set_ylabel("kJ/mol (bars I_H, ticks POL)")
        ax.legend(fontsize=8)
    fig.savefig(out or path.replace(".json", ".png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--pairs", nargs="*", default=["Na-Cl", "Cl-Cl", "Na-Na"])
    r.add_argument("--R", nargs="*", type=float, default=None)
    r.add_argument("--cages", nargs="*", default=[])
    r.add_argument("--basis", default="def2-tzvpd")
    r.add_argument("--xc", default="wb97m-v")
    r.add_argument("--disp", action="store_true")
    r.add_argument("--no-ct", action="store_true")
    p = sub.add_parser("plot")
    p.add_argument("json")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        plot(a.json, a.out)
