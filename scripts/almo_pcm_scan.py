"""ALMO-EDA of an ion pair along a distance scan, in vacuum and in C-PCM at several dielectric constants.

The "decisive scan" of the complete-model plan (Stage D): beyond ~3.6 A the vacuum NaCl ground state is covalent and
the ionic fragmentation labels the wrong state; in a continuum the ion pair stays the ground state. Records every
EDA term, the dipole of each state (the ionic state has |mu| ~ R, the neutral one ~0), the SCF-MI iteration count
and the surface charge sums.

    python scripts/almo_pcm_scan.py run OUT.json [--pair Na Cl] [--R 1.9 2.0 ...] [--eps 1 2 4 78.3553]
                                        [--basis def2-tzvpd] [--xc wb97m-v] [--ng 302]
    python scripts/almo_pcm_scan.py plot OUT.json          # -> OUT.png
    python scripts/almo_pcm_scan.py refrz OUT.json          # recompute the frozen decomposition (disp / mod_pauli) only

``plot`` lives in :func:`plot` so a notebook can call it."""
import argparse
import json
import sys
import time

import numpy as np

from rsfff.qcgen.almo import Fragment, Continuum, almo_eda, BOHR

DEFAULT_R = [1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
DEFAULT_EPS = [1.0, 2.0, 4.0, 78.3553]
TERMS = ["cav", "cls_elec", "disp", "mod_pauli", "frz", "pol", "ct", "int"]


def dipole(mol, dm):
    """Dipole (au) of nuclei + density about the origin."""
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor("int1e_r", comp=3)
    return mol.atom_charges() @ mol.atom_coords() - np.einsum("xij,ji->x", r, dm)


def run(args):
    cat, an = args.pair
    recs = []
    try:
        recs = json.load(open(args.out))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    done = {(r["eps"], r["R"]) for r in recs}
    for eps in args.eps:
        pcm = None if eps == 1.0 else Continuum(eps=eps, lebedev_order={110: 17, 194: 23, 302: 29, 590: 41}[args.ng])
        dm0 = None
        for R in args.R:
            if (eps, R) in done:
                continue
            frags = [Fragment([cat], np.array([[0.0, 0.0, 0.0]]), charge=1),
                     Fragment([an], np.array([[0.0, 0.0, R]]), charge=-1)]
            t = time.time()
            out = almo_eda(frags, xc=args.xc, basis=args.basis, pcm=pcm, frag_dm0=dm0, verbose=0)
            st = out["states"]
            dm0 = st["D_frag_own"]
            rec = {"eps": eps, "R": R, "disp_free": "hf", "terms": {k: float(v) for k, v in out["terms"].items()},
                   "energies": {k: (v if not isinstance(v, list) else [float(x) for x in v]) for k, v in out["energies"].items()},
                   "dipole": {k: float(np.linalg.norm(dipole(st["mol"], st[d]))) for k, d in
                              (("frz", "D_frz"), ("pol", "D_pol"), ("full", "D_full"))},
                   "scfmi_iters": out["info"]["scfmi_iters"], "time": out["info"]["time"]}
            if out["solvent"] is not None:
                s = out["solvent"]
                rec["solvent"] = {"q_sum_frz": float(s["q_frz"].sum()), "q_sum_full": float(s["q_full"].sum()),
                                  "G_frz": float(s["G_frz"]), "G_pol": float(s["G_pol"]), "G_full": float(s["G_full"]),
                                  "G_frag_own": [float(x) for x in s["G_frag_own"]], "npoints": int(len(s["V_frz"]))}
            recs.append(rec)
            json.dump(recs, open(args.out, "w"), indent=1)
            tm = rec["terms"]
            print(f"eps {eps:7.2f} R {R:4.2f}: frz {tm['frz']:8.2f} pol {tm['pol']:7.2f} ct {tm['ct']:7.2f} int {tm['int']:8.2f} "
                  f"| mu full {rec['dipole']['full'] * BOHR:5.2f} eA  ({out['info']['scfmi_iters']} it, {time.time() - t:.0f}s)",
                  flush=True)


def refrz(args):
    """Recompute cav / cls_elec / disp / mod_pauli of every record (fragments + frozen state only; pol and ct kept)."""
    recs = json.load(open(args.out))
    cat, an = args.pair
    dm0 = None
    for rec in sorted(recs, key=lambda r: (r["eps"], r["R"])):
        if rec.get("disp_free") == args.disp_free:
            continue
        eps, R = rec["eps"], rec["R"]
        pcm = None if eps == 1.0 else Continuum(eps=eps, lebedev_order={110: 17, 194: 23, 302: 29, 590: 41}[args.ng])
        frags = [Fragment([cat], np.array([[0.0, 0.0, 0.0]]), charge=1), Fragment([an], np.array([[0.0, 0.0, R]]), charge=-1)]
        t = time.time()
        out = almo_eda(frags, xc=args.xc, basis=args.basis, pcm=pcm, frag_dm0=dm0, with_pol=False, disp_free=args.disp_free)
        dm0 = out["states"]["D_frag_own"]
        for k in ("cav", "cls_elec", "cls_elec_vac", "cls_elec_scr", "disp", "mod_pauli", "frz"):
            rec["terms"][k] = float(out["terms"][k])
        rec["disp_free"] = args.disp_free
        json.dump(recs, open(args.out, "w"), indent=1)
        tm = rec["terms"]
        print(f"eps {eps:7.2f} R {R:4.2f}: frz {tm['frz']:8.2f} disp {tm['disp']:7.2f} mod_pauli {tm['mod_pauli']:8.2f} ({time.time() - t:.0f}s)", flush=True)


def plot(path, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = json.load(open(path))
    epss = sorted({r["eps"] for r in recs})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, term in zip(axes.ravel(), TERMS):
        for eps in epss:
            rs = sorted((r for r in recs if r["eps"] == eps and term in r["terms"]), key=lambda r: r["R"])
            ax.plot([r["R"] for r in rs], [r["terms"][term] for r in rs], "o-", ms=3,
                    label="vacuum" if eps == 1.0 else f"eps = {eps:g}")
        ax.set_title(term)
        ax.set_xlabel("R / A")
        ax.set_ylabel("kJ/mol")
        ax.axhline(0, color="k", lw=0.5)
    axes[0, 0].legend()
    fig.savefig(out or path.replace(".json", ".png"), dpi=130)
    # the state check: dipole of the full SCF vs R
    fig2, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    for eps in epss:
        rs = sorted((r for r in recs if r["eps"] == eps), key=lambda r: r["R"])
        ax.plot([r["R"] for r in rs], [r["dipole"]["full"] * BOHR for r in rs], "o-", ms=3,
                label="vacuum" if eps == 1.0 else f"eps = {eps:g}")
    ax.plot(DEFAULT_R, DEFAULT_R, "k--", lw=0.5, label="e R (ionic)")
    ax.set_xlabel("R / A"); ax.set_ylabel("|mu(full SCF)| / e A"); ax.legend()
    fig2.savefig((out or path.replace(".json", ".png")).replace(".png", "_dipole.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--pair", nargs=2, default=["Na", "Cl"])
    r.add_argument("--R", nargs="+", type=float, default=DEFAULT_R)
    r.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    r.add_argument("--basis", default="def2-tzvpd")
    r.add_argument("--xc", default="wb97m-v")
    r.add_argument("--ng", type=int, default=302)
    f = sub.add_parser("refrz")
    f.add_argument("out")
    f.add_argument("--pair", nargs=2, default=["Na", "Cl"])
    f.add_argument("--basis", default="def2-tzvpd")
    f.add_argument("--xc", default="wb97m-v")
    f.add_argument("--ng", type=int, default=302)
    f.add_argument("--disp-free", default="hf")
    p = sub.add_parser("plot")
    p.add_argument("json")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "refrz":
        refrz(a)
    else:
        plot(a.json, a.out)
