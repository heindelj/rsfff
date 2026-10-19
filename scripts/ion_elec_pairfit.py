"""Refit the ions' permanent shells against *frozen-density pair scans* (complete model plan, A3).

    python scripts/ion_elec_pairfit.py DB [--params IN] [--out OUT] [--lam 1.0] [--ct-sets DIR --walk-eda DIR]

The ESP fit of ion_params.py determines each ion's density from its own potential on scaled vdW surfaces; in
clusters the penetration electrostatics then overbinds by ~6 kJ/mol per ion (ion ct ff plan, R6), which
ion_elec_refit.py fixed by refining the shells against ALMO cls_elec of the EDA frames. Stage A showed that the
pyscf frozen-density electrostatics *is* cls_elec (within 0.01 kJ/mol per pair), so the same refinement can be
done from monomer densities alone: here the shells (Na: b; Cl: b1, b2 and the split n1) are refined against
frozen pair-energy scans Na-Na, Na-Cl, Cl-Cl computed with ``elst_ref`` (exact four-centre ee), with the same
penalty lam * sum (ln p/p0)^2 toward the ESP fit. No EDA enters.

Default: ``--three --r-min 2.3 --lam 1.0`` -> params/ions_wb97mv_tzvpd_pf.json (M2 of the plan, 2026-10-19): per-ion rms vs cls_elec 0.123
kJ/mol on all 3144 frames, clusters -0.025 / 0.165 (EDA refit: 0.130, -0.075 / 0.237). Two Cl shells cannot fit the frozen
Na-Cl curve below ~2.3 A and at 2.8 A at once (residual -1.4 kJ/mol); the third shell fixes it.

The scans are generated once into ``DB/pair_scans_ions.json`` (1.4-8 A; 0.05 A steps to 4 A, 0.25 beyond; the loss
uses R >= 1.8 A, the range the EDA frames sample, with points weighted equally). With ``--ct-sets`` the result is
evaluated against ALMO cls_elec on the CT sets and the walk EDA frames, next to the ESP fit and (``--eda-params``)
the EDA refit, so the three parameter sets can be compared on the same frames.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import kernels as K
from rsfff.qcgen import elst_db as Db
from rsfff.qcgen import elst_ref as E

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ion_elec_refit import KJ, Shells, flat_pairs, ion_potential  # noqa: E402

import dataclasses


class Shells3(Shells):
    """Shells plus a third Cl shell: shell 2 split into (n2 - n3, b2) and (n3, b3), both free, started as a small
    diffuse/compact perturbation (n3 = 0.5 e, b3 = 0.75 b2) and held near the start by the same penalty."""

    def __init__(self, tab: IonTable):
        super().__init__(tab)
        self.p0["n3_Cl"] = torch.tensor(0.5, dtype=torch.float64)
        self.p0["b3_Cl"] = 0.75 * tab.shell_b[self.cl, 1]
        for k in ("n3_Cl", "b3_Cl"):
            self.ln[k] = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def table(self):
        v = self.values()
        nt = self.tab0.Z[self.cl] - self.tab0.charge[self.cl]
        S = self.tab0.shell_b.shape[1]
        b = torch.ones(self.tab0.shell_b.shape[0], max(S, 3), dtype=torch.float64)   # pad like IonTable: b = 1, n = 0
        n = torch.zeros_like(b)
        b[:, :S] = self.tab0.shell_b
        n[:, :S] = self.tab0.shell_n
        b[self.na, 0] = v["b_Na"]
        b[self.cl, 0], b[self.cl, 1], b[self.cl, 2] = v["b1_Cl"], v["b2_Cl"], v["b3_Cl"]
        n[self.cl, 0], n[self.cl, 1], n[self.cl, 2] = v["n1_Cl"], nt - v["n1_Cl"] - v["n3_Cl"], v["n3_Cl"]
        return dataclasses.replace(self.tab0, shell_b=b, shell_n=n)

warnings.filterwarnings("ignore")
ANG = 1.0 / 0.529177210903
ZMAP = {"Na": 11, "Cl": 17}
DBNAME = {"Na": "Nap", "Cl": "Clm"}
PAIRS = (("Na", "Na"), ("Na", "Cl"), ("Cl", "Cl"))


def pair_scans(db: Db.MonomerDB, path: Path):
    """{'Na-Cl': {'R': [...A], 'E': [...kJ/mol]}, ...}: frozen-density pair energies, exact ee."""
    if path.exists():
        return json.load(open(path))
    R = np.concatenate([np.arange(1.4, 4.0, 0.05), np.arange(4.0, 8.01, 0.25)])
    out = {}
    for a, b in PAIRS:
        ma, mb = db.get(DBNAME[a]), db.get(DBNAME[b])
        A = ma.place(shift=-ma.coords[0])
        e = []
        for r in R:
            B = mb.place(shift=np.array([0.0, 0.0, r * ANG]) - mb.coords[0])
            e.append(E.pair_energy(A, B, exact_ee=True)["total"] * KJ)
        out[f"{a}-{b}"] = {"R": R.tolist(), "E": e}
        print(f"scan {a}-{b}: {len(R)} points", flush=True)
    json.dump(out, open(path, "w"))
    return out


def scan_frames(scans, r_min):
    frames, ref = [], []
    for key, d in scans.items():
        a, b = key.split("-")
        for r, e in zip(d["R"], d["E"]):
            if r < r_min:
                continue
            frames.append(([a, b], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])))
            ref.append(e)
    return frames, torch.tensor(ref, dtype=torch.float64)


def cls_elec_frames(ct_sets, walk_eda):
    from ionct_parity import elec_labels
    from rsfff.train.train_ionct import _geom_key, load_frames
    frames = load_frames(ct_sets, walk_eda)
    el = elec_labels(ct_sets, walk_eda)
    fr = [(sym, X) for sym, X, *_ in frames]
    return fr, torch.tensor([el[_geom_key(sym, X)] for sym, X in fr], dtype=torch.float64)


def energies_of(frames, tab):
    pos, sp, pairs, fidx, n = flat_pairs([(s, X) for s, X in frames], tab)
    e = K.permanent_pair_energy(pos, sp, pairs, tab) * KJ
    return torch.zeros(len(frames), dtype=torch.float64).index_add(0, fidx, e), n


def report(name, frames, ref, tab):
    with torch.no_grad():
        e, n = energies_of(frames, tab)
    err = (e - ref) / n
    big = n > 3
    pair = n == 2
    return {"params": name, "per_ion_rms": float(err.pow(2).mean().sqrt()), "per_ion_mean": float(err.mean()),
            "pairs_rms": float(err[pair].pow(2).mean().sqrt()) if pair.any() else None,
            "clusters_per_ion_mean": float(err[big].mean()) if big.any() else None,
            "clusters_per_ion_rms": float(err[big].pow(2).mean().sqrt()) if big.any() else None, "n_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--eda-params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_el.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--r-min", type=float, default=1.8)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--three", action="store_true", help="a third Cl shell (Shells3)")
    ap.add_argument("--ct-sets", default=None)
    ap.add_argument("--walk-eda", default=None)
    a = ap.parse_args()
    db = Db.MonomerDB(a.db)
    scans = pair_scans(db, Path(a.db) / "pair_scans_ions.json")
    frames, ref = scan_frames(scans, a.r_min)
    tab = IonTable.load(a.params)
    sh = (Shells3 if a.three else Shells)(tab)
    opt = torch.optim.LBFGS(sh.parameters(), lr=0.5, max_iter=a.iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        e, n = energies_of(frames, sh.table())
        loss = (((e - ref) / n) ** 2).mean() + a.lam * sh.penalty()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        t1 = sh.table()
        r = torch.linspace(2.0, 10.0, 81, dtype=torch.float64)
        dphi = {s: float(((ion_potential(t1, k, r) - ion_potential(tab, k, r)).abs() / ion_potential(tab, k, r).abs()).max())
                for k, s in enumerate(tab.symbols)}
    rep = {"lam": a.lam, "r_min": a.r_min, "params_old": {k: float(v) for k, v in sh.p0.items()},
           "params_new": {k: float(v) for k, v in sh.values().items()},
           "scan_fit": {"esp": report("esp", frames, ref, tab), "pairfit": report("pairfit", frames, ref, t1)},
           "max_rel_dphi_free_ion_r2-10bohr": dphi}
    # per pair type, per R: residual of the fit
    with torch.no_grad():
        e1, _ = energies_of(frames, t1)
        e0, _ = energies_of(frames, tab)
    k = 0
    rep["scan_residuals"] = {}
    for key, d in scans.items():
        rows = [(r, float(e0[k + i] - ref[k + i]), float(e1[k + i] - ref[k + i]))
                for i, r in enumerate([x for x in d["R"] if x >= a.r_min])]
        k += len(rows)
        rep["scan_residuals"][key] = {"R": [x[0] for x in rows], "esp": [x[1] for x in rows], "pairfit": [x[2] for x in rows]}
    if a.ct_sets:
        fr, cl = cls_elec_frames(a.ct_sets, a.walk_eda)
        rep["cls_elec"] = [report("esp", fr, cl, tab), report("pairfit", fr, cl, t1)]
        if a.eda_params and Path(a.eda_params).exists():
            rep["cls_elec"].append(report("eda_refit", fr, cl, IonTable.load(a.eda_params)))
    print(json.dumps({k: v for k, v in rep.items() if k != "scan_residuals"}, indent=1))
    if a.out:
        P = json.load(open(a.params))
        out = copy.deepcopy(P)
        v = sh.values()
        out["ions"]["Na"]["shells"][0]["b"] = float(v["b_Na"])
        out["ions"]["Cl"]["shells"][0]["b"] = float(v["b1_Cl"])
        out["ions"]["Cl"]["shells"][1]["b"] = float(v["b2_Cl"])
        out["ions"]["Cl"]["shells"][0]["n"] = float(v["n1_Cl"])
        out["ions"]["Cl"]["shells"][1]["n"] = float(t1.shell_n[sh.cl, 1])
        if a.three:
            out["ions"]["Cl"]["shells"].append({"n": float(t1.shell_n[sh.cl, 2]), "b": float(t1.shell_b[sh.cl, 2])})
        out["elec_pairfit"] = rep
        out.pop("channels", None)
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
