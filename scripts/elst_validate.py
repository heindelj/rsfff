"""A1 (complete model plan): qcgen frozen-density electrostatics (pyscf) vs Q-Chem ALMO-EDA ``cls_elec``.

    python scripts/elst_validate.py DB SET [SET ...] [--exact] [--limit N] [--out results.jsonl]

DB is an elst_db root (monomers/<name>.npz + index.json). SET is a ct_model ct-*.jsonl (one JSON row per EDA,
kJ/mol) or an rsfff EDA extxyz (eda_cls_elec in Hartree). Monatomic fragments are placed by translation only;
polyatomic fragments need a density at the frame's own geometry (not handled here -> skipped).

Per frame it records E_ref (pyscf frozen electrostatics, robust-fit ee or exact), cls_elec, the difference,
the closest inter-fragment contact and the number of fragments. Energies in kJ/mol.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rsfff.qcgen import elst_ref as E

KJ = 2625.4996
ANG = 1.0 / 0.529177210903
ION = {("Na", 1): "Nap", ("Cl", -1): "Clm", ("Li", 1): "Lip", ("K", 1): "Kp", ("F", -1): "Fm", ("Br", -1): "Brm"}


def frames_jsonl(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("calc") != "eda2" or "eda" not in r or r["eda"].get("cls_elec") is None:
            continue
        yield {"id": r["job"][:12], "symbols": r["symbols"], "coords": np.asarray(r["coords"], float),
               "fragment_idx": np.asarray(r["fragment_idx"]), "fragment_charges": list(r["fragment_charges"]),
               "cls_elec": float(r["eda"]["cls_elec"]), "tags": r.get("tags", {})}


def frames_extxyz(path):
    from ase.io import read
    for k, at in enumerate(read(path, index=":")):
        if "eda_cls_elec" not in at.info:
            continue
        fi = np.asarray(at.arrays["fragment_idx"])
        sym = at.get_chemical_symbols()
        q = []
        for f in range(fi.max() + 1):
            s = [sym[i] for i in np.where(fi == f)[0]]
            q.append(1 if s == ["Na"] else -1 if s == ["Cl"] else 0)
        yield {"id": f"{Path(path).stem}:{k}", "symbols": sym, "coords": at.positions.copy(), "fragment_idx": fi,
               "fragment_charges": q, "cls_elec": float(at.info["eda_cls_elec"]) * KJ, "tags": {"source": Path(path).stem}}


def place_frame(db, fr, cache):
    placed = []
    for f in range(fr["fragment_idx"].max() + 1):
        idx = np.where(fr["fragment_idx"] == f)[0]
        if len(idx) != 1:
            return None
        key = ION.get((fr["symbols"][idx[0]], fr["fragment_charges"][f]))
        if key is None:
            return None
        if key not in cache:
            cache[key] = E.Monomer.load(db / "monomers" / f"{key}.npz")
        mono = cache[key]
        shift = fr["coords"][idx[0]] * ANG - mono.coords[0]
        placed.append(mono.place(shift=shift))
    return placed


def contact(fr):
    X, fi = fr["coords"], fr["fragment_idx"]
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    m = fi[:, None] != fi[None]
    return float(d[m].min()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="elst_validate.jsonl")
    a = ap.parse_args()
    db = Path(a.db)
    cache = {}
    out = open(a.out, "a")
    for s in a.sets:
        gen = frames_extxyz(s) if s.endswith(".xyz") or s.endswith(".extxyz") else frames_jsonl(s)
        n = 0
        t0 = time.time()
        for fr in gen:
            placed = place_frame(db, fr, cache)
            if placed is None:
                continue
            e = E.interaction_energy(placed, exact_ee=a.exact) * KJ
            rec = {"set": Path(s).stem, "id": fr["id"], "n_frag": len(placed), "contact": contact(fr),
                   "e_ref": e, "cls_elec": fr["cls_elec"], "diff": e - fr["cls_elec"],
                   "per_pair": (e - fr["cls_elec"]) / (len(placed) * (len(placed) - 1) / 2),
                   "tags": {k: fr["tags"][k] for k in ("scan", "scan_r", "scan_angle", "mbe_order", "mbe_n", "config_type", "source") if k in fr["tags"]}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            if a.limit and n >= a.limit:
                break
        print(f"{Path(s).stem}: {n} frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
