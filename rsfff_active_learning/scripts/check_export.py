#!/usr/bin/env python3
"""Does a store export reproduce the qchem_roundtrip training files?

    python scripts/check_export.py EXPORT_DIR OLD_FILE_OR_DIR [...]
    python scripts/check_export.py /tmp/export $RSFFF_DATA/train/data/clusters

Matches frames by geometry (Q-Chem's standard orientation makes the same calculation come
back with the same coordinates) and, for every match, compares energy, forces, the EDA terms,
the fragment labels and the multipoles. Reports per size: matched, only in the old files, only
in the export, and the largest difference of each label. Exit status 1 on any mismatch above
1e-9 (relative for multipoles) -- the store path must be bitwise the old path, or explain why.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train" / "scripts"))

from check_data import geometry_key  # noqa: E402

LABELS = ("energy", "fragment_energies", "fragment_dipoles", "fragment_second_moments",
          "dipole", "quadrupole")


def load(paths):
    from easyal import iter_extxyz

    out = {}
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.xyz")) if p.is_dir() else [p]
        for f in files:
            for fr in iter_extxyz(f):
                key = geometry_key(fr["arrays"]["species"], np.asarray(fr["arrays"]["pos"]))
                out.setdefault(key, (fr, f.name))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path)
    ap.add_argument("old", nargs="+")
    a = ap.parse_args(argv)
    new = load([a.export / "eda", a.export / "force"])
    old = load(a.old)
    sizes = defaultdict(lambda: {"matched": 0, "only_old": 0, "only_new": 0})
    worst = defaultdict(float)
    for key, (fr, _) in old.items():
        n = len(fr["arrays"]["species"]) // 3
        if key not in new:
            sizes[n]["only_old"] += 1
            continue
        sizes[n]["matched"] += 1
        nf = new[key][0]
        worst["forces"] = max(worst["forces"], float(np.abs(
            np.asarray(fr["arrays"]["forces"]) - np.asarray(nf["arrays"]["forces"])).max()))
        for k in [k for k in fr["info"] if k.startswith("eda_")] + list(LABELS):
            if k in fr["info"] and k in nf["info"]:
                x = np.atleast_1d(np.asarray(fr["info"][k], float))
                y = np.atleast_1d(np.asarray(nf["info"][k], float))
                scale = max(1.0, float(np.abs(x).max())) if k in ("dipole", "quadrupole") else 1.0
                worst[k] = max(worst[k], float(np.abs(x - y).max()) / scale)
            elif k in fr["info"]:
                worst[f"{k} (missing in export)"] += 1
    for key, (fr, _) in new.items():
        if key not in old:
            sizes[len(fr["arrays"]["species"]) // 3]["only_new"] += 1
    print(f"{'size':>5s} {'matched':>8s} {'only_old':>9s} {'only_new':>9s}")
    for n in sorted(sizes):
        s = sizes[n]
        print(f"{'w' + str(n):>5s} {s['matched']:8d} {s['only_old']:9d} {s['only_new']:9d}")
    print("largest differences on matched frames:")
    bad = False
    for k, v in sorted(worst.items()):
        flag = v > 1e-9
        bad |= flag
        print(f"  {k:32s} {v:.3g}{'   <-- MISMATCH' if flag else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
