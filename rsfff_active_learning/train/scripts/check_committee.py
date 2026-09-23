#!/usr/bin/env python3
"""Load a trained committee the way the active-learning loop will, and look at it.

    python scripts/check_committee.py runs/film_committee [--frames 20] [--device cpu]

Loads through ``rsfff_al.committee.Committee.load`` -- on the CPU by default,
which is where the loop's sampling and selection run, so a committee fitted on GPUs that
passes this is one the loop can use. Then, on the first ``--frames`` frames of every cluster
file the members were fitted on (these include training frames: this is a sanity check, not
a holdout score), it reports per size:

    E/water    committee-mean total energy vs the label, kJ/mol per water
    F_MAE      committee-mean forces vs the label (converted from Hartree/Bohr), kJ/mol/A
    sig_E/w    member spread of the energy, kJ/mol per water
    sig_F      DeePMD model_devi_f (largest per-atom force deviation), kJ/mol/A

Exit status is non-zero if anything is non-finite or a member fails to load.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
warnings.filterwarnings("ignore", message="torch_cluster is not installed")

KJMOL = 2625.499639
MEV = 27211.386245988
#: the cluster files store forces in Hartree/Bohr (see rsfff.train.data); the model is per Angstrom
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("committee", type=Path)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from easyal import read_extxyz

    from rsfff_al.assess import score_frames
    from rsfff_al.committee import Committee

    manifest = json.loads((args.committee / "committee.json").read_text())
    c = Committee.load(args.committee, device=args.device)
    print(f"{c.n_members} member(s) loaded on {args.device}; val losses "
          f"{[m.get('val_loss') for m in manifest['members']]}")
    files = [Path(p) for p in manifest.get("cluster_files") or []]
    files += [Path(p) for p in manifest.get("force_files") or []]
    ok = True
    print(f"{'file':44s} {'n':>4s} {'E/water':>8s} {'F_MAE':>8s} {'sig_E/w':>8s} {'sig_F':>8s}"
          "   (kJ/mol, kJ/mol/A)")
    for path in files:
        frames = read_extxyz(path)[:args.frames]
        rows = score_frames(c, frames)
        ok &= not any(r["failed"] for r in rows)
        mean = lambda k: float(np.mean([r[k] for r in rows if k in r])) if rows else math.nan  # noqa: E731
        print(f"{path.parent.name + '/' + path.name:44s} {len(rows):4d} {mean('e_err'):8.3f} "
              f"{mean('f_mae'):8.3f} {mean('sigma_e'):8.4f} {mean('sigma_f'):8.3f}")
    print("all predictions finite" if ok else "NON-FINITE predictions")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
