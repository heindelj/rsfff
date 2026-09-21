#!/usr/bin/env python3
"""Load a trained committee the way the active-learning loop will, and look at it.

    python scripts/check_committee.py runs/film_committee [--frames 20] [--device cpu]

Loads through ``active_learning/committee.py`` (``Committee.load``) -- on the CPU by default,
which is where the loop's sampling and selection run, so a committee fitted on GPUs that
passes this is one the loop can use. Then, on the first ``--frames`` frames of every cluster
file the members were fitted on (these include training frames: this is a sanity check, not
a holdout score), it reports per size:

    E_MAE      committee-mean total energy vs the label, kJ/mol per frame
    F_MAE      committee-mean forces vs the label (converted from Hartree/Bohr), kJ/mol/A
    sigma_E    member spread of the energy, meV/atom
    sigma_F    DeePMD model_devi_f (largest per-atom force deviation), kJ/mol/A

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
sys.path.insert(0, str(HERE.parent.parent / "active_learning"))
warnings.filterwarnings("ignore", message="torch_cluster is not installed")

KJMOL = 2625.499639
MEV = 27211.386245988
#: the cluster files store forces in Hartree/Bohr (see rsfff.train.data); the model is per Angstrom
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def read_frames(path: Path, limit: int):
    """``(species, positions, energy, forces)`` for the first ``limit`` frames."""
    from ase.io import iread
    out = []
    for atoms in iread(str(path), index=f":{limit}", format="extxyz"):
        info, arrays = atoms.info, atoms.arrays
        energy = info.get("energy")
        forces = arrays.get("forces")
        if forces is None:
            try:
                forces = atoms.get_forces()
            except Exception:  # noqa: BLE001
                forces = None
        if energy is None:
            try:
                energy = atoms.get_potential_energy()
            except Exception:  # noqa: BLE001
                energy = None
        if forces is not None:
            forces = np.asarray(forces) * BOHR_PER_ANGSTROM
        out.append((atoms.get_chemical_symbols(), atoms.get_positions(), energy, forces))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("committee", type=Path)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from committee import Committee  # active_learning/committee.py
    manifest = json.loads((args.committee / "committee.json").read_text())
    c = Committee.load(args.committee, device=args.device)
    print(f"{c.n_members} member(s) loaded on {args.device}; val losses "
          f"{[m.get('val_loss') for m in manifest['members']]}")

    files = [Path(p) for p in manifest["training_data"]
             if Path(p).name.startswith("w") and Path(p).suffix == ".xyz"]
    ok = True
    print(f"{'file':32s} {'n':>3s} {'E_MAE':>9s} {'F_MAE':>8s} {'sigma_E':>9s} {'sigma_F':>8s}")
    for path in files:
        e_err, f_err, s_e, s_f = [], [], [], []
        for species, pos, energy, forces in read_frames(path, args.frames):
            e, f = c.predict(species, pos)
            n = len(species)
            if not (np.isfinite(e).all() and np.isfinite(f).all()):
                ok = False
            if energy is not None:
                e_err.append(abs(e.mean() - energy) * KJMOL)
            if forces is not None:
                f_err.append(np.abs(f.mean(0) - forces).mean() * KJMOL)
            if c.n_members > 1:
                s_e.append(np.std(e, ddof=1) / n * MEV)
                dev = f - f.mean(0, keepdims=True)
                s_f.append(np.sqrt((dev ** 2).sum(-1).mean(0)).max() * KJMOL)
        mean = lambda v: float(np.mean(v)) if v else math.nan  # noqa: E731
        print(f"{path.name:32s} {len(e_err) or len(s_f):3d} {mean(e_err):9.3f} "
              f"{mean(f_err):8.3f} {mean(s_e):9.4f} {mean(s_f):8.3f}")
    print("all predictions finite" if ok else "NON-FINITE predictions")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
