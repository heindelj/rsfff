"""Evaluate a ``train_bonding`` checkpoint on an extxyz, per molecule and frame kind.

    python scripts/eval_bonding.py ckpt/<run>/best.pt data/stretch.xyz [--json out.json]
    python scripts/eval_bonding.py ckpt/<run>/best.pt data/train.xyz --split val
    python scripts/eval_bonding.py ckpt/<run>/best.pt data/probe.xyz --split val \\
        --vacuum data/train.xyz data/stretch.xyz

Per frame, where the file carries the label: energy relative to the separated atoms (kcal/mol),
force MAE (kcal/mol/Ang), relative Hessian error, |dipole error| (D), quadrupole MAE (e*bohr^2),
polarizability MAE (bohr^3) and isotropic relative error, and for probe frames paired with the vacuum
frame at the same geometry (``--vacuum``) the interaction-energy and force-change errors.
``--split val|train`` evaluates the trainer's split of that file. The code is
:mod:`rsfff.train.bonding_eval`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsfff.train.bonding_eval import evaluate, load_model, print_summary, split_frames, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--ref", type=Path, default=None, help="reference energies json")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--vacuum", type=Path, nargs="*", default=None,
                    help="vacuum files to pair probe frames with (interaction energies)")
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--no-moments", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--film", nargs="*", default=[], help="config.film overrides, key=value (floats)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    over = {k: float(v) for k, v in (s.split("=", 1) for s in a.film)}
    model, ck = load_model(a.checkpoint, a.ref, film_overrides=over, device=a.device)
    idx = None if a.split == "all" else split_frames(a.data, a.split, ck["config"])
    rows = evaluate(model, a.data, idx=idx, hessian=not a.no_hessian,
                    moments=not a.no_moments, vacuum=a.vacuum)
    summary = summarize(rows)
    print_summary(summary)
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
