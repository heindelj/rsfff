#!/usr/bin/env python3
"""Split a harvest into training structures and negative-space structures.

    python scripts/curate_negative_space.py qchem_roundtrip/biased_sampling/*/ \
        --out data/negative_space --select 400

The first, unguarded harvest is the immediate source: 35% of its 5511 frames have the contested
proton spanning an O-O that is far too long, and 5.6% have a hydrogen bound to nothing. Those
were treated as waste. They are not -- they are exactly the configurations the model currently
thinks are cheap, which is *why* the bias walked into them, and a reference label saying they
are expensive is what closes that hole. Recovering them costs one pass over files already on
disk.

What qualifies
--------------
Only the **geometry** rejects. A frame the model could not evaluate -- non-finite energy, a
force off the scale, a collapsed pair of nuclei -- says nothing about where the true surface is
steep, only that the trajectory left the region the model can describe. Worse, a collapsed
frame will not converge in Q-Chem and its energy would dominate any loss it appeared in. The
distinction is carried by :class:`run_reactive_md.Defect`, and `--min-distance` here is the
floor below which a geometry is not labelable at all.

Diversity, not volume
---------------------
Negative examples are abundant and highly correlated -- consecutive frames of one trajectory
segment are the same stranded proton. `--select` runs farthest-point sampling in the model's
own descriptor (:mod:`rsfff.md.similarity`) so the labelled set spans the failure mode rather
than sampling one corner of it a hundred times. A training set swamped by repulsive geometries
teaches the model where not to be at the expense of where to be, so keep this set a fraction of
the positive one.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch                                                              # noqa: E402
from ase.io import read, write                                            # noqa: E402

from run_reactive_md import geometry_defect, infer_charge                 # noqa: E402
from setup_ion_cluster_jobs import ensure_job_dir, molecule_block, rem_block  # noqa: E402
from rsfff.md import FeatureMetric, load_mediated_model                    # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="harvest directories or .xyz files")
    p.add_argument("--out", default="data/negative_space")
    p.add_argument("--select", type=int, default=0,
                   help="farthest-point select this many of the negatives; 0 keeps all")
    p.add_argument("--checkpoint", default="checkpoints/ion_mediator_v4_full/best.pt")
    p.add_argument("--min-distance", type=float, default=0.75,
                   help="Angstrom; below this a geometry is not labelable at all and is "
                        "dropped rather than kept as negative space. Higher than the sampler's "
                        "0.6 on purpose -- this set is going to Q-Chem.")
    p.add_argument("--max-oh", type=float, default=1.45)
    p.add_argument("--max-oo", type=float, default=2.75)
    p.add_argument("--charge", type=int)
    p.add_argument("--stage-force", nargs="?", const="qchem_roundtrip/force/negative_space",
                   help="also write Q-Chem force inputs for the selected structures into this "
                        "job dir (default qchem_roundtrip/force/negative_space). Force and not "
                        "EDA: a stranded proton is off any single fragmentation, so the "
                        "decomposition is the thing the mediator has to decide rather than "
                        "something to supervise, and what the model needs here is the total "
                        "energy and the forces saying this geometry is expensive.")
    p.add_argument("--method", default="wB97M-V")
    p.add_argument("--basis", default="def2-TZVPD")
    args = p.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files += sorted(path.glob("*.xyz"))
        elif path.exists():
            files.append(path)
    if not files:
        raise SystemExit("no .xyz files found in the given paths")

    positive, negative, reasons = [], [], Counter()
    dropped = 0
    for path in files:
        try:
            frames = read(str(path), index=":")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skipping {path}: {exc}", flush=True)
            continue
        for atoms in frames:
            charge = args.charge if args.charge is not None else infer_charge(atoms)
            defect = geometry_defect(atoms, min_distance=args.min_distance,
                                     max_oh=args.max_oh, max_oo=args.max_oo)
            atoms.info["charge"] = charge
            if defect is None:
                positive.append(atoms)
            elif defect.negative_space:
                atoms.info["rejection"] = defect.reason
                # Keep the class, not the number: "contested H spans a 3.12 Angstrom O-O" is a
                # different string every frame and useless as a label.
                kind = ("stranded_h" if defect.reason.startswith("H stranded")
                        else "long_oo")
                atoms.info["rejection_kind"] = kind
                reasons[kind] += 1
                negative.append(atoms)
            else:
                dropped += 1

    total = len(positive) + len(negative) + dropped
    print(f"{total} frames from {len(files)} files", flush=True)
    print(f"  {len(positive):6d} pass the guard")
    print(f"  {len(negative):6d} negative space  " + ", ".join(
        f"{k}={v}" for k, v in sorted(reasons.items())))
    print(f"  {dropped:6d} unlabelable (collapsed or diverged), dropped", flush=True)

    if args.select and len(negative) > args.select:
        torch.set_default_dtype(torch.float64)
        model, _cfg, _state = load_mediated_model(args.checkpoint)
        by_charge: dict[int, list[int]] = {}
        for i, a in enumerate(negative):
            by_charge.setdefault(int(a.info["charge"]), []).append(i)
        picked: list[int] = []
        for charge, idx in sorted(by_charge.items()):
            # Fit and select per charge: a hydronium and a hydroxide cluster share no fragment
            # composition, so a single pooled selection would mostly be choosing between them.
            share = max(1, round(args.select * len(idx) / len(negative)))
            metric = FeatureMetric.fit(model, [(negative[i], charge) for i in idx[:40]])
            desc = [metric.fragment_descriptors(negative[i], charge) for i in idx]
            chosen = metric.farthest_point(desc, min(share, len(idx)))
            picked += [idx[c] for c in chosen]
            print(f"  charge {charge:+d}: selected {len(chosen)} of {len(idx)}")
        negative = [negative[i] for i in sorted(picked)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neg_path = out / "negative_space.xyz"
    write(str(neg_path), negative, format="extxyz")
    print(f"\nwrote {len(negative)} structures -> {neg_path}", flush=True)
    if args.stage_force:
        job = Path(args.stage_force)
        ensure_job_dir(job)
        for i, atoms in enumerate(negative):
            stem = f"negative_{i:04d}_q{int(atoms.info['charge']):+d}"
            text = (molecule_block(atoms.get_chemical_symbols(),
                                   np.asarray(atoms.get_positions()),
                                   int(atoms.info["charge"]))
                    + "\n\n" + rem_block("force", args.method, args.basis, 64000, 10000) + "\n")
            (job / "inputs" / f"{stem}.in").write_text(text)
        print(f"staged {len(negative)} force jobs -> {job}/inputs", flush=True)
    else:
        print("Pass --stage-force to write Q-Chem force inputs for these. Force and not EDA: "
              "they are off any single fragmentation, so the decomposition is what the mediator "
              "has to decide rather than something to supervise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
