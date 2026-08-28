#!/usr/bin/env python3
"""Score and curate harvested structures in the model's own descriptor.

    # how far is each harvested frame from the training corpus?
    python scripts/feature_similarity.py score \
        qchem_roundtrip/biased_sampling/h3o+_w2/transition_structures.xyz --charge 1

    # pick 100 maximally different frames to spend Q-Chem time on
    python scripts/feature_similarity.py select \
        qchem_roundtrip/biased_sampling/h3o+_w2/transition_structures.xyz \
        --charge 1 -n 100 --out selected.xyz

    # what is the pairwise similarity of a handful of structures?
    python scripts/feature_similarity.py compare a.xyz b.xyz --charge 1

``score`` writes ``novelty`` and ``fragment_novelty`` into each frame's extxyz header, so the
numbers travel with the geometry. ``select`` is the one that saves money: a harvest is heavily
autocorrelated, and labelling a random subset spends most of the budget on near-duplicates.

Novelty is measured against a reference set, by default the ion frames of
``data/wb97mv_tzvpd`` -- the geometries the model was actually fitted on. Compare **within a
cluster size**: a size mismatch leaves fragments unmatched, which dominates the score and
swamps whatever the geometry is doing. The tool warns when the reference and the input disagree
about size.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch                                                              # noqa: E402
from ase.io import read, write                                            # noqa: E402

from rsfff.md import FeatureMetric, load_mediated_model                    # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/ion_mediator_v4_full/best.pt"
CORPUS = {
    +1: ["data/wb97mv_tzvpd/w1_h3o+_wb97mv_tzvpd.xyz",
         "data/wb97mv_tzvpd/w2_h3o+_wb97mv_tzvpd.xyz"],
    -1: ["data/wb97mv_tzvpd/w1_oh-_wb97mv_tzvpd.xyz",
         "data/wb97mv_tzvpd/w2_oh-_wb97mv_tzvpd.xyz"],
}


def infer_charge(atoms) -> int:
    z = atoms.get_atomic_numbers()
    return int((z == 1).sum() - 2 * (z == 8).sum())


def load_reference(charge: int, paths, limit: int):
    if paths:
        frames = []
        for p in paths:
            frames += read(p, index=":")
    else:
        frames = []
        for p in CORPUS.get(charge, []):
            if Path(p).exists():
                frames += read(p, index=":")
    if not frames:
        raise SystemExit(
            f"no reference frames for charge {charge:+d}; pass --reference explicitly"
        )
    step = max(1, len(frames) // limit)
    return frames[::step][:limit]


def build(args, charge: int):
    torch.set_default_dtype(torch.float64)
    model, _cfg, _state = load_mediated_model(args.checkpoint)
    reference = load_reference(charge, args.reference, args.reference_limit)
    metric = FeatureMetric.fit(model, [(a, charge) for a in reference])
    return metric, [metric.fragment_descriptors(a, charge) for a in reference], reference


def _warn_size_mismatch(frames, reference) -> None:
    n_in = {int((a.get_atomic_numbers() == 8).sum()) for a in frames}
    n_ref = {int((a.get_atomic_numbers() == 8).sum()) for a in reference}
    if not (n_in & n_ref):
        print(f"WARNING: the input has {sorted(n_in)} oxygens and the reference {sorted(n_ref)}. "
              f"With no size in common every comparison leaves fragments unmatched, so the "
              f"novelty numbers measure the size difference and little else.", flush=True)


def cmd_score(args) -> int:
    frames = read(args.path, index=args.index)
    charge = args.charge if args.charge is not None else infer_charge(frames[0])
    metric, reference, ref_frames = build(args, charge)
    _warn_size_mismatch(frames, ref_frames)

    rows = []
    for a in frames:
        d = metric.fragment_descriptors(a, charge)
        system, fragment = metric.novelty(d, reference)
        a.info["novelty"] = float(system)
        a.info["fragment_novelty"] = float(fragment)
        rows.append((system, fragment))
    novelty = np.array([r[0] for r in rows])

    print(f"{len(frames)} frames, charge {charge:+d}, {len(reference)} reference structures")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  novelty p{q:<3d} = {np.percentile(novelty, q):.4f}")
    if args.out:
        write(args.out, frames, format="extxyz")
        print(f"wrote {args.out} with novelty in the header")
    return 0


def cmd_select(args) -> int:
    frames = read(args.path, index=args.index)
    charge = args.charge if args.charge is not None else infer_charge(frames[0])
    metric, _reference, _rf = build(args, charge)
    descriptors = [metric.fragment_descriptors(a, charge) for a in frames]
    picked = sorted(metric.farthest_point(descriptors, args.n, seed=args.seed))

    def spread(idx):
        vals = [metric.match(descriptors[i], descriptors[j]).score
                for k, i in enumerate(idx) for j in idx[k + 1:]]
        return float(np.mean(vals)) if vals else float("nan")

    baseline = list(range(min(args.n, len(frames))))
    print(f"selected {len(picked)} of {len(frames)}")
    print(f"  mean pairwise similarity, selected : {spread(picked):.4f}")
    print(f"  mean pairwise similarity, first N  : {spread(baseline):.4f}  (lower is more diverse)")
    if args.out:
        write(args.out, [frames[i] for i in picked], format="extxyz")
        print(f"wrote {args.out}")
    return 0


def cmd_compare(args) -> int:
    frames = []
    for p in args.paths:
        frames += read(p, index=":")
    charge = args.charge if args.charge is not None else infer_charge(frames[0])
    torch.set_default_dtype(torch.float64)
    model, _cfg, _state = load_mediated_model(args.checkpoint)
    metric = FeatureMetric.fit(model, [(a, charge) for a in frames])
    descriptors = [metric.fragment_descriptors(a, charge) for a in frames]

    print(f"pairwise similarity of {len(frames)} structures (charge {charge:+d}):")
    print("      " + "".join(f"{j:>8d}" for j in range(len(frames))))
    for i, di in enumerate(descriptors):
        row = "".join(f"{metric.match(di, dj).score:8.4f}" for dj in descriptors)
        print(f"  {i:3d} {row}")
    return 0


def main(argv=None) -> int:
    # The shared options go on a parent parser as well as the top level, so they are accepted
    # on either side of the subcommand. argparse otherwise rejects `score FILE --charge 1`,
    # which is the order everyone types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    common.add_argument("--charge", type=int, help="default: inferred from the composition")
    common.add_argument("--reference", nargs="*", default=None,
                        help="reference structures; default is the training corpus for this charge")
    common.add_argument("--reference-limit", type=int, default=40,
                        help="subsample the reference to this many frames; novelty costs one "
                             "match per reference frame per input frame")

    p = argparse.ArgumentParser(description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", parents=[common],
                       help="novelty of every frame against the reference set")
    s.add_argument("path")
    s.add_argument("--index", default=":")
    s.add_argument("--out", help="write the frames back out with novelty in the header")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("select", parents=[common],
                   help="farthest-point selection of n diverse frames")
    s.add_argument("path")
    s.add_argument("--index", default=":")
    s.add_argument("-n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("compare", parents=[common], help="pairwise similarity matrix")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
