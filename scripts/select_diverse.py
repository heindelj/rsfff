#!/usr/bin/env python3
"""Cut a harvest down to a structurally diverse subset before spending Q-Chem time on it.

    python scripts/select_diverse.py data/cluster_sweep/transition_structures.xyz \
        -n 2000 --out data/cluster_sweep/transition_selected.xyz

A sweep is heavily autocorrelated by construction. Each isomer contributes ~120 frames taken
20 steps apart from a handful of trajectory segments, so consecutive keeps are the same
structure with the protons jiggled. Labelling all of them spends most of the budget proving the
same point, and for EDA -- three jobs per structure -- the waste is tripled.

Selection is **stratified by (charge, number of oxygens)** and farthest-point within each
stratum. Stratifying is not a refinement: the surrogate descriptor
(:meth:`~rsfff.md.similarity.FeatureMetric.system_descriptor`) pools fragments by composition
and so carries no cluster size, and comparing a 3-water cluster to a 7-water one in that space
is meaningless. Splitting first makes every comparison a like-for-like one and has the
side benefit that every size keeps a share of the budget instead of being crowded out by
whichever size happened to be sampled most.

The report prints mean pairwise similarity before and after, computed with the *real*
best-match kernel on a random subsample, so the number says what the selection actually bought
rather than restating the surrogate it was chosen with.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch                                                              # noqa: E402
from ase.io import read, write                                            # noqa: E402

from rsfff.md import FeatureMetric, load_mediated_model                    # noqa: E402


def stratum(atoms) -> tuple[int, int]:
    z = atoms.get_atomic_numbers()
    n_o = int((z == 8).sum())
    return int((z == 1).sum() - 2 * n_o), n_o


def mean_similarity(metric, frames, charge, k=40, seed=0) -> float:
    """Mean best-match similarity over a random sample of pairs. The honest measure."""
    rng = np.random.default_rng(seed)
    if len(frames) < 2:
        return float("nan")
    idx = rng.choice(len(frames), size=min(k, len(frames)), replace=False)
    desc = [metric.fragment_descriptors(frames[i], charge) for i in idx]
    vals = [metric.match(desc[a], desc[b]).score
            for a in range(len(desc)) for b in range(a + 1, len(desc))]
    return float(np.mean(vals)) if vals else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("-n", "--select", type=int, required=True, help="total structures to keep")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="checkpoints/ion_mediator_v4_full/best.pt")
    p.add_argument("--fit-frames", type=int, default=60,
                   help="frames used to fit the z-scoring statistics")
    p.add_argument("--report-pairs", type=int, default=30,
                   help="sample size for the before/after similarity report; 0 skips it")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    frames = read(args.path, index=":")
    print(f"{len(frames)} frames from {args.path}", flush=True)

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, a in enumerate(frames):
        groups[stratum(a)].append(i)
    print(f"{len(groups)} strata (charge, n_oxygen): "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(groups.items())), flush=True)

    model, _cfg, _state = load_mediated_model(args.checkpoint)
    step = max(1, len(frames) // args.fit_frames)
    metric = FeatureMetric.fit(
        model, [(a, int(a.info.get("charge", stratum(a)[0]))) for a in frames[::step]])
    print(f"metric fitted on {len(frames[::step])} frames", flush=True)

    picked: list[int] = []
    t0 = time.time()
    for key, idx in sorted(groups.items()):
        charge, n_o = key
        share = max(1, round(args.select * len(idx) / len(frames)))
        vecs, keep = [], []
        for j in idx:
            v = metric.system_descriptor(frames[j], charge)
            if v is not None:
                vecs.append(v)
                keep.append(j)
        if not vecs:
            continue
        widths = {len(v) for v in vecs}
        if len(widths) > 1:
            # A frame whose composition differs from the rest of its stratum -- an ion that
            # dissociated, say -- cannot share the vector space. Drop the minority rather than
            # padding, which would place them all at a spurious common point.
            common = max(widths, key=lambda w: sum(len(v) == w for v in vecs))
            vecs, keep = zip(*[(v, j) for v, j in zip(vecs, keep) if len(v) == common])
            vecs, keep = list(vecs), list(keep)
        chosen = metric.farthest_point_vectors(np.asarray(vecs), min(share, len(vecs)),
                                               seed=args.seed)
        picked += [keep[c] for c in chosen]
        print(f"  charge {charge:+d}, {n_o} O: {len(chosen):5d} of {len(idx):6d}", flush=True)

    picked = sorted(set(picked))
    print(f"\nselected {len(picked)} of {len(frames)} in {time.time() - t0:.0f} s", flush=True)

    if args.report_pairs:
        big = max(groups.items(), key=lambda kv: len(kv[1]))
        charge = big[0][0]
        before = mean_similarity(metric, [frames[i] for i in big[1]], charge,
                                 k=args.report_pairs, seed=args.seed)
        after_idx = [i for i in picked if stratum(frames[i]) == big[0]]
        after = mean_similarity(metric, [frames[i] for i in after_idx], charge,
                                k=args.report_pairs, seed=args.seed)
        print(f"mean pairwise similarity in the largest stratum {big[0]}: "
              f"{before:.4f} -> {after:.4f}  (lower is more diverse)", flush=True)

    write(args.out, [frames[i] for i in picked], format="extxyz")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
