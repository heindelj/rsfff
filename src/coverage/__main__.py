"""python -m rsfff.coverage {build,score,select}

  build   the training hull of a train_ionct run (runs since the hull change write RUN/hull.pt themselves)
  score   per-frame distance of candidate structures (extxyz) to a hull -> csv
  select  guard + greedy hull extension over candidate structures -> extxyz (+ optionally the extended hull)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read(paths, every=1):
    from ase.io import read
    out = []
    for p in paths:
        for k, a in enumerate(read(p, f"::{every}")):
            out.append((str(p), k * every, a))
    return out


def _params(a, hull):
    p = a.params or hull.meta.get("params")
    if not p or not Path(p).exists():
        raise SystemExit(f"parameter table {p!r} not found; pass --params")
    return p


def _features(frames, params, hull):
    from .select import featurizer_for_params, frame_features
    from ..ff.complete.environment import inv_form_of
    feat = featurizer_for_params(params, inv_form_of(hull.feature_names))
    return [frame_features(feat, at.numbers, at.positions) for _, _, at in frames]


def cmd_build(a):
    from .hull import FeatureHull
    from .runs import run_training_features
    F, scale, names, args, info = run_training_features(a.run, a.cache)
    extra = 0
    if a.extra_xyz:
        from .select import featurizer_for_params, frame_features
        from ..ff.complete.environment import inv_form_of
        feat = featurizer_for_params(a.params or args["params"], inv_form_of(names))
        ex = [frame_features(feat, at.numbers, at.positions) for _, _, at in _read(a.extra_xyz)]
        extra = len(ex)
        F = np.concatenate([F, *ex])
    info.update(params=str(a.params or args["params"]), extra_frames=extra)
    h = FeatureHull.fit(F, scale, k=a.k, feature_names=names, meta=info, verbose=True)
    out = Path(a.out or Path(a.run) / "hull.pt")
    h.save(out)
    print(f"wrote {out}")


def cmd_score(a):
    from .hull import FeatureHull
    from .select import Guard, score_frames
    h = FeatureHull.load(a.hull)
    frames = _read(a.xyz, a.every)
    S = score_frames(h, _features(frames, _params(a, h), h))
    g = Guard(a.like, a.unlike, a.detached)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "index", "n", "d_max", "d_mean", "n_out", "guard"])
        for (f, i, at), s in zip(frames, S):
            w.writerow([f, i, len(at), f"{s[0]:.6g}", f"{s[1]:.6g}", int(s[2]),
                        g.reason(at.get_chemical_symbols(), at.positions) or "ok"])
    print(f"{len(frames)} frames, {int((S[:, 2] > 0).sum())} with atoms outside the hull; "
          f"{int(S[:, 2].sum())} of {sum(len(x[2]) for x in frames)} atoms outside -> {a.out}")


def cmd_select(a):
    from ase.io import write
    from .hull import FeatureHull
    from .select import Guard, greedy_extend
    h = FeatureHull.load(a.hull)
    frames = _read(a.xyz, a.every)
    g = Guard(a.like, a.unlike, a.detached)
    ok = [x for x in frames if g(x[2].get_chemical_symbols(), x[2].positions)]
    print(f"{len(frames)} candidates, {len(ok)} pass the guard", flush=True)
    picked, scores, h2 = greedy_extend(h, _features(ok, _params(a, h), h), a.n, score=a.score, verbose=a.verbose)
    sel = []
    for rank, (i, s) in enumerate(zip(picked, scores)):
        f, k, at = ok[i]
        at = at.copy()
        at.info.update(hull_rank=rank, hull_score=s, source_file=f, source_index=k)
        sel.append(at)
    write(a.out, sel, format="extxyz")
    print(f"picked {len(sel)} frames -> {a.out}; hull support {len(h.points)} -> {len(h2.points)}")
    if a.save_hull:
        h2.meta = dict(h2.meta, extended_by=str(a.out), n_extended=len(sel))
        h2.save(a.save_hull)
        print(f"extended hull -> {a.save_hull}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rsfff.coverage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="training hull of a train_ionct run")
    b.add_argument("--run", required=True, help="train_ionct output directory (model.pt, train.log, predictions.csv)")
    b.add_argument("--cache", help="the run's frame cache (default: the one named in train.log)")
    b.add_argument("--k", type=int, default=6, help="whitened PCA components the hull lives in (default 6)")
    b.add_argument("--extra-xyz", nargs="*", default=[], help="more training structures (e.g. the force frames)")
    b.add_argument("--params", help="parameter table for --extra-xyz (default: the run's)")
    b.add_argument("--out", help="default RUN/hull.pt")
    b.set_defaults(fn=cmd_build)
    for name, fn, hlp in (("score", cmd_score, "distance of candidates to a hull"),
                          ("select", cmd_select, "greedy hull extension over candidates")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--hull", required=True)
        p.add_argument("--xyz", nargs="+", required=True, help="candidate structures (extxyz)")
        p.add_argument("--params", help="parameter table (default: the one recorded in the hull)")
        p.add_argument("--every", type=int, default=1, help="read every n-th frame")
        p.add_argument("--like", type=float, default=0.8, help="guard: min like distance / (r_i + r_j)")
        p.add_argument("--unlike", type=float, default=0.6, help="guard: min unlike distance / (r_i + r_j)")
        p.add_argument("--detached", type=float, default=6.0, help="guard: max distance to the nearest atom (A)")
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "select":
            p.add_argument("-n", type=int, required=True, help="frames to pick")
            p.add_argument("--score", choices=("max", "sum"), default="max")
            p.add_argument("--save-hull", help="write the hull extended by the picks here")
            p.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
