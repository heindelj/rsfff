"""Parity (correlation) plots of a ``train_bonding`` checkpoint against its labels.

    python scripts/parity_bonding.py ckpt/<run>/best.pt --out parity.png \\
        --set train=data/train.xyz:train --set val=data/train.xyz:val --set held-out=data/far.xyz \\
        --probe train=data/probe.xyz:train --probe val=data/probe.xyz:val \\
        --vacuum data/train.xyz data/far.xyz --max-frames 150

Energies relative to each molecule's ``minimum`` frame, force components, Hessian elements, dipole /
traceless quadrupole / polarizability components, and the probe interaction energy and force change,
one panel each, one colour per set (at most three). The code is :mod:`rsfff.train.bonding_eval`
(``collect``, ``plot_parity``), importable from a notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rsfff.train.bonding_eval import (collect, load_model, minimum_energies, plot_parity, save_parity,
                                      split_frames)


def _spec(s):
    """``label=path[:train|val]`` -> (label, path, split)."""
    label, rest = s.split("=", 1)
    path, _, split = rest.partition(":")
    return label, path, split or "all"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--set", action="append", default=[], help="label=vacuum.xyz[:train|val]")
    ap.add_argument("--probe", action="append", default=[], help="label=probe.xyz[:train|val]")
    ap.add_argument("--vacuum", nargs="*", default=[], help="vacuum files (energy minima, probe pairing)")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("parity.png"))
    ap.add_argument("--npz", type=Path, default=None, help="also save the collected arrays")
    ap.add_argument("--title", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--film", nargs="*", default=[], help="config.film overrides, key=value (floats)")
    a = ap.parse_args()
    over = {k: float(v) for k, v in (s.split("=", 1) for s in a.film)}
    model, ck = load_model(a.checkpoint, film_overrides=over, device=a.device)
    cfg = ck["config"]
    minima = minimum_energies(model, a.vacuum or [_spec(s)[1] for s in a.set])
    sets: dict = {}
    for s in a.set:
        label, path, split = _spec(s)
        sets[label] = collect(model, path, idx=split_frames(path, split, cfg, a.max_frames), minima=minima)
    for s in a.probe:
        label, path, split = _spec(s)
        d = collect(model, path, idx=split_frames(path, split, cfg, a.max_frames), hessian=False,
                    moments=False, vacuum=a.vacuum)
        sets.setdefault(label, {}).update(d)
    if a.npz:
        save_parity(sets, a.npz)
    plot_parity(sets, a.out, title=a.title)


if __name__ == "__main__":
    main()
