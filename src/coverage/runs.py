"""Training-set invariants of an existing train_ionct run (for runs that predate the hull.pt it now writes)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

__all__ = ["run_training_features"]


def _key(m):
    return (m.get("workflow"), m.get("parent"), int(m.get("order") or 0), round(float(m.get("ho_ratio") or 0.0), 8))


def run_training_features(run_dir, cache=None):
    """(raw invariants of every training atom (N, F), feat_scale (F,), feature names, model args, info) of a
    train_ionct run: the cache named in train.log (or ``cache``), split by the run's predictions.csv (frames
    matched on workflow / parent / order / ho_ratio)."""
    import torch
    run = Path(run_dir)
    ck = torch.load(run / "model.pt", weights_only=False)
    scale = ck["state"]["feat_scale"].double().numpy()
    if cache is None:
        m = re.search(r"^cache: (.+)$", (run / "train.log").read_text(), re.M)
        if not m:
            raise SystemExit(f"{run}/train.log names no cache; pass --cache")
        cache = Path(m.group(1))
        if not cache.is_absolute():         # train_ionct logs it relative to its working directory
            for base in (Path.cwd(), *run.resolve().parents):
                if (base / cache).exists():
                    cache = base / cache
                    break
    caches = torch.load(cache, weights_only=False)
    split = {}
    with open(run / "predictions.csv") as fh:
        for r in csv.DictReader(fh):
            split[(r["workflow"], r["parent"], int(r["order"]), round(float(r["ho_ratio"]), 8))] = r["split"]
    from ..ff.complete.environment import env_invariants, inv_form_of, PROBE_NAMES
    names = ck.get("feature_names")
    form = inv_form_of(names)
    probes = ck["args"].get("env_probes") or PROBE_NAMES
    tr, miss = [], 0
    for c in caches:
        s = split.get(_key(c.meta))
        miss += s is None
        if s == "train":
            f = env_invariants(c.env, probes, form) if c.env else c.feats
            tr.append(f.double().numpy())
    if miss:
        print(f"warning: {miss} of {len(caches)} cached frames not found in predictions.csv", flush=True)
    info = {"run": str(run), "cache": str(cache), "n_frames": len(tr), "unmatched": miss}
    return np.concatenate(tr), scale, names, ck["args"], info
