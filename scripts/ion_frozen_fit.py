"""Fit a frozen pair term of the ion potential to its ALMO-EDA label:

* ``--term disp`` (default): Tang-Toennies C6 + C8 per species, optionally an environment scale of C6, vs ``disp``;
* ``--term pauli``: the CMM monopole Slater Pauli kernel, optionally an environment scale of qP, vs ``mod_pauli``.

    python scripts/ion_frozen_fit.py --ct-sets CT_SETS --walk-eda DIR --cache CACHE --out DIR [--term pauli] [--nn]

Same frames, deduplication and train/val/test split as train_ionct (test = the held-out parent clusters with all
their subsets). The environment invariants come from the ion CT cache (the same SOAP features as the CT model).
Loss: per-ion squared error (kJ/mol) on the train frames. Writes TERM.pt (IonDispersion.load / IonPauli.load) and predictions.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsfff.ff.ionct.dispersion import IonDispersion
from rsfff.ff.ionct.frozen_pauli import IonPauli
from rsfff.train.train_ionct import DEFAULT_TEST, KJ, load_frames, split

SYM = ("Na", "Cl")


def flatten(frames, key="disp"):
    pos, sp, bi, y, n, pairs = [], [], [], [], [], []
    off = 0
    for k, (sym, X, lab, meta) in enumerate(frames):
        pairs.append(torch.triu_indices(len(sym), len(sym), 1) + off)
        off += len(sym)
        pos.append(torch.as_tensor(X, dtype=torch.float64))
        sp.append(torch.tensor([SYM.index(s) for s in sym]))
        bi.append(torch.full((len(sym),), k, dtype=torch.long))
        y.append(lab.get(key, float("nan")))
        n.append(len(sym))
    return (torch.cat(pos), torch.cat(sp), torch.cat(bi), torch.tensor(y, dtype=torch.float64),
            torch.tensor(n, dtype=torch.float64), torch.cat(pairs, 1))


GROUPS = (("all", lambda m, sym: True),
          ("Na-Cl pairs", lambda m, sym: len(sym) == 2 and sorted(sym) == ["Cl", "Na"]),
          ("Cl-Cl pairs", lambda m, sym: sym == ["Cl", "Cl"]),
          ("trimers", lambda m, sym: len(sym) == 3),
          ("whole clusters n>3", lambda m, sym: m["full"] and len(sym) > 3))


def report(e, frames, tags, y):
    err = ((e - y) * KJ).numpy()
    for s in ("train", "val", "test"):
        for name, sel in GROUPS:
            k = [q for q, (sym, X, lab, m) in enumerate(frames) if tags[q] == s and sel(m, sym) and np.isfinite(err[q])]
            if k:
                ref = (y[k] * KJ).numpy()
                print(f"  {s:5s} {name:20s} n={len(k):5d}  MAE {np.abs(err[k]).mean():.3f}  bias {err[k].mean():+.3f}  "
                      f"max {np.abs(err[k]).max():.2f} kJ/mol  rel {100 * np.abs(err[k]).sum() / np.abs(ref).sum():.1f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--cache", required=True, help="ion CT feature cache aligned with load_frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--term", default="disp", choices=("disp", "pauli"))
    ap.add_argument("--nn", action="store_true", help="environment scale of each ion's C6 (disp) or qP (pauli)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--no-c8", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(a.ct_sets, a.walk_eda)
    tags = split(frames, DEFAULT_TEST, seed=0)
    pos, sp, bi, y, n, pairs = flatten(frames, a.term)
    caches = torch.load(a.cache, weights_only=False)
    assert len(caches) == len(frames), "cache does not match the frames"
    F = torch.cat([c.feats for c in caches]).double()
    tr = torch.tensor([t == "train" for t in tags]) & torch.isfinite(y)
    nz = F.abs().sum(1) > 0
    scale = F[tr[bi] & nz].std(0).clamp(min=1e-6)
    kw = dict(n_feat=F.shape[1] if a.nn else None, hidden=a.hidden, feat_scale=scale)
    model = (IonDispersion.initial(SYM, use_c8=not a.no_c8, **kw) if a.term == "disp" else IonPauli.initial(SYM, **kw))
    B = len(frames)
    with torch.no_grad():
        e0 = model(pos, sp, bi, B, pairs, F)
    print("initial (prior constants, pairwise):")
    report(e0, frames, tags, y)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.02)
    for it in range(a.steps):
        opt.zero_grad()
        e = model(pos, sp, bi, B, pairs, F)
        loss = (((e - y) * KJ / n)[tr] ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        e = model(pos, sp, bi, B, pairs, F)
        ls = model.env_scale(sp, F)
    print("fitted constants:", {k: [round(v, 4) for v in vv] for k, vv in model.constants().items()})
    if a.nn:
        for s_, x in enumerate(SYM):
            m = (sp == s_) & nz
            q = torch.quantile(torch.exp((2 if a.term == "disp" else 1) * ls[m]),
                               torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
            print(f"  environment scale of {'C6' if a.term == 'disp' else 'qP'}, {x}: 5/50/95 % = "
                  + " / ".join(f"{v:.3f}" for v in q.tolist()))
    report(e, frames, tags, y)
    model.save(out / f"{a.term}.pt", args=vars(a))
    rows = [{"split": tags[q], "workflow": m["workflow"], "parent": m["parent"], "n": len(sym), "full": bool(m["full"]),
             f"{a.term}_ref": float(y[q]) * KJ, a.term: float(e[q]) * KJ} for q, (sym, X, lab, m) in enumerate(frames)]
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    json.dump({"constants": model.constants(), "args": vars(a)}, open(out / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
