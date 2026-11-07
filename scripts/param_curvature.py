"""Geometric smoothness of a complete-model run's emitted parameters: rms second difference (per A^2) of the log
Pauli weight, log C6 and log q_D / log q_A along random per-atom displacements, on fixed cluster frames.

    python scripts/param_curvature.py CACHE.pt RUN/model.pt [RUN2/model.pt ...] [--frames 24 --delta 0.02 --fields 3]

Works for any complete model (old heads or the m33 shared environment, hybrid CT included), so runs trained with and
without the curvature prior compare on the same footing. Run from research/rsfff (the runs' relative paths)."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rsfff.ff.complete import batched_precompute
from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import ATOMIC_NUMBER as Zmap
from rsfff.train.train_ionct import build_model, radial_table


def log_params(model, bt):
    sp = bt["species"]
    I = model.scaled(bt["feats"])
    if getattr(model, "env", None) is not None:
        model.env_prepare(bt, sp)
    w, _, C6, _ = model.frozen_params(I, sp)
    q = model.ct_charges_from(I, sp)
    lw = torch.log(w.clamp(min=1e-300)).max(1).values if w.dim() == 2 else torch.log(w)
    return {"pauli": lw, "C6": torch.log(C6), "qD": torch.log(q[:, 0]), "qA": torch.log(q[:, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--min-ions", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--fields", type=int, default=3)
    a = ap.parse_args()
    caches = [c for c in torch.load(a.cache, weights_only=False) if c.species.shape[0] >= a.min_ions][:a.frames]
    print(f"{len(caches)} frames with >= {a.min_ions} ions; delta {a.delta} A; {a.fields} displacement fields")
    print(f"{'run':48s} " + " ".join(f"{k:>8s}" for k in ("pauli", "C6", "qD", "qA")))
    for path in a.models:
        ck = torch.load(path, weights_only=False)
        args = ck["args"]
        tab, chan = IonTable.load(args["params"]), CT.ChannelTable.load(args["params"])
        model = build_model(args, tab, chan, ck["n_feat"], None, ck.get("feature_names"))
        model.load_state_dict(ck["state"])
        model.eval()
        table = radial_table(tab, args["params"], args.get("cache_dir"))
        pos = torch.cat([c.pos for c in caches]).double()
        Z = torch.tensor([Zmap[tab.symbols[int(s)]] for c in caches for s in c.species])
        fr = torch.cat([torch.full((c.species.shape[0],), k, dtype=torch.long) for k, c in enumerate(caches)])
        g = torch.Generator().manual_seed(7)
        acc = {}
        with torch.no_grad():
            for _ in range(a.fields):
                xi = torch.randn(pos.shape, generator=g, dtype=pos.dtype)
                xi = xi / xi.norm(dim=1, keepdim=True)
                us = [log_params(model, batched_precompute(pos + s * a.delta * xi, Z, fr, tab, table, chan,
                                                           inv_form=model.inv_form)) for s in (0.0, 1.0, -1.0)]
                for k in us[0]:
                    d2 = (us[1][k] - 2 * us[0][k] + us[2][k]) / a.delta ** 2
                    acc.setdefault(k, []).append(d2)
        rms = {k: float(torch.cat(v).pow(2).mean().sqrt()) for k, v in acc.items()}
        print(f"{path[-48:]:48s} " + " ".join(f"{rms[k]:8.2f}" for k in ("pauli", "C6", "qD", "qA")))


if __name__ == "__main__":
    main()
