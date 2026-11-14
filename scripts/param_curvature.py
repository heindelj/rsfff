"""Geometric smoothness of a cmm run's emitted parameters: rms second difference (per A^2) of the log
Pauli weight, log C6 and log q_D / log q_A along random per-atom displacements, on fixed cluster frames.

    python scripts/param_curvature.py CACHE.pt RUN/model.pt [RUN2/model.pt ...] [--frames 24 --delta 0.02 --fields 3]

CACHE.pt: a cmm trainer's frame cache (or any list with .pos / .species); the models: cmm checkpoints, run
tarballs or legacy complete-model checkpoints (rsfff.ff.cmm.load_cmm). Runs trained with and without the
curvature prior compare on the same footing."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rsfff.ff.cmm import batched_precompute, load_cmm
from rsfff.ff.cmm.params import ATOMIC_NUMBER as Zmap
from rsfff.train.cmm import radial_table


def log_params(model, bt):
    sp = bt["species"]
    model.environment(bt)
    w, _, C6, _ = model.frozen_params(sp)
    q = model.ct_charges(sp)
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
        model = load_cmm(path)
        tab, chan = model.tab, model.chan
        table = radial_table(tab, model.params_path, None)
        pos = torch.cat([c.pos for c in caches]).double()
        Z = torch.tensor([Zmap[tab.symbols[int(s)]] for c in caches for s in c.species])
        fr = torch.cat([torch.full((c.species.shape[0],), k, dtype=torch.long) for k, c in enumerate(caches)])
        g = torch.Generator().manual_seed(7)
        acc = {}
        with torch.no_grad():
            for _ in range(a.fields):
                xi = torch.randn(pos.shape, generator=g, dtype=pos.dtype)
                xi = xi / xi.norm(dim=1, keepdim=True)
                us = [log_params(model, batched_precompute(pos + s * a.delta * xi, Z, fr, tab, table, chan))
                      for s in (0.0, 1.0, -1.0)]
                for k in us[0]:
                    d2 = (us[1][k] - 2 * us[0][k] + us[2][k]) / a.delta ** 2
                    acc.setdefault(k, []).append(d2)
        rms = {k: float(torch.cat(v).pow(2).mean().sqrt()) for k, v in acc.items()}
        print(f"{path[-48:]:48s} " + " ".join(f"{rms[k]:8.2f}" for k in ("pauli", "C6", "qD", "qA")))


if __name__ == "__main__":
    main()
