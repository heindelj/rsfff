#!/usr/bin/env python
"""Per-geometry Slater-damped multipole fits to an elst_db database (rsfff.qcgen.elst_fit).

    python scripts/elst_fit.py ~/research/rsfff/data/elst_db --model slater_r2 --out fits_slater_r2.json
    python scripts/elst_fit.py DB --model slater_r2 --grad 0.5 --out fits_slater_r2_grad.json   # + field gradients

Models: slater_r2 / slater_r1 (FF form: point nucleus Z + Slater 1s shell, rank 2 / 1), point_r2 (point
multipoles, fitted to the >= 1.4 x vdW shells only).
"""
import argparse

import torch

from rsfff.qcgen import elst_db as D
from rsfff.qcgen import elst_fit as F

MODELS = {"slater_r2": dict(max_rank=2), "slater_r1": dict(max_rank=1),
          "point_r2": dict(penetration=False, min_scale=1.4)}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("db")
ap.add_argument("--model", default="slater_r2", choices=list(MODELS))
ap.add_argument("--out", required=True)
ap.add_argument("--names", nargs="*")
ap.add_argument("--grad", type=float, default=0.0, help="weight of the field-gradient term (adds field_grad to grids)")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
db = D.MonomerDB(a.db)
kw = dict(MODELS[a.model])
if a.grad:
    for n in a.names or db.index:
        D.monomer_grid(db, n, gradients=True)
    kw["w_grad"] = a.grad
fits = F.fit_db(db, names=a.names, **kw)
F.save_fits(a.out, fits, meta={"model": a.model, "w_grad": a.grad})
print(f"{len(fits)} fits -> {a.out}")
