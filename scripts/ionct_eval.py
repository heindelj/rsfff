"""Diagnostics of a trained ion CT model (rsfff.train.train_ionct): channel types, learned overlap gaps, confinement.

    python scripts/ionct_eval.py RUN_DIR --ct-sets CT_SETS [--walk-eda DIR] [--cache CACHE]

Per split, for the whole clusters: the model's channel energies (second order, -W/Delta, per directed channel)
summed by channel type against ALMO-CTA's; the learned overlap gap Delta^ov by channel type and its correlation with
the acceptor's coordination; the confinement c_alpha against the ion's nearest-neighbour distance. Writes
channels.csv and atoms.csv into RUN_DIR.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import CHANNEL_TYPES, IonCTModel, collate
from rsfff.train.train_ionct import KJ, build_cache, load_frames, split

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    ck = torch.load(run / "model.pt", weights_only=False)
    args = ck["args"]
    tab, chan = IonTable.load(args["params"]), CT.ChannelTable.load(args["params"])
    frames = load_frames(a.ct_sets, a.walk_eda or args.get("walk_eda"), args["ct_max_pair"], args.get("max_ho_ratio", 1.6))
    tags = split(frames, args["test"], seed=args["seed"])
    caches = build_cache(frames, tab, chan, None, a.cache or args.get("cache"))
    model = IonCTModel(tab, chan, ck["n_feat"], hidden=args["hidden"], wkb=args["wkb"],
                       c_range=tuple(args.get("c_range", (0.05, 1.25))), confine=args.get("confine"),
                       pseudo=args.get("pseudo", False), joint=args.get("joint", False), ct_pol=args.get("ct_pol", False), gap_form=args.get("gap_form", "pair"),
                       tie_c=args.get("tie_c", False), coupling_resp=args.get("coupling_resp", "barrier"),
                       ct_form=args.get("ct_form", "channels"), pol_form=args.get("pol_form", "confine"), pauli_nn=args.get("pauli_nn", False))
    model.load_state_dict(ck["state"])
    model.eval()
    ch_rows, at_rows = [], []
    with torch.no_grad():
        for c, tag in zip(caches, tags):
            n = c.species.shape[0]
            bt = collate([c])
            out = model(bt)
            sym = [tab.symbols[k] for k in c.species.tolist()]
            D = torch.cdist(c.pos, c.pos) + 1e3 * torch.eye(n)
            for k in range(n):
                at_rows.append({"split": tag, "parent": c.meta["parent"], "n": n, "full": c.meta["full"], "ion": sym[k],
                                "r_nn": float(D[k].min()) if n > 1 else float("nan"),
                                "n_nb_3.2": int((D[k] < 3.2).sum()), "c_alpha": out["c"][k, 0].item(),
                                "c_C": out["c"][k, 1].item()})
            E = c.meta.get("cta_E")
            if E is None or not c.meta["full"] or n < 3:
                continue
            E = np.nan_to_num(np.array([[np.nan if v is None else v for v in x] for x in E], float))[:n, :n]
            i, j = c.ch_index
            e2 = (-out["W"] / out["Delta"]).numpy() * KJ
            for q in range(i.shape[0]):
                a_, b_ = int(i[q]), int(j[q])
                nA = int((D[b_] < 3.2).sum()) - int(D[b_, a_] < 3.2)
                ch_rows.append({"split": tag, "parent": c.meta["parent"], "n": n, "type": CHANNEL_TYPES[int(c.ch_type[q])],
                                "r": float(c.ch_r[q]), "E_cta": E[a_, b_], "E_model": float(e2[q]),
                                "dov": out["dov"][q].item(), "Delta": out["Delta"][q].item(), "W": out["W"][q].item(),
                                "nA_other": nA})
    for name, rows in (("channels", ch_rows), ("atoms", at_rows)):
        with open(run / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    import pandas as pd
    ch = pd.DataFrame(ch_rows)
    g = ch.groupby(["split", "type"]).agg(E_cta=("E_cta", "sum"), E_model=("E_model", "sum"), n=("E_cta", "size"),
                                          dov_mean=("dov", "mean"))
    print(g.round(2).to_string())
    cl = ch.groupby(["split", "parent", "type"])[["E_cta", "E_model"]].sum().reset_index()
    for t in ("Cl>Na", "Cl>Cl", "Na>Cl"):
        x = cl[(cl.type == t) & (cl.split == "test")]
        if len(x):
            print(f"test {t}: per-cluster sum, ALMO {x.E_cta.mean():.2f} model {x.E_model.mean():.2f} "
                  f"r {np.corrcoef(x.E_cta, x.E_model)[0, 1]:.2f}")
    at = pd.DataFrame(at_rows)
    print(at[at.full].groupby(["ion", "n_nb_3.2"])[["c_alpha", "c_C"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
