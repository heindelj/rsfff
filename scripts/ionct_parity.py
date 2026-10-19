"""Parity of the ion CT force field's energy terms against ALMO-EDA: frozen electrostatics, polarisation, CT.

    python scripts/ionct_parity.py RUN_DIR --ct-sets CT_SETS [--walk-eda DIR] [--cache CACHE]

Electrostatics is the fixed shaped-density permanent term (no network) against ALMO's cls_elec; E_pol and E_CT are
the trained model's against ALMO's pol and ct. One point per deduplicated frame of the training data (every split);
CT frames outside the label mask (ionic-covalent mixing) are shown greyed. Writes parity.csv and ionct_parity.png
into RUN_DIR.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.model import IonCTModel, collate
from rsfff.train.train_ionct import KJ, _geom_key, build_cache, load_frames, split

warnings.filterwarnings("ignore")


def elec_labels(ct_sets, walk_eda):
    """{geometry key: ALMO cls_elec (kJ/mol)}."""
    out = {}
    for wf in ("ct-walk", "ct-mbe", "ct-sat", "ct-scan"):
        p = Path(ct_sets) / f"{wf}.jsonl"
        for line in open(p) if p.exists() else []:
            r = json.loads(line)
            if r.get("stage") == "label" and "eda" in r and r.get("symbols") and not (r.get("tags") or {}).get("field"):
                out.setdefault(_geom_key(r["symbols"], r["coords"]), r["eda"]["cls_elec"])
    if walk_eda:
        from ase.io import read
        for f in sorted(Path(walk_eda).glob("*.xyz")):
            for a in read(f, ":"):
                if "eda_cls_elec" in a.info:
                    out.setdefault(_geom_key(a.get_chemical_symbols(), a.positions), a.info["eda_cls_elec"] * KJ)
    return out


def frame(run, ct_sets, walk_eda=None, cache=None):
    run = Path(run)
    ck = torch.load(run / "model.pt", weights_only=False)
    a = ck["args"]
    tab, chan = IonTable.load(a["params"]), CT.ChannelTable.load(a["params"])
    walk_eda = walk_eda or a.get("walk_eda")
    frames = load_frames(ct_sets, walk_eda, a["ct_max_pair"], a.get("max_ho_ratio", 1.6))
    tags = split(frames, a["test"], seed=a["seed"])
    caches = build_cache(frames, tab, chan, None, cache or a.get("cache"))
    model = IonCTModel(tab, chan, ck["n_feat"], hidden=a["hidden"], wkb=a["wkb"],
                       c_range=tuple(a.get("c_range", (0.05, 1.25))), confine=a.get("confine"),
                       pseudo=a.get("pseudo", False), joint=a.get("joint", False), ct_pol=a.get("ct_pol", False), gap_form=a.get("gap_form", "pair"),
                       tie_c=a.get("tie_c", False), coupling_resp=a.get("coupling_resp", "barrier"),
                       ct_form=a.get("ct_form", "channels"), pol_form=a.get("pol_form", "confine"), pauli_nn=a.get("pauli_nn", False))
    model.load_state_dict(ck["state"])
    model.eval()
    elec = elec_labels(ct_sets, walk_eda)
    rows = []
    with torch.no_grad():
        for k in range(0, len(caches), 64):
            chunk = caches[k:k + 64]
            out = model(collate(chunk))
            for q, c in enumerate(chunk):
                sym, X, lab, meta = frames[k + q]
                n = len(sym)
                e_el = K.permanent_pair_energy(c.pos, c.species, torch.triu_indices(n, n, 1), tab).sum().item() * KJ
                rows.append({"split": tags[k + q], "workflow": meta["workflow"], "parent": meta["parent"], "n": n,
                             "full": bool(meta["full"]), "ct_mask": bool(lab["ct_mask"]),
                             "elec_ref": elec.get(_geom_key(sym, X), np.nan), "elec": e_el,
                             "pol_ref": lab["pol"] * KJ, "pol": out["e_pol"][q].item() * KJ,
                             "ct_ref": lab["ct"] * KJ, "ct": out["e_ct"][q].item() * KJ,
                             "pauli_ref": lab.get("pauli", np.nan) * KJ,
                             "pauli": out.get("e_pauli", out["e_pol"] * 0)[q].item() * KJ})
    return pd.DataFrame(rows)


def _stats(x, y):
    d = y - x
    return np.abs(d).mean(), np.sqrt((d ** 2).mean()), np.corrcoef(x, y)[0, 1] ** 2


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    terms = [("elec", "frozen electrostatics (fixed)"), ("pol", "polarisation"), ("ct", "charge transfer")]
    if "pauli" in df and (df.pauli != 0).any():
        terms.append(("pauli", "frozen Pauli (kernel) vs mod_pauli"))
    fig, axs = plt.subplots(2, len(terms), figsize=(5 * len(terms), 9.4))
    style = {"train": dict(color="0.55", ms=3, alpha=0.35, zorder=1),
             "val": dict(color="tab:blue", ms=3.5, alpha=0.6, zorder=2),
             "test": dict(color="tab:orange", ms=4.5, alpha=0.9, zorder=3)}
    for col, (t, title) in enumerate(terms):
        d = df[np.isfinite(df[f"{t}_ref"])]
        ok = d if t != "ct" else d[d.ct_mask]
        ax = axs[0, col]
        if t == "ct":
            bad = d[~d.ct_mask]
            ax.plot(bad[f"{t}_ref"], bad[t], "x", color="0.75", ms=3, zorder=0, label="masked (ionic–covalent)")
        for s, kw in style.items():
            g = ok[ok.split == s]
            ax.plot(g[f"{t}_ref"], g[t], "o", ls="none", mec="none", label=f"{s} ({len(g)})", **kw)
        lo = min(ok[f"{t}_ref"].min(), ok[t].min())
        hi = max(ok[f"{t}_ref"].max(), ok[t].max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"({'abcd'[col]}) {title}")
        ax.set_xlabel(f"ALMO {dict(elec='cls_elec', pauli='mod_pauli').get(t, t)} (kJ/mol)")
        ax.set_ylabel("model (kJ/mol)")
        txt = []
        for s in ("train", "test"):
            g = ok[ok.split == s]
            if len(g) > 2:
                mae, rms, r2 = _stats(g[f"{t}_ref"].to_numpy(), g[t].to_numpy())
                gf = g[g.full & (g.n > 2)]
                mae_f = np.abs(gf[t] - gf[f"{t}_ref"]).mean() if len(gf) else np.nan
                txt.append(f"{s}: MAE {mae:.2f}, RMS {rms:.2f}, R² {r2:.4f}\n   clusters MAE {mae_f:.2f}")
        ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.9))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        # per-ion error against size
        ax = axs[1, col]
        for s, kw in style.items():
            g = ok[ok.split == s]
            jit = np.random.default_rng(0).uniform(-0.2, 0.2, len(g))
            ax.plot(g.n + jit, (g[t] - g[f"{t}_ref"]) / g.n, "o", ls="none", mec="none", **kw)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("ions in frame")
        ax.set_ylabel("(model − ALMO) / n (kJ/mol per ion)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    df = frame(a.run, a.ct_sets, a.walk_eda, a.cache)
    df.to_csv(Path(a.run) / "parity.csv", index=False)
    plot(df, Path(a.run) / "ionct_parity.png")
    print("missing elec labels:", int(df.elec_ref.isna().sum()), "of", len(df))
    for t in ("elec", "pol", "ct", "pauli"):
        if f"{t}_ref" not in df or not (df[t] != 0).any():
            continue
        d = df[np.isfinite(df[f"{t}_ref"]) & (df.ct_mask if t == "ct" else True)]
        print(t, d.groupby("split").apply(lambda g: pd.Series(_stats(g[f"{t}_ref"].to_numpy(), g[t].to_numpy()),
                                                              index=["mae", "rms", "r2"])).round(3).to_string())


if __name__ == "__main__":
    main()
