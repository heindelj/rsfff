"""P0 of the m33 plan: how well do the complete model's pair kernels describe the dimers on their own?

Builds the model of a train_ionct run (its model.pt carries the arguments), and compares on every two-ion frame of the
run's data:
  (a) the run as trained (heads on),
  (b) heads zeroed with the run's per-species constants,
  (c) heads zeroed with the frozen-term constants (Pauli shell weights and exponent scale, C6 and dispersion exponent)
      refitted to the dimers alone -- what the kernels can do with per-species constants only.
Pauli and dispersion per pair type and distance bin. CT is reported for R < --ct-max (default 3.6 A) only and never
fitted: the vacuum NaCl dimer's CT mixes with the radical (neutral) asymptote beyond that.

Run from research/rsfff:
    python scripts/pair_kernel_audit.py RUN/model.pt --out audit.json [--remap /Users/x/research/ct_model=/path]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import channel_types
from rsfff.train.train_ionct import KJ, build_cache, build_model, cache_key, collate, featurizer_for, load_frames

BINS = (0.0, 2.6, 3.0, 3.6, 5.0, 99.0)
FROZEN_CONSTANTS = ("pauli_lnw", "pauli_lnbs", "disp_lnc60", "disp_lnb")


def _remap(p, rules):
    s = str(p)
    for a, b in rules:
        s = s.replace(a, b)
    return s


def zero_heads(model):
    """Zero every environment head's output layer and linear term (outputs exactly 0 -> parameters = constants)."""
    from rsfff.ff.complete.heads import EnvironmentHead
    for m in model.modules():
        if isinstance(m, EnvironmentHead):
            torch.nn.init.zeros_(m.lin.weight)
            torch.nn.init.zeros_(m.net[-1].weight)
            torch.nn.init.zeros_(m.net[-1].bias)
    env = getattr(model, "env", None)              # m33 shared environment, if present
    if env is not None and hasattr(env, "zero_"):
        env.zero_()


def frozen_energies(model, bt):
    sp = bt["species"]
    I = model.scaled(bt["feats"])
    if getattr(model, "env", None) is not None:
        model.env_prepare(bt, sp)
    e_p, e_d, _ = model.frozen_terms(I, bt, sp)
    return e_p, e_d


def table(model, caches, bs=128, with_ct=True, ct_max=3.6):
    rows = []
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            chunk = caches[k:k + bs]
            bt = collate(chunk)
            if with_ct:
                out = model(bt)
                e_p, e_d, e_c = out["e_pauli"], out["e_disp"], out["e_ct"]
            else:
                e_p, e_d = frozen_energies(model, bt)
                e_c = torch.full_like(e_p, float("nan"))
            for q, c in enumerate(chunk):
                sym = tuple(sorted(model.tab.symbols[s] for s in c.species.tolist()))
                r = float((c.pos[0] - c.pos[1]).norm())
                rows.append(dict(pair="-".join(sym), r=r,
                                 pauli=(e_p[q].item() - c.labels.get("pauli", np.nan)) * KJ,
                                 disp=(e_d[q].item() - c.labels.get("disp", np.nan)) * KJ,
                                 pauli_ref=c.labels.get("pauli", np.nan) * KJ, disp_ref=c.labels.get("disp", np.nan) * KJ,
                                 ct=((e_c[q].item() - c.labels["ct"]) * KJ) if r < ct_max else np.nan,
                                 ct_ref=c.labels["ct"] * KJ if r < ct_max else np.nan))
    return rows


def summarize(rows):
    out = {}
    for pair in sorted({r["pair"] for r in rows}):
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = [r for r in rows if r["pair"] == pair and lo <= r["r"] < hi]
            if not sel:
                continue
            d = {"n": len(sel)}
            for key in ("pauli", "disp", "ct"):
                e = np.array([r[key] for r in sel], float)
                y = np.array([r[f"{key}_ref"] for r in sel], float)
                ok = np.isfinite(e) & np.isfinite(y)
                if ok.any():
                    d[f"{key}_mae"] = round(float(np.abs(e[ok]).mean()), 3)
                    d[f"{key}_ref_mean"] = round(float(np.abs(y[ok]).mean()), 3)
            out[f"{pair} [{lo:.1f},{hi:.1f})"] = d
    return out


def refit_frozen(model, caches, steps=400, lr=0.02):
    """Fit the frozen-term constants to the dimers' Pauli and dispersion labels (heads zeroed)."""
    params = [p for n, p in model.named_parameters() if n in FROZEN_CONSTANTS]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    bt = collate(caches)
    yp = torch.tensor([c.labels.get("pauli", np.nan) for c in caches], dtype=torch.float64)
    yd = torch.tensor([c.labels.get("disp", np.nan) for c in caches], dtype=torch.float64)
    mp, md = torch.isfinite(yp), torch.isfinite(yd)
    for it in range(steps):
        opt.zero_grad()
        e_p, e_d = frozen_energies(model, bt)
        loss = ((e_p - torch.nan_to_num(yp)).abs() * mp).sum() / mp.sum() + ((e_d - torch.nan_to_num(yd)).abs() * md).sum() / md.sum()
        loss.backward()
        opt.step()
    return {n: p.detach().tolist() for n, p in model.named_parameters() if n in FROZEN_CONSTANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--remap", nargs="*", default=[], help="path rewrites OLD=NEW for the run's data paths")
    ap.add_argument("--ct-max", type=float, default=3.6)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    rules = [tuple(s.split("=", 1)) for s in a.remap]
    ck = torch.load(a.model, weights_only=False)
    args = dict(ck["args"])
    for k in ("ct_sets", "walk_eda", "params", "cache_dir"):
        if args.get(k):
            args[k] = _remap(args[k], rules)
    args["extra_sets"] = [_remap(p, rules) for p in (args.get("extra_sets") or [])]
    tab, chan = IonTable.load(args["params"]), CT.ChannelTable.load(args["params"])
    feat = featurizer_for(args, tab)
    frames = load_frames(args["ct_sets"], args.get("walk_eda"), args.get("ct_max_pair", 3.6), args.get("max_ho_ratio", 1.4),
                         species=tab.symbols, types=channel_types(tab.symbols), ct_dimers=True,
                         extra_sets=[f"seed-{x}-{k}" for x in args.get("salts", []) for k in ("scan", "mbe", "sat")],
                         extra_files=args["extra_sets"])
    lean = (args.get("ct_form") == "direct") and not args.get("full_precompute", False)
    frontier = args.get("model") == "complete" and not args.get("no_ct_gap", False)
    two = [f for f in frames if len(f[0]) == 2]
    print(f"{len(two)} two-ion frames of {len(frames)}", flush=True)
    path = Path(args["cache_dir"]) / f"cache_{cache_key(args['params'], args['env'], lean, two, frontier)}.pt" if args.get("cache_dir") else None
    dimers = build_cache(two, tab, chan, feat, path, lean=lean, workers=a.workers, frontier=frontier)
    model = build_model(args, tab, chan, ck["n_feat"], None, ck.get("feature_names"))
    model.load_state_dict(ck["state"], strict=False)
    model.eval()
    res = {"model": a.model, "n_dimers": len(dimers)}
    res["trained"] = summarize(table(model, dimers, ct_max=a.ct_max))
    zero_heads(model)
    res["heads_off"] = summarize(table(model, dimers, ct_max=a.ct_max))
    res["refit_constants"] = refit_frozen(model, dimers)
    res["heads_off_refit"] = summarize(table(model, dimers, with_ct=False))
    Path(a.out).write_text(json.dumps(res, indent=1))
    keys = sorted(res["trained"])
    print(f"{'pair / R bin (A)':22s} {'n':>4s} | {'Pauli MAE: trained / off / refit (|ref|)':>44s} | {'disp: trained / off / refit (|ref|)':>40s} | CT trained / off")
    for k in keys:
        t, o, f = res["trained"][k], res["heads_off"].get(k, {}), res["heads_off_refit"].get(k, {})
        print(f"{k:22s} {t['n']:4d} | {t.get('pauli_mae', np.nan):8.2f} {o.get('pauli_mae', np.nan):8.2f} {f.get('pauli_mae', np.nan):8.2f} ({t.get('pauli_ref_mean', np.nan):7.1f}) | "
              f"{t.get('disp_mae', np.nan):7.2f} {o.get('disp_mae', np.nan):7.2f} {f.get('disp_mae', np.nan):7.2f} ({t.get('disp_ref_mean', np.nan):6.1f}) | "
              f"{t.get('ct_mae', np.nan):6.2f} {o.get('ct_mae', np.nan):6.2f}")


if __name__ == "__main__":
    main()
