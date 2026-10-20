"""What the network emits along the pair scans, and what the model makes of it (zig-zag diagnosis).

    python scripts/ionct_pairscan.py RUN_DIR [--step 0.05]

For Na-Cl and Cl-Cl pairs on a fine R grid: the network outputs (confinement c_alpha / c_C of Cl, pseudo-potential
strengths A_ij, overlap gaps Delta^ov per channel), the derived channel quantities (W, Delta, Stark shift), the
energies (E_pol, E_CT, with the ALMO pair scan) and the finite-field polarisabilities along / perpendicular to the
bond (with ALMO's ct-polar values; the alpha-trained distances marked). Then the attribution: alpha_CT recomputed
with one network output at a time replaced by its smooth fit in R (a cubic in R over the grid) -- the output whose
smoothing removes the zig-zag is the one carrying it. Writes pairscan.csv and ionct_pairscan.png into RUN_DIR.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import CHANNEL_TYPES, PSEUDO_TYPES, IonCTModel, collate, precompute
from rsfff.train.train_ionct import F_SET, KJ, POLAR_TEST, polar_fields

warnings.filterwarnings("ignore")


def load(run):
    ck = torch.load(Path(run) / "model.pt", weights_only=False)
    a = ck["args"]
    tab, chan = IonTable.load(a["params"]), CT.ChannelTable.load(a["params"])
    from rsfff.train.train_ionct import build_model
    m = build_model(a, tab, chan, ck["n_feat"], ck["state"].get("feat_scale"), ck.get("feature_names"))
    m.load_state_dict(ck["state"])
    m.eval()
    from rsfff.train.train_ionct import featurizer_for
    feat = featurizer_for(a, tab)
    return m, tab, chan, feat, a


class Override:
    """Replace one network output by given values inside the model's forward (for the attribution)."""

    def __init__(self, m):
        self.m, self.vals = m, {}
        self._conf, self._gap, self._ps = m.confinement, m.overlap_gap, m.pseudo_strength

    def __enter__(self):
        m, v = self.m, self.vals
        if "c" in v:
            m.confinement = lambda h, h0, sp=None: v["c"]()
        if "dov" in v:
            m.overlap_gap = lambda h, idx, r, ty: v["dov"]()
        if "A" in v:
            m.pseudo_strength = lambda h, idx, r, ty: v["A"]()
        return self

    def __exit__(self, *e):
        self.m.confinement, self.m.overlap_gap, self.m.pseudo_strength = self._conf, self._gap, self._ps


def scan(m, tab, chan, feat, pair, R):
    caches = [precompute(list(pair), [[0, 0, 0], [0, 0, r]], tab, chan, feat) for r in R]
    B = len(caches)
    bt7 = collate([c for c in caches for _ in range(7)])
    fields = polar_fields(B)
    bt = collate(caches)
    return caches, bt, bt7, fields


def alphas(m, bt7, fields, B):
    o = m(bt7, field=fields)
    e = torch.stack([o["e_pol"], o["e_ct"]], -1).reshape(B, 7, 2)
    a = -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / F_SET ** 2          # (B, 3 axes, [pol, ct])
    return a


def smooth(R, y, deg=3):
    return np.polyval(np.polyfit(R, y, deg), R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=float, default=0.05)
    a_ = ap.parse_args()
    m, tab, chan, feat, args = load(a_.run)
    # ALMO: pair scans (energies) and ct-polar (alpha)
    scanref = {}
    for line in open(Path(args["ct_sets"]) / "ct-scan.jsonl"):
        r = json.loads(line)
        t = r.get("tags") or {}
        if r.get("stage") == "label" and "eda" in r and len(r.get("symbols") or []) == 2 and not t.get("field"):
            key = "-".join(sorted(r["symbols"]))
            d = float(np.linalg.norm(np.subtract(r["coords"][0], r["coords"][1])))
            scanref.setdefault(key, []).append((d, r["eda"]["pol"], r["eda"]["ct"]))
    from ionct_polar import qm_polar
    qm = qm_polar(Path(args["ct_sets"]) / "ct-polar.jsonl")
    rows = []
    for pair, (r0, r1) in ((("Na", "Cl"), (2.0, 6.0)), (("Cl", "Cl"), (2.9, 6.2))):
        R = np.round(np.arange(r0, r1 + 1e-9, a_.step), 4)
        caches, bt, bt7, fields = scan(m, tab, chan, feat, pair, R)
        B = len(R)
        with torch.no_grad():
            o = m(bt)
            a = alphas(m, bt7, fields, B)
            sp = bt["species"]
            # per frame, per ion / channel / pseudo pair
            cl = (sp == tab.symbols.index("Cl")).nonzero().squeeze(1)
            c_cl = o["c"][cl].reshape(B, -1, 2)[:, 0]                       # first Cl of each frame
            # direct CT: the donor / acceptor charges of the first ion of each species per frame
            qct = {}
            if getattr(m, "ct_form", "channels") == "direct" and o.get("q_ct") is not None and o["q_ct"].numel():
                for sym in set(pair):
                    idx = (sp == tab.symbols.index(sym)).nonzero().squeeze(1)
                    qct[sym] = o["q_ct"][idx].reshape(B, -1, 2)[:, 0]
            names = [CHANNEL_TYPES[t] for t in bt["ch_type"].tolist()]
            fr = bt["ch_frame"]
            st = o.get("stark", torch.zeros_like(o["dov"]))
            per = {}
            for q, nm in enumerate(names):
                k = int(fr[q])
                d = per.setdefault((k, nm), {"dov": o["dov"][q].item(), "W": o["W"][q].item(),
                                             "Delta": o["Delta"][q].item(), "stark": st[q].item(),
                                             "p": o["p"][q].item()})
            psA = {}
            if m.pseudo:
                for q, (i, j) in enumerate(bt["ps_index"].T.tolist()):
                    k = int(bt["frame"][i])
                    psA[(k, PSEUDO_TYPES[int(bt["ps_type"][q])])] = o["A_ps"][q].item()
        # attribution: smooth one output at a time (over R) and recompute alpha
        att = {}
        Rt = torch.tensor(R)
        with torch.no_grad():
            base_c = o["c"].clone()
            base_dov = o["dov"].clone()
            base_A = o["A_ps"].clone()
            frame_atom = bt["frame"]
            frame_ch = bt["ch_frame"]
            frame_ps = bt["frame"][bt["ps_index"][0]] if m.pseudo else None

            def smoothed(vals, frame_of, keys):
                out = vals.clone()
                for kk in torch.unique(keys):
                    sel = keys == kk
                    per_frame = torch.zeros(B, dtype=vals.dtype)
                    per_frame[frame_of[sel]] = vals[sel]
                    sm = torch.tensor(smooth(R, per_frame.numpy()))
                    out[sel] = sm[frame_of[sel]]
                return out

            def rep7(x):           # expand per-frame quantities to the 7-field batch (geometry-major)
                return x

            # smoothed outputs on the base batch, expanded for the stencil batch
            cs = torch.stack([smoothed(base_c[:, 0], frame_atom, sp * 0 + sp), smoothed(base_c[:, 1], frame_atom, sp)], 1)
            dovs = smoothed(base_dov, frame_ch, bt["ch_type"])
            As = smoothed(base_A, frame_ps, bt["ps_type"]) if m.pseudo else None

            def expand(x, frame_of_base):
                # base items are frame-major; the stencil batch repeats each frame 7 times
                parts = [x[frame_of_base == k] for k in range(B)]
                return torch.cat([p for p in parts for _ in range(7)])

            variants = {"c": ("c", lambda: expand(cs, frame_atom)), "dov": ("dov", lambda: expand(dovs, frame_ch))}
            if m.pseudo:
                variants["A"] = ("A", lambda: expand(As, frame_ps))
            for name, (key, fn) in variants.items():
                ov = Override(m)
                ov.vals[key] = fn
                with ov:
                    att[name] = alphas(m, bt7, fields, B)
        ref = {round(d, 3): (p, c) for d, p, c in scanref.get("-".join(sorted(pair)), [])}
        for k, r in enumerate(R):
            pid = f"pair_{'-'.join(pair)}_r{r:.3f}"
            q = qm.get(pid, (None, None, {}))[2]
            row = {"pair": "-".join(pair), "r": r, "c_alpha_Cl": c_cl[k, 0].item(), "c_C_Cl": c_cl[k, 1].item(),
                   "e_pol": o["e_pol"][k].item() * KJ, "e_ct": o["e_ct"][k].item() * KJ,
                   "a_pol_par": a[k, 2, 0].item(), "a_pol_perp": a[k, 0, 0].item(),
                   "a_ct_par": a[k, 2, 1].item(), "a_ct_perp": a[k, 0, 1].item(),
                   "alpha_trained": bool(pid in qm and pid not in set(args.get("polar_test") or POLAR_TEST)
                                         and args.get("w_alpha", 0) > 0 and not args.get("holdout_scans"))}
            for name, aa in att.items():
                row[f"a_ct_par_smooth_{name}"] = aa[k, 2, 1].item()
                row[f"a_pol_par_smooth_{name}"] = aa[k, 2, 0].item()
            for sym, qq in qct.items():
                row[f"qD_{sym}"], row[f"qA_{sym}"] = qq[k, 0].item(), qq[k, 1].item()
            for (kk, nm), d in per.items():
                if kk == k:
                    for f, v in d.items():
                        row[f"{f}[{nm}]"] = v
            for (kk, nm), v in psA.items():
                if kk == k:
                    row[f"A[{nm}]"] = v
            near = [d for d in ref if abs(d - r) < 1e-3]
            if near:
                row["qm_pol"], row["qm_ct"] = ref[near[0]]
            if q:
                ax_par = "z"
                ax_perp = [x for x in q if x != "z"]
                row["qm_ct_par"] = q[ax_par]["ct"]
                row["qm_mi_par"] = q[ax_par]["mi"]
                if ax_perp:
                    row["qm_ct_perp"] = q[ax_perp[0]]["ct"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a_.run) / "pairscan.csv", index=False)
    global HOLDOUT
    HOLDOUT = bool(args.get("holdout_scans"))
    plot(df, Path(a_.run) / "ionct_pairscan.png")
    for p, g in df.groupby("pair"):
        print(p)
        cols = [c for c in g.columns if c.startswith(("a_ct_par", "dov[", "c_alpha", "A["))]
        print(g[["r"] + cols].iloc[::4].round(3).to_string(index=False))


HOLDOUT = False


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(13, 14), sharex="col")
    for col, pair in enumerate(("Na-Cl", "Cl-Cl")):
        g = df[df.pair == pair].sort_values("r")
        R = g.r
        tr = g[g.alpha_trained]
        # energies
        ax = axs[0, col]
        ax.plot(R, g.e_pol, "C0-", label="model E_pol")
        ax.plot(R, g.e_ct, "C3-", label="model E_CT")
        if "qm_pol" in g:
            ax.plot(R, g.qm_pol, "C0o", mfc="none", ms=4, label="ALMO pol")
            ax.plot(R, g.qm_ct, "C3o", mfc="none", ms=4, label="ALMO ct")
        ax.set_ylabel("kJ/mol")
        ax.set_title(f"{pair}: energies (pair scans {'HELD OUT' if HOLDOUT else 'in training'})")
        ax.legend(fontsize=8)
        # confinement and pseudo
        ax = axs[1, col]
        ax.plot(R, g.c_alpha_Cl, "k-", label="c_α (Cl)")
        ax.plot(R, g.c_C_Cl, "k--", label="c_C (Cl)")
        for c in [c for c in g.columns if c.startswith("A[")]:
            mx = g[c].abs().max()
            if np.isfinite(mx) and mx > 0:
                ax.plot(R, g[c] / mx, label=f"{c} / max ({mx:.2g})")
        for r in tr.r.unique():
            ax.axvline(r, color="0.8", lw=0.8, zorder=0)
        ax.set_title("network: confinement, pseudo-potential (grey: α-trained R)")
        ax.legend(fontsize=7)
        direct = "qD_Cl" in g.columns and g["qD_Cl"].notna().any()
        ax = axs[2, col]
        if direct:
            # direct CT: the donor / acceptor charges along R (exactly 1 for an isolated ion)
            for sym, ls in (("Cl", "-"), ("Na", "--")):
                if f"qD_{sym}" in g and g[f"qD_{sym}"].notna().any():
                    ax.plot(R, g[f"qD_{sym}"], "C0", ls=ls, label=f"qD ({sym})")
                    ax.plot(R, g[f"qA_{sym}"], "C1", ls=ls, label=f"qA ({sym})")
            ax.axhline(1.0, color="0.6", lw=0.8)
            ax.set_ylabel("e")
            ax.set_title("network: direct CT charges (1 = isolated)")
        else:
            for c in [c for c in g.columns if c.startswith("dov[")]:
                ax.plot(R, g[c], label=f"Δ^ov {c[4:-1]}")
            for c in [c for c in g.columns if c.startswith("stark[")]:
                ax.plot(R, g[c], ":", label=f"Stark {c[6:-1]}")
            ax.set_ylabel("Eh")
            ax.set_title("network: overlap gaps; Stark shift")
        for r in tr.r.unique():
            ax.axvline(r, color="0.8", lw=0.8, zorder=0)
        ax.legend(fontsize=7)
        # alpha with attribution
        ax = axs[3, col]
        if direct:
            # the direct form has no field response (α_CT = 0): show α_pol against ALMO's α_MI instead
            ax.plot(R, g.a_pol_par, "C0-", lw=2, label="α_pol∥ model")
            ax.plot(R, g.a_pol_perp, "C0--", lw=1, label="α_pol⊥ model")
            for name, ls in (("c", ":"), ("A", "-.")):
                k = f"a_pol_par_smooth_{name}"
                if k in g:
                    ax.plot(R, g[k], "C0", ls=ls, lw=1, label=f"∥ with {name} smoothed in R")
            if "qm_mi_par" in g:
                q = g.dropna(subset=["qm_mi_par"])
                ax.plot(q.r, q.qm_mi_par, "C0o", mfc="none", label="ALMO α_MI∥")
                qt = q[q.alpha_trained]
                ax.plot(qt.r, qt.qm_mi_par, "C0o", ms=6, alpha=0.4, label="(in α loss)")
            ax.set_title("α_pol along the bond vs ALMO α_MI (direct CT: α_CT ≡ 0)")
        else:
            ax.plot(R, g.a_ct_par, "C3-", lw=2, label="α_CT∥ model")
            for name, ls in (("dov", "--"), ("c", ":"), ("A", "-.")):
                k = f"a_ct_par_smooth_{name}"
                if k in g:
                    ax.plot(R, g[k], "C3", ls=ls, lw=1, label=f"… with {name} smoothed in R")
            ax.plot(R, g.a_pol_par - g.a_pol_par.iloc[-1], "C0-", lw=1, label="α_pol∥ (shifted)")
            if "qm_ct_par" in g:
                q = g.dropna(subset=["qm_ct_par"])
                ax.plot(q.r, q.qm_ct_par, "C3o", mfc="none", label="ALMO α_CT∥")
                qt = q[q.alpha_trained]
                ax.plot(qt.r, qt.qm_ct_par, "C3o", ms=6, alpha=0.4, label="(in α loss)")
            ax.set_title("α along the bond; attribution")
        ax.set_xlabel("R (Å)")
        ax.set_ylabel("bohr³")
        ax.legend(fontsize=7)
        for a in axs[:, col]:
            a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
