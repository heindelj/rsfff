"""Finite-field polarisabilities of a trained ion CT model against ALMO-EDA (ct-polar).

    python scripts/ionct_polar.py RUN_DIR --ct-sets CT_SETS [--out FIG]

For every ct-polar geometry the network outputs (confinement c, overlap gaps Delta^ov) are evaluated once at zero
field and held fixed; the induction (E_pol) and CT (E_CT, with the induced multipoles and the field in phi-bar) are
then solved at F = +-0.002 au along each axis. alpha = -d2E/dF2 by central differences, the same stencil as the
ALMO side: alpha_full = -d2E_tot/dF2, alpha_CT = -d2E_CT/dF2, alpha_MI = alpha_full - alpha_CT. The model's alpha
without CT is that of E_pol alone (free-ion response included; the permanent energy is linear in F). The field
rows are not in the training set. Writes polar.csv and the figure into RUN_DIR.
"""
from __future__ import annotations

import argparse
import collections
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import IonCTModel

warnings.filterwarnings("ignore")
F_SET, KJ = 0.002, 2625.4996394799
ZMAP = {"Na": 11, "Cl": 17}
#: CT variants (same network outputs): 'ct' the trained form (joint solve if trained with it), 'ct_post' CT on the
#: polarized reference without feedback, 'ct_0' with the WKB response of W switched off
SHAPE_N = {"lin2": 2, "tri3": 3, "sq4": 4, "tet4": 4, "oct6": 6}


def _field_of(tags):
    f = tags.get("field") or ""
    if not f:
        return "", 0.0
    ax, v = f.split(":")
    return ax, round(float(v), 6)


def qm_polar(path):
    """{polar_id: (symbols, coords, {axis: {full, ct, pol, mi}})} from ct-polar.jsonl (bohr^3)."""
    g = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        t = r.get("tags") or {}
        if r.get("stage") == "label" and "polar_id" in t and "eda" in r:
            g[t["polar_id"]][_field_of(t)] = r
    fn = {"full": lambda r: r["energy"], "ct": lambda r: r["eda"]["ct"] / KJ, "pol": lambda r: r["eda"]["pol"] / KJ}
    out = {}
    for pid, d in g.items():
        if ("", 0.0) not in d:
            continue
        r0, rec = d[("", 0.0)], {}
        for ax in "xyz":
            if (ax, F_SET) in d and (ax, -F_SET) in d:
                rp, rm = d[(ax, F_SET)], d[(ax, -F_SET)]
                rec[ax] = {k: -(f(rp) + f(rm) - 2 * f(r0)) / F_SET ** 2 for k, f in fn.items()}
                rec[ax]["mi"] = rec[ax]["full"] - rec[ax]["ct"]
        out[pid] = (r0["symbols"], np.asarray(r0["coords"], float), rec)
    return out


class Evaluator:
    def __init__(self, run):
        ck = torch.load(Path(run) / "model.pt", weights_only=False)
        a = ck["args"]
        self.tab, self.chan = IonTable.load(a["params"]), CT.ChannelTable.load(a["params"])
        from rsfff.train.train_ionct import build_model
        self.model = build_model(a, self.tab, self.chan, ck["n_feat"], ck["state"].get("feat_scale"), ck.get("feature_names"))
        self.model.load_state_dict(ck["state"])
        # ct-polar geometries the alpha loss trained on (none unless --w-alpha)
        self.alpha_trained = None
        if a.get("w_alpha", 0) > 0:
            self.alpha_held_out = set(a.get("polar_test") or [])
            self.holdout_pairs = bool(a.get("holdout_scans"))
            self.alpha_trained = True
        self.model.eval()
        from rsfff.train.train_ionct import featurizer_for
        self.feat = featurizer_for(a, self.tab)

    @torch.no_grad()
    def alpha(self, symbols, X):
        """{axis: {pol, ct, ct_0, ct_post, total}} (bohr^3) and the confinement c, through the model's own forward
        (cached operators, pseudo-potential drive, joint solve when trained with it) at the 7-field stencil."""
        from rsfff.ff.ionct.model import precompute
        from rsfff.train.train_ionct import polar_batch
        m = self.model
        pb = polar_batch([precompute(symbols, X, self.tab, self.chan, self.feat)])
        e = {}
        wkb = m.wkb
        for key, w in (("ct", wkb), ("ct_0", "none")):
            m.wkb = w
            lam = None
            saved = None
            if key == "ct_0" and getattr(m, "gap_form", "pair") == "density":    # no coupling response
                if getattr(m, "coupling_resp", "barrier") == "frontier":       # (tied: also no CT-state Stark)
                    saved = [m.aorb_acc_raw.data.clone(), m.aorb_don_raw.data.clone()]
                    m.aorb_acc_raw.data.fill_(-40.0)
                    m.aorb_don_raw.data.fill_(-40.0)
                else:
                    lam = m.lam.data.clone()
                    m.lam.data.zero_()
            o = m(pb["bt"], field=pb["field"])
            if lam is not None:
                m.lam.data.copy_(lam)
            if saved is not None:
                m.aorb_acc_raw.data.copy_(saved[0])
                m.aorb_don_raw.data.copy_(saved[1])
            e["pol"] = o["e_pol"]
            e[key] = o["e_ct"]
            if key == "ct":
                e["ct_post"] = o["e_ct_post"]
                c = o["c"][:len(symbols)]
        m.wkb = wkb
        rec = {}
        for k, ax in enumerate("xyz"):
            p, mi = 1 + 2 * k, 2 + 2 * k
            d = {key: (-(v[p] + v[mi] - 2 * v[0]) / F_SET ** 2).item() for key, v in e.items()}
            rec[ax] = {**d, "total": d["pol"] + d["ct"]}
        return rec, c


def frame(run, ct_sets):
    ev = Evaluator(run)
    qm = qm_polar(Path(ct_sets) / "ct-polar.jsonl")
    rows = []
    for pid, (sym, X, q) in sorted(qm.items()):
        mod, c = ev.alpha(sym, X)
        free = float(sum(ev.tab.alpha[ev.tab.species_index(torch.tensor([ZMAP[a]]))].item() for a in sym))
        cl = [k for k, a in enumerate(sym) if a == "Cl"]
        kind = pid.split("_")[0]
        # the pair / trimer axes in the molecular frame: along the pair (or the Na...Na line), perpendicular
        for ax in q:
            d = {"id": pid, "kind": kind, "axis": ax, "free": free, "c_alpha_Cl": float(c[cl, 0].mean()),
                 "trained": bool(ev.alpha_trained and pid not in ev.alpha_held_out
                                 and not (getattr(ev, "holdout_pairs", False) and pid.startswith("pair_")))}
            if kind == "pair":
                d["pair"], d["r"] = pid.split("_")[1], float(pid.split("_r")[1])
                u = X[1] - X[0]
                d["orient"] = "par" if abs(u["xyz".index(ax)]) > 0.5 * np.linalg.norm(u) else "perp"
            elif kind == "shell":
                d["shape"] = pid.split("_")[1]
                d["n_na"] = SHAPE_N[d["shape"]]
            else:
                d["angle"] = int(pid.split("_a")[1])
            d.update({f"qm_{k}": v for k, v in q[ax].items()})
            d.update({f"m_{k}": v for k, v in mod[ax].items()})
            rows.append(d)
    return pd.DataFrame(rows)


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"full": "k", "mi": "tab:blue", "ct": "tab:red"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))

    def series(ax, x, g, title, xlabel, iso=True):
        g = g.sort_values(x)
        X = g[x].to_numpy()
        for k, lab in (("full", "total"), ("mi", "no CT"), ("ct", "CT")):
            mk = "m_total" if k == "full" else ("m_pol" if k == "mi" else "m_ct")
            ax.plot(X, g[f"qm_{k}"], "o", color=C[k], mfc="none", ms=6, label=f"ALMO {lab}")
            if "trained" in g and g["trained"].any():
                t = g["trained"] > 0.5
                ax.plot(X[t.to_numpy()], g[f"qm_{k}"][t], "o", color=C[k], alpha=0.3, ms=6, mec="none",
                        label="(filled: in the α loss)" if k == "full" else None)
            ax.plot(X, g[mk], "-", color=C[k], lw=1.8, label=f"model {lab}")
        if (g["m_ct_post"] - g["m_ct"]).abs().max() > 1e-6:
            ax.plot(X, g["m_ct_post"], "--", color=C["ct"], lw=1.0, label="model CT, no feedback")
        ax.plot(X, g["m_ct_0"], ":", color=C["ct"], lw=1.2, label="model CT, no coupling response (WKB / λ=0)")
        if iso:
            ax.axhline(g["free"].iloc[0], color="0.6", ls=":", lw=1, label="free ions")
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\alpha$ (bohr$^3$)")

    nacl = df[(df.kind == "pair") & (df.pair == "Na-Cl")]
    # average the two perpendicular axes
    par = nacl[nacl.orient == "par"]
    perp = nacl[nacl.orient == "perp"].groupby("r", as_index=False).mean(numeric_only=True)
    series(axs[0, 0], "r", par, r"(a) Na$^+$Cl$^-$ pair, $\alpha_\parallel$", "R (Å)")
    series(axs[0, 1], "r", perp, r"(b) Na$^+$Cl$^-$ pair, $\alpha_\perp$", "R (Å)")
    clcl = df[(df.kind == "pair") & (df.pair == "Cl-Cl")]
    if len(clcl):
        par2 = clcl[clcl.orient == "par"]
        series(axs[0, 2], "r", par2, r"(c) Cl$^-$Cl$^-$ pair, $\alpha_\parallel$", "R (Å)")
    # shells and trimers: the isotropic average
    sh = df[df.kind == "shell"].groupby(["shape", "n_na"], as_index=False).mean(numeric_only=True)
    sh = sh.sort_values(["n_na", "shape"]).reset_index(drop=True)
    sh["pos"] = np.arange(len(sh))
    series(axs[1, 0], "pos", sh, r"(d) Cl$^-$(Na$^+$)$_n$ shells, R = 2.8 Å, $\bar\alpha$", "")
    axs[1, 0].set_xticks(sh["pos"], [f"{s}" for s in sh["shape"]])
    tr = df[df.kind == "trimer"].groupby("angle", as_index=False).mean(numeric_only=True)
    series(axs[1, 1], "angle", tr, r"(e) Na$^+$Cl$^-$Na$^+$, R = 2.8 Å, $\bar\alpha$", "angle (deg)")
    # parity over everything (per axis)
    ax = axs[1, 2]
    sel = df[~((df.kind == "pair") & (df.pair == "Na-Cl") & (df.r > 3.6))]
    for k, mk, lab in (("mi", "m_pol", "no CT"), ("ct", "m_ct", "CT"), ("full", "m_total", "total")):
        ax.plot(sel[f"qm_{k}"], sel[mk], "o", color=C[k], ms=4, alpha=0.7, label=lab)
    lim = [min(sel.qm_ct.min(), 0) - 2, max(sel.qm_full.max(), sel.m_total.max()) + 5]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"ALMO $\alpha$ (bohr$^3$)")
    ax.set_ylabel(r"model $\alpha$ (bohr$^3$)")
    ax.set_title(r"(f) all axes (Na–Cl pairs R ≤ 3.6 Å)")
    ax.legend(fontsize=8)
    for a in axs.flat[:5]:
        a.grid(alpha=0.25)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = frame(a.run, a.ct_sets)
    df.to_csv(Path(a.run) / "polar.csv", index=False)
    plot(df, a.out or Path(a.run) / "ionct_polar.png")
    cols = ["qm_full", "m_total", "qm_mi", "qm_pol", "m_pol", "qm_ct", "m_ct", "m_ct_post", "m_ct_0"]
    print(df.groupby(["kind"])[cols].mean().round(2))
    print(df[df.kind == "pair"].groupby(["pair", "r", "orient"])[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
