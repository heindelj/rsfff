"""Figure of a trained ion CT model against ALMO-EDA: python scripts/ionct_plots.py RUN [--compare RUN_ECT_ONLY]
[--baseline ionct_ct_check.json] [--out RUN/ionct_results.png]. Reads predictions.csv, channels.csv, atoms.csv
(scripts/ionct_eval.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"test": "#e34948", "val": "#eda100", "train": "#9a9a9a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    pr = pd.read_csv(run / "predictions.csv")
    ch = pd.read_csv(run / "channels.csv")
    at = pd.read_csv(run / "atoms.csv")
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.8), gridspec_kw={"hspace": 0.42, "wspace": 0.32})

    a0 = ax[0, 0]
    for sp in ("train", "val", "test"):
        x = pr[(pr.split == sp) & pr.ct_mask]
        a0.scatter(-x.ct_ref, -x.ct, s=4 if sp == "train" else 7, color=C[sp], alpha=0.5 if sp == "train" else 0.8,
                   lw=0, label=f"{sp} ({len(x)})")
        f = x[x.full & (x.n >= 4)]
        a0.scatter(-f.ct_ref, -f.ct, s=30, marker="*", color=C[sp], edgecolor="k", lw=0.4)
    if a.baseline:
        b = pd.DataFrame(json.load(open(a.baseline)))
        a0.scatter(-b.ct, -b["smooth" if "smooth" in b else "one_sided"], s=10, marker="x", color="#2a78d6", lw=0.8, label="dimer-only model (clusters)")
    lim = 80
    a0.plot([0, lim], [0, lim], color="0.5", lw=0.7, ls=":")
    t = pr[(pr.split == "test") & pr.ct_mask]
    tf = t[t.full & (t.n >= 4)]
    a0.set(xlim=(0, lim), ylim=(0, lim), xlabel="ALMO $-E_\\mathrm{CT}$ / kJ mol$^{-1}$", ylabel="model $-E_\\mathrm{CT}$",
           title=f"(a) CT: test MAE {np.abs(t.ct - t.ct_ref).mean():.2f}; clusters (★) "
                 f"{np.abs(tf.ct - tf.ct_ref).mean():.2f} kJ/mol")
    a0.legend(fontsize=6.5, loc="upper left")

    a1 = ax[0, 1]
    for sp in ("train", "val", "test"):
        x = pr[pr.split == sp]
        a1.scatter(-x.pol_ref, -x.pol, s=4 if sp == "train" else 7, color=C[sp], alpha=0.5, lw=0)
        f = x[x.full & (x.n >= 4)]
        a1.scatter(-f.pol_ref, -f.pol, s=30, marker="*", color=C[sp], edgecolor="k", lw=0.4)
    lim = max(-pr.pol_ref.min(), -pr.pol.min()) * 1.05
    a1.plot([0, lim], [0, lim], color="0.5", lw=0.7, ls=":")
    t = pr[pr.split == "test"]
    tf = t[t.full & (t.n >= 4)]
    a1.set(xlabel="ALMO $-E_\\mathrm{pol}$ / kJ mol$^{-1}$", ylabel="model $-E_\\mathrm{pol}$",
           title=f"(b) POL: test MAE {np.abs(t.pol - t.pol_ref).mean():.2f}, clusters {np.abs(tf.pol - tf.pol_ref).mean():.1f}")

    a2 = ax[0, 2]
    types = ["Cl>Na", "Cl>Cl", "Na>Cl", "Na>Na"]
    test = ch[ch.split == "test"]
    ref = [-test[test.type == ty].E_cta.sum() for ty in types]
    mod = [-test[test.type == ty].E_model.sum() for ty in types]
    w = 0.27
    xs = np.arange(len(types))
    a2.bar(xs - w, ref, w, color="k", label="ALMO-CTA")
    a2.bar(xs, mod, w, color=C["test"], label="model (+ channel term)" if a.compare else "model")
    if a.compare:
        cc = pd.read_csv(Path(a.compare) / "channels.csv")
        cc = cc[cc.split == "test"]
        a2.bar(xs + w, [-cc[cc.type == ty].E_model.sum() for ty in types], w, color="#9a9a9a", label="model (E_CT only)")
    a2.set_xticks(xs, [t.replace(">", "→") for t in types])
    a2.set(ylabel="$-\\sum E$ over test clusters / kJ mol$^{-1}$", title="(c) channel types, test clusters")
    a2.legend(fontsize=6.5)

    a3 = ax[1, 0]
    f = at[at.full & (at.n >= 2)]
    for ion, col in (("Cl", "#1baf7a"), ("Na", "#6250d6")):
        y = f[f.ion == ion]
        a3.scatter(y.r_nn, y.c_alpha, s=6, color=col, alpha=0.6, lw=0, label=f"{ion}: $c_\\alpha$")
    a3.axhline(1, color="0.5", lw=0.7, ls=":")
    a3.set(xlabel="nearest-neighbour distance / Å", ylabel="confinement $\\alpha/\\alpha_\\mathrm{free}$",
           title="(d) learned confinement (whole clusters)", ylim=(0, 1.35))
    a3.legend(fontsize=6.5)

    a4 = ax[1, 1]
    for ty, col in (("Cl>Na", "#e34948"), ("Cl>Cl", "#2a78d6")):
        y = ch[ch.type == ty]
        a4.scatter(y.r, y.dov, s=5, color=col, alpha=0.5, lw=0, label=ty.replace(">", "→"))
    a4.set(xlabel="channel distance / Å", ylabel="learned $\\Delta^\\mathrm{ov}$ / $E_h$", title="(e) learned overlap gap")
    a4.legend(fontsize=6.5)

    a5 = ax[1, 2]
    y = ch[ch.type == "Cl>Na"]
    for k in sorted(y.nA_other.unique()):
        z = y[y.nA_other == k]
        a5.boxplot(z.dov, positions=[k], widths=0.6, showfliers=False)
    a5.set(xlabel="acceptor Na$^+$: other neighbours within 3.2 Å", ylabel="learned $\\Delta^\\mathrm{ov}$ (Cl→Na) / $E_h$",
           title="(f) overlap gap vs acceptor coordination")
    for x_ in ax.ravel():
        x_.title.set_fontsize(8.5)
        x_.title.set_ha("left")
        x_.title.set_x(0)
    out = a.out or run / "ionct_results.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
