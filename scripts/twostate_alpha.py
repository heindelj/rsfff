"""CT polarizability of the two-state channel CT: the NaCl dimer along R for a trained model, and the universal
two-state curve alpha |e| / R^2 against Delta / |e| (H2+ at Delta = 0).

    python scripts/twostate_alpha.py RUN/model.pt [--out fig.png] [--outer 30]

Importable: ``dimer_scan`` and ``universal`` return the arrays, ``plot`` draws them. Run from research/rsfff."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rsfff.ff.complete import batched_precompute
from rsfff.ff.complete import twostate as TS
from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.train.train_ionct import build_model, radial_table

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "ink": "#0b0b0b",
     "ink2": "#52514e", "grid": "#e4e3df", "shade": "#f0efec"}


def load(path):
    ck = torch.load(path, weights_only=False)
    args = ck["args"]
    tab, chan = IonTable.load(args["params"]), CT.ChannelTable.load(args["params"])
    model = build_model(args, tab, chan, ck["n_feat"], None, ck.get("feature_names"))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, tab, chan, radial_table(tab, args["params"], args.get("cache_dir"))


def isolated_alpha(e, D, R_bohr):
    """alpha_par = R^2 / k of one isolated channel (NaN where Delta <= e: no two-state minimum with energy e)."""
    e, D = np.asarray(e, float), np.asarray(D, float)
    ok = D > e
    W2 = np.where(ok, e * (e - D), np.nan)
    return R_bohr ** 2 * 2 * W2 / (D - 2 * e) ** 3


def dimer_scan(model, tab, table, chan, Rs, outer=(1, 30), dF=5e-4):
    """Per R (A): the Cl- -> Na+ channel's raw and floored gap, e_c, and alpha_CT,par (bohr^3) of the model by finite
    field along the bond (both channels, coupled to the induced moments) for each number of outer steps, plus the
    isolated two-state alpha of the two channels at the floored and at the raw gap."""
    out = {k: [] for k in ("R", "D_raw", "D_floor", "D_table", "e", "a_iso_floor", "a_iso_raw", "a_iso_table")}
    for n in outer:
        out[f"a_model_{n}"] = []
    cl = tab.symbols.index("Cl")
    for R in Rs:
        X = torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=torch.float64)
        bt = batched_precompute(X, torch.tensor([11, 17]), torch.zeros(2, dtype=torch.long), tab, table, chan,
                                inv_form=model.inv_form)
        sp = bt["species"]
        with torch.no_grad():
            if getattr(model, "env", None) is not None:
                model.env_prepare(bt, sp)
            I = model.scaled(bt["feats"])
            *_, e_c, Dl, Draw = model.twostate_operator(bt, sp, model.ct_charges_from(I, sp), 9 if model.breath else 8)
        c = int((sp[bt["ch_index"][0]] == cl).nonzero()[0])
        i, j = bt["ch_index"]
        with torch.no_grad():     # the same gaps with the table I, A (no learned shifts)
            Dtab = Draw - model.ts_dI[sp[i]] + model.ts_dA[sp[j]]
        rb = R / K.BOHR_ANG
        out["R"].append(R)
        out["D_raw"].append(float(Draw[c]))
        out["D_floor"].append(float(Dl[c]))
        out["D_table"].append(float(Dtab[c]))
        out["a_iso_table"].append(float(np.sum(isolated_alpha(e_c.numpy(), Dtab.numpy(), rb))))
        out["e"].append(float(e_c[c]))
        out["a_iso_floor"].append(float(np.nansum(isolated_alpha(e_c.numpy(), Dl.numpy(), rb))))
        out["a_iso_raw"].append(float(np.sum(isolated_alpha(e_c.numpy(), Draw.numpy(), rb))))
        for n in outer:
            model.ct_outer, model.ct_outer_tol = n, 1e-13
            E = []
            for f in (0.0, dF, -dF):
                with torch.no_grad():
                    E.append(float(model(bt, field=torch.tensor([[f, 0.0, 0.0]], dtype=torch.float64))["e_ct"][0]))
            out[f"a_model_{n}"].append(-(E[1] + E[2] - 2 * E[0]) / dF ** 2)
    return {k: np.array(v) for k, v in out.items()}


def universal(x):
    """alpha |e| / R^2 of one two-state channel against x = Delta / |e| (exact, defined for x > -1) and the
    weak-coupling (perturbative) limit 2 / x^2."""
    x = np.asarray(x, float)
    exact = np.where(x > -1, 2 * (1 + x) / (x + 2) ** 3, np.nan)
    with np.errstate(divide="ignore"):
        pert = np.where(x > 0, 2 / x ** 2, np.nan)
    return exact, pert


def _style(ax):
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["ink2"])
    ax.tick_params(colors=C["ink2"], labelsize=9)


def plot(scan, path, title="NaCl dimer, two-state CT (linear response model)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    R = scan["R"]
    # (a) alpha_CT along R
    ax = axs[0]
    ax.plot(R, scan["a_model_1"], color=C["blue"], lw=2, label="model, linear response")
    if "a_model_30" in scan:
        ax.plot(R, scan["a_model_30"], color=C["orange"], lw=2, ls="--", label="model, converged")
    ax.plot(R, scan["a_iso_floor"], color=C["aqua"], lw=2, ls=":", label="isolated channels, floored gap")
    ax.plot(R, scan["a_iso_raw"], color=C["ink2"], lw=1.5, ls="-.", label="isolated channels, raw learned gap")
    ax.plot(R, scan["a_iso_table"], color=C["yellow"], lw=2, label="isolated channels, table I, A gap")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 2e4)
    ax.set_xlabel("R (Å)", color=C["ink"])
    ax.set_ylabel(r"$\alpha_{\rm CT,\parallel}$ (bohr$^3$)", color=C["ink"])
    ax.set_title("(a) CT polarizability along the bond", fontsize=10, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    _style(ax)
    # (b) gap and direct energy along R (two panels' worth of scale: gap only, e marked as the Delta = e boundary)
    ax = axs[1]
    ax.plot(R, scan["D_raw"], color=C["blue"], lw=2, label=r"$\Delta$ raw (learned I, A)")
    ax.plot(R, scan["D_floor"], color=C["aqua"], lw=2, ls=":", label=r"$\Delta$ floored (0.03 Eh)")
    ax.plot(R, scan["D_table"], color=C["yellow"], lw=2, label=r"$\Delta$ with table I, A")
    ax.plot(R, scan["e"], color=C["orange"], lw=2, ls="--", label=r"$e_c$ (Cl$\to$Na); no solution below")
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("R (Å)", color=C["ink"])
    ax.set_ylabel("Hartree", color=C["ink"])
    ax.set_title(r"(b) Cl$^-\to$Na$^+$ gap and direct energy", fontsize=10, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    # (c) universal two-state curve
    ax = axs[2]
    x = np.linspace(-1.0, 8, 2000)
    exact, pert = universal(x)
    ax.axvspan(-1.6, -1.0, color=C["shade"], lw=0)
    ax.text(-1.55, 0.49, r"$\Delta < e$:" + "\nno\nminimum\nwith\nenergy $e$", fontsize=7.5, color=C["ink2"], va="top")
    ax.plot(x, exact, color=C["blue"], lw=2, label="two-state (exact curvature)")
    ax.plot(x, pert, color=C["orange"], lw=2, ls="--", label=r"weak coupling $2|e|R^2/\Delta^2$")
    ax.plot([0], [0.25], "o", ms=8, color=C["ink"], mec="white", mew=1.5, zorder=5)
    ax.annotate(r"H$_2^+$: $\Delta = 0$, $\alpha = R^2/4W$", (0, 0.25), (1.0, 0.27), fontsize=8, color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.8))
    ax.plot([-0.5], [0.5 / 1.5 ** 3 * 2], "o", ms=6, color=C["blue"], mec="white", mew=1.2, zorder=5)
    ax.annotate(r"max $0.30\,R^2/|e|$ at $\Delta = e/2$", (-0.5, 0.296), (0.2, 0.36), fontsize=8, color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.8))
    ax.set_xlim(-1.6, 8)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"$\Delta / |e|$", color=C["ink"])
    ax.set_ylabel(r"$\alpha_\parallel\,|e| / R^2$", color=C["ink"])
    ax.set_title("(c) one channel: finite at zero gap", fontsize=10, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    fig.suptitle(title, fontsize=11, color=C["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default="twostate_alpha.png")
    ap.add_argument("--outer", type=int, default=30)
    a = ap.parse_args()
    model, tab, chan, table = load(a.model)
    Rs = np.round(np.arange(2.2, 6.01, 0.1), 2)
    scan = dimer_scan(model, tab, table, chan, Rs, outer=(1, a.outer))
    if a.outer != 30:
        scan["a_model_30"] = scan.pop(f"a_model_{a.outer}")
    plot(scan, a.out)
    np.savez(a.out.rsplit(".", 1)[0] + ".npz", **scan)
    for k in range(0, len(Rs), 4):
        print(f"R {Rs[k]:.1f}: D_raw {scan['D_raw'][k]:+.4f} D_floor {scan['D_floor'][k]:.4f} e {scan['e'][k]:+.5f} "
              f"alpha LR {scan['a_model_1'][k]:.2f} conv {scan['a_model_30'][k]:.2f} iso_floor {scan['a_iso_floor'][k]:.2f} "
              f"iso_raw {scan['a_iso_raw'][k]:.2f}")


if __name__ == "__main__":
    main()
