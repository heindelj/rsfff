"""The three-panel abstract figure written by ``scripts/hbond_pauli_classifier.py``.

Panel a  the distribution of the model's oxygen Pauli repulsion charge, split by the
         Kumar/Schmidt/Skinner ``r``-``psi`` hydrogen-bond label, with the dividing line
         that best reproduces that label and how accurately it does so.
Panel b  monomer molecular polarizability, model vs Q-Chem, as sorted eigenvalues.
Panel c  predicted-vs-reference correlation for every ALMO-EDA channel the model fits.

Reads the ``.npz`` and writes ``.pdf`` + ``.png`` beside it. Nothing is recomputed here
except metrics that follow from arrays already stored, so the figure and the numbers in it
can never drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# Categorical slots 1 and 2 of the validated default palette; the pair clears every
# all-pairs gate on a light surface (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1).
BONDED = "#2a78d6"      # blue
FREE = "#eb6834"        # orange
MODEL = "#2a78d6"
REFERENCE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"

#: The blue ramp, ordinal steps 250..700. Cluster size is an *ordered* quantity, so it gets
#: a one-hue ramp rather than categorical hues; the light end stops at step 250, the lightest
#: step that still clears 2:1 against the light chart surface.
SIZE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]

EDA_LABEL = {
    "elst": "frozen electrostatics",
    "pauli": "Pauli repulsion",
    "disp": "dispersion",
    "induction": "induction (pol + CT)",
}


def best_threshold(score, label):
    """``(accuracy, cut)`` for the best "positive when ``score > cut``" rule."""
    order = np.argsort(score, kind="stable")
    lab = np.asarray(label, bool)[order]
    n, n_true = lab.size, int(lab.sum())
    fn = np.concatenate(([0], np.cumsum(lab)))
    tn = np.arange(n + 1) - fn
    acc = ((n_true - fn) + tn) / n
    k = int(np.argmax(acc))
    s = np.sort(score)
    cut = s[0] - 1.0 if k == 0 else (s[-1] + 1.0 if k == n else 0.5 * (s[k - 1] + s[k]))
    return float(acc[k]), float(cut)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def strip_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_a(ax, d):
    """Per-acceptor-oxygen ``q_O`` histograms, the dividing line, and the accuracy box.

    The histogram is over **acceptor oxygens**, not over ``O...H`` pairs, because ``q_O``
    is one number per atom: every pair an oxygen takes part in carries the same charge, so
    a pair-level histogram of it plots the same value several times and blurs the very
    separation the panel is about. The pair-level numbers are still quoted in the box --
    including ``r`` alone, which is the null hypothesis, because the ``r``-``psi`` occupancy
    map decays over 0.343 A and so the label it assigns is very nearly a distance cutoff.
    """
    n_acc, q_o = d["group_n_accepted"], d["group_q_o"]
    accepts = n_acc >= 1
    # `best_threshold` is written for "positive above the cut"; accepting oxygens sit
    # *below* in q_O, so the score is negated and the returned cut negated back.
    acc_atom, cut_atom = best_threshold(-q_o, accepts)
    cut_atom = -cut_atom

    lo, hi = np.percentile(q_o, [0.05, 99.95])
    bins = np.linspace(lo, hi, 90)
    peak = 0.0
    for mask, color, label in (
        (accepts, BONDED, f"accepts $\\geq$1 H-bond  ($n$={int(accepts.sum()):,})"),
        (~accepts, FREE, f"accepts none  ($n$={int((~accepts).sum()):,})"),
    ):
        counts, _, _ = ax.hist(q_o[mask], bins=bins, color=color, alpha=0.55, lw=0.0,
                               label=label)
        ax.hist(q_o[mask], bins=bins, histtype="step", color=color, lw=1.2)
        peak = max(peak, float(counts.max()))

    ax.axvline(cut_atom, color=INK, lw=1.1, ls=(0, (4, 2.5)), zorder=4)
    ax.annotate(
        f"dividing line\n$q_{{\\rm O}}$ = {cut_atom:.3f} $e$",
        xy=(cut_atom, 0.42), xycoords=("data", "axes fraction"),
        xytext=(-6, 0), textcoords="offset points",
        ha="right", va="center", fontsize=7.5, color=INK,
    )

    ax.set_xlabel(r"oxygen Pauli repulsion charge  $q_{\rm O}$  ($e$)")
    ax.set_ylabel("acceptor oxygens")
    ax.set_xlim(lo, hi)
    # Headroom for the legend and the metrics box, both of which live in the empty band
    # above the taller histogram rather than on top of it. Keyed off the tallest bin so the
    # reserved band stays the same fraction of the axes whatever the data does.
    ax.set_ylim(0.0, 1.52 * peak)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.012, 1.005), handlelength=1.0,
              labelcolor=INK, borderaxespad=0.0, handletextpad=0.5, fontsize=7.0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip_spines(ax)

    text = (
        f"agreement with $r$-$\\psi$\n"
        f"atoms  $q_{{\\rm O}}$ cut          {100 * acc_atom:.1f}%\n"
        f"pairs  ($r$, $q_{{\\rm O}}$) line   {100 * float(d['acc_2d']):.1f}%\n"
        f"pairs  $r$ alone (null)  {100 * float(d['acc_r']):.1f}%\n"
        f"pairs  $q_{{\\rm O}}$ alone       {100 * float(d['acc_q']):.1f}%"
    )
    ax.text(0.988, 0.995, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, family="monospace", color=INK, linespacing=1.5, zorder=10,
            bbox={"boxstyle": "round,pad=0.38", "fc": "#fcfcfb", "ec": GRID, "lw": 0.7})
    return acc_atom, cut_atom


def panel_b(ax, d):
    """Model vs Q-Chem monomer polarizability, as the three sorted eigenvalues plus ``<a>``.

    Eigenvalues rather than tensor components: the reference monomers sit in arbitrary
    orientations, so ``alpha_xx`` averaged over them measures how the geometries happen to be
    rotated, not how the molecule responds. The eigenvalues are rigid-motion invariant, and
    for water they are the three physically distinct directions -- in-plane perpendicular to
    the bisector, along the bisector, and out of plane.

    Error bars are the standard deviation over the thermal ensemble, not an uncertainty:
    they show that the *spread* the model produces matches the spread in the label, which a
    comparison of means alone would hide.
    """
    pred, ref = d["alpha_eig_pred"], d["alpha_eig_ref"]
    pred = np.column_stack([pred, pred.mean(axis=1)])
    ref = np.column_stack([ref, ref.mean(axis=1)])
    names = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\bar{\alpha}$"]

    x = np.arange(pred.shape[1], dtype=float)
    width = 0.36
    for offset, values, color, label in (
        (-width / 2, pred, MODEL, "rsfff"),
        (+width / 2, ref, REFERENCE, "wB97M-V/def2-TZVPD"),
    ):
        ax.bar(x + offset, values.mean(axis=0), width, color=color, lw=0, zorder=3,
               label=label, yerr=values.std(axis=0),
               error_kw={"lw": 0.8, "ecolor": INK_2, "capsize": 2.0, "capthick": 0.8,
                         "zorder": 4})

    mae = np.abs(pred - ref).mean(axis=0)
    top = (np.maximum(pred.mean(axis=0), ref.mean(axis=0))
           + np.maximum(pred.std(axis=0), ref.std(axis=0)))
    for xi, value, hi in zip(x, mae, top):
        ax.annotate(f"MAE {value:.2f}", (xi, hi), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", fontsize=6.6, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"polarizability (a.u., $a_0^3$)")
    ax.set_ylim(0.0, 1.42 * float(top.max()))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.012, 1.005), handlelength=1.0,
              labelcolor=INK, borderaxespad=0.0, handletextpad=0.5, fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip_spines(ax)
    ax.text(0.99, 0.995, f"{pred.shape[0]:,} thermal\nH$_2$O monomers",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2,
            linespacing=1.4)
    return mae


def panel_c(axes, d, fig):
    """A parity plot per EDA channel, points colored by cluster size.

    No shared axis: the four channels span wildly different magnitudes (electrostatics
    reaches -1900 kJ/mol where dispersion reaches -450), and forcing a common scale would
    compress three of the four into a corner. Each panel is square with its own limits and
    its own ``y = x``, which is what makes the residual readable.
    """
    terms = [str(t) for t in d["eda_terms"]]
    n_frag = d["e_n_frag"]
    sizes = np.unique(n_frag)
    cmap = ListedColormap(np.array(SIZE_RAMP)[
        np.linspace(0, len(SIZE_RAMP) - 1, len(sizes)).round().astype(int)
    ])
    edges = np.concatenate(([sizes[0] - 0.5], 0.5 * (sizes[:-1] + sizes[1:]),
                            [sizes[-1] + 0.5]))
    norm = BoundaryNorm(edges, cmap.N)

    # Big clusters are rare and are the interesting end of the range, so draw them last
    # rather than letting 8000 dimers bury them.
    order = np.argsort(n_frag, kind="stable")

    handle = None
    for ax, term in zip(axes, terms):
        pred, ref = d[f"e_{term}_pred"], d[f"e_{term}_ref"]
        lo = min(pred.min(), ref.min())
        hi = max(pred.max(), ref.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, lw=0.8,
                ls=(0, (4, 3)), zorder=2)
        handle = ax.scatter(ref[order], pred[order], c=n_frag[order], cmap=cmap, norm=norm,
                            s=5.5, lw=0, alpha=0.85, zorder=3, rasterized=True)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")

        mae = float(np.abs(pred - ref).mean())
        r2 = 1.0 - float(((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())
        ax.text(0.055, 0.945, f"MAE {mae:.2f}\n$R^2$ {r2:.4f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.3, color=INK, linespacing=1.4, zorder=6,
                bbox={"boxstyle": "round,pad=0.25", "fc": "#fcfcfb", "ec": GRID,
                      "lw": 0.6})
        ax.set_title(EDA_LABEL.get(term, term), fontsize=7.6, color=INK, pad=3)
        ax.locator_params(nbins=4)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        strip_spines(ax)

    for ax in axes:
        ax.set_xlabel("wB97M-V ALMO-EDA (kJ/mol)", fontsize=7)
    axes[0].set_ylabel("rsfff (kJ/mol)", fontsize=7)

    bar = fig.colorbar(handle, ax=list(axes), fraction=0.020, pad=0.022, aspect=14)
    bar.set_label("water molecules in cluster", fontsize=7.5, color=INK)
    bar.set_ticks([2, 5, 10, 15, 20, 23])
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="notebooks/figures/hbond_pauli_manybody.npz")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    style()

    fig = plt.figure(figsize=(7.7, 4.7))
    # The parity axes are square and width-limited, so row 2's height only has to cover the
    # square plus its title and xlabel; any more becomes dead space between the rows.
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.63), hspace=0.44,
                             left=0.082, right=0.945, top=0.945, bottom=0.10)
    top = outer[0].subgridspec(1, 2, width_ratios=(1.32, 1.0), wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    # A 1x4 strip, not 2x2: the parity axes are forced square (equal x and y limits are what
    # puts `y = x` at 45 degrees and makes the residual readable), and in a 2x2 that squareness
    # is set by the row height, leaving the columns half-empty.
    grid = outer[1].subgridspec(1, 4, wspace=0.42)
    ax_c = [fig.add_subplot(grid[0, j]) for j in range(4)]

    acc_atom, cut_atom = panel_a(ax_a, d)
    mae_alpha = panel_b(ax_b, d)
    panel_c(ax_c, d, fig)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c[0], "c")):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=10, color=INK, pad=6)

    out = Path(args.npz).with_suffix("")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    print(f"per-oxygen q_O cut {cut_atom:.4f} e -> {100 * acc_atom:.2f}% agreement")
    print("polarizability MAE (a.u.): "
          + "  ".join(f"{n} {v:.3f}" for n, v in
                      zip(("a1", "a2", "a3", "iso"), mae_alpha)))
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
