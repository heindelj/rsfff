"""The two-panel abstract figure written by ``scripts/hbond_pauli_classifier.py``.

Panel A  the distribution of the model's oxygen Pauli repulsion charge, split by the
         Kumar/Schmidt/Skinner ``r``-``psi`` hydrogen-bond label, with the dividing line
         that best reproduces that label and how accurately it does so.
Panel B  the 2-body / beyond-2-body split of induction, dispersion and exchange in a
         23-molecule cluster.

Reads the ``.npz`` and writes ``.pdf`` + ``.png`` beside it. Nothing is recomputed here
except the classifier metrics that follow from arrays already stored, so the figure and the
numbers in it can never drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rsfff.ff.units import KJMOL_PER_HARTREE

# Categorical slots 1 and 2 of the validated default palette; the pair clears every
# all-pairs gate on a light surface (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1).
BONDED = "#2a78d6"      # blue
FREE = "#eb6834"        # orange
TWO_BODY = "#2a78d6"
MANY_BODY = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"

CHANNEL_LABEL = {
    "induction": "Induction\n(pol + CT)",
    "disp": "Dispersion",
    "pauli": "Exchange\n(Pauli rep.)",
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


def panel_a(ax, d) -> None:
    """Per-acceptor-oxygen ``q_O`` histograms, the dividing line, and the accuracy box.

    The histogram is over **acceptor oxygens**, not over ``O...H`` pairs, because ``q_O``
    is one number per atom: every pair an oxygen takes part in carries the same charge, so
    a pair-level histogram of it plots the same value several times and blurs the very
    separation the panel is about. The pair-level classification the caption quotes is in
    the inset, where the second coordinate ``r`` is available to resolve *which* hydrogen.
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
        xy=(cut_atom, 0.295), xycoords=("data", "axes fraction"),
        xytext=(-6, 0), textcoords="offset points",
        ha="right", va="center", fontsize=7, color=INK,
    )

    ax.set_xlabel(r"oxygen Pauli repulsion charge  $q_{\rm O}$  ($e$)")
    ax.set_ylabel("acceptor oxygens")
    ax.set_xlim(lo, hi)
    # Headroom for the legend, the metrics box and the inset, all of which live in the
    # empty band above the taller histogram rather than on top of it. Keyed off the tallest
    # bin so the reserved band stays the same fraction of the axes whatever the data does.
    ax.set_ylim(0.0, 1.78 * peak)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.012, 1.005), handlelength=1.0,
              labelcolor=INK, borderaxespad=0.0, fontsize=7.0, handletextpad=0.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    text = (
        f"agreement with $r$-$\\psi$\n"
        f"atoms  $q_{{\\rm O}}$ cut          {100 * acc_atom:.1f}%\n"
        f"pairs  ($r$, $q_{{\\rm O}}$) line   {100 * float(d['acc_2d']):.1f}%\n"
        f"pairs  $r$ alone (null)  {100 * float(d['acc_r']):.1f}%\n"
        f"pairs  $q_{{\\rm O}}$ alone       {100 * float(d['acc_q']):.1f}%"
    )
    ax.text(0.988, 0.875, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, family="monospace", color=INK, linespacing=1.5, zorder=10,
            bbox={"boxstyle": "round,pad=0.38", "fc": "#fcfcfb", "ec": GRID, "lw": 0.7})
    return acc_atom, cut_atom


def panel_a_inset(ax, d) -> None:
    """The pair-level ``(r, q_O)`` plane with the fitted straight dividing line.

    The boundary comes out almost vertical, and that is the honest result the inset exists
    to show: the ``r``-``psi`` occupancy map falls off with a 0.343 A decay length, so the
    label it assigns is very nearly a distance cutoff and a distance cutoff alone already
    reproduces it. ``q_O`` tilts the line but does not move the accuracy.
    """
    r, q, lab = d["r"], d["q_o"], d["hb_occ"].astype(bool)
    keep = np.linspace(0, r.size - 1, min(r.size, 6000)).round().astype(int)
    for mask, color in ((lab[keep], BONDED), (~lab[keep], FREE)):
        ax.scatter(r[keep][mask], q[keep][mask], s=1.6, lw=0, alpha=0.30,
                   color=color, rasterized=True)

    w_r, w_q, cut, r_mean, r_std, q_mean, q_std = d["line_direction"]
    rr = np.linspace(r.min(), r.max(), 200)
    # w_r*(r-rm)/rs + w_q*(q-qm)/qs = cut  ->  solve for q
    qq = q_mean + q_std * (cut - w_r * (rr - r_mean) / r_std) / w_q
    inside = (qq > np.percentile(q, 0.2)) & (qq < np.percentile(q, 99.8))
    ax.plot(rr[inside], qq[inside], color=INK, lw=1.1, ls=(0, (4, 2.5)))

    ax.set_xlabel(r"$r$ (H$\cdots$O, $\AA$)", fontsize=6.6, labelpad=1.5)
    ax.set_ylabel(r"$q_{\rm O}$ ($e$)", fontsize=6.6, labelpad=1.5)
    ax.set_ylim(*np.percentile(q, [0.2, 99.8]))
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.set_title("all O$\\cdots$H pairs", fontsize=6.6, color=INK_2, pad=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_b(ax, d) -> None:
    """Grouped 2-body / beyond-2-body bars, in kJ/mol, one group per EDA channel.

    Grouped and not stacked on purpose: induction and dispersion are attractive while
    exchange is repulsive, and dispersion's many-body part has the *opposite* sign to its
    2-body part. Stacked segments across a zero line stop meaning "parts of a whole", so
    they would read as smaller than they are.
    """
    channels = [str(c) for c in d["mbe_channels"]]
    total = d["mbe_total"] * KJMOL_PER_HARTREE
    two = d["mbe_two_body"] * KJMOL_PER_HARTREE
    many = total - two

    x = np.arange(len(channels), dtype=float)
    width = 0.36
    ax.bar(x - width / 2, two, width, color=TWO_BODY, lw=0, label="2-body  $E^{(2)}$",
           zorder=3)
    ax.bar(x + width / 2, many, width, color=MANY_BODY, lw=0,
           label="beyond 2-body  $E - E^{(2)}$", zorder=3)

    for xi, value in zip(x - width / 2, two):
        off = 4 if value >= 0 else -4
        ax.annotate(f"{value:,.0f}", (xi, value), textcoords="offset points",
                    xytext=(0, off), ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=7, color=INK)
    for xi, value, tot in zip(x + width / 2, many, total):
        off = 4 if value >= 0 else -4
        # Percent of the *magnitude* of the total. A signed ratio reads as negative for
        # dispersion, whose many-body part opposes its 2-body part -- which is a real
        # feature, but the bar already shows it, and the signed number invites reading
        # "-6%" as a small many-body term rather than one that cancels 6% of the pair sum.
        ax.annotate(f"{value:,.0f}  ({100 * abs(value / tot):.0f}%)",
                    (xi, value), textcoords="offset points", xytext=(0, off),
                    ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=7, color=INK)

    ax.axhline(0.0, color=INK_2, lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABEL.get(c, c) for c in channels])
    ax.set_ylabel("interaction energy (kJ/mol)")
    ax.set_ylim(1.45 * min(two.min(), many.min(), 0.0),
                1.32 * max(two.max(), many.max(), 0.0))
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.855), handlelength=1.1,
              labelcolor=INK, borderaxespad=0.0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.02, 0.985,
            f"(H$_2$O)$_{{{int(d['mbe_n_fragments'])}}}$: "
            f"{int(d['mbe_n_dimers'])} dimers vs. the full cluster\n"
            r"labels: kJ/mol, then $|E-E^{(2)}|\,/\,|E|$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=INK_2,
            linespacing=1.4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="notebooks/figures/hbond_pauli_manybody.npz")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    style()

    fig = plt.figure(figsize=(7.4, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.18, 1.0), wspace=0.30,
                          left=0.075, right=0.985, top=0.90, bottom=0.175)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    acc_atom, cut_atom = panel_a(ax_a, d)
    inset = ax_a.inset_axes([0.585, 0.325, 0.395, 0.255])
    panel_a_inset(inset, d)
    panel_b(ax_b, d)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=10, color=INK, pad=6)

    out = Path(args.npz).with_suffix("")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    print(f"per-oxygen q_O cut {cut_atom:.4f} e -> {100 * acc_atom:.2f}% agreement")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
