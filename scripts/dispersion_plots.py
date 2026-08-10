"""Figures for the trained dispersion model.

Palette follows the project data-viz conventions: a single-hue blue ramp for continuous
magnitude (how much of the energy the network supplied), fixed categorical slots for
identity (element, model variant), and a blue/red diverging pair for signed quantities.
Text stays in ink colors so identity is never carried by colored type.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- palette -------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order, never cycled
#: single-hue sequential ramp (steps 100 -> 700), for continuous magnitude
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("rsfff_blue", SEQ_STEPS)
DIVERGING = LinearSegmentedColormap.from_list(
    "rsfff_div", ["#184f95", "#6da7ec", "#f0efec", "#e88a89", "#c22a2a"]
)

CLUSTER_LABEL = {"w2": "dimer", "w3": "trimer", "w4": "tetramer", "w5": "pentamer"}


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "font.size": 10,
    })


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _unit_line(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)


# --- figures -------------------------------------------------------------------

def correlation_panels(preds: dict, path=None, title=None):
    """Predicted vs reference dispersion (top) and the residual (bottom), per cluster size.

    Points are colored by the fraction of the pair energy the correction head supplied.
    The residual row exists because the correlation alone is too tight to read -- at
    R^2 > 0.999 every point sits on the diagonal and the parity plot shows nothing about
    where the model is actually working.
    """
    tags = list(preds)
    # The correction contributes ~1% on average, so a fixed 0..1 (or even 0..0.25) scale
    # renders every point at the palest step. Clip to the bulk of the observed range.
    vmax = float(np.percentile(
        np.concatenate([p.corr_fraction for p in preds.values()]), 98
    ))
    fig, axes = plt.subplots(2, len(tags), figsize=(3.1 * len(tags), 6.0),
                             gridspec_kw={"height_ratios": [1, 0.85]})

    for col, tag in enumerate(tags):
        p = preds[tag]
        lo = min(p.ref.min(), p.pred.min())
        hi = max(p.ref.max(), p.pred.max())
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        ax = axes[0, col]
        _unit_line(ax, lo, hi)
        ax.scatter(p.ref, p.pred, c=p.corr_fraction, cmap=SEQ, vmin=0.0, vmax=vmax,
                   s=6, lw=0.0, alpha=0.8, rasterized=True)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{CLUSTER_LABEL[tag]}  ({tag})")
        ax.text(0.04, 0.96,
                f"MAE {p.mae:.3f}\nRMSE {p.rmse:.3f}\n$R^2$ {p.r2:.4f}\nn = {len(p.ref)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK_2,
                linespacing=1.5)
        _despine(ax)

        ax = axes[1, col]
        ax.axhline(0.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        sc = ax.scatter(p.ref, p.error, c=p.corr_fraction, cmap=SEQ, vmin=0.0, vmax=vmax,
                        s=6, lw=0.0, alpha=0.8, rasterized=True)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("EDA dispersion  (kJ/mol)")
        _despine(ax)

    axes[0, 0].set_ylabel("model dispersion  (kJ/mol)")
    axes[1, 0].set_ylabel("model - EDA  (kJ/mol)")

    cb = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.012)
    cb.set_label(f"fraction from correction head  (clipped at {vmax:.1%})",
                 color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=INK_MUTED, labelcolor=INK_MUTED, labelsize=8)
    if title:
        fig.suptitle(title, y=1.0, fontsize=12, color=INK)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def parameter_panels(records, priors, path=None):
    """Learned effective C6 per element, and where the damping exponents ended up.

    Each element gets its own C6 panel because the two differ by more than an order of
    magnitude -- a shared axis would flatten hydrogen to a spike. The damping exponents get
    a dot plot rather than a histogram: with ``environment_b`` off they are per-species
    scalars, and a histogram of two constants is two spikes carrying one number each.
    """
    elements = [(1, "H"), (8, "O")]
    fig = plt.figure(figsize=(8.6, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.75], wspace=0.32)

    for col, (z, sym) in enumerate(elements):
        ax = fig.add_subplot(gs[0, col])
        values = records["c6"][records["Z"] == z]
        color = SERIES[col]
        ax.hist(values, bins=70, color=color, alpha=0.9, lw=0)
        prior = priors[z][0]
        top = ax.get_ylim()[1]
        ax.axvline(prior, color=INK, lw=1.2, ls=(0, (4, 3)))
        ax.annotate(f"prior\n{prior:.3g}", xy=(prior, top), xytext=(4, -2),
                    textcoords="offset points", va="top", fontsize=8, color=INK_2,
                    linespacing=1.4)
        ax.set_title(f"{sym}    effective $C_6$", loc="left")
        ax.set_xlabel(r"$C_6$  ($E_h\,a_0^6$)")
        # Keep the stats clear of the prior line: put them on whichever side it isn't.
        left = prior > values.mean()
        ax.text(0.03 if left else 0.97, 0.72,
                f"mean {values.mean():.3g}\n"
                f"{values.min():.3g} – {values.max():.3g}\n"
                f"spread {100 * values.std() / abs(values.mean()):.0f}%",
                transform=ax.transAxes, ha="left" if left else "right", va="top",
                fontsize=8.5, color=INK_2, linespacing=1.5)
        ax.set_yticks([])
        _despine(ax)
        ax.spines["left"].set_visible(False)
        if col == 0:
            ax.set_ylabel("atoms", color=INK_2)

    ax = fig.add_subplot(gs[0, 2])
    for i, (z, sym) in enumerate(elements):
        learned = records["b"][records["Z"] == z].mean()
        prior = priors[z][1]
        ax.plot([prior, learned], [i, i], color=GRID, lw=2.5, solid_capstyle="round",
                zorder=1)
        ax.scatter([prior], [i], s=52, facecolor=SURFACE, edgecolor=INK_MUTED, lw=1.4,
                   zorder=2)
        ax.scatter([learned], [i], s=52, color=SERIES[i], lw=0, zorder=3)
        ax.annotate(f"{learned:.3f}", xy=(learned, i), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(elements)))
    ax.set_yticklabels([sym for _, sym in elements])
    ax.set_ylim(len(elements) - 0.4, -0.6)   # H on top, matching the C6 panels' order
    ax.set_xlabel(r"damping exponent $b$  ($a_0^{-1}$)")
    ax.set_title("prior → learned", loc="left")
    ax.grid(axis="y", visible=False)
    _despine(ax)

    fig.suptitle("Effective dispersion parameters emitted per atom",
                 y=1.04, fontsize=12, color=INK)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def parameter_vs_environment(records, priors, path=None):
    """Effective C6 against the atom's closest inter-fragment contact.

    Cluster size turns out to be the wrong axis -- the medians barely move from dimer to
    pentamer. What the descriptors actually respond to is *local* engagement: a hydrogen
    donating an H-bond sits ~1.8 A from the acceptor oxygen, a free OH hydrogen sits well
    beyond 2.5 A, and that is the split behind the bimodal histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    for ax, (z, sym), color in zip(axes, [(1, "H"), (8, "O")], SERIES):
        m = records["Z"] == z
        x, y = records["contact"][m], records["c6"][m]
        finite = np.isfinite(x)
        x, y = x[finite], y[finite]
        ax.scatter(x, y, s=5, color=color, alpha=0.25, lw=0, rasterized=True)

        # running median, so the trend is readable through the point cloud
        edges = np.quantile(x, np.linspace(0, 1, 26))
        centers, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x >= lo) & (x < hi)
            if sel.sum() > 20:
                centers.append(0.5 * (lo + hi))
                med.append(np.median(y[sel]))
        ax.plot(centers, med, color=INK, lw=1.8, zorder=3)
        ax.axhline(priors[z][0], color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"prior {priors[z][0]:.3g}", xy=(ax.get_xlim()[1], priors[z][0]),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=INK_2)
        ax.set_title(f"{sym}", loc="left")
        ax.set_xlabel("closest contact in another molecule  (Å)")
        _despine(ax)
    axes[0].set_ylabel(r"effective $C_6$  ($E_h\,a_0^6$)")
    fig.suptitle("The coefficients track local H-bonding, not cluster size",
                 y=1.02, fontsize=12, color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def many_body_panels(mbe, path=None):
    """Where the model's dispersion sits in the many-body expansion."""
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.5))
    tags = ["w3", "w4", "w5"]

    # (a) mean magnitude by order, grouped per cluster size
    ax = axes[0]
    orders = [2, 3, 4, 5]
    width = 0.8 / len(tags)
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        vals = []
        for o in orders:
            key = {2: "two_body", 3: "e3", 4: "e4", 5: "e5"}[o]
            vals.append(np.abs(mbe[key][m]).mean())
        offs = (i - (len(tags) - 1) / 2) * width
        ax.bar([o + offs for o in orders], vals, width=width * 0.92,
               color=SERIES[i], label=CLUSTER_LABEL[tag], lw=0)
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("expansion order $k$")
    ax.set_ylabel(r"mean $|E^{(k)}|$  (kJ/mol)")
    ax.set_title("Magnitude by many-body order", loc="left")
    ax.legend(loc="upper right")
    _despine(ax)

    # (b) the beyond-pairwise energy itself. Plotted signed and in kJ/mol rather than as a
    # share of the total: the total is attractive and the many-body part is repulsive, so a
    # ratio comes out negative and reads as an error when it is the physics.
    ax = axes[1]
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        ax.hist(mbe["many_body"][m], bins=34, color=SERIES[i], alpha=0.6, lw=0,
                label=CLUSTER_LABEL[tag])
    ax.axvline(0.0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("beyond-pairwise energy  (kJ/mol)")
    ax.set_ylabel("clusters")
    ax.set_title("Non-additivity is repulsive", loc="left")
    ax.legend(loc="upper right")
    _despine(ax)

    # (c) where the non-additivity comes from. Cluster size is identity here (three
    # discrete values), so it gets categorical colors and a legend, not a continuous bar.
    ax = axes[2]
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        ax.scatter(mbe["mb_ff"][m], mbe["mb_corr"][m], color=SERIES[i], s=13, lw=0,
                   alpha=0.75, label=CLUSTER_LABEL[tag])
    ax.set_xlabel("many-body from effective $C_6$  (kJ/mol)")
    ax.set_ylabel("from correction head  (kJ/mol)")
    ax.set_title("Source of the non-additivity", loc="left")
    ax.legend(loc="lower left")
    _despine(ax)

    # (d) the closest thing to a validation available. The model's 2-body sum is built
    # only from isolated dimer evaluations, and dimers are the one place we have direct
    # reference data (2401 w2 frames, MAE 0.045 kJ/mol) -- so `reference - model 2-body`
    # is a decent proxy for the *true* many-body dispersion. Dropping the model's
    # many-body terms should therefore hurt, and hurt more as clusters grow.
    ax = axes[3]
    x = np.arange(len(tags))
    for i, (label, key) in enumerate((("full model", "total"), ("2-body only", "two_body"))):
        vals = [np.abs(mbe[key][mbe["tag"] == t] - mbe["ref"][mbe["tag"] == t]).mean()
                for t in tags]
        pos = x + (i - 0.5) * 0.38
        ax.bar(pos, vals, width=0.35, color=SERIES[i], label=label, lw=0)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_LABEL[t] for t in tags])
    ax.set_ylabel("MAE vs EDA dispersion  (kJ/mol)")
    ax.set_title("Cost of discarding the many-body terms", loc="left")
    ax.legend(loc="upper left")
    _despine(ax)

    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def additive_comparison(variants: dict, path=None):
    """Accuracy by cluster size for each model variant.

    ``variants`` maps a label to its ``{tag: Predictions}`` dict. The informative
    comparison is the full model against **intra-fragment**, which has identical capacity
    but is strictly pairwise-additive: any gap there is many-body content and nothing else.
    The per-species variant is a weaker control -- it removes flexibility *and*
    non-additivity at once, so on dimers (where many-body is zero by definition) its gap
    measures flexibility alone.
    """
    labels = list(variants)
    tags = list(next(iter(variants.values())))
    x = np.arange(len(tags))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, label in enumerate(labels):
        vals = [variants[label][t].mae for t in tags]
        pos = x + (i - (len(labels) - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.9, color=SERIES[i], label=label, lw=0)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLUSTER_LABEL[t]}\n({t})" for t in tags])
    ax.set_ylabel("MAE vs EDA dispersion  (kJ/mol)")
    ax.set_title("Isolating many-body content from raw flexibility", loc="left")
    ax.legend(loc="upper left")
    _despine(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
