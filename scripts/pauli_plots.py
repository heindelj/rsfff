"""Figures for the trained Pauli model, plus the range-separation diagnostic.

Shares the palette and style with :mod:`dispersion_plots` so the two halves of the notebook
read as one document: single-hue blue ramp for continuous magnitude, fixed categorical slots
for identity, text always in ink colors.

The load-bearing figure here is :func:`switching_functions`. Every term in the model carries
two independent distance scales -- where the *physics* hands over (the dispersion Fermi
midpoint ``r0``, learned) and where the *network* is allowed to contribute (the pair head's
compact envelope ``r_on``/``r_off``, fixed). Those are set in different places, in different
units of intent, and nothing in the training loop forces them to agree. Drawing them on one
axis over the actual pair-distance distribution is the only way to see the overlap region
where both are active at once -- which is exactly where gauge leakage lives.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dispersion_plots import (
    CLUSTER_LABEL,
    GRID,
    INK,
    INK_2,
    INK_MUTED,
    SEQ,
    SERIES,
    _despine,
    _unit_line,
    use_style,
)

__all__ = [
    "use_style",
    "correlation_panels",
    "parameter_panels",
    "rank_comparison",
    "switching_functions",
    "many_body_panels",
]

SURFACE_EDGE = "#fcfcfb"
ELEMENT_COLOR = {1: SERIES[0], 8: SERIES[1]}
ELEMENT_LABEL = {1: "H", 8: "O"}


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def pairwise_switch(r, r_on, r_off):
    """Numpy mirror of ``rsfff.mlip.switch.pairwise_switch`` (1 below r_on, 0 above r_off)."""
    return 1.0 - _smoothstep((r - r_on) / (r_off - r_on))


def fermi_switch(r, r0, alpha):
    """Numpy mirror of ``rsfff.ff.damping.fermi_switch`` (0 short range, 1 long range)."""
    return 1.0 / (1.0 + np.exp(-alpha * (r - r0)))


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def correlation_panels(preds: dict, path=None, title=None):
    """Predicted vs reference Pauli (top) and the residual (bottom), per cluster size.

    Points are colored by the share of the energy the correction head supplied. The
    residual row is not decoration: at R^2 > 0.999 every point sits on the diagonal and the
    parity plot alone says nothing about where the model actually works.
    """
    tags = list(preds)
    share = {t: np.abs(p["corr"]) / np.maximum(np.abs(p["ff"]) + np.abs(p["corr"]), 1e-30)
             for t, p in preds.items()}
    vmax = float(np.percentile(np.concatenate(list(share.values())), 98))

    fig, axes = plt.subplots(2, len(tags), figsize=(3.1 * len(tags), 6.0),
                             gridspec_kw={"height_ratios": [1, 0.85]})
    for col, tag in enumerate(tags):
        p = preds[tag]
        err = p["pred"] - p["ref"]
        lo = min(p["ref"].min(), p["pred"].min())
        hi = max(p["ref"].max(), p["pred"].max())
        pad = 0.04 * (hi - lo)

        ax = axes[0, col]
        _unit_line(ax, lo - pad, hi + pad)
        sc = ax.scatter(p["ref"], p["pred"], c=share[tag], cmap=SEQ, vmin=0.0, vmax=vmax,
                        s=5, lw=0, alpha=0.75)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{CLUSTER_LABEL.get(tag, tag)}  (n = {len(err)})")
        ax.set_xlabel("reference  eda_mod_pauli  (kJ/mol)")
        if col == 0:
            ax.set_ylabel("predicted  (kJ/mol)")
        ax.text(0.04, 0.95,
                f"MAE {np.abs(err).mean():.3f}\nRMSE {np.sqrt((err ** 2).mean()):.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK_2)
        _despine(ax)

        ax = axes[1, col]
        ax.axhline(0.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.scatter(p["ref"], err, c=share[tag], cmap=SEQ, vmin=0.0, vmax=vmax,
                   s=5, lw=0, alpha=0.75)
        ax.set_xlabel("reference  (kJ/mol)")
        if col == 0:
            ax.set_ylabel("residual  (kJ/mol)")
        _despine(ax)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015)
    cbar.set_label("share of the energy from the correction head", fontsize=9, color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(title or "Pauli repulsion: Slater multipole backbone + pair correction",
                 fontsize=12, color=INK, y=0.98)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def parameter_panels(params: dict, priors: dict, path=None):
    """Learned Pauli multipoles: charge, damping exponent, |mu|, |Q| -- one row per element.

    Per-element rows rather than shared axes, because O's Pauli charge is ~14x H's: on one
    linear axis the hydrogen distribution collapses to a line and the figure shows nothing
    about the parameter that actually dominates close H-bond contacts.

    Charge and exponent carry their pyCMM prior as an open marker -- the statement worth
    reading is the *displacement* from a fitted classical model. The dipole and quadrupole
    panels have no prior: those heads are zero-initialized, so the reference is the origin
    and any nonzero width is something the fit decided it needed.
    """
    zs = [z for z in (8, 1) if (params["Z"] == z).any()]
    specs = [
        ("q", "Pauli charge  (e)", True),
        ("b", "damping exponent  (1/bohr)", True),
        ("mu", "$|\\mu|$  (e$\\cdot$bohr)", False),
        ("quad", "$|\\Theta|$  (e$\\cdot$bohr$^2$)", False),
    ]
    fig, axes = plt.subplots(len(zs), 4, figsize=(12.5, 1.9 * len(zs) + 1.1), squeeze=False)

    for row, z in enumerate(zs):
        for col, (key, label, has_prior) in enumerate(specs):
            ax = axes[row][col]
            d = params[key][params["Z"] == z]
            if np.allclose(d, 0.0):
                ax.text(0.5, 0.5, "not enabled", transform=ax.transAxes, ha="center",
                        va="center", color=INK_MUTED, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            elif d.std() < 1e-9:
                # A per-species-only parameter is a single value; a histogram of it is a
                # spike that reads as a plotting bug. Show the number instead.
                ax.axvline(d.mean(), color=ELEMENT_COLOR[z], lw=2.5)
                ax.set_xlim(min(d.mean(), priors[z][1]) - 0.15,
                            max(d.mean(), priors[z][1]) + 0.15)
                ax.set_yticks([])
                if has_prior:
                    ax.axvline(priors[z][1], color=INK_2, lw=1.2, ls=(0, (3, 2)))
            else:
                ax.hist(d, bins=45, color=ELEMENT_COLOR[z], alpha=0.65, edgecolor="none")
                ax.axvline(d.mean(), color=ELEMENT_COLOR[z], lw=1.8)
                ax.set_yticks([])
                if has_prior:
                    ax.axvline(priors[z][0], color=INK_2, lw=1.2, ls=(0, (3, 2)))
            if row == len(zs) - 1:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(ELEMENT_LABEL[z], rotation=0, ha="right", va="center",
                              fontsize=12, color=INK)
            _despine(ax)

    axes[0][0].plot([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="pyCMM prior")
    axes[0][0].plot([], [], color=INK_2, lw=1.8, label="learned mean")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Learned Pauli multipoles, pooled over w2-w5", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def rank_comparison(variants: dict, path=None):
    """MAE by cluster size across multipole ranks and the intra-fragment control."""
    tags = list(next(iter(variants.values())))
    x = np.arange(len(tags))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for k, (label, row) in enumerate(variants.items()):
        ax.bar(x + (k - (len(variants) - 1) / 2) * width,
               [row[t] for t in tags], width * 0.9,
               color=SERIES[k % len(SERIES)], label=label, edgecolor=SURFACE_EDGE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_LABEL.get(t, t) for t in tags])
    ax.set_ylabel("MAE vs eda_mod_pauli  (kJ/mol)")
    ax.set_title("What each layer of the multipole expansion buys")
    ax.legend(loc="upper left")
    _despine(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# The range-separation diagnostic
# ---------------------------------------------------------------------------

def switching_functions(pair_r, e_ff, e_corr, *, disp_r0, disp_alpha=8.0,
                        corr_window=(4.0, 5.0), taper_window=(6.0, 7.0),
                        disp_taper_window=(9.0, 10.0), path=None):
    """Every distance scale in the model on one axis, over the real pair distribution.

    Three questions this answers that no single number does:

    1. **How far apart are the learned handover and the delta-learning cutoff?** The
       dispersion Fermi midpoint ``r0`` is learned under a penalty pushing it down; the pair
       head's envelope is a fixed hyperparameter. They are set independently, and the gap
       between them is the band where the full analytic term and the neural correction are
       *both* at full strength -- the region where a systematic backbone error can hide
       inside the network instead of showing up as a fit error.
    2. **Does the correction actually stay short-ranged?** Plotted as the cumulative share
       of |dE| against r, which is the honest version of "the envelope goes to zero at 5 A".
    3. **Are the cutoffs in the right place at all?** The pair-distance histogram shows
       where the pairs are; a switch that sits in an empty region is doing nothing.
    """
    # Three stacked panels on one shared distance axis rather than two y-scales on one
    # panel: counts and cumulative fractions do not belong on a shared axis.
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.6, 0.75]})
    r = np.linspace(0.0, max(taper_window[1], disp_taper_window[1]) + 0.6, 700)

    ax = axes[0]
    ax.plot(r, fermi_switch(r, disp_r0, disp_alpha), color=SERIES[0], lw=2.0,
            label=f"dispersion handover $S_{{Fermi}}$ — LEARNED, $r_0$ = {disp_r0:.2f} $\\AA$")
    ax.plot(r, pairwise_switch(r, *corr_window), color=SERIES[1], lw=2.0,
            label=f"correction envelope $W$ — FIXED, {corr_window[0]:.0f}-{corr_window[1]:.0f} $\\AA$")
    ax.plot(r, pairwise_switch(r, *taper_window), color=SERIES[2], lw=2.0, ls=(0, (5, 2)),
            label=f"Pauli neighbor taper ({taper_window[0]:.0f}-{taper_window[1]:.0f} $\\AA$)")
    ax.plot(r, pairwise_switch(r, *disp_taper_window), color=SERIES[3], lw=2.0,
            ls=(0, (5, 2)),
            label=f"dispersion neighbor taper ({disp_taper_window[0]:.0f}-{disp_taper_window[1]:.0f} $\\AA$)")

    # Where the analytic term and the correction are both at full strength. For dispersion
    # that starts at the learned r0; the Pauli backbone has no handover at all, so its
    # overlap with the correction runs all the way in.
    ax.axvspan(disp_r0, corr_window[0], color=SERIES[4], alpha=0.10, lw=0)
    ax.axvspan(0.0, disp_r0, color=SERIES[2], alpha=0.07, lw=0)
    ax.annotate(
        f"dispersion: backbone + $\\Delta E$\nboth full strength\n"
        f"({disp_r0:.2f}-{corr_window[0]:.1f} $\\AA$)",
        xy=((disp_r0 + corr_window[0]) / 2, 0.45), ha="center", va="center",
        fontsize=8.2, color=INK_2,
    )
    ax.annotate(
        "Pauli has\nno handover:\n$\\Delta E$ overlaps\nthe backbone\nall the way in",
        xy=(disp_r0 / 2, 0.45), ha="center", va="center", fontsize=7.6, color=INK_2,
    )
    ax.set_ylabel("switch value")
    ax.set_xlim(0, r[-1])
    ax.set_ylim(-0.04, 1.16)
    ax.set_title("Learned range separation vs the fixed delta-learning cutoff")
    ax.legend(loc="center right", fontsize=8.2)
    _despine(ax)

    ax = axes[1]
    ax.hist(pair_r, bins=90, range=(0, r[-1]), color=SERIES[0], alpha=0.35,
            edgecolor="none")
    for x, color in ((disp_r0, SERIES[0]), (corr_window[0], SERIES[1])):
        ax.axvline(x, color=color, lw=1.2, ls=(0, (2, 2)))
    ax.set_ylabel("inter-fragment pairs")
    _despine(ax)

    ax = axes[2]
    order = np.argsort(pair_r)
    r_sorted = pair_r[order]
    for arr, color, label in ((e_ff, SERIES[0], "$|E_{FF}|$  (Slater multipoles)"),
                              (e_corr, SERIES[1], "$|\\Delta E|$  (correction head)")):
        mag = np.abs(arr)[order]
        total = mag.sum()
        if total <= 0:
            continue
        ax.plot(r_sorted, np.cumsum(mag) / total, color=color, lw=2.0, label=label)
    for x, color in ((disp_r0, SERIES[0]), (corr_window[0], SERIES[1])):
        ax.axvline(x, color=color, lw=1.2, ls=(0, (2, 2)))
    ax.set_ylabel("cumulative share of |energy|")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlabel("pair distance  $r$  ($\\AA$)")
    ax.set_xlim(0, r[-1])
    _despine(ax)

    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Many-body expansion
# ---------------------------------------------------------------------------

def many_body_panels(mbe, path=None):
    """The Pauli many-body expansion, split into force-field and correction sources.

    ``SlaterPauli`` is a *pair sum*, so with per-species multipoles it would be strictly
    additive and every ``E^(k>=3)`` identically zero. Two mechanisms can make it non-additive
    anyway, and this figure separates them:

    - **force field** -- the emitted charges, dipoles and quadrupoles are functions of the
      environment, so deleting a neighbor changes the multipoles of the molecules that
      remain. That is real many-body physics carried by an explicit functional form.
    - **correction head** -- the pair features are environment-aware too, so the neural
      delta is non-additive by the same route, but with no physical constraint on it.

    Which one dominates is the question the delta-learning arrangement lives or dies on: a
    correction supplying most of the non-additivity means the physics has leaked into the
    network.
    """
    tags = [t for t in ("w3", "w4", "w5") if (mbe["tag"] == t).any()]
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.6))

    # (a) magnitude by order -- log scale, because 2-body dwarfs the rest by ~2 decades.
    ax = axes[0]
    orders = [2, 3, 4, 5]
    width = 0.8 / max(len(tags), 1)
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        vals = [np.abs(mbe[f"total_{ {2: 'two_body', 3: 'e3', 4: 'e4', 5: 'e5'}[o] }"][m]).mean()
                for o in orders]
        offs = (i - (len(tags) - 1) / 2) * width
        ax.bar([o + offs for o in orders], vals, width=width * 0.92, color=SERIES[i],
               label=CLUSTER_LABEL[tag], lw=0)
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("expansion order $k$")
    ax.set_ylabel(r"mean $|E^{(k)}|$  (kJ/mol)")
    ax.set_title("Magnitude by many-body order", loc="left")
    ax.legend(loc="upper right")
    _despine(ax)

    # (b) the same orders >= 3, signed and split by source. This is the panel the question
    # is actually about. The correction's bars are typically far too small to see, which is
    # itself the result -- so they are labeled with their value rather than left invisible.
    ax = axes[1]
    ax.axhline(0.0, color=INK, lw=1.0)
    sub_orders = [3, 4, 5]
    group = 0.8 / max(len(tags), 1)
    inner = group / 2
    span = max(
        abs(mbe[f"{c}_{k}"].mean())
        for c in ("ff", "corr") for k in ("e3", "e4", "e5")
    )
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        for j, (comp, hatch, label) in enumerate(
            (("ff", None, "force field (multipoles)"), ("corr", "////", "correction head"))
        ):
            vals = [mbe[f"{comp}_{ {3: 'e3', 4: 'e4', 5: 'e5'}[o] }"][m].mean()
                    for o in sub_orders]
            offs = (i - (len(tags) - 1) / 2) * group + (j - 0.5) * inner
            ax.bar([o + offs for o in sub_orders], vals, width=inner * 0.88,
                   color=SERIES[i], hatch=hatch, edgecolor=SURFACE_EDGE, lw=0.6,
                   label=label if i == 0 else None)
    biggest_corr = max(
        abs(mbe[f"corr_{k}"].mean()) for k in ("e3", "e4", "e5")
    )
    if biggest_corr < 0.02 * span:
        # Its bars are invisible at this scale, which *is* the result -- say so in words
        # rather than leaving the reader to wonder whether the series failed to plot.
        ax.text(0.5, 0.06, f"correction head: $|E^{{(k)}}| < ${biggest_corr:.0e} kJ/mol"
                           f" at every order",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color=INK_MUTED)
    ax.set_xticks(sub_orders)
    ax.set_xlabel("expansion order $k$")
    ax.set_ylabel(r"mean $E^{(k)}$  (kJ/mol)")
    ax.set_title("Non-additivity by source (signed)", loc="left")
    # Bars run negative at k=3, so the upper left is the empty quadrant.
    ax.legend(loc="upper left", fontsize=8.0)
    _despine(ax)

    # (c) the split as a share, so a near-zero correction reads as "0%" rather than as a
    # missing bar. Signed magnitudes are summed as |.| because the two sources can oppose.
    ax = axes[2]
    ff_share, corr_share = [], []
    for tag in tags:
        m = mbe["tag"] == tag
        a = np.abs(mbe["ff_many_body"][m]).mean()
        b = np.abs(mbe["corr_many_body"][m]).mean()
        ff_share.append(100 * a / max(a + b, 1e-30))
        corr_share.append(100 * b / max(a + b, 1e-30))
    x = np.arange(len(tags))
    ax.bar(x, ff_share, 0.55, color=SERIES[0], lw=0)
    ax.bar(x, corr_share, 0.55, bottom=ff_share, color=SERIES[1], hatch="////",
           edgecolor=SURFACE_EDGE, lw=0.6)
    # Direct labels rather than a legend: one segment is invisible at this scale, so a
    # legend swatch for it would be the only place it appears.
    for xi, (f, c) in enumerate(zip(ff_share, corr_share)):
        ax.text(xi, 52, f"{f:.1f}%", ha="center", va="center", color=SURFACE_EDGE,
                fontsize=11, weight="bold")
        ax.text(xi, 44, "force field", ha="center", va="center", color=SURFACE_EDGE,
                fontsize=8)
        ax.text(xi, 102, f"correction {c:.2f}%", ha="center", va="bottom",
                fontsize=8, color=SERIES[1])
    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_LABEL[t] for t in tags])
    ax.set_ylim(0, 118)
    ax.set_ylabel("share of $|$beyond-pairwise$|$  (%)")
    ax.set_title("Source of the non-additivity", loc="left")
    _despine(ax)

    # (d) does the learned non-additivity track the reference? The model's 2-body sum is
    # built only from isolated dimer evaluations, and dimers are where direct reference
    # data exists -- so `reference - model 2-body` is a proxy for the true many-body Pauli.
    # Not fully independent (the model was trained on these totals), but the slope is
    # informative.
    ax = axes[3]
    x = mbe["ref"] - mbe["total_two_body"]
    y = mbe["total_many_body"]
    for i, tag in enumerate(tags):
        m = mbe["tag"] == tag
        ax.scatter(x[m], y[m], color=SERIES[i], s=13, lw=0, alpha=0.75,
                   label=CLUSTER_LABEL[tag])
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    _unit_line(ax, lo - pad, hi + pad)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    slope = np.polyfit(y, x, 1)[0]
    r = np.corrcoef(y, x)[0, 1]
    ax.text(0.04, 0.95, f"r = {r:.4f}\nslope = {slope:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color=INK_2)
    ax.set_xlabel("reference $-$ model 2-body  (kJ/mol)")
    ax.set_ylabel("model many-body  (kJ/mol)")
    ax.set_title("Does it track the reference?", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    _despine(ax)

    fig.suptitle("Many-body expansion of the Pauli repulsion", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
