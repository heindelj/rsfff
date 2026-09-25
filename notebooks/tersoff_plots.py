"""Diagnostics for the Tersoff model against the pairing model: the ablation, drawn.

The code behind ``notebooks/tersoff_plots.ipynb``; the notebook sets knobs and calls in here
(``%autoreload`` picks up edits). Everything :mod:`pairing_plots` does -- the Q-Chem scans,
the one-body decomposition, bond orders and formal charges along the scans, the monomer
parity -- works unchanged on a Tersoff model (its state comes back in the pairing model's
:class:`ElectronicState`), so this module re-exports it and adds what the explicit bond order
specifically needs to show:

* :func:`dimer_scan` / :func:`plot_dimer` -- the discriminator: a hydrogen-bonded water dimer
  along the O-O distance, with the covalent ``p(O-H)``, the hydrogen bond's ``p(O...H)``, its
  *raw* order ``b = clip(J/kappa)`` (what the saturation rule has to squeeze), the donor
  hydrogen's total order and the co-membership the classical channels see.
* :func:`plot_pt_state` -- the shared proton's two bond orders, their sum (the under-filling
  of the per-atom rule shows here), and the oxygens' formal charges and capacities along the
  proton-transfer scans, one row per model.
* :func:`plot_valence` -- capacities and coordination of the oxygens along every scan.

Models: :func:`pairing_plots.build_prior_model` builds either model at its priors from its
config (``tersoff_saturation="rebo"`` and the other ``tersoff_*`` fields can be overridden),
:func:`pairing_plots.load_model` loads a checkpoint of either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
if str(ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(ROOT / "notebooks"))

import pairing_plots as _pp  # noqa: E402

_pp.FIG_PREFIX = "tersoff"          # this notebook's figures: notebooks/figures/tersoff_*

from pairing_plots import (  # noqa: E402,F401
    MODEL_COLORS,
    SCANS,
    Scan,
    ScanResult,
    build_prior_model,
    evaluate_all,
    evaluate_scan,
    is_pairing,
    load_model,
    load_scans,
    model_kind,
    monomer_parity,
    plot_bend,
    plot_bond_orders,
    plot_formal_charges,
    plot_monomer_parity,
    plot_proton_transfer,
    plot_stretches,
    savefig,
)
from rsfff.train.data import Batch  # noqa: E402

__all__ = [
    "build_prior_model", "build_prior_tersoff_model", "dimer_batch", "dimer_scan",
    "plot_dimer", "plot_pt_state", "plot_valence", "prior_models", "savefig_tersoff",
]


def savefig_tersoff(fig, name):
    return savefig(fig, name, prefix="tersoff")


def build_prior_tersoff_model(config_path=ROOT / "configs" / "water_tersoff.yaml", **film_over):
    return build_prior_model(config_path, **film_over)


def prior_models(*, rebo: bool = True, pairing: bool = True) -> dict:
    """``{label: model}`` at the priors: tersoff (waterfill), optionally tersoff:rebo and pairing."""
    models = {"tersoff (priors)": build_prior_tersoff_model()[0]}
    if rebo:
        models["tersoff:rebo (priors)"] = build_prior_tersoff_model(tersoff_saturation="rebo")[0]
    if pairing:
        models["pairing (priors)"] = build_prior_model(ROOT / "configs" / "water_pairing.yaml")[0]
    return models


# ---------------------------------------------------------------------------
# the dimer


def dimer_batch(r_oo: float, dtype=None) -> Batch:
    """A linear hydrogen-bonded water dimer: the donor O-H on the O-O axis, the acceptor's
    plane perpendicular to it. Atoms: ``[O_d, H_d (donated), H_d', O_a, H_a, H_a']``."""
    dtype = dtype or torch.get_default_dtype()
    donor = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
    c, s = 0.5, 0.8660254
    acceptor = torch.tensor([
        [r_oo, 0.0, 0.0],
        [r_oo + 0.9572 * c, 0.0, 0.9572 * s],
        [r_oo + 0.9572 * c, 0.0, -0.9572 * s],
    ])
    positions = torch.cat((donor, acceptor)).to(dtype)
    return Batch(
        positions=positions,
        atomic_numbers=torch.tensor([8, 1, 1, 8, 1, 1]),
        batch_idx=torch.zeros(6, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1, dtype=dtype),
        fragment_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
        fragment_charge=torch.zeros(2, dtype=dtype),
        fragment_two_s=torch.zeros(2, dtype=dtype),
        fragment_to_batch=torch.zeros(2, dtype=torch.long),
        n_fragments=2,
    )


def _pair(out, i, j):
    """Position of the candidate pair ``(i, j)`` in ``out.sub_index``, or None."""
    pi = out.pair_index[:, out.sub_index]
    hit = torch.nonzero((pi[0] == min(i, j)) & (pi[1] == max(i, j)), as_tuple=False)
    return None if hit.numel() == 0 else int(hit[0])


def dimer_scan(models: dict, r_oo=np.linspace(2.5, 4.0, 31)) -> dict[str, dict[str, np.ndarray]]:
    """``{label: {quantity: (n,)}}`` along the O-O distance for every bond-order model.

    Quantities: ``E`` (Hartree), ``p_OH`` (the donor's covalent bond), ``p_HB`` (the hydrogen
    bond), ``b_HB`` (its raw order), ``c_HB`` (its co-membership), ``n_H`` (the donated
    hydrogen's total bond order), ``u_H``.
    """
    from rsfff.ff.tersoff.bond_order import raw_bond_order

    out_all: dict[str, dict[str, np.ndarray]] = {}
    for label, model in models.items():
        if not is_pairing(model):
            continue
        rows = {k: np.full(len(r_oo), np.nan) for k in ("E", "p_OH", "p_HB", "b_HB", "c_HB", "n_H", "u_H")}
        for k, r in enumerate(r_oo):
            with torch.no_grad():
                out = model(dimer_batch(float(r)))
            rows["E"][k] = float(out.energy.sum())
            a, b = _pair(out, 0, 1), _pair(out, 1, 3)
            if a is not None:
                rows["p_OH"][k] = float(out.bond_order[a])
            if b is not None:
                rows["p_HB"][k] = float(out.bond_order[b])
                rows["c_HB"][k] = float(out.p_intra[out.sub_index[b]])
                rows["b_HB"][k] = float(raw_bond_order(out.coupling[b], out.kappa_pair[b], model.temperature))
            sp = out.pair_index[:, out.sub_index]
            n = torch.zeros(6).index_add_(0, sp[0], out.bond_order).index_add_(0, sp[1], out.bond_order)
            rows["n_H"][k] = float(n[1])
            rows["u_H"][k] = float(out.unpaired[1])
        out_all[label] = rows
    return out_all


def plot_dimer(r_oo, scans, path=None):
    """Left: the dimer's energy relative to its own minimum. Middle: the hydrogen bond's raw
    and saturated orders and its co-membership. Right: the covalent bond and the donated
    hydrogen's total."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    from rsfff.ff.units import KJMOL_PER_HARTREE

    for label, rows in scans.items():
        color = MODEL_COLORS.get(label)
        e = rows["E"]
        axes[0].plot(r_oo, (e - np.nanmin(e)) * KJMOL_PER_HARTREE, "-", color=color, label=label)
        axes[1].plot(r_oo, rows["b_HB"], ":", color=color, label=f"{label}: raw $b$(O···H)")
        axes[1].plot(r_oo, rows["p_HB"], "-", color=color, label=f"{label}: $p$(O···H)")
        axes[1].plot(r_oo, rows["c_HB"], "--", color=color, lw=0.9, label=f"{label}: $c$(O···H)")
        axes[2].plot(r_oo, rows["p_OH"], "-", color=color, label=f"{label}: $p$(O-H)")
        axes[2].plot(r_oo, rows["n_H"], "--", color=color, lw=0.9, label=f"{label}: $\\sum_j p_{{Hj}}$")
    axes[0].set_ylabel("dimer energy relative to its minimum (kJ/mol)")
    axes[1].set_ylabel("hydrogen-bond order")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[2].set_ylabel("covalent order of the donor O-H")
    for ax in axes:
        ax.set_xlabel("O-O ($\\AA$)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        savefig_tersoff(fig, path)
    return fig


# ---------------------------------------------------------------------------
# the proton-transfer state and the capacities along the scans


def plot_pt_state(scans, results, name="h5o2+_pt", path=None):
    """One row per bond-order model: the shared proton's two orders and their sum (left), the
    oxygens' formal charges (middle) and capacities against their coordination (right),
    one curve set per O-O distance."""
    scan = scans[name]
    z = scan.frames[0].get_atomic_numbers()
    oxygens = [int(i) for i in np.flatnonzero(z == 8)]
    labels = [lab for lab, res in results[name].items() if res.bond_order]
    groups = scan.groups()
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
    fig, axes = plt.subplots(len(labels), 3, figsize=(15, 3.8 * len(labels)), squeeze=False)
    for row, label in zip(axes, labels):
        res = results[name][label]
        pairs = list(res.bond_order)
        for g, color in zip(groups, cmap):
            sel = scan.group == g
            x = scan.coord[sel]
            p1, p2 = res.bond_order[pairs[0]][sel], res.bond_order[pairs[1]][sel]
            row[0].plot(x, p1, "-", color=color, lw=1.1, label=f"{pairs[0]} (O-O {g:.2f})")
            row[0].plot(x, p2, "--", color=color, lw=1.1)
            row[0].plot(x, p1 + p2, ":", color=color, lw=1.6)
            for o, ls in zip(oxygens, ("-", "--")):
                row[1].plot(x, res.formal_charge[sel, o], ls, color=color, lw=1.1,
                            label=f"q O{o} (O-O {g:.2f})" if ls == "-" else None)
                row[2].plot(x, res.valence[sel, o], ls, color=color, lw=1.1,
                            label=f"v O{o} (O-O {g:.2f})" if ls == "-" else None)
                row[2].plot(x, res.coordination[sel, o], ls, color=color, lw=0.7, alpha=0.5)
        row[0].set_ylabel(f"{label}\nshared proton: $p_1$ (solid), $p_2$ (dashed), sum (dotted)")
        row[1].set_ylabel("formal charge of the oxygens")
        row[2].set_ylabel("capacity $v$ (thick) and $\\sum_j p$ (thin) of the oxygens")
        row[0].set_ylim(-0.05, 1.1)
        for ax in row:
            ax.set_xlabel(scan.xlabel)
            ax.legend(fontsize=6)
    fig.suptitle(name)
    fig.tight_layout()
    if path:
        savefig_tersoff(fig, path)
    return fig


def plot_valence(scans, results, names=None, path=None):
    """Capacity and coordination of every oxygen along each scan, per model."""
    names = [n for n in (names or SCANS) if n in scans]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        scan = scans[name]
        z = scan.frames[0].get_atomic_numbers()
        oxygens = [int(i) for i in np.flatnonzero(z == 8)]
        for label, res in results[name].items():
            if res.valence is None:
                continue
            color = MODEL_COLORS.get(label)
            groups = scan.groups() if scan.kind == "pt" else [None]
            for g in groups:
                sel = np.ones(scan.n, bool) if g is None else scan.group == g
                for o, ls in zip(oxygens, ("-", "--")):
                    ax.plot(scan.coord[sel], res.valence[sel, o], ls, color=color, lw=1.1,
                            label=f"{label}: v O{o}" if g in (None, groups[0]) else None)
                    ax.plot(scan.coord[sel], res.coordination[sel, o], ls, color=color, lw=0.6, alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel(scan.xlabel)
        ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("capacity (thick) and coordination (thin)")
    fig.tight_layout()
    if path:
        savefig_tersoff(fig, path)
    return fig
