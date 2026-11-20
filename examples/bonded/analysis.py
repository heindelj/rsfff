"""Read a bonded-model run (rsfff.train.bonded) and plot its held-out scans.

    python examples/bonded/analysis.py examples/bonded/run        # writes scans.png and prints the summary

or, from a notebook (analysis.ipynb): ``import analysis; analysis.summary(run); analysis.plot_scans(run)``.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KCAL = 627.509474
LABELS = (("e_mae", "E mae (kcal/mol)"), ("f_rmse", "F rmse (kcal/mol/A)"), ("dipole_rmse", "dipole (e bohr)"),
          ("quadrupole_rmse", "second moment (e bohr^2)"), ("polarizability_rmse", "alpha (bohr^3)"),
          ("dipole_derivatives_rmse", "dmu/dR (e)"), ("polarizability_derivatives_rmse", "dalpha/dR (bohr^2)"),
          ("hessian_rmse", "Hessian (Ha/bohr^2)"))


def load(run):
    """(metrics dict, held-out rows as dicts) of a run directory."""
    run = Path(run)
    metrics = json.loads((run / "metrics.json").read_text())
    path = run / "holdout.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    return metrics, rows


def summary(run, file=sys.stdout):
    """Print validation and held-out errors per data file, the trained model next to its tables alone."""
    metrics, _ = load(run)
    for split, const in (("validation", "constant"), ("holdout", "holdout_constant")):
        print(f"\n{split} (tables alone in parentheses)", file=file)
        for name, r in metrics[split].items():
            c = metrics[const][name]
            print(f"  {name} ({r['n']} frames)", file=file)
            for key, label in LABELS:
                if key in r:
                    print(f"    {label:28s} {r[key]:10.4f}  ({c[key]:.4f})", file=file)
    print("\nharmonic frequencies at the model's minimum, cm^-1 (reference)", file=file)
    for name, r in metrics["minima"].items():
        ref = r.get("reference_frequencies")
        print(f"  {name:12s} " + " ".join(f"{w:7.0f}" for w in r["frequencies"])
              + ("" if ref is None else "   (" + " ".join(f"{w:.0f}" for w in ref) + ")"), file=file)
    return metrics


def scans(run):
    """{(file, scan name): (reference energies, model energies)} in kcal/mol relative to each scan's reference
    minimum, in file order (the scan coordinate's order)."""
    _, rows = load(run)
    out = defaultdict(lambda: ([], []))
    for r in rows:
        ref, mod = out[(r["file"], r["config_type"])]
        ref.append(float(r["energy_reference"]))
        mod.append(float(r["energy_model"]))
    res = {}
    for key, (ref, mod) in out.items():
        ref, mod = np.array(ref), np.array(mod)
        res[key] = ((ref - ref.min()) * KCAL, (mod - ref.min()) * KCAL)
    return res


def plot_scans(run, path=None, columns=4):
    """One panel per held-out scan: reference (points) and model (line). Returns the figure."""
    import matplotlib.pyplot as plt
    data = scans(run)
    if not data:
        raise ValueError(f"{run} has no held-out frames (data.holdout in the config)")
    rows = -(-len(data) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.6 * rows), squeeze=False)
    for ax, ((file, name), (ref, mod)) in zip(axes.flat, sorted(data.items())):
        x = np.arange(len(ref))
        ax.plot(x, ref, "o", ms=4, color="black", label="reference")
        ax.plot(x, mod, "-", color="tab:red", label="model")
        ax.set_title(f"{file}: {name.replace('scan_', '')}", fontsize=8)
        ax.set_xlabel("scan point", fontsize=8)
        ax.set_ylabel("kcal/mol", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(data):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/bonded/run")
    summary(run_dir)
    import matplotlib
    matplotlib.use("Agg")
    plot_scans(run_dir, run_dir / "scans.png")
    print(f"\nwrote {run_dir / 'scans.png'}")
