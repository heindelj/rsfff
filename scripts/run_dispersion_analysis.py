"""Generate every dispersion figure and print the summary numbers.

    python scripts/run_dispersion_analysis.py

Runs the same code the notebook does (``notebooks/eda_rsff_plotting.ipynb``); this entry
point exists so the figures can be regenerated headlessly and the numbers checked without
a kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dispersion_plots as dp  # noqa: E402
from dispersion_analysis import (  # noqa: E402
    CLUSTERS,
    atom_parameters,
    load_clusters,
    load_model,
    many_body_table,
    predict,
)

FIGDIR = Path("notebooks/figures")
CKPT = "checkpoints/eda_water_disp/best.pt"
CONFIG = "configs/eda_water_disp_full.yaml"
CKPT_ADD = "checkpoints/eda_water_disp_additive/best.pt"
CONFIG_ADD = "configs/eda_water_disp_additive.yaml"
CKPT_INTRA = "checkpoints/eda_water_disp_intra/best.pt"
CONFIG_INTRA = "configs/eda_water_disp_intra.yaml"
PER_CLUSTER = 167          # ~500 clusters over trimers/tetramers/pentamers


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    dp.use_style()

    model, cfg, state = load_model(CKPT, CONFIG)
    print(f"model: epoch {state['epoch']}  val_loss {state['val_loss']:.4f}  "
          f"r0 = {float(model.r0.detach()):.3f} A")
    datasets = load_clusters()

    # --- accuracy -------------------------------------------------------------
    preds = {tag: predict(model, datasets[tag], tag) for tag in CLUSTERS}
    print("\ncluster    n     MAE      RMSE     R^2      <|E_corr|/|E_tot|>")
    for tag, p in preds.items():
        print(f"{tag:<9}{len(p.ref):<6}{p.mae:<9.4f}{p.rmse:<9.4f}{p.r2:<9.5f}"
              f"{p.corr_fraction.mean():.4f}")
    allp = np.concatenate([p.error for p in preds.values()])
    print(f"overall MAE {np.abs(allp).mean():.4f} kJ/mol over {len(allp)} frames")

    dp.correlation_panels(preds, path=FIGDIR / "disp_correlation.png")
    print(f"wrote {FIGDIR / 'disp_correlation.png'}")

    # --- learned parameters ---------------------------------------------------
    _, priors = atom_parameters(model, datasets["w2"], n_frames=1)
    combined = {k: np.concatenate([
        atom_parameters(model, datasets[t], n_frames=150)[0][k] for t in CLUSTERS
    ]) for k in ("Z", "c6", "b", "n_fragments", "contact")}
    print("\nlearned effective parameters (over w2-w5):")
    for z, sym in ((1, "H"), (8, "O")):
        m = combined["Z"] == z
        c6, b = combined["c6"][m], combined["b"][m]
        print(f"  {sym}: C6 {c6.mean():7.3f} +/- {c6.std():.3f} "
              f"(prior {priors[z][0]:.3f}, range {c6.min():.3f}-{c6.max():.3f})   "
              f"b {b.mean():5.3f} +/- {b.std():.3f} (prior {priors[z][1]:.3f})")

    dp.parameter_panels(combined, priors, path=FIGDIR / "disp_parameters.png")
    dp.parameter_vs_environment(combined, priors, path=FIGDIR / "disp_parameters_env.png")
    print(f"wrote {FIGDIR / 'disp_parameters.png'}, {FIGDIR / 'disp_parameters_env.png'}")

    # --- many-body expansion --------------------------------------------------
    mbe = many_body_table(model, datasets, PER_CLUSTER)
    print(f"\nmany-body expansion over {len(mbe['total'])} strongly-interacting clusters:")
    print("cluster   n     total     2-body    3-body    4-body    5-body   |MB|/|tot|")
    for tag in ("w3", "w4", "w5"):
        m = mbe["tag"] == tag
        print(f"{tag:<10}{m.sum():<6}{mbe['total'][m].mean():<10.3f}"
              f"{mbe['two_body'][m].mean():<10.3f}{mbe['e3'][m].mean():<10.3f}"
              f"{mbe['e4'][m].mean():<10.3f}{mbe['e5'][m].mean():<10.3f}"
              f"{np.abs(mbe['many_body'][m] / mbe['total'][m]).mean() * 100:.2f}%")
    print(f"  non-additivity source: from effective C6 {mbe['mb_ff'].mean():+.4f}, "
          f"from correction head {mbe['mb_corr'].mean():+.4f} kJ/mol (means)")

    print("\ncost of discarding the many-body terms (MAE vs eda_disp, kJ/mol):")
    print(f"{'cluster':<10}{'full':<10}{'2-body only':<14}{'model MB':<12}"
          f"{'reference asks for':<20}")
    for tag in ("w3", "w4", "w5"):
        m = mbe["tag"] == tag
        full = np.abs(mbe["total"][m] - mbe["ref"][m]).mean()
        two = np.abs(mbe["two_body"][m] - mbe["ref"][m]).mean()
        print(f"{tag:<10}{full:<10.4f}{two:<14.4f}"
              f"{(mbe['total'][m] - mbe['two_body'][m]).mean():<+12.4f}"
              f"{(mbe['ref'][m] - mbe['two_body'][m]).mean():<+20.4f}")
    slope = np.polyfit(mbe["total"] - mbe["two_body"], mbe["ref"] - mbe["two_body"], 1)[0]
    r = np.corrcoef(mbe["total"] - mbe["two_body"], mbe["ref"] - mbe["two_body"])[0, 1]
    print(f"  model many-body vs (reference - model 2-body): r = {r:.4f}, slope = {slope:.3f}")

    dp.many_body_panels(mbe, path=FIGDIR / "disp_many_body.png")
    print(f"wrote {FIGDIR / 'disp_many_body.png'}")

    # --- controls -------------------------------------------------------------
    variants = {"environment-aware (full)": preds}
    for label, ckpt, conf in (
        ("intra-fragment (additive, same capacity)", CKPT_INTRA, CONFIG_INTRA),
        ("per-species only (additive, rigid)", CKPT_ADD, CONFIG_ADD),
    ):
        if not Path(ckpt).exists():
            print(f"\nskipping control '{label}': {ckpt} not found")
            continue
        m, _, s = load_model(ckpt, conf)
        variants[label] = {tag: predict(m, datasets[tag], tag) for tag in CLUSTERS}
        print(f"\ncontrol '{label}': val_loss {s['val_loss']:.4f}")

    print("\nMAE by cluster size (kJ/mol):")
    print(f"{'variant':<44}" + "".join(f"{t:<10}" for t in CLUSTERS))
    for label, p in variants.items():
        print(f"{label:<44}" + "".join(f"{p[t].mae:<10.4f}" for t in CLUSTERS))
    if len(variants) > 1:
        dp.additive_comparison(variants, path=FIGDIR / "disp_additive_control.png")
        print(f"wrote {FIGDIR / 'disp_additive_control.png'}")

    intra = variants.get("intra-fragment (additive, same capacity)")
    if intra is not None:
        print("\nmany-body gain (full vs the strictly-additive model of equal capacity):")
        for tag in CLUSTERS:
            gain = intra[tag].mae - preds[tag].mae
            print(f"  {tag}: {intra[tag].mae:.4f} -> {preds[tag].mae:.4f}  "
                  f"({gain:+.4f} kJ/mol, {intra[tag].mae / preds[tag].mae:.2f}x)")

    np.savez(
        FIGDIR / "disp_analysis.npz",
        **{f"{t}_{k}": getattr(preds[t], k)
           for t in CLUSTERS for k in ("pred", "ref", "ff", "corr")},
        **{f"mbe_{k}": v for k, v in mbe.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
