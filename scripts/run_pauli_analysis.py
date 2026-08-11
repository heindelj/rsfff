"""Generate every Pauli figure and print the summary numbers.

    python scripts/run_pauli_analysis.py

Runs the same code the notebook does (``notebooks/eda_rsff_plotting.ipynb``); this entry
point exists so the figures can be regenerated headlessly and the numbers checked without a
kernel.
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

import pauli_plots as pp  # noqa: E402
from pauli_analysis import (  # noqa: E402
    CLUSTERS,
    atom_parameters,
    load_clusters,
    load_model,
    many_body_table,
    predict,
)
from rsfff.ff.pauli import DEFAULT_PAULI_PRIOR  # noqa: E402

FIGDIR = Path("notebooks/figures")

#: (label, checkpoint, config). The rank ladder plus the additive control.
RUNS = [
    ("rank 2 (+quadrupoles)",
     "checkpoints/eda_water_pauli_q/best.pt", "configs/eda_water_pauli_quad.yaml"),
    ("rank 1 (charges + dipoles)",
     "checkpoints/eda_water_pauli/best.pt", "configs/eda_water_pauli.yaml"),
    ("rank 1, intra-fragment features",
     "checkpoints/eda_water_pauli_intra/best.pt", "configs/eda_water_pauli_intra.yaml"),
]

#: The dispersion model's learned Fermi midpoint, for the range-separation figure.
DISP_CKPT = "checkpoints/eda_water_disp/best.pt"
DISP_CONFIG = "configs/eda_water_disp_full.yaml"

PER_CLUSTER = 167          # ~500 clusters over trimers/tetramers/pentamers


def _learned_disp_r0(default=1.481):
    """The dispersion model's learned range-separation midpoint, in Angstrom."""
    if not Path(DISP_CKPT).exists():
        print(f"  (dispersion checkpoint missing; using r0 = {default} A)")
        return default
    import torch.nn.functional as F
    state = torch.load(DISP_CKPT, map_location="cpu", weights_only=False)
    raw = state["model_state"]["dispersion.r0_raw"]
    return float(F.softplus(raw))


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pp.use_style()
    datasets = load_clusters()

    table, headline = {}, None
    for label, ckpt, conf in RUNS:
        if not Path(ckpt).exists():
            print(f"skipping '{label}': {ckpt} not found")
            continue
        model, cfg, state = load_model(ckpt, conf)
        preds = {tag: predict(model, ds, cfg.pauli.target) for tag, ds in datasets.items()}
        row = {}
        print(f"\n[{label}] epoch {state['epoch']}  val_loss {state['val_loss']:.4f}  "
              f"max_rank {cfg.pauli.max_rank}")
        print(f"{'cluster':<9}{'MAE':<9}{'RMSE':<9}{'R^2':<10}{'corr share'}")
        for tag, p in preds.items():
            err = p["pred"] - p["ref"]
            share = np.abs(p["corr"]) / np.maximum(np.abs(p["ff"]) + np.abs(p["corr"]), 1e-30)
            row[tag] = float(np.abs(err).mean())
            print(f"{tag:<9}{row[tag]:<9.4f}{np.sqrt((err ** 2).mean()):<9.4f}"
                  f"{np.corrcoef(p['pred'], p['ref'])[0, 1] ** 2:<10.5f}"
                  f"{share.mean() * 100:.2f}%")
        table[label] = row
        if headline is None:
            headline = (label, model, cfg, preds)

    if headline is None:
        print("no Pauli checkpoints found; train one first")
        return 1
    label, model, cfg, preds = headline

    pp.correlation_panels(preds, path=FIGDIR / "pauli_correlation.png",
                          title=f"Pauli repulsion: {label}")
    print(f"\nwrote {FIGDIR / 'pauli_correlation.png'}")

    params = atom_parameters(model, datasets, n_frames=150, target=cfg.pauli.target)
    pp.parameter_panels(params, DEFAULT_PAULI_PRIOR, path=FIGDIR / "pauli_parameters.png")
    print(f"wrote {FIGDIR / 'pauli_parameters.png'}")
    for z, sym in ((1, "H"), (8, "O")):
        m = params["Z"] == z
        print(f"  {sym}: q {params['q'][m].mean():7.3f} (prior {DEFAULT_PAULI_PRIOR[z][0]:.3f})"
              f"   b {params['b'][m].mean():5.3f} (prior {DEFAULT_PAULI_PRIOR[z][1]:.3f})"
              f"   |mu| {params['mu'][m].mean():.4f}   |Q| {params['quad'][m].mean():.4f}")

    if len(table) > 1:
        pp.rank_comparison(table, path=FIGDIR / "pauli_rank_comparison.png")
        print(f"wrote {FIGDIR / 'pauli_rank_comparison.png'}")
        print("\nMAE by cluster size (kJ/mol):")
        print(f"{'variant':<34}" + "".join(f"{t:<10}" for t in CLUSTERS))
        for lbl, row in table.items():
            print(f"{lbl:<34}" + "".join(f"{row[t]:<10.4f}" for t in CLUSTERS))

    # --- range separation vs the delta-learning cutoff ------------------------
    pair = predict(model, datasets["w3"], cfg.pauli.target, n_frames=400, with_pairs=True)
    r0 = _learned_disp_r0()
    pp.switching_functions(
        pair["pair_r"], pair["pair_e_ff"], pair["pair_e_corr"],
        disp_r0=r0,
        corr_window=(cfg.pauli.corr_r_on, cfg.pauli.corr_r_off),
        taper_window=(cfg.pauli.cutoff - cfg.pauli.taper_width, cfg.pauli.cutoff),
        path=FIGDIR / "pauli_switching.png",
    )
    print(f"wrote {FIGDIR / 'pauli_switching.png'}")

    r, ff, corr = pair["pair_r"], np.abs(pair["pair_e_ff"]), np.abs(pair["pair_e_corr"])
    print("\nrange separation vs delta-learning cutoff:")
    print(f"  dispersion learned Fermi midpoint r0 = {r0:.3f} A")
    print(f"  Pauli correction envelope            = {cfg.pauli.corr_r_on:.1f}-"
          f"{cfg.pauli.corr_r_off:.1f} A (fixed hyperparameter)")
    print(f"  gap where both are at full strength  = {r0:.2f}-{cfg.pauli.corr_r_on:.1f} A "
          f"({cfg.pauli.corr_r_on - r0:.2f} A wide)")
    inside = (r >= r0) & (r <= cfg.pauli.corr_r_on)
    print(f"  {inside.mean() * 100:.1f}% of inter-fragment pairs fall in that band, "
          f"carrying {ff[inside].sum() / max(ff.sum(), 1e-30) * 100:.1f}% of |E_FF| "
          f"and {corr[inside].sum() / max(corr.sum(), 1e-30) * 100:.1f}% of |dE|")
    for frac in (0.5, 0.9, 0.99):
        order = np.argsort(r)
        for name, mag in (("|E_FF|", ff), ("|dE|", corr)):
            if mag.sum() <= 0:
                continue
            c = np.cumsum(mag[order]) / mag.sum()
            print(f"  {name:>7} reaches {frac:.0%} of its total by r = "
                  f"{r[order][np.searchsorted(c, frac)]:.2f} A")

    # --- many-body expansion --------------------------------------------------
    mbe = many_body_table(model, datasets, PER_CLUSTER, target=cfg.pauli.target)
    print(f"\nmany-body expansion over {len(mbe['total_total'])} strongly-interacting "
          f"clusters:")
    print("cluster   n     total     2-body    3-body    4-body    5-body   |MB|/|tot|")
    for tag in ("w3", "w4", "w5"):
        m = mbe["tag"] == tag
        print(f"{tag:<10}{m.sum():<6}{mbe['total_total'][m].mean():<10.3f}"
              f"{mbe['total_two_body'][m].mean():<10.3f}{mbe['total_e3'][m].mean():<10.3f}"
              f"{mbe['total_e4'][m].mean():<10.3f}{mbe['total_e5'][m].mean():<10.3f}"
              f"{np.abs(mbe['total_many_body'][m] / mbe['total_total'][m]).mean() * 100:.2f}%")
    print("\nsource of the non-additivity (mean beyond-pairwise, kJ/mol):")
    print(f"{'cluster':<10}{'force field':<14}{'correction head':<18}{'corr share'}")
    for tag in ("w3", "w4", "w5"):
        m = mbe["tag"] == tag
        a = np.abs(mbe["ff_many_body"][m]).mean()
        b = np.abs(mbe["corr_many_body"][m]).mean()
        print(f"{tag:<10}{mbe['ff_many_body'][m].mean():<+14.4f}"
              f"{mbe['corr_many_body'][m].mean():<+18.3e}{b / max(a + b, 1e-30) * 100:.3f}%")
    x, y = mbe["ref"] - mbe["total_two_body"], mbe["total_many_body"]
    print(f"  model many-body vs (reference - model 2-body): "
          f"r = {np.corrcoef(y, x)[0, 1]:.4f}, slope = {np.polyfit(y, x, 1)[0]:.3f}")

    pp.many_body_panels(mbe, path=FIGDIR / "pauli_many_body.png")
    print(f"wrote {FIGDIR / 'pauli_many_body.png'}")

    np.savez(
        FIGDIR / "pauli_analysis.npz",
        **{f"mbe_{k}": v for k, v in mbe.items()},
        **{f"{t}_{k}": preds[t][k] for t in CLUSTERS for k in ("pred", "ref", "ff", "corr")},
        **{f"param_{k}": v for k, v in params.items()},
        pair_r=r, pair_e_ff=pair["pair_e_ff"], pair_e_corr=pair["pair_e_corr"],
        disp_r0=r0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
