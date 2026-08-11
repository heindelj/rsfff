"""Fit the intermolecular Pauli repulsion term against an EDA component.

    python -m rsfff.train.train_pauli configs/eda_water_pauli.yaml

Trains :class:`rsfff.ff.pauli.SlaterPauli` -- Slater-damped multipolar repulsion with
learned Pauli multipoles, plus a term-specific short-range pair correction -- directly on
``batch.eda["mod_pauli"]`` from the Q-Chem ALMO-EDA water clusters. Standalone, like
``train_dispersion``: nothing here touches the Phase-1 monomer/mixture stack.

The loss is

    MSE((E_pred - E_target)/energy_scale)  +  corr_l2_weight * mean((dE/energy_scale)^2)

Normalizing by ``energy_scale`` before squaring is not cosmetic. During the dispersion work
a penalty weight quoted in Hartree against a squared error in Hartree^2 (~1e-7) came out
700x too large and the optimizer simply minimized the penalty while the MAE got *worse*.
Keeping every weight dimensionless makes that class of mistake visible.

There are no forces in this data (EDA single points carry no nuclear gradient), so the fit
is energy-only. Suggested curriculum, each stage localizing where the residual lives:

  1. everything frozen (``learn_q/learn_b/environment_q: false, max_rank: 0,
     correction: false``) -- an evaluation, not a fit. Expect ~7 kJ/mol MAE on w2 at
     correlation 0.99, the pyCMM priors' natural accuracy.
  2. per-species q only (``learn_q: true``) -- two parameters. Expect ~2 kJ/mol; the priors
     have the shape right and the scale off by ~18%.
  3. add per-species b.
  4. add the q environment MLP.
  5. ``max_rank: 1`` -- enable the dipole head (bit-identical at init, so this is a pure
     addition).
  6. add the pair correction.

Watch ``ff_mae`` against ``mae``. Dispersion ended with its correction carrying 1.7% of the
energy; if Pauli's correction takes a large share, the missing piece is most likely the
quadrupole (pyCMM's ``K_quad_pauli`` is non-zero for both elements) and the answer is to
port rank 2 in ``rsfff.ff.multipole`` rather than to grow the network.

Caveat on the target: ``eda_mod_pauli`` is Q-Chem EDA2's ``cls_pauli - disp``, so it pairs
cleanly with the separately fit dispersion model, but the split between the two is a
property of the decomposition rather than of nature. The fitted Pauli multipoles are not
transferable to a different EDA scheme.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# macOS: torch's bundled libomp + conda's llvm-openmp abort with OMP Error #15
# unless this is set before the first OpenMP runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from ..features.features import FlatLambdaSOAPFeaturizer  # noqa: E402
from ..ff.multipole import irrep2_to_spherical  # noqa: E402
from ..ff.pauli import (  # noqa: E402
    PauliModel,
    PauliMultipoleHeads,
    SlaterPauli,
    build_pauli_priors,
)
from ..ff.units import KJMOL_PER_HARTREE  # noqa: E402
from ..mlip.pair_heads import PairEnergyHead  # noqa: E402
from .config import Config, load_config  # noqa: E402
from .data import load_datasets, split_indices  # noqa: E402
from .train import _iter_minibatches  # noqa: E402
from .train_eem import resolve_device  # noqa: E402


def build_pauli_model(features_cfg, pcfg, neighbor_types) -> PauliModel:
    """Featurizer + :class:`SlaterPauli` at the per-species priors, as one module."""
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        selected_lambdas=features_cfg.selected_lambdas,
        backend=features_cfg.backend,
        density_channels=features_cfg.density_channels,
    )
    p0 = featurizer.feature_dims[0]
    p1 = featurizer.feature_dims.get(1)
    p2 = featurizer.feature_dims.get(2)
    n_species = len(neighbor_types)
    if pcfg.max_rank >= 1 and pcfg.learn_dipole and p1 is None:
        raise ValueError(
            "pauli.max_rank >= 1 emits dipoles, which need lambda=1 features; set "
            "features.selected_lambdas: [0, 1, 2] (or pauli.max_rank: 0)"
        )
    log_q, log_b, mu_scale, quad_scale = build_pauli_priors(neighbor_types)
    params = PauliMultipoleHeads(
        p0, p1, n_species,
        log_q_prior=log_q, log_b_prior=log_b, dipole_scale=mu_scale,
        p2=p2, quad_scale=quad_scale,
        irrep2_to_spherical=irrep2_to_spherical(featurizer.backend.irrep6_to_voigt()),
        emb_dim=pcfg.emb_dim, hidden=pcfg.hidden, depth=pcfg.depth,
        equiv_channels=pcfg.equiv_channels, max_rank=pcfg.max_rank,
        learn_q=pcfg.learn_q, environment_q=pcfg.environment_q,
        learn_b=pcfg.learn_b, environment_b=pcfg.environment_b,
        learn_dipole=pcfg.learn_dipole, learn_quadrupole=pcfg.learn_quadrupole,
    )
    correction = None
    if pcfg.correction:
        if pcfg.corr_r_off > features_cfg.cutoff:
            raise ValueError(
                f"pauli.corr_r_off ({pcfg.corr_r_off}) exceeds the feature cutoff "
                f"({features_cfg.cutoff}); past it the pair head reads features whose atoms "
                f"cannot see each other"
            )
        correction = PairEnergyHead(
            p0, n_species,
            emb_dim=pcfg.emb_dim, hidden=pcfg.corr_hidden, depth=pcfg.corr_depth,
            n_radial=pcfg.corr_n_radial, r_on=pcfg.corr_r_on, r_off=pcfg.corr_r_off,
            energy_scale=pcfg.corr_energy_scale,
        )
    pauli = SlaterPauli(
        params, correction,
        cutoff=pcfg.cutoff, taper_width=pcfg.taper_width,
        max_rank=pcfg.max_rank, inter_only=pcfg.inter_only,
    )
    return PauliModel(featurizer, pauli, intra_fragment=pcfg.intra_fragment_features)


def _run_epoch(model, dataset, indices, config: Config, device, *, optimizer=None, seed=0):
    """One pass over ``indices``. Trains if ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    pcfg, tcfg = config.pauli, config.train
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for mb in _iter_minibatches(indices, tcfg.batch_size, shuffle=training, seed=seed):
        batch = dataset.flat_batch(mb).to(device)
        if batch.eda is None or pcfg.target not in batch.eda:
            raise ValueError(
                f"batch carries no eda['{pcfg.target}'] label; the dataset must come from "
                f"scripts/parse_qchem_eda.py (available: "
                f"{sorted(batch.eda) if batch.eda else 'none'})"
            )
        target = batch.eda[pcfg.target]

        with torch.set_grad_enabled(training):
            out = model(batch)
            err = out.energy - target
            scale = pcfg.energy_scale
            loss = (err / scale).pow(2).mean()
            if pcfg.corr_l2_weight > 0.0:
                loss = loss + pcfg.corr_l2_weight * (out.e_pair_corr / scale).pow(2).mean()

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if tcfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()

        n = len(mb)
        ff_mag = out.e_pair_ff.detach().abs().mean().clamp(min=1e-30)
        metrics = {
            "loss": float(loss.detach()),
            # kJ/mol throughout: the targets are ~1e-2 Ha and unreadable in Hartree.
            "mae": float(err.detach().abs().mean()) * KJMOL_PER_HARTREE,
            "rmse": float(err.detach().pow(2).mean().sqrt()) * KJMOL_PER_HARTREE,
            "ff_mae": float((out.energy_ff.detach() - target).abs().mean()) * KJMOL_PER_HARTREE,
            "corr_frac": float(out.e_pair_corr.detach().abs().mean() / ff_mag),
        }
        if out.mu is not None:
            metrics["mu"] = float(out.mu.detach().norm(dim=-1).mean())
        if out.quad_s is not None:
            metrics["quad"] = float(out.quad_s.detach().norm(dim=-1).mean())
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v * n
            counts[k] = counts.get(k, 0) + n

    return {k: sums[k] / max(counts[k], 1) for k in sums}


def _fmt(metrics: dict[str, float], keys) -> str:
    return "  ".join(f"{k} {metrics[k]:.4g}" for k in keys if k in metrics)


_LOG_KEYS = ("loss", "mae", "rmse", "ff_mae", "corr_frac", "mu", "quad")


def _report_params(model, neighbor_types):
    with torch.no_grad():
        q = (model.params.log_q_prior + model.params.d_log_q).exp()
        b = (model.params.log_b_prior + model.params.d_log_b).exp()
    print(f"per-species q (e):       {dict(zip(neighbor_types, q.tolist()))}")
    print(f"per-species b (1/bohr):  {dict(zip(neighbor_types, b.tolist()))}")


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    pcfg = config.pauli
    print(f"[{config.run_name}] device={device} dtype={config.dtype} target=eda_{pcfg.target}")

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if pcfg.inter_only and not dataset.has_fragments:
        raise ValueError(
            "the Pauli term is defined over inter-fragment pairs, but the dataset has no "
            "fragment partition; the extxyz needs a `fragment_idx` column"
        )
    neighbor_types = dataset.unique_atomic_numbers
    print(f"loaded {len(dataset)} frames; species Z={neighbor_types}")

    model = build_pauli_model(config.features, pcfg, neighbor_types).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"max_rank {pcfg.max_rank} (dipoles={'on' if pcfg.max_rank >= 1 and pcfg.learn_dipole else 'off'}), "
        f"fit term 1.0 at {pcfg.energy_scale * KJMOL_PER_HARTREE:.3g} kJ/mol error"
    )

    # Baseline before any training: the Slater backbone at the pyCMM priors, which is the
    # number every later epoch has to beat for the learned parts to be earning their place.
    base = _run_epoch(model, dataset, val_idx, config, device)
    print(f"untrained: {_fmt(base, _LOG_KEYS)}")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.train.learning_rate, weight_decay=config.train.weight_decay,
    )
    ckpt_dir = Path(config.checkpoint_root) / config.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(config.train.epochs):
        t0 = time.time()
        tr = _run_epoch(model, dataset, train_idx, config, device,
                        optimizer=optimizer, seed=config.data.seed + epoch)
        do_eval = (
            (epoch + 1) % config.train.eval_every == 0 or epoch == config.train.epochs - 1
        )
        line = f"epoch {epoch+1:4d}  train: {_fmt(tr, _LOG_KEYS)}  ({time.time()-t0:.1f}s)"
        if do_eval and len(val_idx) > 0:
            va = _run_epoch(model, dataset, val_idx, config, device)
            print(line)
            print(f"            val:   {_fmt(va, _LOG_KEYS)}")
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {"model_state": model.state_dict(), "neighbor_types": neighbor_types,
                     "config": config, "epoch": epoch, "val_loss": best_val},
                    ckpt_dir / "best.pt",
                )
        else:
            print(line)

    _report_params(model, neighbor_types)
    print(f"done; best val loss {best_val:.4e}; checkpoints in {ckpt_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit Slater-damped multipolar Pauli repulsion to an EDA component."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
