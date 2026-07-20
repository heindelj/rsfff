"""Multi-target training for the charge-aware MLIP.

    rsfff-train-charge configs/h2o_h3o_charge.yaml

Fits the :class:`ChargeAwareModel` on energies, forces, dipoles, dipole
derivatives, and polarizabilities simultaneously, with the charge SCF inside the
loop and the isolated-species integer-charge anchors as an auxiliary loss. Every
loss term is reported separately (the doc's requirement) via a metrics dict.

Notes on the loop:
- ``positions.requires_grad_`` is set *before* the model call so the geometry
  cache carries the graph (forces, d mu/dR).
- The applied field starts at zeros with ``requires_grad=True`` every step; the
  field-route polarizability is ``d mu / dF`` through the SCF attach.
- ``dmu_dr_every`` strides the dipole-derivative term (3 extra double-backwards).
- MPS is not supported for the charge path (batched linalg + float64 gaps);
  device resolution falls back to CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..mlip import build_charge_model
from .config import Config, load_config
from .data import (
    load_datasets,
    load_isolated_species,
    load_reference_energies,
    split_indices,
)
from .loss import charge_model_loss, isolated_species_loss
from .train import _iter_minibatches


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # no MPS: batched linalg.solve/eigvalsh gaps


def _run_epoch(
    model, dataset, indices, config: Config, device, iso_batch, *, optimizer=None, seed=0
):
    """One pass over ``indices``. Trains if ``optimizer`` is given, else evaluates.

    Returns a dict of per-frame-weighted mean metrics (keys may be absent when a
    term was skipped on every step, e.g. strided dmu/dR at eval).
    """
    training = optimizer is not None
    model.train(training)
    tcfg = config.train
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_frames = 0
    step = 0

    for mb in _iter_minibatches(indices, tcfg.batch_size, shuffle=training, seed=seed):
        batch = dataset.flat_batch(mb).to(device)
        batch.positions.requires_grad_(True)
        field = torch.zeros(batch.n_systems, 3, device=device, requires_grad=True)
        dmu_w = tcfg.dmu_dr_weight if step % max(tcfg.dmu_dr_every, 1) == 0 else 0.0

        with torch.enable_grad():
            out = model(batch, field=field, freeze_charges=config.charge.freeze_charges)
            loss, metrics = charge_model_loss(
                out, batch,
                energy_weight=tcfg.energy_weight,
                force_weight=tcfg.force_weight,
                dipole_weight=tcfg.dipole_weight,
                dmu_dr_weight=dmu_w,
                alpha_weight=tcfg.alpha_weight,
                alpha_consistency_weight=tcfg.alpha_consistency_weight,
                q_l2_weight=tcfg.q_l2_weight,
                create_graph=training,
            )
            if tcfg.iso_weight > 0.0 and iso_batch is not None:
                iso_loss, iso_max = isolated_species_loss(
                    model, iso_batch, weight=tcfg.iso_weight
                )
                loss = loss + iso_loss
                metrics["iso_e_max"] = iso_max

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), tcfg.grad_clip
                )
            optimizer.step()

        bs = int(mb.shape[0])
        metrics["loss"] = float(loss.detach())
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v * bs
            counts[k] = counts.get(k, 0) + bs
        n_frames += bs
        step += 1

    return {k: sums[k] / max(counts[k], 1) for k in sums}


def _fmt(metrics: dict[str, float], keys) -> str:
    return "  ".join(f"{k} {metrics[k]:.3e}" for k in keys if k in metrics)


_LOG_KEYS = (
    "loss", "energy_mae", "force_mae", "dipole_mae", "dmu_dr_mae",
    "alpha_mae", "alpha_head_mae", "iso_e_max", "q_abs_mean",
    "scf_iters", "scf_residual",
)


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device)
    print(f"[{config.run_name}] device={device} dtype={config.dtype}")

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    neighbor_types = dataset.unique_atomic_numbers
    e0 = load_reference_energies(config.data.reference_energies, neighbor_types)
    print(f"loaded {len(dataset)} frames; species Z={neighbor_types}")

    iso_batch = None
    if config.data.isolated_species:
        iso_batch = load_isolated_species(
            config.data.isolated_species, dtype=torch.get_default_dtype()
        ).to(device)
        print(f"isolated-species anchors: {iso_batch.n_systems} systems")
    elif config.train.iso_weight > 0.0:
        raise ValueError("train.iso_weight > 0 but data.isolated_species is not set")

    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    model = build_charge_model(config.features, config.charge, neighbor_types, e0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.train.learning_rate, weight_decay=config.train.weight_decay,
    )

    ckpt_dir = Path(config.checkpoint_root) / config.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(config.train.epochs):
        t0 = time.time()
        tr = _run_epoch(model, dataset, train_idx, config, device, iso_batch,
                        optimizer=optimizer, seed=config.data.seed + epoch)
        do_eval = (epoch + 1) % config.train.eval_every == 0 or epoch == config.train.epochs - 1
        if do_eval and len(val_idx) > 0:
            va = _run_epoch(model, dataset, val_idx, config, device, iso_batch)
            print(f"epoch {epoch+1:4d}  train: {_fmt(tr, _LOG_KEYS)}  ({time.time()-t0:.1f}s)")
            print(f"            val:   {_fmt(va, _LOG_KEYS)}")
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {"model_state": model.state_dict(), "neighbor_types": neighbor_types,
                     "config_path": None, "epoch": epoch, "val_loss": best_val},
                    ckpt_dir / "best.pt",
                )
        else:
            print(f"epoch {epoch+1:4d}  train: {_fmt(tr, _LOG_KEYS)}  ({time.time()-t0:.1f}s)")

    print(f"done; best val loss {best_val:.4e}; checkpoints in {ckpt_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the charge-aware MLIP.")
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
