"""Single-stage energy + force training for the total-energy MLIP.

    rsfff-train configs/h2o_energy.yaml

Loads a labeled extxyz dataset, references each molecular energy against the
open-shell isolated-atom energies (``data/atomic_references.json``), and fits the
``EnergyModel`` with an Adam loop on a joint energy + force loss. Forces come from
autograd of the predicted energy w.r.t. positions.
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

from ..mlip import build_model
from .config import Config, load_config
from .data import load_extxyz, load_reference_energies, split_indices
from .loss import compute_forces, energy_force_loss


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _iter_minibatches(indices: torch.Tensor, batch_size: int, *, shuffle: bool, seed: int):
    idx = indices
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = indices[torch.randperm(indices.shape[0], generator=g)]
    for start in range(0, idx.shape[0], batch_size):
        yield idx[start : start + batch_size]


def _run_epoch(model, dataset, indices, config, device, *, optimizer=None, seed=0):
    """One pass over ``indices``. Trains if ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    tcfg = config.train
    totals = torch.zeros(3)  # loss, energy_mae, force_mae (weighted by batch size)
    n_frames = 0

    for mb in _iter_minibatches(indices, tcfg.batch_size, shuffle=training, seed=seed):
        batch = dataset.flat_batch(mb).to(device)
        batch.positions.requires_grad_(True)
        # Forces need a graph even at eval; keep create_graph only while training.
        with torch.enable_grad():
            energy = model(batch)
            forces = compute_forces(energy, batch.positions, create_graph=training)
            loss, e_mae, f_mae = energy_force_loss(
                energy, forces, batch,
                energy_weight=tcfg.energy_weight, force_weight=tcfg.force_weight,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), tcfg.grad_clip
                )
            optimizer.step()
        bs = int(mb.shape[0])
        totals += torch.tensor([loss.item(), e_mae.item(), f_mae.item()]) * bs
        n_frames += bs

    return (totals / max(n_frames, 1)).tolist()


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device)
    print(f"[{config.run_name}] device={device} dtype={config.dtype}")

    dataset = load_extxyz(config.data.path, dtype=torch.get_default_dtype())
    neighbor_types = dataset.unique_atomic_numbers
    e0 = load_reference_energies(config.data.reference_energies, neighbor_types)
    print(f"loaded {len(dataset)} frames; species Z={neighbor_types}")
    print(f"reference energies (Ha): "
          f"{{{', '.join(f'{z}:{e:.4f}' for z, e in zip(neighbor_types, e0.tolist()))}}}")

    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    model = build_model(config.features, config.mlip, neighbor_types, e0).to(device)
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
        tr = _run_epoch(model, dataset, train_idx, config, device,
                        optimizer=optimizer, seed=config.data.seed + epoch)
        do_eval = (epoch + 1) % config.train.eval_every == 0 or epoch == config.train.epochs - 1
        if do_eval and len(val_idx) > 0:
            va = _run_epoch(model, dataset, val_idx, config, device)
            print(f"epoch {epoch+1:4d}  train loss {tr[0]:.4e}  "
                  f"E_mae {tr[1]:.4e}  F_mae {tr[2]:.4e}  |  "
                  f"val loss {va[0]:.4e}  E_mae {va[1]:.4e}  F_mae {va[2]:.4e}  "
                  f"({time.time()-t0:.1f}s)")
            if va[0] < best_val:
                best_val = va[0]
                torch.save(
                    {"model_state": model.state_dict(), "neighbor_types": neighbor_types,
                     "epoch": epoch, "val_loss": best_val},
                    ckpt_dir / "best.pt",
                )
        else:
            print(f"epoch {epoch+1:4d}  train loss {tr[0]:.4e}  "
                  f"E_mae {tr[1]:.4e}  F_mae {tr[2]:.4e}  ({time.time()-t0:.1f}s)")

    print(f"done; best val loss {best_val:.4e}; checkpoints in {ckpt_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the total-energy MLIP.")
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
