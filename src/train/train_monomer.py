"""Phase-1 training for the frozen monomer stack.

    rsfff-train-monomer configs/monomer_h2o_h3o_oh.yaml

Fits :class:`rsfff.mlip.MonomerModel` on energies, forces, dipoles, dipole derivatives, and
polarizabilities, with the isolated-atom reference states as free-atom anchors. Everything is a
single ordinary autograd graph: the SQE solve is an exact linear solve, so there is no inner
iteration and nothing to diverge.

This is the phase whose output gets **frozen permanently** (docs/mixture_of_diabatic_
embeddings.md §4.3): all of the model's asymptotic guarantees -- dissociation limits, IE-EA
differences, the gate's Coulomb bias -- rest on the checkpoint this script produces. Validate
it before building on it.

Notes:
- ``positions.requires_grad_`` is set before the model call (forces, d mu/dR).
- The polarizability is the model's analytic output; no applied-field tensor is needed.
- ``dmu_dr_every`` strides the dipole-derivative term (3 extra double-backwards).
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

from ..mlip import AtomicStateReference, DiabaticStateLibrary, build_monomer_model
from .config import Config, load_config
from .data import (
    load_atomic_reference_batch,
    load_datasets,
    load_reference_energies,
    split_indices,
)
from .loss import atomic_reference_loss, monomer_loss
from .train import _iter_minibatches
from .train_eem import resolve_device


def _run_epoch(
    model, dataset, indices, config: Config, device, anchors, *, optimizer=None, seed=0
):
    """One pass over ``indices``. Trains if ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    tcfg = config.train
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = 0

    for mb in _iter_minibatches(indices, tcfg.batch_size, shuffle=training, seed=seed):
        batch = dataset.flat_batch(mb).to(device)
        batch.positions.requires_grad_(True)
        dmu_w = tcfg.dmu_dr_weight if step % max(tcfg.dmu_dr_every, 1) == 0 else 0.0

        with torch.enable_grad():
            out = model(batch)
            loss, metrics = monomer_loss(
                out, batch,
                energy_weight=tcfg.energy_weight,
                force_weight=tcfg.force_weight,
                dipole_weight=tcfg.dipole_weight,
                dmu_dr_weight=dmu_w,
                alpha_weight=tcfg.alpha_weight,
                q_l2_weight=tcfg.q_l2_weight,
                create_graph=training,
            )
            if anchors is not None and tcfg.atomic_ref_weight > 0.0:
                anchor_batch, states = anchors
                a_loss, a_metrics = atomic_reference_loss(
                    model, anchor_batch, states,
                    energy_weight=tcfg.atomic_ref_weight,
                    alpha_weight=tcfg.free_alpha_weight,
                    unbound_weight=tcfg.unbound_weight,
                )
                loss = loss + a_loss
                metrics.update(a_metrics)

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
        step += 1

    return {k: sums[k] / max(counts[k], 1) for k in sums}


def _fmt(metrics: dict[str, float], keys) -> str:
    return "  ".join(f"{k} {metrics[k]:.3e}" for k in keys if k in metrics)


_LOG_KEYS = (
    "loss", "energy_mae", "force_mae", "dipole_mae", "dmu_dr_mae", "alpha_mae",
    "atomic_e_max", "atomic_alpha_max", "q_abs_mean", "s_mean", "p_abs_max", "eta_min",
)


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    print(f"[{config.run_name}] device={device} dtype={config.dtype}")

    if not config.data.diabatic_states:
        raise ValueError(
            "the monomer stack needs data.diabatic_states: the channel graph comes from the "
            "diabatic assignment, not from a distance rule (see rsfff.mlip.diabats)"
        )
    library = DiabaticStateLibrary.from_yaml(config.data.diabatic_states)
    dataset = load_datasets(
        config.data.path, dtype=torch.get_default_dtype(), library=library
    )
    neighbor_types = dataset.unique_atomic_numbers
    e0 = load_reference_energies(config.data.reference_energies, neighbor_types)
    print(
        f"loaded {len(dataset)} frames; species Z={neighbor_types}; "
        f"diabatic states: {library.keys}"
    )

    anchors = None
    states = None
    if config.data.atomic_reference_states:
        states = AtomicStateReference.from_json(
            config.data.atomic_reference_states,
            neighbor_types,
            dtype=torch.get_default_dtype(),
        )
        anchor_batch = load_atomic_reference_batch(states, neighbor_types).to(device)
        states = states.to(device)
        anchors = (anchor_batch, states)
        n_unbound = int((~states.bound).sum())
        print(
            f"atomic reference states: {len(states)} ({states.symbols})"
            + (f"; {n_unbound} unbound, weight {config.train.unbound_weight}"
               if n_unbound else "")
        )
    elif config.train.atomic_ref_weight > 0.0:
        raise ValueError(
            "train.atomic_ref_weight > 0 but data.atomic_reference_states is not set"
        )

    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    model = build_monomer_model(
        config.features, config.monomer, config.sqe, neighbor_types, e0, states
    ).to(device)
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
        tr = _run_epoch(model, dataset, train_idx, config, device, anchors,
                        optimizer=optimizer, seed=config.data.seed + epoch)
        do_eval = (
            (epoch + 1) % config.train.eval_every == 0 or epoch == config.train.epochs - 1
        )
        if do_eval and len(val_idx) > 0:
            va = _run_epoch(model, dataset, val_idx, config, device, anchors)
            print(f"epoch {epoch+1:4d}  train: {_fmt(tr, _LOG_KEYS)}  ({time.time()-t0:.1f}s)")
            print(f"            val:   {_fmt(va, _LOG_KEYS)}")
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {"model_state": model.state_dict(), "neighbor_types": neighbor_types,
                     "epoch": epoch, "val_loss": best_val},
                    ckpt_dir / "best.pt",
                )
        else:
            print(f"epoch {epoch+1:4d}  train: {_fmt(tr, _LOG_KEYS)}  ({time.time()-t0:.1f}s)")

    print(f"done; best val loss {best_val:.4e}; checkpoints in {ckpt_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train the Phase-1 frozen monomer stack (reference states + SQE)."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
