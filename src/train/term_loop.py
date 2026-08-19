"""The shared training loop for the standalone force-field terms.

``train_dispersion``, ``train_pauli`` and ``train_elec`` all fit one explicit physical term
plus a pairwise neural correction against one ``batch.eda[...]`` component, and all three had
the same forty-line epoch loop. Factored here at the third instance -- two is where a shared
abstraction is guessed at, three is where its shape is known.

What the terms keep for themselves is what actually differs: how the model is built, which
penalties the loss carries beyond the fit term, and which diagnostics are worth printing.
Those arrive as callbacks rather than as flags, so adding a term does not mean editing this
file.

**The one invariant worth stating loudly:** squared errors are divided by ``energy_scale``
*before* squaring, so the fit term is O(1) at an error of that size and every penalty weight
is a plain dimensionless number. During the dispersion work a penalty quoted in Hartree
against a squared error in Hartree^2 (~1e-7) came out 700x too large, and the optimizer
duly minimized the penalty while the MAE got worse. Keeping the normalization here means no
term can reintroduce that mistake by accident.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..ff.units import KJMOL_PER_HARTREE
from .train import _iter_minibatches


def eda_component_fit(out, batch, cfg):
    """The default fit term: one EDA component against ``out.energy``.

    Returns ``(loss, metrics, target)``. This is exactly what ``run_epoch`` did
    inline before a term arrived (the 1-body head) whose target is per *fragment*
    rather than per frame and whose output carries no ``energy_ff``.
    """
    if batch.eda is None or cfg.target not in batch.eda:
        raise ValueError(
            f"batch carries no eda['{cfg.target}'] label; the dataset must come from "
            f"scripts/parse_roundtrip.py or scripts/parse_qchem_eda.py (available: "
            f"{sorted(batch.eda) if batch.eda else 'none'})"
        )
    target = batch.eda[cfg.target]
    err = out.energy - target
    loss = (err / cfg.energy_scale).pow(2).mean()

    # Share of the *system* energy the correction supplies, not of the mean per-pair
    # magnitude. Those differ enormously once pair energies carry mixed signs: in
    # electrostatics the attractive and repulsive pairs largely cancel, so a correction
    # worth 0.1% of the mean |pair energy| can still move the total by 10 kJ/mol. The
    # per-pair version reads reassuringly small exactly when it should not.
    ff, corr = out.energy_ff.detach().abs(), out.energy_corr.detach().abs()
    metrics = {
        # kJ/mol throughout: the targets are ~1e-2 Ha and unreadable in Hartree.
        "mae": float(err.detach().abs().mean()) * KJMOL_PER_HARTREE,
        "rmse": float(err.detach().pow(2).mean().sqrt()) * KJMOL_PER_HARTREE,
        "ff_mae": float((out.energy_ff.detach() - target).abs().mean()) * KJMOL_PER_HARTREE,
        "corr_share": float((corr / (ff + corr).clamp(min=1e-30)).mean()),
    }
    return loss, metrics, target


def parameter_groups(model, weight_decay: float) -> list[dict]:
    """Optimizer groups, exempting modules that declare ``no_weight_decay``.

    The flag is set in exactly one place, :func:`rsfff.mlip.heads.zero_init_readout`, which is
    also where the reasoning and the measurements live: a block whose readout starts at zero
    gets *no loss gradient at all* on the first step, weight decay is then the only force
    acting on it, and it does not recover -- five such blocks in this model were measured going
    to zero over a staged fit, taking the many-body dispersion, the per-channel compliance and
    the quadrupole anisotropy with them, silently.

    Nothing else is exempt. Per-species tables, the featurizer and the plainly-initialized
    heads all keep decaying, because they have live gradients to push back with.
    """
    decayed, exempt = [], []
    seen: set[int] = set()
    for module in model.modules():
        if not getattr(module, "no_weight_decay", False):
            continue
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                exempt.append(p)
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            decayed.append(p)
    groups = [{"params": decayed, "weight_decay": float(weight_decay)}]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return groups


def build_scheduler(optimizer, train_cfg):
    """The per-epoch learning-rate schedule, or ``None`` for a fixed rate.

    ``cosine`` anneals to ``learning_rate * lr_final_factor`` over ``epochs``, stepped once
    per epoch rather than per step so the printed rate is the one the whole epoch ran at.

    The reason it exists is written up in :attr:`rsfff.train.config.TrainConfig.lr_schedule`:
    the split of ``fragment_energy`` between ``E_internal`` and ``E_atom`` is unlabeled, hence
    a flat direction, and along a flat direction a fixed step size does not converge -- it sets
    a diffusion amplitude. Annealing shrinks that amplitude toward the end of the stage. It
    does not make the direction any less flat, so it is a mitigation and not a fix; what makes
    the number stop moving is something that actually pins the split.
    """
    name = str(getattr(train_cfg, "lr_schedule", "none") or "none").lower()
    if name in ("", "none"):
        return None
    if name != "cosine":
        raise ValueError(
            f"unknown train.lr_schedule {name!r}; supported: 'none', 'cosine'"
        )
    final = float(getattr(train_cfg, "lr_final_factor", 0.05))
    if not 0.0 <= final <= 1.0:
        raise ValueError(
            f"train.lr_final_factor must be in [0, 1], got {final}; it is a *fraction* of "
            f"train.learning_rate, not an absolute rate"
        )
    # `T_max = epochs - 1`, not `epochs`: the schedule is read *before* each epoch runs and
    # stepped after, so with `T_max = epochs` the floor would be reached one step past the end
    # and the last epoch would run at ~1.3x it. Minus one makes the final epoch run at exactly
    # `learning_rate * lr_final_factor`, which is what the config field says it does.
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(train_cfg.epochs) - 1, 1),
        eta_min=float(train_cfg.learning_rate) * final,
    )


#: Below this Frobenius norm an ``equiv_reduce`` is treated as a deleted block rather than a
#: trained value, and :func:`warm_start` reinitializes it. Its initialization norm is
#: ``sqrt(equiv_channels)`` (5.66 at the default 32), and the observed dead value was a
#: denormal 2e-315, so anything in between separates the two cases cleanly.
_DEAD_NORM = 1.0e-6


def warm_start(model, path: str | None) -> None:
    """Load what fits from an earlier checkpoint, and say exactly what did not.

    Staged fits need this: the polarized level is meant to start from the frozen fit and the
    charge-transfer level from the polarized one, because each label is defined as a
    *difference* against the level below it. Fitting ``ct`` on top of a badly-fit ``pol``
    makes the label meaningless rather than merely harder.

    The load is deliberately non-strict, because a stage genuinely changes some shapes: the
    correction trunk's input widens when the electrostatic environment starts feeding the bond
    channel. A grown tensor is **zero-padded rather than reinitialized** -- the saved block is
    copied into the leading slice and the new columns are set to zero, so the new input starts
    inert and the model starts at exactly the checkpoint it was given.

    That distinction is worth more than it looks. Dropping the tensor and reinitializing it
    scrambles the *whole* shared trunk, not just the part the new input feeds: measured on the
    frozen-to-polarized step, a reinitialized ``pair_head.trunk.0.weight`` took ``elst_mae``
    from 0.46 to 3.2 kJ/mol and put -237 kJ/mol per fragment of intramolecular electrostatics
    back, because the per-pair range separation is read from the same trunk. Zero-padding
    leaves all of that untouched.

    Everything skipped or left at initialization is counted and printed. A stage reporting far
    more of either than the shape changes it introduced has loaded the wrong checkpoint.

    **An ``equiv_reduce`` at zero is refused rather than loaded.** It is not a trained
    value but a block that weight decay deleted, and loading it re-enters a deadlock that
    training cannot leave -- see ``_DEAD_NORM`` and :func:`rsfff.mlip.heads.zero_init_readout`.
    Every
    equivariant head now exempts itself from decay, so this only fires on checkpoints written
    before that fix; it fires loudly, because a silently dead head is what it is guarding
    against.

    **Zero-padding is right for a weight and wrong for a parameter table.** Growing the input
    of a linear layer should leave the new input inert, which zero does. Growing a per-species
    bias table -- ``cquad0_raw`` from one column to three when ``elec.anisotropic_cquad`` is
    switched on -- should *replicate* the trained value, because equal entries are what make
    the anisotropic head reduce to the isotropic one; zeroing gives ``softplus(0) + floor``
    for the new eigenvalues and jumps the model. Nothing in the staged configs changes that
    flag mid-chain for exactly this reason. If you ever need to, replicate rather than pad.
    """
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt.get("model_state", ckpt)
    current = model.state_dict()
    take, padded, dropped, revived = {}, [], [], []
    for k, v in saved.items():
        if k not in current:
            continue
        want = current[k]
        if k.endswith("equiv_reduce") and float(v.norm()) < _DEAD_NORM:
            # A channel reduction that reached zero is not a trained value, it is a deleted
            # block: `equiv_reduce == 0` zeroes its head's output, which zeroes the gradient
            # into the zero-initialized gate, which holds the gate at zero and keeps
            # `equiv_reduce`'s own gradient at zero. Loading it re-enters the deadlock, and no
            # amount of further training leaves it. Keep this stage's fresh initialization
            # instead -- the head restarts, which is strictly better than staying dead.
            # `rsfff.mlip.heads.zero_init_readout` documents how they got there.
            revived.append(f"{k} (norm {float(v.norm()):.3g})")
            continue
        if want.shape == v.shape:
            take[k] = v
        elif v.dim() == want.dim() and all(a <= b for a, b in zip(v.shape, want.shape)):
            grown = torch.zeros_like(want)
            grown[tuple(slice(0, n) for n in v.shape)] = v
            take[k] = grown
            padded.append(f"{k} {tuple(v.shape)}->{tuple(want.shape)}")
        else:
            dropped.append(f"{k} {tuple(v.shape)} vs {tuple(want.shape)}")
    revived_keys = {r.split(" ", 1)[0] for r in revived}
    missing = [k for k in current if k not in take and k not in revived_keys]
    model.load_state_dict(take, strict=False)
    parts = [f"warm start from {path}: loaded {len(take)}/{len(current)} tensors"]
    if padded:
        parts.append(f"{len(padded)} zero-padded ({', '.join(padded[:3])})")
    if dropped:
        parts.append(f"{len(dropped)} DROPPED, incompatible ({', '.join(dropped[:3])})")
    if revived:
        parts.append(
            f"{len(revived)} REINITIALIZED, saved value was a deleted block "
            f"({', '.join(revived[:3])})"
        )
    if missing:
        parts.append(f"{len(missing)} left at initialization")
    print("; ".join(parts), flush=True)


def run_epoch(
    model,
    dataset,
    indices,
    cfg,                       # the term's own config block (needs target, energy_scale)
    train_cfg,
    device,
    *,
    optimizer=None,
    seed: int = 0,
    fit_term=None,             # (out, batch, cfg) -> (Tensor, dict, target); default above
    penalties=None,            # (out, batch, cfg) -> dict[str, Tensor], extra loss terms
    diagnostics=None,          # (out, batch, target) -> dict[str, float]
    grad_positions: bool = False,
):
    """One pass over ``indices``. Trains if ``optimizer`` is given, else evaluates.

    Returns per-sample means of every metric, so train and validation lines are comparable
    regardless of how the last minibatch was sized.

    ``grad_positions`` turns on the autograd path a force term needs: positions become
    leaves and gradients stay enabled even during evaluation, because ``-dE/dR`` is itself
    computed by a backward pass. Terms without force labels leave it off and pay nothing.
    """
    training = optimizer is not None
    fit_term = fit_term if fit_term is not None else eda_component_fit
    model.train(training)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for mb in _iter_minibatches(indices, train_cfg.batch_size, shuffle=training, seed=seed):
        batch = dataset.flat_batch(mb).to(device)
        if grad_positions:
            batch.positions.requires_grad_(True)

        with torch.set_grad_enabled(training or grad_positions):
            out = model(batch)
            loss, fit_metrics, target = fit_term(out, batch, cfg)
            extra = penalties(out, batch, cfg) if penalties is not None else {}
            for value in extra.values():
                loss = loss + value

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if train_cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

        metrics = {"loss": float(loss.detach()), **fit_metrics}
        # Penalties are summed into the loss; without logging them the loss can be
        # dominated by a term nothing in the printed line accounts for.
        metrics.update({k: float(v.detach()) for k, v in extra.items()})
        if diagnostics is not None:
            metrics.update(diagnostics(out, batch, target))

        n = len(mb)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v * n
            counts[k] = counts.get(k, 0) + n

    return {k: sums[k] / max(counts[k], 1) for k in sums}


def fmt(metrics: dict[str, float], keys) -> str:
    return "  ".join(f"{k} {metrics[k]:.4g}" for k in keys if k in metrics)


def fit(
    model,
    dataset,
    config,
    cfg,                       # the term's config block
    device,
    train_idx,
    val_idx,
    *,
    log_keys,
    fit_term=None,
    penalties=None,
    diagnostics=None,
    grad_positions: bool = False,
    report=None,               # (model) -> None, printed once at the end
    after_warm_start=None,     # (model) -> None, between the warm start and the optimizer
):
    """Baseline evaluation, then the training loop with best-checkpoint saving.

    The untrained line is printed before anything is optimized on purpose: it is the number
    every later epoch has to beat for the learned parts to be earning their place, and for
    these terms the physical backbone alone is often already close. With ``train.init_from``
    set it is the *warm-started* baseline, which is the number a staged fit actually starts
    from.

    ``after_warm_start`` runs in the one window where a stage can read the state it inherited
    and act on it: the checkpoint is loaded, and the optimizer does not exist yet, so freezing
    a parameter there actually keeps it out of the optimizer rather than merely zeroing its
    gradient. :meth:`rsfff.train.train_unified.AnchorTerms.snapshot_frozen_level` is the
    caller, and both things it does need exactly this window.
    """
    def epoch(indices, **kw):
        return run_epoch(
            model, dataset, indices, cfg, config.train, device,
            fit_term=fit_term, penalties=penalties, diagnostics=diagnostics,
            grad_positions=grad_positions, **kw
        )

    warm_start(model, config.train.init_from)
    if after_warm_start is not None:
        after_warm_start(model)

    base = epoch(val_idx)
    print(f"untrained: {fmt(base, log_keys)}", flush=True)

    groups = parameter_groups(model, config.train.weight_decay)
    optimizer = torch.optim.Adam(groups, lr=config.train.learning_rate)
    if len(groups) > 1:
        n = sum(p.numel() for p in groups[1]["params"])
        print(
            f"weight decay {config.train.weight_decay} on all but {n} exempt parameters",
            flush=True,
        )
    scheduler = build_scheduler(optimizer, config.train)
    ckpt_dir = Path(config.checkpoint_root) / config.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for ep in range(config.train.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr = epoch(train_idx, optimizer=optimizer, seed=config.data.seed + ep)
        if scheduler is not None:
            scheduler.step()
        do_eval = (
            (ep + 1) % config.train.eval_every == 0 or ep == config.train.epochs - 1
        )
        suffix = f", lr {lr:.2e}" if scheduler is not None else ""
        line = (
            f"epoch {ep+1:4d}  train: {fmt(tr, log_keys)}  "
            f"({time.time()-t0:.1f}s{suffix})"
        )
        if do_eval and len(val_idx) > 0:
            va = epoch(val_idx)
            print(line, flush=True)
            print(f"            val:   {fmt(va, log_keys)}", flush=True)
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {"model_state": model.state_dict(),
                     "neighbor_types": dataset.unique_atomic_numbers,
                     "config": config, "epoch": ep, "val_loss": best_val},
                    ckpt_dir / "best.pt",
                )
        else:
            print(line, flush=True)

    if report is not None:
        report(model)
    print(f"done; best val loss {best_val:.4e}; checkpoints in {ckpt_dir}", flush=True)
    return best_val
