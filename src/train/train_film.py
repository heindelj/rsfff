"""Fit the film model: two data streams, one optimizer, no freeze.

    python -m rsfff.train.train_film configs/water_film.yaml

Mirrors :mod:`rsfff.train.train_expert` for the ``rsfff.ff.film`` generation. The streams and
their labels are the same -- the model behind them changed:

======================  =======  ==================================================
stream                  env      supervises
======================  =======  ==================================================
fragment views          ``= 0``  ``fragment_energy`` (Morse + angle + intra classical),
                                 fragment dipole and quadrupole (the permanent heads)
the dedicated monomers  ``= 0``  the same, plus molecular polarizability (the response
                                 family) and true one-body forces
clusters                live     ``eda_cls_elec``, ``eda_mod_pauli``, ``eda_disp``,
                                 ``eda_pol + eda_ct``, cluster forces
======================  =======  ==================================================

What to watch, in order of what will bite
-----------------------------------------
``bond_var``
    the geometry-independence regularizer's argument: how far the bonded parameters wander
    from their per-type tables across the batch. Rising means the Morse "parameters" are
    becoming a disguised pair energy; the per-parameter ``std`` diagnostics say which one.
``env_*``
    per-quantity ``|theta - theta_0|``. The film analogue of v4's env sector; same reading.
``ind_mae`` against ``env_bond_*``
    the split of induction between the multipole relaxation and the bonded-parameter shift
    (the learned analogue of pyCMM's field-dependent Morse). Both are physical; the ratio is
    what has to stay honest.
``q_res``
    the exact charge projection's residual. This is an architecture check, not a fit metric:
    anything above 1e-12 is a bug.
``cg_ind`` / ``cg_fail``
    as in v4: a climbing count is the early warning, ``cg_fail`` must stay 0.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from ..ff.units import KJMOL_PER_HARTREE
from ..mlip.heads import env_parameters
from .build_film import build_film_model
from .config import Config, load_config, stage_config
from .data import (
    fragment_view,
    load_cluster_datasets,
    load_reference_energies,
    split_indices_grouped,
)
from .loss import (
    compute_forces,
    fragment_multipole_loss,
    fragment_polarizability_loss,
    onebody_anchor_loss,
)
from .term_loop import fit
from .train_eem import resolve_device
from .train_expert import load_anchor_datasets

#: Channel -> the ``eda_*`` header it is fitted against. Identical to v4's mapping.
_TARGETS = {"elst": "cls_elec", "pauli": "mod_pauli", "disp": "disp"}
_INDUCTION_COMPONENTS = ("pol", "ct")

_LOG_KEYS = (
    "loss", "ob_mae", "elst_mae", "pauli_mae", "disp_mae", "ind_mae", "e_tot_mae",
    "f_clu", "frag_mae", "dip_mae", "quad_mae", "alpha_mae", "f_mae",
    "bonded", "bond_var", "q_res",
    "r0_elst", "r0_pauli", "r0_disp",
    "env_norm", "env_c6", "env_eta", "env_bond_d", "env_bond_r_eq",
    "cg_ind", "cg_fail",
)


def film_fit(out, batch, cfg: Config, *, training: bool = True, with_forces: bool = True):
    """The cluster-stream loss: four EDA channels, the (default-off) total, and the forces.

    ``fragment_energy`` is reported (``ob_mae``) and not fitted here, for v4's reason: on a
    cluster batch it is exactly the same function of the same weights as on the fragment-view
    stream.
    """
    x = cfg.film
    metrics = {}
    loss = batch.energy.new_zeros(())

    ob_err = out.fragment_energy - batch.fragment_energy
    metrics["ob_mae"] = float(ob_err.detach().abs().mean()) * KJMOL_PER_HARTREE
    metrics["bonded"] = float(out.energy_bonded.detach().mean()) * KJMOL_PER_HARTREE
    metrics["env_norm"] = float(out.env_norm.detach().mean())
    for name, shift in out.env_shift.items():
        metrics[f"env_{name}"] = float(shift.detach().mean())

    weights = {"elst": x.elst_weight, "pauli": x.pauli_weight, "disp": x.disp_weight}
    for name, key in _TARGETS.items():
        if key not in (batch.eda or {}):
            raise KeyError(
                f"the dataset has no eda_{key} label, which the {name} channel fits; "
                f"available: {sorted(batch.eda or {})}"
            )
        err = out.interaction[name] - batch.eda[key]
        loss = loss + weights[name] * (err / x.energy_scale).pow(2).mean()
        metrics[f"{name}_mae"] = float(err.detach().abs().mean()) * KJMOL_PER_HARTREE

    if x.induction:
        missing = [k for k in _INDUCTION_COMPONENTS if k not in (batch.eda or {})]
        if missing:
            raise KeyError(
                f"induction is fitted against eda_pol + eda_ct and the dataset has no "
                f"{', '.join('eda_' + m for m in missing)}"
            )
        target = sum(batch.eda[k] for k in _INDUCTION_COMPONENTS)
        err = out.interaction["induction"] - target
        loss = loss + x.induction_weight * (err / x.energy_scale).pow(2).mean()
        metrics["ind_mae"] = float(err.detach().abs().mean()) * KJMOL_PER_HARTREE

    if batch.energy is not None:
        e_err = out.energy - batch.energy
        metrics["e_tot_mae"] = float(e_err.detach().abs().mean()) * KJMOL_PER_HARTREE
        if x.total_energy_weight > 0.0:
            loss = loss + x.total_energy_weight * (e_err / x.energy_scale).pow(2).mean()

    if x.force_weight > 0.0 and with_forces:
        if batch.forces is None:
            raise ValueError("film.force_weight > 0 but the dataset carries no forces")
        if not batch.positions.requires_grad:
            raise ValueError(
                "film.force_weight > 0 but batch.positions is not a leaf; the training "
                "loop must be entered with grad_positions=True"
            )
        forces = compute_forces(out.energy, batch.positions, create_graph=training)
        f_err = (forces - batch.forces) / x.force_scale
        loss = loss + x.force_weight * f_err.pow(2).sum(-1).mean()
        metrics["f_clu"] = float((forces - batch.forces).detach().abs().mean())

    if out.solver:
        n_iter, converged, pd_fail = out.solver["ind"]
        metrics["cg_ind"] = float(n_iter)
        metrics["cg_fail"] = float((~converged).sum() + pd_fail.sum())
    return loss, metrics, batch.fragment_energy


def strided_fit_term(model, force_every: int = 1):
    """:func:`film_fit` with the cluster force term applied every k-th training step."""
    every = max(int(force_every), 1)
    state = {"step": 0}

    def term(out, batch, cfg):
        training = bool(model.training)
        if training:
            state["step"] += 1
        with_forces = (not training) or state["step"] % every == 0
        return film_fit(out, batch, cfg, training=training, with_forces=with_forces)

    return term


def _bonded_variance(out) -> torch.Tensor:
    """Mean squared feature-dependent deviation of the bonded parameters (theta_0 branch)."""
    d = out.parameters.bonded0.delta_iso
    return d.pow(2).mean() if d is not None and d.numel() else out.energy.new_zeros(())


class FilmStreams:
    """The isolated streams plus the film model's penalties.

    The v4 ``IsolatedStreams`` restated for the film output: same fragment-view and monomer
    anchor streams (minibatched -- the fixed 500-frame anchor was once ~95% of wall time),
    same r0 barriers, the env L1 acting on the film's per-quantity shifts, plus the new
    **bonded variance** regularizer.
    """

    def __init__(
        self,
        model,
        device,
        *,
        fragment_dataset,
        fragment_batch_size: int,
        anchor_datasets=(),
        anchor_batch_size: int = 32,
        anchor_force_every: int = 5,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.fragments = fragment_dataset
        self.fragment_batch_size = int(fragment_batch_size)
        self.anchors = list(anchor_datasets)
        self.anchor_batch_size = int(anchor_batch_size)
        self.anchor_force_every = max(int(anchor_force_every), 1)
        self.generator = torch.Generator().manual_seed(int(seed))
        self._step = 0
        self._metrics: dict[str, float] = {}

    def _draw(self, dataset, size, training, *, grad_positions: bool = False):
        n = len(dataset)
        if n == 0:
            return None
        size = min(size, n)
        idx = (
            torch.randperm(n, generator=self.generator)[:size] if training
            else torch.arange(size)
        )
        batch = dataset.flat_batch(idx).to(self.device)
        if grad_positions:
            batch.positions.requires_grad_(True)
        return batch, idx

    def penalties(self, out, batch, cfg: Config):
        x = cfg.film
        extra: dict = {}
        self._metrics = {}

        # --- range separation (v4 semantics: barrier + log-space spread) ------------------
        if x.r0_weight > 0.0:
            inter = ~out.is_intra
            if bool(inter.any()):
                extra["r0"] = x.r0_weight * torch.stack([
                    (out.r0_pair[name][inter] - floor[inter].exp()).clamp(min=0.0).mean()
                    for name, floor in out.log_r0_prior_pair.items()
                ]).sum()
        if x.r0_spread_weight > 0.0:
            extra["r0_spread"] = x.r0_spread_weight * torch.stack([
                (out.r0[name].log() - prior).pow(2).mean()
                for name, prior in out.log_r0_prior.items()
            ]).sum()

        # --- the environment penalty -------------------------------------------------------
        unknown = set(x.env_penalty_weights) - set(out.env_shift)
        if unknown:
            raise KeyError(
                f"film.env_penalty_weights names {sorted(unknown)}, which the model does "
                f"not report; available: {sorted(out.env_shift)}"
            )
        for name, shift in out.env_shift.items():
            weight = float(x.env_penalty_weights.get(name, x.env_penalty_weight))
            if weight > 0.0:
                extra[f"env_{name}_pen"] = weight * shift.mean()

        # --- the geometry-independence regularizer ------------------------------------------
        if x.bonded_variance_weight > 0.0:
            extra["bond_var"] = x.bonded_variance_weight * _bonded_variance(out)

        # --- the fragment-view stream ---------------------------------------------------------
        training = bool(self.model.training)
        if training:
            self._step += 1

        drawn = self._draw(self.fragments, self.fragment_batch_size, training)
        if drawn is not None:
            frag_batch, _ = drawn
            # `with_induction=False` is an optimization here, not a correctness need: the
            # chi-trick makes a lone fragment's induction exactly zero either way.
            frag_out = self.model(frag_batch, with_induction=False)
            err = frag_out.fragment_energy - frag_batch.fragment_energy
            extra["frag"] = x.fragment_weight * (err / x.energy_scale).pow(2).mean()
            self._metrics["frag_mae"] = (
                float(err.detach().abs().mean()) * KJMOL_PER_HARTREE
            )
            mm, mm_metrics = fragment_multipole_loss(
                frag_out, frag_batch,
                dipole_weight=cfg.elec.dipole_weight,
                quadrupole_weight=cfg.elec.quadrupole_weight,
                dipole_scale=cfg.elec.dipole_scale,
                quadrupole_scale=cfg.elec.quadrupole_scale,
            )
            extra.update({k: x.fragment_weight * v for k, v in mm.items()})
            self._metrics.update(mm_metrics)
            if x.bonded_variance_weight > 0.0:
                extra["bond_var_frag"] = (
                    x.bonded_variance_weight * _bonded_variance(frag_out)
                )

        # --- the dedicated monomer sets --------------------------------------------------------
        w = x.anchor_weight
        for i, anchor in enumerate(self.anchors):
            drawn = self._draw(
                anchor, self.anchor_batch_size, training, grad_positions=True
            )
            if drawn is None:
                continue
            anchor_batch, _ = drawn
            with torch.enable_grad():
                anchor_out = self.model(
                    anchor_batch,
                    with_polarizability=cfg.elec.polarizability_weight > 0.0,
                    with_induction=False,
                )

            onebody_cfg = cfg.onebody
            if training and self._step % self.anchor_force_every != 0:
                onebody_cfg = replace(onebody_cfg, force_weight=0.0)
            terms, metrics = onebody_anchor_loss(
                self.model, anchor_batch, onebody_cfg, out=anchor_out, training=training
            )
            ap, ap_metrics = fragment_polarizability_loss(
                anchor_out, anchor_batch,
                weight=cfg.elec.polarizability_weight,
                scale=cfg.elec.polarizability_scale,
            )
            tag = "" if len(self.anchors) == 1 else f"_{i}"
            extra.update({f"{k}{tag}": w * v for k, v in {**terms, **ap}.items()})
            self._metrics.update(
                {f"{k}{tag}": v for k, v in {**metrics, **ap_metrics}.items()}
            )
        return extra

    def diagnostics(self, out, batch, target):
        """Charge-projection residual, r0 per channel, bonded-parameter spread, streams."""
        q = out.charges.detach()
        n_frag = int(batch.n_fragments)
        per_frag = q.new_zeros(n_frag).index_add_(0, batch.fragment_idx, q)
        want = (
            per_frag.new_zeros(n_frag) if batch.fragment_charge is None
            else batch.fragment_charge.to(per_frag.dtype)
        )
        metrics = {"q_res": float((per_frag - want).abs().max()), **self._metrics}
        for name, value in out.r0.items():
            metrics[f"r0_{name}"] = float(value.detach().mean())
        b0 = out.parameters.bonded0
        for name in ("r_eq", "d", "k"):
            values = getattr(b0, name).detach()
            if values.numel() > 1:
                metrics[f"std_{name}"] = float(values.std())
        return metrics


def _train_once(config: Config):
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = resolve_device(config.device, config.dtype)

    clusters = load_cluster_datasets(
        config.data.path, dtype=dtype, fragmentations=config.data.fragmentations
    )
    if not clusters.has_fragments:
        raise ValueError(
            "the film model routes every pair to a per-fragment or per-frame label, so the "
            "dataset needs a `fragment_idx` column"
        )
    neighbor_types = tuple(clusters.unique_atomic_numbers)
    train_idx, val_idx = split_indices_grouped(
        clusters._group_id, config.data.holdout_fraction, config.data.seed
    )
    fragments = fragment_view(clusters, train_idx)
    anchors = load_anchor_datasets(config.data.monomer_path, dtype=dtype)
    reference_energies = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(dtype)

    torch.manual_seed(config.train.seed)
    model = build_film_model(
        config.features, config.film, neighbor_types, reference_energies
    ).to(device=device, dtype=dtype)

    n_all = sum(p.numel() for p in model.parameters())
    n_env = sum(p.numel() for _n, p in env_parameters(model))
    n_geoms = int(torch.unique(clusters._group_id).shape[0])
    print(
        f"{len(clusters)} cluster frames over {n_geoms} geometries, "
        f"{len(train_idx)}/{len(val_idx)} train/val; {len(fragments)} fragment views; "
        f"{sum(len(a) for a in anchors)} monomer anchor frames in {len(anchors)} stream(s)",
        flush=True,
    )
    print(
        f"{n_all} parameters, {n_env} in the environment embedders "
        f"(conditioning_mode={config.film.conditioning_mode})",
        flush=True,
    )

    streams = FilmStreams(
        model, device,
        fragment_dataset=fragments,
        fragment_batch_size=config.film.fragment_batch_size,
        anchor_datasets=anchors,
        anchor_batch_size=config.film.anchor_batch_size,
        anchor_force_every=config.film.anchor_force_every,
        seed=config.data.seed,
    )

    return fit(
        model, clusters, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS,
        fit_term=strided_fit_term(model, config.film.force_every),
        penalties=streams.penalties,
        diagnostics=streams.diagnostics,
        grad_positions=config.film.force_weight > 0.0,
    )


def _train_staged(config: Config):
    init_from = ""
    result = None
    for stage in config.stages:
        cfg = stage_config(config, stage, init_from)
        print(f"\n=== stage {stage.name} -> {cfg.run_name} ===", flush=True)
        result = _train_once(cfg)
        best = Path(cfg.checkpoint_root) / cfg.run_name / "best.pt"
        if not best.exists():
            raise RuntimeError(
                f"stage {stage.name!r} wrote no checkpoint to {best}; the next stage has "
                f"nothing to warm start from"
            )
        init_from = str(best)
    return result


def train(config: Config):
    return _train_staged(config) if config.stages else _train_once(config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    train(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
