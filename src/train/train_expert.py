"""Fit the fragment-expert model: two data streams, one optimizer, no freeze.

    python -m rsfff.train.train_expert configs/water_expert.yaml

Two streams, drawn together on every step:

======================  =======  ==================================================
stream                  ``eta``  supervises
======================  =======  ==================================================
fragment views          ``= 0``  ``fragment_energy``, fragment dipole and quadrupole
the dedicated monomers  ``= 0``  the same, plus molecular polarizability and true
                                 one-body forces
clusters                live     ``eda_cls_elec``, ``eda_mod_pauli``, ``eda_disp``,
                                 ``eda_pol + eda_ct``, cluster forces
======================  =======  ==================================================

The fragment-view stream is every fragment of every cluster as a frame of its own -- roughly
34k monomer geometries against the 499 in the dedicated set, sampled exactly where clusters
actually go (:func:`rsfff.train.data.fragment_view`).

Why there is no freeze
----------------------
The obvious schedule is: fit the isolated sector on the monomer streams, freeze it, then let
the clusters fit only the environment weights. It is wrong, and the reason is worth stating
because it is not obvious. ``C6``, ``b_disp``, the Pauli multipoles and ``r0`` have **no
monomer-side label at all** -- they appear only in inter-fragment channels, and the
intra-fragment classical remainder that survives the range separation is well under a kJ/mol
per fragment, so monomer data cannot constrain them even in principle. Freezing them after a
monomer stage would pin them at noise and then require the environment weights to carry the
entire dispersion and Pauli parameterization: the environment sector would *become* the model,
which is precisely the pathology this design exists to prevent.

So ``theta_0`` and ``theta`` share their fragment-half weights, cluster data supervises both,
and the guarantee comes from the architecture instead:

    ``theta(h, 0) = theta_0(h)`` identically, and no ``eta`` enters ``fragment_energy``, so the
    one-body sector is exactly isolated at every point in training.

``tests/test_one_body_isolation.py`` is where that is checked. ``d_ob`` below is the running
confirmation: it is the fragment-view error and it is a real measurement, not a formality.

What decides the split is ``expert.env_penalty_weight``, acting on ``theta - theta_0`` per
quantity. The EDA channels fix the sum; that decides how much of it may be environmental.

What to watch, in order of what will bite
-----------------------------------------
``env_c6`` / ``env_b_disp`` / ``env_pauli_multipole`` / ``env_e_bond``
    **The numbers this design exists to produce.** How far each parameter moved between
    ``theta_0`` and ``theta``. If a channel's explanation lives here rather than in the
    fragment, the penalty is too loose.
``env_norm``
    ``||eta||``. Exactly zero on a cluster batch would mean the environment slot is not being
    built at all -- it is a property of the geometry, not of the weights, so unlike the old
    ``EnvironmentResidual`` it cannot be flattened by weight decay.
``ind_ff`` against ``env_e_bond``
    the split of induction between a real relaxation of the multipoles and a shift in the
    bonding parameters. Both are physical; the ratio is what has to stay honest. Two previous
    charge-transfer channels were 96.7% and 100% neural respectively.
``internal`` / ``bond``
    the unlabeled split inside ``fragment_energy``. Only their sum is fitted, so this is a flat
    direction: watch the spread over the last epochs, not the values.
``cg_ind`` / ``cg_fail``
    a climbing iteration count is the early warning that the coupled functional is losing
    positive definiteness, well before a NaN. ``cg_fail`` must stay 0.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from ..ff.units import KJMOL_PER_HARTREE
from ..mlip.heads import core_parameters, env_parameters
from ..mlip.reference_states import AtomicStateReference
from .build_expert import build_expert_model
from .config import Config, load_config, stage_config
from .data import (
    concatenate_datasets,
    fragment_view,
    load_cluster_datasets,
    load_extxyz,
    load_reference_energies,
    split_indices_grouped,
)
from .loss import (
    applicability_loss,
    compute_forces,
    fragment_multipole_loss,
    fragment_polarizability_loss,
    onebody_anchor_loss,
)
from .term_loop import fit
from .train_eem import resolve_device

#: Channel -> the ``eda_*`` header it is fitted against.
_TARGETS = {"elst": "cls_elec", "pauli": "mod_pauli", "disp": "disp"}

#: Induction is fitted against the **sum** of these two. They were separate levels once,
#: separated only by their labels and by which feature stream the bonding term read -- and that
#: swap is a route to the label needing no charge to move, which ate the whole of ``ct``.
_INDUCTION_COMPONENTS = ("pol", "ct")

_LOG_KEYS = (
    "loss", "ob_mae", "elst_mae", "pauli_mae", "disp_mae", "ind_mae", "e_tot_mae",
    "f_clu", "frag_mae", "dip_mae", "quad_mae", "alpha_mae", "f_mono",
    "internal", "bond", "q_res", "applicability", "app_acc",
    "r0_elst", "r0_pauli", "r0_disp",
    # The environment sector, per quantity. See the module docstring.
    "env_norm", "env_c6", "env_b_disp", "env_pauli_multipole", "env_e_bond",
    "ind_ff", "cg_ind", "cg_fail",
)


def expert_fit(out, batch, cfg: Config, *, training: bool = True, with_forces: bool = True):
    """The cluster-stream loss: four EDA channels, the total, and the forces.

    ``fragment_energy`` is **not** fitted here. On a cluster batch it is exactly the same
    function of exactly the same weights as it is on the fragment-view stream, so fitting it in
    both places would double its weight for no extra information. It is reported as ``ob_mae``,
    which is worth watching precisely because it is not being optimized against this batch: it
    is the running check that the one-body sector is what the isolated stream says it is.
    """
    x = cfg.expert
    metrics = {}
    loss = batch.energy.new_zeros(())

    ob_err = out.fragment_energy - batch.fragment_energy
    metrics["ob_mae"] = float(ob_err.detach().abs().mean()) * KJMOL_PER_HARTREE
    metrics["internal"] = float(out.energy_internal.detach().mean()) * KJMOL_PER_HARTREE
    metrics["bond"] = float(out.energy_bond.detach().mean()) * KJMOL_PER_HARTREE
    metrics["env_norm"] = float(out.env_norm.detach().mean())
    for name, value in out.env_shift.items():
        metrics[f"env_{name}"] = float(value.detach())

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
                f"{', '.join('eda_' + m for m in missing)}; the files from "
                f"scripts/parse_roundtrip.py carry both"
            )
        target = sum(batch.eda[k] for k in _INDUCTION_COMPONENTS)
        err = out.interaction["induction"] - target
        loss = loss + x.induction_weight * (err / x.energy_scale).pow(2).mean()
        metrics["ind_mae"] = float(err.detach().abs().mean()) * KJMOL_PER_HARTREE
        if out.energy_frozen_total is not None and out.level_ind is not None:
            relax = out.level_ind.energy - out.energy_frozen_total
            metrics["ind_ff"] = float(relax.detach().mean()) * KJMOL_PER_HARTREE

    # Which decomposition of this geometry is the best one. Contributes only where the batch
    # actually holds competing fragmentations of one geometry -- see `applicability_loss`.
    app_terms, app_metrics = applicability_loss(
        out, batch,
        weight=x.applicability_weight,
        temperature=x.applicability_temperature,
    )
    for value in app_terms.values():
        loss = loss + value
    metrics.update(app_metrics)

    if batch.energy is not None:
        e_err = out.energy - batch.energy
        metrics["e_tot_mae"] = float(e_err.detach().abs().mean()) * KJMOL_PER_HARTREE
        if x.total_energy_weight > 0.0:
            loss = loss + x.total_energy_weight * (e_err / x.energy_scale).pow(2).mean()

    if x.force_weight > 0.0 and with_forces:
        if batch.forces is None:
            raise ValueError(
                "expert.force_weight > 0 but the dataset carries no forces; the merged "
                "eda+force files from scripts/parse_roundtrip.py do (data/wb97mv_tzvpd/*)"
            )
        if not batch.positions.requires_grad:
            raise ValueError(
                "expert.force_weight > 0 but batch.positions is not a leaf; the training loop "
                "must be entered with grad_positions=True"
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
    """:func:`expert_fit` with the cluster force term applied every k-th training step.

    ``model.training`` is what separates a training step from an evaluation one, and it is the
    only reliable signal once the force terms are on: they make grad enabled during evaluation
    as well, so the grad context reads ``True`` on a validation epoch.
    """
    every = max(int(force_every), 1)
    state = {"step": 0}

    def term(out, batch, cfg):
        training = bool(model.training)
        if training:
            state["step"] += 1
        with_forces = (not training) or state["step"] % every == 0
        return expert_fit(out, batch, cfg, training=training, with_forces=with_forces)

    return term


class IsolatedStreams:
    """The two ``eta = 0`` streams, plus the range and environment penalties.

    Held as one object because they share a step counter and a device, and because every one of
    them is a *penalty* in :func:`rsfff.train.term_loop.fit`'s sense -- a term added to the loss
    that does not come from the cluster batch.
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
        #: One per label set; see the anchor block in :meth:`penalties`.
        self.anchors = list(anchor_datasets)
        self.anchor_batch_size = int(anchor_batch_size)
        self.anchor_force_every = max(int(anchor_force_every), 1)
        self.generator = torch.Generator().manual_seed(int(seed))
        self._step = 0
        self._metrics: dict[str, float] = {}

    def _draw(self, dataset, size, training, *, grad_positions: bool = False):
        """A random subset while training, a fixed prefix while evaluating.

        The fixed slice matters: a validation number computed on a different random draw each
        epoch is not comparable between epochs, and these terms are several kJ/mol.

        ``grad_positions`` makes the batch's coordinates a leaf, which the monomer force term
        needs -- that force is a backward pass through the model, and it runs on evaluation
        epochs too so that ``f_mono`` means the same thing on every validation line.
        """
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
        x = cfg.expert
        extra: dict = {}
        self._metrics = {}

        # --- the range separation ---------------------------------------------------------
        if x.r0_weight > 0.0:
            # A one-sided barrier holding r0 at or below its element prior, per channel: moving
            # energy out of the physical backbone and into the parameters has to be earned.
            # Zero, with zero gradient, wherever r0 already sits at the prior. An unbounded
            # downward *pull* is what once put -39 kJ/mol per fragment of "intramolecular
            # dispersion" between covalently bonded atoms, with nothing opposing it because
            # intra classical energy is exactly degenerate with the bonding term.
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

        # --- the environment penalty ------------------------------------------------------
        # The dial. See `ExpertConfig.env_penalty_weight` for why it is L1 and why it acts on
        # the parameters rather than on a feature norm.
        unknown = set(x.env_penalty_weights) - set(out.env_shift)
        if unknown:
            raise KeyError(
                f"expert.env_penalty_weights names {sorted(unknown)}, which the model does not "
                f"report; available: {sorted(out.env_shift)}"
            )
        for name, shift in out.env_shift.items():
            weight = float(x.env_penalty_weights.get(name, x.env_penalty_weight))
            if weight > 0.0:
                extra[f"env_{name}_pen"] = weight * shift

        # --- the fragment-view stream -----------------------------------------------------
        training = bool(self.model.training)
        if training:
            self._step += 1

        drawn = self._draw(self.fragments, self.fragment_batch_size, training)
        if drawn is not None:
            frag_batch, _ = drawn
            # No coupled solve: on a lone fragment the label says induction is zero, and
            # the model's residue there is an artifact of how the levels are defined.
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

        # --- the dedicated monomer sets ---------------------------------------------------
        # The only data carrying a molecular polarizability -- the one label that says how the
        # charges and dipoles *move* rather than where they sit -- and true one-body forces.
        #
        # Several streams rather than one, because the files disagree about what they carry:
        # the bare-ion AIMD trajectories have energies and forces, the `*_pol` references add
        # multipoles and polarizabilities, and `concatenate_datasets` refuses to mix label
        # sets (rightly -- the alternative is silently dropping supervision). Every loss term
        # below already no-ops on an absent label, so each stream contributes what it has.
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
                    # Same reason as the fragment stream, plus one more:
                    # `onebody_anchor_loss` takes its force from `out.energy` on the premise
                    # that a one-fragment system has `energy == fragment_energy`. This is
                    # what makes that exact.
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
            # Suffixed per stream: two streams writing `anchor_e` would have the second
            # silently replace the first in the loss, and the printed line would show one
            # number for two different things.
            tag = "" if len(self.anchors) == 1 else f"_{i}"
            extra.update({f"{k}{tag}": w * v for k, v in {**terms, **ap}.items()})
            self._metrics.update(
                {f"{k}{tag}": v for k, v in {**metrics, **ap_metrics}.items()}
            )
        return extra

    def diagnostics(self, out, batch, target):
        """Charge conservation, the learned ``r0`` per channel, and the stream metrics."""
        res = out.response
        q = res.charges.detach()
        n_frag = int(batch.n_fragments)
        per_frag = q.new_zeros(n_frag).index_add_(0, batch.fragment_idx, q)
        want = (
            per_frag.new_zeros(n_frag) if batch.fragment_charge is None
            else batch.fragment_charge.to(per_frag.dtype)
        )
        metrics = {"q_res": float((per_frag - want).abs().max()), **self._metrics}
        for name, value in out.r0.items():
            metrics[f"r0_{name}"] = float(value.detach().mean())
        return metrics


def _freeze_experts(model) -> None:
    """Freeze everything but the mediator. See ``ExpertConfig.freeze_experts`` for why.

    v4 note: with the per-composition experts gone this freezes the featurizer, the
    fragment-state block and the whole decoder -- everything the mediator's weight multiplies.
    That is still the intent the name records: the mixture's total-energy loss reaches the
    parameterization directly and the mediator only through a scalar.

    Applied after the warm start, so the frozen values are the checkpoint's rather than the
    initialization's -- freezing before loading would pin the experts at random weights.
    """
    frozen = live = 0
    for name, p in model.named_parameters():
        if name.startswith("mediator"):
            live += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    if not live:
        raise ValueError(
            "freeze_experts froze every parameter: the model has no `mediator` submodule, so "
            "this stage would run an optimizer over nothing"
        )
    print(
        f"freeze_experts: {frozen} force-field parameters frozen, {live} mediator ones live",
        flush=True,
    )


def _freeze_core(model) -> None:
    """The environment-only ablation. **Not** part of the schedule; see the module docstring."""
    n = 0
    for _name, p in core_parameters(model):
        p.requires_grad_(False)
        n += p.numel()
    live = sum(p.numel() for _n, p in env_parameters(model))
    print(
        f"freeze_core: {n} isolated-sector parameters frozen, {live} environment ones live",
        flush=True,
    )


#: The optional label fields an anchor file may or may not carry. Two files agreeing on all
#: of these can be concatenated; disagreeing on any one of them means `concatenate_datasets`
#: would refuse, and rightly -- it has no way to fill the gap.
_ANCHOR_LABELS = (
    "_forces", "_dipole", "_polarizability", "_dipole_derivatives",
    "_fragment_energy", "_fragment_dipole", "_fragment_second_moment",
)


def load_anchor_datasets(paths, *, dtype) -> list:
    """Monomer anchor files as a list of streams, one per distinct label set.

    The five monomer files in ``data/wb97mv_tzvpd`` do not carry the same labels. The bare-ion
    AIMD trajectories (``h3o+``, ``oh-``) have energies and true one-body forces and nothing
    else -- an AIMD run prints no multipoles per step -- while ``h2o_*_pol`` and the three
    ``*_opt_*_pol`` references add dipoles, second moments and polarizabilities. Concatenating
    those would mean dropping a label for the frames that have it, so files are grouped by
    what they carry and each group becomes a stream of its own.

    In practice this returns two streams for that set, not five: everything with
    polarizabilities lands together, and the two bare-ion trajectories land together.
    """
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]

    by_labels: dict[tuple, list] = {}
    for path in paths:
        dataset = load_extxyz(path, dtype=dtype)
        signature = tuple(getattr(dataset, name) is not None for name in _ANCHOR_LABELS)
        by_labels.setdefault(signature, []).append(dataset)
    return [concatenate_datasets(group) for group in by_labels.values()]


def _train_once(config: Config):
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = resolve_device(config.device, config.dtype)

    clusters = load_cluster_datasets(
        config.data.path, dtype=dtype, fragmentations=config.data.fragmentations
    )
    if not clusters.has_fragments:
        raise ValueError(
            "the expert model routes every pair to a per-fragment or per-frame label, so the "
            "dataset needs a `fragment_idx` column"
        )
    neighbor_types = tuple(clusters.unique_atomic_numbers)
    # By *geometry*, not by frame. Several decompositions of one geometry share every
    # nucleus, so a frame-wise split would put one in training and its alternative in
    # validation and report the result as held out.
    train_idx, val_idx = split_indices_grouped(
        clusters._group_id, config.data.holdout_fraction, config.data.seed
    )

    # Split the clusters *first*, then explode. A fragment of a validation cluster must not
    # appear in the training stream: the two streams share every weight, so that would be
    # leakage of exactly the quantity `frag_mae` is meant to be validating.
    fragments = fragment_view(clusters, train_idx)

    anchors = load_anchor_datasets(config.data.monomer_path, dtype=dtype)

    reference_energies = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(dtype)
    atomic_states = (
        AtomicStateReference.from_json(
            config.data.atomic_reference_states, neighbor_types, dtype=dtype
        )
        if config.data.atomic_reference_states else None
    )

    # Seed *before* building. Every head calls `torch.randn` for its species tables and its
    # equivariant reductions, so an unseeded build starts from whatever global RNG state the
    # process is in -- and two fits that differ only in `env_penalty_weight` would then differ
    # in their initialization too, which is exactly the comparison this model exists to make.
    torch.manual_seed(config.train.seed)
    model = build_expert_model(
        config, neighbor_types, reference_energies, atomic_states
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
        f"{n_all} parameters, {n_env} ({100 * n_env / n_all:.1f}%) in the environment slot",
        flush=True,
    )

    streams = IsolatedStreams(
        model, device,
        fragment_dataset=fragments,
        fragment_batch_size=config.expert.fragment_batch_size,
        anchor_datasets=anchors,
        anchor_batch_size=config.expert.anchor_batch_size,
        anchor_force_every=config.expert.anchor_force_every,
        seed=config.data.seed,
    )

    penalties, diagnostics = streams.penalties, streams.diagnostics
    log_keys = _LOG_KEYS
    if config.expert.mediator:
        mixture = _build_mixture_stream(config, model, clusters, train_idx, device, dtype)
        penalties = _chain_penalties(streams.penalties, mixture.penalties)
        diagnostics = _chain_diagnostics(streams.diagnostics, mixture.diagnostics)
        log_keys = _LOG_KEYS + _MIXTURE_LOG_KEYS

    return fit(
        model, clusters, config, config, device, train_idx, val_idx,
        log_keys=log_keys,
        fit_term=strided_fit_term(model, config.expert.force_every),
        penalties=penalties,
        diagnostics=diagnostics,
        grad_positions=config.expert.force_weight > 0.0,
        after_warm_start=_after_warm_start(config),
    )


#: What the mixture stream adds to the printed line. ``pi_occ`` is §9's occupancy diagnostic:
#: exactly 0 everywhere means the mediator is off or the trigger never opens, and broadly split
#: everywhere means it is hedging rather than deciding.
_MIXTURE_LOG_KEYS = (
    "mix_e_mae", "mix_f", "mix_ct", "pi_occ", "pi_split", "pi_sharp",
)


def _after_warm_start(config: Config):
    """The freeze hooks this run wants, composed, or ``None``.

    The two are independent and both act after the checkpoint is loaded: ``freeze_core`` is the
    environment-only *ablation* and ``freeze_experts`` is the mediator *stage*.
    """
    hooks = []
    if config.expert.freeze_core:
        hooks.append(_freeze_core)
    if config.expert.freeze_experts:
        hooks.append(_freeze_experts)
    if not hooks:
        return None

    def run(model):
        for hook in hooks:
            hook(model)

    return run


def _merge(dicts, what: str) -> dict:
    """Merge, refusing a key collision.

    Two streams quietly contributing to one loss key is the kind of thing that stays invisible
    until the printed number stops meaning what its name says.
    """
    out: dict = {}
    for d in dicts:
        clash = set(d) & set(out)
        if clash:
            raise ValueError(f"two streams both produce {what} {sorted(clash)}")
        out.update(d)
    return out


def _chain_penalties(*fns):
    """Run several ``penalties`` callables and merge the loss terms they return."""

    def combined(out, batch, cfg):
        return _merge([fn(out, batch, cfg) for fn in fns], "loss term")

    return combined


def _chain_diagnostics(*fns):
    """Run several ``diagnostics`` callables and merge the metrics they return."""

    def combined(out, batch, target):
        return _merge([fn(out, batch, target) for fn in fns], "metric")

    return combined


def _build_mixture_stream(config, model, clusters, train_idx, device, dtype):
    """The mediator, its geometries, and the stream that trains it (``docs/fff_v2.md`` §8).

    The mediator's input width is read off the model's own slots rather than recomputed from
    the feature config: the fragment slot carries the ``(Q_f, 2S_f)`` block appended by
    :class:`rsfff.ff.fragment_state.FragmentStateEmbedding`, so the descriptor is wider than
    the featurizer's own ``feature_dims`` says and a width computed from the config would be
    wrong in a way that only shows up as a shape error deep in a training step.

    **Held out by geometry.** The mixture groups are filtered to the geometries in
    ``train_idx``, so a mediated geometry whose vertices are in validation never trains the
    mediator either.
    """
    from ..ff.mediator import MediatorHead
    from .mixture_data import load_mixture_groups
    from .mixture_stream import MixtureStream

    dataset = load_mixture_groups(config.data.path, dtype=dtype)
    train_geometries = {int(g) for g in clusters._group_id[train_idx].tolist()}
    groups = [g for g in dataset.groups if g.group_id in train_geometries]
    held_out = len(dataset.groups) - len(groups)
    if dataset.groups and not groups:
        raise ValueError(
            f"the mediator has {len(dataset.groups)} mediable geometries but none of them fell "
            f"in the training split; the group ids in mixture_data and load_cluster_datasets "
            f"have drifted apart and the mediator would train on nothing"
        )
    print(
        f"mediator: {dataset.summary()}; {len(groups)} in training, {held_out} held out",
        flush=True,
    )

    widths = _slot_widths(model, groups[0]) if groups else (0, 0)
    mediator = MediatorHead(
        widths[0], widths[1],
        hidden=config.expert.mediator_hidden,
        depth=config.expert.mediator_depth,
        bump=dict(
            lo0=0.0, lo1=0.0,
            hi1=config.expert.mediator_bump_hi1,
            hi0=config.expert.mediator_bump_hi0,
        ),
    ).to(device=device, dtype=dtype)
    # The mediator is a *submodule* of the model, so the optimizer built from
    # `model.parameters()` picks it up and the checkpoint carries it. It owns no force-field
    # parameter and reads no expert's weights; the coupling is one way.
    model.mediator = mediator
    print(
        f"mediator: {sum(p.numel() for p in mediator.parameters())} parameters, "
        f"slots {widths[0]}+{widths[1]}",
        flush=True,
    )

    return MixtureStream(
        model, mediator, groups, device,
        batch_size=config.expert.mixture_batch_size,
        energy_weight=config.expert.mixture_energy_weight,
        force_weight=config.expert.mixture_force_weight,
        energy_scale=config.expert.energy_scale,
        force_scale=config.expert.force_scale,
        ct_weight=config.expert.mixture_ct_weight,
        sharpness_weight=config.expert.mediator_sharpness_weight,
        ct_scale=config.expert.energy_scale,
        force_every=config.expert.mixture_force_every,
        induction=config.expert.induction,
        seed=config.data.seed,
    )


def _slot_widths(model, group) -> tuple[int, int]:
    """``(p_frag, p_env)`` of the joined descriptor, measured on a real group."""
    from ..ff.mixture_model import intra_pairs_unsorted

    with torch.no_grad():
        emission = model.emit(
            group.batch(0), group.fragments[0],
            bond_index=intra_pairs_unsorted(group.fragments[0]),
        )
    frag = int(emission.iso.inv_feats.shape[-1])
    joined = int(emission.joined.inv_feats.shape[-1])
    return frag, joined - frag


def _train_staged(config: Config):
    """Sequential stages, each warm starting from the previous one's best checkpoint.

    Used for a cheap pre-train on the isolated streams before the clusters arrive -- it gets the
    one-body and response sectors into the right basin so the coupled solve does not spend its
    first epochs running on nonsense multipoles. **A warm start, never a freeze**: see the
    module docstring for why freezing the isolated sector is the wrong lever.
    """
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
