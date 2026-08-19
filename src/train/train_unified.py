"""Fit all four channels at once under the unified pair model.

    python -m rsfff.train.train_unified configs/water_unified.yaml

One pair list, one featurization, one response solve, one correction trunk, four targets::

    fragment_energy   <- E0 + E_internal + intra pairs (classical + corrections + bond)
    eda_cls_elec      <- inter pairs, electrostatic channel
    eda_mod_pauli     <- inter pairs, Pauli channel
    eda_disp          <- inter pairs, dispersion channel

Why all four together rather than in sequence
---------------------------------------------
The same argument ``configs/water_onebody_elec.yaml`` makes for the 1-body/electrostatics
pair, extended: the channels now share a trunk, so fitting them in sequence means each stage
partially undoes the last. They also share the response solve, the range-separation heads and
the pair list, and every one of those is under-determined by any single target.

What to watch
-------------
``intra_elst`` / ``intra_pauli`` / ``intra_disp`` are the readouts that matter: classical
energy, in kJ/mol per fragment, sitting inside ``fragment_energy``. They should be small --
the bond channel is supposed to be doing that job.

**Read the energies, not the gates.** ``gate_intra`` was the original diagnostic and it is
actively misleading: a gate of 1e-5 on the Pauli channel is nothing, while a gate of 0.98 on
dispersion was quietly routing -28 kJ/mol per fragment of "dispersion" between covalently
bonded atoms -- ten times the ``ob_mae`` it was being judged on -- and the gate readout gave
no sense of the difference. That leak is what ``unified.intra_classical_weight`` guards
against; the fit will not find it on its own, because the bond channel absorbs whatever
appears and the two are exactly degenerate.

``gate_inter_elst`` should sit at 1.0, and ``r0`` per channel is small enough to read
directly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from ..ff.dispersion import DispersionParameterHeads, build_log_priors
from ..ff.environment import N_PAIR_INVARIANTS
from ..ff.atomic_energy import AtomicStateEnergy
from ..ff.multipole import irrep2_to_spherical
from ..ff.pauli import PauliMultipoleHeads, build_pauli_priors
from ..ff.range_priors import RANGE_CHANNELS, build_range_priors
from ..ff.unified import (
    ClassicalSpec,
    EnvironmentResidual,
    FragmentStateEmbedding,
    RangeSeparationHeads,
    UnifiedPairModel,
)
from ..ff.units import KJMOL_PER_HARTREE
from ..mlip.reference_states import AtomicStateReference
from ..mlip.switch import pairwise_switch
from ..mlip.unified_head import ChannelSpec, UnifiedPairHead
from .config import Config, load_config, stage_config
from .data import (
    load_datasets,
    load_extxyz,
    load_reference_energies,
    split_indices,
)
from .loss import (
    compute_forces,
    fragment_multipole_loss,
    fragment_polarizability_loss,
    free_atom_batch,
    free_atom_polarizability_loss,
    onebody_anchor_loss,
)
from .term_loop import fit
from .train_eem import resolve_device
from .train_elec import build_featurizer, build_response

#: EDA component fitted by each interaction channel. ``pol``/``ct`` are added only when their
#: levels are switched on, so a frozen-level config raises nothing about missing labels.
_TARGETS = {"elst": "cls_elec", "pauli": "mod_pauli", "disp": "disp"}
_LEVEL_TARGETS = {"pol": "pol", "ct": "ct"}

_LOG_KEYS = (
    "loss", "ob_mae", "elst_mae", "pauli_mae", "disp_mae", "pol_mae", "ct_mae",
    "a_mae", "f_mae", "dip_mae", "quad_mae", "e_tot_mae",
    "anchor_e", "anchor_f", "dipole", "quad", "alpha_mae", "f_clu", "intra_ff",
    "q_res", "qO", "dchi", "internal", "bond", "e_atom",
    # `d_internal` is the isolated-fragment internal energy's drift from where this stage
    # started, kJ/mol. It is the quantity that produced the one-body constant offset: the
    # split between `internal` and `bond` carries no label, so a higher level is free to slide
    # it and leave the bond head chasing. Watch this, not `ob_mae`, for that failure.
    # `elst_env` is the environment-dependent electrostatic energy booked as polarization.
    "d_internal", "d_e_atom", "int_drift", "elst_env", "free_alpha_mae",
    "gate_inter_elst", "gate_intra_disp",
    "intra_elst", "intra_pauli", "intra_disp",
    "r0_elst", "r0_pauli", "r0_disp", "env_norm",
    # The polarization/CT split and the solver's health. `pol_ff` is the classical relaxation
    # and must stay <= 0 while the two levels share response parameters; `q_ct` is how much
    # charge actually crossed a fragment boundary; a climbing `cg_ct` is the early warning
    # that the coupled functional is losing positive definiteness.
    "pol_ff", "ct_ff", "q_ct", "cg_pol", "cg_ct", "cg_fail",
)


def build_unified_model(config: Config, neighbor_types, reference_energies, atomic_states=None):
    """The whole model. Reuses the ``elec:``/``dispersion:``/``pauli:`` blocks verbatim."""
    ucfg, ecfg = config.unified, config.elec
    fcfg = config.features
    if ucfg.corr_r_off > fcfg.cutoff or ucfg.bond_r_off > fcfg.cutoff:
        raise ValueError(
            f"unified.corr_r_off ({ucfg.corr_r_off}) / bond_r_off ({ucfg.bond_r_off}) exceed "
            f"the feature cutoff ({fcfg.cutoff}); past it the pair head reads features whose "
            f"atoms cannot see each other"
        )
    if int(ucfg.max_rank) != int(ecfg.max_rank):
        raise ValueError(
            f"unified.max_rank ({ucfg.max_rank}) must match elec.max_rank ({ecfg.max_rank}); "
            f"the response heads decide which multipoles exist"
        )

    featurizer = build_featurizer(fcfg, ecfg, neighbor_types)
    n_species = len(neighbor_types)
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)

    # The fragment-state block widens the invariant input for *every* consumer: the response
    # parameters need fragment charge as much as the range separation and the pair trunk do.
    fragment_state = FragmentStateEmbedding(
        ucfg.fragment_state_dim,
        hidden=ucfg.fragment_state_hidden,
        depth=ucfg.fragment_state_depth,
    )
    p0 = featurizer.feature_dims[0] + fragment_state.dim

    response = build_response(
        featurizer, fcfg, ecfg, neighbor_types, atomic_states,
        p0_extra=fragment_state.dim,
        # Built at every stage, not only the charge-transfer one, so the staged warm starts
        # see the same parameter set throughout and this head is not "left at initialization"
        # exactly when it starts doing work. Unused -- and untrained, since nothing reaches
        # it -- until `charge_transfer` opens the radius-derived channels.
        separate_ct_compliance=ucfg.separate_ct_compliance,
    )

    log_c6, log_b_disp = build_log_priors(neighbor_types, b_prior=config.dispersion.b_prior)
    disp_params = DispersionParameterHeads(
        p0, n_species, log_c6_prior=log_c6, log_b_prior=log_b_disp,
        emb_dim=config.dispersion.emb_dim, hidden=config.dispersion.hidden,
        depth=config.dispersion.depth,
        learn_c6=config.dispersion.learn_c6, environment_c6=config.dispersion.environment_c6,
        learn_b=config.dispersion.learn_b, environment_b=config.dispersion.environment_b,
    )

    log_q, log_b_pauli, mu_scale, quad_scale = build_pauli_priors(neighbor_types)
    pauli_params = PauliMultipoleHeads(
        p0, p1, n_species,
        log_q_prior=log_q, log_b_prior=log_b_pauli, dipole_scale=mu_scale,
        p2=p2, quad_scale=quad_scale,
        irrep2_to_spherical=irrep2_to_spherical(featurizer.backend.irrep6_to_voigt()),
        emb_dim=config.pauli.emb_dim, hidden=config.pauli.hidden, depth=config.pauli.depth,
        equiv_channels=config.pauli.equiv_channels, max_rank=ucfg.max_rank,
        learn_q=config.pauli.learn_q, environment_q=config.pauli.environment_q,
        learn_b=config.pauli.learn_b, environment_b=config.pauli.environment_b,
        learn_dipole=config.pauli.learn_dipole,
        learn_quadrupole=config.pauli.learn_quadrupole,
    )

    range_heads = RangeSeparationHeads(
        p0, n_species,
        log_r0_prior=build_range_priors(neighbor_types),
        alpha_init=ucfg.alpha_init, channels=RANGE_CHANNELS,
        emb_dim=ucfg.r0_emb_dim, hidden=ucfg.r0_hidden, depth=ucfg.r0_depth,
        learn_r0=ucfg.learn_r0, environment_r0=ucfg.environment_r0,
        learn_alpha=ucfg.learn_alpha,
    )

    # The per-atom energy of the electronic state. Built only when asked for, because it and
    # the pair head's bond channel describe the same thing -- an atom's bonding energy -- and
    # having both live at once would reintroduce exactly the unlabeled split this replaces.
    atomic_energy = None
    if ucfg.atomic_energy:
        atomic_energy = AtomicStateEnergy(
            p0, n_species, p1=p1, p2=p2,
            irrep2_to_spherical=(
                irrep2_to_spherical(featurizer.backend.irrep6_to_voigt())
                if p2 is not None else None
            ),
            emb_dim=ucfg.atomic_energy_emb_dim,
            hidden=ucfg.atomic_energy_hidden,
            depth=ucfg.atomic_energy_depth,
            equiv_channels=ucfg.atomic_energy_equiv_channels,
            energy_scale=ucfg.atomic_energy_scale,
        )

    corr = dict(r_on=ucfg.corr_r_on, r_off=ucfg.corr_r_off)
    pair_head = UnifiedPairHead(
        p0, n_species,
        {
            "elst": ChannelSpec(**corr, energy_scale=ucfg.elst_energy_scale),
            "pauli": ChannelSpec(**corr, energy_scale=ucfg.pauli_energy_scale),
            "disp": ChannelSpec(**corr, energy_scale=ucfg.disp_energy_scale),
            "bond": ChannelSpec(
                r_on=ucfg.bond_r_on, r_off=ucfg.bond_r_off,
                energy_scale=ucfg.bond_energy_scale,
            ),
            # The bond channel with the fragment constraint lifted: a bond energy for pairs
            # that *cross* a fragment boundary, which is what charge transfer is allowed to
            # be that polarization is not. Built at every stage so the staged warm starts see
            # one parameter set; only `charge_transfer` ever reads it.
            **({"ct_bond": ChannelSpec(
                r_on=ucfg.bond_r_on, r_off=ucfg.bond_r_off,
                energy_scale=ucfg.ct_bond_energy_scale,
            )} if ucfg.ct_bond else {}),
        },
        range_channels=RANGE_CHANNELS if ucfg.pair_range_separation else (),
        # The side channel the external field enters through. Zero unless polarization is on,
        # which keeps the trunk's input width -- and hence the checkpoint -- unchanged.
        extra_dim=N_PAIR_INVARIANTS if ucfg.polarization else 0,
        emb_dim=ucfg.emb_dim, hidden=ucfg.corr_hidden, depth=ucfg.corr_depth,
        n_radial=ucfg.corr_n_radial,
    )

    environment = None
    if ucfg.environment_features:
        environment = EnvironmentResidual(
            p0, p1, p2, n_species,
            emb_dim=ucfg.emb_dim, hidden=ucfg.env_hidden, depth=ucfg.env_depth,
        )

    return UnifiedPairModel(
        featurizer, response, disp_params, pauli_params, range_heads, pair_head,
        fragment_state, reference_energies,
        atomic_energy=atomic_energy,
        pair_corrections=ucfg.pair_corrections,
        environment=environment,
        max_rank=ucfg.max_rank,
        classical={
            "elst": ClassicalSpec(ucfg.elst_cutoff, ucfg.taper_width),
            "pauli": ClassicalSpec(ucfg.pauli_cutoff, ucfg.taper_width),
            "disp": ClassicalSpec(ucfg.disp_cutoff, ucfg.taper_width),
        },
        polarization=ucfg.polarization,
        charge_transfer=ucfg.charge_transfer,
        ct_channel_cutoff=ucfg.ct_channel_cutoff,
        ct_channel_taper=ucfg.ct_channel_taper,
        ct_compliance_scale=ucfg.ct_compliance_scale,
        cg_rtol=ucfg.cg_rtol, cg_atol=ucfg.cg_atol, cg_maxiter=ucfg.cg_maxiter,
    )


def unified_fit(out, batch, cfg: Config):
    """Four dimensionless terms: the fragment energy and the three EDA components.

    ``cfg`` is the whole :class:`Config`, not one block -- this fit spans ``unified:`` and
    reads ``elec:``/``onebody:`` for the shared pieces. Every error is divided by
    ``unified.energy_scale`` before squaring, the invariant :mod:`rsfff.train.term_loop` is
    built on, so all four weights are plain dimensionless numbers.
    """
    u = cfg.unified
    ob_err = out.fragment_energy - batch.fragment_energy
    loss = u.onebody_weight * (ob_err / u.energy_scale).pow(2).mean()
    metrics = {
        "ob_mae": float(ob_err.detach().abs().mean()) * KJMOL_PER_HARTREE,
        # kJ/mol per fragment, all three, matching `scripts/staged_diagnostics.py` and every
        # other energy in this dict. They were Hartree, which put three columns on a different
        # scale from their neighbours for no reason -- `bond` in particular reads as 1e-05 and
        # is 0.04 kJ/mol.
        "internal": float(out.energy_internal.detach().mean()) * KJMOL_PER_HARTREE,
        # Despite the name this is the intra-fragment *classical* remainder, not a bond
        # channel: `e_corr["bond"]` is identically zero unless 'bond' is listed in
        # `pair_corrections`, so what is left is the gated elst/pauli/disp between
        # same-fragment atoms. Watch `intra_elst`/`intra_pauli`/`intra_disp` for its breakdown.
        "bond": float(out.energy_bond.detach().mean()) * KJMOL_PER_HARTREE,
        # The other half of the unlabeled split inside `fragment_energy`. Only their sum is
        # fitted, so watch the two together: a drift in `e_atom` against a fixed `internal`
        # between stages is the failure this design exists to remove.
        "e_atom": float(out.energy_atom.detach().mean()) * KJMOL_PER_HARTREE,
    }
    if out.elst_env is not None:
        # Already inside `pol`; reported alone because the size of what moved out of
        # `cls_elec` is what says whether the re-partition is doing anything.
        metrics["elst_env"] = float(out.elst_env.detach().abs().mean()) * KJMOL_PER_HARTREE
    weights = {
        "elst": u.elst_weight, "pauli": u.pauli_weight, "disp": u.disp_weight,
        "pol": u.pol_weight, "ct": u.ct_weight,
    }
    targets = dict(_TARGETS)
    if u.polarization:
        targets["pol"] = _LEVEL_TARGETS["pol"]
    if u.charge_transfer:
        targets["ct"] = _LEVEL_TARGETS["ct"]
    for name, key in targets.items():
        if key not in batch.eda:
            raise KeyError(
                f"the dataset has no eda_{key} label, which the {name} channel fits; "
                f"available: {sorted(batch.eda)}"
            )
        err = out.interaction[name] - batch.eda[key]
        loss = loss + weights[name] * (err / u.energy_scale).pow(2).mean()
        metrics[f"{name}_mae"] = float(err.detach().abs().mean()) * KJMOL_PER_HARTREE

    # The cluster total and its gradient. Diagnostics until `total_energy_weight` /
    # `force_weight` are turned on, which should only happen once `pol` and `ct` are on:
    # below that level the decomposition is incomplete by construction, so `energy` is short
    # two channels and fitting it pushes the deficit into whichever channel distorts cheapest.
    #
    # **The EDA component weights stay up when these come on.** The total is one number per
    # frame against six well-posed component targets, and supervised alone it admits every
    # wrong split that happens to sum correctly -- the same degeneracy the pol/ct corrections
    # already have with each other, but now spanning all six channels at once. The components
    # are what keep the decomposition meaningful; the total is what makes it a force field.
    if batch.energy is not None:
        e_err = out.energy - batch.energy
        metrics["e_tot_mae"] = float(e_err.detach().abs().mean()) * KJMOL_PER_HARTREE
        if u.total_energy_weight > 0.0:
            loss = loss + u.total_energy_weight * (e_err / u.energy_scale).pow(2).mean()

    if u.force_weight > 0.0:
        if batch.forces is None:
            raise ValueError(
                "unified.force_weight > 0 but the dataset carries no forces; the merged "
                "eda+force files from scripts/parse_roundtrip.py do (data/wb97mv_tzvpd/*)"
            )
        if not batch.positions.requires_grad:
            raise ValueError(
                "unified.force_weight > 0 but batch.positions is not a leaf; the training "
                "loop must be entered with grad_positions=True"
            )
        # `create_graph` from the context: a training step needs the second-order graph, an
        # evaluation epoch does not and should not pay for one. See UnifiedConfig.force_weight
        # for the measured bias this carries at the pol/CT levels -- the forces are exact, the
        # derivative of the force with respect to the parameters is ~1e-4 relative off,
        # because the coupled solve's adjoint is not itself double-differentiable.
        forces = compute_forces(
            out.energy, batch.positions, create_graph=torch.is_grad_enabled()
        )
        f_err = (forces - batch.forces) / u.force_scale
        loss = loss + u.force_weight * f_err.pow(2).sum(-1).mean()
        metrics["f_clu"] = float((forces - batch.forces).detach().abs().mean())
    return loss, metrics, batch.fragment_energy


def _frozen_level_parameters(model) -> dict[str, list]:
    """Everything on the path from geometry to an *isolated fragment's* observables.

    Three entries, and the list is this long because a shorter one does not work. Reverting a
    trained stage-3 checkpoint's ``response.params`` alone to its stage-1 values left the
    monomer internal energy at -230 kJ/mol against stage 1's -342: ``featurizer.channel_proj``
    had moved 22% over the two higher stages, so the same head weights were reading different
    features. The compliance head is here because it enters the frozen intra-fragment SQE
    solve; charge transfer gets its own head rather than sharing this one, precisely so that
    freezing it costs CT nothing.

    Not here, and not an oversight: ``pauli_params``, ``disp_params`` and the correction trunk
    also reach ``fragment_energy``, through the intra classical channels and the bond channel.
    They have to keep training -- their own labels are inter-fragment -- and they are not a
    problem, because ``fragment_energy`` *has* a label. What went wrong before was the
    unlabeled split *inside* it, between ``E_internal`` and ``energy_bond``; freezing this list
    pins ``E_internal``, so the bond head is fitting a target that holds still instead of
    chasing one that moved 60-110 kJ/mol.
    """
    frozen = {
        "response.params": list(model.response.params.parameters()),
        "response.compliance_head": list(model.response.compliance_head.parameters()),
    }
    # `atomic_energy` *is* the frozen level's definition now: `E_atom^0` is what carries the
    # intramolecular deformation energy inside `fragment_energy`, and the higher levels are
    # the same weights read at a relaxed electronic state. Leaving it trainable above stage 1
    # would hand `pol` and `ct` at weight 30 a direct lever on the one-body zero -- which is
    # precisely how the bond channel it replaces ended up carrying a constant -1.686 kJ/mol
    # per fragment. What pol and ct move instead is the *state*, not this network.
    if getattr(model, "atomic_energy", None) is not None:
        frozen["atomic_energy"] = list(model.atomic_energy.parameters())
    proj = getattr(model.featurizer, "channel_proj", None)
    if proj is not None:
        frozen["featurizer.channel_proj"] = [proj]
    # `fragment_state` augments `h_frag` before the response heads see it, so it is part of
    # the fragment-confined descriptor even though it is identically zero on the neutral
    # singlets this is fit on today. It stops being zero the moment a fragment is charged,
    # which is the case this whole distinction exists for.
    if model.fragment_state is not None and model.fragment_state.dim:
        frozen["fragment_state"] = list(model.fragment_state.parameters())
    return {k: v for k, v in frozen.items() if v}


class AnchorTerms:
    """Penalties and diagnostics: the monomer anchor, the ``r0`` pull, and the gate readouts.

    One object rather than separate closures so the anchor's forward -- which includes a
    backward pass for the forces -- happens once per step and its metrics are reused by the
    logging callback, exactly as in :mod:`rsfff.train.train_onebody_elec`.

    Unlike that module the anchor is **minibatched**. Its force term makes each evaluation a
    second-order backward, and at the full 500 monomer frames that measured at ~95% of a
    training step's wall time -- the same computation repeated once per minibatch of the
    training set, 269 times an epoch. Drawing a fresh subset per step is plain SGD on that
    term. Evaluation draws a fixed leading slice instead, so the validation metric is not
    itself a random variable.

    The force term is additionally **strided**: ``unified.anchor_force_every`` applies it on
    every k-th training step and on every evaluation step. Even minibatched it remained the
    most expensive single item in a step -- 36% of the total on the frozen stage -- because it
    is the only second-order backward in the fit.
    """

    def __init__(self, model, anchor, device, batch_size=0, seed=0,
                 atomic_states=None, neighbor_types=(), force_every=1):
        self.model = model
        self.anchor = anchor            # MoleculeDataset, or None
        self.device = device
        self.batch_size = int(batch_size)
        self.force_every = max(int(force_every), 1)
        #: Counts *training* steps only, so the stride is over optimizer steps rather than
        #: over forwards; an evaluation epoch neither advances it nor skips its own force term.
        self._train_step = 0
        self._gen = torch.Generator().manual_seed(int(seed))
        self._metrics: dict[str, float] = {}
        self._eval_batch = self._eval_idx = None
        # A handful of one-atom systems, built once: the free-atom limit does not depend on
        # anything that varies between steps.
        self._free_batch, self._free_alpha = (None, None)
        if atomic_states is not None:
            self._free_batch, self._free_alpha = free_atom_batch(
                atomic_states, neighbor_types,
                dtype=torch.get_default_dtype(), device=device,
            )
        #: (len(anchor),) the isolated-fragment internal energy at the moment this stage
        #: started, filled by :meth:`snapshot_frozen_level`. ``None`` when not anchoring.
        self._internal_ref = None
        #: The same snapshot for ``E_atom``. Both halves of the unlabeled split are watched,
        #: because which of them carries the energy is a configuration choice: under
        #: ``elec.direct_multipoles`` the on-site sectors leave ``internal_energy`` entirely,
        #: so an alarm on ``internal`` alone would be watching a quantity that is near zero by
        #: construction and would never fire.
        self._atom_ref = None

    def _anchor_batch(self, training: bool):
        """``(batch, indices)``: a monomer batch ready for the force term, or two Nones.

        The indices come back because the drift penalty compares against a per-frame snapshot
        and the training batch is a fresh random subset every step.
        """
        if self.anchor is None:
            return None, None
        n = len(self.anchor)
        take = n if not self.batch_size else min(self.batch_size, n)
        if take == n and self._eval_batch is not None:
            return self._eval_batch, self._eval_idx
        if not training:
            if self._eval_batch is None:
                self._eval_idx = list(range(take))
                self._eval_batch = self._make(self._eval_idx)
            return self._eval_batch, self._eval_idx
        idx = torch.randperm(n, generator=self._gen)[:take].tolist()
        return self._make(idx), idx

    def _make(self, indices):
        batch = self.anchor.flat_batch(indices).to(self.device)
        batch.positions.requires_grad_(True)
        return batch

    def snapshot_frozen_level(self, cfg: Config) -> None:
        """Freeze the fragment-confined path and record the internal energy it produces.

        Called once per stage, after the warm start and before the optimizer is built, so what
        it captures is exactly the previous stage's fitted frozen level.

        The freeze and the snapshot are two halves of one idea and are deliberately issued
        together: the frozen level has no label of its own above stage 1, so unless something
        holds it, ``pol`` and ``ct`` at weight 30 will move it -- and they did, by 62 and 109
        kJ/mol of internal energy and 1 a0^3 of monomer polarizability. See
        :attr:`rsfff.train.config.UnifiedConfig.freeze_frozen_level`.
        """
        u = cfg.unified
        # A pair head that nothing calls. With **every** energy readout off *and* the per-pair
        # range separation off, no gradient ever reaches it. `torch.optim.Adam` skips a
        # parameter whose `.grad` is None, so it would in fact survive untouched -- but that
        # is a property of the optimizer rather than of anything stated here, and the whole
        # point of keeping the module is that its weights are still worth something later.
        # Saying so explicitly also keeps it out of the optimizer state entirely.
        #
        # One live channel is enough to keep the whole head training: the trunk is shared, so
        # `pair_corrections: [elst]` puts a gradient on every parameter except the three silent
        # readouts, and freezing any of it would be wrong.
        head = getattr(self.model, "pair_head", None)
        if head is not None and not self.model.pair_corrections and not head.range_channels:
            n = 0
            for prm in head.parameters():
                prm.requires_grad_(False)
                n += prm.numel()
            if n:
                print(
                    f"pair head inert (no correction channels, no per-pair r0): {n} parameters "
                    f"held at their loaded values and kept out of the optimizer"
                , flush=True)
        if u.freeze_frozen_level:
            frozen = _frozen_level_parameters(self.model)
            n = 0
            for params in frozen.values():
                for p in params:
                    p.requires_grad_(False)
                    n += p.numel()
            print(
                f"frozen level held at its warm start: {n} parameters in "
                f"{', '.join(frozen)} no longer train"
            , flush=True)
        if u.internal_drift_weight > 0.0 and self.anchor is not None:
            batch = self.anchor.flat_batch(range(len(self.anchor))).to(self.device)
            with torch.no_grad():
                snap = self.model(batch)
                self._internal_ref = snap.energy_internal.detach().clone()
                self._atom_ref = snap.energy_atom.detach().clone()
            print(
                f"frozen-level drift anchored on {len(self.anchor)} monomer frames at "
                f"internal {float(self._internal_ref.mean()) * KJMOL_PER_HARTREE:.3f} + "
                f"E_atom {float(self._atom_ref.mean()) * KJMOL_PER_HARTREE:.3f} kJ/mol"
            , flush=True)

    def penalties(self, out, batch, cfg: Config):
        u = cfg.unified
        extra = {}
        if u.corr_l2_weight > 0.0:
            corr = torch.stack(
                [out.interaction_corr[n] for n in out.interaction_corr], dim=0
            )
            extra["corr_l2"] = u.corr_l2_weight * (corr / u.energy_scale).pow(2).mean()
        if u.r0_weight > 0.0:
            # A one-sided barrier holding r0 at or below its element prior, per channel:
            # moving energy out of the physical backbone and into the network has to be
            # earned. Zero, with zero gradient, wherever r0 already sits at the prior.
            #
            # **This used to be an unbounded linear pull downward and that was the bug.** The
            # restriction to inter-fragment pairs was supposed to keep it out of the bonded
            # region, and it does not: `r0_pair` is the per-element base times a *shared trunk*
            # deviation, and the trunk applies one deviation to intra and inter alike. Measured
            # on a trained frozen checkpoint, the dispersion channel's `d_log_r0` had reached
            # -0.42 on intra and inter O-H equally -- a uniform shift, not a discrimination --
            # dropping the pair r0 from 1.22 to 0.79, opening the bonded gate to 0.999 and
            # putting **-39 kJ/mol per fragment** of dispersion between covalently bonded
            # atoms. Nothing opposed it: intramolecular classical energy is exactly degenerate
            # with the bond channel, so the fit is indifferent and the penalty alone decides.
            #
            # As a barrier there is no force pushing r0 below the prior at all, and the prior
            # becomes the floor it is written as. Restricted to inter pairs still, for the
            # original reason -- an intra pair has no business being pulled either way -- but
            # the restriction is no longer what the bonded region is relying on.
            inter = ~out.is_intra
            if bool(inter.any()):
                extra["r0"] = u.r0_weight * torch.stack([
                    (out.r0_pair[name][inter] - floor[inter].exp()).clamp(min=0.0).mean()
                    for name, floor in out.log_r0_prior_pair.items()
                ]).sum()
        if u.r0_spread_weight > 0.0:
            # Keeps a per-atom r0 near its element prior, so no single atom can run away
            # while the mean above stays put. Quadratic in log space, which is where the
            # geometric-mean combination is linear.
            spread = torch.stack([
                (out.r0[name].log() - prior).pow(2).mean()
                for name, prior in out.log_r0_prior.items()
            ])
            extra["r0_spread"] = u.r0_spread_weight * spread.sum()
        if u.intra_classical_weight > 0.0 and bool(out.is_intra.any()):
            # Keep the classical forms out of the region the bond channel is describing.
            # The weight is that channel's own envelope, so this says "do not put classical
            # energy where the bond term already is" rather than "do not put classical energy
            # between same-fragment atoms" -- an intra pair past bond_r_off is left alone, and
            # same-fragment electrostatics at range survives untouched.
            spec = self.model.pair_head.channels["bond"]
            w = pairwise_switch(out.r[out.is_intra], spec.r_on, spec.r_off)
            extra["intra_ff"] = u.intra_classical_weight * torch.stack([
                (w * e[out.is_intra] / u.energy_scale).pow(2).mean()
                for e in out.e_pair_ff.values()
            ]).sum()
        if u.env_weight > 0.0 and out.environment_norm is not None:
            # Bias back toward the fragment-confined description, so environment dependence
            # has to be earned rather than drifted into.
            extra["env"] = u.env_weight * out.environment_norm.pow(2).mean()

        # The exact free-atom limit of the on-site polarizability. Independent of the monomer
        # anchor and of the cluster batch alike -- it is a handful of one-atom systems -- so it
        # is not scaled by `anchor_weight`.
        fa, fa_metrics = free_atom_polarizability_loss(
            self.model, self._free_batch, self._free_alpha,
            weight=u.free_alpha_weight, scale=cfg.elec.polarizability_scale,
        )
        extra.update(fa)
        free_metrics = fa_metrics

        training = torch.is_grad_enabled()
        anchor_batch, anchor_idx = self._anchor_batch(training)
        if anchor_batch is None:
            self._metrics = dict(free_metrics)
            return extra
        if training:
            self._train_step += 1

        # One anchor forward feeding both the energy/force terms and the multipole terms.
        # Grad is forced on because the force term is itself a backward pass and this runs on
        # evaluation epochs too.
        with torch.enable_grad():
            anchor_out = self.model(
                anchor_batch, with_polarizability=cfg.elec.polarizability_weight > 0.0
            )
        w = u.anchor_weight

        # How far the isolated-fragment internal energy has moved since this stage started.
        # The split between `E_internal` and `energy_bond` inside `fragment_energy` is
        # unlabeled gauge -- only their sum is fitted -- so `pol` and `ct` can slide it,
        # and the bond head is left chasing a target that moved by 60-110 kJ/mol. It tracked to
        # within 2-3 kJ/mol and stopped, which is exactly the per-fragment constant that showed
        # up as a one-body offset. Fixing the gauge at its stage-entry value removes the
        # degeneracy instead of reweighting it.
        drift_metric = {}
        if self._internal_ref is not None:
            idx = torch.as_tensor(anchor_idx, device=self.device)
            drift = anchor_out.energy_internal - self._internal_ref[idx]
            d_atom = anchor_out.energy_atom - self._atom_ref[idx]
            # Both halves, summed into one penalty. Their *sum* is the labeled quantity, so
            # penalising the sum would be redundant with `ob_mae`; what has to be held is each
            # one separately, which is what makes this an assertion about the gauge rather
            # than a second copy of the energy loss.
            extra["int_drift"] = u.internal_drift_weight * (
                (drift / u.energy_scale).pow(2).mean()
                + (d_atom / u.energy_scale).pow(2).mean()
            )
            drift_metric["d_internal"] = float(drift.detach().mean()) * KJMOL_PER_HARTREE
            drift_metric["d_e_atom"] = float(d_atom.detach().mean()) * KJMOL_PER_HARTREE
        # The force term, strided. `create_graph=True` makes it the only second-order backward
        # in the fit and the single most expensive item in a step, so `anchor_force_every`
        # applies it every k-th *training* step. Evaluation always keeps it: there the force is
        # one plain backward, and `f_mae` has to mean the same thing on every validation line.
        onebody_cfg = cfg.onebody
        if training and self._train_step % self.force_every != 0:
            onebody_cfg = replace(onebody_cfg, force_weight=0.0)
        terms, self._metrics = onebody_anchor_loss(
            self.model, anchor_batch, onebody_cfg, out=anchor_out
        )
        self._metrics.update(drift_metric)
        self._metrics.update(free_metrics)
        extra.update({k: w * v for k, v in terms.items()})

        mm, mm_metrics = fragment_multipole_loss(
            anchor_out, anchor_batch,
            dipole_weight=cfg.elec.dipole_weight,
            quadrupole_weight=cfg.elec.quadrupole_weight,
            dipole_scale=cfg.elec.dipole_scale,
            quadrupole_scale=cfg.elec.quadrupole_scale,
        )
        extra.update({k: w * v for k, v in mm.items()})
        self._metrics.update(mm_metrics)

        # The only label that constrains how the charges and dipoles *move* rather than where
        # they sit. See `fragment_polarizability_loss` for why that gap matters and for what
        # this still cannot separate.
        ap, ap_metrics = fragment_polarizability_loss(
            anchor_out, anchor_batch,
            weight=cfg.elec.polarizability_weight,
            scale=cfg.elec.polarizability_scale,
        )
        extra.update({k: w * v for k, v in ap.items()})
        self._metrics.update(ap_metrics)
        return extra

    def diagnostics(self, out, batch, target):
        res = out.response
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)
        q = res.charges.detach()
        per_frag = q.new_zeros(n_frag).index_add_(0, frag, q)
        want = (
            per_frag.new_zeros(n_frag) if batch.fragment_charge is None
            else batch.fragment_charge.to(per_frag.dtype)
        )
        metrics = {"q_res": float((per_frag - want).abs().max()), **self._metrics}

        # The readout that tests whether the learned range separation reproduces what the
        # hard mask used to do. See the module docstring for how to read gate_intra.
        intra, inter = out.is_intra, ~out.is_intra
        for name, g in out.gate.items():
            g = g.detach()
            if bool(intra.any()):
                metrics[f"gate_intra_{name}"] = float(g[intra].mean())
            if bool(inter.any()):
                metrics[f"gate_inter_{name}"] = float(g[inter].mean())
            metrics[f"r0_{name}"] = float(out.r0[name].detach().mean())

        # `gate_intra` alone is misleading and cost real time to learn: a gate of 1e-5 on the
        # Pauli channel is nothing, while a gate of 0.99 on dispersion was quietly routing
        # -29 kJ/mol per fragment of "dispersion" between bonded atoms. What matters is the
        # energy, so log the energy. This is intramolecular classical energy sitting inside
        # `fragment_energy`; it should be small, and the bond channel should be doing that job.
        if bool(intra.any()):
            per_frag = KJMOL_PER_HARTREE / max(int(batch.n_fragments), 1)
            for name, e in out.e_pair_ff.items():
                metrics[f"intra_{name}"] = float(e.detach()[intra].sum()) * per_frag

        if out.environment_norm is not None:
            # How much of its surroundings the model has decided it needs. Zero at init by
            # construction, so any growth is the model asking for many-body content.
            metrics["env_norm"] = float(out.environment_norm.detach().mean())

        # The polarization/CT split. `pol_ff`/`ct_ff` are the *classical* relaxations; the
        # rest of each channel is its correction head. Watch the split, not the MAE: pol and
        # ct are both "environment-dependent correction" and are separated only by their
        # labels and by acting on disjoint pair sets, which is the same degeneracy shape as
        # the intra-classical/bond leak that the fit did not police on its own.
        #
        # `pol_ff` must stay <= 0. It is a relaxation of one functional, so it is negative by
        # the variational principle wherever the two levels share response parameters -- which
        # they do exactly at initialization. A positive value means the environment-aware
        # response parameters have drifted far enough to break that, and `env_weight` is the
        # lever.
        for name in ("pol", "ct"):
            if name in out.interaction_ff:
                metrics[f"{name}_ff"] = (
                    float(out.interaction_ff[name].detach().mean()) * KJMOL_PER_HARTREE
                )
        if out.solver:
            for name, (n_iter, converged, pd_fail) in out.solver.items():
                metrics[f"cg_{name}"] = float(n_iter)
                fails = int((~converged).sum()) + int(pd_fail.sum())
                metrics["cg_fail"] = metrics.get("cg_fail", 0.0) + float(fails)
        if out.level_ct is not None:
            # How much charge actually crossed a fragment boundary -- the direct readout of
            # what the CT level bought, and zero if the compliance head closed every channel.
            q = out.level_ct.charges.detach()
            per_frag = q.new_zeros(n_frag).index_add_(0, frag, q)
            want = (
                per_frag.new_zeros(n_frag) if batch.fragment_charge is None
                else batch.fragment_charge.to(per_frag.dtype)
            )
            metrics["q_ct"] = float((per_frag - want).abs().max())

        z = batch.atomic_numbers
        oxygen, hydrogen = z == 8, z == 1
        if bool(oxygen.any()):
            metrics["qO"] = float(q[oxygen].mean())
        if bool(oxygen.any()) and bool(hydrogen.any()):
            chi = res.chi.detach()
            metrics["dchi"] = float(chi[oxygen].mean() - chi[hydrogen].mean())
        return metrics


def train(config: Config):
    """One fit, or a sequence of them when the config declares ``stages:``.

    Staged is the intended way to reach the full decomposition, because each higher level's
    label is a **difference** against the level below it: ``pol`` is ``E_pol - E_frozen`` and
    ``ct`` is ``E_ct - E_pol``. Fitting them all at once makes a badly fit lower level and a
    badly fit new level indistinguishable, and fitting them by hand in three invocations
    invites the stale-checkpoint mistake this loop removes -- each stage warm starts from the
    previous stage's *best* checkpoint automatically, and checkpoints to its own directory so
    it can never overwrite what it started from.

    The model is rebuilt per stage rather than carried over, because turning polarization on
    genuinely changes a shape: the correction trunk widens to take the electrostatic
    environment. :func:`rsfff.train.term_loop.warm_start` zero-pads that tensor, so the new
    input starts inert and the stage begins at exactly the checkpoint it was handed.
    """
    if config.stages:
        return _train_staged(config)
    return _train_once(config)


def _train_staged(config: Config):
    print(
        f"[{config.run_name}] staged fit: "
        + " -> ".join(s.name for s in config.stages)
    , flush=True)
    result, previous = None, ""
    for i, stage in enumerate(config.stages, 1):
        staged = stage_config(config, stage, init_from=previous)
        print(
            f"\n{'=' * 78}\n"
            f"stage {i}/{len(config.stages)}: {stage.name}  "
            f"(epochs {staged.train.epochs}"
            + (f", warm start {staged.train.init_from}" if staged.train.init_from else "")
            + f")\n{'=' * 78}"
        , flush=True)
        result = _train_once(staged)
        previous = str(Path(staged.checkpoint_root) / staged.run_name / "best.pt")
        if not Path(previous).exists():
            raise RuntimeError(
                f"stage {stage.name!r} wrote no checkpoint to {previous}, so the next stage "
                f"has nothing to warm start from. This happens when a stage has no validation "
                f"frames or its loss never improved on the untrained baseline."
            )
    return result


def _train_once(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    names = list(_TARGETS.values())
    if config.unified.polarization:
        names.append("pol")
    if config.unified.charge_transfer:
        names.append("ct")
    print(
        f"[{config.run_name}] device={device} dtype={config.dtype} "
        f"targets=fragment_energy + {' + '.join('eda_' + v for v in names)}"
    , flush=True)

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if not dataset.has_fragments:
        raise ValueError("the unified model needs a `fragment_idx` column")
    if dataset._fragment_energy is None:
        raise ValueError(
            "the dataset carries no `fragment_energies` header, which is the 1-body target; "
            "use the files written by scripts/parse_roundtrip.py"
        )
    neighbor_types = dataset.unique_atomic_numbers
    reference_energies = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(torch.get_default_dtype())
    atomic_states = None
    if config.data.atomic_reference_states:
        atomic_states = AtomicStateReference.from_json(
            config.data.atomic_reference_states, neighbor_types,
            dtype=torch.get_default_dtype(),
        )
    print(
        f"loaded {len(dataset)} frames; species Z={neighbor_types}; "
        f"E0 = {dict(zip(neighbor_types, reference_energies.tolist()))}"
    , flush=True)

    anchor = None
    if config.data.monomer_path:
        anchor = load_extxyz(config.data.monomer_path, dtype=torch.get_default_dtype())
        drawn = config.unified.anchor_batch_size or len(anchor)
        print(
            f"monomer anchor: {len(anchor)} frames from {config.data.monomer_path}, "
            f"{min(drawn, len(anchor))} drawn per step"
        , flush=True)

    model = build_unified_model(
        config, neighbor_types, reference_energies, atomic_states
    ).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    r0_table = {
        name: dict(zip(neighbor_types, model.range_heads.log_r0_prior[c].exp().tolist()))
        for c, name in enumerate(model.range_heads.channel_names)
    }
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"max_rank {config.unified.max_rank}; pair list to {model.cutoff_max} A "
        f"(no fragment mask); r0 priors {r0_table} A, "
        f"alpha init {config.unified.alpha_init} 1/A"
    , flush=True)

    anchor_terms = AnchorTerms(
        model, anchor, device,
        batch_size=config.unified.anchor_batch_size, seed=config.data.seed,
        atomic_states=atomic_states, neighbor_types=neighbor_types,
        force_every=config.unified.anchor_force_every,
    )
    fit(
        model, dataset, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, fit_term=unified_fit,
        penalties=anchor_terms.penalties, diagnostics=anchor_terms.diagnostics,
        after_warm_start=lambda _: anchor_terms.snapshot_frozen_level(config),
        # A cluster force is a backward pass through the whole model, so positions have to be
        # leaves and autograd has to stay on even during evaluation.
        grad_positions=config.unified.force_weight > 0.0,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit the 1-body, electrostatic, Pauli and dispersion channels jointly "
                    "under the unified pair model with learned range separation."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
