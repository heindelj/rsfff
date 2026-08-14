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

import torch

from ..ff.dispersion import DispersionParameterHeads, build_log_priors
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
from .config import Config, load_config
from .data import (
    load_datasets,
    load_extxyz,
    load_reference_energies,
    split_indices,
)
from .loss import fragment_multipole_loss, onebody_anchor_loss
from .term_loop import fit
from .train_eem import resolve_device
from .train_elec import build_featurizer, build_response

#: EDA component fitted by each interaction channel.
_TARGETS = {"elst": "cls_elec", "pauli": "mod_pauli", "disp": "disp"}

_LOG_KEYS = (
    "loss", "ob_mae", "elst_mae", "pauli_mae", "disp_mae",
    "a_mae", "f_mae", "dip_mae", "quad_mae",
    "anchor_e", "anchor_f", "dipole", "quad", "intra_ff",
    "q_res", "qO", "dchi", "internal", "bond",
    "gate_inter_elst", "gate_intra_disp",
    "intra_elst", "intra_pauli", "intra_disp",
    "r0_elst", "r0_pauli", "r0_disp", "env_norm",
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
        },
        range_channels=RANGE_CHANNELS if ucfg.pair_range_separation else (),
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
        environment=environment,
        max_rank=ucfg.max_rank,
        classical={
            "elst": ClassicalSpec(ucfg.elst_cutoff, ucfg.taper_width),
            "pauli": ClassicalSpec(ucfg.pauli_cutoff, ucfg.taper_width),
            "disp": ClassicalSpec(ucfg.disp_cutoff, ucfg.taper_width),
        },
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
        "internal": float(out.energy_internal.detach().mean()),
        "bond": float(out.energy_bond.detach().mean()),
    }
    weights = {
        "elst": u.elst_weight, "pauli": u.pauli_weight, "disp": u.disp_weight
    }
    for name, key in _TARGETS.items():
        if key not in batch.eda:
            raise KeyError(
                f"the dataset has no eda_{key} label, which the {name} channel fits; "
                f"available: {sorted(batch.eda)}"
            )
        err = out.interaction[name] - batch.eda[key]
        loss = loss + weights[name] * (err / u.energy_scale).pow(2).mean()
        metrics[f"{name}_mae"] = float(err.detach().abs().mean()) * KJMOL_PER_HARTREE
    return loss, metrics, batch.fragment_energy


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
    """

    def __init__(self, model, anchor, device, batch_size=0, seed=0):
        self.model = model
        self.anchor = anchor            # MoleculeDataset, or None
        self.device = device
        self.batch_size = int(batch_size)
        self._gen = torch.Generator().manual_seed(int(seed))
        self._metrics: dict[str, float] = {}
        self._eval_batch = None

    def _anchor_batch(self, training: bool):
        """A monomer batch with positions ready for the force term, or None."""
        if self.anchor is None:
            return None
        n = len(self.anchor)
        take = n if not self.batch_size else min(self.batch_size, n)
        if take == n and self._eval_batch is not None:
            return self._eval_batch
        if not training:
            if self._eval_batch is None:
                self._eval_batch = self._make(range(take))
            return self._eval_batch
        idx = torch.randperm(n, generator=self._gen)[:take].tolist()
        return self._make(idx)

    def _make(self, indices):
        batch = self.anchor.flat_batch(indices).to(self.device)
        batch.positions.requires_grad_(True)
        return batch

    def penalties(self, out, batch, cfg: Config):
        u = cfg.unified
        extra = {}
        if u.corr_l2_weight > 0.0:
            corr = torch.stack(
                [out.interaction_corr[n] for n in out.interaction_corr], dim=0
            )
            extra["corr_l2"] = u.corr_l2_weight * (corr / u.energy_scale).pow(2).mean()
        if u.r0_weight > 0.0:
            # A linear pull on r0, per channel: moving energy out of the physical backbone
            # and into the network has to be earned.
            #
            # Restricted to **inter-fragment** pairs, and that restriction is load-bearing.
            # Applied to every pair it pulls the *intra* r0 down too, and there is nothing to
            # push back: whatever intramolecular classical energy appears is absorbed by the
            # bond channel, which can represent it equally well either way. The two are
            # degenerate, so the penalty alone decides -- and it decides in favour of dragging
            # the classical form into the bonding region, which is gauge leakage running
            # backwards. Measured with the unrestricted version: the dispersion channel slid
            # its r0 down until it was putting -29 kJ/mol per fragment of "intramolecular
            # dispersion" between bonded atoms, ten times the ob_mae it was being judged on.
            inter = ~out.is_intra
            if bool(inter.any()):
                extra["r0"] = u.r0_weight * torch.stack(
                    [v[inter].mean() for v in out.r0_pair.values()]
                ).sum()
        if u.r0_spread_weight > 0.0:
            # Keeps a per-atom r0 near its element prior, so no single atom can run away
            # while the mean above stays put. Quadratic in log space, which is where the
            # geometric-mean combination is linear.
            spread = torch.stack(
                [(v.log() - out.log_r0_prior).pow(2).mean() for v in out.r0.values()]
            )
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

        anchor_batch = self._anchor_batch(torch.is_grad_enabled())
        if anchor_batch is None:
            self._metrics = {}
            return extra

        # One anchor forward feeding both the energy/force terms and the multipole terms.
        # Grad is forced on because the force term is itself a backward pass and this runs on
        # evaluation epochs too.
        with torch.enable_grad():
            anchor_out = self.model(anchor_batch)
        w = u.anchor_weight
        terms, self._metrics = onebody_anchor_loss(
            self.model, anchor_batch, cfg.onebody, out=anchor_out
        )
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

        z = batch.atomic_numbers
        oxygen, hydrogen = z == 8, z == 1
        if bool(oxygen.any()):
            metrics["qO"] = float(q[oxygen].mean())
        if bool(oxygen.any()) and bool(hydrogen.any()):
            chi = res.chi.detach()
            metrics["dchi"] = float(chi[oxygen].mean() - chi[hydrogen].mean())
        return metrics


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    print(
        f"[{config.run_name}] device={device} dtype={config.dtype} "
        f"targets=fragment_energy + {' + '.join('eda_' + v for v in _TARGETS.values())}"
    )

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
    )

    anchor = None
    if config.data.monomer_path:
        anchor = load_extxyz(config.data.monomer_path, dtype=torch.get_default_dtype())
        drawn = config.unified.anchor_batch_size or len(anchor)
        print(
            f"monomer anchor: {len(anchor)} frames from {config.data.monomer_path}, "
            f"{min(drawn, len(anchor))} drawn per step"
        )

    model = build_unified_model(
        config, neighbor_types, reference_energies, atomic_states
    ).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    r0_table = {
        name: dict(zip(neighbor_types, model.range_heads.log_r0_prior.exp().tolist()))
        for name in model.range_heads.channel_names
    }
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"max_rank {config.unified.max_rank}; pair list to {model.cutoff_max} A "
        f"(no fragment mask); r0 priors {r0_table['elst']} A, "
        f"alpha init {config.unified.alpha_init} 1/A"
    )

    anchor_terms = AnchorTerms(
        model, anchor, device,
        batch_size=config.unified.anchor_batch_size, seed=config.data.seed,
    )
    fit(
        model, dataset, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, fit_term=unified_fit,
        penalties=anchor_terms.penalties, diagnostics=anchor_terms.diagnostics,
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
