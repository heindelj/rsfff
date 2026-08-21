"""``build_unified_model`` as it stood for ``checkpoints/water_staged/best.pt``. **Do not edit.**

Lifted verbatim from ``rsfff.train.train_unified``; only the imports are rewritten for the new
location. See :mod:`rsfff.ff.v1` for why this is pinned rather than shared.
"""

from __future__ import annotations

from ...mlip.reference_states import AtomicStateReference  # noqa: F401
from ...mlip.unified_head import ChannelSpec, UnifiedPairHead
from ...train.config import Config
from ...train.train_elec import build_featurizer, build_response
from ..dispersion import DispersionParameterHeads, build_log_priors
from ..environment import N_PAIR_INVARIANTS
from ..multipole import irrep2_to_spherical
from ..pauli import PauliMultipoleHeads, build_pauli_priors
from ..range_priors import RANGE_CHANNELS, build_range_priors
from .atomic_energy import AtomicStateEnergy
from .unified import (
    ClassicalSpec,
    EnvironmentResidual,
    FragmentStateEmbedding,
    RangeSeparationHeads,
    UnifiedPairModel,
)

__all__ = ["build_unified_model"]


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
            offset_scale=ucfg.atomic_energy_offset_scale,
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
        # The side channel the external field enters through. Zero unless induction is on,
        # which keeps the trunk's input width -- and hence the checkpoint -- unchanged.
        extra_dim=N_PAIR_INVARIANTS if ucfg.induction else 0,
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
        induction=ucfg.induction,
        cg_rtol=ucfg.cg_rtol, cg_atol=ucfg.cg_atol, cg_maxiter=ucfg.cg_maxiter,
    )
