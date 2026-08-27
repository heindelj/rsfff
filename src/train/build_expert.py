"""Assemble a :class:`rsfff.ff.expert_model.FragmentExpertModel` from the config blocks.

The ``elec:``, ``dispersion:`` and ``pauli:`` blocks are reused verbatim -- what changed with
the fragment-expert model is *who owns* the heads they configure, not how they are configured.

The slot widths are the one genuinely new thing. ``environment_features`` decides whether an
environment slot exists at all; when it does, its per-lambda widths equal the fragment slot's,
because both come from the same featurizer run on complementary halves of the same edge set.
Every head is then built with those widths and is a two-slot head; with the flag off every head
builds its single-slot form, the model is rigorously two-body, and that is the ablation.
"""

from __future__ import annotations

import torch

from ..ff.bond_energy import FragmentBondEnergy
from ..ff.dispersion import DispersionParameterHeads, build_log_priors
from ..ff.decoder import ParameterDecoder
from ..ff.expert import ApplicabilityHead, ExpertBank, FragmentExpert
from ..ff.keys import AtomKeyEncoder
from ..ff.expert_model import ClassicalSpec, FragmentExpertModel
from ..ff.fragment_state import FragmentStateEmbedding
from ..ff.multipole import irrep2_to_spherical
from ..ff.pauli import PauliMultipoleHeads, build_pauli_priors
from ..ff.range_heads import RangeSeparationHeads
from ..ff.range_priors import RANGE_CHANNELS, build_range_priors
from ..ff.slots import SlotDims
from .train_elec import build_featurizer, build_response

__all__ = ["build_expert", "build_expert_model", "slot_dims"]

#: Element symbols by atomic number, for the composition keys the expert bank dispatches on.
_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def slot_dims(featurizer, *, p0_extra: int, environment: bool) -> SlotDims:
    """Per-lambda widths of the two slots.

    The environment slot is the *same featurizer* over the complementary edges, so its widths
    match the fragment slot's -- except at lambda=0, where the fragment slot additionally
    carries the ``(Q_f, 2S_f)`` block. Charge and multiplicity are properties of the fragment,
    so that block does not appear on the environment side.
    """
    p0 = featurizer.feature_dims[0]
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)
    return SlotDims(
        p0_frag=p0 + int(p0_extra),
        p0_env=p0 if environment else 0,
        p1_frag=p1,
        p1_env=(p1 or 0) if environment else 0,
        p2_frag=p2,
        p2_env=(p2 or 0) if environment else 0,
    )


def build_expert(
    key: str,
    featurizer,
    config,
    neighbor_types,
    atomic_states=None,
    *,
    environment: bool = True,
) -> FragmentExpert:
    """One composition's **encoder**, plus its fragment-state block and applicability score.

    v3: the parameter heads are no longer here. They live once, in the shared
    :class:`rsfff.ff.decoder.ParameterDecoder` (:func:`build_decoder`), and this emits the key
    that feeds them. §2 is unchanged -- a composition still gets its own network -- but the
    network now ends at a latent, which is what makes two experts' outputs mixable at all.
    """
    xcfg = config.expert
    n_species = len(neighbor_types)

    fragment_state = FragmentStateEmbedding(
        xcfg.fragment_state_dim,
        hidden=xcfg.fragment_state_hidden,
        depth=xcfg.fragment_state_depth,
    )
    dims = slot_dims(
        featurizer, p0_extra=fragment_state.dim, environment=environment
    )

    encoder = AtomKeyEncoder(
        dims.p0_frag, dims.p0_env,
        dims.p1_frag, dims.p1_env,
        dims.p2_frag, dims.p2_env,
        n_species,
        k0=xcfg.key_dim, k1=xcfg.key_dim_l1, k2=xcfg.key_dim_l2,
        emb_dim=xcfg.key_emb_dim, hidden=xcfg.key_hidden, depth=xcfg.key_depth,
    )

    applicability = (
        ApplicabilityHead(
            dims.p0_frag, dims.p0_env,
            hidden=xcfg.applicability_hidden, depth=xcfg.applicability_depth,
        )
        if xcfg.applicability else None
    )

    return FragmentExpert(
        key,
        encoder=encoder,
        fragment_state=fragment_state,
        applicability=applicability,
    )


def build_decoder(featurizer, config, neighbor_types, atomic_states, key_dims):
    """The **one** parameter decoder every composition shares.

    Built at *key* widths with ``p_env = 0`` throughout: the two-slot split now lives entirely
    in the encoders, so the environment reaches the parameters through exactly one named,
    zero-initialized sector instead of one per head. ``env_parameters`` still finds it and the
    ablation still works.

    The heads are the same classes as before, sized differently. Every physical form, prior,
    positivity constraint and initialization therefore survives the move untouched, which is
    most of what makes this a re-plumbing rather than a new model.
    """
    from types import SimpleNamespace

    xcfg, ecfg, fcfg = config.expert, config.elec, config.features
    n_species = len(neighbor_types)
    k0, k1, k2 = key_dims

    # `build_response` reads only `feature_dims` and `backend`, so a shim at key widths reuses
    # it verbatim rather than growing a second construction path that could drift.
    shim = SimpleNamespace(
        feature_dims={0: k0, **({1: k1} if k1 else {}), **({2: k2} if k2 else {})},
        backend=featurizer.backend,
    )
    to_spherical = (
        irrep2_to_spherical(featurizer.backend.irrep6_to_voigt()) if k2 else None
    )

    response = build_response(
        shim, fcfg, ecfg, neighbor_types, atomic_states, p0_extra=0, slots=None
    )

    log_c6, log_b_disp = build_log_priors(neighbor_types, b_prior=config.dispersion.b_prior)
    disp_params = DispersionParameterHeads(
        k0, n_species, log_c6_prior=log_c6, log_b_prior=log_b_disp, p_env=0,
        emb_dim=config.dispersion.emb_dim, hidden=config.dispersion.hidden,
        depth=config.dispersion.depth,
        learn_c6=config.dispersion.learn_c6, environment_c6=config.dispersion.environment_c6,
        learn_b=config.dispersion.learn_b, environment_b=False,
    )

    log_q, log_b_pauli, mu_scale, quad_scale = build_pauli_priors(neighbor_types)
    pauli_params = PauliMultipoleHeads(
        k0, k1, n_species,
        log_q_prior=log_q, log_b_prior=log_b_pauli, dipole_scale=mu_scale,
        p2=k2, quad_scale=quad_scale, irrep2_to_spherical=to_spherical,
        p_env=0, p1_env=0, p2_env=0,
        emb_dim=config.pauli.emb_dim, hidden=config.pauli.hidden, depth=config.pauli.depth,
        equiv_channels=config.pauli.equiv_channels, max_rank=xcfg.max_rank,
        learn_q=config.pauli.learn_q, environment_q=config.pauli.environment_q,
        learn_b=config.pauli.learn_b, environment_b=False,
        learn_dipole=config.pauli.learn_dipole,
        learn_quadrupole=config.pauli.learn_quadrupole,
    )

    # `r0` reads the element and nothing else. `environment_r0` is forced off because there is
    # no feature here for it to read -- the whole point is that `r0` cannot move when the
    # description of an atom changes. See `rsfff.ff.decoder`.
    range_heads = RangeSeparationHeads(
        0, n_species, log_r0_prior=build_range_priors(neighbor_types),
        alpha_init=xcfg.alpha_init, p_env=0, channels=RANGE_CHANNELS,
        emb_dim=xcfg.r0_emb_dim, hidden=xcfg.r0_hidden, depth=xcfg.r0_depth,
        learn_r0=xcfg.learn_r0, environment_r0=False,
        learn_alpha=xcfg.learn_alpha,
    )

    bond = FragmentBondEnergy(
        k0, n_species, p1=k1, p2=k2, p_env=0, p1_env=0, p2_env=0,
        irrep2_to_spherical=to_spherical,
        emb_dim=xcfg.bond_emb_dim, hidden=xcfg.bond_hidden, depth=xcfg.bond_depth,
        equiv_channels=xcfg.bond_equiv_channels, energy_scale=xcfg.bond_energy_scale,
    )

    return ParameterDecoder(
        response=response, disp_params=disp_params, pauli_params=pauli_params,
        range_heads=range_heads, bond=bond,
    )


def build_expert_model(config, neighbor_types, reference_energies, atomic_states=None):
    """The whole model: one shared featurizer, one expert per configured composition."""
    xcfg, ecfg = config.expert, config.elec
    if int(xcfg.max_rank) != int(ecfg.max_rank):
        raise ValueError(
            f"expert.max_rank ({xcfg.max_rank}) must match elec.max_rank ({ecfg.max_rank}); "
            f"the response heads decide which multipoles exist"
        )
    featurizer = build_featurizer(config.features, ecfg, neighbor_types)
    unknown = [z for z in neighbor_types if int(z) not in _SYMBOLS]
    if unknown:
        raise ValueError(
            f"no element symbol for atomic number(s) {unknown}; add them to "
            f"rsfff.train.build_expert._SYMBOLS so the composition keys can be formed"
        )
    symbols = tuple(_SYMBOLS[int(z)] for z in neighbor_types)

    experts = {
        key: build_expert(
            key, featurizer, config, neighbor_types, atomic_states,
            environment=xcfg.environment_features,
        )
        for key in xcfg.compositions
    }

    # `alpha` and `r0` used to need tying across experts, with a comment explaining that
    # untied copies decay away silently because they receive no gradient. That whole class of
    # bug is gone: there is exactly one decoder, so there is exactly one of each.
    decoder = build_decoder(
        featurizer, config, neighbor_types, atomic_states,
        next(iter(experts.values())).encoder.key_dims,
    )

    return FragmentExpertModel(
        featurizer,
        ExpertBank(experts, symbols),
        decoder,
        reference_energies,
        max_rank=xcfg.max_rank,
        classical={
            # `elst` reads the isolated slot because its label is rigorously pairwise, and its
            # environment difference is polarization. See `rsfff.ff.expert_model`.
            "elst": ClassicalSpec(
                xcfg.elst_cutoff, xcfg.taper_width,
                environment=False, to_induction=xcfg.induction,
            ),
            "pauli": ClassicalSpec(xcfg.pauli_cutoff, xcfg.taper_width),
            "disp": ClassicalSpec(xcfg.disp_cutoff, xcfg.taper_width),
        },
        environment_features=xcfg.environment_features,
        induction=xcfg.induction,
        cg_rtol=xcfg.cg_rtol, cg_atol=xcfg.cg_atol, cg_maxiter=xcfg.cg_maxiter,
    )
