"""Assemble a :class:`rsfff.ff.film.FilmModel` from config blocks.

Mirrors :mod:`rsfff.train.build_expert` for the film generation. The physical heads it builds
are the v4 classes wherever one exists (Pauli, dispersion, range separation, compliance, PSD
alpha), constructed at the **trunk latent width** instead of the raw feature width -- see
:mod:`rsfff.ff.film.heads` for why that is the only change.

Every function takes plain attribute namespaces (the config dataclasses, or a
``types.SimpleNamespace`` in tests), reading with defaults so a minimal namespace works.
"""

from __future__ import annotations

import torch

from bb.features import FlatLambdaSOAPFeaturizer
from bb.graph import DEFAULT_MAX_NUM_NEIGHBORS

from ..ff.dispersion import DispersionParameterHeads, build_log_priors
from ..ff.coupled_solve import DEFAULT_CG_CHECK_EVERY
from ..ff.film.spec import ClassicalSpec
from ..ff.film import (
    BondedParameterHead,
    ConditionedParameterNetwork,
    FilmModel,
    FilmResponseHeads,
    FragmentProjector,
    PermanentMultipoleHeads,
)
from ..ff.film.ece import ECEBondFeatures
from ..ff.film.electrical import ECEElectricalHeads
from ..ff.film.fields import FieldFeatureShift
from ..ff.film.terms import TermParameterHead
from ..ff.fragment_state import FragmentStateEmbedding
from ..ff.multipole import irrep2_to_spherical
from ..ff.nonvariational import NonvariationalModel
from ..ff.pauli import PauliMultipoleHeads, build_pauli_priors
from ..ff.range_heads import RangeSeparationHeads
from ..ff.range_priors import RANGE_CHANNELS, build_range_priors
from ..ff.response import alpha_scale_prior, build_elec_priors

__all__ = ["build_film_model"]


def _get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def build_film_model(
    features_cfg,
    film_cfg,
    neighbor_types,
    reference_energies: torch.Tensor,
) -> FilmModel:
    """The whole film model: one projector, one state block, one conditioned network."""
    neighbor_types = sorted(int(z) for z in neighbor_types)
    n_species = len(neighbor_types)
    if _get(film_cfg, "bonded_nn_residual", False):
        raise NotImplementedError(
            "film.bonded_nn_residual is a reserved hook; the decision on record is the "
            "strict physical form (Morse + cosine angle only). Implement a zero-init "
            "per-atom readout on the bonded family latent if an ablation demands it."
        )

    lambdas = tuple(int(v) for v in _get(features_cfg, "selected_lambdas", (0, 1, 2)))
    if not {0, 1, 2} <= set(lambdas):
        raise ValueError(
            "the film model needs lambda 0, 1 and 2 features (permanent dipoles and "
            "quadrupoles); set features.selected_lambdas: [0, 1, 2]"
        )
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=_get(features_cfg, "cutoff", 5.0),
        n_max=_get(features_cfg, "n_max", 6),
        l_max=_get(features_cfg, "l_max", 3),
        neighbor_types=neighbor_types,
        selected_lambdas=lambdas,
        backend=_get(features_cfg, "backend", "e3nn"),
        density_channels=_get(features_cfg, "density_channels", None),
        max_num_neighbors=_get(
            features_cfg, "max_num_neighbors", DEFAULT_MAX_NUM_NEIGHBORS
        ),
    )
    projector = FragmentProjector(
        featurizer, cross_lambdas=tuple(_get(film_cfg, "cross_lambdas", (0,)))
    )

    state_embedding = FragmentStateEmbedding(
        _get(film_cfg, "fragment_state_dim", 4),
        n_species=n_species,
        hidden=_get(film_cfg, "fragment_state_hidden", 32),
        depth=_get(film_cfg, "fragment_state_depth", 1),
    )
    d_c = state_embedding.dim + 1                     # [k_i, u_i]

    hidden = int(_get(film_cfg, "hidden", 128))       # trunk latent width
    emb_dim = int(_get(film_cfg, "emb_dim", 16))
    head_hidden = int(_get(film_cfg, "head_hidden", 64))
    head_depth = int(_get(film_cfg, "head_depth", 2))
    equiv_channels = int(_get(film_cfg, "equiv_channels", 32))
    max_rank = int(_get(film_cfg, "max_rank", 2))

    p1 = featurizer.feature_dims.get(1)
    p2 = featurizer.feature_dims.get(2)
    to_spherical = irrep2_to_spherical(featurizer.backend.irrep6_to_voigt())

    # Per-bond ECE features (rsfff.ff.film.ece) for the bonded/term heads, opt-in.
    bond_features = None
    mode = str(_get(film_cfg, "bond_features", "none")).lower()
    if mode not in ("none", "ece"):
        raise ValueError(f"film.bond_features must be 'none' or 'ece', got {mode!r}")
    use_latents = True
    if mode == "ece":
        bond_features = ECEBondFeatures(
            neighbor_types,
            n_max=int(_get(film_cfg, "ece_n_max", 4)),
            l_max=int(_get(film_cfg, "ece_l_max", 3)),
            channels=int(_get(film_cfg, "ece_channels", 8)),
            species_dim=int(_get(film_cfg, "ece_species_dim", 4)),
            morse_a=float(_get(film_cfg, "ece_morse_a", 2.0)),
            own_bond=bool(_get(film_cfg, "ece_own_bond", True)),
            dim=int(_get(film_cfg, "ece_dim", 16)),
            hidden=int(_get(film_cfg, "ece_hidden", 32)),
            depth=int(_get(film_cfg, "ece_depth", 1)),
            backend=_get(features_cfg, "backend", "e3nn"),
            neighbour_decay=bool(_get(film_cfg, "ece_neighbour_decay", True)),
        )
        use_latents = bool(_get(film_cfg, "ece_latents", True))
        if not use_latents and _get(film_cfg, "field_features", False):
            raise ValueError("film.field_features shifts the bonded latents, which "
                             "film.ece_latents: false removes; keep the latents or drop the "
                             "field features")
    bond_feat_dim = 0 if bond_features is None else bond_features.dim

    bonded_head = BondedParameterHead(
        hidden, neighbor_types,
        hidden=int(_get(film_cfg, "bonded_hidden", 64)),
        depth=int(_get(film_cfg, "bonded_depth", 1)),
        emb_dim=int(_get(film_cfg, "bonded_emb_dim", 8)),
        impropers=bool(_get(film_cfg, "impropers", False)),
        atom_typing=str(_get(film_cfg, "atom_typing", "element")),
        bond_feat_dim=bond_feat_dim,
        use_latents=use_latents,
    )

    # Torsions / couplings (rsfff.ff.film.terms) and the O(F^2) field features
    # (rsfff.ff.film.fields): built only when switched on, so every existing config and
    # checkpoint keeps its exact parameter set.
    term_head = None
    if _get(film_cfg, "torsions", False) or _get(film_cfg, "couplings", False):
        if _get(film_cfg, "couplings", False) and not _get(film_cfg, "torsions", False):
            raise ValueError("film.couplings needs film.torsions (the torsion couplings ride "
                             "on the torsion head); set torsions: true")
        term_head = TermParameterHead(
            hidden, len(bonded_head.types),
            torsions=bool(_get(film_cfg, "torsions", False)),
            couplings=bool(_get(film_cfg, "couplings", False)),
            opposite=bool(_get(film_cfg, "opposite_coupling", False)),
            n_max=int(_get(film_cfg, "torsion_nmax", 4)),
            hidden=int(_get(film_cfg, "term_hidden", 32)),
            depth=int(_get(film_cfg, "term_depth", 1)),
            emb_dim=int(_get(film_cfg, "bonded_emb_dim", 8)),
            torsion_scale=float(_get(film_cfg, "torsion_scale", 1.0e-3)),
            bond_feat_dim=bond_feat_dim,
            use_latents=use_latents,
        )
    field_features = None
    if _get(film_cfg, "field_features", False):
        field_features = FieldFeatureShift(
            hidden, p1, p2, to_spherical,
            ranks=tuple(_get(film_cfg, "field_ranks", (0, 1, 2))),
            channels=int(_get(film_cfg, "field_channels", 8)),
            n_products=int(_get(film_cfg, "field_products", 16)),
            hidden=int(_get(film_cfg, "field_hidden", 32)),
            scales=tuple(_get(film_cfg, "field_scales", (0.05, 0.02, 0.01))),
        )

    # `film.model`: "film" (converged coupled solve) or "nonvariational" (fixed-K unrolled
    # solve with learned mutual damping, docs/fff_nonvariational.md).
    model_kind = str(_get(film_cfg, "model", "film"))
    if model_kind not in ("film", "nonvariational"):
        raise ValueError(f"film.model must be 'film' or 'nonvariational', got {model_kind!r}")
    nonvariational = model_kind == "nonvariational"

    log_z, log_b_elec, q0_prior = build_elec_priors(neighbor_types)
    permanent_heads = PermanentMultipoleHeads(
        hidden, p1, p2, n_species,
        q0_prior=q0_prior,
        irrep2_to_spherical=to_spherical,
        max_rank=max_rank,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels,
    )

    elec_mode = str(_get(film_cfg, "electrical_features", "none")).lower()
    if elec_mode not in ("none", "ece"):
        raise ValueError(f"film.electrical_features must be 'none' or 'ece', got {elec_mode!r}")
    if elec_mode == "ece" and bond_features is None:
        raise ValueError("film.electrical_features: ece needs film.bond_features: ece")
    electrical_heads = None
    if elec_mode == "ece":
        electrical_heads = ECEElectricalHeads(
            bond_features, neighbor_types, [z for z, _ in bonded_head.types],
            bonded_head.bond_table.detach(),
            latent_dim=hidden if bool(_get(film_cfg, "elec_latents", True)) else 0,
            emb_dim=int(_get(film_cfg, "elec_emb_dim", 8)),
            hidden=int(_get(film_cfg, "elec_hidden", 64)),
            depth=int(_get(film_cfg, "elec_depth", 2)),
            s_init=float(_get(film_cfg, "s_init", 0.5)),
            flow_kappa=float(_get(film_cfg, "elec_flow_kappa", 0.15)),
            resp_latent_dim=hidden,
            channels=bool(_get(film_cfg, "elec_channels", True)),
            channel_onset=float(_get(film_cfg, "elec_channel_onset", 0.5)),
            channel_width=float(_get(film_cfg, "elec_channel_width", 0.3)),
        )

    response_heads = FilmResponseHeads(
        hidden, p2, n_species,
        log_z_prior=log_z, log_b_prior=log_b_elec,
        irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels,
        learn_alpha=electrical_heads is None,
        eta_init=float(_get(film_cfg, "eta_init", 0.5)),
        eta_floor=float(_get(film_cfg, "eta_floor", 0.05)),
        psd_floor=float(_get(film_cfg, "psd_floor", 1e-4)),
        compliance_cutoff=featurizer.cutoff,
        s_init=float(_get(film_cfg, "s_init", 0.5)),
        # the nonvariational model's heads; absent (no parameters) under `model: film`, so
        # film checkpoints keep loading whatever the config defaults say
        induced_width=nonvariational and bool(_get(film_cfg, "induced_width", True)),
        induced_width_mode=str(_get(film_cfg, "induced_width_mode", "broaden")),
        induced_width_bias_init=float(_get(film_cfg, "induced_width_bias_init", -3.0)),
        induced_quadrupoles=nonvariational and bool(_get(film_cfg, "induced_quadrupoles", True)),
        cquad_init=float(_get(film_cfg, "cquad_init", 1.0)),
        cquad_floor=float(_get(film_cfg, "cquad_floor", 1.0e-4)),
        alpha_scale=(
            alpha_scale_prior(neighbor_types) if bool(_get(film_cfg, "alpha_prior", True)) else None
        ),
    )

    log_q, log_b_pauli, mu_scale, quad_scale = build_pauli_priors(neighbor_types)
    pauli_heads = PauliMultipoleHeads(
        hidden, p1, n_species,
        log_q_prior=log_q, log_b_prior=log_b_pauli, dipole_scale=mu_scale,
        p2=p2, quad_scale=quad_scale, irrep2_to_spherical=to_spherical,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels, max_rank=max_rank,
        environment_q=True, environment_b=False,
    )

    log_c6, log_b_disp = build_log_priors(
        neighbor_types, b_prior=_get(film_cfg, "disp_b_prior", "per_element")
    )
    disp_heads = DispersionParameterHeads(
        hidden, n_species,
        log_c6_prior=log_c6, log_b_prior=log_b_disp,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        environment_c6=True, environment_b=False,
    )

    network = ConditionedParameterNetwork(
        p_in=featurizer.feature_dims[0],
        p_cross=projector.cross_dims[0],
        d_c=d_c,
        bonded_head=bonded_head,
        permanent_heads=permanent_heads,
        response_heads=response_heads,
        pauli_heads=pauli_heads,
        disp_heads=disp_heads,
        block_dim=int(_get(film_cfg, "block_dim", 64)),
        hidden=hidden,
        depth=int(_get(film_cfg, "depth", 2)),
        conditioning_mode=str(_get(film_cfg, "conditioning_mode", "film")),
        film_hidden=int(_get(film_cfg, "film_hidden", 32)),
        film_depth=int(_get(film_cfg, "film_depth", 1)),
        gate_a0=float(_get(film_cfg, "gate_a0", 0.5)),
        term_head=term_head,
        field_features=field_features,
        bond_features=bond_features,
        electrical_heads=electrical_heads,
    )

    # `film.nonbonded`: "range_separated" (Fermi-switched; the default, so configs pickled
    # into older checkpoints keep loading) or "exclusions" (hard 1-2/1-3 exclusions, no range
    # heads, strictly non-reactive). See rsfff.ff.film.model.
    nonbonded = str(_get(film_cfg, "nonbonded", "range_separated"))
    range_heads = None
    if nonbonded == "range_separated":
        range_heads = RangeSeparationHeads(
            0, n_species,
            log_r0_prior=build_range_priors(neighbor_types),
            alpha_init=float(_get(film_cfg, "alpha_init", 40.0)),
            p_env=0, channels=RANGE_CHANNELS,
            environment_r0=False,
        )

    cls, extra = FilmModel, {}
    if nonvariational:
        cls, extra = NonvariationalModel, dict(
            n_iter=int(_get(film_cfg, "n_iter", 3)),
            iterate_weights=bool(_get(film_cfg, "iterate_weights", True)),
            with_residual=bool(_get(film_cfg, "solve_residual", False)),
        )

    return cls(
        projector,
        state_embedding,
        network,
        range_heads,
        reference_energies,
        **extra,
        nonbonded=nonbonded,
        exclude_through=int(_get(film_cfg, "exclude_through", 3)),
        max_rank=max_rank,
        classical={
            "elst": ClassicalSpec(
                float(_get(film_cfg, "elst_cutoff", 12.0)),
                float(_get(film_cfg, "taper_width", 1.0)),
                environment=False,
            ),
            "pauli": ClassicalSpec(
                float(_get(film_cfg, "pauli_cutoff", 7.0)),
                float(_get(film_cfg, "taper_width", 1.0)),
            ),
            "disp": ClassicalSpec(
                float(_get(film_cfg, "disp_cutoff", 10.0)),
                float(_get(film_cfg, "taper_width", 1.0)),
            ),
        },
        induction=bool(_get(film_cfg, "induction", True)),
        cg_rtol=float(_get(film_cfg, "cg_rtol", 1.0e-9)),
        cg_atol=float(_get(film_cfg, "cg_atol", 1.0e-12)),
        cg_maxiter=int(_get(film_cfg, "cg_maxiter", 100)),
        cg_check_every=int(_get(film_cfg, "cg_check_every", DEFAULT_CG_CHECK_EVERY)),
    )
