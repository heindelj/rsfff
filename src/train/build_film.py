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

from ..features.features import FlatLambdaSOAPFeaturizer
from ..ff.dispersion import DispersionParameterHeads, build_log_priors
from ..ff.expert_model import ClassicalSpec
from ..ff.film import (
    BondedParameterHead,
    ConditionedParameterNetwork,
    FilmModel,
    FilmResponseHeads,
    FragmentProjector,
    PermanentMultipoleHeads,
)
from ..ff.fragment_state import FragmentStateEmbedding
from ..ff.multipole import irrep2_to_spherical
from ..ff.pauli import PauliMultipoleHeads, build_pauli_priors
from ..ff.range_heads import RangeSeparationHeads
from ..ff.range_priors import RANGE_CHANNELS, build_range_priors
from ..ff.response import build_elec_priors
from ..neighbors import DEFAULT_MAX_NUM_NEIGHBORS

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

    bonded_head = BondedParameterHead(
        hidden, neighbor_types,
        hidden=int(_get(film_cfg, "bonded_hidden", 64)),
        depth=int(_get(film_cfg, "bonded_depth", 1)),
        emb_dim=int(_get(film_cfg, "bonded_emb_dim", 8)),
    )

    log_z, log_b_elec, q0_prior = build_elec_priors(neighbor_types)
    permanent_heads = PermanentMultipoleHeads(
        hidden, p1, p2, n_species,
        q0_prior=q0_prior,
        irrep2_to_spherical=to_spherical,
        max_rank=max_rank,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels,
    )

    response_heads = FilmResponseHeads(
        hidden, p2, n_species,
        log_z_prior=log_z, log_b_prior=log_b_elec,
        irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels,
        eta_init=float(_get(film_cfg, "eta_init", 0.5)),
        eta_floor=float(_get(film_cfg, "eta_floor", 0.05)),
        psd_floor=float(_get(film_cfg, "psd_floor", 1e-4)),
        compliance_cutoff=featurizer.cutoff,
        s_init=float(_get(film_cfg, "s_init", 0.5)),
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
    )

    range_heads = RangeSeparationHeads(
        0, n_species,
        log_r0_prior=build_range_priors(neighbor_types),
        alpha_init=float(_get(film_cfg, "alpha_init", 40.0)),
        p_env=0, channels=RANGE_CHANNELS,
        environment_r0=False,
    )

    return FilmModel(
        projector,
        state_embedding,
        network,
        range_heads,
        reference_energies,
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
    )
