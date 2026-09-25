"""Assemble a :class:`rsfff.ff.pairing.PairingModel` from config blocks.

The film builder with the bonded head swapped for the pairing heads
(:mod:`rsfff.ff.pairing.heads`) and the pairing knobs of :class:`FilmConfig` passed through.
Everything the two builders share is one function here, so a change to the permanent /
response / Pauli / dispersion heads reaches both models.
"""

from __future__ import annotations

import torch

from ..features.features import FlatLambdaSOAPFeaturizer
from ..ff.dispersion import DispersionParameterHeads, build_log_priors
from ..ff.expert_model import ClassicalSpec
from ..ff.film import FilmResponseHeads, FragmentProjector, PermanentMultipoleHeads
from ..ff.fragment_state import FragmentStateEmbedding
from ..ff.multipole import irrep2_to_spherical
from ..ff.pairing import PairingHeads, PairingModel, PairingParameterNetwork
from ..ff.pairing.reference import ChargedAtomicReference
from ..ff.pairing.electronic_state import capacity_table, chi_eta_table
from ..ff.pairing.heads import build_pairing_priors
from ..ff.pauli import DEFAULT_PAULI_DIPOLE_SCALE, DEFAULT_PAULI_QUAD_SCALE, PauliMultipoleHeads, build_pauli_priors
from ..ff.range_heads import RangeSeparationHeads
from ..ff.range_priors import RANGE_CHANNELS, build_range_priors
from ..ff.response import build_elec_priors
from ..neighbors import DEFAULT_MAX_NUM_NEIGHBORS

__all__ = ["MODEL_BUILDERS", "build_pairing_model", "build_model"]


def _get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def build_pairing_model(
    features_cfg,
    film_cfg,
    neighbor_types,
    reference_energies: torch.Tensor,
    atomic_states=None,
    *,
    model_cls=PairingModel,
    heads_cls=PairingHeads,
    heads_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> PairingModel:
    """``atomic_states`` (an ``AtomicStateReference``) turns on the charge-dependent atomic
    reference ``E0(Z, q)``; without it every atom is referenced neutrally, as in the film.

    ``model_cls`` / ``heads_cls`` (+ their extra kwargs) let a model that shares the pairing
    assembly -- :mod:`rsfff.train.build_tersoff` -- reuse this builder whole.
    """
    neighbor_types = sorted(int(z) for z in neighbor_types)
    n_species = len(neighbor_types)

    lambdas = tuple(int(v) for v in _get(features_cfg, "selected_lambdas", (0, 1, 2)))
    if not {0, 1, 2} <= set(lambdas):
        raise ValueError(
            "the pairing model needs lambda 0, 1 and 2 features; set "
            "features.selected_lambdas: [0, 1, 2]"
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
    # the pairing model conditions on the topology pass's (q_i, u_i), not on a fragment key;
    # the state embedding is kept for the constructor signature only

    hidden = int(_get(film_cfg, "hidden", 128))
    emb_dim = int(_get(film_cfg, "emb_dim", 16))
    head_hidden = int(_get(film_cfg, "head_hidden", 64))
    head_depth = int(_get(film_cfg, "head_depth", 2))
    equiv_channels = int(_get(film_cfg, "equiv_channels", 32))
    max_rank = int(_get(film_cfg, "max_rank", 2))

    p1 = featurizer.feature_dims.get(1)
    p2 = featurizer.feature_dims.get(2)
    to_spherical = irrep2_to_spherical(featurizer.backend.irrep6_to_voigt())

    log_z, log_b_elec, q0_prior = build_elec_priors(neighbor_types)
    permanent_heads = PermanentMultipoleHeads(
        hidden, p1, p2, n_species,
        q0_prior=q0_prior, irrep2_to_spherical=to_spherical, max_rank=max_rank,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth, equiv_channels=equiv_channels,
    )
    response_heads = FilmResponseHeads(
        hidden, p2, n_species,
        log_z_prior=log_z, log_b_prior=log_b_elec,
        irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth, equiv_channels=equiv_channels,
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

    valence, log_qv, log_bv, log_kappa = build_pairing_priors(neighbor_types)
    capacity = capacity_table(neighbor_types)
    chi, eta = chi_eta_table(neighbor_types, atomic_states)
    # the valence dipole / quadrupole heads borrow the Pauli output scales: same operator,
    # same role (a learning-rate convenience, not a prior)
    mu_scale_v = torch.tensor([DEFAULT_PAULI_DIPOLE_SCALE.get(int(z), 0.3) for z in neighbor_types])
    quad_scale_v = torch.tensor([DEFAULT_PAULI_QUAD_SCALE.get(int(z), 0.5) for z in neighbor_types])
    head_kwargs = dict(
        valence=valence, log_q_prior=log_qv, log_b_prior=log_bv, log_kappa_prior=log_kappa,
        dipole_scale=mu_scale_v, p2=p2, quad_scale=quad_scale_v,
        irrep2_to_spherical=to_spherical,
        emb_dim=emb_dim, hidden=head_hidden, depth=head_depth,
        equiv_channels=equiv_channels, max_rank=max_rank,
        environment_q=True,
        environment_b=bool(_get(film_cfg, "pairing_environment_b", True)),
        environment_kappa=bool(_get(film_cfg, "pairing_environment_kappa", True)),
        environment_valence=bool(_get(film_cfg, "pairing_environment_valence", True)),
        capacity=capacity, chi=chi, eta=eta,
    )
    head_kwargs.update(heads_kwargs or {})
    pairing_heads = heads_cls(hidden, p1, n_species, **head_kwargs)
    topology_hidden = int(_get(film_cfg, "topology_hidden", 64))
    topology_heads = heads_cls(topology_hidden, p1, n_species, **head_kwargs)

    network = PairingParameterNetwork(
        pairing_heads=pairing_heads,
        topology_heads=topology_heads,
        topology_hidden=topology_hidden,
        topology_depth=int(_get(film_cfg, "topology_depth", 2)),
        p_in=featurizer.feature_dims[0],
        p_cross=projector.cross_dims[0],
        d_c=2,                                    # [formal charge, unpaired count]
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
    taper = float(_get(film_cfg, "taper_width", 1.0))
    reference = ChargedAtomicReference.from_states(reference_energies, atomic_states)
    return model_cls(
        projector, state_embedding, network, range_heads, reference,
        max_rank=max_rank,
        classical={
            "elst": ClassicalSpec(float(_get(film_cfg, "elst_cutoff", 12.0)), taper, environment=False),
            "pauli": ClassicalSpec(float(_get(film_cfg, "pauli_cutoff", 7.0)), taper),
            "disp": ClassicalSpec(float(_get(film_cfg, "disp_cutoff", 10.0)), taper),
        },
        induction=bool(_get(film_cfg, "induction", True)),
        cg_rtol=float(_get(film_cfg, "cg_rtol", 1.0e-9)),
        cg_atol=float(_get(film_cfg, "cg_atol", 1.0e-12)),
        cg_maxiter=int(_get(film_cfg, "cg_maxiter", 100)),
        cg_check_every=int(_get(film_cfg, "cg_check_every", 1)),
        pairing_cutoff=float(_get(film_cfg, "pairing_cutoff", 4.0)),
        pairing_taper=float(_get(film_cfg, "pairing_taper", 1.0)),
        temperature=float(_get(film_cfg, "pairing_temperature", 0.002)),
        range_gate=str(_get(film_cfg, "range_gate", "bond_order")),
        include_13=bool(_get(film_cfg, "include_13", True)),
        bo_tol=float(_get(film_cfg, "bo_tol", 1.0e-8)),
        bo_maxiter=int(_get(film_cfg, "bo_maxiter", 60)),
        bo_device=str(_get(film_cfg, "bo_device", "cpu")),
        state_cache=bool(_get(film_cfg, "state_cache", True)),
        **(model_kwargs or {}),
    )


def _build_film(features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states=None):
    from .build_film import build_film_model

    return build_film_model(features_cfg, film_cfg, neighbor_types, reference_energies)


def _build_tersoff(features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states=None):
    from .build_tersoff import build_tersoff_model

    return build_tersoff_model(
        features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states
    )


#: ``film.model`` -> builder ``(features_cfg, film_cfg, neighbor_types, reference_energies,
#: atomic_states)``. The film references neutrally and ignores ``atomic_states``.
MODEL_BUILDERS = {
    "film": _build_film,
    "pairing": build_pairing_model,
    "tersoff": _build_tersoff,
}


def build_model(features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states=None):
    """``film.model`` dispatch (:data:`MODEL_BUILDERS`) from the same config blocks."""
    kind = str(_get(film_cfg, "model", "film"))
    try:
        builder = MODEL_BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"film.model must be one of {tuple(MODEL_BUILDERS)}, got {kind!r}"
        ) from None
    return builder(features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states)
