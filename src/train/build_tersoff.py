"""Assemble a :class:`rsfff.ff.tersoff.TersoffModel` from config blocks.

The pairing builder with the heads and model classes swapped and the ``tersoff_*`` knobs of
:class:`FilmConfig` passed through; everything else (features, projector, permanent /
response / Pauli / dispersion heads, priors, classical specs) is the pairing model's, by
construction.
"""

from __future__ import annotations

import torch

from ..ff.tersoff import TersoffHeads, TersoffModel
from .build_pairing import _get, build_pairing_model

__all__ = ["build_tersoff_model"]


def build_tersoff_model(
    features_cfg,
    film_cfg,
    neighbor_types,
    reference_energies: torch.Tensor,
    atomic_states=None,
) -> TersoffModel:
    return build_pairing_model(
        features_cfg, film_cfg, neighbor_types, reference_energies, atomic_states,
        model_cls=TersoffModel,
        heads_cls=TersoffHeads,
        heads_kwargs=dict(
            formal_charge_readout=bool(_get(film_cfg, "tersoff_formal_charge_readout", True)),
        ),
        model_kwargs=dict(
            saturation=str(_get(film_cfg, "tersoff_saturation", "waterfill")),
            saturation_width=float(_get(film_cfg, "tersoff_saturation_width", 0.02)),
            fill_steps=int(_get(film_cfg, "tersoff_fill_steps", 8)),
            formal_charge=str(_get(film_cfg, "tersoff_formal_charge", "overload")),
            formal_charge_width=float(_get(film_cfg, "tersoff_formal_charge_width", 0.1)),
            reconcile_width=float(_get(film_cfg, "tersoff_reconcile_width", 2.0e-3)),
            dual_sweeps=int(_get(film_cfg, "tersoff_dual_sweeps", 0)),
            overbinding_penalty=float(_get(film_cfg, "tersoff_overbinding_penalty", 0.0)),
        ),
    )
