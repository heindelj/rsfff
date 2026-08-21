"""Rebuild the v1 model from a checkpoint alone. **Do not edit.**

``checkpoints/water_staged/best.pt`` embeds the full :class:`rsfff.train.config.Config` it was
trained under, so nothing here reads a YAML file. That matters: the config *blocks* the live tree
uses have moved on, and a v1 checkpoint should not depend on a v1 config still sitting on disk in a
loadable state. It depends only on itself and on the two reference-data JSONs.

One consequence, recorded because it is a real constraint on the live tree: the embedded config is
a pickled dataclass, so :class:`rsfff.train.config.Config` and ``UnifiedConfig`` must keep their
module path and class names. Unpickling restores ``__dict__`` directly without calling
``__init__``, so *adding* or *removing* fields on those classes is safe -- an old instance simply
keeps the attributes it was pickled with, which is exactly what the archived builder reads.
Renaming or relocating the classes is not.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...mlip.reference_states import AtomicStateReference
from ...train.data import load_reference_energies
from .build import build_unified_model

__all__ = ["load_v1_checkpoint"]

#: Repo root, four levels up from ``src/ff/v1/loader.py``.
_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def load_v1_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
):
    """``(model, config, neighbor_types)`` -- the v1 model with its trained weights.

    ``strict=True`` on purpose. A missing or unexpected tensor here means the shared heads this
    package imports from the live tree have drifted, which is the one failure mode the archive
    exists to catch; loading anyway would silently benchmark a partly-initialized model.
    """
    state = torch.load(_resolve(checkpoint_path), map_location="cpu", weights_only=False)
    config = state["config"]
    neighbor_types = tuple(int(z) for z in state["neighbor_types"])
    dtype = dtype or (torch.float64 if config.dtype == "float64" else torch.float32)

    reference_energies = load_reference_energies(
        _resolve(config.data.reference_energies), neighbor_types
    ).to(dtype)
    atomic_states = None
    if config.data.atomic_reference_states:
        atomic_states = AtomicStateReference.from_json(
            _resolve(config.data.atomic_reference_states), neighbor_types, dtype=dtype
        )

    model = build_unified_model(config, neighbor_types, reference_energies, atomic_states)
    model.load_state_dict(state["model_state"], strict=True)
    return model.to(device=device, dtype=dtype).eval(), config, neighbor_types
