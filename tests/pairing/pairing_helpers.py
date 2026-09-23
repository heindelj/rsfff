"""Builders for the pairing-model tests: a small model from the config namespaces."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "film"))
from film_helpers import water_cluster_batch  # noqa: E402,F401

from rsfff.train.build_pairing import build_pairing_model  # noqa: E402


def make_model(**film_over):
    feat = SimpleNamespace(cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2), density_channels=4)
    film = dict(hidden=32, block_dim=16, head_hidden=16, equiv_channels=8)
    film.update(film_over)
    ref = torch.tensor([-0.5, -75.0], dtype=torch.get_default_dtype())
    return build_pairing_model(feat, SimpleNamespace(**film), [1, 8], ref)
