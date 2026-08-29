"""Shared builders for the film-model tests."""

from __future__ import annotations

import torch

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.film import FragmentProjector, StateDescriptor
from rsfff.train.data import Batch


def water_cluster_batch(n_waters: int = 2, jitter: float = 0.05, seed: int = 7) -> Batch:
    """``n_waters`` waters on a line, one fragment each, mildly jittered."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor(
        [[0.0, 0.0, 0.1174], [0.0, 0.7572, -0.4696], [0.0, -0.7572, -0.4696]]
    )
    frames = []
    for k in range(n_waters):
        offset = torch.tensor([2.8 * k, 0.0, 0.0])
        frames.append(base + offset + jitter * torch.randn(base.shape, generator=g))
    positions = torch.cat(frames)
    n = 3 * n_waters
    return Batch(
        positions=positions.to(torch.get_default_dtype()),
        atomic_numbers=torch.tensor([8, 1, 1] * n_waters),
        batch_idx=torch.zeros(n, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.repeat_interleave(torch.arange(n_waters), 3),
        fragment_charge=torch.zeros(n_waters),
        fragment_two_s=torch.zeros(n_waters),
        fragment_to_batch=torch.zeros(n_waters, dtype=torch.long),
        n_fragments=n_waters,
    )


def make_projector(cross_lambdas=(0,), **over) -> FragmentProjector:
    cfg = dict(
        cutoff=5.0, n_max=3, l_max=2, neighbor_types=[1, 8],
        selected_lambdas=(0, 1, 2), density_channels=4,
    )
    cfg.update(over)
    return FragmentProjector(
        FlatLambdaSOAPFeaturizer(**cfg), cross_lambdas=cross_lambdas
    )


def make_state(batch, projector) -> StateDescriptor:
    return StateDescriptor.from_batch(
        batch, projector.species_index(batch.atomic_numbers),
        projector.featurizer.n_species,
    )
