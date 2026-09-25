"""Builders for the Tersoff-model tests: the pairing helpers with the model swapped."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "film"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pairing"))
from film_helpers import water_cluster_batch  # noqa: E402,F401

from rsfff.train.build_pairing import build_pairing_model  # noqa: E402
from rsfff.train.build_tersoff import build_tersoff_model  # noqa: E402

FEAT = SimpleNamespace(cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2), density_channels=4)
FILM = dict(hidden=32, block_dim=16, head_hidden=16, equiv_channels=8)
REF = [-0.5, -75.0]


def make_model(**film_over):
    film = dict(FILM)
    film.update(film_over)
    ref = torch.tensor(REF, dtype=torch.get_default_dtype())
    return build_tersoff_model(FEAT, SimpleNamespace(**film), [1, 8], ref)


def make_pairing_model(**film_over):
    film = dict(FILM)
    film.update(film_over)
    ref = torch.tensor(REF, dtype=torch.get_default_dtype())
    return build_pairing_model(FEAT, SimpleNamespace(**film), [1, 8], ref)


def water_dimer_batch(r_oo: float = 2.9):
    """A hydrogen-bonded water dimer (donor O-H pointing at the acceptor O along x)."""
    from rsfff.train.data import Batch

    donor = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],                              # the donated hydrogen, on the O-O axis
        [-0.2400, 0.9266, 0.0],
    ])
    c, s = 0.5, 0.8660254
    acceptor = torch.tensor([
        [r_oo, 0.0, 0.0],
        [r_oo + 0.9572 * c, 0.0, 0.9572 * s],
        [r_oo + 0.9572 * c, 0.0, -0.9572 * s],
    ])
    positions = torch.cat((donor, acceptor))
    return Batch(
        positions=positions.to(torch.get_default_dtype()),
        atomic_numbers=torch.tensor([8, 1, 1, 8, 1, 1]),
        batch_idx=torch.zeros(6, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
        fragment_charge=torch.zeros(2),
        fragment_two_s=torch.zeros(2),
        fragment_to_batch=torch.zeros(2, dtype=torch.long),
        n_fragments=2,
    )


def ion_batch():
    """H3O+ and OH-, one frame each."""
    from rsfff.train.data import Batch

    h3o = torch.tensor([
        [0.0, 0.0, 0.0754], [0.9408, 0.0, -0.2010],
        [-0.4704, -0.8147, -0.2010], [-0.4704, 0.8147, -0.2010],
    ])
    oh = torch.tensor([[0.0, 0.0, -0.1072], [0.0, 0.0, 0.8577]])
    positions = torch.cat((h3o, oh))
    return Batch(
        positions=positions.to(torch.get_default_dtype()),
        atomic_numbers=torch.tensor([8, 1, 1, 1, 8, 1]),
        batch_idx=torch.tensor([0, 0, 0, 0, 1, 1]),
        n_systems=2,
        energy=torch.zeros(2),
        fragment_idx=torch.tensor([0, 0, 0, 0, 1, 1]),
        fragment_charge=torch.tensor([1.0, -1.0]),
        fragment_two_s=torch.zeros(2),
        fragment_to_batch=torch.tensor([0, 1]),
        n_fragments=2,
    )


def zundel_batch(r_oo: float = 2.4, shift: float = 0.0):
    """H5O2+ with the shared proton on the O-O axis, ``shift`` Angstrom off the midpoint."""
    from rsfff.train.data import Batch

    c, s = 0.5, 0.8660254
    o1 = torch.tensor([[0.0, 0.0, 0.0]])
    o2 = torch.tensor([[r_oo, 0.0, 0.0]])
    h_shared = torch.tensor([[0.5 * r_oo + shift, 0.0, 0.0]])
    h1 = torch.tensor([[-0.9572 * c, 0.9572 * s, 0.0], [-0.9572 * c, -0.9572 * s, 0.0]])
    h2 = torch.tensor([[r_oo + 0.9572 * c, 0.0, 0.9572 * s], [r_oo + 0.9572 * c, 0.0, -0.9572 * s]])
    positions = torch.cat((o1, h1, h_shared, o2, h2))
    return Batch(
        positions=positions.to(torch.get_default_dtype()),
        atomic_numbers=torch.tensor([8, 1, 1, 1, 8, 1, 1]),
        batch_idx=torch.zeros(7, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        fragment_charge=torch.tensor([1.0, 0.0]),
        fragment_two_s=torch.zeros(2),
        fragment_to_batch=torch.zeros(2, dtype=torch.long),
        n_fragments=2,
    )
