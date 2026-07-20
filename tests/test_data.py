"""Data loading: shapes, units, and the dipole-derivative translational sum rule."""

import pytest
import torch

from rsfff.train.data import load_datasets, load_extxyz

from conftest import DATA_H2O, DATA_H3O


@pytest.fixture(scope="module")
def mixed():
    return load_datasets([DATA_H2O, DATA_H3O], dtype=torch.float64)


def test_response_labels_loaded(mixed):
    b = mixed.flat_batch([0, 1, 500, 501])
    assert b.total_charge.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert b.dipole.shape == (4, 3)
    assert b.polarizability.shape == (4, 3, 3)
    assert b.dipole_derivatives.shape == (b.positions.shape[0], 3, 3)
    # polarizability is symmetric to psi4 precision
    asym = (b.polarizability - b.polarizability.transpose(-1, -2)).abs().max()
    assert asym < 1e-5


def test_units_are_model_units(mixed):
    """Dipole ~ e*Angstrom: water's gas-phase dipole is ~1.85 D = 0.385 e*A."""
    b = mixed.flat_batch(list(range(20)))
    norms = b.dipole.norm(dim=-1)
    assert 0.2 < norms.mean() < 0.6  # e*A scale, not e*a0 (which would be ~0.73)
    # alpha isotropic part of water ~9.9 a0^3 = 2.77 e^2 A^2 / Ha
    iso = b.polarizability.diagonal(dim1=-2, dim2=-1).mean(-1)
    assert 1.5 < iso.mean() < 4.0


def test_dipole_derivative_sum_rule(mixed):
    """Translational sum rule sum_i d mu_b / d R_{i,a} = Q delta_ab.

    Validates the (atom, d/dR, mu) layout, the charge header, and the (absence
    of) unit conversion in one shot.
    """
    b = mixed.flat_batch([0, 250, 500, 750])
    eye = torch.eye(3, dtype=torch.float64)
    for m in range(b.n_systems):
        s = b.dipole_derivatives[b.batch_idx == m].sum(dim=0)
        assert (s - b.total_charge[m] * eye).abs().max() < 1e-4


def test_single_file_loader_matches(mixed):
    ds = load_extxyz(DATA_H2O, dtype=torch.float64)
    a = ds.flat_batch([7])
    c = mixed.flat_batch([7])
    assert torch.equal(a.positions, c.positions)
    assert torch.equal(a.dipole, c.dipole)
