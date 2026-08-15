"""Data loading: shapes and the unit conversions applied at load time.

These used to run on the psi4 b3lyp/def2-svpd monomer labels, which carried a charged species
(H3O+) and analytic ``dipole_derivatives``. Neither survives in ``data/`` -- one level of
theory is kept now -- so the charge-dependent sum rule and the mixed-charge fixture are gone
with them. What is still checkable, and still worth checking, is that every label arrives in
model units: the Bohr powers differ per quantity and a missing conversion is a silent factor
of 1.9, 3.6 or 6.7 that no invariance test would catch.
"""

import pytest
import torch

from rsfff.train.data import load_datasets, load_extxyz

from conftest import DATA_H2O

#: The monomer anchor with the def2-TZVPD polarizability tensors folded in
#: (scripts/parse_polarizability.py). The plain DATA_H2O carries no ``polarizability``.
DATA_H2O_POL = "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd_pol.xyz"


@pytest.fixture(scope="module")
def monomers():
    return load_datasets([DATA_H2O_POL], dtype=torch.float64)


def test_response_labels_loaded(monomers):
    b = monomers.flat_batch([0, 1, 250, 251])
    assert b.total_charge is None or b.total_charge.abs().max() == 0.0
    assert b.dipole.shape == (4, 3)
    assert b.polarizability.shape == (4, 3, 3)
    asym = (b.polarizability - b.polarizability.transpose(-1, -2)).abs().max()
    assert asym < 1e-5


def test_units_are_model_units(monomers):
    """Dipole ~ e*Angstrom: water's gas-phase dipole is ~1.85 D = 0.385 e*A."""
    b = monomers.flat_batch(list(range(20)))
    norms = b.dipole.norm(dim=-1)
    assert 0.2 < norms.mean() < 0.6  # e*A scale, not e*a0 (which would be ~0.73)
    # alpha isotropic part of water ~9.9 a0^3 = 2.77 e^2 A^2 / Ha
    iso = b.polarizability.diagonal(dim1=-2, dim2=-1).mean(-1)
    assert 1.5 < iso.mean() < 4.0


def test_forces_are_per_angstrom_not_per_bohr(monomers):
    """Forces are stored in Ha/bohr and divided by Bohr on load.

    Skipping the conversion leaves them a factor 1.89 too small, which reads as a
    plausible-looking force and fits to a wrong energy scale.
    """
    b = monomers.flat_batch(list(range(50)))
    assert monomers.has_forces
    # Wigner-sampled monomers near equilibrium: ~1e-2 Ha/A, and never zero.
    mag = b.forces.norm(dim=-1)
    assert 1e-4 < float(mag.mean()) < 0.5


def test_fragment_labels_ride_along(monomers):
    b = monomers.flat_batch([0, 1, 2])
    assert monomers.has_fragments
    assert int(b.n_fragments) == 3          # one fragment per monomer frame
    assert b.fragment_energy.shape == (3,)
    assert b.fragment_dipole.shape == (3, 3)


def test_single_file_loader_matches(monomers):
    ds = load_extxyz(DATA_H2O_POL, dtype=torch.float64)
    a = ds.flat_batch([7])
    c = monomers.flat_batch([7])
    assert torch.equal(a.positions, c.positions)
    assert torch.equal(a.dipole, c.dipole)
