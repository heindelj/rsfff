"""Frozen-density reference electrostatics (rsfff.qcgen.elst_ref).

Water dimer, PBE/def2-SVP (cheap): the robust-fit pair energy against the exact four-centre frozen
electrostatics; a rotated stored density against an SCF at the rotated geometry; fields against finite
differences of the potential; save/load.
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")

import pyscf.lib

pyscf.lib.num_threads(1)

from rsfff.qcgen import elst_ref as E
from rsfff.qcgen.backend import make_mf, make_mol

_XC, _BASIS = "pbe", "def2-svp"
_SYM = ["O", "H", "H"]
_XYZ = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
_ROT = np.array([[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]])   # proper rotation
_SHIFT_ANG = np.array([0.4, -0.3, 2.4])


def _scf(xyz):
    mol = make_mol(_SYM, xyz, basis=_BASIS)
    mf = make_mf(mol, _XC)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mol, mf.make_rdm1()


@pytest.fixture(scope="module")
def water():
    molA, dmA = _scf(_XYZ)
    molB, dmB = _scf(_XYZ @ _ROT.T + _SHIFT_ANG)
    A = E.Monomer.from_density(molA, dmA)
    return molA, dmA, molB, dmB, A


def test_rotation_matrix_is_proper():
    assert np.allclose(_ROT @ _ROT.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(_ROT), 1.0)


def test_pair_energy_matches_exact_frozen(water):
    molA, dmA, molB, dmB, A = water
    exact = E.exact_frozen_elst(molA, dmA, molB, dmB)
    pa, pb = A.place(), A.place(_ROT, _SHIFT_ANG / E.BOHR)
    robust = E.pair_energy(pa, pb)["total"]
    four = E.pair_energy(pa, pb, exact_ee=True)["total"]
    assert abs(robust - exact) < 2e-6
    assert abs(four - exact) < 1e-7


def test_rotated_density_matches_rotated_scf(water):
    molA, dmA, molB, dmB, A = water
    pb_rot = A.place(_ROT, _SHIFT_ANG / E.BOHR)
    pb_scf = E.Monomer.from_density(molB, dmB).place()
    pts = np.random.default_rng(0).normal(size=(12, 3)) * 3 + pb_rot.com
    pts = pts[np.linalg.norm(pts[:, None] - pb_rot.coords[None], axis=2).min(1) > 1.0]
    assert np.abs(E.potential(pb_rot, pts) - E.potential(pb_scf, pts)).max() < 1e-6


def test_field_is_minus_grad_potential(water):
    A = water[4]
    ps = [A.place(), A.place(_ROT, _SHIFT_ANG / E.BOHR)]
    pts = np.array([[0.5, 0.3, 8.0], [3.0, -1.0, 2.0], [-2.0, 2.0, -3.0]])
    h = 1e-4
    fd = np.stack([-(E.potential(ps, pts + h * e) - E.potential(ps, pts - h * e)) / (2 * h) for e in np.eye(3)], 1)
    assert np.abs(E.field(ps, pts) - fd).max() < 1e-7


def test_far_field_is_dipolar(water):
    A = water[4]
    p = A.place()
    mu = -np.einsum("ij,xji->x", p.dm, p.mol.intor("int1e_r")) + p.charges @ p.coords
    x = np.array([0.0, 0.0, 2000.0]) + p.com
    d = x - p.com
    approx = mu @ d / np.linalg.norm(d) ** 3
    assert abs(E.potential(p, x[None])[0] / approx - 1) < 1e-3


def test_save_load_roundtrip(water, tmp_path):
    A = water[4]
    A.save(tmp_path / "w.npz")
    B = E.Monomer.load(tmp_path / "w.npz")
    pa, pb = A.place(), A.place(_ROT, _SHIFT_ANG / E.BOHR)
    assert np.isclose(E.pair_energy(pa, pb)["total"], E.pair_energy(B.place(), pb)["total"], atol=1e-12)
