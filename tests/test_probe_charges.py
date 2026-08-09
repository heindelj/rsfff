"""Correctness tests for probe-charge placement and the induced potential/field/
field-gradient.

The electrostatics are checked two ways: against the closed form for a single
point charge, and against a finite-difference of the potential itself (so the
field and field gradient are verified as true derivatives of V, in atomic
units). Placement is checked against the shell-membership constraint and the
prescribed total charge.
"""

import numpy as np
import pytest
from pyscf.data import nist

from rsfff.qcgen.probe_charges import (
    FEATURE_LENGTH,
    field_gradient_to_matrix,
    place_charges,
    probe_features,
    probe_fields,
    sample_surface_points,
)

_BOHR = nist.BOHR


def _water():
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ])
    return symbols, coords


def test_surface_points_on_shell():
    _, coords = _water()
    cutoff = 5.0
    pts = sample_surface_points(coords, cutoff=cutoff, n_points=200,
                                rng=np.random.default_rng(0))
    d = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2)
    # Every point clears every atom by at least the cutoff...
    assert (d >= cutoff * (1 - 1e-6)).all()
    # ...and sits exactly on some atom's sphere (its nearest atom is at cutoff).
    assert np.allclose(d.min(axis=1), cutoff, atol=1e-6)


def test_total_charge_and_count():
    _, coords = _water()
    for dist in ("equal", "uniform", "normal"):
        pos, q = place_charges(coords, cutoff=5.0, n_charges=17,
                               total_charge=-0.37, rng=np.random.default_rng(1),
                               charge_dist=dist, spread=0.5)
        assert pos.shape == (17, 3)
        assert q.shape == (17,)
        assert q.sum() == pytest.approx(-0.37)


def test_single_charge_analytic():
    # One atom at the origin, one charge on the +z axis 5 A away.
    coords = np.zeros((1, 3))
    cpos = np.array([[0.0, 0.0, 5.0]])
    q = np.array([1.3])
    feat = probe_fields(coords, cpos, q)[0]

    d = 5.0 / _BOHR  # separation in Bohr
    # V = q/d; the atom is on the -z side of the charge, so E points -z.
    assert feat[0] == pytest.approx(q[0] / d)
    assert feat[1:4] == pytest.approx([0.0, 0.0, -q[0] / d**2])

    # G_ij = q (d^2 delta_ij - 3 s_i s_j)/d^5 with s = (0,0,-d):
    #   G_xx = G_yy = q/d^3,  G_zz = -2q/d^3,  off-diagonals 0.
    G = field_gradient_to_matrix(feat[4:])
    expected = np.diag([q[0] / d**3, q[0] / d**3, -2.0 * q[0] / d**3])
    assert G == pytest.approx(expected, abs=1e-12)
    assert np.trace(G) == pytest.approx(0.0, abs=1e-12)  # Laplace: div E = 0


def _potential(coords_ang, cpos, q):
    """Bare potential (a.u.) at each row of ``coords_ang`` -- FD reference."""
    R = np.asarray(coords_ang, float) / _BOHR
    P = np.asarray(cpos, float) / _BOHR
    d = np.linalg.norm(R[:, None, :] - P[None, :, :], axis=2)
    return (q / d).sum(axis=1)


def test_field_and_gradient_match_finite_difference():
    rng = np.random.default_rng(3)
    _, coords = _water()
    cpos, q = place_charges(coords, cutoff=5.0, n_charges=12,
                            total_charge=0.4, rng=rng, charge_dist="normal",
                            spread=0.6)
    feat = probe_fields(coords, cpos, q)

    h = 1e-4  # Angstrom displacement for FD
    natoms = coords.shape[0]
    E_fd = np.zeros((natoms, 3))
    G_fd = np.zeros((natoms, 3, 3))
    for i in range(3):
        dp = coords.copy(); dp[:, i] += h
        dm = coords.copy(); dm[:, i] -= h
        Vp = _potential(dp, cpos, q)
        Vm = _potential(dm, cpos, q)
        # E = -dV/dx; convert the Angstrom step to Bohr for an a.u. derivative.
        E_fd[:, i] = -(Vp - Vm) / (2 * h / _BOHR)
        # second derivative of V -> field gradient G = -d2V/dxi dxj
        Ep = probe_fields(dp, cpos, q)[:, 1:4]
        Em = probe_fields(dm, cpos, q)[:, 1:4]
        G_fd[:, :, i] = (Ep - Em) / (2 * h / _BOHR)

    assert feat[:, 1:4] == pytest.approx(E_fd, abs=1e-6)
    G = field_gradient_to_matrix(feat[:, 4:])
    assert G == pytest.approx(G_fd, abs=1e-6)


def test_feature_shape():
    _, coords = _water()
    feat, pos, q = probe_features(coords, cutoff=5.0, n_charges=8,
                                  total_charge=0.0, rng=np.random.default_rng(4))
    assert feat.shape == (3, FEATURE_LENGTH)
    assert pos.shape == (8, 3)
    assert q.shape == (8,)
