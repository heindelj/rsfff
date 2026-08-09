"""Correctness of the QM/MM probe-response labeling.

Two checks, both on a tiny H2/STO-3G system so they stay cheap:
  * with all probe charges set to zero the embedded calculation must reproduce
    the plain (no-field) reference energy/dipole/quadrupole/polarizability;
  * a real probe charge must actually polarize the monomer (induced dipole
    shifts toward the charge), confirming the embedding is live.
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")

# Single-threaded: multi-threaded pyscf DFT segfaults under the macOS libomp
# duplicate-runtime conflict when several SCF calcs run in one process.
import pyscf.lib

pyscf.lib.num_threads(1)

from rsfff.qcgen.compute import compute_reference_data
from rsfff.qcgen.probe_response import compute_response_under_charges

_XC, _BASIS = "pbe", "sto-3g"
_SYM = ["H", "H"]
_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def test_zero_charge_reduces_to_plain_reference():
    ref = compute_reference_data(_SYM, _COORDS, 0, 0, _XC, _BASIS)
    # A single zero-magnitude charge contributes no potential -> identical SCF.
    res = compute_response_under_charges(
        _SYM, _COORDS, np.array([[0.0, 0.0, 6.0]]), np.array([0.0]),
        0, 0, _XC, _BASIS,
    )
    assert res["converged"]
    assert res["energy"] == pytest.approx(ref["energy"], abs=1e-8)
    assert res["dipole"] == pytest.approx(ref["dipole"], abs=1e-7)
    assert res["quadrupole"] == pytest.approx(ref["quadrupole"], abs=1e-7)
    assert res["polarizability"] == pytest.approx(ref["polarizability"], abs=1e-5)


def test_probe_charge_polarizes_monomer():
    # A positive charge on the +z axis pulls electron density toward +z; the
    # electron cloud's centroid shifts +z, so the induced dipole points -z.
    ref = compute_reference_data(_SYM, _COORDS, 0, 0, _XC, _BASIS)
    res = compute_response_under_charges(
        _SYM, _COORDS, np.array([[0.0, 0.0, 6.0]]), np.array([0.8]),
        0, 0, _XC, _BASIS,
    )
    # Homonuclear H2 has zero permanent dipole; any nonzero dipole here is
    # induced by the probe charge (the embedding is live and polarizing).
    assert np.linalg.norm(ref["dipole"]) < 1e-6
    mu = res["dipole"]
    assert abs(mu[2]) > 1e-3 and mu[2] < 0.0            # induced, along -z
    assert abs(mu[0]) < 1e-8 and abs(mu[1]) < 1e-8       # axial symmetry preserved
