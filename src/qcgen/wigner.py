"""Finite-temperature Wigner sampling from a harmonic Hessian.

This is the normal-mode / thermal-Gaussian sampler from
``scripts/sample_wigner.py``, refactored to consume plain numpy arrays (Hessian,
masses, geometry) instead of a psi4 wavefunction, so it works with the
pyscf/gpu4pyscf compute core. The math is unchanged: displacements are drawn in
mass-weighted normal coordinates with per-mode standard deviation

    sigma_q = (1 / sqrt(2 w)) * sqrt(coth(w / (2 kB T)))

which reduces to the pure zero-point (ZPE) width at ``T = 0``.

All internal work is done in atomic units; inputs are converted from the
conventional units the compute stage stores (Angstrom, amu).
"""

from __future__ import annotations

import numpy as np
from pyscf.data import nist

# atomic units per conventional unit
_AMU2AU = nist.AMU2AU  # electron masses per amu
_BOHR = nist.BOHR  # Angstrom per Bohr
_KB_AU = nist.BOLTZMANN / nist.HARTREE2J  # Hartree per Kelvin


def sample_wigner(coords_ang, masses_amu, hessian, n_samples=500,
                  temperature=300.0, rng=None):
    """Draw ``n_samples`` geometries from the finite-T Wigner distribution.

    Parameters
    ----------
    coords_ang : (natoms, 3) array
        Equilibrium (optimized) geometry in Angstrom.
    masses_amu : (natoms,) array
        Atomic masses in amu.
    hessian : (3*natoms, 3*natoms) array
        Cartesian Hessian in Hartree/Bohr^2 (atomic units).
    n_samples : int
        Number of geometries to generate.
    temperature : float
        Sampling temperature in Kelvin. ``0`` gives pure zero-point sampling.
    rng : numpy.random.Generator, optional
        Seeded generator for reproducibility; defaults to ``np.random.default_rng()``.

    Returns
    -------
    (n_samples, natoms, 3) array of geometries in Angstrom.
    """
    if rng is None:
        rng = np.random.default_rng()

    coords_bohr = np.asarray(coords_ang, float) / _BOHR
    natoms = coords_bohr.shape[0]

    masses_au = np.asarray(masses_amu, float) * _AMU2AU
    m_3n = np.repeat(masses_au, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)

    # Mass-weighted Hessian -> normal modes.
    hessian = np.asarray(hessian, float).reshape(3 * natoms, 3 * natoms)
    mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
    evals, evecs = np.linalg.eigh(mw_hessian)

    # Keep vibrational modes only (drop translations/rotations near zero, and
    # any tiny negative numerical eigenvalues).
    vib = np.where(evals > 1e-6)[0]
    omega = np.sqrt(evals[vib])  # angular frequencies (a.u.)
    L = evecs[:, vib]

    if temperature > 0.0:
        coth = 1.0 / np.tanh(omega / (2.0 * _KB_AU * temperature))
        scale = np.sqrt(coth)
    else:
        scale = np.ones_like(omega)
    sigma_q = (1.0 / np.sqrt(2.0 * omega)) * scale

    geoms = np.empty((n_samples, natoms, 3))
    coord0_bohr = coords_bohr
    for k in range(n_samples):
        q = rng.normal(loc=0.0, scale=sigma_q)
        dx = (inv_sqrt_m * (L @ q)).reshape(natoms, 3)
        geoms[k] = (coord0_bohr + dx) * _BOHR
    return geoms
