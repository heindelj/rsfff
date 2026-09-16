"""Harmonic frequencies from a Cartesian Hessian, and what each normal mode does to a bond.

Model-agnostic on purpose: everything takes a ``(3N, 3N)`` Hessian in Hartree/Angstrom^2 and
masses in amu, so the same code serves the film model, a reference calculation, or a test
potential.

Translations and rotations are projected out of the mass-weighted Hessian rather than being
identified afterwards by "the six smallest eigenvalues". At a minimum the two agree; at a
loosely converged geometry they do not, and a rotation leaking in at 30i cm^-1 is exactly the
kind of thing that gets mistaken for a real soft mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LocalMode",
    "VibrationalModes",
    "bond_displacements",
    "local_mode",
    "vibrational_analysis",
]

_HARTREE_J = 4.3597447222060e-18
_AMU_KG = 1.66053906892e-27
_ANG_M = 1.0e-10
_C_CM_S = 2.99792458e10

#: sqrt(Hartree / (amu * Angstrom^2)) expressed as a wavenumber: nu~ = FREQ_FACTOR * sqrt(k/m).
FREQ_FACTOR = np.sqrt(_HARTREE_J / (_AMU_KG * _ANG_M**2)) / (2.0 * np.pi * _C_CM_S)


@dataclass
class VibrationalModes:
    """The vibrational subspace only: translations and rotations have been removed.

    frequencies : (M,) wavenumbers in cm^-1, **negative for imaginary** modes
    modes       : (M, N, 3) Cartesian displacement vectors, each normalized to unit
                  mass-weighted norm -- i.e. sqrt(m_i) * mode is orthonormal. This is the
                  convention in which a displacement of fixed length means a fixed amount of
                  vibrational energy, so stepping along different modes is comparable.
    reduced_mass: (M,) amu, the usual 1 / sum_i |c_i|^2 with c the normalized Cartesian mode
    n_removed   : how many trans/rot degrees of freedom were projected out (6, or 5 linear)
    """

    frequencies: np.ndarray
    modes: np.ndarray
    reduced_mass: np.ndarray
    n_removed: int


def _trans_rot_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the trans/rot subspace in mass-weighted coordinates."""
    sqrt_m = np.sqrt(masses)
    com = (masses[:, None] * positions).sum(0) / masses.sum()
    rel = positions - com

    vectors = []
    for axis in range(3):
        t = np.zeros_like(positions)
        t[:, axis] = 1.0
        vectors.append((sqrt_m[:, None] * t).reshape(-1))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        vectors.append((sqrt_m[:, None] * np.cross(e, rel)).reshape(-1))

    basis = []
    for v in vectors:
        for b in basis:
            v = v - (b @ v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:                       # a linear molecule loses one rotation here
            basis.append(v / norm)
    return np.asarray(basis)


def vibrational_analysis(hessian, positions, masses) -> VibrationalModes:
    """Diagonalize the projected mass-weighted Hessian."""
    hessian = np.asarray(hessian, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses, dtype=float)
    n = masses.size

    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    mass_weighted = hessian * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

    basis = _trans_rot_basis(positions, masses)
    projector = np.eye(3 * n) - basis.T @ basis
    projected = projector @ mass_weighted @ projector

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projected + projected.T))
    # The projected trans/rot eigenvalues are zero to round-off; drop exactly that many by
    # taking the largest (3N - n_removed) eigenvalues, which is well defined even when a real
    # mode is imaginary and therefore also negative.
    keep = np.argsort(eigenvalues)[basis.shape[0]:]
    keep = keep[np.argsort(eigenvalues[keep])]
    eigenvalues = eigenvalues[keep]
    vectors_mw = eigenvectors[:, keep].T                       # (M, 3N), mass-weighted

    frequencies = np.sign(eigenvalues) * FREQ_FACTOR * np.sqrt(np.abs(eigenvalues))
    cartesian = (vectors_mw * inv_sqrt_m[None, :]).reshape(-1, n, 3)
    reduced_mass = 1.0 / (cartesian**2).sum(axis=(1, 2))
    return VibrationalModes(
        frequencies=frequencies,
        modes=cartesian,
        reduced_mass=reduced_mass,
        n_removed=int(basis.shape[0]),
    )


def bond_displacements(positions, bonds, mode, *, step: float) -> np.ndarray:
    """How much each bond length changes when the geometry steps along ``mode``.

    Returns the bond-length change for a ``+step`` displacement, taken as half the difference
    between the ``+step`` and ``-step`` geometries so the bond's curvature cancels. This is
    the "take a step along the mode and see which O-H moves" assignment, done numerically
    rather than by reading off an internal-coordinate B-matrix.
    """
    positions = np.asarray(positions, dtype=float)
    bonds = np.asarray(bonds, dtype=int)
    plus = positions + step * mode
    minus = positions - step * mode
    r_plus = np.linalg.norm(plus[bonds[:, 1]] - plus[bonds[:, 0]], axis=1)
    r_minus = np.linalg.norm(minus[bonds[:, 1]] - minus[bonds[:, 0]], axis=1)
    return 0.5 * (r_plus - r_minus)


@dataclass
class LocalMode:
    """One O-H oscillator isolated by isotopic decoupling.

    frequency   : cm^-1 of the mode that actually moves the target bond
    localization: fraction of that mode's total bond-length change carried by the target
                  bond. This is the check that the decoupling worked -- it should be within
                  a percent of 1, and anything lower means the substituted masses were not
                  heavy enough to separate this oscillator from its neighbours.
    character   : sum of squared bond-length changes under a unit-norm Cartesian step, the
                  same discriminator the normal-mode path uses
    mode_index  : which mode of the substituted problem was selected
    """

    frequency: float
    localization: float
    character: float
    mode_index: int


def local_mode(
    hessian, positions, masses, bonds, target: int, *, heavy_mass: float, step: float
) -> LocalMode:
    """The frequency of one bond's stretch, with every other hydrogen mass-substituted.

    Isotopic substitution leaves the Hessian untouched -- the potential does not know about
    nuclear masses -- so this is a re-diagonalization of the *same* matrix under a different
    mass vector, not another electronic-structure evaluation. Raising the mass of every
    hydrogen except the target's pushes their stretches far below the one remaining O-H and
    kills the resonant coupling that delocalizes normal modes in a cluster. What comes back is
    a genuinely local oscillator, which is the object Badger's rule is a statement about.

    ``heavy_mass = 2.014`` is deuterium and reproduces the HOD-in-D2O experiment that
    established the rule; a large value (1e5) decouples completely and is the cleaner choice
    for a pure model diagnostic. Both are available because they answer different questions.

    The selected mode is the one that changes the target bond most, rather than "the highest
    frequency": the two agree whenever the decoupling worked, and when it did not, the
    displacement test says so through ``localization`` instead of silently returning a
    neighbour's mode.
    """
    masses = np.asarray(masses, dtype=float).copy()
    bonds = np.asarray(bonds, dtype=int)
    target_h = int(bonds[target, 1])
    hydrogens = np.flatnonzero(masses < 3.0)          # before substitution: the real H
    masses[hydrogens] = heavy_mass
    masses[target_h] = 1.008

    modes = vibrational_analysis(hessian, positions, masses)
    delta_r = np.array([
        bond_displacements(positions, bonds, mode / np.linalg.norm(mode), step=step)
        for mode in modes.modes
    ])
    weight = (delta_r / step) ** 2
    index = int(np.abs(weight[:, target]).argmax())
    return LocalMode(
        frequency=float(modes.frequencies[index]),
        localization=float(weight[index, target] / weight[index].sum()),
        character=float(weight[index].sum()),
        mode_index=index,
    )
