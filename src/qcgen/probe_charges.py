"""Probe-charge placement and the electrostatic potential/field/field-gradient
they induce at each atom.

Purpose
-------
This is the geometric/electrostatic front end for parameterizing a *monomer*
response independently of the EDA (dimer) part of the model. The idea: take a
(Wigner-sampled) monomer geometry, scatter point charges on a shell around it,
and record what those charges do at every atom -- the electrostatic potential,
field, and field gradient. Sampling many random charge configurations traces out
the distribution of local electrostatic environments the monomer must respond
to; the quantum response to each configuration (density polarization, induced
multipoles, ...) is then labeled separately with pyscf using these same charges
as background point charges.

Charge placement
----------------
Charges live on the outer envelope of spheres of radius ``cutoff`` centered on
every atom: a point is admissible iff its nearest atom sits at exactly
``cutoff`` (i.e. the point lies on some atom's sphere and inside no other's).
With ``cutoff`` well beyond bonding distances this is a smooth shell wrapping the
whole molecule at a fixed standoff, so no probe charge is ever near a nuclear
singularity. ``n_charges`` points are drawn from this shell and carry a
prescribed ``total_charge`` (which need not be an integer).

Units
-----
Geometry and ``cutoff`` are in **Angstrom** (matching the rest of the pipeline);
charges are in units of the elementary charge ``e``. All returned electrostatic
quantities are in **atomic units**, consistent with :mod:`rsfff.qcgen.compute`:

  * potential      V        Hartree / e
  * field          E = -grad V        Hartree / (e * a0)
  * field gradient dE_i/dx_j          Hartree / (e * a0^2)

The field gradient is the gradient of the electric field, ``G_ij = dE_i/dx_j``.
Because ``E`` is curl-free (``E = -grad V``) this tensor is symmetric, so only
its six upper-triangular components are stored, in the order given by
:data:`FIELD_GRADIENT_ORDER`. Note ``G_ij = -d^2 V / dx_i dx_j``; the sign
distinguishes it from the potential Hessian / EFG convention.
"""

from __future__ import annotations

import numpy as np
from pyscf.data import nist

_BOHR = nist.BOHR  # Angstrom per Bohr

# Upper-triangular (row-major) packing of the symmetric 3x3 field gradient into
# the 6 trailing entries of the per-atom feature vector.
FIELD_GRADIENT_ORDER = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)

# Length of the per-atom electrostatic feature vector: 1 potential + 3 field +
# 6 unique field-gradient components.
FEATURE_LENGTH = 1 + 3 + 6


# ---------------------------------------------------------------------------
# Surface sampling
# ---------------------------------------------------------------------------
def _random_unit_vectors(n, rng):
    """``n`` directions drawn uniformly on the unit sphere, shape (n, 3)."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample_surface_points(coords_ang, cutoff=5.0, n_points=1, rng=None,
                          oversample=8, max_batches=64, tol=1e-9):
    """Draw ``n_points`` points on the molecule's outer ``cutoff`` shell.

    A point is accepted iff its nearest atom is at distance ``cutoff`` -- it lies
    on one atom's sphere of radius ``cutoff`` and no other atom is closer than
    ``cutoff``. Points are produced by rejection sampling: a random atom and a
    random direction give a candidate on that atom's sphere, which is kept when
    it clears every other atom. This samples the exposed shell uniformly by area.

    Parameters
    ----------
    coords_ang : (natoms, 3) array
        Atomic positions in Angstrom.
    cutoff : float
        Sphere radius / standoff distance in Angstrom.
    n_points : int
        Number of accepted points to return.
    rng : numpy.random.Generator, optional
        Defaults to ``np.random.default_rng()``.
    oversample : int
        Candidates generated per still-needed point in each rejection batch.
    max_batches : int
        Safety cap on rejection rounds before giving up.
    tol : float
        Relative slack on the "no closer atom" test, guarding equidistant ties.

    Returns
    -------
    (n_points, 3) array of accepted points in Angstrom.
    """
    if rng is None:
        rng = np.random.default_rng()
    coords = np.asarray(coords_ang, float)
    natoms = coords.shape[0]
    thresh = cutoff * (1.0 - tol)

    accepted = []
    n_have = 0
    for _ in range(max_batches):
        if n_have >= n_points:
            break
        n_try = max(oversample * (n_points - n_have), oversample)
        atom_idx = rng.integers(natoms, size=n_try)
        dirs = _random_unit_vectors(n_try, rng)
        cand = coords[atom_idx] + cutoff * dirs  # (n_try, 3)

        # Distance from each candidate to every atom; keep candidates whose
        # closest atom is no nearer than the cutoff (they generate at exactly
        # cutoff on their own atom, so this makes that atom the nearest).
        d = np.linalg.norm(cand[:, None, :] - coords[None, :, :], axis=2)
        keep = (d >= thresh).all(axis=1)
        good = cand[keep]
        if len(good):
            accepted.append(good)
            n_have += len(good)

    if n_have < n_points:
        raise RuntimeError(
            f"only found {n_have}/{n_points} surface points after {max_batches} "
            f"batches (cutoff={cutoff}); increase max_batches/oversample or check "
            "the geometry"
        )
    return np.concatenate(accepted, axis=0)[:n_points]


def _assign_charges(n_charges, total_charge, rng, charge_dist, spread):
    """Return ``n_charges`` charges summing to ``total_charge``.

    ``"equal"`` splits the total evenly. ``"uniform"`` / ``"normal"`` draw an
    independent per-charge magnitude (spread ``spread``) and then shift every
    charge by the same constant so the sum lands exactly on ``total_charge`` --
    a shift keeps the sampled spread intact where a rescale would not.
    """
    if charge_dist == "equal":
        return np.full(n_charges, total_charge / n_charges)
    if charge_dist == "uniform":
        c = rng.uniform(-spread, spread, size=n_charges)
    elif charge_dist == "normal":
        c = rng.normal(0.0, spread, size=n_charges)
    else:
        raise ValueError(f"unknown charge_dist: {charge_dist!r}")
    return c + (total_charge - c.sum()) / n_charges


def place_charges(coords_ang, cutoff=5.0, n_charges=1, total_charge=0.0,
                  rng=None, charge_dist="equal", spread=1.0, **surface_kwargs):
    """Place ``n_charges`` probe charges on the ``cutoff`` shell of a molecule.

    Combines :func:`sample_surface_points` (positions) with
    :func:`_assign_charges` (magnitudes). The charges sum to ``total_charge``
    (any real value); ``charge_dist`` controls whether they are identical
    (``"equal"``) or individually randomized around the mean (``"uniform"`` /
    ``"normal"``, per-charge ``spread``).

    Returns
    -------
    positions : (n_charges, 3) array
        Charge positions in Angstrom.
    charges : (n_charges,) array
        Charge magnitudes in ``e``, summing to ``total_charge``.
    """
    if rng is None:
        rng = np.random.default_rng()
    positions = sample_surface_points(
        coords_ang, cutoff=cutoff, n_points=n_charges, rng=rng, **surface_kwargs
    )
    charges = _assign_charges(n_charges, total_charge, rng, charge_dist, spread)
    return positions, charges


# ---------------------------------------------------------------------------
# Electrostatics: potential / field / field gradient at the atoms
# ---------------------------------------------------------------------------
def probe_fields(coords_ang, charge_positions_ang, charges):
    """Potential, field, and field gradient at each atom from point charges.

    All positions are given in Angstrom and converted to Bohr internally; the
    returned quantities are in atomic units (see the module docstring). Because
    probe charges live on the ``cutoff`` shell, the atom-charge separations are
    bounded well away from zero and no regularization is needed.

    Parameters
    ----------
    coords_ang : (natoms, 3) array
        Atomic positions (evaluation points) in Angstrom.
    charge_positions_ang : (ncharges, 3) array
        Point-charge positions in Angstrom.
    charges : (ncharges,) array
        Point-charge magnitudes in ``e``.

    Returns
    -------
    (natoms, 10) array
        Per atom: ``[V, Ex, Ey, Ez, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]`` in atomic
        units, with the field-gradient block ordered by
        :data:`FIELD_GRADIENT_ORDER`.
    """
    R = np.asarray(coords_ang, float) / _BOHR            # (natoms, 3), Bohr
    P = np.asarray(charge_positions_ang, float) / _BOHR  # (ncharges, 3), Bohr
    q = np.asarray(charges, float)                       # (ncharges,)

    s = R[:, None, :] - P[None, :, :]        # (natoms, ncharges, 3), Bohr
    d = np.linalg.norm(s, axis=2)            # (natoms, ncharges)
    inv_d = 1.0 / d
    inv_d3 = inv_d**3
    inv_d5 = inv_d**5

    # Potential  V = sum_k q_k / d_k
    V = (q * inv_d).sum(axis=1)                                  # (natoms,)

    # Field  E_i = -dV/dx_i = sum_k q_k s_i / d^3
    E = np.einsum("k,akc,ak->ac", q, s, inv_d3)                 # (natoms, 3)

    # Field gradient  G_ij = dE_i/dx_j = sum_k q_k (d^2 delta_ij - 3 s_i s_j)/d^5
    delta = np.eye(3)
    d2 = d**2                                                    # (natoms, ncharges)
    tensor = d2[:, :, None, None] * delta[None, None, :, :] \
        - 3.0 * s[:, :, :, None] * s[:, :, None, :]             # (natoms, nc, 3, 3)
    G = np.einsum("k,akij,ak->aij", q, tensor, inv_d5)         # (natoms, 3, 3)

    natoms = R.shape[0]
    out = np.empty((natoms, FEATURE_LENGTH))
    out[:, 0] = V
    out[:, 1:4] = E
    for col, (i, j) in enumerate(FIELD_GRADIENT_ORDER):
        out[:, 4 + col] = G[:, i, j]
    return out


def field_gradient_to_matrix(g6):
    """Expand a 6-vector (or (..., 6)) field gradient back to a symmetric 3x3."""
    g6 = np.asarray(g6, float)
    mat = np.zeros(g6.shape[:-1] + (3, 3))
    for col, (i, j) in enumerate(FIELD_GRADIENT_ORDER):
        mat[..., i, j] = g6[..., col]
        mat[..., j, i] = g6[..., col]
    return mat


def probe_features(coords_ang, cutoff=5.0, n_charges=1, total_charge=0.0,
                   rng=None, charge_dist="equal", spread=1.0, **surface_kwargs):
    """Convenience wrapper: place charges and return everything in one call.

    Returns
    -------
    features : (natoms, 10) array
        Per-atom potential/field/field-gradient in atomic units.
    positions : (n_charges, 3) array
        Probe-charge positions in Angstrom.
    charges : (n_charges,) array
        Probe-charge magnitudes in ``e``.
    """
    positions, charges = place_charges(
        coords_ang, cutoff=cutoff, n_charges=n_charges,
        total_charge=total_charge, rng=rng, charge_dist=charge_dist,
        spread=spread, **surface_kwargs,
    )
    features = probe_fields(coords_ang, positions, charges)
    return features, positions, charges
