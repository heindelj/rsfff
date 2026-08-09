"""Quantum monomer response to external probe charges (pyscf QM/MM embedding).

The companion to :mod:`rsfff.qcgen.probe_charges`. Given a monomer geometry and a
set of probe point charges (from ``probe_charges.place_charges``), embed the
charges as a static background potential in the QM Hamiltonian
(``pyscf.qmmm.mm_charge``) and label the *polarized monomer*: energy, forces on
the QM atoms, dipole, quadrupole, and dipole polarizability. Paired with the
classical potential/field/field-gradient descriptors that the same charges
induce at each atom, these become the (input, label) records for fitting the
monomer response independently of the EDA/dimer part of the model.

What the QM/MM energy contains
------------------------------
``mm_charge`` adds the interaction of the QM nuclei and QM electron density with
the fixed MM charges, but *not* the MM-MM Coulomb energy or any vdW/bonded term.
The reported forces are on the QM atoms only (they do include the pull of the MM
charges). The dipole/quadrupole are moments of the QM subsystem (QM nuclei +
electrons) in the polarizing field -- exactly the induced monomer multipoles.

Units and conventions match :mod:`rsfff.qcgen.compute` (atomic units; forces =
-dE/dR in Hartree/Bohr; dipole in e*a0; quadrupole = nuclear-electronic second
moment in e*a0^2; polarizability alpha_ij = dmu_i/dF_j in a0^3). The MM charge
positions are supplied in Angstrom (as produced by ``probe_charges``).
"""

from __future__ import annotations

import numpy as np
from pyscf import qmmm
from pyscf.data import nist

from .backend import HAVE_GPU, make_mf, make_mol, to_numpy
from .compute import (
    _dipole,
    _polarizability_cpscf,
    _quadrupole,
    _warn_once,
)

_BOHR = nist.BOHR

# Same sign convention as compute._dipole_with_field: the field enters as
# +field . <mu|r|nu>, calibrated so the induced dipole is parallel to the field.
_FIELD_SIGN = 1.0


def _wrap_mm(mf, mm_coords_ang, mm_charges):
    """Embed MM point charges (positions in Angstrom) into ``mf``'s 1e potential."""
    return qmmm.mm_charge(
        mf, np.asarray(mm_coords_ang, float), np.asarray(mm_charges, float),
        unit="Angstrom",
    )


def _dipole_with_field_mm(mol, xc, spin, mm_coords_ang, mm_charges, field):
    """QM dipole (a.u.) under both the MM charges and a uniform electric field.

    Rebuilds the mean field so the finite-field polarizability fallback sees the
    same MM embedding as the reference SCF, then adds ``+field . r`` to the core
    Hamiltonian on top of the MM potential.
    """
    mf = _wrap_mm(make_mf(mol, xc, spin), mm_coords_ang, mm_charges)
    h0 = mf.get_hcore()
    ao_dip = mol.intor("int1e_r", comp=3)
    pert = _FIELD_SIGN * np.einsum("x,xij->ij", field, ao_dip)
    h1 = h0 + pert
    mf.get_hcore = lambda *a, **k: h1
    mf.kernel()
    return to_numpy(mf.dip_moment(mol, mf.make_rdm1(), unit="au", verbose=0))


def _polarizability_fd_mm(mol, xc, spin, mm_coords_ang, mm_charges, field_step):
    """Polarizability by central difference of the QM dipole under +/- field."""
    alpha = np.zeros((3, 3))
    for axis in range(3):
        f = np.zeros(3)
        f[axis] = field_step
        mu_p = _dipole_with_field_mm(mol, xc, spin, mm_coords_ang, mm_charges, +f)
        mu_m = _dipole_with_field_mm(mol, xc, spin, mm_coords_ang, mm_charges, -f)
        alpha[:, axis] = (mu_p - mu_m) / (2.0 * field_step)
    return 0.5 * (alpha + alpha.T)


def compute_response_under_charges(
    symbols, coords, mm_coords, mm_charges, charge, spin, xc, basis,
    response="cpscf", field_step=1e-3,
):
    """Label one geometry's monomer response to a set of background charges.

    Parameters
    ----------
    symbols, coords : QM monomer (coords in Angstrom).
    mm_coords : (M, 3) array
        Probe-charge positions in Angstrom.
    mm_charges : (M,) array
        Probe-charge magnitudes in ``e``.
    charge, spin : QM subsystem charge and pyscf spin (Nalpha - Nbeta).
    xc, basis : DFT functional and basis set.
    response : "cpscf" (analytic, default) or "finite-difference" for the
        polarizability, mirroring :func:`compute.compute_reference_data`.
    field_step : finite-field step (a.u.) for the finite-difference path.

    Returns
    -------
    dict with keys ``energy``, ``forces``, ``dipole``, ``quadrupole``,
    ``polarizability``, ``response_method`` (which polarizability path ran), and
    ``converged`` -- all host numpy / atomic units.
    """
    if HAVE_GPU:  # pragma: no cover - GPU-only path
        # pyscf.qmmm wraps a pyscf mean field; the gpu4pyscf objects that
        # backend.make_mf returns on CUDA are not compatible. Wire this to
        # gpu4pyscf.qmmm before using it on the GPU backend.
        raise NotImplementedError(
            "probe_response uses pyscf.qmmm and is CPU-only; the GPU backend is "
            "not wired yet"
        )

    mol = make_mol(symbols, coords, charge, spin, basis)
    mf = _wrap_mm(make_mf(mol, xc, spin), mm_coords, mm_charges)
    energy = float(mf.kernel())

    forces = -to_numpy(mf.nuc_grad_method().kernel())
    dipole = _dipole(mf, mol)
    quadrupole = _quadrupole(mf, mol)

    used = response
    if response == "cpscf":
        try:
            polarizability = _polarizability_cpscf(mf, mol, spin)
        except (NotImplementedError, ImportError, AttributeError) as exc:
            used = "finite-difference"
            _warn_once(f"CPSCF polarizability unavailable ({type(exc).__name__}: "
                       f"{exc}); falling back to finite field")
    if used != "cpscf":
        polarizability = _polarizability_fd_mm(
            mol, xc, spin, mm_coords, mm_charges, field_step
        )

    return {
        "energy": energy,
        "forces": forces,
        "dipole": dipole,
        "quadrupole": quadrupole,
        "polarizability": polarizability,
        "response_method": used,
        "converged": bool(mf.converged),
    }
