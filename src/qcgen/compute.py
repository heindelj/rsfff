"""Quantum-chemistry operations for the reference-data pipeline.

Backend-agnostic (see :mod:`rsfff.qcgen.backend`): geometry optimization,
harmonic frequencies, and per-structure property labeling
(energy/forces/dipole/quadrupole/polarizability/dipole-derivatives), all at a
chosen ``xc``/``basis`` (production target: ``wB97M-V``/``def2-svpd``).

Property conventions (atomic units):
  * forces        = -dE/dR             (Hartree/Bohr)
  * dipole        total dipole         (e*a0)
  * quadrupole    Cartesian 2nd moment (e*a0^2), nuclear - electronic, origin 0
  * polarizability alpha_ij = dmu_i/dF_j via finite field (a0^3)
  * dipole_derivs [atom, dR, dmu] = dmu/dR via finite nuclear displacement (e)

Design choices worth knowing:
  * Polarizability uses a finite electric field injected into ``get_hcore``; only
    the perturbed *density's* dipole is read back, so the (field-inconsistent)
    analytic gradient is never used -- correct and backend-portable.
  * Dipole derivatives (APT) use finite nuclear displacement of the analytic
    dipole rather than the field/interchange route, sidestepping the fact that
    pyscf's nuclear gradient does not see a custom ``get_hcore`` field term.
  * The harmonic Hessian is analytic where supported and falls back to
    finite-difference of analytic gradients for cases pyscf lacks (notably UKS
    + non-local correlation, which raises NotImplementedError in pyscf 2.14).
"""

from __future__ import annotations

import numpy as np
from pyscf.data import nist

from .backend import as_backend_array, make_mf, make_mol, to_numpy

_BOHR = nist.BOHR  # Angstrom per Bohr

# Sign of the finite-field dipole perturbation added to the core Hamiltonian.
# Calibrated so the induced dipole is parallel to the field (positive diagonal
# polarizability); see tests/verification.
_FIELD_SIGN = 1.0


# ---------------------------------------------------------------------------
# Low-level SCF helpers
# ---------------------------------------------------------------------------
def _run_scf(mol, xc, spin, dm0=None):
    """Run one DFT SCF; return ``(mf, energy, dm1)`` with dm on the backend."""
    mf = make_mf(mol, xc, spin)
    e = mf.kernel(dm0=dm0)
    return mf, float(e), mf.make_rdm1()


def _dipole(mf, mol):
    """Total dipole (a.u.) of the mean-field's current density, as host numpy."""
    return to_numpy(mf.dip_moment(mol, mf.make_rdm1(), unit="au", verbose=0))


def _dipole_with_field(mol, xc, spin, field, dm0=None):
    """Dipole (a.u.) of the density converged under a uniform electric ``field``.

    The field enters as ``_FIELD_SIGN * field . <mu|r|nu>`` added to the core
    Hamiltonian. Only the resulting density's dipole is returned.
    """
    mf = make_mf(mol, xc, spin)
    h0 = mf.get_hcore()
    ao_dip = mol.intor("int1e_r", comp=3)  # (3, nao, nao), host numpy
    pert = _FIELD_SIGN * np.einsum("x,xij->ij", field, ao_dip)
    h1 = h0 + as_backend_array(pert)
    mf.get_hcore = lambda *a, **k: h1
    mf.kernel(dm0=dm0)
    return to_numpy(mf.dip_moment(mol, mf.make_rdm1(), unit="au", verbose=0))


def _quadrupole(mf, mol):
    """Cartesian second-moment quadrupole (e*a0^2): nuclear - electronic, origin 0."""
    dm = to_numpy(mf.make_rdm1())
    if dm.ndim == 3:  # UKS returns (2, nao, nao)
        dm = dm[0] + dm[1]
    Z = mol.atom_charges()
    R = mol.atom_coords()  # Bohr
    nuc = np.einsum("a,ai,aj->ij", Z, R, R)
    nao = mol.nao
    rr = mol.intor("int1e_rr").reshape(3, 3, nao, nao)
    elec = np.einsum("ijmn,mn->ij", rr, dm)
    return nuc - elec


# ---------------------------------------------------------------------------
# Stage 1: geometry optimization
# ---------------------------------------------------------------------------
def optimize_geometry(symbols, coords, charge, spin, xc, basis,
                      maxsteps=100, conv_params=None):
    """Optimize ``symbols``/``coords`` at ``xc``/``basis``.

    Returns ``(opt_coords_ang, energy)``. Uses geomeTRIC through pyscf's
    ``geometric_solver`` (works with gpu4pyscf mean-field objects too).
    """
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize

    mol = make_mol(symbols, coords, charge, spin, basis)
    mf = make_mf(mol, xc, spin)
    params = {"maxsteps": maxsteps}
    if conv_params:
        params.update(conv_params)
    mol_eq = geometric_optimize(mf, **params)

    opt_coords = mol_eq.atom_coords(unit="Angstrom")
    _, energy, _ = _run_scf(mol_eq, xc, spin)
    return opt_coords, energy


# ---------------------------------------------------------------------------
# Stage 2: harmonic frequencies (Hessian)
# ---------------------------------------------------------------------------
def _analytic_hessian(mf):
    """Analytic Cartesian Hessian as (3N, 3N); raise if unsupported."""
    H = to_numpy(mf.Hessian().kernel())  # (natm, natm, 3, 3)
    natm = H.shape[0]
    return np.transpose(H, (0, 2, 1, 3)).reshape(3 * natm, 3 * natm)


def _fd_hessian(mol, xc, spin, step=1e-3):
    """Finite-difference Hessian from analytic gradients (central, 6N grads).

    Robust fallback for functionals/references whose analytic Hessian pyscf
    lacks (e.g. UKS + NLC). ``step`` is in Bohr.
    """
    base = mol.atom_coords()  # Bohr
    natm = base.shape[0]
    n3 = 3 * natm
    H = np.zeros((n3, n3))
    _, _, dm0 = _run_scf(mol, xc, spin)
    for k in range(n3):
        a, c = divmod(k, 3)
        gpm = []
        for sign in (+1.0, -1.0):
            disp = base.copy()
            disp[a, c] += sign * step
            m2 = mol.set_geom_(disp, unit="Bohr", inplace=False)
            mf2 = make_mf(m2, xc, spin)
            mf2.kernel(dm0=dm0)
            gpm.append(to_numpy(mf2.nuc_grad_method().kernel()).reshape(n3))
        H[k] = (gpm[0] - gpm[1]) / (2.0 * step)
    return 0.5 * (H + H.T)


def harmonic_frequencies(symbols, coords, charge, spin, xc, basis,
                         fd_step=1e-3):
    """Compute the harmonic Hessian and normal-mode frequencies.

    Returns a dict with ``hessian`` (3N,3N, Hartree/Bohr^2), ``masses_amu``,
    ``coords_ang``, ``symbols``, ``freqs_cm`` (signed: negative = imaginary),
    ``energy``, and ``hessian_method`` ("analytic" or "finite-difference").
    """
    mol = make_mol(symbols, coords, charge, spin, basis)
    mf, energy, _ = _run_scf(mol, xc, spin)

    try:
        hessian = _analytic_hessian(mf)
        method = "analytic"
    except NotImplementedError:
        hessian = _fd_hessian(mol, xc, spin, step=fd_step)
        method = "finite-difference"

    masses_amu = np.asarray(mol.atom_mass_list())
    freqs_cm = _hessian_to_freqs(hessian, masses_amu)
    return {
        "symbols": list(symbols),
        "coords_ang": np.asarray(coords),
        "masses_amu": masses_amu,
        "hessian": hessian,
        "freqs_cm": freqs_cm,
        "energy": energy,
        "hessian_method": method,
    }


def _hessian_to_freqs(hessian, masses_amu):
    """Signed harmonic frequencies (cm^-1); negative entries are imaginary modes."""
    m_3n = np.repeat(np.asarray(masses_amu) * nist.AMU2AU, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)
    mw = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
    evals = np.linalg.eigvalsh(mw)
    # omega (a.u.) -> cm^-1: hartree-to-cm^-1 of the energy hbar*omega.
    au2cm = nist.HARTREE2J / (nist.PLANCK * nist.LIGHT_SPEED_SI) / 100.0
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * au2cm
    return freqs


# ---------------------------------------------------------------------------
# Stage 4: full property labeling of one structure
# ---------------------------------------------------------------------------
def compute_reference_data(symbols, coords, charge, spin, xc, basis,
                           field_step=1e-3, disp_step=1e-3):
    """Label one geometry with energy/forces/dipole/quadrupole/polarizability/APT.

    ``field_step`` (a.u. field) drives the polarizability; ``disp_step`` (Bohr)
    drives the dipole-derivative (APT) finite differences. Returns a dict of
    host numpy arrays in atomic units matching the extxyz schema.
    """
    mol = make_mol(symbols, coords, charge, spin, basis)
    mf, energy, dm0 = _run_scf(mol, xc, spin)

    forces = -to_numpy(mf.nuc_grad_method().kernel())
    dipole = _dipole(mf, mol)
    quadrupole = _quadrupole(mf, mol)

    # Polarizability: central difference of the dipole under +/- field.
    polarizability = np.zeros((3, 3))
    for axis in range(3):
        f = np.zeros(3)
        f[axis] = field_step
        mu_p = _dipole_with_field(mol, xc, spin, +f, dm0=dm0)
        mu_m = _dipole_with_field(mol, xc, spin, -f, dm0=dm0)
        polarizability[:, axis] = (mu_p - mu_m) / (2.0 * field_step)
    polarizability = 0.5 * (polarizability + polarizability.T)

    # Dipole derivatives (APT): central difference of the dipole under +/-
    # displacement of each nuclear Cartesian. Index [atom, dR, dmu].
    natm = len(symbols)
    base = mol.atom_coords()  # Bohr
    dipole_derivatives = np.zeros((natm, 3, 3))
    for a in range(natm):
        for c in range(3):
            mus = []
            for sign in (+1.0, -1.0):
                disp = base.copy()
                disp[a, c] += sign * disp_step
                m2 = mol.set_geom_(disp, unit="Bohr", inplace=False)
                mf2, _, _ = _run_scf(m2, xc, spin, dm0=dm0)
                mus.append(_dipole(mf2, m2))
            dipole_derivatives[a, c, :] = (mus[0] - mus[1]) / (2.0 * disp_step)

    return {
        "energy": energy,
        "forces": forces,
        "dipole": dipole,
        "quadrupole": quadrupole,
        "polarizability": polarizability,
        "dipole_derivatives": dipole_derivatives,
    }
