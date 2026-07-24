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

from .backend import HAVE_GPU, as_backend_array, make_mf, make_mol, to_numpy

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
# Analytic (CPSCF) response: polarizability and dipole derivatives
# ---------------------------------------------------------------------------
_warned = set()


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"[qcgen] {msg}", flush=True)


def _polarizability_cpscf(mf, mol, spin):
    """Static dipole polarizability (a0^3) from coupled-perturbed SCF.

    Uses the backend's validated CPHF response: ``gpu4pyscf.properties`` on GPU,
    ``pyscf.prop.polarizability`` (the pyscf-properties extension) on CPU.
    Costs 3 CPHF solves instead of 6 field-perturbed SCF runs.
    """
    if HAVE_GPU:  # pragma: no cover - GPU-only path
        from gpu4pyscf.properties import polarizability as gpu_pol

        return np.asarray(to_numpy(gpu_pol.eval_polarizability(mf)))

    if int(spin) == 0:
        from pyscf.prop.polarizability.rks import Polarizability
    else:
        from pyscf.prop.polarizability.uks import Polarizability
    return np.asarray(Polarizability(mf).polarizability())


def _apt_cpscf(mf, mol, spin):
    """Dipole derivatives dmu/dR (a.u.) from nuclear coupled-perturbed SCF.

    No library exposes the atomic polar tensor directly (gpu4pyscf's IR module
    builds it internally but returns only intensities), so it is assembled here
    from the same CPSCF first-order orbitals the Hessian uses:

        dmu_x/dR_{a,y} = -4 <r_x . dm1_y>  -  <d(r_x)/dR_a . dm0>  +  Z_a delta_xy

    The three terms are the density response, the derivative of the dipole
    integrals w.r.t. the basis functions riding on atom ``a``, and the nuclear
    contribution. Contractions run on the host; only ``solve_mo1`` is on-backend.
    Costs 3N CPSCF solves instead of 6N displaced SCF runs.

    Returns ``(natm, 3, 3)`` indexed ``[atom, dR, dmu]``.
    """
    if int(spin) != 0:
        # The 4x prefactor below assumes doubly-occupied orbitals; the
        # open-shell generalization is not implemented, so use numerical APTs.
        raise NotImplementedError("CPSCF APT is implemented for closed shell only")

    hessobj = mf.Hessian()
    h1ao = hessobj.make_h1(mf.mo_coeff, mf.mo_occ)
    mo1, _ = hessobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao)

    mo_coeff = to_numpy(mf.mo_coeff)
    mo_occ = to_numpy(mf.mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = to_numpy(mf.make_rdm1())

    nao, natm = mol.nao, mol.natm
    r_ao = mol.intor("int1e_r")  # <u| r |v>, (3, nao, nao)
    # d/dR of the dipole integrals, non-zero only for AOs centred on the atom.
    drdx = -mol.intor("int1e_irp").reshape(3, 3, nao, nao)
    aoslices = mol.aoslice_by_atom()

    apt = np.zeros((3, 3, natm))  # [dmu, dR, atom]
    for ia in range(natm):
        p0, p1 = aoslices[ia][2], aoslices[ia][3]
        h11 = np.zeros((3, 3, nao, nao))
        h11[:, :, :, p0:p1] += drdx[:, :, :, p0:p1]
        h11[:, :, p0:p1, :] += drdx[:, :, :, p0:p1].transpose(0, 1, 3, 2)

        # Backend convention differs: pyscf's solve_mo1 already returns the
        # first-order orbitals in the AO basis (it applies mo_coeff internally),
        # while gpu4pyscf returns them in the MO basis. Both come back shaped
        # (3, nao, nocc) whenever nao == nmo, so this cannot be inferred from
        # the array itself -- transform only on the GPU path.
        m1 = to_numpy(mo1[ia])
        if HAVE_GPU:  # pragma: no cover - GPU-only path
            m1 = np.einsum("up,ypi->yui", mo_coeff, m1, optimize=True)
        # dm1 is the density response; the factor 4 below supplies double
        # occupancy (x2) and the complex-conjugate term (x2).
        dm1 = np.einsum("yui,vi->yuv", m1, mocc, optimize=True)

        t = -np.einsum("xuv,yuv->xy", r_ao, dm1, optimize=True) * 4.0
        t -= np.einsum("xyuv,vu->xy", h11, dm0, optimize=True)
        t += mol.atom_charge(ia) * np.eye(3)
        apt[:, :, ia] = t

    # [dmu, dR, atom] -> [atom, dR, dmu]
    return apt.transpose(2, 1, 0)


# ---------------------------------------------------------------------------
# Numerical (finite-difference) response -- fallback and cross-check
# ---------------------------------------------------------------------------
def _polarizability_fd(mol, xc, spin, dm0, field_step):
    """Polarizability by central difference of the dipole under +/- field."""
    alpha = np.zeros((3, 3))
    for axis in range(3):
        f = np.zeros(3)
        f[axis] = field_step
        mu_p = _dipole_with_field(mol, xc, spin, +f, dm0=dm0)
        mu_m = _dipole_with_field(mol, xc, spin, -f, dm0=dm0)
        alpha[:, axis] = (mu_p - mu_m) / (2.0 * field_step)
    return 0.5 * (alpha + alpha.T)


def _apt_fd(mol, xc, spin, dm0, disp_step):
    """Dipole derivatives by central difference under nuclear displacement.

    Returns ``(natm, 3, 3)`` indexed ``[atom, dR, dmu]``.
    """
    natm = mol.natm
    base = mol.atom_coords()  # Bohr
    apt = np.zeros((natm, 3, 3))
    for a in range(natm):
        for c in range(3):
            mus = []
            for sign in (+1.0, -1.0):
                disp = base.copy()
                disp[a, c] += sign * disp_step
                m2 = mol.set_geom_(disp, unit="Bohr", inplace=False)
                mf2, _, _ = _run_scf(m2, xc, spin, dm0=dm0)
                mus.append(_dipole(mf2, m2))
            apt[a, c, :] = (mus[0] - mus[1]) / (2.0 * disp_step)
    return apt


# ---------------------------------------------------------------------------
# Stage 4: full property labeling of one structure
# ---------------------------------------------------------------------------
def compute_reference_data(symbols, coords, charge, spin, xc, basis,
                           response="cpscf", with_dipole_derivatives=False,
                           field_step=1e-3, disp_step=1e-3):
    """Label one geometry with energy/forces/dipole/quadrupole/polarizability.

    ``with_dipole_derivatives`` (default ``False``) adds the atomic polar tensor
    dmu/dR. It is off by default because it dominates the cost of this routine:
    even analytically it needs a Hessian object plus 3N coupled-perturbed solves
    (and 6N SCF runs numerically), versus 3 solves for everything else. When
    ``False`` the ``dipole_derivatives`` key is absent from the result and from
    the extxyz header.

    ``response`` selects how the polarizability and dipole derivatives are
    obtained:

      * ``"cpscf"`` (default) -- analytic coupled-perturbed SCF. Costs
        ``3 + 3N`` linear solves on top of one SCF, versus ``6 + 6N`` full SCF
        runs, and carries no finite-difference truncation error. Falls back
        automatically where unsupported (e.g. UKS + non-local correlation).
      * ``"finite-difference"`` -- numerical differentiation, using
        ``field_step`` (a.u. field) and ``disp_step`` (Bohr). Always available;
        useful as an independent cross-check.

    The returned dict reports which path ran under ``response_method``. All
    values are host numpy arrays in atomic units matching the extxyz schema.
    """
    mol = make_mol(symbols, coords, charge, spin, basis)
    mf, energy, dm0 = _run_scf(mol, xc, spin)

    forces = -to_numpy(mf.nuc_grad_method().kernel())
    dipole = _dipole(mf, mol)
    quadrupole = _quadrupole(mf, mol)

    used = response
    if response == "cpscf":
        try:
            polarizability = _polarizability_cpscf(mf, mol, spin)
            if with_dipole_derivatives:
                dipole_derivatives = _apt_cpscf(mf, mol, spin)
        except (NotImplementedError, ImportError, AttributeError) as exc:
            # e.g. UKS + non-local correlation, or a backend without an analytic
            # response path. Numerical differentiation always works.
            used = "finite-difference"
            _warn_once(f"CPSCF response unavailable ({type(exc).__name__}: {exc});"
                       " falling back to finite difference")
    if used != "cpscf":
        polarizability = _polarizability_fd(mol, xc, spin, dm0, field_step)
        if with_dipole_derivatives:
            dipole_derivatives = _apt_fd(mol, xc, spin, dm0, disp_step)

    data = {
        "energy": energy,
        "forces": forces,
        "dipole": dipole,
        "quadrupole": quadrupole,
        "polarizability": polarizability,
        "response_method": used,
    }
    if with_dipole_derivatives:
        data["dipole_derivatives"] = dipole_derivatives
    return data
