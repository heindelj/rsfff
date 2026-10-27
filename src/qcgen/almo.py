"""ALMO-EDA in pyscf: fragments -> frozen -> SCF-MI (polarization) -> full SCF (charge transfer), with a uniform
electric field and a polarizable continuum (C-PCM), every state variational in its own reaction field.

The decomposition (closed-shell fragments, RKS)

    E_int = E_cav + E_cls_elec + E_disp + E_mod_pauli + E_pol + E_ct
            \\___________ E_frz ____________/

is a chain of constrained minima of the same functional  E[D] = E_KS[D] + E_field[D] + G_PCM[D]:

* fragments: each fragment alone, in the field and in its *own* cavity (a solvated monomer);
* frozen: the antisymmetrized product of the fragment orbitals (one density, no variables), the surface charges
  re-equilibrated to it in the complex cavity;
* SCF-MI: the energy minimized over absolutely localized MOs (each fragment's occupied orbitals expanded in its own
  AOs only; Stoll/Gianinetti projected equations with the reaction field in the Fock matrix);
* full: the unconstrained SCF in the field and the continuum.

so a force field that minimizes E_FF + G_PCM jointly under the matching constraints (permanent moments only; +
induced moments; + charge transfer) reproduces the same terms. The frozen interaction is split the way the
quasiclassical ALMO-EDA does, extended to the continuum:

* E_cav      = sum_A ( G_cplx[rho_A] - G_own[rho_A] ): each fragment's (solvated) density moved from its own cavity
               into the complex cavity -- the desolvation by the cavity change (0 in vacuum);
* E_cls_elec = Coulomb interaction of the frozen fragment densities + nuclei, plus the solvent-screened cross term
               G_cplx[sum_A rho_A] - sum_A G_cplx[rho_A] (= q_A . V_B for C-PCM): classical electrostatics in the
               continuum, which a force field with fixed moments and surface charges reproduces exactly up to the
               moment representation;
* E_disp     = [E_xc(D_frz) - sum_A E_xc(P~_A)] - [E_x^HF(D_frz) - sum_A E_x^HF(P~_A)]: the exchange-correlation
               interaction of the orthogonally decomposed fragment densities P~_A beyond Hartree-Fock exchange --
               Q-Chem's orthogonal frozen decomposition (EDA2 option sets 1-5) with its default dispersion-free
               functional (DISP_FREE_X = HF); the P~_A minimize the fragments' distortion energies
               (:func:`orthogonal_decomposition`). ``frozen_decomp="classical"`` uses the undistorted fragment
               densities (EDA_CLS_DISP), ``disp_free="nlc"`` the functional without its VV10;
* E_mod_pauli = the rest of E_frz: kinetic + exchange + the local xc change on antisymmetrization.

Field: a uniform field F (au) enters as h1 = +F.r (phi = -F.r: an electron at potential phi has energy -phi) plus the
nuclear term; for charged systems the absolute energies are origin-dependent (origin = the input frame's 0) but every
EDA term is not (the same nuclei, in the same frame, in every state). Any object with ``h1(mol)`` and
``nuclear(mol)`` (e.g. :class:`rsfff.qcgen.pol_ref.Environment`) can be passed instead of a uniform field.

Continuum: pyscf's ``solvent.pcm`` (C-PCM by default: f = (eps-1)/eps, SWIG switching, Lebedev 302 points per
sphere, Bondi radii x 1.2 with H = 1.1 A -- Q-Chem's defaults). The surface potentials are the Gaussian-smeared ones
pyscf uses (point i carries a Gaussian of exponent zeta_i^2), so the force-field continuum (:mod:`rsfff.ff.cpcm`)
evaluates its potentials the same way. Every state's surface potential and charges are returned for the
QM/FF consistency tests.

Grids: by default every fragment is evaluated on the supermolecule's DFT grids (``shared_grid=True``), so grid
errors cancel in the differences; the fragment SCFs still use their own cavity and basis.

Units: Hartree inside, kJ/mol in ``terms``; coordinates in Angstrom on input, bohr inside (``BOHR``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field as dfield

import numpy as np
import scipy.linalg as sla
from pyscf import df, gto, lib
from pyscf.data import radii as _radii
from pyscf.data.elements import charge as _z

from .backend import make_mf, make_mol, to_numpy

BOHR = 0.52917721092
KJ = 2625.4996394799  # Hartree -> kJ/mol
__all__ = ["Fragment", "UniformField", "Continuum", "almo_eda", "scf_mi", "frozen_density", "surface_potential",
           "pcm_energy", "BOHR", "KJ"]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclass
class Fragment:
    symbols: list
    coords: np.ndarray          # (n, 3) Angstrom, in the common frame
    charge: int = 0
    spin: int = 0               # pyscf convention (2S); only 0 is supported today

    @property
    def natm(self):
        return len(self.symbols)


@dataclass
class UniformField:
    """Uniform electric field F (au): phi(r) = -F.r, h1 = +F.r, nuclear term sum_a Z_a (-F.R_a)."""
    field: np.ndarray

    def h1(self, mol):
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            return np.einsum("x,xij->ij", np.asarray(self.field, float), mol.intor("int1e_r", comp=3))

    def nuclear(self, mol):
        return float(-(mol.atom_charges() @ (mol.atom_coords() @ np.asarray(self.field, float))))

    def potential(self, points):
        return -np.asarray(points, float).reshape(-1, 3) @ np.asarray(self.field, float)


@dataclass
class Continuum:
    """A pyscf ``solvent.pcm`` continuum. ``radii``: {symbol: radius in Angstrom} overriding the scaled Bondi table
    (the final sphere radius, no further scaling); ``vdw_scale`` scales the table radii (pyscf/Q-Chem: 1.2)."""
    eps: float = 78.3553
    method: str = "C-PCM"           # C-PCM | COSMO | IEF-PCM | SS(V)PE
    lebedev_order: int = 29         # 302 points per sphere
    vdw_scale: float = 1.2
    radii: dict = dfield(default_factory=dict)
    switching: str = "SWIG"         # SWIG | ISWIG

    def radii_table(self):
        """Per-Z sphere radii in bohr as pyscf builds them (modified Bondi, H = 1.1 A, times vdw_scale)."""
        from pyscf.solvent.pcm import modified_Bondi
        tab = self.vdw_scale * modified_Bondi
        for sym, r_ang in self.radii.items():
            tab[_z(sym)] = float(r_ang) / BOHR
        return tab

    def build(self, mol):
        from pyscf.solvent import pcm
        obj = pcm.PCM(mol)
        obj.method = self.method
        obj.eps = float(self.eps)
        obj.lebedev_order = self.lebedev_order
        obj.vdw_scale = self.vdw_scale
        obj.surface_discretization_method = self.switching
        obj.radii_table = self.radii_table()
        obj.verbose = 0
        obj.build()
        return obj

    def spec(self):
        return {"eps": self.eps, "method": self.method, "lebedev_order": self.lebedev_order,
                "vdw_scale": self.vdw_scale, "radii": dict(self.radii), "switching": self.switching}


# ---------------------------------------------------------------------------
# continuum helpers (the same quantities the force field evaluates)
# ---------------------------------------------------------------------------
def surface_potential(pcmobj, dm=None, atoms=None):
    """Potential V_i at the surface points of ``pcmobj`` (Gaussian-smeared, pyscf's convention) of the nuclei in
    ``atoms`` (indices into pcmobj.mol; None = all) and of the electron density ``dm`` (AO matrix of pcmobj.mol;
    None = nuclei only)."""
    mol = pcmobj.mol
    surf = pcmobj.surface
    if atoms is None:
        v = np.array(pcmobj.v_grids_n, float)
    else:
        atoms = list(atoms)
        v = np.zeros(len(surf["grid_coords"]))
        if atoms:
            fakemol = gto.fakemol_for_charges(surf["grid_coords"], expnt=surf["charge_exp"] ** 2)
            fakemol_nuc = gto.fakemol_for_charges(mol.atom_coords()[atoms])
            v_ng = gto.mole.intor_cross(mol._add_suffix("int2c2e"), fakemol_nuc, fakemol)
            v = mol.atom_charges()[atoms] @ v_ng
    if dm is not None:
        v = v - pcmobj._get_v(np.asarray(dm).reshape(1, mol.nao, mol.nao))[0]
    return v


def pcm_charges(pcmobj, v):
    """Apparent surface charges q = K^-1 R v of the surface potential v."""
    K, R = pcmobj._intermediates["K"], pcmobj._intermediates["R"]
    return np.linalg.solve(K, R @ np.asarray(v, float))


def pcm_energy(pcmobj, v):
    """G = 1/2 q . v, the continuum's free energy for a solute whose surface potential is v (the equilibrated
    charges q = K^-1 R v; for C-PCM, -f/2 v^T S^-1 v)."""
    return 0.5 * float(pcm_charges(pcmobj, v) @ np.asarray(v, float))


# ---------------------------------------------------------------------------
# orbital algebra
# ---------------------------------------------------------------------------
def orth(C, S):
    """Lowdin S-orthonormalization of the columns of C (same span)."""
    if C.shape[1] == 0:
        return C
    w, U = np.linalg.eigh(C.T @ S @ C)
    return C @ (U * w ** -0.5) @ U.T


def proj_space(cols, Cocc_other, S, tol=1e-8):
    """Orthonormal basis X of span(cols) projected out of span(Cocc_other) (S-orthonormal); (X, M) with
    cols @ M the own-AO representatives of X."""
    Y = cols - Cocc_other @ (Cocc_other.T @ S @ cols) if Cocc_other.shape[1] else cols
    w, U = np.linalg.eigh(Y.T @ S @ Y)
    keep = w > tol * w.max()
    M = U[:, keep] / np.sqrt(w[keep])
    return Y @ M, M


def density(Cs, S):
    """Closed-shell density 2 C (C^T S C)^-1 C^T of the determinant spanned by the (non-orthogonal) columns of Cs."""
    C = np.hstack(Cs)
    return 2 * C @ np.linalg.solve(C.T @ S @ C, C.T)


def frozen_density(C_blocks, S):
    return density(C_blocks, S)


# ---------------------------------------------------------------------------
# the supersystem functional
# ---------------------------------------------------------------------------
class _System:
    """E[D] = E_KS[D] + E_field + G_PCM[D] on one molecule, and its Fock matrix."""

    def __init__(self, mol, xc, env=None, pcm: Continuum | None = None, grids=None, nlcgrids=None, grid_level=None):
        self.mol = mol
        self.mf = make_mf(mol, xc)
        self.mf.small_rho_cutoff = 0.0          # never prune a (shared) grid by one state's density
        if grid_level is not None:
            self.mf.grids.level, self.mf.nlcgrids.level = grid_level
        if grids is not None:
            self.mf.grids = grids
        if nlcgrids is not None:
            self.mf.nlcgrids = nlcgrids
        self.S = mol.intor("int1e_ovlp")
        self.h = to_numpy(self.mf.get_hcore())
        self.enuc = float(mol.energy_nuc())
        self.env = env
        if env is not None:
            self.h = self.h + env.h1(mol)
            self.enuc += env.nuclear(mol)
        self.pcm = pcm.build(mol) if pcm is not None else None

    def energy_fock(self, dm):
        vhf = self.mf.get_veff(self.mol, dm)
        e = float(self.mf.energy_elec(dm, self.h, vhf)[0]) + self.enuc
        F = self.h + to_numpy(vhf)
        epcm = 0.0
        if self.pcm is not None:
            epcm, vp = self.pcm._get_vind(np.asarray(dm))
            e += float(epcm)
            F = F + vp
        return e, F, float(epcm)

    def energy(self, dm):
        return self.energy_fock(dm)[0]

    def nlc_energy(self, dm):
        """The nonlocal-correlation (VV10) energy of the density; 0 for a functional without one."""
        mf = self.mf
        if not mf._numint.libxc.is_nlc(mf.xc) or mf.nlc is False:
            return 0.0
        return float(mf._numint.nr_nlc_vxc(self.mol, mf.nlcgrids, mf.xc, np.asarray(dm))[1])

    def xc_energy(self, dm):
        """The full exchange-correlation energy of the density: semilocal xc + the functional's exact-exchange part
        + VV10 (pyscf's ``vhf.exc``)."""
        return float(self.mf.get_veff(self.mol, np.asarray(dm)).exc)

    def hf_exchange_energy(self, dm):
        """-1/4 tr(D K[D]): the Hartree-Fock exchange energy of the (closed-shell) density -- Q-Chem's default
        dispersion-free functional (DISP_FREE_X = HF, DISP_FREE_C = NONE)."""
        K = to_numpy(self.mf.get_k(self.mol, np.asarray(dm)))
        return -0.25 * float(np.einsum("ij,ji->", K, dm))

    def scf(self, dm0=None, conv_tol=1e-10, max_cycle=100):
        """The unconstrained SCF of this functional (pyscf's own PCM-in-SCF loop when a continuum is set)."""
        mf = make_mf(self.mol, self.mf.xc)
        mf.small_rho_cutoff = 0.0
        mf.nlc = self.mf.nlc
        mf.grids, mf.nlcgrids = self.mf.grids, self.mf.nlcgrids
        if self.env is not None:
            h, enuc = self.h, self.enuc
            mf.get_hcore = lambda *a, **k: h
            mf.energy_nuc = lambda *a, **k: enuc
        if self.pcm is not None:
            mf = mf.PCM(self.pcm)
        mf.conv_tol, mf.max_cycle = conv_tol, max_cycle
        mf.kernel(dm0=dm0)
        if not mf.converged:
            mf = mf.newton()
            mf.kernel(mf.make_rdm1())
            if not mf.converged:
                raise RuntimeError("SCF did not converge")
        return mf


# ---------------------------------------------------------------------------
# SCF-MI
# ---------------------------------------------------------------------------
def scf_mi(system: _System, C0, slices, relax=None, max_cycle=100, tol=1e-9, gtol=2e-5, diis_space=12,
           verbose=False, log=None):
    """Absolutely localized SCF: minimize system.energy over determinants whose fragment-k occupied orbitals are
    expanded in fragment k's AOs (``slices[k]``), starting from the frozen orbitals ``C0`` (list of (nao, no_k)
    blocks). ``relax``: the fragments allowed to relax (default all). Returns (C, E, info)."""
    S, nao = system.S, system.mol.nao
    n = len(C0)
    relax = list(range(n)) if relax is None else list(relax)
    eye = np.eye(nao)
    Cs = [c.copy() for c in C0]
    no = [c.shape[1] for c in Cs]
    e, F, _ = system.energy_fock(density(Cs, S))
    e_frz = e

    def spaces(Cs):
        out = {}
        for k in relax:
            Coth = orth(np.hstack([Cs[j] for j in range(n) if j != k]) if n > 1 else np.zeros((nao, 0)), S)
            out[k] = proj_space(eye[:, slices[k]], Coth, S)
        return out

    def grad(F, Cs, Xs):
        g = []
        for k in relax:
            X = Xs[k][0]
            O, _ = np.linalg.qr(X.T @ S @ Cs[k])
            V = sla.null_space(O.T)
            g.append((V.T @ (X.T @ F @ X) @ O).ravel())
        return np.concatenate(g) if g else np.zeros(0)

    diis = lib.diis.DIIS()
    diis.space = diis_space
    e_prev, gmax, it = e, np.inf, 0
    for it in range(max_cycle):
        Xs = spaces(Cs)
        g = grad(F, Cs, Xs)
        gmax = float(np.abs(g).max()) if g.size else 0.0
        if verbose:
            print(f"  scf-mi it {it:3d} E {e:.10f} dE {e - e_prev:+.2e} |g| {gmax:.2e}", flush=True)
        if log is not None:
            log.append((it, e, gmax))
        if it > 0 and abs(e - e_prev) < tol and gmax < gtol:
            break
        Fd = diis.update(F, g) if it > 0 else F
        newC = list(Cs)
        for k in relax:
            X, M = Xs[k]
            w, U = np.linalg.eigh(X.T @ Fd @ X)
            newC[k] = eye[:, slices[k]] @ (M @ U[:, :no[k]])     # own-AO coefficients: the ALMO form
        Cs = newC
        e_prev = e
        e, F, _ = system.energy_fock(density(Cs, S))
    else:
        raise RuntimeError(f"SCF-MI did not converge in {max_cycle} iterations (|g| {gmax:.1e})")
    return Cs, e, {"iters": it + 1, "gmax": gmax, "e_frz": e_frz}


# ---------------------------------------------------------------------------
# the orthogonal frozen decomposition (Q-Chem's EDA2 option sets 1-5)
# ---------------------------------------------------------------------------
def _fragment_hcore(system: _System, atom_slices):
    """h_A = T + V_nuc(A) (+ the field) in the supermolecule basis, per fragment."""
    mol = system.mol
    T = mol.intor("int1e_kin")
    out = []
    for a0, a1 in atom_slices:
        v = np.zeros_like(T)
        for a in range(a0, a1):
            with mol.with_rinv_origin(mol.atom_coord(a)):
                v -= mol.atom_charge(a) * mol.intor("int1e_rinv")
        h = T + v
        if system.env is not None:
            h = h + system.env.h1(mol)
        out.append(h)
    return out


def orthogonal_decomposition(system: _System, C_blocks, atom_slices, D_blocks, tol=1e-6, max_iter=200, verbose=False):
    """Split the frozen density P_frz = sum_A P~_A into mutually orthogonal fragment densities that minimize
    Omega = sum_A (E_A[P~_A] - E_A[P_A]) (Q-Chem's FRZ_ORTHO_DECOMP; E_A the fragment's own functional in the
    supermolecule basis), starting from the Lowdin-orthogonalized frozen orbitals. The variables are the rotations
    between occupied orbitals of different fragments, U = exp(K); the exact gradient of Omega(K) comes from the
    Frechet derivative of the matrix exponential, and L-BFGS does the minimization. Returns (P~_A, Omega, info)."""
    from scipy.optimize import minimize
    mol, mf, S = system.mol, system.mf, system.S
    C = np.hstack(C_blocks)
    w, U = np.linalg.eigh(C.T @ S @ C)
    Cu0 = C @ (U * w ** -0.5) @ U.T                          # Lowdin: orthonormal, blocks closest to the originals
    no = [c.shape[1] for c in C_blocks]
    off = np.cumsum([0] + no)
    nocc = off[-1]
    frag_of = np.concatenate([np.full(n, k) for k, n in enumerate(no)])
    iu, ju = np.triu_indices(nocc, 1)
    sel = frag_of[iu] != frag_of[ju]
    iu, ju = iu[sel], ju[sel]
    hA = _fragment_hcore(system, atom_slices)

    def frag_energy_fock(k, P):
        vhf = mf.get_veff(mol, P)
        e = float(np.einsum("ij,ji->", hA[k], P)) + float(vhf.ecoul) + float(vhf.exc)
        return e, hA[k] + to_numpy(vhf)

    E0 = [frag_energy_fock(k, D_blocks[k])[0] for k in range(len(no))]
    state = {"n": 0}

    def unpack(x):
        K = np.zeros((nocc, nocc))
        K[iu, ju] = x
        return K - K.T

    def fun(x):
        K = unpack(x)
        Uk = sla.expm(K)
        Cu = Cu0 @ Uk
        om, GU = 0.0, np.zeros((nocc, nocc))
        Ps = []
        for k in range(len(no)):
            c = Cu[:, off[k]:off[k + 1]]
            P = 2 * c @ c.T
            ek, Fk = frag_energy_fock(k, P)
            om += ek - E0[k]
            GU[:, off[k]:off[k + 1]] = 4 * (Cu0.T @ Fk @ Cu0 @ Uk)[:, off[k]:off[k + 1]]
            Ps.append(P)
        GK = sla.expm_frechet(K.T, GU, compute_expm=False)   # d/dK <GU, expm(K)>
        g = GK[iu, ju] - GK[ju, iu]
        state["n"] += 1
        state["last"] = (om, Ps)
        if verbose:
            print(f"  ortho-decomp eval {state['n']} Omega {om * KJ:.6f} kJ/mol |g| {np.abs(g).max():.2e}", flush=True)
        return om, g

    res = minimize(fun, np.zeros(len(iu)), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-14})
    om, _ = fun(res.x)
    om, Ps = state["last"]
    gmax = float(np.abs(res.jac).max())
    if gmax > 10 * tol:
        raise RuntimeError(f"orthogonal decomposition did not converge (|g| {gmax:.1e}, {res.message})")
    return Ps, om, {"evals": state["n"], "gmax": gmax}


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------
def _cross_coulomb(system: _System, D_blocks, atom_slices):
    """Classical Coulomb interaction between the fragment charge distributions (nuclei + frozen densities), in vacuum."""
    mol, mf = system.mol, system.mf
    n = len(D_blocks)
    Vn = []
    for k in range(n):
        v = np.zeros((mol.nao, mol.nao))
        for a in range(*atom_slices[k]):
            with mol.with_rinv_origin(mol.atom_coord(a)):
                v += mol.atom_charge(a) * mol.intor("int1e_rinv")
        Vn.append(v)
    Z, R = mol.atom_charges(), mol.atom_coords()
    e = 0.0
    for A in range(n):
        JA = to_numpy(mf.get_j(mol, D_blocks[A]))
        for B in range(A + 1, n):
            e += float(np.einsum("ij,ji->", JA, D_blocks[B]))
            e -= float(np.einsum("ij,ji->", Vn[B], D_blocks[A])) + float(np.einsum("ij,ji->", Vn[A], D_blocks[B]))
            for a in range(*atom_slices[A]):
                for b in range(*atom_slices[B]):
                    e += Z[a] * Z[b] / np.linalg.norm(R[a] - R[b])
    return e


def _disp_part(system: _System, dm, disp_free):
    """The density's share of E_disp: E_xc - E_x^HF ("hf") or E_VV10 ("nlc")."""
    if disp_free == "hf":
        return system.xc_energy(dm) - system.hf_exchange_energy(dm)
    if disp_free == "nlc":
        return system.nlc_energy(dm)
    raise ValueError(f"disp_free {disp_free!r} (hf | nlc)")


def almo_eda(fragments, xc="wb97m-v", basis="def2-tzvpd", field=None, env=None, pcm: Continuum | None = None,
             shared_grid=True, conv_tol=1e-10, scfmi_tol=1e-9, scfmi_gtol=2e-5, verbose=0, with_ct=True,
             return_states=True, frag_dm0=None, disp_free="hf", grid_level=None, with_pol=True,
             frozen_decomp="orthogonal", ortho_tol=1e-5):
    """ALMO-EDA of ``fragments`` (list of :class:`Fragment`, closed shell). ``field``: uniform field (3,) au, or
    ``env``: any object with ``h1(mol)`` / ``nuclear(mol)``; ``pcm``: a :class:`Continuum` (None: vacuum);
    ``frag_dm0``: initial fragment densities (e.g. the converged ones of a neighbouring geometry in a scan).
    ``frozen_decomp`` / ``disp_free``: the dispersion term. "orthogonal" (Q-Chem's EDA2 option sets 1-5,
    FRZ_ORTHO_DECOMP): the frozen density is split into orthogonal fragment densities P~_A minimizing the fragments'
    distortion sum_A (E_A[P~_A] - E_A[P_A]) (:func:`orthogonal_decomposition`), and E_disp = [E_xc(D_frz) - sum_A
    E_xc(P~_A)] - [E_xc^DF(D_frz) - sum_A E_xc^DF(P~_A)]; "classical" (EDA_CLS_DISP) uses the undistorted fragment
    densities P_A instead. ``disp_free`` = "hf" (DISP_FREE_X = HF, DISP_FREE_C = NONE, Q-Chem's default: E_xc^DF =
    Hartree-Fock exchange) or "nlc" (the functional without its VV10). Reproduces Q-Chem's E_disp (water dimer
    wB97X-V -8.64 vs -8.67 kJ/mol across basis sets; the classical variant is ~1.5 kJ/mol short).
    ``grid_level``: (xc, nlc) pyscf grid levels (default pyscf's 3 and 1).

    Returns a dict: ``terms`` (kJ/mol: cav, cls_elec, disp, mod_pauli, frz, pol, ct, int), ``energies`` (Hartree:
    fragments, frz, scfmi, full), ``solvent`` (per state: the surface potential V, charges q and G = 1/2 q.V on the
    complex cavity; the fragments' own-cavity G; the surface geometry), ``states`` (densities and orbitals), ``info``."""
    if env is not None and field is not None:
        raise ValueError("pass either field or env")
    if field is not None:
        env = UniformField(np.asarray(field, float))
    if any(f.spin for f in fragments):
        raise NotImplementedError("open-shell fragments (UKS SCF-MI) are not implemented")
    t0 = time.time()
    # the supermolecule: fragments in order, so every fragment's AOs and atoms are contiguous
    symbols = [s for f in fragments for s in f.symbols]
    coords = np.vstack([np.asarray(f.coords, float).reshape(-1, 3) for f in fragments])
    charge = sum(f.charge for f in fragments)
    mol = make_mol(symbols, coords, charge=charge, spin=0, basis=basis)
    ao_sl = mol.aoslice_by_atom()
    atom_slices, ao_slices = [], []
    a0 = 0
    for f in fragments:
        a1 = a0 + f.natm
        atom_slices.append((a0, a1))
        ao_slices.append(slice(int(ao_sl[a0, 2]), int(ao_sl[a1 - 1, 3])))
        a0 = a1
    sup = _System(mol, xc, env=env, pcm=pcm, grid_level=grid_level)
    grids, nlcgrids = (sup.mf.grids, sup.mf.nlcgrids) if shared_grid else (None, None)
    if shared_grid:
        grids.build()
        if sup.mf._numint.libxc.is_nlc(xc):
            nlcgrids.build()

    # --- fragments, each in the field and in its own cavity
    frag_sys, frag_mf, C0, D_blocks, e_frag, e_frag_pcm, e_nlc_frag = [], [], [], [], [], [], []
    for k, f in enumerate(fragments):
        mol_k = make_mol(f.symbols, f.coords, charge=f.charge, spin=0, basis=basis)
        sys_k = _System(mol_k, xc, env=env, pcm=pcm, grids=grids, nlcgrids=nlcgrids)
        mf_k = sys_k.scf(dm0=None if frag_dm0 is None else frag_dm0[k], conv_tol=conv_tol)
        C = to_numpy(mf_k.mo_coeff)[:, to_numpy(mf_k.mo_occ) > 0]
        Cb = np.zeros((mol.nao, C.shape[1]))
        Cb[ao_slices[k]] = C
        Dk = np.zeros((mol.nao, mol.nao))
        Dk[ao_slices[k], ao_slices[k]] = to_numpy(mf_k.make_rdm1())
        frag_sys.append(sys_k)
        frag_mf.append(mf_k)
        C0.append(Cb)
        D_blocks.append(Dk)
        e_frag.append(float(mf_k.e_tot))
        e_frag_pcm.append(float(mf_k.with_solvent.e) if pcm is not None else 0.0)
        e_nlc_frag.append(_disp_part(sup, Dk, disp_free) if frozen_decomp == "classical" else 0.0)
        if verbose:
            print(f"fragment {k}: E {e_frag[-1]:.10f}  ({time.time() - t0:.0f}s)", flush=True)

    # --- frozen
    D_frz = density(C0, sup.S)
    e_frz, F_frz, g_frz = sup.energy_fock(D_frz)
    e_cls_vac = _cross_coulomb(sup, D_blocks, atom_slices)
    e_cav, e_scr, solvent = 0.0, 0.0, None
    if pcm is not None:
        p = sup.pcm
        V_A = [surface_potential(p, D_blocks[k], atoms=range(*atom_slices[k])) for k in range(len(fragments))]
        G_A_cplx = [pcm_energy(p, v) for v in V_A]
        V_sum = np.sum(V_A, 0)
        G_sum = pcm_energy(p, V_sum)
        e_cav = sum(G_A_cplx) - sum(e_frag_pcm)
        e_scr = G_sum - sum(G_A_cplx)
        V_frz = surface_potential(p, D_frz)
        solvent = {"grid_coords": p.surface["grid_coords"].copy(), "charge_exp": p.surface["charge_exp"].copy(),
                   "switch_fun": p.surface["switch_fun"].copy(), "weights": p.surface["weights"].copy(),
                   "R_vdw": p.surface["R_vdw"].copy(), "gslice_by_atom": list(p.surface["gslice_by_atom"]),
                   "f_epsilon": p._intermediates["f_epsilon"],
                   "V_frag_cplx": V_A, "G_frag_cplx": G_A_cplx, "G_frag_own": list(e_frag_pcm),
                   "V_sup": V_sum, "G_sup": G_sum,
                   "V_frz": V_frz, "q_frz": pcm_charges(p, V_frz), "G_frz": g_frz}
    ortho = None
    if frozen_decomp == "orthogonal":
        P_orth, omega, ortho = orthogonal_decomposition(sup, C0, atom_slices, D_blocks, tol=ortho_tol, verbose=verbose > 1)
        ortho["omega"] = omega
        e_disp = _disp_part(sup, D_frz, disp_free) - sum(_disp_part(sup, P, disp_free) for P in P_orth)
    elif frozen_decomp == "classical":
        e_disp = _disp_part(sup, D_frz, disp_free) - sum(e_nlc_frag)
    else:
        raise ValueError(f"frozen_decomp {frozen_decomp!r} (orthogonal | classical)")
    e_frz_int = e_frz - sum(e_frag)
    e_pauli = e_frz_int - e_cav - e_cls_vac - e_scr - e_disp
    if verbose:
        print(f"frozen: E {e_frz:.10f}  E_frz {e_frz_int * KJ:.3f} kJ/mol  ({time.time() - t0:.0f}s)", flush=True)

    # --- SCF-MI
    log = []
    if not with_pol:
        out = {"terms": {k: v * KJ for k, v in {"cav": e_cav, "cls_elec": e_cls_vac + e_scr, "cls_elec_vac": e_cls_vac,
                                                 "cls_elec_scr": e_scr, "disp": e_disp, "mod_pauli": e_pauli,
                                                 "frz": e_frz_int}.items()},
               "energies": {"fragments": e_frag, "frz": e_frz}, "solvent": solvent,
               "info": {"xc": xc, "basis": basis, "pcm": None if pcm is None else pcm.spec(), "time": time.time() - t0,
                        "ortho": ortho}}
        if return_states:
            out["states"] = {"mol": mol, "system": sup, "fragment_mf": frag_mf, "C_frz": C0, "D_frag": D_blocks,
                             "D_frz": D_frz, "ao_slices": ao_slices, "atom_slices": atom_slices,
                             "D_frag_own": [to_numpy(m.make_rdm1()) for m in frag_mf]}
        return out
    C_pol, e_pol_tot, info_mi = scf_mi(sup, C0, ao_slices, tol=scfmi_tol, gtol=scfmi_gtol, verbose=verbose > 1, log=log)
    D_pol = density(C_pol, sup.S)
    e_pol = e_pol_tot - e_frz
    if verbose:
        print(f"scf-mi: E {e_pol_tot:.10f}  pol {e_pol * KJ:.3f} kJ/mol  ({info_mi['iters']} it, {time.time() - t0:.0f}s)",
              flush=True)
    if pcm is not None:
        _, _, g_pol = sup.energy_fock(D_pol)
        V_pol = surface_potential(sup.pcm, D_pol)
        solvent.update({"V_pol": V_pol, "q_pol": pcm_charges(sup.pcm, V_pol), "G_pol": g_pol})

    # --- full SCF
    e_full, D_full, mf_full, e_ct = None, None, None, None
    if with_ct:
        mf_full = sup.scf(dm0=D_pol, conv_tol=conv_tol)
        e_full = float(mf_full.e_tot)
        D_full = to_numpy(mf_full.make_rdm1())
        e_ct = e_full - e_pol_tot
        if verbose:
            print(f"full: E {e_full:.10f}  ct {e_ct * KJ:.3f} kJ/mol  ({time.time() - t0:.0f}s)", flush=True)
        if pcm is not None:
            V_full = surface_potential(sup.pcm, D_full)
            solvent.update({"V_full": V_full, "q_full": pcm_charges(sup.pcm, V_full),
                            "G_full": float(mf_full.with_solvent.e)})

    terms = {"cav": e_cav, "cls_elec": e_cls_vac + e_scr, "cls_elec_vac": e_cls_vac, "cls_elec_scr": e_scr,
             "disp": e_disp, "mod_pauli": e_pauli, "frz": e_frz_int, "pol": e_pol}
    if e_ct is not None:
        terms["ct"] = e_ct
    terms["int"] = e_frz_int + e_pol + (e_ct or 0.0)
    out = {
        "terms": {k: v * KJ for k, v in terms.items()},
        "energies": {"fragments": e_frag, "frz": e_frz, "scfmi": e_pol_tot, "full": e_full},
        "solvent": solvent,
        "info": {"xc": xc, "basis": basis, "field": None if field is None else list(map(float, field)),
                 "pcm": None if pcm is None else pcm.spec(), "scfmi_iters": info_mi["iters"],
                 "scfmi_gmax": info_mi["gmax"], "scfmi_log": log, "time": time.time() - t0, "ortho": ortho,
                 "fragments": [{"symbols": list(f.symbols), "coords": np.asarray(f.coords, float).tolist(),
                                "charge": f.charge} for f in fragments]},
    }
    if return_states:
        out["states"] = {"mol": mol, "system": sup, "fragment_mf": frag_mf, "C_frz": C0, "C_pol": C_pol,
                         "D_frag": D_blocks, "D_frz": D_frz, "D_pol": D_pol, "D_full": D_full, "mf_full": mf_full,
                         "ao_slices": ao_slices, "atom_slices": atom_slices,
                         "D_frag_own": [to_numpy(m.make_rdm1()) for m in frag_mf]}
    return out
