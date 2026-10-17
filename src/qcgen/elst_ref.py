"""Reference electrostatics from density-fitted monomer densities (pyscf).

Build a database of monomer charge densities once, then place monomers rigidly and get exact
frozen-density electrostatics -- the SAPT/EDA first-order electrostatic energy -- together with
potentials and fields at arbitrary points and rigid-body forces/torques, from two-centre integrals
only. No dimer calculation, no new SCF per placement: milliseconds per pair. The intended use is
fitting the electrostatic part of a force-field functional (permanent multipoles, Slater exponents,
shaped damping densities) deep into the charge-penetration region.

The monomer density
-------------------
Each monomer keeps its SCF AO density (total, spherical basis) and a Coulomb-metric fit of it in an
auxiliary basis (charge constrained). The fit alone is *not* accurate enough for potentials: its error
in an external potential is first order (water, def2-universal-jkfit: ~2e-4 au at 5 bohr, dipole off by
0.7 %). So nuclear attraction, potentials and fields use the exact AO density (three-index integrals
with point functions), and the electron-electron term uses the robust fitting form
(rho_A|fit_B) + (fit_A|rho_B) - (fit_A|fit_B), whose error is second order: <= 1e-6 Eh down to 1.8 A
O...O (water dimer, def2-SVPD), ~10 ms per pair. ``exact_ee=True`` uses the four-centre block (~1e-8 Eh).
Nuclear charges are pyscf's ``atom_charge`` -- with an ECP the core is removed from both nucleus and
electrons; the ECP's non-local part is not electrostatics and is not represented.

Placement
---------
A placement is a proper rotation ``rot`` (3x3) and a translation ``shift`` (Bohr) applied to the
monomer's stored frame: r' = rot @ r + shift. The AO density and the fit coefficients rotate shell by
shell with real solid-harmonic rotation matrices built numerically from pyscf's own AO ordering
(convention-free); no SCF per placement.

Conventions: atomic units throughout (Bohr, Hartree). Potentials are of the *charge* density
(nuclei positive, electrons negative); fields are E = -grad(phi); a force is -dE/dX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from pyscf import df, dft, gto
from pyscf.data import nist

BOHR = nist.BOHR  # Angstrom per Bohr


# ---------------------------------------------------------------------------
# density fitting
# ---------------------------------------------------------------------------
def _aux_integrals(auxmol: gto.Mole) -> np.ndarray:
    """int chi_Q(r) dr for every auxiliary function (numerical; only s-shells are non-zero)."""
    grids = dft.gen_grid.Grids(auxmol)
    grids.level = 5
    grids.build(with_non0tab=False)
    ao = dft.numint.eval_ao(auxmol, grids.coords)
    return ao.T @ grids.weights


def fit_density(mol: gto.Mole, dm: np.ndarray, auxbasis: str = "def2-universal-jkfit"):
    """Coulomb-metric fit of the electron density of ``dm`` (AO, total) with the charge constrained.

    Returns ``(auxmol, coef)``.
    """
    dm = np.asarray(dm)
    if dm.ndim == 3:            # unrestricted (alpha, beta)
        dm = dm[0] + dm[1]
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    j2c = auxmol.intor("int2c2e")
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")      # (nao, nao, naux)
    b = np.einsum("ijq,ji->q", j3c, dm)
    s = _aux_integrals(auxmol)
    n_e = float(np.einsum("ij,ji->", dm, mol.intor("int1e_ovlp")))
    # minimise (c J c - 2 b c) subject to s.c = n_e
    Jb = np.linalg.solve(j2c, b)
    Js = np.linalg.solve(j2c, s)
    lam = (s @ Jb - n_e) / (s @ Js)
    return auxmol, Jb - lam * Js


# ---------------------------------------------------------------------------
# rotation of real solid harmonics (pyscf AO order), built numerically
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _probe_mol(lmax: int) -> gto.Mole:
    basis = {"X": [[l, [1.0, 1.0]] for l in range(lmax + 1)]}
    return gto.M(atom="X 0 0 0", basis=basis, spin=None, charge=0, verbose=0, unit="Bohr")


def harmonic_rotation(lmax: int, rot: np.ndarray) -> list[np.ndarray]:
    """[D_l for l = 0..lmax]: coefficients of a function rotated by ``rot`` are c' = D_l @ c.

    A function f(r) = sum_m c_m g_m(r) rotated rigidly, f'(r) = f(rot^T r), equals sum_m c'_m g_m(r).
    """
    mol = _probe_mol(lmax)
    rng = np.random.default_rng(7)
    u = rng.normal(size=(64, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    g = mol.eval_gto("GTOval_sph", u)            # g_m(u)
    gr = mol.eval_gto("GTOval_sph", u @ rot)     # g_m(rot^T u)  (rows: u_i @ rot = (rot^T u_i)^T)
    out, k = [], 0
    for l in range(lmax + 1):
        n = 2 * l + 1
        A, B = g[:, k:k + n], gr[:, k:k + n]
        # f'(u) = sum_m c_m g_m(rot^T u) = sum_m' c'_m' g_m'(u)  ->  B c = A c'  ->  c' = (A^+ B) c
        D = np.linalg.lstsq(A, B, rcond=None)[0]
        out.append(D)
        k += n
    return out


# ---------------------------------------------------------------------------
# monomers
# ---------------------------------------------------------------------------
def _rotate_blocks(mol: gto.Mole, rot: np.ndarray) -> np.ndarray:
    """Block-diagonal rotation U of a spherical basis: rotated coefficients are U @ c (U @ D @ U.T)."""
    ls = mol._bas[:, gto.ANG_OF]
    Ds = harmonic_rotation(int(ls.max()), rot)
    U = np.zeros((mol.nao, mol.nao))
    offs = mol.ao_loc_nr()
    for ib in range(mol.nbas):
        l = int(ls[ib])
        n = 2 * l + 1
        for k in range(mol.bas_nctr(ib)):
            p0 = offs[ib] + k * n
            U[p0:p0 + n, p0:p0 + n] = Ds[l]
    return U


@dataclass
class Monomer:
    """A monomer's electron density in its own (stored) frame: the AO density and its Coulomb fit."""

    symbols: list
    coords: np.ndarray            # (natm, 3) Bohr
    basis: str
    ecp: str | None
    charge: int
    spin: int
    dm: np.ndarray                # (nao, nao) total AO density
    auxbasis: str
    coef: np.ndarray              # (naux,) Coulomb-metric fit of the electron density
    meta: dict = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_density(cls, mol: gto.Mole, dm, auxbasis="def2-universal-jkfit", **meta):
        dm = np.asarray(dm)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        _, coef = fit_density(mol, dm, auxbasis)
        ecp = mol.ecp if isinstance(mol.ecp, str) else (repr(mol.ecp) if mol.ecp else None)
        return cls(symbols=[mol.atom_symbol(i) for i in range(mol.natm)], coords=mol.atom_coords().copy(),
                   basis=mol.basis if isinstance(mol.basis, str) else repr(mol.basis), ecp=ecp,
                   charge=int(mol.charge), spin=int(mol.spin), dm=dm, auxbasis=auxbasis, coef=coef, meta=meta)

    @classmethod
    def from_scf(cls, symbols, coords_ang, charge=0, spin=0, xc="wb97m-v", basis="def2-tzvpd",
                 auxbasis="def2-universal-jkfit", conv_tol=1e-10):
        from .backend import make_mf, make_mol, to_numpy
        mol = make_mol(symbols, coords_ang, charge=charge, spin=spin, basis=basis)
        mf = make_mf(mol, xc, spin=spin)
        mf.conv_tol = conv_tol
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("monomer SCF did not converge")
        dm = to_numpy(mf.make_rdm1())
        return cls.from_density(mol, dm, auxbasis, xc=xc, energy=float(mf.e_tot))

    # -- io ---------------------------------------------------------------
    def save(self, path):
        np.savez(path, symbols=np.asarray(self.symbols), coords=self.coords, basis=self.basis,
                 ecp=str(self.ecp), charge=self.charge, spin=self.spin, dm=self.dm, auxbasis=self.auxbasis,
                 coef=self.coef, meta=np.asarray([repr(self.meta)]))

    @classmethod
    def load(cls, path):
        import ast
        z = np.load(path, allow_pickle=False)
        ecp = str(z["ecp"])
        return cls(symbols=[str(s) for s in z["symbols"]], coords=z["coords"], basis=str(z["basis"]),
                   ecp=None if ecp == "None" else ecp, charge=int(z["charge"]), spin=int(z["spin"]),
                   dm=z["dm"], auxbasis=str(z["auxbasis"]), coef=z["coef"],
                   meta=ast.literal_eval(str(z["meta"][0])))

    # -- molecules at given coordinates -------------------------------------
    def _atoms(self, coords):
        return [(s, tuple(x)) for s, x in zip(self.symbols, coords)]

    def mol(self, coords=None) -> gto.Mole:
        c = self.coords if coords is None else coords
        basis = self.basis if not self.basis.startswith("{") else eval(self.basis)  # noqa: S307 (own repr)
        kw = {} if self.ecp is None else {"ecp": self.ecp if not self.ecp.startswith("{") else eval(self.ecp)}  # noqa: S307
        return gto.M(atom=self._atoms(c), unit="Bohr", basis=basis, charge=self.charge, spin=self.spin,
                     symmetry=False, verbose=0, **kw)

    def auxmol(self, coords=None) -> gto.Mole:
        return df.addons.make_auxmol(self.mol(coords), self.auxbasis)

    def place(self, rot=None, shift=None) -> "Placed":
        rot = np.eye(3) if rot is None else np.asarray(rot, float)
        shift = np.zeros(3) if shift is None else np.asarray(shift, float)
        if abs(np.linalg.det(rot) - 1) > 1e-8 or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-8):
            raise ValueError("rot must be a proper rotation")
        coords = self.coords @ rot.T + shift
        mol = self.mol(coords)
        aux = df.addons.make_auxmol(mol, self.auxbasis)
        dm, coef = self.dm, self.coef
        if not np.allclose(rot, np.eye(3)):
            U = _rotate_blocks(mol, rot)
            dm = U @ dm @ U.T
            coef = _rotate_blocks(aux, rot) @ coef
        charges = np.asarray([mol.atom_charge(i) for i in range(mol.natm)], float)
        return Placed(self, coords, charges, mol, dm, aux, coef, rot, shift)


@dataclass
class Placed:
    monomer: Monomer
    coords: np.ndarray
    charges: np.ndarray           # nuclear (valence, with an ECP) charges
    mol: gto.Mole
    dm: np.ndarray
    aux: gto.Mole
    coef: np.ndarray
    rot: np.ndarray
    shift: np.ndarray

    @property
    def com(self):
        return self.coords.mean(0)


# ---------------------------------------------------------------------------
# potentials, fields, energies
# ---------------------------------------------------------------------------
def _points_mol(points):
    return gto.fakemol_for_charges(np.asarray(points, float).reshape(-1, 3))


def _electron_potential(p: Placed, points) -> np.ndarray:
    """Potential of p's electron *number* density at points (exact AO density): sum_ij D_ij (ij|delta_R)."""
    j3 = df.incore.aux_e2(p.mol, _points_mol(points), intor="int3c2e", aosym="s1")   # (nao, nao, npts)
    return np.einsum("ijp,ji->p", j3, p.dm)


def potential(placed, points) -> np.ndarray:
    """Electrostatic potential of the placed charge densities (nuclei + electrons) at ``points`` (Bohr)."""
    points = np.asarray(points, float).reshape(-1, 3)
    phi = np.zeros(len(points))
    for p in _as_list(placed):
        phi -= _electron_potential(p, points)
        r = np.linalg.norm(points[:, None, :] - p.coords[None, :, :], axis=2)
        phi += (p.charges[None, :] / r).sum(1)
    return phi


def field(placed, points) -> np.ndarray:
    """Electric field E = -grad(phi) of the placed charge densities at ``points`` (Bohr), shape (npts, 3)."""
    points = np.asarray(points, float).reshape(-1, 3)
    E = np.zeros((len(points), 3))
    for p in _as_list(placed):
        # phi_e(R) = -sum D_ij (ij|g_R); d/dR (ij|g_R) = -(ij|grad g_R) = -int3c2e_ip2
        ip2 = df.incore.aux_e2(p.mol, _points_mol(points), intor="int3c2e_ip2", aosym="s1", comp=3)
        E -= np.einsum("xijp,ji->px", ip2, p.dm)
        d = points[:, None, :] - p.coords[None, :, :]
        r = np.linalg.norm(d, axis=2)
        E += (p.charges[None, :, None] * d / r[..., None] ** 3).sum(1)
    return E


def field_gradient(placed, points) -> np.ndarray:
    """Field gradient G_ab = dE_a/dr_b = -d_a d_b phi at ``points`` (Bohr): (npts, 3, 3), symmetric, traceless
    away from the charge density (Laplace); inside the density its trace is 4 pi rho (Poisson)."""
    points = np.asarray(points, float).reshape(-1, 3)
    G = np.zeros((len(points), 3, 3))
    for p in _as_list(placed):
        # phi_e = -sum D_ij (ij|g_R);  d_a d_b (ij|g_R) = (ij| d_a d_b g) = int3c2e_ipip2  ->  G_e = +D . ipip2
        # a point exponent of 1e10 (not pyscf's default 1e16, which loses the diagonal of the second
        # derivatives to round-off); the smearing width 1e-5 bohr is far below anything resolved here
        pm = gto.fakemol_for_charges(points, expnt=1e10)
        ipip = df.incore.aux_e2(p.mol, pm, intor="int3c2e_ipip2", aosym="s1", comp=9)
        G += np.einsum("xijp,ji->px", ipip, p.dm).reshape(-1, 3, 3)
        d = points[:, None, :] - p.coords[None, :, :]
        r = np.linalg.norm(d, axis=2)
        dd = 3 * d[..., :, None] * d[..., None, :] - (r ** 2)[..., None, None] * np.eye(3)
        G -= (p.charges[None, :, None, None] * dd / r[..., None, None] ** 5).sum(1)
    return G


def pair_energy(A: Placed, B: Placed, exact_ee: bool = False) -> dict:
    """Frozen-density electrostatic interaction of two placed monomers (Hartree), by parts.

    Nuclear terms use the exact AO densities. The electron-electron term uses the robust density-fitting
    form (rho_A|fit_B) + (fit_A|rho_B) - (fit_A|fit_B), whose error is second order in the fit residuals;
    ``exact_ee`` evaluates the four-centre (AA|BB) block instead.
    """
    if exact_ee:
        m = A.mol + B.mol
        nA, nb = A.mol.nbas, m.nbas
        eri = m.intor("int2e", shls_slice=(0, nA, 0, nA, nA, nb, nA, nb), aosym="s1")
        ee = float(np.einsum("ijkl,ji,lk->", eri, A.dm, B.dm))
    else:
        rA_fB = np.einsum("ijq,ji->q", df.incore.aux_e2(A.mol, B.aux, intor="int3c2e", aosym="s1"), A.dm) @ B.coef
        fA_rB = A.coef @ np.einsum("ijq,ji->q", df.incore.aux_e2(B.mol, A.aux, intor="int3c2e", aosym="s1"), B.dm)
        fA_fB = A.coef @ gto.intor_cross("int2c2e", A.aux, B.aux) @ B.coef
        ee = float(rA_fB + fA_rB - fA_fB)
    en = -float(B.charges @ _electron_potential(A, B.coords))     # A electrons - B nuclei
    ne = -float(A.charges @ _electron_potential(B, A.coords))     # A nuclei - B electrons
    d = A.coords[:, None, :] - B.coords[None, :, :]
    nn = float((A.charges[:, None] * B.charges[None, :] / np.linalg.norm(d, axis=2)).sum())
    return {"ee": ee, "en": en, "ne": ne, "nn": nn, "total": ee + en + ne + nn}


def interaction_energy(placed, exact_ee: bool = False) -> float:
    """Sum of frozen-density pair electrostatics over all pairs of placed monomers."""
    ps = _as_list(placed)
    return sum(pair_energy(ps[i], ps[j], exact_ee)["total"] for i in range(len(ps)) for j in range(i + 1, len(ps)))


def rigid_force_torque(placed, k: int, h: float = 1e-4):
    """Force and torque (about its centroid) on placed monomer ``k`` from all the others.

    The force is analytic in principle; here both come from central differences of the energy with the
    monomer's density moved rigidly (translation h Bohr, rotation h rad), which is exact to O(h^2) and
    cheap (two-centre integrals only).
    """
    ps = _as_list(placed)
    me = ps[k]
    others = [p for i, p in enumerate(ps) if i != k]

    def e_with(rot, shift):
        # place: r' = R_new r0 + t_new, with R_new = rot @ R, t_new chosen to rotate about the centroid
        R_new = rot @ me.rot
        c0 = me.com
        t_new = rot @ (me.shift - c0) + c0 + shift
        q = me.monomer.place(R_new, t_new)
        return sum(pair_energy(q, o)["total"] for o in others)

    F, T = np.zeros(3), np.zeros(3)
    I = np.eye(3)
    for x in range(3):
        s = np.zeros(3); s[x] = h
        F[x] = -(e_with(I, s) - e_with(I, -s)) / (2 * h)
        T[x] = -(e_with(_axis_rot(x, h), np.zeros(3)) - e_with(_axis_rot(x, -h), np.zeros(3))) / (2 * h)
    return F, T


def random_rotation(rng=None) -> np.ndarray:
    """A uniformly distributed proper rotation (QR of a Gaussian matrix, sign-fixed)."""
    rng = np.random.default_rng(rng)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def place_pair(A: Monomer, B: Monomer, distance: float, rng=None, rot_a=None, rot_b=None, direction=None):
    """A at the origin (centroid), B with its centroid at ``distance`` Bohr along ``direction``; random
    orientations unless given. Returns (placed A, placed B)."""
    rng = np.random.default_rng(rng)
    ra = random_rotation(rng) if rot_a is None else rot_a
    rb = random_rotation(rng) if rot_b is None else rot_b
    u = rng.normal(size=3) if direction is None else np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    pa = A.place(ra, -ra @ A.coords.mean(0))
    pb = B.place(rb, -rb @ B.coords.mean(0) + distance * u)
    return pa, pb


def min_contact(A: Placed, B: Placed) -> float:
    """Shortest interatomic distance between two placed monomers (Bohr)."""
    return float(np.linalg.norm(A.coords[:, None] - B.coords[None], axis=2).min())


def _axis_rot(x, a):
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][x]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def _as_list(placed):
    return [placed] if isinstance(placed, Placed) else list(placed)


# ---------------------------------------------------------------------------
# exact reference (for validation): frozen AO densities in the joint basis
# ---------------------------------------------------------------------------
def exact_frozen_elst(molA: gto.Mole, dmA, molB: gto.Mole, dmB) -> float:
    """E_elst^(1) of two frozen AO densities (no fitting), molecules already in their placed frames."""
    mol = molA + molB
    nA = molA.nao
    D = [np.zeros((mol.nao, mol.nao)) for _ in range(2)]
    D[0][:nA, :nA] = dmA
    D[1][nA:, nA:] = dmB
    vj = mol.RKS().get_j(mol, D[1])
    ee = float(np.einsum("ij,ji->", D[0], vj))
    en = ne = 0.0
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            v = mol.intor("int1e_rinv")
        z = mol.atom_charge(ia)
        if ia < molA.natm:
            ne -= z * float(np.einsum("ij,ji->", D[1], v))
        else:
            en -= z * float(np.einsum("ij,ji->", D[0], v))
    cA, cB = molA.atom_coords(), molB.atom_coords()
    zA = np.array([molA.atom_charge(i) for i in range(molA.natm)])
    zB = np.array([molB.atom_charge(i) for i in range(molB.natm)])
    nn = float((zA[:, None] * zB[None, :] / np.linalg.norm(cA[:, None] - cB[None], axis=2)).sum())
    return ee + en + ne + nn
