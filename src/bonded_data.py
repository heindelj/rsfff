"""Generate the training data of the bonded model with pyscf: geometries and labels, one script, no other code.

    rsfff-bonded-data all --species h2o h3o+ oh- --out data/bonded --workers 8
    rsfff-bonded-data reference --species h2o --out data/bonded
    rsfff-bonded-data sample data/bonded/h2o_opt.xyz --out data/bonded/h2o_geoms.xyz --wigner 400 --scans
    rsfff-bonded-data label data/bonded/h2o_geoms.xyz --out data/bonded/h2o.xyz --raman --esp --workers 8

(``rsfff-bonded-data`` is this module, ``python -m rsfff.bonded_data``; it imports nothing else from rsfff.)
Needs only numpy, scipy and pyscf. Default level: wB97M-V/def2-TZVPD (``--xc``, ``--basis``).

``--backend gpu4pyscf`` runs every SCF, gradient and Hessian on a GPU with gpu4pyscf (same protocol, same labels;
``--gpus N`` spreads the ``--workers`` over N devices). ``--analytic-hessian`` (either backend) takes the Hessian
from the program's analytic code instead of differences of gradients, and for a closed shell also the dipole
derivatives, from the same coupled-perturbed solution (gpu4pyscf's own IR code does this; its Raman code takes
the polarizability derivatives from finite fields, as here). When both exist the finite-field dipole derivatives
are kept as a check: the largest difference is written as ``dipole_derivatives_check`` and reported.

Stages
------
``reference``  optimize (BFGS on analytic gradients), Hessian by central differences of gradients, harmonic
               frequencies; writes ``<name>_opt.xyz`` with every label below plus ``hessian`` and ``frequencies``.
``sample``     geometries from a reference file: ``--wigner N`` normal-mode samples from the harmonic Wigner
               distribution at ``--temperature`` (``--classical``: Boltzmann), and ``--scans``: rigid scans of every
               bond stretch (``--stretch`` x r_e), every angle (``--bend`` degrees either way), for a centre
               with three neighbours the umbrella through planar to the inverted structure, and a full turn about
               every rotatable bond. Scan frames carry
               ``config_type=scan_*`` so that they can be held out.
``label``      labels for every frame of an extxyz file; resumable (one part file per frame in ``<out>.parts/``),
               ``--workers`` frames in parallel.
``all``        the three in sequence for each species.

Labels (all atomic units; positions in Angstrom; moments about the coordinate origin; derivatives with respect to
nuclear positions in bohr). Everything beyond the first line comes from finite fields, so it works for any
functional pyscf has gradients for (VV10 included):

    energy, forces, dipole, quadrupole (9: the primitive second moment sum q r r), homo
    --fields (default)   6 SCF + gradients at +-F along x, y, z:
        polarizability (9)                      d mu_i / d F_j
        dipole_quadrupole_polarizability (27)   d M2_ab / d F_c, layout [a, b, c]
        dipole_derivatives (N*3*3)              d mu_k / d R_ai = d force_ai / d F_k, layout [atom, i, k]
    --raman              6 more at +-(F_j + F_k):
        polarizability_derivatives (N*3*9)      d alpha_jk / d R_ai, layout [atom, i, jk]
    --field-gradient     12 at +-G for the six symmetric field-gradient components (energy +1/2 r.G.r per electron):
        quadrupole_polarizability (81)          d M2_ab / d G_cd (G symmetric), layout [ab, cd]
        quadrupole_derivatives (N*3*9)          d M2_ab / d R_ai, layout [atom, i, ab]
    --hessian            6N gradients (or --analytic-hessian): hessian ((3N)^2, [atom a, i], [atom b, j]) and
                         frequencies (cm^-1)
    --esp                the electrostatic potential on shells around the molecule, at zero field and its
                         response to each field component, in ``<out>.esp.npz``: points_<frame> (P, 3) bohr,
                         esp_<frame> (P,), esp_response_<frame> (3, P) = d V / d F_k   (needs --fields)

Conventions: a field F adds -q F.r to the energy of a charge q, so mu = -dE/dF; forces are -dE/dR. The frame keeps
its input coordinates (no reorientation), ``charge``, ``multiplicity``, ``bonds`` and ``config_type``.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BOHR = 0.529177210903
CM = 219474.6313
AMU = 1822.888486
KB = 3.166811563e-6                    # Hartree / K
MASS = {"H": 1.00782503, "C": 12.0, "N": 14.0030740, "O": 15.99491462, "F": 18.99840316, "P": 30.97376200,
        "S": 31.97207117, "Cl": 34.96885268}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80, "Cl": 1.75}       # Angstrom
SPECIES = {      # starting geometries (Angstrom), charge, multiplicity
    "h2o": (["O", "H", "H"], [[0.0, 0.0, 0.117], [-0.763, 0.0, -0.468], [0.763, 0.0, -0.468]], 0, 1),
    "h3o+": (["O", "H", "H", "H"], [[0.0, 0.0, 0.075], [0.941, 0.0, -0.201], [-0.470, 0.815, -0.201],
                                    [-0.470, -0.815, -0.201]], 1, 1),
    "oh-": (["O", "H"], [[0.0, 0.0, -0.107], [0.0, 0.0, 0.858]], -1, 1),
    "h2o2": (["O", "O", "H", "H"], [[0.0, 0.725, -0.055], [0.0, -0.725, -0.055], [0.78, 0.89, 0.44], [-0.78, -0.89, 0.44]], 0, 1),
}
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02}      # Angstrom
SYM_G = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


# ---- extxyz -----------------------------------------------------------------------------------------------------
def default_bonds(sym):
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    if len(sym) == 2 and not heavy:
        return [(0, 1)]
    if len(heavy) != 1:
        raise ValueError("a frame without bonds=... needs exactly one heavy atom; give the covalent topology otherwise")
    return [(heavy[0], i) for i in range(len(sym)) if i != heavy[0]]


def perceive_bonds(sym, pos_ang, scale=1.25):
    """Bonds of ONE reference structure from covalent radii. Used only where a starting structure comes without a
    topology; every later frame carries the bonds it was given, so a stretched bond never drops out."""
    pos = np.asarray(pos_ang)
    return [(i, j) for i in range(len(sym)) for j in range(i + 1, len(sym))
            if np.linalg.norm(pos[i] - pos[j]) < scale * (COVALENT[sym[i]] + COVALENT[sym[j]])]


def read_frames(path):
    """[{sym, pos (n, 3) Angstrom, info}] of an extxyz file; info values are strings."""
    out, lines, i = [], Path(path).read_text().splitlines(), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n, head = int(lines[i].split()[0]), lines[i + 1]
        info = {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
                for m in re.finditer(r'([A-Za-z_]\w*)=(?:"([^"]*)"|(\S+))', head)}
        cols = info.pop("Properties", "species:S:1:pos:R:3").split(":")
        off = 0
        for name, _, w in zip(cols[0::3], cols[1::3], cols[2::3]):
            if name == "pos":
                break
            off += int(w)
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        out.append({"sym": [r[0] for r in rows], "pos": np.array([[float(x) for x in r[off:off + 3]] for r in rows]), "info": info})
        i += 2 + n
    return out


def frame_meta(fr):
    info = fr["info"]
    if "bonds" in info:
        bonds = np.array(info["bonds"].split(), dtype=int).reshape(-1, 2).tolist()
    elif sum(s != "H" for s in fr["sym"]) <= 1:
        bonds = default_bonds(fr["sym"])
    else:
        raise ValueError("this frame has several heavy atoms and no bonds=...: run the reference stage on it first "
                         "(it perceives the topology once and writes it), or add bonds=\"i j i j ...\"")
    return int(float(info.get("charge", 0))), int(float(info.get("multiplicity", 1))), [tuple(b) for b in bonds]


def _fmt(v):
    a = np.asarray(v, dtype=float).reshape(-1)
    return f"{a[0]:.12e}" if a.size == 1 and np.ndim(v) == 0 else '"' + " ".join(f"{x:.10e}" for x in a) + '"'


def write_frames(path, frames):
    """frames: [{sym, pos, forces (optional), info: {key: str | number | array}}]."""
    lines = []
    for fr in frames:
        forces = fr.get("forces")
        props = "species:S:1:pos:R:3" + (":forces:R:3" if forces is not None else "")
        items = [f"Properties={props}"]
        for k, v in fr["info"].items():
            items.append(f"{k}={v}" if isinstance(v, str) and " " not in v else (f'{k}="{v}"' if isinstance(v, str) else f"{k}={_fmt(v)}"))
        lines += [str(len(fr["sym"])), " ".join(items)]
        for a, s in enumerate(fr["sym"]):
            row = f"{s:2s} " + " ".join(f"{x:18.12f}" for x in fr["pos"][a])
            if forces is not None:
                row += " " + " ".join(f"{x:20.12e}" for x in forces[a])
            lines.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# ---- electronic structure ---------------------------------------------------------------------------------------
class Level:
    def __init__(self, xc="wb97m-v", basis="def2-tzvpd", grid=(99, 590), conv=1e-11, threads=None, backend="pyscf",
                 analytic_hessian=False):
        if backend not in ("pyscf", "gpu4pyscf"):
            raise ValueError(f"backend {backend!r}: pyscf or gpu4pyscf")
        self.xc, self.basis, self.grid, self.conv, self.threads = xc, basis, tuple(grid), conv, threads
        self.backend, self.analytic_hessian = backend, bool(analytic_hessian)

    def mol(self, sym, pos_ang, charge, mult):
        from pyscf import gto
        return gto.M(atom=[(s, tuple(p)) for s, p in zip(sym, np.asarray(pos_ang))], basis=self.basis, charge=charge,
                     spin=mult - 1, unit="Angstrom", verbose=0)


def _np(x):
    """A numpy array from a numpy or cupy array (or a tuple of them)."""
    if isinstance(x, (tuple, list)):
        return np.stack([_np(v) for v in x])
    return x.get() if hasattr(x, "get") else np.asarray(x)


class State:
    """One SCF of a molecule in a uniform field F and field gradient G (zero by default), with its gradient.

    The field enters the SCF through the core Hamiltonian. Its nuclear energy and its part of the gradient are added
    here, from the density, so the same code serves pyscf and gpu4pyscf (whose gradient objects do not see it)."""

    def __init__(self, level: Level, mol, F=None, G=None, dm0=None):
        from pyscf import lib
        if level.threads:
            lib.num_threads(level.threads)
        self.mol, self.level = mol, level
        self.gpu = level.backend == "gpu4pyscf"
        if self.gpu:
            import cupy
            from gpu4pyscf import dft, scf
            to_dev = cupy.asarray
        else:
            from pyscf import dft, scf
            to_dev = np.asarray
        self.F = np.zeros(3) if F is None else np.asarray(F, dtype=float)
        self.G = np.zeros((3, 3)) if G is None else np.asarray(G, dtype=float)
        hf = level.xc.lower() == "hf"
        mf = (scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)) if hf else (dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol))
        if not hf:
            mf.xc = level.xc
            mf.grids.atom_grid = level.grid
            mf.grids.prune = None
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = level.conv, 1e-7, 200
        self.field_on = bool(np.any(self.F) or np.any(self.G))
        self.Z, self.R = mol.atom_charges().astype(float), mol.atom_coords()
        if self.field_on:
            with mol.with_common_orig((0.0, 0.0, 0.0)):
                r = mol.intor("int1e_r")
                rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
            h = to_dev(_np(mf.get_hcore()) + np.einsum("k,kij->ij", self.F, r) + 0.5 * np.einsum("ab,abij->ij", self.G, rr))
            mf.get_hcore = lambda *a, **k: h
        mf.kernel(dm0=None if dm0 is None else to_dev(dm0))
        if not mf.converged and hasattr(mf, "newton"):
            try:
                mf = mf.newton()
                mf.kernel(dm0=mf.make_rdm1())
            except Exception:                                    # no second-order solver for this object
                pass
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        # the nuclei in the field: -Z (F.R + 1/2 R.G.R)
        e_nuc_field = -float(self.Z @ (self.R @ self.F + 0.5 * np.einsum("na,ab,nb->n", self.R, self.G, self.R)))
        self.mf, self.energy = mf, float(mf.e_tot) + e_nuc_field
        self.dm_spin = _np(mf.make_rdm1())
        self.dm = self.dm_spin if self.dm_spin.ndim == 2 else self.dm_spin[0] + self.dm_spin[1]

    def moments(self):
        """(dipole (3,), primitive second moment (3, 3)) about the coordinate origin."""
        mol = self.mol
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            r = mol.intor("int1e_r")
            rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
        mu = self.Z @ self.R - np.einsum("kij,ji->k", r, self.dm)
        m2 = np.einsum("n,na,nb->ab", self.Z, self.R, self.R) - np.einsum("abij,ji->ab", rr, self.dm)
        return mu, m2

    def forces(self):
        """(N, 3) -dE/dR, Hartree/bohr, field terms included."""
        mol = self.mol
        g = self.mf.nuc_grad_method()
        if hasattr(g, "grid_response"):
            g.grid_response = True
        grad = _np(g.kernel()).copy()
        if self.field_on:
            with mol.with_common_orig((0.0, 0.0, 0.0)):
                irp = mol.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)           # <mu| r_k d_x nu>
                irrp = mol.intor("int1e_irrp").reshape(3, 3, 3, mol.nao, mol.nao)      # <mu| r_a r_b d_x nu>
            o_ip = np.einsum("k,kxij->xij", self.F, irp) + 0.5 * np.einsum("ab,abxij->xij", self.G, irrp)
            for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                # d<mu|O|nu>/dR_a = -<mu|O d nu> (nu on a) - its transpose; the density is symmetric
                grad[a] -= 2.0 * np.einsum("xij,ij->x", o_ip[:, :, p0:p1], self.dm[:, p0:p1])
            grad -= self.Z[:, None] * (self.F[None, :] + self.R @ self.G)
        return -grad

    def analytic_hessian(self):
        """((3N, 3N) Hessian, dipole derivatives (N, 3, 3) [atom, i, k] or None) from the program's analytic
        second derivatives, at zero field. The dipole derivatives (closed shells only) reuse the Hessian's
        coupled-perturbed orbitals: d mu_k / d R_ai = Z_a delta_ik - Tr[dD/dR_ai r_k] - Tr[D d r_k / d R_ai]."""
        if self.field_on:
            raise ValueError("the analytic Hessian is for the zero-field state")
        mol, mf, n = self.mol, self.mf, self.mol.natm
        hobj = mf.Hessian()
        hess = _np(hobj.kernel()).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        hess = 0.5 * (hess + hess.T)
        if mol.spin != 0:
            return hess, None
        mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
        atmlst = range(n)
        h1ao = hobj.make_h1(mo_coeff, mo_occ, None, atmlst)
        if self.gpu:
            fx = hobj.gen_vind(mo_coeff, mo_occ)
            mo1, _ = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, fx, atmlst, hobj.max_memory, None)
            c = _np(mo_coeff)
            mo1 = [np.einsum("up,ypi->yui", c, _np(mo1[a])) for a in atmlst]          # gpu4pyscf: mo1 in the MO basis
        else:
            mo1, _ = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, None, atmlst)
            mo1 = [np.asarray(mo1[a]) for a in atmlst]                                # pyscf: already AO x occupied
        c, occ = _np(mo_coeff), _np(mo_occ)
        mocc = c[:, occ > 0]
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            r = mol.intor("int1e_r")
            irp = mol.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)
        apt = np.zeros((n, 3, 3))
        for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
            dm1 = 2.0 * np.einsum("xui,vi->xuv", mo1[a], mocc)
            dm1 = dm1 + dm1.transpose(0, 2, 1)
            apt[a] = -np.einsum("kuv,xvu->xk", r, dm1)
            apt[a] += 2.0 * np.einsum("kxij,ij->xk", irp[:, :, :, p0:p1], self.dm[:, p0:p1])
            apt[a] += self.Z[a] * np.eye(3)
        return hess, apt

    def esp(self, points):
        """(P,) electrostatic potential of the molecule's own charge distribution at ``points`` (bohr)."""
        v = np.einsum("n,pn->p", self.Z, 1.0 / np.linalg.norm(points[:, None, :] - self.R[None], axis=-1))
        for i in range(0, len(points), 200):
            v[i:i + 200] -= np.einsum("pij,ji->p", self.mol.intor("int1e_grids", grids=points[i:i + 200]), self.dm)
        return v

    def homo(self):
        mo_e, occ = _np(self.mf.mo_energy), _np(self.mf.mo_occ)
        return float(mo_e[occ > 0].max())


def esp_points(sym, pos_ang, scales=(1.4, 1.6, 1.8, 2.0, 2.5), per_sphere=40):
    """Points (bohr) on scaled van der Waals shells around the atoms, outside every atom's sphere of that scale."""
    k = np.arange(per_sphere) + 0.5
    polar, azim = np.arccos(1 - 2 * k / per_sphere), np.pi * (1 + 5 ** 0.5) * k
    unit = np.stack([np.sin(polar) * np.cos(azim), np.sin(polar) * np.sin(azim), np.cos(polar)], 1)
    rad = np.array([VDW[s] for s in sym])
    pts = []
    for scale in scales:
        for a in range(len(sym)):
            p = pos_ang[a] + scale * rad[a] * unit
            d = np.linalg.norm(p[:, None, :] - pos_ang[None], axis=-1)
            pts.append(p[(d >= scale * rad[None] - 1e-9).all(1)])
    return np.concatenate(pts) / BOHR


def frequencies(hessian, sym):
    """Harmonic frequencies (cm^-1, ascending; imaginary as negative) and the mass-weighted eigen-system."""
    n = len(sym)
    m = np.repeat([MASS[s] * AMU for s in sym], 3)
    w2, modes = np.linalg.eigh(hessian / np.sqrt(m[:, None] * m[None, :]))
    n_vib = 3 * n - (5 if n == 2 else 6)
    order = np.argsort(np.abs(w2))[-n_vib:]
    order = order[np.argsort(w2[order])]
    return np.sign(w2[order]) * np.sqrt(np.abs(w2[order])) * CM, w2[order], modes[:, order], m


def numerical_hessian(level, sym, pos_ang, charge, mult, step=0.005, dm0=None):
    """(3N, 3N) Hartree/bohr^2 by central differences of analytic gradients (step in bohr)."""
    n = len(sym)
    h = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for i in range(3):
            f = []
            for s in (1, -1):
                p = np.array(pos_ang, dtype=float)
                p[a, i] += s * step * BOHR
                f.append(State(level, level.mol(sym, p, charge, mult), dm0=dm0).forces())
            h[3 * a + i] = -(f[0] - f[1]).reshape(-1) / (2 * step)
    return 0.5 * (h + h.T)


def label_frame(level, sym, pos_ang, charge, mult, fields=True, raman=False, field_gradient=False, hessian=False,
                esp=False, f=0.002, g=0.002):
    """Every requested label of one geometry: {key: array} (atomic units)."""
    n = len(sym)
    mol = level.mol(sym, pos_ang, charge, mult)
    s0 = State(level, mol)
    f0 = s0.forces()
    mu0, m20 = s0.moments()
    out = {"energy": s0.energy, "forces": f0, "dipole": mu0, "quadrupole": m20, "homo": s0.homo()}
    pts = esp_points(sym, np.asarray(pos_ang)) if esp else None
    if esp:
        out["esp_points"], out["esp"] = pts, s0.esp(pts)
    if fields or raman:
        eye = np.eye(3)
        st = {(j, sgn): State(level, mol, F=sgn * f * eye[j], dm0=s0.dm_spin) for j in range(3) for sgn in (1, -1)}
        ff = {k: v.forces() for k, v in st.items()}
        mom = {k: v.moments() for k, v in st.items()}
        alpha = np.stack([(mom[j, 1][0] - mom[j, -1][0]) / (2 * f) for j in range(3)], 1)                # [i, j]
        out["polarizability"] = 0.5 * (alpha + alpha.T)
        out["dipole_quadrupole_polarizability"] = np.stack([(mom[c, 1][1] - mom[c, -1][1]) / (2 * f) for c in range(3)], -1)
        out["dipole_derivatives"] = np.stack([(ff[k, 1] - ff[k, -1]) / (2 * f) for k in range(3)], -1)   # [a, i, k]
        if esp:
            out["esp_response"] = np.stack([(st[k, 1].esp(pts) - st[k, -1].esp(pts)) / (2 * f) for k in range(3)])
        if raman:
            d = np.zeros((n, 3, 3, 3))
            for j in range(3):
                d[:, :, j, j] = (ff[j, 1] + ff[j, -1] - 2 * f0) / f ** 2
            for j in range(3):
                for k in range(j + 1, 3):
                    fp = State(level, mol, F=f * (eye[j] + eye[k]), dm0=s0.dm_spin).forces()
                    fm = State(level, mol, F=-f * (eye[j] + eye[k]), dm0=s0.dm_spin).forces()
                    d[:, :, j, k] = d[:, :, k, j] = 0.5 * ((fp + fm - 2 * f0) / f ** 2 - d[:, :, j, j] - d[:, :, k, k])
            out["polarizability_derivatives"] = d.reshape(n, 3, 9)
    if field_gradient:
        c = np.zeros((3, 3, 3, 3))
        dq = np.zeros((n, 3, 3, 3))
        for a, b in SYM_G:
            S = np.zeros((3, 3))
            S[a, b] = S[b, a] = 1.0
            sp, sm = (State(level, mol, G=sgn * g * S, dm0=s0.dm_spin) for sgn in (1, -1))
            w = 1.0 if a == b else 0.5                          # an off-diagonal S moves G_ab and G_ba together
            c[:, :, a, b] = c[:, :, b, a] = w * (sp.moments()[1] - sm.moments()[1]) / (2 * g)
            # E(G) = E0 - 1/2 G : M2, so d force / d g = 1/2 S : dM2/dR = (1 or 2) * 1/2 dM2_ab/dR
            dq[:, :, a, b] = dq[:, :, b, a] = (sp.forces() - sm.forces()) / (2 * g) * (2.0 if a == b else 1.0)
        out["quadrupole_polarizability"] = c.reshape(9, 9)
        out["quadrupole_derivatives"] = dq.reshape(n, 3, 9)
    if hessian:
        if level.analytic_hessian:
            h, apt = s0.analytic_hessian()
            if apt is not None:
                if "dipole_derivatives" in out:
                    out["dipole_derivatives_check"] = float(np.abs(out["dipole_derivatives"] - apt).max())
                out["dipole_derivatives"] = apt
        else:
            h = numerical_hessian(level, sym, pos_ang, charge, mult, dm0=s0.dm_spin)
        out["hessian"], out["frequencies"] = h, frequencies(h, sym)[0]
    return out


# ---- geometries -------------------------------------------------------------------------------------------------
def optimize(level, sym, pos_ang, charge, mult, gtol=2e-5):
    """BFGS on the analytic gradient; returns positions (Angstrom)."""
    from scipy.optimize import minimize
    cache = {}

    def fun(x):
        s = State(level, level.mol(sym, x.reshape(-1, 3) * BOHR, charge, mult), dm0=cache.get("dm"))
        cache["dm"] = s.dm_spin
        return s.energy, -s.forces().reshape(-1)

    x0 = np.asarray(pos_ang, dtype=float).reshape(-1) / BOHR
    sol = minimize(fun, x0, jac=True, method="BFGS", options={"gtol": gtol, "maxiter": 200})
    return sol.x.reshape(-1, 3) * BOHR


def wigner_samples(pos_ang, hessian, sym, n, temperature=300.0, classical=False, seed=0):
    """(n, N, 3) Angstrom: normal-mode samples about ``pos_ang`` from the harmonic Wigner (or Boltzmann) density."""
    _, w2, modes, m = frequencies(hessian, sym)
    if (w2 <= 0).any():
        raise ValueError("the reference structure has an imaginary frequency; it is not a minimum")
    w = np.sqrt(w2)
    kt = KB * temperature
    var = kt / w2 if classical else (0.5 / w) / np.tanh(0.5 * w / max(kt, 1e-12))
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, len(w))) * np.sqrt(var)                       # mass-weighted normal coordinates
    dx = (q @ modes.T) / np.sqrt(m)
    return np.asarray(pos_ang)[None] + dx.reshape(n, -1, 3) * BOHR


def _rotate(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * (axis @ v) * (1 - math.cos(angle))


def scans(sym, pos_ang, bonds, stretch=(0.75, 1.6), bend=35.0, points=13):
    """[(config_type, positions)]: rigid scans about ``pos_ang`` of every bond, every angle, every umbrella and every
    rotatable bond (a full turn)."""
    pos, out = np.asarray(pos_ang, dtype=float), []
    nbr = {a: [] for a in range(len(sym))}
    for i, j in bonds:
        nbr[i].append(j)
        nbr[j].append(i)
    for i, j in bonds:                                         # move the lighter end (the one with fewer neighbours)
        a, b = (i, j) if len(nbr[j]) <= len(nbr[i]) else (j, i)
        for s in np.linspace(stretch[0], stretch[1], points):
            p = pos.copy()
            p[b] = pos[a] + s * (pos[b] - pos[a])
            out.append((f"scan_stretch_{i}_{j}", p))
    for a, ns in nbr.items():
        for x in range(len(ns)):
            for y in range(x + 1, len(ns)):
                i, k = ns[x], ns[y]
                u, v = pos[i] - pos[a], pos[k] - pos[a]
                axis = np.cross(u, v)
                if np.linalg.norm(axis) < 1e-8:
                    continue
                for d in np.radians(np.linspace(-bend, bend, points)):
                    p = pos.copy()
                    p[i], p[k] = pos[a] + _rotate(u, axis, -0.5 * d), pos[a] + _rotate(v, axis, 0.5 * d)
                    out.append((f"scan_bend_{i}_{a}_{k}", p))
        if len(ns) == 3:                                       # umbrella: the centre through the neighbours' plane
            c = pos[ns].mean(0)
            normal = np.cross(pos[ns[1]] - pos[ns[0]], pos[ns[2]] - pos[ns[0]])
            normal /= np.linalg.norm(normal)
            h0 = (pos[a] - c) @ normal
            if abs(h0) > 1e-3:
                for t in np.linspace(1.3, -1.3, 2 * points - 1):
                    p = pos.copy()
                    p[a] = pos[a] + (t - 1.0) * h0 * normal
                    out.append((f"scan_umbrella_{a}", p))
    for b, c in bonds:                                         # torsion: rotate c's side about b-c, if it is not a ring
        if len(nbr[b]) < 2 or len(nbr[c]) < 2:
            continue
        side, stack = {c}, [c]
        while stack:
            for x in nbr[stack.pop()]:
                if x not in side and x != b:
                    side.add(x)
                    stack.append(x)
        if any(b in nbr[x] for x in side if x != c):           # the bond is in a ring
            continue
        axis = pos[c] - pos[b]
        for ang in np.radians(np.linspace(0.0, 360.0, 2 * points - 1, endpoint=False)[1:]):
            p = pos.copy()
            for x in side:
                p[x] = pos[c] + _rotate(pos[x] - pos[c], axis, ang)
            out.append((f"scan_torsion_{b}_{c}", p))
    return out


# ---- stages -----------------------------------------------------------------------------------------------------
SCALAR = ("energy", "homo", "dipole_derivatives_check")


def _record(sym, pos, charge, mult, bonds, lab, extra):
    info = {"energy": float(lab["energy"]), "charge": str(charge), "multiplicity": str(mult),
            "bonds": " ".join(f"{i} {j}" for i, j in bonds), "units": "atomic", **extra}
    for k, v in lab.items():
        if k not in ("energy", "forces", "esp", "esp_points", "esp_response"):
            info[k] = float(v) if k in SCALAR else np.asarray(v)
    return {"sym": sym, "pos": np.asarray(pos), "forces": lab["forces"], "info": info}


def _pick_gpu(queue):
    """Worker initializer: bind this process to one GPU before anything imports cupy."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(queue.get())


def _label_one(job):
    idx, sym, pos, charge, mult, part, level_kw, flags = job
    if part.exists():
        return idx, 0.0
    t0 = time.time()
    lab = label_frame(Level(**level_kw), sym, pos, charge, mult, **flags)
    tmp = part.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in lab.items()})
    os.replace(tmp, part)
    return idx, time.time() - t0


def label_file(geoms, out, level_kw, flags, workers=1, threads=None, method="", gpus=1):
    frames = read_frames(geoms)
    out = Path(out)
    parts = Path(str(out) + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    threads = threads or max(1, (os.cpu_count() or 1) // max(workers, 1))
    jobs = [(i, fr["sym"], fr["pos"], *frame_meta(fr)[:2], parts / f"frame_{i:06d}.npz", {**level_kw, "threads": threads}, flags)
            for i, fr in enumerate(frames)]
    todo = sum(not j[5].exists() for j in jobs)
    print(f"{geoms}: {len(frames)} frames, {todo} to compute, {workers} at a time with {threads} threads each", flush=True)
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        ctx, init = mp.get_context("spawn"), {}
        if level_kw.get("backend") == "gpu4pyscf":
            queue = ctx.Queue()
            for w in range(workers):
                queue.put(w % max(gpus, 1))
            init = {"initializer": _pick_gpu, "initargs": (queue,)}
        with ProcessPoolExecutor(workers, mp_context=ctx, **init) as pool:
            for k, (i, dt) in enumerate(pool.map(_label_one, jobs)):
                if dt:
                    print(f"  frame {i} done in {dt:.0f} s ({k + 1}/{len(jobs)}, {time.time() - t0:.0f} s elapsed)", flush=True)
    else:
        for k, job in enumerate(jobs):
            i, dt = _label_one(job)
            if dt:
                print(f"  frame {i} done in {dt:.0f} s ({k + 1}/{len(jobs)}, {time.time() - t0:.0f} s elapsed)", flush=True)
    records, esp = [], {}
    for i, fr in enumerate(frames):
        lab = dict(np.load(parts / f"frame_{i:06d}.npz"))
        charge, mult, bonds = frame_meta(fr)
        extra = {"config_type": fr["info"].get("config_type", "sample"), "frame": str(i), "method": method}
        records.append(_record(fr["sym"], fr["pos"], charge, mult, bonds, lab, extra))
        for k in ("esp_points", "esp", "esp_response"):
            if k in lab:
                esp[f"{k.replace('esp_points', 'points')}_{i}"] = lab[k]
    write_frames(out, records)
    if esp:
        np.savez(Path(str(out).rsplit(".", 1)[0] + ".esp.npz"), **esp)
    checks = [r["info"]["dipole_derivatives_check"] for r in records if "dipole_derivatives_check" in r["info"]]
    if checks:
        print(f"  analytic against finite-field dipole derivatives: largest difference {max(checks):.1e} e over {len(checks)} frames")
    unbound = [r["info"]["frame"] for r in records if r["info"]["homo"] > 0]
    if unbound:
        print(f"  warning: {len(unbound)} frames have a positive HOMO energy (an unbound electron): frames {' '.join(unbound[:20])}")
    print(f"wrote {out} ({len(records)} frames)" + (" and its .esp.npz" if esp else ""), flush=True)


def reference(name, level_kw, out_dir, flags, xyz=None, charge=0, mult=1, method=""):
    level = Level(**level_kw)
    if xyz:
        fr = read_frames(xyz)[0]
        sym, pos = fr["sym"], fr["pos"]
        if "bonds" in fr["info"]:
            bonds = frame_meta(fr)[2]
        else:
            bonds = perceive_bonds(sym, pos)
        charge = int(float(fr["info"].get("charge", charge)))
        mult = int(float(fr["info"].get("multiplicity", mult)))
    else:
        sym, pos, charge, mult = SPECIES[name]
        bonds = perceive_bonds(sym, pos)
    t0 = time.time()
    pos = optimize(level, sym, pos, charge, mult)
    lab = label_frame(level, sym, pos, charge, mult, **{**flags, "hessian": True})
    path = Path(out_dir) / f"{name}_opt.xyz"
    write_frames(path, [_record(sym, pos, charge, mult, bonds, lab, {"config_type": "minimum", "method": method})])
    if "esp" in lab:
        np.savez(Path(out_dir) / f"{name}_opt.esp.npz", points_0=lab["esp_points"], esp_0=lab["esp"],
                 **({"esp_response_0": lab["esp_response"]} if "esp_response" in lab else {}))
    print(f"{name}: optimized and labelled in {time.time() - t0:.0f} s; frequencies " + " ".join(f"{w:.0f}" for w in lab["frequencies"])
          + (f"; analytic against finite-field dipole derivatives {lab['dipole_derivatives_check']:.1e} e"
             if "dipole_derivatives_check" in lab else ""))
    return path


def sample(ref, out, wigner=400, temperature=600.0, classical=False, do_scans=True, stretch=(0.75, 1.6), bend=35.0,
           points=13, seed=0):
    fr = read_frames(ref)[0]
    charge, mult, bonds = frame_meta(fr)
    n = len(fr["sym"])
    base = {"charge": str(charge), "multiplicity": str(mult), "bonds": " ".join(f"{i} {j}" for i, j in bonds)}
    frames = []
    if wigner:
        if "hessian" not in fr["info"]:
            raise ValueError(f"{ref} has no hessian= (run the reference stage)")
        h = np.array(fr["info"]["hessian"].split(), dtype=float).reshape(3 * n, 3 * n)
        for p in wigner_samples(fr["pos"], h, fr["sym"], wigner, temperature, classical, seed):
            frames.append({"sym": fr["sym"], "pos": p, "info": {**base, "config_type": "wigner"}})
    if do_scans:
        for tag, p in scans(fr["sym"], fr["pos"], bonds, stretch, bend, points):
            frames.append({"sym": fr["sym"], "pos": p, "info": {**base, "config_type": tag}})
    write_frames(out, frames)
    print(f"wrote {out}: {wigner} normal-mode samples at {temperature:.0f} K, {len(frames) - wigner} scan frames")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--xc", default="wb97m-v")
        p.add_argument("--basis", default="def2-tzvpd")
        p.add_argument("--grid", type=int, nargs=2, default=(99, 590))
        p.add_argument("--threads", type=int, default=None, help="threads per SCF (default: cores / workers)")
        p.add_argument("--no-fields", action="store_true", help="energy, forces and moments only")
        p.add_argument("--raman", action="store_true")
        p.add_argument("--field-gradient", action="store_true")
        p.add_argument("--hessian", action="store_true")
        p.add_argument("--analytic-hessian", action="store_true",
                       help="Hessians (and closed-shell dipole derivatives) from the analytic code, not from gradients")
        p.add_argument("--backend", choices=("pyscf", "gpu4pyscf"), default="pyscf")
        p.add_argument("--gpus", type=int, default=1, help="gpu4pyscf: devices to spread the workers over")
        p.add_argument("--esp", action="store_true")
        p.add_argument("--field", type=float, default=0.002, help="field strength, atomic units")
        p.add_argument("--gradient", type=float, default=0.002, help="field-gradient strength, atomic units")

    def sampling(p):
        p.add_argument("--wigner", type=int, default=400)
        p.add_argument("--temperature", type=float, default=600.0)
        p.add_argument("--classical", action="store_true")
        p.add_argument("--scans", action="store_true")
        p.add_argument("--stretch", type=float, nargs=2, default=(0.75, 1.6))
        p.add_argument("--bend", type=float, default=35.0)
        p.add_argument("--points", type=int, default=13)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("reference")
    p.add_argument("--species", nargs="+", default=[], help=f"built in: {' '.join(SPECIES)}")
    p.add_argument("--xyz", help="or one molecule from an (ext)xyz file; --name, --charge, --multiplicity. Without "
                                 "bonds=... in the file its topology is perceived once, from this structure")
    p.add_argument("--name")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--out", default="data/bonded")
    common(p)
    p = sub.add_parser("sample")
    p.add_argument("reference")
    p.add_argument("--out", required=True)
    sampling(p)
    p = sub.add_parser("label")
    p.add_argument("geometries")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    common(p)
    p = sub.add_parser("all")
    p.add_argument("--species", nargs="+", required=True)
    p.add_argument("--out", default="data/bonded")
    p.add_argument("--workers", type=int, default=1)
    common(p)
    sampling(p)
    a = ap.parse_args(argv)

    if a.stage == "sample":
        sample(a.reference, a.out, a.wigner, a.temperature, a.classical, a.scans, tuple(a.stretch), a.bend, a.points, a.seed)
        return
    level_kw = {"xc": a.xc, "basis": a.basis, "grid": tuple(a.grid), "backend": a.backend,
                "analytic_hessian": a.analytic_hessian}
    flags = {"fields": not a.no_fields, "raman": a.raman, "field_gradient": a.field_gradient, "hessian": a.hessian,
             "esp": a.esp, "f": a.field, "g": a.gradient}
    if a.esp and a.no_fields:
        print("--esp without fields: the zero-field potential only", file=sys.stderr)
    method = f"{a.xc}/{a.basis}"
    if a.stage == "reference":
        kw = {**level_kw, "threads": a.threads}
        if a.xyz:
            reference(a.name or Path(a.xyz).stem, kw, a.out, flags, a.xyz, a.charge, a.multiplicity, method)
        for name in a.species:
            reference(name, kw, a.out, flags, method=method)
    elif a.stage == "label":
        label_file(a.geometries, a.out, level_kw, flags, a.workers, a.threads, method, a.gpus)
    else:
        for name in a.species:
            ref = reference(name, {**level_kw, "threads": a.threads}, a.out, flags, method=method)
            geoms = sample(ref, Path(a.out) / f"{name}_geoms.xyz", a.wigner, a.temperature, a.classical, True,
                           tuple(a.stretch), a.bend, a.points, a.seed)
            label_file(geoms, Path(a.out) / f"{name}.xyz", level_kw, flags, a.workers, a.threads, method, a.gpus)


if __name__ == "__main__":
    main()
