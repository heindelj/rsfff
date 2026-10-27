"""ALMO-style (SCF-MI) polarization of Cl- in Na_n shells with the Pauli constraint switched on and off.

Each fragment's occupied orbitals live in its own AOs, projected out of the other fragments' occupied space
(antisymmetrized product = the orthogonalized one; the energy depends only on the span), as in ALMO-EDA.
Polarization = E(relaxed) - E(frozen), the frozen state built from the isolated fragment orbitals.

Modes (Cl- relaxes; Na+ frozen unless noted):
  P    : Cl- own KS functional, ghost Na basis; Na+ enters ONLY through orthogonality (pure Pauli / confinement drive)
  CP   : P + the Coulomb potential of the frozen Na+ (nuclei + density)  -- Coulomb + orthogonality, no inter-fragment xc
  XP   : supersystem functional minus the Cl-/Na+ Coulomb interaction      -- Pauli = orthogonality + inter-fragment xc, no Coulomb
  full : supersystem KS functional of all fragments, Na+ frozen           -- = ALMO with frozen cations
  alm  : supersystem functional, every fragment relaxes                   -- = ALMO pol (validation vs Q-Chem)
(C, Coulomb only, is the FD SCF of ion_cage_response.py.)
Validated: alm reproduces Q-Chem ALMO-EDA pol (Na-Cl 2.6 A: -46.77 vs -46.76 kJ/mol). P / CP / XP are not a usable
Coulomb/Pauli split: without the cation's Coulomb attraction the orthogonalization tails at the cation cores are not
balanced, and the Pauli-only relaxation is huge (XP: -84 pair, -312 oct6 at 2.6 A vs full -44 / -54) -- kept as probes.
Usage: ion_cage_scfmi.py OUT.json shape:R ... [--modes=XP,full,alm] [-v]"""
import json, sys, time
from dataclasses import replace
import numpy as np
import scipy.linalg as sla
from pyscf import gto, lib

from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import pol_ref as PR
from rsfff.qcgen.backend import make_mf
from ion_cage_response import shell, cl, na, pcl, mf_cl, dm_na, na_c0, BOHR, KJ

C_cl0 = mf_cl.mo_coeff[:, mf_cl.mo_occ > 0]
pna0 = na.place()
mf_na0 = PR.isolated(pna0)
C_na0 = mf_na0.mo_coeff[:, mf_na0.mo_occ > 0]
nao_cl, nao_na = C_cl0.shape[0], C_na0.shape[0]


def sup_mol(sites_ang, ghost):
    atoms = [("Cl", (0.0, 0.0, 0.0))] + [("GHOST-Na" if ghost else "Na", tuple(s)) for s in sites_ang]
    n = len(sites_ang)
    return gto.M(atom=atoms, unit="Angstrom", basis="def2-tzvpd", charge=-1 if ghost else n - 1, verbose=0, symmetry=False)


def orth(C, S):
    """Lowdin-orthonormalize the columns of C (same span)."""
    s = C.T @ S @ C
    w, U = np.linalg.eigh(s)
    return C @ (U * w ** -0.5) @ U.T


def proj_space(cols, Cocc_other, S, tol=1e-8):
    """Orthonormal basis X = (1 - P_other) cols M of span(cols) projected out of span(Cocc_other)
    (Cocc_other S-orthonormal). Returns (X, M): cols @ M is the own-AO representative of X."""
    Y = cols - Cocc_other @ (Cocc_other.T @ S @ cols) if Cocc_other.shape[1] else cols
    s = Y.T @ S @ Y
    w, U = np.linalg.eigh(s)
    keep = w > tol * w.max()
    M = U[:, keep] / np.sqrt(w[keep])
    return Y @ M, M


def density(Cs, S):
    C = np.hstack(Cs)
    return 2 * C @ np.linalg.solve(C.T @ S @ C, C.T)


def scf_mi(mode, sites_ang, max_cycle=80, tol=1e-8, verbose=False):
    n = len(sites_ang)
    ghost = mode in ("P", "CP")
    mol = sup_mol(sites_ang, ghost)
    mf = make_mf(mol, "wb97m-v")
    S = mol.intor("int1e_ovlp")
    h = mf.get_hcore()
    enuc = mol.energy_nuc()
    if mode in ("CP", "XP"):
        env = PR.Environment(partners=[replace(na.place(np.eye(3), s / BOHR - na_c0), dm=dm_na) for s in sites_ang])
        # CP: add the Na+ Coulomb potential to the ghost-basis Cl- problem; XP: remove it from the supersystem
        # (constant nuclear terms dropped: they cancel in E - E_frz)
        h = h + env.h1(mol) if mode == "CP" else h
    nao = mol.nao
    sl = [slice(0, nao_cl)] + [slice(nao_cl + k * nao_na, nao_cl + (k + 1) * nao_na) for k in range(n)]
    eye = np.eye(nao)
    C0 = []
    for k, s in enumerate(sl):
        c = np.zeros((nao, (C_cl0 if k == 0 else C_na0).shape[1]))
        c[s] = C_cl0 if k == 0 else C_na0
        C0.append(c)
    relax = [0] if mode in ("P", "CP", "full", "XP") else list(range(n + 1))
    in_density = [0] if ghost else list(range(n + 1))

    Cna_fix = orth(np.hstack(C0[1:]), S)

    def energy_fock(Cs):
        if ghost:   # Cl- density alone, of its orbitals orthogonalized to the frozen Na+ occupied space
            Cp = Cs[0] - Cna_fix @ (Cna_fix.T @ S @ Cs[0])
            dm = density([Cp], S)
        else:
            dm = density(Cs, S)
        vhf = mf.get_veff(mol, dm)
        e = mf.energy_elec(dm, h, vhf)[0] + enuc
        return e, h + vhf

    if mode == "XP":   # remove exactly the Coulomb potential of the frozen (orthonormalized) Na+ set used here
        Cna_x = orth(np.hstack(C0[1:]), S)
        vna = np.zeros_like(h)
        for a in range(1, mol.natm):
            with mol.with_rinv_origin(mol.atom_coord(a)):
                vna -= mol.atom_charge(a) * mol.intor("int1e_rinv")
        h = h - vna - mf.get_j(mol, 2 * Cna_x @ Cna_x.T)
    # frozen state: isolated orbitals (the span is what matters)
    Cs = [c.copy() for c in C0]
    e_frz, F = energy_fock(Cs)
    no_k = {k: C0[k].shape[1] for k in relax}

    def spaces(Cs):
        out = {}
        for k in relax:
            Coth = orth(np.hstack([Cs[j] for j in range(n + 1) if j != k]), S)
            out[k] = proj_space(eye[:, sl[k]], Coth, S)
        return out

    def grad(F, Cs, Xs):
        g = []
        for k in relax:
            X = Xs[k][0]
            O, _ = np.linalg.qr(X.T @ S @ Cs[k])
            V = sla.null_space(O.T)
            g.append((V.T @ (X.T @ F @ X) @ O).ravel())
        return np.concatenate(g)

    diis = lib.diis.DIIS(); diis.space = 12
    e_old = e_frz
    for it in range(max_cycle):
        Xs = spaces(Cs)
        g = grad(F, Cs, Xs)
        if verbose:
            print(f"  {mode} it {it} E {e_old:.10f} |g| {np.abs(g).max():.2e}", flush=True)
        if it > 0 and abs(e - e_prev) < tol and np.abs(g).max() < 2e-5:
            break
        Fd = diis.update(F, g) if it > 0 else F
        newC = list(Cs)
        for k in relax:
            X, M = Xs[k]
            w, U = np.linalg.eigh(X.T @ Fd @ X)
            newC[k] = eye[:, sl[k]] @ (M @ U[:, :no_k[k]])          # stored in own AOs (ALMO form)
        Cs = newC
        e_prev = e_old
        e, F = energy_fock(Cs)
        e_old = e
    return {"e_frz": e_frz, "e": e_old, "pol": (e_old - e_frz) * KJ, "iters": it + 1, "gmax": float(np.abs(g).max())}


if __name__ == "__main__":
    out = sys.argv[1]
    modes = ["full", "alm"]
    args = [a for a in sys.argv[2:] if not a.startswith("-")]
    for a in sys.argv[2:]:
        if a.startswith("--modes"):
            modes = a.split("=")[1].split(",")
    recs = []
    for c in args:
        shape, R = c.split(":")[0], float(c.split(":")[1])
        rec = {"shape": shape, "R": R}
        for m in modes:
            t = time.time()
            r = scf_mi(m, shell(shape, R), verbose="-v" in sys.argv)
            rec[m] = r
            print(f"{shape:5s} R {R:.2f} {m:4s}: pol {r['pol']:8.2f} kJ/mol  ({r['iters']} it, {time.time() - t:.0f}s)", flush=True)
            recs_out = recs + [rec]
            json.dump(recs_out, open(out, "w"), indent=1)
        recs.append(rec)
