"""Dispersion-relevant response properties of the free ions from electronic structure (PySCF).

For each closed-shell ion: the full linear-response (TDDFT / TDHF) spectrum from the A and B matrices, in the
oscillator form of each multipole polarizability at imaginary frequency,

    alpha_l(i w) = sum_k f_k / (W_k^2 + w^2),       Q_l0 = z,  (3 z^2 - r^2) / 2,  (5 z^3 - 3 z r^2) / 2,

from which the Casimir-Polder integrals are closed form (int_0^inf dw / ((a^2 + w^2)(b^2 + w^2)) = pi / (2ab(a+b))):

    C6 = (3/pi) int a1 a1,   C8 = (15/2pi) int (a1 a2' + a2 a1'),   C10 = (14/pi) int (a1 a3' + a3 a1') + (35/pi) int a2 a2'

and the one-pole (London) frequencies  w_l = (4 / pi) int alpha_l(i w)^2 dw / alpha_l(0)^2  (so C6_ss = 3/4 a1^2 w1).

    python scripts/cmm_ion_spectra.py validate            hydrogen atom, exact sum over states: C6 6.499, C8 124.4, C10 3285.8
    python scripts/cmm_ion_spectra.py run SPECTRA.json [xc] [basis] [ions ...]

The spectra file is the input of the cmm response dispersion's table (``CmmConfig(disp_model="response")``):

    python -m rsfff.ff.cmm.disp_table SPECTRA.json params/disp_<level>.json

(params/spectra_wb97mv_tzvpd.json -> params/disp_wb97mv_tzvpd.json are wB97M-V / def2-TZVPD, VV10 in the orbitals but
not in the response kernel.) For another level of theory rerun both and point ``disp_table`` at the new file.
"""
import json
import sys
import time

import numpy as np
from pyscf import dft, gto, scf
from pyscf.tdscf import rhf as tdrhf

IONS = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "F": -1, "Cl": -1, "Br": -1, "I": -1}
ECP = {"Rb", "Cs", "I"}


def multipole_ops(mol):
    """AO matrices of Q_10, Q_20, Q_30 about the origin."""
    r = mol.intor("int1e_r")
    rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
    rrr = mol.intor("int1e_rrr").reshape(3, 3, 3, mol.nao, mol.nao)
    r2 = rr[0, 0] + rr[1, 1] + rr[2, 2]
    zr2 = rrr[2, 0, 0] + rrr[2, 1, 1] + rrr[2, 2, 2]
    return [r[2], 0.5 * (3 * rr[2, 2] - r2), 0.5 * (5 * rrr[2, 2, 2] - 3 * zr2)]


class Spectrum:
    """alpha(i w) = sum_k f_k / (W_k^2 + w^2)."""

    def __init__(self, W, f):
        keep = f > 1e-14 * f.max()
        self.W, self.f = np.asarray(W)[keep], np.asarray(f)[keep]

    def alpha(self, w=0.0):
        return float((self.f / (self.W ** 2 + w * w)).sum())

    def cp(self, other):
        """int_0^inf alpha(i w) alpha'(i w) dw."""
        a, b = self.W[:, None], other.W[None, :]
        return float((self.f[:, None] * other.f[None, :] * np.pi / (2 * a * b * (a + b))).sum())

    def pole(self):
        """The one-pole frequency: (4 / pi) int alpha^2 / alpha(0)^2."""
        return 4 / np.pi * self.cp(self) / self.alpha() ** 2


def coefficients(A, B):
    """C6, C8, C10 (and its quadrupole-quadrupole part) between two sets of [dipole, quadrupole, octupole] spectra."""
    c6 = 3 / np.pi * A[0].cp(B[0])
    c8 = 15 / (2 * np.pi) * (A[0].cp(B[1]) + A[1].cp(B[0]))
    c10qq = 35 / np.pi * A[1].cp(B[1])
    c10 = 14 / np.pi * (A[0].cp(B[2]) + A[2].cp(B[0])) + c10qq
    return c6, c8, c10, c10qq


def hydrogen():
    """Exact (in the basis) sum over states of the hydrogen atom."""
    ex = [0.005 * 2.2 ** k for k in range(16)]
    basis = {"H": gto.basis.parse("\n".join(
        f"H {lab}\n  {e:.8f} 1.0" for lab, es in (("S", ex + [x * 2.2 ** 16 / 0.005 * 0.005 for x in (1.0, 2.2, 4.84, 10.6)]),
                                                 ("P", ex[:14]), ("D", ex[:13]), ("F", ex[:12])) for e in es))}
    mol = gto.M(atom="H 0 0 0", basis=basis, spin=1, verbose=0)
    S, h = mol.intor("int1e_ovlp"), mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    from scipy.linalg import eigh
    e, c = eigh(h, S)
    sp = []
    for op in multipole_ops(mol):
        t = c[:, 0] @ op @ c[:, 1:]
        sp.append(Spectrum(e[1:] - e[0], 2 * (e[1:] - e[0]) * t * t))
    return e[0], sp


def ion(sym, xc="wb97m-v", basis="def2-tzvpd", verbose=0, conf=None):
    """[dipole, quadrupole, octupole] spectra of the ion, plus (energy, n_occ, n_virt). ``conf`` = (k, r0, n): a
    confining wall k max(r - r0, 0)^n added to the core Hamiltonian (on the DFT grid)."""
    mol = gto.M(atom=f"{sym} 0 0 0", basis=basis, ecp=basis if sym in ECP else None, charge=IONS[sym], spin=0,
                verbose=verbose, symmetry=False)
    if xc.lower() == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 4
    if conf is not None:
        g = dft.gen_grid.Grids(mol)
        g.level = 5
        g.build()
        ao = dft.numint.eval_ao(mol, g.coords)
        rr = np.linalg.norm(g.coords, axis=1)
        k, r0, n = conf
        v = k * np.clip(rr - r0, 0, None) ** n
        vmat = (ao * (g.weights * v)[:, None]).T @ ao
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *a, **kw: h0 + vmat
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    if xc.lower() != "hf":
        mf.nlc = False                        # VV10 stays in the orbitals; its kernel is left out of the response
    A, B = tdrhf.get_ab(mf)
    no, nv = A.shape[0], A.shape[1]
    A, B = A.reshape(no * nv, no * nv), B.reshape(no * nv, no * nv)
    w, V = np.linalg.eigh(A - B)
    assert w.min() > 0, f"A - B not positive definite ({w.min():.2e})"
    S = (V * np.sqrt(w)) @ V.T
    W2, U = np.linalg.eigh(S @ (A + B) @ S)
    assert W2.min() > 0, f"instability ({W2.min():.2e})"
    co, cv = mf.mo_coeff[:, :no], mf.mo_coeff[:, no:]
    sp = []
    for op in multipole_ops(mol):
        d = (co.T @ op @ cv).reshape(-1)
        t = U.T @ (S @ d)
        sp.append(Spectrum(np.sqrt(W2), 4 * t * t))
    return sp, (mf.e_tot, no, nv)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        e0, sp = hydrogen()
        print(f"H atom: E = {e0:.6f} (-0.5); alpha_1,2,3 = {sp[0].alpha():.4f}, {sp[1].alpha():.4f}, {sp[2].alpha():.3f}"
              f" (4.5, 15, 131.25)")
        c6, c8, c10, c10qq = coefficients(sp, sp)
        print(f"H-H: C6 = {c6:.4f} (6.4990), C8 = {c8:.3f} (124.399), C10 = {c10:.1f} (3285.8); one-pole frequencies"
              f" {sp[0].pole():.4f}, {sp[1].pole():.4f}, {sp[2].pole():.4f}")
    else:
        out = sys.argv[2]
        xc = sys.argv[3] if len(sys.argv) > 3 else "wb97m-v"
        basis = sys.argv[4] if len(sys.argv) > 4 else "def2-tzvpd"
        ions = sys.argv[5:] or list(IONS)
        try:
            res = json.load(open(out))
        except FileNotFoundError:
            res = {}
        for x in ions:
            t = time.time()
            sp, (e, no, nv) = ion(x, xc, basis)
            res[x] = {"xc": xc, "basis": basis, "E": e, "nocc": no, "nvirt": nv,
                      "alpha": [s.alpha() for s in sp], "pole": [s.pole() for s in sp],
                      "W": [s.W.tolist() for s in sp], "f": [s.f.tolist() for s in sp]}
            print(f"{x:2s} {xc} / {basis}: nocc {no} nvirt {nv}; alpha_1,2,3 = {sp[0].alpha():9.3f} {sp[1].alpha():10.2f}"
                  f" {sp[2].alpha():11.1f}; poles {sp[0].pole():.3f} {sp[1].pole():.3f} {sp[2].pole():.3f}"
                  f"   [{time.time() - t:.0f} s]", flush=True)
            json.dump(res, open(out, "w"))
