"""Finite-field dipole polarizabilities of the free ions and their CT radicals (the neutral atoms), wB97M-V/def2-TZVPD
(UKS for the open shells, def2 ECPs for Cs and I; field 0.002 au along x, y, z). The values in
rsfff.ff.cmm.model.RADICAL_ALPHA (``CmmConfig.ct_alpha_radical``) come from

    python scripts/cmm_radical_alpha.py Li:1:0:1 Na:1:0:1 K:1:0:1 Cs:1:0:1 F:-1:0:1 Cl:-1:0:1 Br:-1:0:1 I:-1:0:1

(symbol:ion charge:ion 2S:radical 2S). The ions reproduce the species table's alpha. Needs pyscf."""
import sys, numpy as np
from pyscf import gto, dft
F = 0.002
def energy(sym, charge, spin, field, xc="wb97m-v"):
    mol = gto.M(atom=f"{sym} 0 0 0", basis="def2-tzvpd", ecp="def2-tzvpd", charge=charge, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc = xc
    mf.nlc = "vv10" if "-v" in xc else ""
    mf.grids.level = 5
    mf.conv_tol = 1e-11
    h = mf.get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        d = mol.intor("int1e_r")
    mf.get_hcore = lambda *a: h + np.einsum("x,xij->ij", field, d)
    if spin:   # an occupied p_z hole (halogens) / s electron (alkalis): break the symmetry the same way for every field
        dm0 = mf.get_init_guess()
        mf.kernel(dm0)
    else:
        mf.kernel()
    return mf.e_tot
def alpha(sym, charge, spin):
    e0 = energy(sym, charge, spin, np.zeros(3))
    a = []
    for x in range(3):
        f = np.zeros(3); f[x] = F
        ep, em = energy(sym, charge, spin, f), energy(sym, charge, spin, -f)
        a.append(-(ep + em - 2 * e0) / F ** 2)
    return np.array(a)
for sym, q_ion, s_ion, s_rad in [(s, int(q), int(a), int(b)) for s, q, a, b in (x.split(":") for x in sys.argv[1:])]:
    ai = alpha(sym, q_ion, s_ion)
    ar = alpha(sym, 0, s_rad)
    print(f"{sym}: ion {ai.round(2)} mean {ai.mean():.2f}   radical {ar.round(2)} mean {ar.mean():.2f}", flush=True)
