"""Free atom / ion static dipole polarizabilities by finite field (pyscf): the dissociation limits of
the ECE electrical heads (``rsfff.ff.film.electrical.FREE_ATOM_ALPHA``).

    python scripts/free_atom_alpha.py --xc b3lyp --basis def2-svpd --out free_atom_alpha.json

alpha = -(E(+F) + E(-F) - 2 E(0)) / F^2 per axis, averaged over x, y, z: open-shell UKS ground states
are not spherically symmetric, and the average is the free-atom value. Use the level of the training
data, so the limit is the one the data would reach.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

#: (symbol, charge, multiplicity) of the ground states computed by default
STATES = [("H", 0, 2), ("C", 0, 3), ("N", 0, 4), ("O", 0, 3), ("F", 0, 2), ("S", 0, 3),
          ("C", 1, 2), ("N", 1, 3), ("O", 1, 4), ("C", -1, 4), ("O", -1, 2), ("F", -1, 1),
          ("S", -1, 2)]


def energy(mol, field, xc, dm0=None):
    mf = mol.RKS(xc=xc) if mol.spin == 0 else mol.UKS(xc=xc)
    mf.conv_tol, mf.max_cycle = 1e-12, 300
    mf.grids.level = 4
    h0 = mf.get_hcore()
    r = mol.intor("int1e_r")
    mf.get_hcore = lambda *a: h0 + np.einsum("x,xij->ij", field, r)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.make_rdm1())
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {mol.atom}")
    return mf.e_tot, mf.make_rdm1()


def main():
    import pyscf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xc", default="b3lyp")
    ap.add_argument("--basis", default="def2-svpd")
    ap.add_argument("--field", type=float, default=0.002)
    ap.add_argument("--out", default="free_atom_alpha.json")
    a = ap.parse_args()
    out = {}
    for sym, q, mult in STATES:
        mol = pyscf.M(atom=[(sym, (0.0, 0.0, 0.0))], basis=a.basis, charge=q, spin=mult - 1,
                      symmetry=False, verbose=0)
        e0, dm = energy(mol, np.zeros(3), a.xc)
        axes = []
        for ax in range(3):
            f = np.zeros(3)
            f[ax] = a.field
            ep, _ = energy(mol, f, a.xc, dm)
            em, _ = energy(mol, -f, a.xc, dm)
            axes.append(-(ep + em - 2 * e0) / a.field ** 2)
        out[f"{sym}{q:+d}"] = {"Z": int(pyscf.gto.charge(sym)), "charge": q, "multiplicity": mult,
                               "alpha_iso": float(np.mean(axes)), "alpha_axes": [float(v) for v in axes]}
        print(f"{sym}{q:+d} (2S+1={mult}): {np.mean(axes):.3f}  axes {np.round(axes, 3)}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
