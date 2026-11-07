"""The induced density of the central Cl- in Na+ shells, by angular channel: ALMO-style SCF-MI (frozen Na+, the
supersystem functional: rsfff scripts/ion_cage_scfmi.py mode "full", = ALMO with frozen cations) and the
Coulomb-only linear response (frozen Na+ densities, ion_cage_response.py "FD lin").

The SCF-MI Cl- orbitals live in the Cl AOs only (ALMO form), so rho_Cl = 2 C (C^T S C)^-1 C^T on the Cl block is the
absolutely localized Cl- density; drho = rho_Cl - rho_Cl(isolated). Reports d<r^2>, the electrons moved by each l
channel (1/2 int |drho_l|) and the l = 0 radial profile 4 pi r^2 drho_0(r).

    python ion_cage_ldens.py OUT.json shape:R ...     (results: data/elst_db/validation/cage_ldens.json)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft

SCRIPTS = Path.home() / "dev/rsfff/scripts"
sys.path.insert(0, str(SCRIPTS))
import ion_cage_response as CR                                       # noqa: E402
from pyscf import lib as _lib                                          # noqa: E402
_lib.einsum = lambda *a, **k: np.einsum(*a, optimize=True)          # pyscf's einsum breaks on this numpy

src = (SCRIPTS / "ion_cage_scfmi.py").read_text().split('if __name__ == "__main__":')[0]
src = src.replace('return {"e_frz": e_frz, "e": e_old,', 'return {"C": Cs[0], "e_frz": e_frz, "e": e_old,')
SM = {"__name__": "scfmi"}
exec(compile(src, "ion_cage_scfmi.py", "exec"), SM)

mol = CR.mol                                   # isolated Cl- (the Cl block of the supersystem basis)
ncl = mol.nao
S = mol.intor("int1e_ovlp")
R2 = mol.intor("int1e_r2")                     # origin at the Cl nucleus
C0 = SM["C_cl0"]
D0 = 2 * C0 @ np.linalg.solve(C0.T @ S @ C0, C0.T)
pts, wr, wa, uv, rr = CR.pts, CR.wr, CR.wa, CR.uv, CR.rr
ao = CR.ao_cl
LMAX = 6


def analyse(dD):
    rho = dft.numint.eval_rho(mol, ao, dD).reshape(len(rr), len(uv))      # electron density change
    w3 = wr[:, None] * rr[:, None] ** 2
    out = {"dn": float(np.sum(w3 * rho * wa[None])), "dr2": float(np.einsum("ij,ji->", dD, R2)), "moved_l": []}
    for l in range(LMAX + 1):
        c = (rho * wa[None]) @ CR.Yang[l].T                               # (NR, 2l+1) radial coefficients
        rho_l = c @ CR.Yang[l]
        out["moved_l"].append(0.5 * float(np.sum(w3 * np.abs(rho_l) * wa[None])))
        if l == 0:
            out["r"] = (rr * CR.BOHR).tolist()
            out["p0"] = (4 * np.pi * rr ** 2 * c[:, 0] * CR.Yang[0][0, 0]).tolist()   # e / bohr
    return out


if __name__ == "__main__":
    recs = []
    for arg in sys.argv[2:]:
        shape, R = arg.split(":")[0], float(arg.split(":")[1])
        t = time.time()
        r = SM["scf_mi"]("full", CR.shell(shape, R))
        C = r["C"][:ncl]
        D = 2 * C @ np.linalg.solve(C.T @ S @ C, C.T)
        almo = analyse(D - D0)
        res = CR.PR.polarize(CR.pcl, CR.environment(CR.shell(shape, R) / CR.BOHR, "FD"), mf0=CR.mf_cl, linear=True)
        coul = analyse(res["ddm_lin"])
        rec = {"shape": shape, "R": R, "pol": r["pol"], "almo": almo, "coulomb": coul}
        recs.append(rec)
        json.dump(recs, open(sys.argv[1], "w"))
        f = lambda a: f"d<r2> {a['dr2']:+7.3f}  dN {a['dn']:+.1e}  moved l0..4 " + " ".join(f"{x:.4f}" for x in a["moved_l"][:5])
        print(f"{shape:5s} R {R:.2f} pol {r['pol']:7.2f} [{time.time() - t:.0f}s]\n   ALMO    {f(almo)}\n   Coulomb {f(coul)}",
              flush=True)
