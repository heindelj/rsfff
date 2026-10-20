"""l-decomposition of the CPKS polarization energy of an ion by a point charge, E_l = 1/2 int drho phi_l. Usage: ion_response_ldecomp.py Cl 1.0 2.0 2.6 3.0"""
import sys, numpy as np
from pyscf import dft
from scipy.special import eval_legendre
from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import pol_ref as PR
BOHR = 0.529177210903; KJ = 2625.4996
ion = sys.argv[1]; q = float(sys.argv[2])
mono = E.Monomer.load(f"/home/claude/a1/db_new/monomers/{ {'Cl':'Clm','Na':'Nap'}[ion]}.npz")
placed = mono.place(); mol = placed.mol
mf0 = PR.isolated(placed)
g = dft.gen_grid.Grids(mol); g.level = 4; g.build()
ao = dft.numint.eval_ao(mol, g.coords)
r = np.linalg.norm(g.coords, axis=1); cos = g.coords[:, 2] / np.maximum(r, 1e-12)
LMAX = 6
for R in [float(x) for x in sys.argv[3:]]:
    Rb = R / BOHR
    env = PR.Environment(charge_coords=np.array([[0.0, 0.0, Rb]]), charges=np.array([q]))
    res = PR.polarize(placed, env, mf0=mf0)
    for tag, dm in (("lin", res["ddm_lin"]), ("scf", res["ddm"])):
        rho = -dft.numint.eval_rho(mol, ao, dm)           # electron density change -> charge density change
        El = []
        for l in range(LMAX + 1):
            rl, rg = np.minimum(r, Rb), np.maximum(r, Rb)
            phi = q * rl ** l / rg ** (l + 1) * eval_legendre(l, cos)
            El.append(0.5 * np.sum(g.weights * rho * phi) * KJ)
        tot = np.sum(g.weights * rho * q / np.sqrt(r * r + Rb * Rb - 2 * r * Rb * cos)) * 0.5 * KJ
        ref = res["e_pol_lin"] if tag == "lin" else res["e_pol"]
        print(f"R {R:4.2f} {tag}: E_pol {ref*KJ:9.3f} (grid {tot:9.3f}) | " + " ".join(f"l{l} {e:8.3f}" for l, e in enumerate(El)) + f" | sum {sum(El):9.3f}", flush=True)
