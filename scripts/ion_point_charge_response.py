"""Ion polarized by a point charge at R (pyscf full SCF and CPKS) vs the single-shape classical parent: the shell-split check (complete model plan, 2026-10-20). Usage: ion_point_charge_response.py Cl 1.0"""
import json, sys, numpy as np, torch
from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import pol_ref as PR
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable
BOHR = K.BOHR_ANG; KJ = 2625.4996
tab = IonTable.load("/home/claude/env/rsfff/params/ions_wb97mv_tzvpd_pf2.json")
ion, q = sys.argv[1], float(sys.argv[2])
name = {"Cl": "Clm", "Na": "Nap"}[ion]
mono = E.Monomer.load(f"/home/claude/a1/db_new/monomers/{name}.npz")
placed = mono.place()
mf0 = PR.isolated(placed)
s = tab.symbols.index(ion)
P = K._poly_map(torch.float64, "cpu")
def classical(R_ang, q):
    r = torch.tensor([R_ang / BOHR], dtype=torch.float64)
    dr = torch.tensor([[0.0, 0.0, R_ang / BOHR]], dtype=torch.float64)
    out = {}
    for par, (bp, wp) in (("d", (tab.b_dip[s:s+1], tab.w_dip[s:s+1])), ("q", (tab.b_quad[s:s+1], tab.w_quad[s:s+1]))):
        lam = 1 - K.shaped_one_center_damp(bp * r, wp, 2)
        D = (q * K._tensor(dr, lam)[:, 0, :]) @ P          # (1, 8): [-E (3), G (5)]
        out[par] = D[0]
    Ld, Lq = out["d"][:3], out["q"][3:]
    e_d = -0.5 * float(tab.alpha[s]) * float((Ld ** 2).sum())
    e_q = -0.5 * float(tab.C[s]) * float((Lq ** 2).sum())
    mu = float(tab.alpha[s]) * float(Ld.norm())
    # undamped (point) values for reference
    Rb = R_ang / BOHR
    e_d0 = -0.5 * float(tab.alpha[s]) * (q / Rb ** 2) ** 2
    return e_d, e_q, mu, e_d0
rows = []
for R in (2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0):
    env = PR.Environment(charge_coords=np.array([[0.0, 0.0, R / BOHR]]), charges=np.array([q]))
    r = PR.polarize(placed, env, mf0=mf0)
    e_d, e_q, mu, e_d0 = classical(R, q)
    row = dict(R=R, e_scf=r["e_pol"] * KJ, e_lin=r["e_pol_lin"] * KJ, mu_scf=abs(r["dmu"][2]), mu_lin=abs(r["dmu_lin"][2]),
               e_cls=(e_d + e_q) * KJ, e_cls_d=e_d * KJ, e_cls_q=e_q * KJ, mu_cls=mu, e_point_d=e_d0 * KJ)
    rows.append(row)
    print(f"R {R:4.2f}  E_pol scf {row['e_scf']:9.3f} lin {row['e_lin']:9.3f} | classical {row['e_cls']:9.3f} (dip {row['e_cls_d']:8.3f} quad {row['e_cls_q']:8.3f}; point-dip {row['e_point_d']:9.3f}) | mu scf {row['mu_scf']:.4f} lin {row['mu_lin']:.4f} cls {row['mu_cls']:.4f}", flush=True)
json.dump(rows, open(f"/home/claude/a1/shellcheck/{ion}_q{q:+.1f}.json", "w"), indent=1)
