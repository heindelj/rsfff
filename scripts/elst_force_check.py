"""A2 (complete model plan): electrostatic forces of the frozen-density reference along the NaCl pair scans.

Three comparisons per pair type (Na-Cl, Cl-Cl, Na-Na), kJ/mol/A:
  F_ref   -dE_ref/dR by central differences of the pyscf frozen energy with a fine step (h = 1e-3 A);
  F_rigid rigid_force_torque (elst_ref; z-component on monomer B);
  F_eda   -d cls_elec/dR from a cubic spline through the Q-Chem scan points (FD error ~ h^2 f''' noted).
Also the splined pyscf curve differentiated the same way (F_ref_spl), so the spline error is separable
from the density difference.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from rsfff.qcgen import elst_ref as E

KJ = 2625.4996
ANG = 1.0 / 0.529177210903
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("db", help="elst_db root")
ap.add_argument("scan", help="ct_model ct-scan.jsonl")
ap.add_argument("--out", default="elst_force_check.json")
a = ap.parse_args()
db = Path(a.db) / "monomers"
mono = {"Na": E.Monomer.load(db / "Nap.npz"), "Cl": E.Monomer.load(db / "Clm.npz")}

rows = [json.loads(l) for l in open(a.scan)]
rows = [r for r in rows if r.get("calc") == "eda2" and r["tags"].get("scan") == "pair"]


def energy(a, b, R):
    A = mono[a].place(shift=np.zeros(3) - mono[a].coords[0])
    B = mono[b].place(shift=np.array([0, 0, R * ANG]) - mono[b].coords[0])
    return E.interaction_energy([A, B]) * KJ, (A, B)


out = {}
for pair in ("Na-Cl", "Cl-Cl", "Na-Na"):
    a, b = pair.split("-")
    sel = sorted([r for r in rows if r["tags"]["scan_pair"] == pair], key=lambda r: r["tags"]["scan_r"])
    R = np.array([r["tags"]["scan_r"] for r in sel])
    eda = np.array([r["eda"]["cls_elec"] for r in sel])
    e_ref = np.array([energy(a, b, x)[0] for x in R])
    h = 1e-3
    f_ref = np.array([-(energy(a, b, x + h)[0] - energy(a, b, x - h)[0]) / (2 * h) for x in R])
    f_rigid = []
    for x in R:
        _, (A, B) = energy(a, b, x)
        F, _T = E.rigid_force_torque([A, B], 1)
        f_rigid.append(F[2] * KJ * ANG)  # Hartree/Bohr -> kJ/mol/A on B along +z: -dE/dR
    f_rigid = np.array(f_rigid)
    f_eda = -CubicSpline(R, eda)(R, 1)
    f_ref_spl = -CubicSpline(R, e_ref)(R, 1)
    out[pair] = dict(R=R.tolist(), e_ref=e_ref.tolist(), eda=eda.tolist(), f_ref=f_ref.tolist(),
                     f_rigid=f_rigid.tolist(), f_eda=f_eda.tolist(), f_ref_spl=f_ref_spl.tolist())
    print(f"\n{pair}:  R  |  E_ref-cls_elec  |  F_ref  F_rigid  F_eda(spline)  F_ref(spline)   [kJ/mol, kJ/mol/A]")
    for i in range(len(R)):
        print(f"{R[i]:5.2f} {e_ref[i]-eda[i]:9.4f} | {f_ref[i]:10.3f} {f_rigid[i]:10.3f} {f_eda[i]:10.3f} {f_ref_spl[i]:10.3f}")
    inner = R <= 3.6
    print(f"  max|F_ref-F_rigid| = {np.abs(f_ref-f_rigid).max():.2e}; "
          f"rms(F_ref_spl - F_eda) = {np.sqrt(np.mean((f_ref_spl-f_eda)**2)):.4f} (all), "
          f"{np.sqrt(np.mean((f_ref_spl-f_eda)[inner]**2)):.4f} (R<=3.6); "
          f"spline-vs-fine FD rms (pyscf curve) = {np.sqrt(np.mean((f_ref_spl-f_ref)[1:-1]**2)):.4f}")
json.dump(out, open(a.out, "w"))
