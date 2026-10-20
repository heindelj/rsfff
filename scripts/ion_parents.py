"""Response parents on the permanent shell exponents (complete model plan, 2026-10-20).

Per ion: the CPKS response to a +1 point charge at R (pyscf, same functional / basis as the elst_db monomer), the
energy decomposed by angular momentum E_l = 1/2 int drho phi_l, and then

    l = 1  ->  dipole parent:     weights over r^0..r^3 on each permanent shell exponent (alpha from the table)
    l = 2  ->  quadrupole parent: same (C from the table)
    l = 0  ->  breathing shape:   one term, free exponent, and its polarizability a0:  E_0 = -1/2 a0 (f_0(R) / R)^2
               (f_0 the potential complement of the shape at the charge -- what a penetrating charge sees)

written as ions[s]["parents"] = {"dip": [{b, w}], "quad": [...], "breath": [{b, w}], "a0"} into a copy of the
parameter file (IonTable reads multi-term parents; the legacy b_ind / w_dip stay as the T = 1 fallback).

    ion_parents.py --params params/ions_wb97mv_tzvpd_pf2.json --out params/ions_wb97mv_tzvpd_pf2p.json [--ions Na Cl]
                   [--scan-dir DIR]     # reuse <ion>_resp_scan.json there instead of running pyscf
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import eval_legendre

from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.multipole import shaped_one_center_damp

BOHR = K.BOHR_ANG
KJ = 2625.4996
DB = os.environ.get("ELST_DB", os.path.expanduser("~/research/rsfff/data/elst_db"))
DBNAME = {"Na": "Nap", "Cl": "Clm", "Li": "Lip", "K": "Kp", "F": "Fm", "Br": "Brm"}
RS = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0, 7.0, 8.0]


def response_scan(ion, out_json):
    out = json.load(open(out_json)) if Path(out_json).exists() else []
    if all(any(abs(o["R"] - R) < 1e-6 for o in out) for R in RS):
        return out
    from pyscf import dft
    from rsfff.qcgen import elst_ref as E
    from rsfff.qcgen import pol_ref as PR
    placed = E.Monomer.load(f"{DB}/monomers/{DBNAME[ion]}.npz").place()
    mol = placed.mol
    mf0 = PR.isolated(placed)
    g = dft.gen_grid.Grids(mol)
    g.level = 4
    g.build()
    ao = dft.numint.eval_ao(mol, g.coords)
    r = np.linalg.norm(g.coords, axis=1)
    cos = g.coords[:, 2] / np.maximum(r, 1e-12)
    for R in RS:
        if any(abs(o["R"] - R) < 1e-6 for o in out):
            continue
        Rb = R / BOHR
        env = PR.Environment(charge_coords=np.array([[0.0, 0.0, Rb]]), charges=np.array([1.0]))
        res = PR.polarize(placed, env, mf0=mf0)
        rho = -dft.numint.eval_rho(mol, ao, res["ddm_lin"])
        rl, rg = np.minimum(r, Rb), np.maximum(r, Rb)
        El = [0.5 * float(np.sum(g.weights * rho * rl ** l / rg ** (l + 1) * eval_legendre(l, cos))) for l in range(5)]
        out.append({"R": R, "E_l": El, "e_lin": res["e_pol_lin"], "e_scf": res["e_pol"]})
        out.sort(key=lambda o: o["R"])
        json.dump(out, open(out_json, "w"), indent=1)
        print(f"  {ion} R {R:4.1f}: " + " ".join(f"l{l} {e * KJ:8.3f}" for l, e in enumerate(El)), flush=True)
    return out


def _damp(b_terms, w_terms, R, rank):
    f = 0
    for b, w in zip(b_terms, w_terms):
        f = f + shaped_one_center_damp(b * R, w[None].expand(R.shape[0], 4), rank)
    return f


def _energies(b_terms, w_terms, R, alpha, C):
    lam = 1 - _damp(b_terms, w_terms, R, 2)
    dr = torch.stack([torch.zeros_like(R), torch.zeros_like(R), R], 1)
    D = K._tensor(dr, lam)[:, 0, :] @ K._poly_map(torch.float64, "cpu")
    return -0.5 * alpha * (D[:, :3] ** 2).sum(1), -0.5 * C * (D[:, 3:] ** 2).sum(1)


def fit_ion(ion, tab, scan, r_min=2.0):
    s = tab.symbols.index(ion)
    R = torch.tensor([d["R"] for d in scan], dtype=torch.float64) / BOHR
    El = torch.tensor([d["E_l"] for d in scan], dtype=torch.float64)
    bsh = [float(b) for b, n in zip(tab.shell_b[s], tab.shell_n[s]) if n > 0]
    alpha, C = float(tab.alpha[s]), float(tab.C[s])
    sel = (R * BOHR >= r_min).numpy()
    rel = lambda e, ref: float(np.sqrt(np.mean(((e - ref) / ref).numpy()[sel] ** 2)))
    out, report = {}, {}
    for l, key in ((1, "dip"), (2, "quad")):
        def resid(z):
            w = torch.softmax(torch.tensor(z), 0).reshape(len(bsh), 4)
            e = _energies(bsh, list(w), R, alpha, C)[l - 1]
            return (torch.log(-e) - torch.log(-El[:, l])).numpy()[sel]
        best = None
        for k in range(4 * len(bsh)):
            z0 = np.zeros(4 * len(bsh))
            z0[k] = 3.0
            res = least_squares(resid, z0)
            if best is None or res.cost < best.cost:
                best = res
        w = torch.softmax(torch.tensor(best.x), 0).reshape(len(bsh), 4)
        e = _energies(bsh, list(w), R, alpha, C)[l - 1]
        out[key] = [{"b": b, "w": [round(x, 6) for x in row]} for b, row in zip(bsh, w.tolist()) if max(row) > 1e-6]
        e_old = _energies(list(getattr(tab, f"b_{key}")[s]), list(getattr(tab, f"w_{key}")[s]), R, alpha, C)[l - 1]
        report[key] = {"rel_rms": rel(e, El[:, l]), "rel_rms_table": rel(e_old, El[:, l])}
    E0 = El[:, 0]
    sel0 = sel & (E0.abs().numpy() * KJ > 0.002)
    if sel0.sum() >= 4:
        def model0(p):
            w = torch.softmax(torch.tensor(p[2:]), 0)[None].expand(len(R), 4)
            f = shaped_one_center_damp(np.exp(p[1]) * R, w, 0)[0]
            return -0.5 * np.exp(p[0]) * (f / R) ** 2
        best = None
        for b0 in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
            for k in range(4):
                z = [0.0] * 4
                z[k] = 3.0
                res = least_squares(lambda p: (torch.log(-model0(p)[sel0]) - torch.log(-E0[sel0])).numpy(),
                                    x0=[np.log(20.0), np.log(b0)] + z)
                if best is None or res.cost < best.cost:
                    best = res
        p = best.x
        out["breath"] = [{"b": float(np.exp(p[1])), "w": [round(x, 6) for x in torch.softmax(torch.tensor(p[2:]), 0).tolist()]}]
        out["a0"] = float(np.exp(p[0]))
        report["breath"] = {"a0": out["a0"], "b": out["breath"][0]["b"],
                            "rel_rms": float(np.sqrt(np.mean((((model0(p) - E0) / E0).numpy()[sel0]) ** 2)))}
    else:
        out["a0"] = 0.0
        report["breath"] = "negligible"
    out["source"] = "ion_parents.py: CPKS point-charge response by l, R >= %.1f A" % r_min
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ions", nargs="*", default=None)
    ap.add_argument("--scan-dir", default=".")
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    doc = json.load(open(a.params))
    for ion in a.ions or tab.symbols:
        scan = response_scan(ion, Path(a.scan_dir) / f"{ion}_resp_scan.json")
        parents, report = fit_ion(ion, tab, scan)
        doc["ions"][ion]["parents"] = parents
        print(ion, json.dumps(report), flush=True)
    doc["parents_note"] = "parents: per-shell response shapes from the CPKS point-charge response (scripts/ion_parents.py)"
    json.dump(doc, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
