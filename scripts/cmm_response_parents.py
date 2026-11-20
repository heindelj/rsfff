"""Dipole response parents from the free ion's own linear-response density.

The model writes an induced dipole density as -mu d/dz g(r), g the normalized "parent". A finite-field SCF gives the
ion's first-order density d_rho(r) = E f(r) cos(theta), which fixes the parent, g'(r) = -f(r) / alpha: its shape is
not a free choice. This script fits the table's parent family to it,

    g(r) = sum_t sum_k w[t, k] N_k r^k e^{-b[t] r},      b[t] the table's exponents (unchanged), sum w = 1, w >= 0,

by least squares in the Coulomb metric (density fitting): minimize <d_rho - d_rho_model | 1/r12 | d_rho - d_rho_model>
(relative to <d_rho|1/r12|d_rho>), plus ``--outside-weight`` times the mean squared relative error of the response
potential's damping at 2-8 A (below). The Coulomb norm bounds the error of the response potential everywhere, inside
the ion included, and it is the inner product of the dipole shape's self-energy

    u1 = (4 pi / 3) <g|g> = 2 E_Hartree[d_rho] / mu^2,

whose product with alpha decides the Coulomb Gram certificate (alpha u1 < 1: the induction operator is positive
definite at any geometry) and the room of the saturating hyperpolarization (rsfff.ff.cmm, hyperpol = "saturating").
Physically 1 / alpha = k_s + u1 + u_xc (independent-particle stiffness, Hartree, exchange-correlation), so alpha u1 < 1
says the stiffness without the Hartree term is positive. For the hydrogen atom the exact parent,
(3 + 6 r + 2 r^2) e^{-2 r}, is in the family (weights 1/6, 1/2, 1/3 at the density's exponent).

The parents that scripts/ion_parents.py fits use the same response seen from outside (the energy of a point charge
at R >= 2 A), which leaves the inner shape open: Li+, Cs+ and F- end up with 80 % Coulomb error and Cs+ with
alpha u1 > 1. The Coulomb norm alone has the opposite bias: it is dominated by the inner density and at these fixed
exponents lets the damping at contact distances drift (2-7 % too weak at 2.0-2.6 A for Cs+, F-, Cl-, I-).
``outside`` is that measure, the rms relative error of phi(R) R^2 / alpha at ion_parents.py's distances >= 2 A; with
the default weight (100) both stay small: Coulomb error <= 4 %, outside error <= 1.6 %, alpha u1 within 0.06 of the
response's. ``--outside-weight 0`` is the pure density fit.

    python scripts/cmm_response_parents.py --params IN.json --out OUT.json [--ions Na Cl ...] [--xc wb97m-v]
                                           [--basis def2-tzvpd] [--outside-weight 100] [--report REPORT.json]

OUT.json is IN.json with ions[s]["parents"]["dip"] replaced (the previous one kept as "dip_old") and a top-level
"dipole_parents_note". Atomic units.
"""
import argparse
import json
import math

import numpy as np
from pyscf import dft, gto
from scipy.optimize import minimize
from scipy.special import gammainc

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "F": -1, "Cl": -1, "Br": -1, "I": -1}
ECP = {"Rb", "Cs", "I"}
FIELD = 0.002
BOHR = 0.529177210903
R_OUT = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0, 7.0, 8.0]       # A (ion_parents.py's, >= 2 A)


def response(sym, xc, basis):
    """(mol, grids, dD, alpha): the electron density-matrix response per unit field along z, by central differences."""
    D = []
    for field in (+FIELD, -FIELD):
        mol = gto.M(atom=f"{sym} 0 0 0", basis=basis, charge=CHARGE[sym], ecp=basis if sym in ECP else None, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = xc
        if xc.lower().endswith("-v"):
            mf.nlc = "vv10"
        mf.grids.level = 5
        mf.conv_tol = 1e-11
        h0, z = mf.get_hcore(), mol.intor("int1e_r")[2]
        mf.get_hcore = lambda *a, h0=h0, z=z, field=field: h0 + field * z
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{sym}: SCF not converged at field {field}")
        D.append(mf.make_rdm1())
    dD = (D[0] - D[1]) / (2 * FIELD)
    return mol, mf, dD, -float(np.einsum("ij,ji", dD, z))


def shape_ft(n, b, k):
    """Fourier transform of the normalized density N r^n e^{-b r}."""
    th, m = np.arctan2(k, b), n + 2
    return b ** (n + 3) * np.sin(m * th) / (m * k * (b * b + k * k) ** (m / 2))


def enclosed(n, b, r):
    """The charge of the normalized density N r^n e^{-b r} inside r (the damping of its dipole's potential)."""
    return gammainc(n + 3, b * r)


def gram(shapes, kmax=400.0, n=80000):
    """U[k, l] = (4 pi / 3) <g_k | g_l>: the Coulomb overlap of two dipole shapes per unit dipoles."""
    k = np.linspace(kmax / n, kmax, n)
    G = np.array([shape_ft(m, b, k) for b, m in shapes])
    return 2 / (3 * math.pi) * np.trapezoid(G[:, None, :] * G[None, :, :] * k * k, k, axis=2)


def fit_ion(sym, ion, xc, basis, lam=100.0):
    mol, mf, dD, alpha = response(sym, xc, basis)
    u1 = float(np.einsum("ij,ji", mf.get_j(mol, dD), dD)) / alpha ** 2
    terms = ion["parents"]["dip"]
    shapes = [(float(t["b"]), m) for t in terms for m in range(4)]
    w_old = np.array([(list(t["w"]) + [0.0] * 4)[:4][m] for t in terms for m in range(4)], float)
    U = gram(shapes)
    g = mf.grids
    drho = -dft.numint.eval_rho(mol, dft.numint.eval_ao(mol, g.coords), dD)          # charge density response
    r = np.linalg.norm(g.coords, axis=1).clip(1e-12)
    cz = g.coords[:, 2] / r
    v = np.array([float((g.weights * drho * cz * enclosed(m, b, r) / r ** 2).sum()) for b, m in shapes]) / alpha
    err = lambda w: float((u1 - 2 * w @ v + w @ U @ w) / u1)                                  # noqa: E731
    # the damping seen from outside: phi(R) R^2 / alpha of the response density at R on the field axis, and the model's
    Rs = np.array(R_OUT) / BOHR
    q_qm = []
    for R in Rs:
        with mol.with_rinv_origin((0.0, 0.0, float(R))):
            q_qm.append(-float(np.einsum("ij,ji", dD, mol.intor("int1e_rinv"))) * R * R / alpha)
    q_qm = np.array(q_qm)
    E = np.array([[enclosed(m, b, R) for b, m in shapes] for R in Rs])
    A = E / q_qm[:, None]
    out_err = lambda ww: float(np.sqrt(np.mean((A @ ww - 1) ** 2)))                           # noqa: E731
    fun = lambda w: (w @ U @ w - 2 * w @ v) / u1 + lam * float(np.mean((A @ w - 1) ** 2))     # noqa: E731
    jac = lambda w: (2 * U @ w - 2 * v) / u1 + lam * 2 * A.T @ (A @ w - 1) / len(q_qm)        # noqa: E731
    best = None
    for start in (w_old / max(w_old.sum(), 1e-12), np.full(len(shapes), 1.0 / len(shapes))):
        o = minimize(fun, start, jac=jac, bounds=[(0, 1)] * len(shapes),
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                     options={"ftol": 1e-16, "maxiter": 2000})
        if best is None or o.fun < best.fun:
            best = o
    w = np.where(best.x < 5e-7, 0.0, best.x)
    w = np.round(w / w.sum(), 6)
    w[int(np.argmax(w))] += 1.0 - float(w.sum())
    w = np.round(w, 6)                                                                      # six digits, sum exactly 1
    rep = {"alpha_response": alpha, "u1_response": u1, "alpha_u1_response": alpha * u1,
           "alpha_u1": float(alpha * w @ U @ w), "alpha_u1_old": float(alpha * w_old @ U @ w_old),
           "coulomb_error": err(w), "coulomb_error_old": err(w_old),
           "outside": out_err(w), "outside_old": out_err(w_old), "outside_weight": lam,
           "level": f"{xc}/{basis}, field {FIELD}"}
    new = [{"b": t["b"], "w": [float(x) for x in w[4 * i:4 * i + 4]]} for i, t in enumerate(terms)]
    rep["_arrays"] = {"U": U, "v": v, "u1": u1, "alpha": alpha, "E": E, "q_qm": q_qm, "w_old": w_old, "shapes": np.array(shapes)}
    return new, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ions", nargs="*")
    ap.add_argument("--xc", default="wb97m-v")
    ap.add_argument("--basis", default="def2-tzvpd")
    ap.add_argument("--outside-weight", type=float, default=100.0,
                    help="weight of the outside damping's mean squared relative error next to the Coulomb error "
                         "(0: density fit only)")
    ap.add_argument("--report")
    ap.add_argument("--dump", help="directory for the fit's arrays per ion (U, v, u1, the outside damping), <ion>_dipresp.npz")
    a = ap.parse_args()
    tab = json.load(open(a.params))
    report = {}
    for sym in (a.ions or list(tab["ions"])):
        ion = tab["ions"][sym]
        new, rep = fit_ion(sym, ion, a.xc, a.basis, a.outside_weight)
        arr = rep.pop("_arrays")
        if a.dump:
            np.savez(f"{a.dump}/{sym}_dipresp.npz", **arr)
        if abs(rep["alpha_response"] - ion["alpha"]) > 0.02 * ion["alpha"]:
            print(f"  note: {sym} finite-field alpha {rep['alpha_response']:.3f} against the table's {ion['alpha']:.3f}")
        pr = ion["parents"]
        pr.setdefault("dip_old", pr["dip"])
        pr["dip"] = new
        pr["dip_source"] = ("cmm_response_parents.py: Coulomb-metric fit of the finite-field response density "
                            f"({rep['level']}; outside weight {a.outside_weight:g}); alpha u1 {rep['alpha_u1']:.3f} "
                            f"(response {rep['alpha_u1_response']:.3f}, dip_old {rep['alpha_u1_old']:.3f}); Coulomb error "
                            f"{rep['coulomb_error']:.4f} (dip_old {rep['coulomb_error_old']:.4f}); outside 2-8 A rms "
                            f"{rep['outside']:.4f} (dip_old {rep['outside_old']:.4f})")
        report[sym] = rep
        print(f"{sym:2s} alpha u1 {rep['alpha_u1']:.3f} (response {rep['alpha_u1_response']:.3f}, old {rep['alpha_u1_old']:.3f}) | "
              f"Coulomb error {rep['coulomb_error']:.4f} (old {rep['coulomb_error_old']:.4f}) | outside 2-8 A rms "
              f"{rep['outside']:.4f} (old {rep['outside_old']:.4f}) | w {[t['w'] for t in new]}", flush=True)
        tab["dipole_parents_note"] = ("dipole parents: Coulomb-metric fit of each free ion's finite-field response density on "
                                      f"the previous exponents, outside weight {a.outside_weight:g} "
                                      "(scripts/cmm_response_parents.py); the point-charge-scan parents are kept as "
                                      "parents['dip_old']")
        json.dump(tab, open(a.out, "w"), indent=1)
        if a.report:
            json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
