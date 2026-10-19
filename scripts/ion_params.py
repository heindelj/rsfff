"""Fixed physical parameters of the monatomic ions for the ion CT force field (ion ct ff plan, R1).

    python scripts/ion_params.py DB [--out params/ions_wb97mv_tzvpd.json] [--shapes DB/shapes/ion_shells.json]

DB is an ``rsfff.qcgen.elst_db`` database (wB97M-V/def2-TZVPD) with the ions' inner grids and their uniform field and
field-gradient responses (``scripts/pol_db.py field DB --names Nap Clm --inner --field-gradients``).

Per ion, everything is fitted once to monomer references and then frozen:

* permanent density: point nucleus of charge Z plus Slater 1s shells carrying Z - q electrons (``elst_fit``; Z fixed,
  Na+ 9 with one shell, Cl- 11 with two shells);
* induced response (``joint_fit``: permanent and induced fitted together with the induced exponent tied to the
  outermost permanent shell; Cl- with a free ratio): the free-ion dipole polarizability alpha (exact), the quadrupole
  polarizability C (fixed to the reference field-gradient response), and the shapes of the induced dipole and
  quadrupole densities (exponent b_ind, weights over r^0..r^3);
* frontier densities: the Fukui functions (f- for an anion donor, f+ for a cation acceptor) as one r^3 e^{-br} shape
  (the closest power the shaped kernels carry; ``shapes/ion_shells.json``), and for the other role the outermost
  permanent shell;
* the frontier energies of the CT channels: I_donor, the free ion's electron-removal energy (Cl-: EA(Cl), UKS
  Delta-SCF; Na+: the second ionisation energy of Na, experimental), and A_acceptor, the electron-attachment
  energy (Na+: IP(Na), UKS Delta-SCF; Cl-: none, Cl2- is unbound -- channels into Cl- take their intrinsic gap
  from the pair scans).

Units: atomic units (bohr, e, Hartree); alpha bohr^3, C in the pol_fit convention (Theta = C g).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rsfff.qcgen import elst_db as Db
from rsfff.qcgen import elst_fit as EF
from rsfff.qcgen import joint_fit as J
from rsfff.qcgen import pol_fit as PF
from rsfff.qcgen import pol_ref as R

EV = 27.211386
#: name in the database, element symbol, nuclear charge Z (fixed), permanent shells, free b_ind/b_perm ratio
IONS = {"Na": ("Nap", 11, 9.0, 1, False), "Cl": ("Clm", 17, 11.0, 2, True)}
#: the frontier (Fukui) shape used for the role the Fukui function describes
FUKUI_ROLE = {"Na": "acceptor", "Cl": "donor"}
#: second ionisation energy of Na (NIST, eV): Na+ as a donor
IP2_EV = {"Na": 47.28636}
FUKUI_K = 3


def fit_ion(db, sym):
    name, z, zv, n_shells, free = IONS[sym]
    rec = R.field_response(db, name, field_gradients=True)
    m = db.get(name)
    grid = Db.monomer_grid(db, name, scales=EF.INNER_SCALES, cache="grids_inner")
    perm = EF.FitConfig(max_rank=0, n_shells=n_shells, z={z: zv}, b_min=1.2, taper=(5.0, 6.0))
    pol = PF.PolConfig(quadrupole=True, n_shape=3, alpha="exact", c_fixed={z: PF.reference_c(rec)})
    cfg = J.JointConfig(perm=perm, pol=pol, perm_min_scale=0.5, free_ratio=free)
    jf, met = J.fit_joint(grid, rec, [z], db.index[name]["charge"], m.coords, cfg)
    sp = jf.sync()
    b, sw = (x.detach().numpy()[0] for x in sp.shells())
    q, Z = float(sp.q.detach()[0]), float(sp.Z.detach()[0])
    pol_fit = jf.pol

    def weights(w):      # drop shape powers the fit switched off (softmax tails), so the kernels skip them
        w = np.where(w < 1e-6, 0.0, w)
        return (w / w.sum()).tolist()

    Pi = pol_fit.blocks().detach().numpy()[0]
    out = {
        "symbol": sym, "db_name": name, "charge": q, "Z": Z,
        "shells": [{"n": float((Z - q) * w), "b": float(bb)} for bb, w in sorted(zip(b, sw), key=lambda t: -t[0])],
        "alpha": float(np.trace(Pi[:3, :3]) / 3), "alpha_ref": float(np.trace(rec["alpha"]) / 3),
        "C": float(np.trace(Pi[3:, 3:]) / 5) if Pi.shape[0] > 3 else None,
        "b_ind": float(pol_fit.b().detach()[0]), "w_dip": weights(pol_fit.w().detach().numpy()[0]),
        "b_quad": float(pol_fit.b_quad().detach()[0]), "w_quad": weights(pol_fit.w_quad().detach().numpy()[0]),
        "ratio": float(met["joint"]["ratio"][jf.elements.index(z)]),
        "fit": {"perm_phi_rel": {s: v["phi_rms"] / v["phi_ref_rms"] for s, v in met["joint"]["perm"].items()},
                "resp_F": met["joint"]["resp"]["scales"], "resp_G": met["joint"]["resp"].get("scales_G")},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--out", default="params/ions_wb97mv_tzvpd.json")
    ap.add_argument("--shapes", default=None, help="ion_shells.json (default DB/shapes/ion_shells.json)")
    ap.add_argument("--ions", nargs="*", default=list(IONS))
    a = ap.parse_args()
    db = Db.MonomerDB(a.db)
    shapes = json.load(open(a.shapes or Path(a.db) / "shapes" / "ion_shells.json"))["fukui"]
    out_path = Path(a.out)
    table = json.load(open(out_path)) if out_path.exists() else {}
    table.setdefault("level", "wB97M-V/def2-TZVPD")
    table.setdefault("units", "atomic (bohr, e, Hartree); alpha bohr^3; C: Theta = C g, g_k = (1/3) B_k : G")
    ions = table.setdefault("ions", {})
    t0 = time.time()
    for sym in a.ions:
        p = fit_ion(db, sym)
        fk = shapes[IONS[sym][0]]
        outer = min(s["b"] for s in p["shells"])
        frontier = {"donor": {"k": 0, "b": outer}, "acceptor": {"k": 0, "b": outer}}
        frontier[FUKUI_ROLE[sym]] = {"k": FUKUI_K, "b": float(fk["fits"][str(FUKUI_K)][0]),
                                     "fit_err": float(fk["fits"][str(FUKUI_K)][1]), "source": "Fukui"}
        p["frontier"] = frontier
        if FUKUI_ROLE[sym] == "donor":            # X-: removing the frontier electron costs EA(X)
            p["I_donor"], p["A_acceptor"] = float(fk["I_eV"]) / EV, None
        else:                                     # M+: attaching one gains IP(M); removing one costs IP2(M)
            p["I_donor"], p["A_acceptor"] = IP2_EV[sym] / EV, float(fk["I_eV"]) / EV
        ions[sym] = p
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(out_path, "w"), indent=1)
        print(f"{sym}: Z {p['Z']} shells {[(round(s['n'], 3), round(s['b'], 3)) for s in p['shells']]} "
              f"alpha {p['alpha']:.3f} (ref {p['alpha_ref']:.3f}) C {p['C']} b_ind {p['b_ind']:.3f} "
              f"w {np.round(p['w_dip'], 3).tolist()} b_quad {p['b_quad']:.3f} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
