"""Channel analysis of the Cl- polarization in Na_n shells.

Exact at linear order (spherical chi, per-site additive drives): E(config) = sum_l G_l(config) e_l(R), with
G_l = sum_jk P_l(cos theta_jk) the coherence factor of channel l and e_l(R) the single-site channel energy.
Fitting e_l(R) to the nine shapes at each R gives an l-resolved single-site energy for ANY reference, including
ALMO (where the drive is Coulomb + Pauli) -- the fit residual measures the nonlinearity / non-additivity.
The Na+ share is taken from the frozen-cation SCF-MI where available (ALMO - full), else the FD CPKS estimate.
Usage: ion_cage_channels.py RESPONSE.json [SCFMI.json] [--lmax 3]"""
import sys
import json
import numpy as np
from scipy.special import eval_legendre

from ion_cage_response import shell

ALMO = {  # Q-Chem ALMO-EDA wB97M-V/def2-TZVPD: (pol, mod_pauli) kJ/mol
    ("pair", 2.6): (-46.7605, 61.4515), ("pair", 2.8): (-41.5297, 34.8896), ("pair", 3.2): (-31.2614, 11.8591),
    ("a90", 2.6): (-89.4416, 121.0268), ("a90", 2.8): (-77.6608, 68.6634), ("a90", 3.2): (-56.9033, 23.3201),
    ("a109", 2.6): (-64.7332, 119.5196), ("a109", 2.8): (-55.582, 67.8102), ("a109", 3.2): (-40.1898, 23.0435),
    ("a120", 2.6): (-53.0627, 118.8875), ("a120", 2.8): (-45.2009, 67.4531), ("a120", 3.2): (-32.3952, 22.9311),
    ("a180", 2.6): (-19.6921, 117.1046), ("a180", 2.8): (-15.0643, 66.8618), ("a180", 3.2): (-9.365, 22.6141),
    ("tri3", 2.6): (-26.723, 172.2747), ("tri3", 2.8): (-19.49, 98.0389), ("tri3", 3.2): (-10.8626, 33.2225),
    ("sq4", 2.6): (-38.908, 226.6269), ("sq4", 2.8): (-26.1516, 128.7499), ("sq4", 3.2): (-11.9706, 43.6425),
    ("tet4", 2.6): (-34.104, 225.4167), ("tet4", 2.8): (-23.1753, 128.145), ("tet4", 3.2): (-10.8784, 43.4272),
    ("oct6", 2.6): (-63.4068, 328.5431), ("oct6", 2.8): (-39.6951, 186.2739), ("oct6", 3.2): (-14.5499, 63.0843),
}
SHAPES = ["pair", "a90", "a109", "a120", "a180", "tri3", "sq4", "tet4", "oct6"]
LFIT = 4  # l = 0..3 (l = 4 is ~0 in every CPKS reference; fitting it only absorbs noise)


def G(shape, l):
    u = shell(shape, 1.0)
    return float(eval_legendre(l, np.clip(u @ u.T, -1, 1)).sum())


def fit(values, R):
    A = np.array([[G(s, l) for l in range(LFIT)] for s in SHAPES])
    y = np.array([values[(s, R)] for s in SHAPES])
    e, *_ = np.linalg.lstsq(A, y, rcond=None)
    return e, y - A @ e


if __name__ == "__main__":
    if "--lmax" in sys.argv:
        LFIT = int(sys.argv[sys.argv.index("--lmax") + 1]) + 1
    files = [x for x in sys.argv[1:] if x.endswith(".json")]
    recs = {(r["shape"], r["R"]): r for r in json.load(open(files[0]))}
    full = {}
    if len(files) > 1:
        for r in json.load(open(files[1])):
            if "full" in r:
                full[(r["shape"], r["R"])] = r["full"]["pol"]
    print("coherence factors G_l:")
    for s in SHAPES:
        print(f"  {s:5s} " + " ".join(f"{G(s, l):7.3f}" for l in range(LFIT)))
    out = {}
    for R in (2.6, 2.8, 3.2):
        print(f"\n===== R = {R} A")
        na = {s: (ALMO[(s, R)][0] - full[(s, R)]) if (s, R) in full else recs[(s, R)]["FD"]["na_lin_total"] for s in SHAPES}
        almo_cl = {(s, R): ALMO[(s, R)][0] - na[s] for s in SHAPES}
        print("shape  ALMO  [full]  Na+ share | Cl- : ALMO-Na  FD scf  FD lin  PC scf  PC lin | Pauli eff = ALMO_Cl - FD scf")
        for s in SHAPES:
            r = recs[(s, R)]
            f = full.get((s, R))
            print(f"{s:5s} {ALMO[(s, R)][0]:7.1f} {f if f is None else round(f, 1)!s:>7s} {na[s]:6.1f} | {almo_cl[(s, R)]:7.1f} "
                  f"{r['FD']['e_scf']:7.1f} {r['FD']['e_lin']:7.1f} {r['PC']['e_scf']:7.1f} {r['PC']['e_lin']:7.1f} | "
                  f"{almo_cl[(s, R)] - r['FD']['e_scf']:7.1f}")
        rows = {}
        for name, vals in (("PC lin", {(s, R): recs[(s, R)]["PC"]["e_lin"] for s in SHAPES}),
                           ("PC scf", {(s, R): recs[(s, R)]["PC"]["e_scf"] for s in SHAPES}),
                           ("FD lin", {(s, R): recs[(s, R)]["FD"]["e_lin"] for s in SHAPES}),
                           ("FD scf", {(s, R): recs[(s, R)]["FD"]["e_scf"] for s in SHAPES}),
                           ("ALMO-Na", almo_cl)):
            e, res = fit(vals, R)
            rows[name] = (e, res)
        pair_lin = {k: recs[("pair", R)][k]["El_lin"][:LFIT] for k in ("PC", "FD")}
        print("single-site channel energies e_l (kJ/mol)      l0      l1      l2      l3      l4 | fit rms  max|res|")
        for k in ("PC", "FD"):
            print(f"  {k} pair CPKS (exact)              " + " ".join(f"{x:7.2f}" for x in pair_lin[k]))
        for name, (e, res) in rows.items():
            print(f"  {name:8s} fitted over 9 shapes        " + " ".join(f"{x:7.2f}" for x in e)
                  + f" | {np.sqrt(np.mean(res ** 2)):6.2f} {np.abs(res).max():6.2f}")
        e_a = rows["ALMO-Na"][0]
        print("  ALMO/FD-lin ratio per l           " + " ".join(f"{a / b:7.2f}" if abs(b) > 0.05 else "     --" for a, b in zip(e_a, pair_lin["FD"])))
        print("  ALMO/PC-lin ratio per l           " + " ".join(f"{a / b:7.2f}" if abs(b) > 0.05 else "     --" for a, b in zip(e_a, pair_lin["PC"])))
        out[R] = {n: {"e_l": e.tolist(), "res": res.tolist()} for n, (e, res) in rows.items()}
        out[R]["pair_lin"] = pair_lin
    json.dump({str(k): v for k, v in out.items()}, open(files[0].replace(".json", "_channels.json"), "w"), indent=1)
