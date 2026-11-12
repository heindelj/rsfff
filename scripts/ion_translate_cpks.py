"""CPKS split of the induced dipole between the permanent shells, for the shell-translation exchange
(rsfff.ff.complete.contract.TranslateTable; theory/parameter_free_xpol.tex).

Each shell t translates rigidly (electron moment n_t delta_t). The 2 x 2 compliance A of the two shells' translations is
fitted to the free ion's cached CPKS l = 1 response (uniform field + shell probes at R >= 1.7 A): Pi_kl ~ D_k^T A D_l,
D_k,t = <-d_z (n_t s_t) | v_k>. The free-field response then splits the moment as f_t = n_t (A n)_t / n^T A n
(sum f = 1; the anions come out anti-correlated: the compact shell shifts against the field). One-shell ions: f = (1, 0).

The same for l = 2 and the volume-preserving shell strain (rsfff.ff.complete.gram.StrainTable): the first-order strained
density n_t r s_t' P_2, D_k,t = n_t (4 pi / 5) int r^3 s_t' f_k dr against the r^2 and r_<^2 / r_>^3 probes, and the
split of the induced second moment f2_t = D_t (A D)_t / D^T A D with D_t = D_power,t (written as "f2", "A2").

    python scripts/ion_translate_cpks.py PARAMS.json OUT.json [--db data/elst_db]      (run from research/rsfff)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from rsfff.ff.ionct.params import IonTable

BOHR = 0.529177210903
DBNAME = {"F": "Fm", "Cl": "Clm", "Br": "Brm", "I": "Im", "Li": "Lip", "Na": "Nap", "K": "Kp", "Rb": "Rbp", "Cs": "Csp"}


def fit_ion(shells, rec, r_min_ang=1.69, lq=1):
    lab = [tuple(x) for x in rec["labels"]]
    idx = [k for k, (l, m, kd, p) in enumerate(lab) if l == lq and (kd == "power" or p * BOHR >= r_min_ang)]
    P = rec["Pi"][np.ix_(idx, idx)]
    r = np.linspace(0, 60, 240001)[1:]
    dr = r[1] - r[0]
    s1 = lambda b: b ** 3 / (8 * np.pi) * np.exp(-b * r)
    if lq == 1:
        frad = lambda kd, p: r if kd == "power" else np.minimum(r, p) / np.maximum(r, p) ** 2
        D = np.array([[n * b * (4 * np.pi / 3) * np.sum(s1(b) * frad(lab[k][2], lab[k][3]) * r * r) * dr
                       for n, b in shells] for k in idx])
    else:
        frad = lambda kd, p: r * r if kd == "power" else np.minimum(r, p) ** 2 / np.maximum(r, p) ** 3
        D = np.array([[-n * b * (4 * np.pi / 5) * np.sum(s1(b) * frad(lab[k][2], lab[k][3]) * r ** 3) * dr
                       for n, b in shells] for k in idx])
    d = np.sqrt(np.diag(P))
    S = np.outer(d, d)

    def A_of(x):
        L = np.zeros((2, 2))
        L[np.tril_indices(2)] = x
        return L @ L.T
    res = min((least_squares(lambda x: ((D @ A_of(x) @ D.T - P) / S).ravel(),
                             np.array([math.sqrt(s0), c, math.sqrt(s0)]))
               for s0 in np.logspace(-2, 1, 7) for c in (-0.5, 0.0, 0.5)), key=lambda q: q.cost)
    A = A_of(res.x)
    n = np.array([s[0] for s in shells]) if lq == 1 else D[0]          # the uniform-field (power) probe's moments
    f = n * (A @ n) / (n @ A @ n)
    R = (D @ A @ D.T - P) / S
    sfx = "" if lq == 1 else "2"
    return {"A" + sfx: A.tolist(), "f" + sfx: f.tolist(), "alpha_fit" + sfx: float(n @ A @ n),
            "alpha_cpks" + sfx: float(P[0, 0]), "rel_rms" + sfx: float(np.sqrt(np.mean(R ** 2))),
            "n_probes" + sfx: len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("out")
    ap.add_argument("--db", default="data/elst_db")
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    out = {"source": "scripts/ion_translate_cpks.py: CPKS l = 1, 2, uniform field + shells R >= 1.7 A", "params": a.params,
           "ions": {}}
    for s, sym in enumerate(tab.symbols):
        shells = [(float(tab.shell_n[s, t]), float(tab.shell_b[s, t])) for t in range(tab.shell_n.shape[1])
                  if float(tab.shell_n[s, t]) > 0]
        if len(shells) < 2:
            out["ions"][sym] = {"f": [1.0, 0.0], "f2": [1.0, 0.0]}
        else:
            z = np.load(Path(a.db) / "probe_multipole" / f"{DBNAME[sym]}.npz", allow_pickle=True)
            rec = {"labels": z["labels"].tolist(), "Pi": z["Pi"]}
            out["ions"][sym] = {**fit_ion(shells[:2], rec), **fit_ion(shells[:2], rec, lq=2)}
        print(sym, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v)
                               for k, v in out["ions"][sym].items()}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
