"""CPKS stiffness of the shell contraction modes (rsfff.ff.complete.contract; theory/parameter_free_xpol.tex, step 1).

For every breathing ion (a0 > 0 in the ion table) the two l = 0 modes m_t of :func:`contraction_modes` and the cached
CPKS response of the free ion to the l = 0 shell probes at R >= 1.7 A (the m34 recipe's probe set,
qcgen.probe_multipole.monomer_multipole_response): Pi_kl ~ D_k^T A D_l with D_k,t = <m_t | v_k>, A = H0^-1 positive
definite (Cholesky), least squares on the scale-free residuals (Pi - model)_kl / sqrt(Pi_kk Pi_ll).

    python scripts/ion_contract_cpks.py PARAMS.json OUT.json [--db data/elst_db]      (run from research/rsfff)

Writes {"ions": {symbol: {"A": 2x2, "rel_rms", "rel_rms_diag", "eigA"}}, ...}; non-breathing ions are left out.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from rsfff.ff.complete.contract import breathing_species, contraction_modes
from rsfff.ff.ionct.params import IonTable

BOHR = 0.529177210903
DBNAME = {"F": "Fm", "Cl": "Clm", "Br": "Brm", "I": "Im", "Li": "Lip", "Na": "Nap", "K": "Kp", "Rb": "Rbp", "Cs": "Csp"}


def fit_ion(tab, s, rec, r_min_ang=1.69):
    lab = [tuple(x) for x in rec["labels"]]
    idx = [k for k, (l, m, kd, p) in enumerate(lab) if l == 0 and kd == "shell" and p * BOHR >= r_min_ang]
    P = rec["Pi"][np.ix_(idx, idx)]
    d = np.sqrt(np.clip(np.diag(P), 1e-300, None))
    S = np.outer(d, d)
    r = np.linspace(0, 60.0, 120001)[1:]
    dr = r[1] - r[0]
    modes = contraction_modes(tab, s, r)
    D = np.array([[np.sum(4 * np.pi * r * r * m / np.maximum(r, lab[k][3])) * dr for m in modes] for k in idx])

    def A_of(x):
        L = np.zeros((2, 2))
        L[np.tril_indices(2)] = x
        return L @ L.T

    def resid(x):
        return ((D @ A_of(x) @ D.T - P) / S).ravel()

    best = None
    for s0 in np.logspace(-3, 2, 11):
        for c in (-0.5, 0.0, 0.5):
            x0 = np.array([math.sqrt(s0), c * math.sqrt(s0), math.sqrt(s0)])
            res = least_squares(resid, x0)
            if best is None or res.cost < best.cost:
                best = res
    A = A_of(best.x)
    R = resid(best.x).reshape(P.shape)
    return {"A": A.tolist(), "rel_rms": float(np.sqrt(np.mean(R ** 2))),
            "rel_rms_diag": float(np.sqrt(np.mean(np.diag(R) ** 2))), "eigA": np.linalg.eigvalsh(A).tolist(),
            "n_probes": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("out")
    ap.add_argument("--db", default="data/elst_db")
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    out = {"source": "scripts/ion_contract_cpks.py: CPKS l = 0 shell probes, R >= 1.7 A", "params": a.params, "ions": {}}
    br = breathing_species(tab)
    for s, sym in enumerate(tab.symbols):
        if not br[s]:
            continue
        z = np.load(Path(a.db) / "probe_multipole" / f"{DBNAME[sym]}.npz", allow_pickle=True)
        out["ions"][sym] = fit_ion(tab, s, {"labels": z["labels"].tolist(), "Pi": z["Pi"]})
        print(sym, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in out["ions"][sym].items()}),
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
