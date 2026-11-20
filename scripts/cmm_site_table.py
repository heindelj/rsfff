"""The species table of the site-potential cmm (CmmConfig.pauli_form = "site") from a base table and three monomer /
pair-fit inputs:

  --shells   {ion: {Z, n[], b[]}} or a table: the permanent shells fitted to ln n(r) of the free ion's density on its
             valence window, populations free, Z_eff = q + their sum (non-integer; claude/qm_embedding_coulomb_shells.md)
  --response directory of es2_<ion>.json: the free ion's density n(r) and its response densities u_l(r) to
             r P1 and r^2 P2 (per unit strength; K0_l = alpha, C) on a radial grid
  --pauli    {params: {ion: {amp[], scale}}}: the site potentials fitted to the EDA's mod_pauli on those shells

It replaces shells and Z, refits the dipole and quadrupole parents on the new shell exponents, adds pauli_site, and
keeps the dispersion damping exponent the base table implied (outermost shell x the Pauli scale) as disp_range.
The parents are fitted in the amplitude metric, int (u - u_fit)^2 / n r^2 dr with n the model's shells, normalized
(sum of the weights = 1): the dipole's g' against u_1 / alpha, the quadrupole's -(g'' - g' / r) / 3 against u_2 / C.
Each parent's Coulomb certificate rho = alpha u1 (C u2) is reported and must stay below 1.

    python scripts/cmm_site_table.py --base BASE.json --shells SHELLS.json --response DIR --pauli PAULI.json --out OUT.json
"""
import argparse, json, math
from pathlib import Path
import numpy as np

N_SHAPE = 4


def _terms(shells, r):
    """g, g', g'' of every basis function N_k r^k e^{-b_t r} (t over the shells, k = 0..3): (T * 4, R) each."""
    g, g1, g2 = [], [], []
    for sh in shells:
        b = sh["b"]
        e = np.exp(-b * r)
        for k in range(N_SHAPE):
            N = b ** (k + 3) / (4 * math.pi * math.factorial(k + 2))
            rk = r ** k
            rk1 = k * r ** (k - 1) if k >= 1 else 0 * r
            rk2 = k * (k - 1) * r ** (k - 2) if k >= 2 else 0 * r
            g.append(N * rk * e); g1.append(N * (rk1 - b * rk) * e); g2.append(N * (rk2 - 2 * b * rk1 + b * b * rk) * e)
    return np.array(g), np.array(g1), np.array(g2)


def _constrained(B, t, wt):
    """min sum wt (c.B - t)^2 with sum c = 1."""
    A = (B * wt) @ B.T
    rhs = (B * wt) @ t
    n = len(A)
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n] = 1; M[n, :n] = 1
    c = np.linalg.lstsq(M, np.append(rhs, 1.0), rcond=1e-12)[0][:n]
    res = c @ B - t
    return c, math.sqrt(np.sum(wt * res * res) / np.sum(wt * t * t))


def _self_coulomb(r, u, l):
    """Coulomb self-energy of u(r) P_l at unit amplitude."""
    from scipy.integrate import cumulative_trapezoid as ct
    w = np.gradient(r)
    a = ct(u * r ** (l + 2), r, initial=0)
    b = -ct((u * r ** (1 - l))[::-1], r[::-1], initial=0)[::-1]
    phi = (4 * np.pi / (2 * l + 1)) * (a / r ** (l + 1) + b * r ** l)
    return (4 * np.pi / (2 * l + 1)) * np.sum(w * r * r * u * phi)


def refit_parents(shells, es):
    """({dip, quad} parent term lists, report) on the shells' exponents from the response densities ``es``."""
    r = np.array(es["r"]); nes = np.array(es["n"])
    nmod = sum(x["n"] * x["b"] ** 3 / (8 * math.pi) * np.exp(-x["b"] * r) for x in shells)
    m = (nes > 1e-9) & (nmod > 1e-12)
    wt = np.gradient(r) * r * r / np.where(m, nmod, 1.0) * m
    g, g1, g2 = _terms(shells, r)
    out, rep = {}, {}
    for key, l, basis in (("dip", 1, g1), ("quad", 2, -(g2 - g1 / r) / 3)):
        K0 = es["K0"][str(l)]
        if K0 < 1e-8:
            continue
        t = np.array(es["u"][str(l)]) / K0
        c, resid = _constrained(basis, t, wt)
        out[key] = [{"b": sh["b"], "w": [float(x) for x in c[i * N_SHAPE:(i + 1) * N_SHAPE]]} for i, sh in enumerate(shells)]
        rep[key] = {"residual": resid, "certificate": float(K0 * _self_coulomb(r, c @ basis, l))}
    return out, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True); ap.add_argument("--shells", required=True)
    ap.add_argument("--response", required=True); ap.add_argument("--pauli", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = json.load(open(a.base))
    sh = json.load(open(a.shells)); sh = sh.get("ions", sh)
    pa = json.load(open(a.pauli))["params"]
    report = {}
    for s, ion in doc["ions"].items():
        v = sh[s]
        shells = v["shells"] if "shells" in v else [{"n": n, "b": b} for n, b in zip(v["n"], v["b"])]
        shells = sorted(({"n": float(x["n"]), "b": float(x["b"])} for x in shells), key=lambda x: -x["b"])
        Z = float(ion["charge"]) + sum(x["n"] for x in shells)
        if abs(Z - float(v["Z"])) > 1e-6:
            raise ValueError(f"{s}: Z_eff {v['Z']} is not charge + the shells' electrons ({Z})")
        if len(shells) != len(pa[s]["amp"]):
            raise ValueError(f"{s}: {len(shells)} shells but {len(pa[s]['amp'])} site-potential amplitudes")
        ion["shells_pairfit"] = ion["shells"]; ion["Z_pairfit"] = ion["Z"]
        if ion.get("pauli"):          # the dispersion damping exponent the pair-fit table started from
            ion["disp_range"] = float(min(x["b"] for x in ion["shells"]) * ion["pauli"]["s"])
        ion["shells"], ion["Z"] = shells, Z
        ion["shells_note"] = "ln n(r) of the free ion on its valence window, free populations; Z = charge + their sum (scripts/cmm_site_table.py)"
        ion["pauli_site"] = {"a": [float(x) for x in pa[s]["amp"]], "s": float(pa[s]["scale"]),
                             "form": "sum_t a_t e^{-beta r} (1/r + beta/2), beta = s b_t; E_ij = <n_i|v_j> + <n_j|v_i>"}
        es = json.load(open(Path(a.response) / f"es2_{s}.json"))
        par, rep = refit_parents(shells, es)
        old = ion.get("parents") or {}
        ion["parents"] = {**old, **par, "source_dip_quad": "scripts/cmm_site_table.py: free-ion response densities, 1/n metric, shell exponents"}
        for k in ("dip", "quad"):
            if k in old and k in par:
                ion["parents"][k + "_previous"] = old[k]
        report[s] = {"Z": Z, "shells": shells, **rep}
        line = f"{s:>2}: Z_eff {Z:7.3f}  shells " + " ".join(f"({x['n']:.3f}, {x['b']:.3f})" for x in shells)
        for k in ("dip", "quad"):
            if k in rep:
                line += f"  {k}: residual {rep[k]['residual']:.3f} certificate {rep[k]['certificate']:.3f}"
                if not rep[k]["certificate"] < 1:
                    raise ValueError(f"{s}: the {k} parent's Coulomb certificate is {rep[k]['certificate']:.3f} >= 1")
        print(line)
    doc["built_by"] = str(doc.get("built_by", "")) + " + scripts/cmm_site_table.py (density shells, site Pauli, 1/n parents)"
    json.dump(doc, open(a.out, "w"), indent=1)
    json.dump(report, open(str(a.out).replace(".json", ".report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
