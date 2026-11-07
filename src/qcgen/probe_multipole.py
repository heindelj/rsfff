"""Multipole-resolved static response matrices of a monomer from radial probe potentials (CPKS).

The electric polarizability is the response to the l = 1 probe r C_1^m (a uniform field); its rank-2 analogue
(r^2 C_2^m, a uniform field gradient) gives the quadrupole polarizability. The l = 0 member of the hierarchy has no
uniform probe (r^0 is a constant, which cannot move charge at fixed electron number), so the breathing response is
characterized by a *set* of spherical probes v_k(r) and its response matrix

    Pi_kl = -d^2 E / d lambda_k d lambda_l,      H = H0 + sum_k lambda_k v_k(r)   (v acting on each electron)

from coupled-perturbed KS (``pol_ref.linear_response``: fixed electron number, full hybrid kernel except VV10).
Probes are radial functions times Racah-normalized real harmonics C_l^m (C_l^0 = P_l(cos theta)):

    ("power", n)   r^n                  n = l: the uniform field (l = 1) / field gradient (l = 2); n = 2 at l = 0: r^2
    ("shell", R)   r_<^l / r_>^(l+1)    the l-channel of a unit point charge at distance R (R in Bohr); for l = 0
                                        1 / max(r, R), so a point charge's monopole response is a single shell probe
    ("gauss", a)   r^l exp(-a r^2)      smooth, localized probes (a in Bohr^-2)

One CPKS solve for all probes of all ranks gives the full matrix, cross-rank blocks included: the (l = 1, l = 2) block
of the power probes is the dipole-quadrupole polarizability A (zero for any centrosymmetric monomer, e.g. a free
atom or ion), and its radial resolution comes from the shell / Gaussian probes.

Breathing parent fit (:func:`fit_breathing`): the complete model's l = 0 term moves charge b between the nucleus and
a normalized Slater-type shape g(r) = sum_k w_k N_k r^k e^{-beta r} (the table's ``parents.breath``; the mode
delta_0 - g carries no net charge), so its drive by a spherical probe v is D[v] = v(0) - <g|v> and its energy
-1/2 a0 D[v]^2; across probes Pi^model_kl = a0 D_k D_l. A point charge at R gives D = f_0(beta R) / R, the
penetration form ``scripts/ion_parents.py`` fitted to the diagonal of the shell block alone. Fitting the whole matrix
(off-diagonal couplings, r^2 and Gaussian probes inside the density) fixes a0 and the shape separately; the spectrum
of Pi says how much of the l = 0 response one mode can carry. Exchange (Pauli-pressure) breathing is not here: it is
the model's own exchange polarization, as for the dipoles.

Units: atomic (Bohr, Hartree). Closed-shell monomers (CPKS).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import pol_ref as PR

__all__ = ["racah", "probe_values", "multipole_response", "monomer_multipole_response", "breathing_drive",
           "fit_breathing", "select_probes", "default_probes"]


def racah(l: int, X: np.ndarray) -> np.ndarray:
    """Real Racah-normalized harmonics C_l^m(r^) at points X (P, 3) about the origin: (2l+1, P), m order
    l = 1: (z, x, y); l = 2: (z0, xz, yz, x2-y2, xy). C_l^0 = P_l(cos theta)."""
    r = np.linalg.norm(X, axis=1)
    u = X / np.maximum(r, 1e-300)[:, None]
    x, y, z = u.T
    if l == 0:
        return np.ones((1, len(X)))
    if l == 1:
        return np.stack([z, x, y])
    if l == 2:
        s3 = math.sqrt(3.0)
        return np.stack([0.5 * (3 * z * z - 1), s3 * x * z, s3 * y * z, 0.5 * s3 * (x * x - y * y), s3 * x * y])
    raise ValueError("l <= 2")


def _radial(kind, p, l, r):
    if kind == "power":
        return r ** p
    if kind == "shell":
        rl, rg = np.minimum(r, p), np.maximum(r, p)
        return rl ** l / rg ** (l + 1)
    if kind == "gauss":
        return r ** l * np.exp(-p * r * r)
    raise ValueError(kind)


def probe_values(probes, X):
    """Probe potentials on points X (relative to the origin): (n_probe, P) and labels (l, m, kind, p)."""
    r = np.linalg.norm(X, axis=1)
    vals, labels = [], []
    for l, kind, p, ms in probes:
        C = racah(l, X)
        f = _radial(kind, p, l, r)
        for m in ms:
            vals.append(f * C[m])
            labels.append((l, m, kind, float(p)))
    return np.array(vals), labels


def default_probes(m_all=False, shells_ang=(0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0),
                   gauss=(4.0, 1.0, 0.4, 0.15, 0.06)):
    """(l, kind, param, m's): l = 0 shells (Bohr), Gaussians and r^2; l = 1, 2 the uniform probes plus the
    same shells (the distributed dipole / quadrupole response). ``m_all``: every m (molecules); else m = 0 only
    (spherical monomers: the other m are degenerate and the cross-m blocks vanish)."""
    B = PR.BOHR
    m1 = range(3) if m_all else (0,)
    m2 = range(5) if m_all else (0,)
    out = [(0, "power", 2, (0,))]
    out += [(0, "shell", R / B, (0,)) for R in shells_ang]
    out += [(0, "gauss", a, (0,)) for a in gauss]
    out += [(1, "power", 1, m1)] + [(1, "shell", R / B, m1) for R in shells_ang[2:]]
    out += [(2, "power", 2, m2)] + [(2, "shell", R / B, m2) for R in shells_ang[2:]]
    return out


def multipole_response(placed, probes=None, mf0=None, xc=None, origin=None, grid_level=6, keep_ddm=False):
    """CPKS response matrix of a placed (closed-shell) monomer to ``probes`` (see :func:`default_probes`).

    Returns {"labels": [(l, m, kind, p)], "Pi": (K, K) = -d2E/dl_k dl_l (symmetrized, Hartree per unit^2),
    "asym": the relative antisymmetric part before symmetrizing (a quadrature / convergence check),
    "origin", "radial": {r, rho_l} the l-projected induced densities on a radial grid (m = 0 probes, one row
    per probe; electrons, sign as the induced electron density for +lambda), "ddm" (with ``keep_ddm``)}."""
    from pyscf import dft
    mol = placed.mol
    mf0 = mf0 or PR.isolated(placed, xc)
    o = PR.charge_center(mol) if origin is None else np.asarray(origin, float)
    probes = probes or default_probes()
    g = dft.gen_grid.Grids(mol)
    g.level = grid_level
    g.build()
    ao = dft.numint.eval_ao(mol, g.coords)
    v, labels = probe_values(probes, g.coords - o)
    V = np.einsum("pi,kp,pj->kij", ao, v * g.weights, ao, optimize=True)
    ddm = PR.linear_response(mf0, V)                          # d dm / d lambda_k (electrons)
    P = -np.einsum("kij,lji->kl", V, ddm)                      # -tr(V_k ddm_l)
    asym = float(np.abs(P - P.T).max() / max(np.abs(P).max(), 1e-300))
    out = {"labels": labels, "Pi": 0.5 * (P + P.T), "asym": asym, "origin": o}
    # l-projected radial profiles of the induced densities (about the origin), for shapes and plots
    rho = np.stack([dft.numint.eval_rho(mol, ao, ddm[k]) for k in range(len(labels))])
    X = g.coords - o
    r = np.linalg.norm(X, axis=1)
    edges = np.concatenate([[0], np.geomspace(0.01, 15.0, 160)])
    rad = {"r": 0.5 * (edges[1:] + edges[:-1])}
    for l in sorted({lab[0] for lab in labels}):
        Cl = racah(l, X)[0]
        idx = np.digitize(r, edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)
        shell_vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        rows = []
        for k, lab in enumerate(labels):
            if lab[1] != 0:
                rows.append(np.full(len(edges) - 1, np.nan))
                continue
            # (2l+1) <C_l^0 rho> over the shell: the l-channel radial amplitude
            acc = np.bincount(idx[ok], weights=(rho[k] * Cl * g.weights)[ok], minlength=len(edges) - 1)
            rows.append((2 * l + 1) * acc / shell_vol)
        rad[f"rho_l{l}"] = np.array(rows)
    out["radial"] = rad
    if keep_ddm:
        out["ddm"] = ddm
    return out


def monomer_multipole_response(db, name, probes=None, cache="probe_multipole", overwrite=False, grid_level=6):
    """:func:`multipole_response` of a database monomer, cached in ``db/<cache>/<name>.npz``."""
    out = Path(db.root) / cache / f"{name}.npz"
    if out.exists() and not overwrite:
        z = np.load(out, allow_pickle=True)
        return {"labels": [tuple(x) for x in z["labels"].tolist()], "Pi": z["Pi"], "asym": float(z["asym"]),
                "origin": z["origin"], "radial": z["radial"].item()}
    m = db.get(name)
    p = m.place()
    rec = multipole_response(p, probes, mf0=PR.isolated(p, m.meta.get("xc"), key=name), grid_level=grid_level)
    out.parent.mkdir(exist_ok=True)
    np.savez(out, labels=np.array(rec["labels"], dtype=object), Pi=rec["Pi"], asym=rec["asym"], origin=rec["origin"],
             radial=np.array(rec["radial"], dtype=object))
    return rec


# ---------------------------------------------------------------------------------------------------------- breathing
def _shape_density(beta, w, r):
    """Normalized g(r) = sum_k w_k N_k r^k e^{-beta r} (k = 0..3), sum w = 1."""
    g = 0.0
    for k, wk in enumerate(w):
        if wk:
            g = g + wk * beta ** (k + 3) / (4 * np.pi * math.factorial(k + 2)) * r ** k * np.exp(-beta * r)
    return g


def breathing_drive(beta, w, labels, inner=None, n=6000, rmax=40.0):
    """D_k = <m|v_k> of the breathing mode m = (inner) - g for each l = 0 probe label (others: nan). ``inner`` None:
    the nucleus (a point; the complete model's form, D = v(0) - <g|v>); else (beta_in, w_in), a second normalized
    shape, so m = g_in - g carries no net charge and has a finite core."""
    r = np.linspace(0, rmax, n + 1)[1:]
    dr = r[1] - r[0]
    g = _shape_density(beta, w, r) * 4 * np.pi * r * r * dr
    gi = None if inner is None else _shape_density(inner[0], inner[1], r) * 4 * np.pi * r * r * dr
    D = []
    for l, m, kind, p in labels:
        if l != 0:
            D.append(np.nan)
            continue
        v = _radial(kind, p, 0, r)
        vin = _radial(kind, p, 0, np.array([0.0]))[0] if gi is None else float(np.sum(gi * v))
        D.append(vin - float(np.sum(g * v)))
    return np.array(D)


def select_probes(labels, r_min_ang=1.0, a_max=0.5, power=True):
    """Indices of the l = 0 probes an environment drive resembles: shells at R >= ``r_min_ang`` (a neighbour's
    charge never sits inside the core), Gaussians no tighter than ``a_max`` (Bohr^-2), and r^2 if ``power``."""
    out = []
    for k, (l, m, kind, p) in enumerate(labels):
        if l != 0:
            continue
        if (kind == "shell" and p * PR.BOHR >= r_min_ang - 1e-9) or (kind == "gauss" and p <= a_max) or \
                (kind == "power" and power):
            out.append(k)
    return out


def fit_breathing(rec, probes=None, inner="nucleus", n_starts=24):
    """Fit the breathing mode to the l = 0 block of ``rec["Pi"]`` restricted to ``probes`` (indices into
    rec["labels"]; default :func:`select_probes`): Pi_kl ~ a0 D_k D_l, least squares on the scale-free residuals
    (Pi_kl - model_kl) / sqrt(Pi_kk Pi_ll). ``inner``: "nucleus" (the complete model's mode delta_0 - g) or "shape"
    (g_in - g, both Slater-type). Returns {a0, beta, w, [beta_in, w_in], rel_rms, rel_rms_diag, spectrum (eigenvalues
    of the scaled block, descending; their sum is the number of probes), rank1_fraction, labels, Pi0, model0}."""
    from scipy.optimize import least_squares
    lab = rec["labels"]
    idx = select_probes(lab) if probes is None else list(probes)
    P = rec["Pi"][np.ix_(idx, idx)]
    labs = [lab[k] for k in idx]
    d = np.sqrt(np.clip(np.diag(P), 1e-300, None))
    S = np.outer(d, d)
    two = inner == "shape"

    def unpack(z):
        a0, beta = np.exp(z[0]), np.exp(z[1])
        w = np.exp(z[2:6] - z[2:6].max()); w = w / w.sum()
        ins = None
        if two:
            wi = np.exp(z[7:11] - z[7:11].max())
            ins = (beta * (1 + np.exp(z[6])), wi / wi.sum())       # the inner shape is the tighter one
        return a0, beta, w, ins

    def model(z):
        a0, beta, w, ins = unpack(z)
        D = breathing_drive(beta, w, labs, ins)
        return a0 * np.outer(D, D)

    def resid(z):
        return ((model(z) - P) / S).ravel()

    nz = 11 if two else 6
    lo = np.full(nz, -8.0); hi = np.full(nz, 8.0)
    lo[0], hi[0] = np.log(1e-4), np.log(1e4)
    lo[1], hi[1] = np.log(0.1), np.log(20.0)
    best = None
    rng = np.random.default_rng(0)
    for _ in range(n_starts):
        z0 = rng.normal(0, 1.0, nz)
        z0[0], z0[1] = np.log(rng.uniform(0.05, 20)), np.log(rng.uniform(0.4, 3.0))
        res = least_squares(resid, np.clip(z0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        if best is None or res.cost < best.cost:
            best = res
    a0, beta, w, ins = unpack(best.x)
    M = model(best.x)
    R = (M - P) / S
    ev = np.linalg.eigvalsh(P / S)[::-1]
    out = {"a0": float(a0), "beta": float(beta), "w": w.tolist(), "rel_rms": float(np.sqrt(np.mean(R ** 2))),
           "rel_rms_diag": float(np.sqrt(np.mean(np.diag(R) ** 2))), "spectrum": ev.tolist(),
           "rank1_fraction": float(ev[0] / ev.sum()), "labels": labs, "Pi0": P, "model0": M, "inner": inner}
    if two:
        out["beta_in"], out["w_in"] = float(ins[0]), ins[1].tolist()
    return out
