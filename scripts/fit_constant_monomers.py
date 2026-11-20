"""Constant-parameter bonded fits of the H2O, H3O+ and OH- monomers (wB97M-V/def2-TZVPD).

The baseline of the molecular cmm reference (project doc ``claude/molecular_reference_plan.md``): the
bonded functional forms with one constant per parameter and no network. It answers which terms each
species needs and gives the initial per-type tables for the parameter-emitting model.

Terms (the forms of ``rsfff.ff.film.bonded`` / ``rsfff.ff.film.terms``, atomic units; X = O, legs X-H):

    morse     D [y^2 - 1],  y = 1 - exp(-beta (r - r_eq)),  beta = sqrt(k / 2D)
    angle     (k_theta / 2) dc^2,  dc = cos(theta) - cos(theta_eq)
    bb        2 rho_bb D y_j y_k                               the two legs of an angle
    ba        2 rho_ba sqrt(D k_theta / 2) (y_j + y_k) dc_jk   an angle with its own legs
    bo        2 rho_bo sqrt(D k_theta / 2) y_l dc_jk           an angle with the opposite bond (degree 3; new)
    aa        2 rho_aa (k_theta / 2) dc_a dc_b                 two angles sharing a leg (degree 3)
    improper  (k_chi / 2) s (s - 2 c_chi),  s = sin^2(chi)     Wilson out-of-plane, per leg (degree 3)

rho = 0.95 tanh(raw), as in ``terms.py``. Each fit has a free energy offset, so D is set by the shape of
the surface (anharmonicity), not by an atomization energy.

The ``tables`` section of the output holds one chosen rung per species (``table`` in ``SPECIES``), keyed by
(Z, degree) types: the initial per-type tables of ``rsfff.ff.bonded.BondedTables``.

Fitted to energies and forces of the sampled frames (80 / 20 split). Not fitted, used as checks: the
harmonic frequencies and the geometry of the optimized structure (``*_opt_*_pol.xyz``).

    python scripts/fit_constant_monomers.py [--data data/wb97mv_tzvpd] [--out params/bonded_monomers_wb97mv_tzvpd.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import scipy.optimize as so
import torch

torch.set_default_dtype(torch.float64)

BOHR = 0.529177210903
KCAL = 627.509474
CM = 219474.6313
AMU = 1822.888486
MASS = {"O": 15.99491462, "H": 1.00782503}
RHO_MAX = 0.95
SIGMA_E = 0.1 / KCAL                    # Ha: 0.1 kcal/mol
SIGMA_F = 1.0 / KCAL * BOHR             # Ha/bohr: 1 kcal/mol/A

SPECIES = {
    "h2o": dict(file="h2o_wb97mv_tzvpd.xyz", opt="h2o_opt_wb97mv_tzvpd_pol.xyz", charge=0,
                table="+bb+ba", ladder=[("morse+angle", ()), ("+bb", ("bb",)), ("+bb+ba", ("bb", "ba"))]),
    "h3o+": dict(file="h3o+_wb97mv_tzvpd.xyz", opt="h3o+_opt_wb97mv_tzvpd_pol.xyz", charge=1, table="+bb+ba+aa+bo",
                 ladder=[("morse+angle", ()), ("+improper", ("imp",)), ("+bb+ba", ("bb", "ba")),
                         ("+bb+ba+aa", ("bb", "ba", "aa")), ("+bb+ba+bo", ("bb", "ba", "bo")),
                         ("+bb+ba+aa+bo", ("bb", "ba", "aa", "bo")),
                         ("+bb+ba+aa+bo+improper", ("bb", "ba", "aa", "bo", "imp"))]),
    "oh-": dict(file="oh-_wb97mv_tzvpd.xyz", opt="oh-_opt_wb97mv_tzvpd_pol.xyz", charge=-1,
                table="morse", ladder=[("morse", ())]),
}


# ---- data -----------------------------------------------------------------------------------------------------
def read_xyz(path):
    """Frames of an extxyz file: symbols, positions (bohr), forces (Ha/bohr), energy (Ha), frequencies or None."""
    frames, lines, i = [], Path(path).read_text().splitlines(), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n, head = int(lines[i].split()[0]), lines[i + 1]
        cols = re.search(r"Properties=(\S+)", head).group(1).split(":")
        off, where = 0, {}
        for name, _, w in zip(cols[0::3], cols[1::3], cols[2::3]):
            where[name] = (off, int(w))
            off += int(w)
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        take = lambda key: np.array([[float(x) for x in r[where[key][0]:where[key][0] + where[key][1]]] for r in rows])
        fr = re.search(r'frequencies="([^"]*)"', head)
        frames.append(dict(sym=[r[0] for r in rows], pos=take("pos") / BOHR, forces=take("forces"),
                           energy=float(re.search(r"(?<![a-z_])energy=(\S+)", head).group(1)),
                           freq=None if fr is None else np.array([float(x) for x in fr.group(1).split()])))
        i += 2 + n
    return frames


def stack(frames):
    sym = frames[0]["sym"]
    assert all(f["sym"] == sym for f in frames) and sym[0] == "O" and set(sym[1:]) == {"H"}
    return (torch.tensor(np.stack([f["pos"] for f in frames])), torch.tensor(np.stack([f["forces"] for f in frames])),
            torch.tensor([f["energy"] for f in frames]))


# ---- the model ------------------------------------------------------------------------------------------------
def names(n_h, terms):
    p = ["r_eq", "ln_D", "ln_k"]
    if n_h >= 2:
        p += ["theta_eq", "ln_k_theta"]
    p += [f"raw_{t}" for t in ("bb", "ba", "bo", "aa") if t in terms]
    if "imp" in terms:
        p += ["c_chi", "ln_k_chi"]
    return p + ["offset"]


def physical(p):
    """The fit variables as physical parameters (atomic units, angles in degrees)."""
    out = {"r_eq": p["r_eq"], "D": math.exp(p["ln_D"]), "k": math.exp(p["ln_k"])}
    out["beta"] = math.sqrt(out["k"] / (2 * out["D"]))
    if "theta_eq" in p:
        out["theta_eq_deg"], out["k_theta"] = math.degrees(p["theta_eq"]), math.exp(p["ln_k_theta"])
    for t in ("bb", "ba", "bo", "aa"):
        if f"raw_{t}" in p:
            out[f"rho_{t}"] = RHO_MAX * math.tanh(p[f"raw_{t}"])
    if "c_chi" in p:
        out["c_chi"], out["k_chi"] = p["c_chi"], math.exp(p["ln_k_chi"])
    out["offset"] = p["offset"]
    return out


def energy(x, p):
    """(M,) bonded energy of (M, 1 + n_h, 3) positions in bohr, O first; ``p``: name -> scalar tensor."""
    b = x[:, 1:] - x[:, :1]
    r = b.norm(dim=-1)
    u = b / r.unsqueeze(-1)
    n_h = r.shape[1]
    D, k = torch.exp(p["ln_D"]), torch.exp(p["ln_k"])
    y = 1.0 - torch.exp(-torch.sqrt(k / (2 * D)) * (r - p["r_eq"]))
    e = (D * (y * y - 1.0)).sum(-1) + p["offset"]
    if n_h < 2:
        return e
    kt = torch.exp(p["ln_k_theta"])
    rho = lambda t: RHO_MAX * torch.tanh(p[f"raw_{t}"]) if f"raw_{t}" in p else None
    pairs = [(j, l) for j in range(n_h) for l in range(j + 1, n_h)]
    dc = {jl: (u[:, jl[0]] * u[:, jl[1]]).sum(-1) - torch.cos(p["theta_eq"]) for jl in pairs}
    s_bt = torch.sqrt(D * kt / 2)
    for (j, l), d in dc.items():
        e = e + 0.5 * kt * d * d
        if rho("bb") is not None:
            e = e + 2 * rho("bb") * D * y[:, j] * y[:, l]
        if rho("ba") is not None:
            e = e + 2 * rho("ba") * s_bt * (y[:, j] + y[:, l]) * d
        if rho("bo") is not None and n_h == 3:
            e = e + 2 * rho("bo") * s_bt * y[:, 3 - j - l] * d
    if rho("aa") is not None and n_h == 3:
        for a in range(len(pairs)):
            for c in range(a + 1, len(pairs)):
                e = e + 2 * rho("aa") * (kt / 2) * dc[pairs[a]] * dc[pairs[c]]
    if "c_chi" in p and n_h == 3:
        kc = torch.exp(p["ln_k_chi"])
        for leg, j, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            n = torch.cross(u[:, j], u[:, l], dim=-1)
            s = (u[:, leg] * n).sum(-1) ** 2 / (n * n).sum(-1).clamp_min(1e-10)
            e = e + 0.5 * kc * s * (s - 2 * p["c_chi"])
    return e


def energy_forces(x, p):
    x = x.clone().requires_grad_(True)
    e = energy(x, p)
    return e.detach(), -torch.autograd.grad(e.sum(), x)[0]


# ---- fit and analysis -----------------------------------------------------------------------------------------
def fit(x, f, e, n_h, terms, start):
    keys = names(n_h, terms)
    v0 = np.array([start.get(k, 0.0) for k in keys])
    to_p = lambda v: {k: torch.tensor(float(a)) for k, a in zip(keys, v)}

    def res(v):
        em, fm = energy_forces(x, to_p(v))
        return torch.cat([(em - e) / SIGMA_E, ((fm - f) / SIGMA_F).reshape(-1)]).numpy()

    lo = np.array([-2.0 if k == "c_chi" else -40.0 if k == "ln_k_chi" else -np.inf for k in keys])
    hi = np.array([1.0 if k == "c_chi" else np.inf for k in keys])
    sol = so.least_squares(res, np.clip(v0, lo, hi), bounds=(lo, hi), x_scale="jac", xtol=1e-14, ftol=1e-14,
                           gtol=1e-12, max_nfev=4000)
    return dict(zip(keys, map(float, sol.x)))


def errors(x, f, e, p):
    em, fm = energy_forces(x, {k: torch.tensor(v) for k, v in p.items()})
    return dict(e_mae=float((em - e).abs().mean()) * KCAL, e_max=float((em - e).abs().max()) * KCAL,
                f_rmse=float(((fm - f) ** 2).mean().sqrt()) * KCAL / BOHR)


def minimum(x0, p):
    """Model minimum from x0 (1 + n_h, 3): geometry, harmonic frequencies (cm^-1), energy."""
    pt = {k: torch.tensor(v) for k, v in p.items()}
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = energy(x[None], pt)[0]
        v.backward()
        return v

    opt.step(closure)
    xe = x.detach()
    n = xe.shape[0]
    h = torch.autograd.functional.hessian(lambda q: energy(q.view(1, n, 3), pt)[0], xe.reshape(-1))
    m = torch.tensor([MASS["O"]] + [MASS["H"]] * (n - 1)).repeat_interleave(3) * AMU
    w2 = torch.linalg.eigvalsh(h / torch.sqrt(m[:, None] * m[None, :]))
    n_vib = 3 * n - (5 if n == 2 else 6)
    freq = (w2[-n_vib:].clamp(min=0).sqrt() * CM).tolist()
    b = xe[1:] - xe[0]
    r = b.norm(dim=-1)
    ang = [math.degrees(math.acos(float(b[j] @ b[l] / (r[j] * r[l])))) for j in range(n - 1) for l in range(j + 1, n - 1)]
    return dict(x=xe, r=float(r.mean()) * BOHR, angle=float(np.mean(ang)) if ang else None, freq=freq,
                energy=float(energy(xe[None], pt)[0]))


def planar_barrier(p, e_min):
    """H3O+ inversion barrier (kcal/mol): the D3h structure relaxed in r, minus the pyramidal minimum."""
    pt = {k: torch.tensor(v) for k, v in p.items()}
    best = math.inf
    for r in np.linspace(0.90, 1.10, 401) / BOHR:
        h = [[r * math.cos(2 * math.pi * a / 3), r * math.sin(2 * math.pi * a / 3), 0.0] for a in range(3)]
        best = min(best, float(energy(torch.tensor([[[0.0, 0.0, 0.0]] + h]), pt)[0]))
    return (best - e_min) * KCAL


def coverage(x):
    b = x[:, 1:] - x[:, :1]
    r = b.norm(dim=-1)
    out = dict(r_min=float(r.min()) * BOHR, r_max=float(r.max()) * BOHR)
    if r.shape[1] >= 2:
        u = b / r.unsqueeze(-1)
        c = torch.stack([(u[:, j] * u[:, l]).sum(-1) for j in range(r.shape[1]) for l in range(j + 1, r.shape[1])], -1)
        th = torch.rad2deg(torch.acos(c.clamp(-1, 1)))
        out.update(theta_min=float(th.min()), theta_max=float(th.max()))
    if r.shape[1] == 3:
        n = torch.cross(u[:, 1], u[:, 2], dim=-1)
        s = (u[:, 0] * n).sum(-1) ** 2 / (n * n).sum(-1)
        out.update(sin2chi_min=float(s.min()), sin2chi_max=float(s.max()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/wb97mv_tzvpd")
    ap.add_argument("--out", default="params/bonded_monomers_wb97mv_tzvpd.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = {"method": "wB97M-V", "basis": "def2-TZVPD", "units": "atomic (angles in degrees)",
              "sigma": {"energy_kcal": 0.1, "force_kcal_per_A": 1.0}, "tables": {"bond": {}, "apex": {}}, "species": {}}
    for name, spec in SPECIES.items():
        frames = read_xyz(Path(args.data) / spec["file"])
        ref = read_xyz(Path(args.data) / spec["opt"])[0]
        x, f, e = stack(frames)
        e = e - ref["energy"]                                # relative to the optimized structure
        perm = np.random.default_rng(args.seed).permutation(len(frames))
        tr, te = torch.tensor(perm[: int(0.8 * len(perm))]), torch.tensor(perm[int(0.8 * len(perm)):])
        n_h = x.shape[1] - 1
        x_ref = torch.tensor(ref["pos"])
        b = x_ref[1:] - x_ref[0]
        start = {"r_eq": float(b.norm(dim=-1).mean()), "ln_D": math.log(0.18), "ln_k": math.log(0.5),
                 "ln_k_theta": math.log(0.15), "c_chi": -0.3, "ln_k_chi": math.log(0.01)}
        if n_h >= 2:
            start["theta_eq"] = math.acos(float(b[0] @ b[1] / (b[0].norm() * b[1].norm())))
        cov = coverage(x)
        print(f"\n=== {name}: {len(frames)} frames, E up to {float(e.max()) * KCAL:.1f} kcal/mol above the minimum; "
              + "  ".join(f"{k} {v:.3f}" for k, v in cov.items()))
        print(f"    reference: r {float(b.norm(dim=-1).mean()) * BOHR:.4f} A"
              + (f"  angle {math.degrees(start['theta_eq']):.2f}" if n_h >= 2 else "")
              + "  freq " + " ".join(f"{w:.0f}" for w in ref["freq"]))
        entry = {"charge": spec["charge"], "frames": len(frames), "coverage": cov,
                 "reference": {"frequencies": ref["freq"].tolist()}, "fits": {}}
        prev = dict(start)
        for label, terms in spec["ladder"]:
            p = fit(x[tr], f[tr], e[tr], n_h, terms, prev)
            prev = {**prev, **{k: v for k, v in p.items() if k not in ("c_chi", "ln_k_chi")}}
            a, t = errors(x[tr], f[tr], e[tr], p), errors(x[te], f[te], e[te], p)
            mn = minimum(x_ref, p)
            phys = physical(p)
            rec = {"terms": ["morse"] + (["angle"] if n_h >= 2 else []) + list(terms), "parameters": phys,
                   "train": a, "test": t, "minimum": {"r_A": mn["r"], "angle_deg": mn["angle"], "frequencies": mn["freq"]}}
            line = (f"  {label:22s} E mae {t['e_mae']:.3f} (train {a['e_mae']:.3f}, max {t['e_max']:.2f}) kcal/mol   "
                    f"F rmse {t['f_rmse']:.2f} (train {a['f_rmse']:.2f}) kcal/mol/A")
            line += f"\n  {'':22s} min: r {mn['r']:.4f}" + (f" angle {mn['angle']:.2f}" if mn["angle"] else "")
            line += "  freq " + " ".join(f"{w:.0f}" for w in mn["freq"])
            if n_h == 3:
                rec["minimum"]["inversion_barrier_kcal"] = planar_barrier(p, mn["energy"])
                line += f"  barrier {rec['minimum']['inversion_barrier_kcal']:.2f} kcal/mol"
            line += f"\n  {'':22s} " + "  ".join(f"{k} {v:.4f}" for k, v in phys.items() if k != "offset")
            print(line)
            entry["fits"][label] = rec
        result["species"][name] = entry
        chosen = entry["fits"][spec["table"]]["parameters"]
        result["tables"]["bond"][f"1.1-8.{n_h}"] = {k: chosen[k] for k in ("r_eq", "D", "k")}
        if n_h >= 2:
            result["tables"]["apex"][f"8.{n_h}"] = {k: v for k, v in chosen.items()
                                                    if k not in ("r_eq", "D", "k", "beta", "offset")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
