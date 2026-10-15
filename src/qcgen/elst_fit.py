"""Per-geometry fits of Slater-damped multipoles to reference electrostatics (rsfff.qcgen.elst_db).

The model is exactly the force field's (:mod:`rsfff.ff.multipole`, ``rsfff.ff.electrostatics``): per
atom a point nucleus ``Z`` and a Slater 1s shell of exponent ``b`` carrying ``[q - Z, mu, Theta]``
(Buckingham, traceless), so the potential of a monomer at a point probe is

    phi(p) = sum_i  e_p^T T(p - r_i) M_i  +  e_p^T [-f_1c(b_i r) T] M_cp_i,      e_p = [1, 0, ...]

(the probe is a bare unit point charge: it has no shell, so only the one-centre penetration of each
atom's shell appears). Fields and field gradients are autograd derivatives with respect to the probe
position. Pair energies use the same contraction as ``slater_elec_pair_energy`` -- point term plus
shell-shell (two-centre, combined exponent sqrt(b_i b_j)) and the two nucleus-shell one-centre terms.

Each monomer geometry is fitted independently (no network): this measures the ceiling of the functional
form and how the parameters move with geometry, before deciding what a network should emit.

Units: atomic units (bohr, e, e bohr, e bohr^2, Hartree).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..ff.multipole import (
    build_polytensor,
    damped_interaction_tensor,
    slater_one_center_damp,
    slater_two_center_damp,
    slater_two_center_damp_unequal,
    spherical_to_cartesian_quadrupole,
)

DTYPE = torch.float64

#: Starting (Z, b) per element: rsfff.ff.response.DEFAULT_ELEC_PRIOR where it has one; otherwise
#: valence-count / IP-scaled guesses (the fit moves them).
ZB_INIT = {1: (0.936, 2.333), 8: (3.616, 2.134), 7: (3.0, 1.91), 9: (4.2, 2.41), 11: (2.141, 2.170),
           17: (6.0, 1.931), 3: (1.5, 3.0), 19: (3.0, 1.8), 35: (6.0, 1.7)}
Q_INIT = {8: -0.39, 1: 0.195, 7: -0.47, 9: -0.24}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class SiteParams:
    """Per-atom parameters of one monomer (torch, atomic units)."""

    q: torch.Tensor       # (N,)
    mu: torch.Tensor      # (N, 3)
    theta: torch.Tensor   # (N, 3, 3) Cartesian, traceless
    Z: torch.Tensor       # (N,)
    b: torch.Tensor       # (N,)

    def polytensors(self, max_rank=2):
        m = build_polytensor(self.q, self.mu, self.theta, max_rank=max_rank)
        m_cp = build_polytensor(self.q - self.Z, self.mu, self.theta, max_rank=max_rank)
        m_z = build_polytensor(self.Z, None, None, max_rank=max_rank)
        return m, m_cp, m_z

    def rotated(self, rot: np.ndarray) -> "SiteParams":
        R = torch.as_tensor(rot, dtype=self.q.dtype)
        return SiteParams(self.q, self.mu @ R.T, R @ self.theta @ R.T, self.Z, self.b)

    def numpy(self):
        return {k: getattr(self, k).detach().cpu().numpy() for k in ("q", "mu", "theta", "Z", "b")}


def potential(coords, sp: SiteParams, points, max_rank=2, penetration=True):
    """phi at ``points`` (P, 3) of the sites at ``coords`` (N, 3): (P,)."""
    N, P = coords.shape[0], points.shape[0]
    dr = (points[:, None, :] - coords[None, :, :]).reshape(-1, 3)          # probe - atom (j - i)
    r = dr.norm(dim=-1)
    rinv = 1.0 / r
    m, m_cp, _ = sp.polytensors(max_rank)
    mi = m.repeat(P, 1)
    T = damped_interaction_tensor(dr, None, rinv, max_rank=max_rank)
    phi = T[:, 0, :].mul(mi).sum(-1)
    if penetration:
        b = sp.b.repeat(P)
        T1 = damped_interaction_tensor(dr, -slater_one_center_damp(b * r, max_rank), rinv, max_rank=max_rank)
        phi = phi + T1[:, 0, :].mul(m_cp.repeat(P, 1)).sum(-1)
    return phi.reshape(P, N).sum(1)


def potential_field(coords, sp, points, max_rank=2, penetration=True, gradient=False, create_graph=True):
    """(phi, E[, dE/dr]) at ``points``: E = -grad phi; the field gradient G_ab = dE_a/dr_b (P, 3, 3)."""
    pts = points.detach().clone().requires_grad_(True)
    phi = potential(coords, sp, pts, max_rank, penetration)
    (g,) = torch.autograd.grad(phi.sum(), pts, create_graph=create_graph or gradient)
    E = -g
    if not gradient:
        return phi, E
    G = torch.stack([torch.autograd.grad(E[:, a].sum(), pts, create_graph=create_graph, retain_graph=True)[0]
                     for a in range(3)], 1)
    return phi, E, G


def pair_energy(ca, spa: SiteParams, cb, spb: SiteParams, max_rank=2, penetration=True, two_center="unequal"):
    """Model electrostatic interaction of two monomers (Hartree): point + penetration, as the FF.

    ``two_center``: "unequal" -- the shell-shell term with each shell's own exponent (HIPPO, exact for two
    Slater 1s densities); "combined" -- the equal-exponent form at sqrt(b_i b_j) (the FF's current default).
    """
    Na, Nb = ca.shape[0], cb.shape[0]
    ii = torch.arange(Na).repeat_interleave(Nb)
    jj = torch.arange(Nb).repeat(Na)
    dr = cb[jj] - ca[ii]
    r = dr.norm(dim=-1)
    rinv = 1.0 / r
    ma, ma_cp, ma_z = spa.polytensors(max_rank)
    mb, mb_cp, mb_z = spb.polytensors(max_rank)

    def c(mi, T, mj):
        return torch.einsum("pa,pab,pb->p", mj, T, mi)

    e = c(ma[ii], damped_interaction_tensor(dr, None, rinv, max_rank=max_rank), mb[jj]).sum()
    if penetration:
        bi, bj = spa.b[ii], spb.b[jj]
        if two_center == "unequal":
            f_ss = slater_two_center_damp_unequal(bi, bj, r, max_rank)
        else:
            f_ss = slater_two_center_damp(torch.sqrt(bi * bj) * r, max_rank)
        t_ss = damped_interaction_tensor(dr, -f_ss, rinv, max_rank=max_rank)
        t_i = damped_interaction_tensor(dr, -slater_one_center_damp(bi * r, max_rank), rinv, max_rank=max_rank)
        t_j = damped_interaction_tensor(dr, -slater_one_center_damp(bj * r, max_rank), rinv, max_rank=max_rank)
        e = e + c(ma_cp[ii], t_ss, mb_cp[jj]).sum() + c(ma_cp[ii], t_i, mb_z[jj]).sum() + c(ma_z[ii], t_j, mb_cp[jj]).sum()
    return e


# ---------------------------------------------------------------------------
# per-geometry fit
# ---------------------------------------------------------------------------
class MonomerFit(torch.nn.Module):
    """Free parameters of one monomer; total charge held exactly."""

    def __init__(self, numbers, charge, max_rank=2, fit_zb=True):
        super().__init__()
        n = len(numbers)
        self.charge, self.max_rank, self.fit_zb = float(charge), max_rank, fit_zb
        q0 = torch.tensor([Q_INIT.get(int(z), 0.0) for z in numbers], dtype=DTYPE)
        q0 = q0 - q0.mean() + self.charge / n
        self.q = torch.nn.Parameter(q0)
        self.mu = torch.nn.Parameter(torch.zeros(n, 3, dtype=DTYPE))
        self.th = torch.nn.Parameter(torch.zeros(n, 5, dtype=DTYPE))
        zb = torch.tensor([ZB_INIT.get(int(z), (max(1.0, 0.6 * _valence(int(z))), 2.0)) for z in numbers], dtype=DTYPE)
        self.logZ = torch.nn.Parameter(zb[:, 0].log(), requires_grad=fit_zb)
        self.logb = torch.nn.Parameter(zb[:, 1].log(), requires_grad=fit_zb)

    def params(self) -> SiteParams:
        q = self.q - self.q.mean() + self.charge / self.q.numel()
        th = spherical_to_cartesian_quadrupole(self.th) if self.max_rank >= 2 else torch.zeros(len(q), 3, 3, dtype=DTYPE)
        mu = self.mu if self.max_rank >= 1 else torch.zeros_like(self.mu)
        return SiteParams(q, mu, th, self.logZ.exp(), self.logb.exp())


SCALE_WEIGHT = {0.6: 0.1, 0.8: 0.3}   # inner shells down-weighted (very large potentials); others 1


def fit_monomer(grid: dict, numbers, charge, max_rank=2, penetration=True, w_field=1.0, w_grad=0.0,
                min_scale=0.0, ridge=1e-5, steps=200, outer=8, init: MonomerFit | None = None):
    """Fit one monomer to its grid (``elst_db.monomer_grid`` record). Returns (MonomerFit, metrics)."""
    X = torch.as_tensor(grid["coords"], dtype=DTYPE)
    sel = grid["scale"] >= min_scale
    P = torch.as_tensor(grid["points"][sel], dtype=DTYPE)
    phi_ref = torch.as_tensor(grid["phi"][sel], dtype=DTYPE)
    E_ref = torch.as_tensor(grid["field"][sel], dtype=DTYPE)
    G_ref = torch.as_tensor(grid["field_grad"][sel], dtype=DTYPE) if w_grad and "field_grad" in grid else None
    w = torch.as_tensor([SCALE_WEIGHT.get(round(float(s), 2), 1.0) for s in grid["scale"][sel]], dtype=DTYPE)
    w = w / w.sum()
    mod = init or MonomerFit(numbers, charge, max_rank, fit_zb=penetration)
    opt = torch.optim.LBFGS([p for p in mod.parameters() if p.requires_grad], lr=1, max_iter=steps,
                            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")

    def loss_fn():
        sp = mod.params()
        out = potential_field(X, sp, P, max_rank, penetration, gradient=G_ref is not None)
        L = (w * (out[0] - phi_ref) ** 2).sum() + w_field * (w * ((out[1] - E_ref) ** 2).sum(1)).sum()
        if G_ref is not None:
            L = L + w_grad * (w * ((out[2] - G_ref) ** 2).sum((1, 2))).sum()
        return L + ridge * ((sp.mu ** 2).sum() + (sp.theta ** 2).sum())

    def closure():
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        return L

    last = None
    for _ in range(outer):           # restart L-BFGS until the loss stops moving (flat charge/dipole gauge)
        L = float(opt.step(closure))
        if last is not None and abs(last - L) <= 1e-10 * max(abs(L), 1e-30):
            break
        last = L
    return mod, grid_metrics(grid, mod.params(), max_rank, penetration)


def grid_metrics(grid, sp: SiteParams, max_rank=2, penetration=True) -> dict:
    """RMS errors of phi (Hartree/e) and |E| per vdW scale, and relative to the reference RMS."""
    X = torch.as_tensor(grid["coords"], dtype=DTYPE)
    P = torch.as_tensor(grid["points"], dtype=DTYPE)
    phi, E = potential_field(X, sp, P, max_rank, penetration, create_graph=False)
    phi, E = phi.detach().numpy(), E.detach().numpy()
    out = {}
    for s in np.unique(grid["scale"]):
        k = grid["scale"] == s
        out[round(float(s), 2)] = {
            "phi_rms": float(np.sqrt(np.mean((phi[k] - grid["phi"][k]) ** 2))),
            "phi_ref_rms": float(np.sqrt(np.mean(grid["phi"][k] ** 2))),
            "E_rms": float(np.sqrt(np.mean(((E[k] - grid["field"][k]) ** 2).sum(1)))),
            "E_ref_rms": float(np.sqrt(np.mean((grid["field"][k] ** 2).sum(1)))),
        }
    return out


def fit_db(db, names=None, log=print, **kw) -> dict:
    """Fit every monomer of an ``elst_db.MonomerDB`` (grids must exist). Returns name -> (SiteParams numpy, metrics)."""
    from . import elst_db as D
    names = names or list(db.index)
    out = {}
    for k, n in enumerate(names):
        g = D.monomer_grid(db, n)
        m = db.get(n)
        numbers = [_Z(s) for s in m.symbols]
        mod, met = fit_monomer(g, numbers, db.index[n]["charge"], **kw)
        out[n] = (mod.params().numpy(), met)
        if log and (k % 10 == 0 or k == len(names) - 1):
            r = met[1.4]["phi_rms"] if 1.4 in met else float("nan")
            log(f"{k + 1}/{len(names)} {n}: phi rms at 1.4 vdW {r * 627.5:.3f} kcal/mol/e")
    return out


def pair_errors(db, frames, fits: dict, max_rank=2, penetration=True, two_center="unequal") -> np.ndarray:
    """Model minus reference e_elst (Hartree) for pair records, from per-geometry fits (rotated into place)."""
    from . import elst_db as D
    err = []
    for f in frames:
        pa, pb = D.reassemble(db, f)
        spa = _sp(fits[f.info["mono_a"]][0]).rotated(pa.rot)
        spb = _sp(fits[f.info["mono_b"]][0]).rotated(pb.rot)
        e = pair_energy(torch.as_tensor(pa.coords), spa, torch.as_tensor(pb.coords), spb, max_rank, penetration,
                        two_center)
        err.append(float(e) - f.info["e_elst"])
    return np.asarray(err)


def _sp(d) -> SiteParams:
    t = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in d.items()}
    return SiteParams(t["q"], t["mu"], t["theta"], t["Z"], t["b"])


def _Z(sym):
    from pyscf.data.elements import charge
    return charge(sym)


def _valence(z):
    shells = [2, 10, 18, 36, 54, 86]
    prev = max([s for s in shells if s < z] or [0])
    return z - prev


# ---------------------------------------------------------------------------
# io and analysis (used by notebooks/elst_fit.ipynb)
# ---------------------------------------------------------------------------
def save_fits(path, fits: dict, meta: dict | None = None):
    """``fits`` (name -> (params numpy dict, metrics)) as JSON."""
    import json
    out = {"meta": meta or {}, "fits": {n: {"params": {k: np.asarray(v).tolist() for k, v in p.items()}, "metrics": m}
                                       for n, (p, m) in fits.items()}}
    with open(path, "w") as f:
        json.dump(out, f)


def load_fits(path) -> dict:
    import json
    with open(path) as f:
        d = json.load(f)
    return {n: ({k: np.asarray(v) for k, v in r["params"].items()}, {float(s): v for s, v in r["metrics"].items()})
            for n, r in d["fits"].items()}


def grid_error_table(db, fits: dict, scales=(0.6, 0.8, 1.0, 1.4, 2.0, 3.0), unit=627.5095):
    """Median phi RMS error per species group and vdW scale (kcal/mol/e by default): rows of dicts."""
    groups = {}
    for n, (_, m) in fits.items():
        g = "ion" if db.index[n]["charge"] != 0 else n.rsplit("_", 1)[0]
        groups.setdefault(g, []).append(m)
    rows = []
    for g, ms in groups.items():
        row = {"group": g, "n": len(ms)}
        for sc in scales:
            row[f"phi@{sc}"] = float(np.median([m[sc]["phi_rms"] for m in ms])) * unit
            row[f"rel@{sc}"] = float(np.median([m[sc]["phi_rms"] / m[sc]["phi_ref_rms"] for m in ms]))
        rows.append(row)
    return rows


def pair_error_table(db, fits: dict, frames, bins=((1.4, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)),
                     max_rank=2, penetration=True, unit=627.5095, two_center="unequal"):
    """MAE (kcal/mol) of the model pair energy from per-geometry fits, by closest-contact bin (Angstrom).
    Returns (rows, contact, error, reference)."""
    err = pair_errors(db, frames, fits, max_rank, penetration, two_center) * unit
    c = np.array([f.info["contact_ang"] for f in frames])
    ref = np.array([f.info["e_elst"] for f in frames]) * unit
    rows = [{"contact": f"{a}-{b}", "n": int(((c >= a) & (c < b)).sum()), "mae": float(np.mean(np.abs(err[(c >= a) & (c < b)]))),
             "ref_mean_abs": float(np.mean(np.abs(ref[(c >= a) & (c < b)])))} for a, b in bins]
    return rows, c, err, ref


def parameter_spread(db, fits: dict, prefix: str):
    """Mean and sd of q, Z, b per atom index over the conformers named ``prefix_*``."""
    rows = [p for n, (p, _) in fits.items() if n.startswith(prefix + "_")]
    sym = db.get(next(n for n in fits if n.startswith(prefix + "_"))).symbols
    out = []
    for i, s in enumerate(sym):
        r = {"atom": f"{s}{i}"}
        for k in ("q", "Z", "b"):
            v = np.array([p[k][i] for p in rows])
            r[k] = float(v.mean())
            r[k + "_sd"] = float(v.std())
        out.append(r)
    return out
