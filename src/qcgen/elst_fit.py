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

Constraints are collected in :class:`FitConfig` so a fit can be pinned down as needed:

* ``n_shells`` -- Slater 1s shells per site (all sites, or per element), each with its own exponent and a
  positive weight (softmax), so the site density stays a positive mixture;
* ``z`` -- the nuclear charge: free, the valence table :data:`Z_VALENCE`, or any per-element values;
* ``b_min`` -- floor on every shell exponent (b = b_min + softplus, shells ordered b_1 < b_2 < ...);
* ``taper`` -- (r_on, r_off) Angstrom: every penetration term (one- and two-centre) is multiplied by a
  C2 smoothstep of its pair distance, so penetration is exactly zero beyond r_off.

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
    shaped_one_center_damp,
    shaped_two_center_damp,
    spherical_to_cartesian_quadrupole,
)

DTYPE = torch.float64
BOHR = 0.52917721092

#: Valence nuclear charges (the "fixed Z" convention; alkali cations carry their outer core shell).
Z_VALENCE = {1: 1.0, 3: 3.0, 6: 4.0, 7: 5.0, 8: 6.0, 9: 7.0, 11: 9.0, 17: 7.0, 19: 9.0, 35: 7.0}

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
    b: torch.Tensor       # (N,) or (N, S) shell exponents
    w: torch.Tensor | None = None   # (N, K+1) shape weights (None = 1s, HIPPO); one shell only
    sw: torch.Tensor | None = None  # (N, S) shell weights, rows sum to 1 (None = one shell)

    def shells(self):
        """(b, sw), both (N, S)."""
        b = self.b if self.b.dim() == 2 else self.b[:, None]
        return b, (self.sw if self.sw is not None else torch.ones_like(b))

    def polytensors(self, max_rank=2):
        m = build_polytensor(self.q, self.mu, self.theta, max_rank=max_rank)
        m_cp = build_polytensor(self.q - self.Z, self.mu, self.theta, max_rank=max_rank)
        m_z = build_polytensor(self.Z, None, None, max_rank=max_rank)
        return m, m_cp, m_z

    def rotated(self, rot: np.ndarray) -> "SiteParams":
        R = torch.as_tensor(rot, dtype=self.q.dtype)
        return SiteParams(self.q, self.mu @ R.T, R @ self.theta @ R.T, self.Z, self.b, self.w, self.sw)

    def numpy(self):
        out = {k: getattr(self, k).detach().cpu().numpy() for k in ("q", "mu", "theta", "Z", "b")}
        for k in ("w", "sw"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k).detach().cpu().numpy()
        return out


def taper(r, r_on, r_off):
    """C2 smoothstep 1 -> 0 between r_on and r_off (same units as r)."""
    x = ((r - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _taper_bohr(tp):
    return None if tp is None else (tp[0] / BOHR, tp[1] / BOHR)


def _one_center(sp: SiteParams, idx, r, max_rank):
    """sum_s sw_s f_1c(b_s r) for sites ``idx`` at distances ``r``: (2*max_rank+1, P)."""
    b, sw = sp.shells()
    if sp.w is not None:
        if b.shape[1] != 1:
            raise ValueError("shaped densities are single-shell")
        return shaped_one_center_damp(b[idx, 0] * r, sp.w[idx], max_rank)
    return sum(sw[idx, s] * slater_one_center_damp(b[idx, s] * r, max_rank) for s in range(b.shape[1]))


def potential(coords, sp: SiteParams, points, max_rank=2, penetration=True, taper_ang=None):
    """phi at ``points`` (P, 3) of the sites at ``coords`` (N, 3): (P,). ``taper_ang``: (r_on, r_off) A."""
    N, P = coords.shape[0], points.shape[0]
    dr = (points[:, None, :] - coords[None, :, :]).reshape(-1, 3)          # probe - atom (j - i)
    r = dr.norm(dim=-1)
    rinv = 1.0 / r
    m, m_cp, _ = sp.polytensors(max_rank)
    mi = m.repeat(P, 1)
    T = damped_interaction_tensor(dr, None, rinv, max_rank=max_rank)
    phi = T[:, 0, :].mul(mi).sum(-1)
    if penetration:
        f1c = _one_center(sp, torch.arange(N).repeat(P), r, max_rank)
        if taper_ang is not None:
            f1c = f1c * taper(r, *_taper_bohr(taper_ang))
        T1 = damped_interaction_tensor(dr, -f1c, rinv, max_rank=max_rank)
        phi = phi + T1[:, 0, :].mul(m_cp.repeat(P, 1)).sum(-1)
    return phi.reshape(P, N).sum(1)


def potential_field(coords, sp, points, max_rank=2, penetration=True, gradient=False, create_graph=True,
                    taper_ang=None):
    """(phi, E[, dE/dr]) at ``points``: E = -grad phi; the field gradient G_ab = dE_a/dr_b (P, 3, 3)."""
    pts = points.detach().clone().requires_grad_(True)
    phi = potential(coords, sp, pts, max_rank, penetration, taper_ang)
    (g,) = torch.autograd.grad(phi.sum(), pts, create_graph=create_graph or gradient)
    E = -g
    if not gradient:
        return phi, E
    G = torch.stack([torch.autograd.grad(E[:, a].sum(), pts, create_graph=create_graph, retain_graph=True)[0]
                     for a in range(3)], 1)
    return phi, E, G


def _w1s(n, K):
    w = torch.zeros(n, K, dtype=DTYPE)
    w[:, 0] = 1.0
    return w


def _pad(w, K):
    return w if w.shape[1] == K else torch.cat([w, w.new_zeros(w.shape[0], K - w.shape[1])], 1)


def pair_energy(ca, spa: SiteParams, cb, spb: SiteParams, max_rank=2, penetration=True, two_center="unequal",
                taper_ang=None):
    """Model electrostatic interaction of two monomers (Hartree): point + penetration, as the FF.

    Multi-shell sites sum the shell-shell terms over shell pairs (weights sw_is sw_jt) and the
    nucleus-shell terms over shells. Shaped densities (``SiteParams.w``, single shell) always use the
    exact shaped two-center factors. ``two_center`` (1s densities): "unequal" -- each shell's own exponent
    (HIPPO, exact for two Slater 1s densities); "combined" -- the equal-exponent form at sqrt(b_i b_j).
    ``taper_ang``: (r_on, r_off) Angstrom, multiplies every penetration term by a smoothstep of r_ij.
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
        ba, swa = spa.shells()
        bb, swb = spb.shells()
        shaped = spa.w is not None or spb.w is not None
        if shaped:
            if ba.shape[1] != 1 or bb.shape[1] != 1:
                raise ValueError("shaped densities are single-shell")
            bi, bj = ba[ii, 0], bb[jj, 0]
            wa = spa.w if spa.w is not None else _w1s(Na, spb.w.shape[1])
            wb = spb.w if spb.w is not None else _w1s(Nb, spa.w.shape[1])
            K = max(wa.shape[1], wb.shape[1])
            wa, wb = _pad(wa, K)[ii], _pad(wb, K)[jj]
            f_ss = shaped_two_center_damp(bi, bj, wa, wb, r, max_rank)
            f_i, f_j = shaped_one_center_damp(bi * r, wa, max_rank), shaped_one_center_damp(bj * r, wb, max_rank)
        else:
            f_ss = 0.0
            for s in range(ba.shape[1]):
                for t in range(bb.shape[1]):
                    bi, bj = ba[ii, s], bb[jj, t]
                    f = (slater_two_center_damp_unequal(bi, bj, r, max_rank) if two_center == "unequal"
                         else slater_two_center_damp(torch.sqrt(bi * bj) * r, max_rank))
                    f_ss = f_ss + swa[ii, s] * swb[jj, t] * f
            f_i, f_j = _one_center(spa, ii, r, max_rank), _one_center(spb, jj, r, max_rank)
        if taper_ang is not None:
            s_r = taper(r, *_taper_bohr(taper_ang))
            f_ss, f_i, f_j = f_ss * s_r, f_i * s_r, f_j * s_r
        t_ss = damped_interaction_tensor(dr, -f_ss, rinv, max_rank=max_rank)
        t_i = damped_interaction_tensor(dr, -f_i, rinv, max_rank=max_rank)
        t_j = damped_interaction_tensor(dr, -f_j, rinv, max_rank=max_rank)
        e = e + c(ma_cp[ii], t_ss, mb_cp[jj]).sum() + c(ma_cp[ii], t_i, mb_z[jj]).sum() + c(ma_z[ii], t_j, mb_cp[jj]).sum()
    return e


# ---------------------------------------------------------------------------
# per-geometry fit
# ---------------------------------------------------------------------------
@dataclass
class FitConfig:
    """Model and constraints of a per-geometry fit (see the module docstring)."""

    max_rank: int = 2
    penetration: bool = True
    n_shells: int | dict = 1                 # all sites, or {atomic number: shells} (missing -> 1)
    z: None | str | dict = None              # None free; "valence" -> Z_VALENCE; {atomic number: Z} (missing -> free)
    b_min: float | None = None               # bohr^-1, floor on all shell exponents
    taper: tuple | None = None               # (r_on, r_off) Angstrom on all penetration terms
    n_shape: int = 0                         # r^k shape functions (single-shell sites only)
    two_center: str = "unequal"

    def shells_for(self, z):
        return int(self.n_shells.get(int(z), 1)) if isinstance(self.n_shells, dict) else int(self.n_shells)

    def z_for(self, z):
        """Fixed Z for element ``z`` or None (free)."""
        if self.z is None:
            return None
        table = Z_VALENCE if self.z == "valence" else self.z
        v = table.get(int(z))
        return None if v is None else float(v)

    def asdict(self):
        from dataclasses import asdict
        d = asdict(self)
        for k in ("n_shells", "z"):
            if isinstance(d[k], dict):
                d[k] = {str(a): v for a, v in d[k].items()}
        return d

    @classmethod
    def fromdict(cls, d):
        d = dict(d)
        for k in ("n_shells", "z"):
            if isinstance(d.get(k), dict):
                d[k] = {int(a): v for a, v in d[k].items()}
        if d.get("taper") is not None:
            d["taper"] = tuple(d["taper"])
        return cls(**d)


def _isp(x):
    """inverse softplus"""
    x = torch.as_tensor(x, dtype=DTYPE)
    return x + torch.log(-torch.expm1(-x))


class MonomerFit(torch.nn.Module):
    """Free parameters of one monomer under a :class:`FitConfig`; total charge held exactly.

    Shell exponents are ordered and floored: b_1 = b_min + softplus(r_1), b_s = b_{s-1} + softplus(r_s)
    (b_min = 0 without a floor); a site with fewer shells than the widest site uses the first (most
    diffuse) ones, the rest carry zero weight.
    """

    def __init__(self, numbers, charge, max_rank=2, fit_zb=True, n_shape=0, config: FitConfig | None = None,
                 spread=(0.7, 1.6)):
        super().__init__()
        cfg = config or FitConfig(max_rank=max_rank, penetration=fit_zb, n_shape=n_shape)
        self.cfg = cfg
        n = len(numbers)
        self.charge, self.max_rank, self.fit_zb = float(charge), cfg.max_rank, cfg.penetration
        q0 = torch.tensor([Q_INIT.get(int(z), 0.0) for z in numbers], dtype=DTYPE)
        q0 = q0 - q0.mean() + self.charge / n
        self.q = torch.nn.Parameter(q0)
        self.mu = torch.nn.Parameter(torch.zeros(n, 3, dtype=DTYPE))
        self.th = torch.nn.Parameter(torch.zeros(n, 5, dtype=DTYPE))
        zb = torch.tensor([ZB_INIT.get(int(z), (max(1.0, 0.6 * _valence(int(z))), 2.0)) for z in numbers], dtype=DTYPE)
        zfix = torch.tensor([np.nan if cfg.z_for(z) is None else cfg.z_for(z) for z in numbers], dtype=DTYPE)
        self.register_buffer("zfix", zfix)
        self.logZ = torch.nn.Parameter(zb[:, 0].log(), requires_grad=cfg.penetration)
        # shells
        ns = [cfg.shells_for(z) for z in numbers]
        S = max(ns)
        self.n_shape = cfg.n_shape
        if cfg.n_shape and S > 1:
            raise ValueError("shape functions are single-shell")
        self.b_min = float(cfg.b_min or 0.0)
        b1 = zb[:, 1].clamp(min=self.b_min + 0.05)
        # start: a single shell at the prior; several spread geometrically over ``spread`` x prior, most diffuse first
        lo = [spread[0] if k > 1 else 1.0 for k in ns]
        fac = torch.tensor(np.array([np.geomspace(l, spread[1], S) if S > 1 else [1.0] for l in lo]), dtype=DTYPE)
        binit = torch.clamp(b1[:, None] * fac, min=self.b_min + 0.05)
        binit = torch.cummax(binit, 1).values + 1e-3 * torch.arange(S, dtype=DTYPE)
        inc = torch.cat([binit[:, :1] - self.b_min, binit[:, 1:] - binit[:, :-1]], 1)
        self.rb = torch.nn.Parameter(_isp(inc), requires_grad=cfg.penetration)
        mask = torch.arange(S)[None, :] < torch.tensor(ns)[:, None]
        self.register_buffer("smask", mask)
        self.swlog = torch.nn.Parameter(torch.zeros(n, S, dtype=DTYPE), requires_grad=cfg.penetration and S > 1)
        if cfg.n_shape:   # softmax logits, started near the pure 1s density
            init = torch.full((n, cfg.n_shape + 1), -4.0, dtype=DTYPE)
            init[:, 0] = 0.0
            self.wlog = torch.nn.Parameter(init)

    def params(self) -> SiteParams:
        q = self.q - self.q.mean() + self.charge / self.q.numel()
        th = spherical_to_cartesian_quadrupole(self.th) if self.max_rank >= 2 else torch.zeros(len(q), 3, 3, dtype=DTYPE)
        mu = self.mu if self.max_rank >= 1 else torch.zeros_like(self.mu)
        w = torch.softmax(self.wlog, dim=1) if self.n_shape else None
        Z = torch.where(torch.isnan(self.zfix), self.logZ.exp(), self.zfix)
        b = self.b_min + torch.cumsum(torch.nn.functional.softplus(self.rb), 1)
        S = b.shape[1]
        if S == 1:
            return SiteParams(q, mu, th, Z, b[:, 0], w)
        sw = torch.softmax(self.swlog.masked_fill(~self.smask, -torch.inf), dim=1)
        return SiteParams(q, mu, th, Z, b, w, sw)


SCALE_WEIGHT = {0.6: 0.1, 0.8: 0.3}   # inner shells down-weighted (very large potentials); others 1
#: Grid scales reaching into the core (physical shells need them: Gauss's law makes outer grids see only the
#: enclosed charge). Use with ``elst_db.monomer_grid(..., scales=INNER_SCALES, cache="grids_inner")``.
#: Initial exponent spreads (x the element prior, most diffuse to tightest) tried by multi-shell fits.
SHELL_STARTS = ((0.7, 1.6), (0.5, 1.1), (1.0, 2.5), (0.9, 4.0), (1.0, 8.0))
INNER_SCALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0)


def _grid_weights(grid, sel, weights):
    """Per-point weights for phi and for E: "fixed" (SCALE_WEIGHT, same for both) or "relative" (each vdW
    scale contributes equally relative to its reference RMS, for phi and E separately)."""
    sc = np.round(grid["scale"][sel].astype(float), 2)
    if weights == "fixed":
        w = torch.as_tensor([SCALE_WEIGHT.get(float(s), 1.0) for s in sc], dtype=DTYPE)
        w = w / w.sum()
        return w, w
    if weights != "relative":
        raise ValueError(weights)
    phi, E = grid["phi"][sel], grid["field"][sel]
    wp, wE = np.zeros(len(sc)), np.zeros(len(sc))
    u = np.unique(sc)
    for s in u:
        k = sc == s
        wp[k] = 1.0 / (k.sum() * np.mean(phi[k] ** 2) * len(u))
        wE[k] = 1.0 / (k.sum() * np.mean((E[k] ** 2).sum(1)) * len(u))
    return torch.as_tensor(wp, dtype=DTYPE), torch.as_tensor(wE, dtype=DTYPE)


def fit_monomer(grid: dict, numbers, charge, max_rank=2, penetration=True, w_field=1.0, w_grad=0.0,
                min_scale=0.0, ridge=1e-5, steps=200, outer=8, init: MonomerFit | None = None, n_shape=0,
                config: FitConfig | None = None, weights="fixed", starts=None):
    """Fit one monomer to its grid (``elst_db.monomer_grid`` record). Returns (MonomerFit, metrics).

    ``config`` sets the model and constraints (default: ``max_rank``, ``penetration``, ``n_shape`` with one
    free shell, free Z, no floor, no taper -- the original model). ``weights``: see :func:`_grid_weights`.
    Multi-shell fits have local minima (a shell collapsing onto another, or swapping roles): they are
    started from each initial exponent spread in ``starts`` (default :data:`SHELL_STARTS`) and the lowest
    loss is kept.
    """
    cfg = config or FitConfig(max_rank=max_rank, penetration=penetration, n_shape=n_shape)
    multi = cfg.penetration and max(cfg.shells_for(z) for z in numbers) > 1
    if init is None and multi:
        best = None
        for sp_ in (starts or SHELL_STARTS):
            m = MonomerFit(numbers, charge, config=cfg, spread=sp_)
            m, met = fit_monomer(grid, numbers, charge, w_field=w_field, w_grad=w_grad, min_scale=min_scale,
                                 ridge=ridge, steps=steps, outer=outer, init=m, config=cfg, weights=weights)
            if best is None or m.loss < best[0].loss:
                best = (m, met)
        return best
    X = torch.as_tensor(grid["coords"], dtype=DTYPE)
    sel = grid["scale"] >= min_scale
    P = torch.as_tensor(grid["points"][sel], dtype=DTYPE)
    phi_ref = torch.as_tensor(grid["phi"][sel], dtype=DTYPE)
    E_ref = torch.as_tensor(grid["field"][sel], dtype=DTYPE)
    G_ref = torch.as_tensor(grid["field_grad"][sel], dtype=DTYPE) if w_grad and "field_grad" in grid else None
    wp, wE = _grid_weights(grid, sel, weights)
    mod = init or MonomerFit(numbers, charge, config=cfg)
    opt = torch.optim.LBFGS([p for p in mod.parameters() if p.requires_grad], lr=1, max_iter=steps,
                            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")

    def loss_fn():
        sp = mod.params()
        out = potential_field(X, sp, P, cfg.max_rank, cfg.penetration, gradient=G_ref is not None, taper_ang=cfg.taper)
        L = (wp * (out[0] - phi_ref) ** 2).sum() + w_field * (wE * ((out[1] - E_ref) ** 2).sum(1)).sum()
        if G_ref is not None:
            L = L + w_grad * (wE * ((out[2] - G_ref) ** 2).sum((1, 2))).sum()
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
    mod.loss = float(loss_fn().detach())
    return mod, grid_metrics(grid, mod.params(), cfg.max_rank, cfg.penetration, cfg.taper)


def grid_metrics(grid, sp: SiteParams, max_rank=2, penetration=True, taper_ang=None) -> dict:
    """RMS errors of phi (Hartree/e) and |E| per vdW scale, and relative to the reference RMS."""
    X = torch.as_tensor(grid["coords"], dtype=DTYPE)
    P = torch.as_tensor(grid["points"], dtype=DTYPE)
    phi, E = potential_field(X, sp, P, max_rank, penetration, create_graph=False, taper_ang=taper_ang)
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


def fit_db(db, names=None, log=print, grid_kw=None, **kw) -> dict:
    """Fit every monomer of an ``elst_db.MonomerDB``. Returns name -> (SiteParams numpy, metrics).
    ``grid_kw`` goes to ``elst_db.monomer_grid`` (e.g. ``dict(scales=INNER_SCALES, cache="grids_inner")``)."""
    from . import elst_db as D
    names = names or list(db.index)
    out = {}
    for k, n in enumerate(names):
        g = D.monomer_grid(db, n, **(grid_kw or {}))
        m = db.get(n)
        numbers = [_Z(s) for s in m.symbols]
        mod, met = fit_monomer(g, numbers, db.index[n]["charge"], **kw)
        out[n] = (mod.params().numpy(), met)
        if log and (k % 10 == 0 or k == len(names) - 1):
            r = met[1.4]["phi_rms"] if 1.4 in met else float("nan")
            log(f"{k + 1}/{len(names)} {n}: phi rms at 1.4 vdW {r * 627.5:.3f} kcal/mol/e")
    return out


def pair_errors(db, frames, fits: dict, max_rank=2, penetration=True, two_center="unequal", taper_ang=None) -> np.ndarray:
    """Model minus reference e_elst (Hartree) for pair records, from per-geometry fits (rotated into place).
    ``fits`` may hold different models per monomer (shell counts, fixed or free Z)."""
    from . import elst_db as D
    err = []
    for f in frames:
        pa, pb = D.reassemble(db, f)
        spa = _sp(fits[f.info["mono_a"]][0]).rotated(pa.rot)
        spb = _sp(fits[f.info["mono_b"]][0]).rotated(pb.rot)
        e = pair_energy(torch.as_tensor(pa.coords), spa, torch.as_tensor(pb.coords), spb, max_rank, penetration,
                        two_center, taper_ang)
        err.append(float(e) - f.info["e_elst"])
    return np.asarray(err)


def _sp(d) -> SiteParams:
    t = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in d.items()}
    return SiteParams(t["q"], t["mu"], t["theta"], t["Z"], t["b"], t.get("w"), t.get("sw"))


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
                     max_rank=2, penetration=True, unit=627.5095, two_center="unequal", taper_ang=None):
    """MAE (kcal/mol) of the model pair energy from per-geometry fits, by closest-contact bin (Angstrom).
    Returns (rows, contact, error, reference)."""
    err = pair_errors(db, frames, fits, max_rank, penetration, two_center, taper_ang) * unit
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
