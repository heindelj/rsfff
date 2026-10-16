"""Site polarizabilities and induced-density shapes fitted to reference response (:mod:`pol_ref`).

The model is the force field's induced dipole carried by a density rather than a point: site i holds an
induced dipole mu_i whose charge density is the dipole of a normalized shape density g_i,

    delta rho_i(r) = -mu_i . grad g_i(r - r_i),      g_i = sum_k w_ik N_k r^k e^{-b_i r},  w_ik >= 0,

so its potential is the one-centre damped dipole term of the permanent model (:mod:`elst_fit`, with the
site's *induced* exponent b_i and shape w_i in place of the permanent shell's):

    delta phi(p) = sum_i mu_i . (p - r_i) [1 - f3_i(|p - r_i|)] / |p - r_i|^3 .

``n_shape = 0`` is a Slater 1s dipole density; ``n_shape >= 2`` contains the exact first-order response
of the hydrogen atom, which is the dipole of g(r) ~ e^{-2r} (3/8 + 3r/4 + r^2/4), i.e. b = 2,
w = (1/6, 1/2, 1/3) and alpha = 9/2 (``HYDROGEN_EXACT``).

Within a small molecule (the database's water, HF, NH3: every pair 1-2 or 1-3) the induced dipoles do not
interact, so mu_i = A_i E(r_i) and, in a uniform field, the molecular polarizability is sum_i A_i. Site
polarizabilities are isotropic (A_i = alpha_i 1) or anisotropic (symmetric positive definite tensors).

Charge flow (``charge_flow=True``): each bond b = (i, j) carries a conductance k_b >= 0 and moves charge down
the potential difference at its ends, dq_i = -k_b (phi_i - phi_j) = -dq_j, with the charge spread over the
site's induced shape density. In a uniform field it adds k_b d_b d_b^T (d_b = r_i - r_j) to the molecular
polarizability -- positive semidefinite, along the bond. Bonds from covalent radii (x ``bond_scale``).

The polarizability as a constraint (``PolConfig.alpha``):

* "free"    -- only the grid data (d phi / dF, d E / dF on the monomer's scaled vdW surfaces);
* "penalty" -- plus ``alpha_weight * |sum A_i - alpha_t|^2 / |alpha_t|^2``;
* "exact"   -- the target by construction. Anisotropic sites without charge flow or fixed sites: the full
  tensor, A_i = M K^-1 S_i K^-T M^T with S_i positive definite, K K^T = sum S_j and M M^T = alpha_t, so
  sum A_i = alpha_t and every site stays positive definite. Otherwise the isotropic mean: all free pieces
  (site tensors, bond terms) share one positive scale fixed by tr(alpha_t) (anisotropy from the data).

``alpha_t`` defaults to the reference polarizability of the record; ``alpha_fixed`` pins chosen elements'
isotropic site polarizabilities (with "exact", the rest share what remains). Exponents: per-site or tied
per element, floored at ``b_min``, or fixed per element (``b_fixed``).

Units: atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import elst_fit as F

DTYPE = torch.float64
HYDROGEN_EXACT = {"alpha": 4.5, "b": 2.0, "w": (1 / 6, 1 / 2, 1 / 3)}


@dataclass
class PolConfig:
    aniso: bool = False
    n_shape: int = 0                     # shape functions r^k, k <= n_shape (0: Slater 1s dipole density)
    tie_elements: bool = False           # one exponent / shape per element instead of per site
    b_min: float | None = None
    b_fixed: dict | None = None          # {atomic number: b}
    alpha: str = "free"                  # "free" | "penalty" | "exact"
    alpha_target: object = None          # (3, 3), scalar, or None (the record's reference alpha)
    alpha_weight: float = 1.0
    alpha_fixed: dict | None = None      # {atomic number: isotropic site alpha} (isotropic sites)
    w_field: float = 1.0
    taper: tuple | None = None           # (r_on, r_off) Angstrom on the induced-density penetration
    charge_flow: bool = False            # bond conductances (charge flow between bonded sites)
    bond_scale: float = 1.25             # bonded if r_ij < bond_scale (r_cov,i + r_cov,j)


def bonds_of(numbers, coords_bohr, scale=1.25):
    """Bonded pairs (i < j) from covalent radii (ase), coordinates in Bohr."""
    from ase.data import covalent_radii
    r = np.array([covalent_radii[int(z)] for z in numbers]) / 0.52917721092
    X = np.asarray(coords_bohr, float)
    return [(i, j) for i in range(len(r)) for j in range(i + 1, len(r))
            if np.linalg.norm(X[i] - X[j]) < scale * (r[i] + r[j])]


def _isp(x):
    x = torch.as_tensor(x, dtype=DTYPE)
    return x + torch.log(-torch.expm1(-x))


class PolFit(torch.nn.Module):
    """Site polarizabilities and induced-density exponents/shapes of one monomer under a :class:`PolConfig`."""

    def __init__(self, numbers, cfg: PolConfig, alpha_ref=None, b_init=None, coords=None):
        super().__init__()
        self.cfg, self.numbers = cfg, [int(z) for z in numbers]
        n = len(numbers)
        self.bonds = bonds_of(numbers, coords, cfg.bond_scale) if cfg.charge_flow else []
        X = torch.zeros(n, 3, dtype=DTYPE) if coords is None else torch.as_tensor(np.asarray(coords, float), dtype=DTYPE)
        self.register_buffer("d_b", torch.stack([X[i] - X[j] for i, j in self.bonds]) if self.bonds
                             else torch.zeros(0, 3, dtype=DTYPE))
        self.lk = torch.nn.Parameter(torch.full((len(self.bonds),), -2.0, dtype=DTYPE))
        at = cfg.alpha_target if cfg.alpha_target is not None else alpha_ref
        if at is not None:
            at = torch.as_tensor(np.asarray(at, float), dtype=DTYPE)
            at = at * torch.eye(3, dtype=DTYPE) if at.ndim == 0 else 0.5 * (at + at.T)
        self.register_buffer("alpha_t", at if at is not None else torch.full((3, 3), float("nan"), dtype=DTYPE))
        if cfg.alpha in ("exact", "penalty") and at is None:
            raise ValueError("alpha constraint needs alpha_target or a reference alpha")
        fixed = cfg.alpha_fixed or {}
        if cfg.aniso and fixed:
            raise ValueError("alpha_fixed applies to isotropic sites")
        self.register_buffer("afix", torch.tensor([fixed.get(z, np.nan) for z in self.numbers], dtype=DTYPE))
        a_bar = float(torch.trace(at) / 3) if at is not None else 1.0 * n
        free = torch.isnan(self.afix)
        # site polarizabilities: start shared equally among free sites
        self.log_a = torch.nn.Parameter(torch.full((n,), np.log(max(a_bar, 1e-3) / max(int(free.sum()), 1)), dtype=DTYPE))
        if cfg.aniso:
            L0 = torch.zeros(n, 6, dtype=DTYPE)
            L0[:, :3] = 0.5 * np.log(max(a_bar, 1e-3) / n)            # log of the Cholesky diagonal
            self.L = torch.nn.Parameter(L0)
        # exponents: one per site, or per element
        keys = sorted(set(self.numbers)) if cfg.tie_elements else list(range(n))
        self.key_of = [keys.index(z) if cfg.tie_elements else i for i, z in enumerate(self.numbers)]
        zkey = keys if cfg.tie_elements else self.numbers
        self.b_min = float(cfg.b_min or 0.0)
        b0 = []
        for k, z in zip(range(len(keys)), zkey):
            bf = (cfg.b_fixed or {}).get(int(z))
            b0.append(bf if bf is not None else (b_init.get(int(z)) if isinstance(b_init, dict) else b_init)
                      or F.ZB_INIT.get(int(z), (0, 2.0))[1])
        self.register_buffer("bfix", torch.tensor([(cfg.b_fixed or {}).get(int(z), np.nan) for z in zkey], dtype=DTYPE))
        self.rb = torch.nn.Parameter(_isp(torch.tensor(b0, dtype=DTYPE).clamp_min(self.b_min + 0.05) - self.b_min))
        if cfg.n_shape:
            self.wlog = torch.nn.Parameter(torch.zeros(len(keys), cfg.n_shape + 1, dtype=DTYPE))

    # -- parameters ---------------------------------------------------------
    def b(self):
        b = self.b_min + torch.nn.functional.softplus(self.rb)
        b = torch.where(torch.isnan(self.bfix), b, self.bfix)
        return b[self.key_of]

    def w(self):
        return None if not self.cfg.n_shape else torch.softmax(self.wlog, 1)[self.key_of]

    def pieces(self):
        """(site tensors A (N, 3, 3), bond conductances k (B,)) with the alpha constraint applied."""
        cfg, n = self.cfg, len(self.numbers)
        I = torch.eye(3, dtype=DTYPE)
        free = torch.isnan(self.afix)
        if cfg.aniso:
            L = torch.zeros(n, 3, 3, dtype=DTYPE)
            L[:, [0, 1, 2], [0, 1, 2]] = self.L[:, :3].exp()
            L[:, [1, 2, 2], [0, 0, 1]] = self.L[:, 3:]
            S = L @ L.transpose(1, 2) + 1e-8 * I
        else:
            S = torch.where(free, self.log_a.exp(), torch.nan_to_num(self.afix))[:, None, None] * I
        k = self.lk.exp()
        if cfg.alpha != "exact":
            return S, k
        if cfg.aniso and not self.bonds and bool(free.all()):      # full tensor exact
            K = torch.linalg.cholesky(S.sum(0))
            M = torch.linalg.cholesky(self.alpha_t)
            P = torch.linalg.solve_triangular(K, torch.linalg.solve_triangular(K, S, upper=False).transpose(1, 2),
                                              upper=False).transpose(1, 2)        # K^-1 S K^-T
            return M @ P @ M.T, k
        # isotropic mean exact: one positive scale on all free pieces
        fixed_tr = (torch.diagonal(S, dim1=1, dim2=2).sum(1) * (~free)).sum()
        free_tr = (torch.diagonal(S, dim1=1, dim2=2).sum(1) * free).sum() + (k * (self.d_b ** 2).sum(1)).sum()
        rest = torch.trace(self.alpha_t) - fixed_tr
        if rest <= 0:
            raise ValueError("fixed site polarizabilities exceed the target")
        c = rest / free_tr
        return torch.where(free[:, None, None], c * S, S), c * k

    def site_alpha(self) -> torch.Tensor:
        """(N, 3, 3) site polarizability tensors."""
        return self.pieces()[0]

    def charge_flow_alpha(self):
        k = self.pieces()[1]
        return torch.einsum("b,bx,by->xy", k, self.d_b, self.d_b) if self.bonds else torch.zeros(3, 3, dtype=DTYPE)

    def molecular_alpha(self):
        A, _ = self.pieces()
        return A.sum(0) + self.charge_flow_alpha()

    def dq_dF(self, k=None):
        """(N, 3) induced site charges per unit uniform field."""
        k = self.pieces()[1] if k is None else k
        dq = torch.zeros(len(self.numbers), 3, dtype=DTYPE)
        for (i, j), kb, d in zip(self.bonds, k, self.d_b):
            dq = dq.index_add(0, torch.tensor([i, j]), torch.stack([kb * d, -kb * d]))
        return dq

    def dq_env(self, env_phi):
        """(N,) induced site charges in an environment with potential ``env_phi`` at the sites."""
        k = self.pieces()[1]
        dq = torch.zeros(len(self.numbers), dtype=DTYPE)
        phi = torch.as_tensor(env_phi, dtype=DTYPE)
        for (i, j), kb in zip(self.bonds, k):
            dq = dq.index_add(0, torch.tensor([i, j]), torch.stack([-kb * (phi[i] - phi[j]), kb * (phi[i] - phi[j])]))
        return dq


def _kernels(coords, b, w, points):
    """Damped interaction tensors probe <- site, (P, N, 4, 4): phi = T[..., 0, :] . [q, mu], E = -T[..., 1:, :] . [q, mu]."""
    from ..ff.multipole import damped_interaction_tensor, shaped_one_center_damp, slater_one_center_damp
    P, N = points.shape[0], coords.shape[0]
    dr = (points[:, None, :] - coords[None, :, :]).reshape(-1, 3)
    r = dr.norm(dim=-1)
    bb = b.repeat(P)
    f = slater_one_center_damp(bb * r, 1) if w is None else shaped_one_center_damp(bb * r, w.repeat(P, 1), 1)
    return damped_interaction_tensor(dr, 1.0 - f, 1.0 / r, max_rank=1).reshape(P, N, 4, 4)


def response_from_kernels(T, A, dq=None):
    """d phi/dF (P, 3), d E/dF (P, 3, 3) from site dipoles A[:, :, x] and charges dq[:, x] per unit F_x."""
    phi = torch.einsum("pna,nax->px", T[:, :, 0, 1:], A)
    E = -torch.einsum("pnba,nax->pbx", T[:, :, 1:, 1:], A)
    if dq is not None:
        phi = phi + torch.einsum("pn,nx->px", T[:, :, 0, 0], dq)
        E = E - torch.einsum("pnb,nx->pbx", T[:, :, 1:, 0], dq)
    return phi, E


def response_potential_field(coords, A, b, w, points, taper_ang=None, create_graph=True, dq=None):
    """d phi / dF_x (P, 3) and d E_a / dF_x (P, 3, 3) of site dipoles mu_i = A_i F (uncoupled sites) and site
    charges dq_i = dq[i] . F, with shaped densities (exponents b, shapes w). Exact damped tensors; with a
    taper the field is taken by autograd of the tapered potential."""
    coords = torch.as_tensor(coords, dtype=DTYPE)
    points = torch.as_tensor(points, dtype=DTYPE)
    if taper_ang is None:
        return response_from_kernels(_kernels(coords, b, w, points), A, dq)
    n = coords.shape[0]
    phis, Es = [], []
    for x in range(3):
        q = torch.zeros(n, dtype=DTYPE) if dq is None else dq[:, x]
        sp = F.SiteParams(q, A[:, :, x], torch.zeros(n, 3, 3, dtype=DTYPE), torch.zeros(n, dtype=DTYPE), b, w)
        phi, E = F.potential_field(coords, sp, points, max_rank=1, create_graph=create_graph, taper_ang=taper_ang)
        phis.append(phi)
        Es.append(E)
    return torch.stack(phis, 1), torch.stack(Es, 2)


def _weights(scale, dphi, dE):
    """Each vdW scale contributes equally, relative to its reference RMS (phi and E separately)."""
    sc = np.round(np.asarray(scale, float), 2)
    u = np.unique(sc)
    wp, wE = np.zeros(len(sc)), np.zeros(len(sc))
    for s in u:
        k = sc == s
        wp[k] = 1.0 / (k.sum() * np.mean(dphi[k] ** 2) * len(u))
        wE[k] = 1.0 / (k.sum() * np.mean(dE[k] ** 2) * 3 * len(u))
    return torch.as_tensor(wp, dtype=DTYPE), torch.as_tensor(wE, dtype=DTYPE)


def fit_field_response(rec: dict, numbers, cfg: PolConfig | None = None, coords=None, min_scale=0.0, steps=300,
                       outer=10, b_starts=None):
    """Fit site polarizabilities and induced-density exponents/shapes to a uniform-field response record
    (``pol_ref.field_response``). Multi-started over initial exponents (``b_starts``: scale factors on the
    element prior; default (0.6, 0.9, 1.4)); the lowest loss is kept. Returns (PolFit, metrics)."""
    cfg = cfg or PolConfig()
    X = torch.as_tensor(rec["coords"] if coords is None else coords, dtype=DTYPE)
    sel = np.asarray(rec["scale"]) >= min_scale
    P = torch.as_tensor(rec["points"][sel], dtype=DTYPE)
    phi_ref = torch.as_tensor(rec["dphi_dF"][sel], dtype=DTYPE)
    E_ref = torch.as_tensor(rec["dE_dF"][sel], dtype=DTYPE)
    wp, wE = _weights(rec["scale"][sel], rec["dphi_dF"][sel], rec["dE_dF"][sel])
    a_ref = np.asarray(rec["alpha"], float)
    best = None
    for f in (b_starts or (0.6, 0.9, 1.4)):
        b_init = {z: f * F.ZB_INIT.get(z, (0, 2.0))[1] for z in set(int(z) for z in numbers)}
        mod = PolFit(numbers, cfg, alpha_ref=a_ref, b_init=b_init, coords=X)
        opt = torch.optim.LBFGS([p for p in mod.parameters() if p.requires_grad], lr=1, max_iter=steps,
                                tolerance_grad=1e-14, tolerance_change=1e-16, history_size=50,
                                line_search_fn="strong_wolfe")

        def loss_fn():
            A, k = mod.pieces()
            dq = mod.dq_dF(k) if mod.bonds else None
            phi, E = response_potential_field(X, A, mod.b(), mod.w(), P, cfg.taper, dq=dq)
            L = (wp[:, None] * (phi - phi_ref) ** 2).sum() + cfg.w_field * (wE[:, None, None] * (E - E_ref) ** 2).sum()
            if cfg.alpha == "penalty":
                d = mod.molecular_alpha() - mod.alpha_t
                L = L + cfg.alpha_weight * (d ** 2).sum() / (mod.alpha_t ** 2).sum()
            return L

        def closure():
            opt.zero_grad()
            L = loss_fn()
            L.backward()
            return L

        last = None
        for _ in range(outer):
            L = float(opt.step(closure))
            if last is not None and abs(last - L) <= 1e-10 * max(abs(L), 1e-30):
                break
            last = L
        mod.loss = float(loss_fn().detach())
        if best is None or mod.loss < best.loss:
            best = mod
    return best, response_metrics(rec, best, coords)


def response_metrics(rec, mod: PolFit, coords=None) -> dict:
    """Per vdW scale: RMS of d phi/dF and |d E/dF| errors relative to the reference RMS; model vs
    reference polarizability (and its charge-flow part); fitted site and bond parameters."""
    X = torch.as_tensor(rec["coords"] if coords is None else coords, dtype=DTYPE)
    P = torch.as_tensor(rec["points"], dtype=DTYPE)
    with torch.no_grad():
        A, k = mod.pieces()
        dq = mod.dq_dF(k) if mod.bonds else None
        w = mod.w()
        phi, E = response_potential_field(X, A, mod.b(), w, P, mod.cfg.taper, create_graph=False, dq=dq)
        Am, Acf = mod.molecular_alpha().numpy(), mod.charge_flow_alpha().numpy()
    phi, E = phi.detach().numpy(), E.detach().numpy()
    out = {"scales": {}}
    for s in np.unique(rec["scale"]):
        sel = rec["scale"] == s
        out["scales"][round(float(s), 2)] = {
            "phi_rel": float(np.sqrt(np.mean((phi[sel] - rec["dphi_dF"][sel]) ** 2) / np.mean(rec["dphi_dF"][sel] ** 2))),
            "E_rel": float(np.sqrt(np.mean((E[sel] - rec["dE_dF"][sel]) ** 2) / np.mean(rec["dE_dF"][sel] ** 2)))}
    An = A.numpy()
    out.update(alpha_model=Am.tolist(), alpha_ref=np.asarray(rec["alpha"]).tolist(), alpha_cf=Acf.tolist(),
               alpha_iso_model=float(np.trace(Am) / 3), alpha_iso_ref=float(np.trace(rec["alpha"]) / 3),
               alpha_iso_cf=float(np.trace(Acf) / 3),
               site_alpha=An.tolist(), site_alpha_iso=(np.trace(An, axis1=1, axis2=2) / 3).tolist(),
               bonds=[list(b) for b in mod.bonds], k_bond=k.numpy().tolist(),
               b=mod.b().detach().numpy().tolist(), w=None if w is None else w.detach().numpy().tolist())
    return out


# ---------------------------------------------------------------------------
# environment records: the same sites responding to the environment's field at the nuclei
# ---------------------------------------------------------------------------
def env_prediction(mod: PolFit, coords, env_field, points, env_phi=None, taper_ang=None):
    """d phi, d E at ``points`` from mu_i = A_i E_env(r_i) and, with charge flow, dq_i from the environment's
    potential at the sites (the local-field model, uncoupled sites). Returns (phi, E, mu, dq)."""
    X = torch.as_tensor(coords, dtype=DTYPE)
    with torch.no_grad():
        A, _ = mod.pieces()
        mu = torch.einsum("nab,nb->na", A, torch.as_tensor(env_field, dtype=DTYPE))
        dq = mod.dq_env(env_phi) if (mod.bonds and env_phi is not None) else torch.zeros(len(mu), dtype=DTYPE)
        n = X.shape[0]
        sp = F.SiteParams(dq, mu, torch.zeros(n, 3, 3, dtype=DTYPE), torch.zeros(n, dtype=DTYPE), mod.b(), mod.w())
    phi, E = F.potential_field(X, sp, torch.as_tensor(points, dtype=DTYPE), max_rank=1, create_graph=False,
                               taper_ang=taper_ang)
    return phi.detach().numpy(), E.detach().numpy(), mu.numpy(), dq.numpy()


def env_errors(db, tag, fits: dict, grid_kw=None):
    """For each record of ``db/pol_env/<tag>`` whose monomer has a fit (``fits[name] = PolFit``): relative RMS
    error of the model's induced potential against the full and the linear reference, induced dipole of
    model / linear / full. Rows of dicts."""
    from . import elst_db as D
    from . import pol_ref as R
    rows, load = R.load_records(db, tag)
    out = []
    for r in rows:
        if r["name"] not in fits:
            continue
        x = load(r["id"])
        g = D.monomer_grid(db, r["name"], **(grid_kw or {}))
        phi, _, mu, dq = env_prediction(fits[r["name"]], g["coords"], x["env_field"], g["points"], x["env_phi"])
        rel = lambda ref: float(np.sqrt(np.mean((phi - ref) ** 2) / np.mean(ref ** 2)))
        out.append({"id": r["id"], "name": r["name"], "kind": r["kind"], "partner": r.get("partner"),
                    "err_full": rel(x["dphi"]), "err_lin": rel(x["dphi_lin"]),
                    "dmu_model": float(np.linalg.norm(mu.sum(0) + dq @ np.asarray(g["coords"]))), "dmu_lin": float(np.linalg.norm(x["dmu_lin"])),
                    "dmu_full": float(np.linalg.norm(x["dmu"]))})
    return out
