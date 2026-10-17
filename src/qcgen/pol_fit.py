"""Site response parameters fitted to reference response (:mod:`pol_ref`): polarizabilities (dipole,
dipole-quadrupole, quadrupole), split-charge flow with hardness, and the induced-density shapes.

Induced densities
-----------------
Every induced site moment is carried by derivatives of one spherical *parent* density per site,

    g_i(r) = sum_k w_ik N_k r^k e^{-b_i r},     w_ik >= 0,  N_k normalizing r^k e^{-br},

a charge dq_i by g_i itself, a dipole by -mu_i . grad g_i, a quadrupole by the second derivatives. Their
potentials are the one-centre damped multipole terms of the permanent model (:mod:`elst_fit`) with the
site's induced exponent b_i and shape w_i:  phi = T(1 - f_1c(b r; w)) . [dq, mu, Theta].
``n_shape = 0`` is a Slater 1s parent; ``n_shape >= 2`` contains the exact first-order dipole response of the
hydrogen atom: the dipole of g ~ e^{-2r} (3/8 + 3r/4 + r^2/4), i.e. b = 2, w = (1/6, 1/2, 1/3), alpha = 9/2.
The quadrupole response has its own parent (``rank_shapes``, default on with quadrupoles): for hydrogen the
exact l = 2 response (Dalgarno-Lewis, f = -(r^2/2 + r^3/3) for V = r^2 P2) is the second derivative of
g ~ e^{-2r} (7/32 + 7r/16 + r^2/3 + r^3/12), i.e. the same b = 2 and w = (7, 21, 32, 20)/80 on r^0..r^3,
with C = 15 in the convention below (``HYDROGEN_EXACT``). ``b_sites`` ties the induced exponents to given per-site values, e.g. the permanent
shell exponents (:func:`permanent_exponents`).

Site response
-------------
Each site responds to the field E_i and the field gradient at its nucleus. The gradient enters through
its spherical drive g_ik = (1/3) B_k : G_i (B_k = Cartesian form of the k-th spherical quadrupole unit
vector, ``rsfff.ff.multipole.spherical_to_cartesian_quadrupole``), the conjugate of the spherical quadrupole
theta (Theta = sum_k theta_k B_k, Buckingham) since U = -(1/3) Theta : G = -theta . g. With x = [E_i, g_i],

    [mu_i, theta_i] = Pi_i x,        Pi_i = [[alpha_i, A_i], [A_i^T, C_i]]  (8 x 8, positive semidefinite),

so U_i = -1/2 x^T Pi_i x <= 0. ``aniso=False``: alpha_i 1_3 (+ c_i 1_5), A_i = 0 (atom-like sites);
``aniso=True``: Pi_i = L L^T with a free lower-triangular L (3 x 3, or 8 x 8 with quadrupoles: A_i is the
site dipole-quadrupole polarizability). ``quadrupole=False`` keeps only the dipole block.

Charge flow (rsfff.mlip.sqe): split charges p_e on the bonds, q = B p, with on-site hardness eta_i and channel
compliance s_e = 1/kappa_e:  E(p) = sum_i [1/2 eta_i q_i^2 + q_i phi_i] + 1/2 sum_e p_e^2 / s_e, so the linear
response to a potential phi at the sites is

    dq = -B M^-1 B^T phi,      M^-1 = S (L S + 1)^-1,  L = B^T H B,

and in a uniform field alpha_cf = R^T B M^-1 B^T R (PSD). ``hardness="fit"`` fits eta (per site, floored),
"none" sets eta = 0 (independent bonds, dq = -s_e (phi_i - phi_j)), a dict {Z: eta} fixes it per element
(e.g. IP - EA, :func:`hardness_from_ip_ea`). A single-site system has no channels, so its hardness is not
determined by its own response. Bonds from covalent radii (x ``bond_scale``).

Within a small molecule (water, HF, NH3: every pair 1-2 or 1-3) the induced moments do not interact.

The polarizability as a constraint (``alpha``)
----------------------------------------------
* "free"    -- only the grid data;
* "penalty" -- plus ``alpha_weight * |alpha_model - alpha_t|^2 / |alpha_t|^2``;
* "exact"   -- by construction. Anisotropic sites, no charge flow, no fixed sites: the full tensor, by the
  congruence Pi_i -> D Pi_i D^T, D = diag(M K^-1, 1_5), K K^T = sum alpha-blocks, M M^T = alpha_t (every Pi_i
  stays PSD). Otherwise the isotropic mean: the free sites' alpha-blocks are scaled (congruence again)
  so that tr(sum alpha_i + alpha_cf + fixed) = tr(alpha_t); the anisotropy then comes from the data plus,
  if ``alpha_weight > 0``, the penalty.

``alpha_t`` defaults to the reference polarizability of the record; ``alpha_fixed`` pins elements'
isotropic site polarizabilities. The quadrupole analogue is ``quad_weight``: a penalty on the molecular
induced traceless second moments per unit field and per unit field gradient (dQ/dF, dQ/dG about the
centre of nuclear charge), which are not additive over sites (site dipoles and charges contribute).

Data: the uniform-field response (3 columns: d phi/dF_x, dE/dF_x on the grid) and, when present, the
uniform field-gradient response (5 columns, G_k = B_k about the centre of nuclear charge) from
``pol_ref.field_response(..., field_gradients=True)``. Every vdW shell of every column set is weighted
equally relative to its reference RMS.

Units: atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import elst_fit as F
from ..ff.multipole import (
    build_polytensor,
    damped_interaction_tensor,
    shaped_one_center_damp,
    slater_one_center_damp,
    spherical_to_cartesian_quadrupole,
)

DTYPE = torch.float64
HYDROGEN_EXACT = {"alpha": 4.5, "b": 2.0, "w": (1 / 6, 1 / 2, 1 / 3),
                  "c": 15.0, "w_quad": (7 / 80, 21 / 80, 32 / 80, 20 / 80)}
QUAD_BASIS = spherical_to_cartesian_quadrupole(torch.eye(5, dtype=DTYPE))      # (5, 3, 3) B_k
EV = 1.0 / 27.211386245988


def grad_drive(G):
    """Spherical drive of a field gradient: (..., 3, 3) -> (..., 5), g_k = (1/3) B_k : G."""
    return torch.einsum("kab,...ab->...k", QUAD_BASIS, torch.as_tensor(G, dtype=DTYPE)) / 3.0


def reference_c(rec):
    """Isotropic quadrupole polarizability of a spherical atom from its field-gradient response record:
    C = Theta_k / g_k averaged over the five columns (Theta = 3/2 traceless second moment, Buckingham)."""
    Q = np.moveaxis(np.asarray(rec["dQ_dG"], float), -1, 0)
    Qt = Q - np.trace(Q, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    th = np.einsum("kab,cab->ck", np.linalg.pinv(QUAD_BASIS.reshape(5, 9).numpy().T).reshape(5, 3, 3), 1.5 * Qt)
    g = grad_drive(QUAD_BASIS).numpy()                      # (5 columns, 5)
    return float(np.trace(th @ np.linalg.inv(g)) / 5)


def hardness_from_ip_ea(ip_ev, ea_ev):
    """Chemical hardness IP - EA (rsfff.mlip.reference_states convention), eV in -> Hartree out."""
    return (ip_ev - ea_ev) * EV


@dataclass
class PolConfig:
    aniso: bool = False
    quadrupole: bool = False             # quadrupole response (C_i; with aniso also the cross term A_i)
    n_shape: int = 0                     # parent shape functions r^k, k <= n_shape (0: Slater 1s)
    rank_shapes: bool = True             # separate parent shape for the quadrupole response (same exponent)
    rank_exponents: bool = False         # ... and a separate exponent too
    quad_weight: float = 0.0             # penalty on the molecular induced (traceless) second moments, dQ/dF and dQ/dG
    tie_elements: bool = False           # one exponent / shape per element instead of per site
    b_min: float | None = 0.5            # floor on the induced exponents (bohr^-1)
    b_fixed: dict | None = None          # {atomic number: b}
    b_sites: object = None               # (N,) fixed per-site exponents (e.g. the permanent shells')
    alpha: str = "free"                  # "free" | "penalty" | "exact"
    alpha_target: object = None          # (3, 3), scalar, or None (the record's reference alpha)
    alpha_weight: float = 1.0
    alpha_fixed: dict | None = None      # {atomic number: isotropic site alpha} (isotropic sites)
    c_fixed: dict | None = None          # {atomic number: isotropic site quadrupole polarizability} (isotropic sites)
    w_field: float = 1.0                 # weight of dE relative to dphi
    w_gradient_data: float = 1.0         # weight of the field-gradient columns relative to the field columns
    taper: tuple | None = None           # (r_on, r_off) Angstrom on the induced-density penetration
    charge_flow: bool = False            # split charges on the bonds
    hardness: object = "fit"             # "fit" | "none" | {atomic number: eta (Hartree)}
    eta_min: float = 1e-3
    bond_scale: float = 1.25             # bonded if r_ij < bond_scale (r_cov,i + r_cov,j)


def bonds_of(numbers, coords_bohr, scale=1.25):
    """Bonded pairs (i < j) from covalent radii (ase), coordinates in Bohr."""
    from ase.data import covalent_radii
    r = np.array([covalent_radii[int(z)] for z in numbers]) / 0.52917721092
    X = np.asarray(coords_bohr, float)
    return [(i, j) for i in range(len(r)) for j in range(i + 1, len(r))
            if np.linalg.norm(X[i] - X[j]) < scale * (r[i] + r[j])]


def permanent_exponents(params: dict, which="outer"):
    """Per-site exponents of a permanent fit (``elst_fit`` params dict): one shell -> b; several shells ->
    the most diffuse ("outer") or the weight average ("mean")."""
    b = np.asarray(params["b"], float)
    if b.ndim == 1:
        return b
    if which == "outer":
        return b.min(1)
    return (b * np.asarray(params["sw"], float)).sum(1)


def _isp(x):
    x = torch.as_tensor(x, dtype=DTYPE)
    return x + torch.log(-torch.expm1(-x))


def _tril(n_par, D, diag_log):
    """Lower-triangular (N, D, D) from (N, D(D+1)/2) parameters (first D on the diagonal, logged)."""
    N = n_par.shape[0]
    L = torch.zeros(N, D, D, dtype=DTYPE)
    i, j = torch.tril_indices(D, D, offset=-1)
    L[:, range(D), range(D)] = n_par[:, :D].exp() if diag_log else n_par[:, :D]
    L[:, i, j] = n_par[:, D:]
    return L


class PolFit(torch.nn.Module):
    """Response parameters of one monomer under a :class:`PolConfig`."""

    def __init__(self, numbers, cfg: PolConfig, alpha_ref=None, b_init=None, coords=None):
        super().__init__()
        self.cfg, self.numbers = cfg, [int(z) for z in numbers]
        n = len(numbers)
        self.D = 8 if cfg.quadrupole else 3
        X = torch.zeros(n, 3, dtype=DTYPE) if coords is None else torch.as_tensor(np.asarray(coords, float), dtype=DTYPE)
        self.register_buffer("X", X)
        # charge flow graph
        self.bonds = bonds_of(numbers, X.numpy(), cfg.bond_scale) if cfg.charge_flow else []
        Bm = torch.zeros(n, len(self.bonds), dtype=DTYPE)
        for e, (i, j) in enumerate(self.bonds):
            Bm[i, e], Bm[j, e] = 1.0, -1.0
        self.register_buffer("Bm", Bm)
        self.ls = torch.nn.Parameter(torch.full((len(self.bonds),), -1.0, dtype=DTYPE))
        hard = cfg.hardness
        self.register_buffer("eta_fix", torch.tensor(
            [0.0 if hard == "none" else (hard.get(z, np.nan) if isinstance(hard, dict) else np.nan) for z in self.numbers],
            dtype=DTYPE))
        self.leta = torch.nn.Parameter(_isp(torch.full((n,), 0.3, dtype=DTYPE)))
        # alpha target
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
        a0 = max(a_bar, 1e-3) / max(int(free.sum()), 1)
        # site response blocks
        if cfg.aniso:
            P0 = torch.zeros(n, self.D * (self.D + 1) // 2, dtype=DTYPE)
            P0[:, :3] = 0.5 * np.log(a0)
            P0[:, 3:self.D] = 0.5 * np.log(2.0 * a0 ** (4 / 3))   # quadrupole diagonal: c ~ 2 alpha^(4/3) (H: 15 vs 4.5)
            self.Lp = torch.nn.Parameter(P0)
        else:
            self.log_a = torch.nn.Parameter(torch.full((n,), np.log(a0), dtype=DTYPE))
            if cfg.quadrupole:
                self.log_c = torch.nn.Parameter(torch.full((n,), np.log(2.0 * a0 ** (4 / 3)), dtype=DTYPE))
        self.register_buffer("cfix", torch.tensor([(cfg.c_fixed or {}).get(z, np.nan) for z in self.numbers], dtype=DTYPE))
        # exponents and shapes
        keys = sorted(set(self.numbers)) if cfg.tie_elements else list(range(n))
        self.key_of = [keys.index(z) if cfg.tie_elements else i for i, z in enumerate(self.numbers)]
        zkey = keys if cfg.tie_elements else self.numbers
        self.b_min = float(cfg.b_min or 0.0)
        b0 = []
        for z in zkey:
            bf = (cfg.b_fixed or {}).get(int(z))
            b0.append(bf if bf is not None else (b_init.get(int(z)) if isinstance(b_init, dict) else b_init)
                      or F.ZB_INIT.get(int(z), (0, 2.0))[1])
        self.register_buffer("bfix", torch.tensor([(cfg.b_fixed or {}).get(int(z), np.nan) for z in zkey], dtype=DTYPE))
        self.register_buffer("bsite", torch.full((n,), np.nan, dtype=DTYPE) if cfg.b_sites is None
                             else torch.as_tensor(np.asarray(cfg.b_sites, float), dtype=DTYPE))
        self.rb = torch.nn.Parameter(_isp(torch.tensor(b0, dtype=DTYPE).clamp_min(self.b_min + 0.05) - self.b_min),
                                     requires_grad=not bool((~torch.isnan(self.bsite)).all()))
        if cfg.n_shape:
            self.wlog = torch.nn.Parameter(torch.zeros(len(keys), cfg.n_shape + 1, dtype=DTYPE))
            if cfg.quadrupole and cfg.rank_shapes:
                self.wlog_q = torch.nn.Parameter(torch.zeros(len(keys), cfg.n_shape + 1, dtype=DTYPE))
        if cfg.quadrupole and cfg.rank_exponents:
            self.rb_q = torch.nn.Parameter(self.rb.detach().clone(), requires_grad=self.rb.requires_grad)

    # -- parameters ---------------------------------------------------------
    #: Externally supplied induced exponents (N,), e.g. the permanent shells' in a joint fit
    #: (:mod:`rsfff.qcgen.joint_fit`); takes precedence over everything below when set.
    b_ext = None

    def b(self):
        if self.b_ext is not None:
            return self.b_ext
        b = self.b_min + torch.nn.functional.softplus(self.rb)
        b = torch.where(torch.isnan(self.bfix), b, self.bfix)[self.key_of]
        return torch.where(torch.isnan(self.bsite), b, self.bsite)

    def w(self):
        return None if not self.cfg.n_shape else torch.softmax(self.wlog, 1)[self.key_of]

    def w_quad(self):
        if not (self.cfg.n_shape and self.cfg.quadrupole and self.cfg.rank_shapes):
            return self.w()
        return torch.softmax(self.wlog_q, 1)[self.key_of]

    def b_quad(self):
        if not (self.cfg.quadrupole and self.cfg.rank_exponents) or self.b_ext is not None:
            return self.b()
        b = self.b_min + torch.nn.functional.softplus(self.rb_q)
        b = torch.where(torch.isnan(self.bfix), b, self.bfix)[self.key_of]
        return torch.where(torch.isnan(self.bsite), b, self.bsite)

    def kernels(self, points, b=None):
        """Interaction tensors of this model's induced densities at ``points`` (per-rank parents)."""
        rank = 2 if self.cfg.quadrupole else 1
        b = self.b() if b is None else b
        if rank == 2 and self.cfg.rank_exponents:
            T = kernels(self.X, b, self.w(), points, 2)
            Tq = kernels(self.X, self.b_quad(), self.w_quad(), points, 2)
            return torch.cat([T[..., :4], Tq[..., 4:]], -1)
        return kernels(self.X, b, self.w(), points, rank, w_quad=self.w_quad() if rank == 2 else None)

    def second_moments(self, dq, mu, Theta, origin):
        """Molecular traceless second moment (C, 3, 3) of per-column site moments about ``origin``:
        sum_i [dq_i d_i d_i^T + d_i mu_i^T + mu_i d_i^T]_traceless + 2/3 Theta_i (Buckingham Theta_i)."""
        d = self.X - torch.as_tensor(np.asarray(origin, float), dtype=DTYPE)
        M = torch.einsum("cn,na,nb->cab", dq, d, d) + torch.einsum("na,cnb->cab", d, mu) + torch.einsum("cna,nb->cab", mu, d)
        M = M - torch.diagonal(M, dim1=1, dim2=2).sum(-1)[:, None, None] * torch.eye(3, dtype=DTYPE) / 3
        if Theta is not None:
            M = M + 2.0 / 3.0 * Theta.sum(1)
        return M

    def compliance(self):
        return self.ls.exp()

    def hardness(self):
        eta = self.cfg.eta_min + torch.nn.functional.softplus(self.leta)
        return torch.where(torch.isnan(self.eta_fix), eta, self.eta_fix)

    def flow_kernel(self):
        """K = B M^-1 B^T (N, N): dq = -K phi."""
        if not self.bonds:
            return torch.zeros(len(self.numbers), len(self.numbers), dtype=DTYPE)
        S = torch.diag(self.compliance())
        L = self.Bm.T @ torch.diag(self.hardness()) @ self.Bm
        Minv = S @ torch.linalg.inv(L @ S + torch.eye(len(self.bonds), dtype=DTYPE))
        return self.Bm @ Minv @ self.Bm.T

    def charge_flow_alpha(self):
        K = self.flow_kernel()
        a = self.X.T @ K @ self.X
        return 0.5 * (a + a.T)

    def _raw_blocks(self):
        n, D = len(self.numbers), self.D
        if self.cfg.aniso:
            L = _tril(self.Lp, D, True)
            return L @ L.transpose(1, 2) + 1e-10 * torch.eye(D, dtype=DTYPE)
        free = torch.isnan(self.afix)
        a = torch.where(free, self.log_a.exp(), torch.nan_to_num(self.afix))
        Pi = torch.zeros(n, D, D, dtype=DTYPE)
        Pi[:, range(3), range(3)] = a[:, None]
        if self.cfg.quadrupole:
            c = torch.where(torch.isnan(self.cfix), self.log_c.exp(), self.cfix)
            Pi[:, range(3, 8), range(3, 8)] = c[:, None]
        return Pi

    def full_tensor_exact(self):
        return self.cfg.alpha == "exact" and self.cfg.aniso and not self.bonds and bool(torch.isnan(self.afix).all())

    def blocks(self):
        """Site response matrices Pi (N, D, D) with the alpha constraint applied."""
        Pi = self._raw_blocks()
        if self.cfg.alpha != "exact":
            return Pi
        D = self.D
        if self.full_tensor_exact():
            K = torch.linalg.cholesky(Pi[:, :3, :3].sum(0))
            M = torch.linalg.cholesky(self.alpha_t)
            T3 = M @ torch.linalg.inv(K)
            Dm = torch.block_diag(T3, torch.eye(D - 3, dtype=DTYPE)) if D > 3 else T3
            return Dm @ Pi @ Dm.T
        free = torch.isnan(self.afix)
        tr = torch.diagonal(Pi[:, :3, :3], dim1=1, dim2=2).sum(1)
        rest = torch.trace(self.alpha_t) - (tr * ~free).sum() - torch.trace(self.charge_flow_alpha())
        c = rest.clamp_min(1e-6 * torch.trace(self.alpha_t)) / (tr * free).sum()
        d = torch.cat([c.sqrt().repeat(3), torch.ones(D - 3, dtype=DTYPE)])
        Dm = torch.diag(d)
        return torch.where(free[:, None, None], Dm @ Pi @ Dm, Pi)

    def site_alpha(self):
        return self.blocks()[:, :3, :3]

    def molecular_alpha(self):
        return self.site_alpha().sum(0) + self.charge_flow_alpha()

    # -- induced moments ----------------------------------------------------
    def induced(self, E, g, phi, Pi=None):
        """Induced site moments for perturbation columns: E (C, N, 3), g (C, N, 5) or None, phi (C, N).
        Returns dq (C, N), mu (C, N, 3), Theta (C, N, 3, 3) or None."""
        Pi = self.blocks() if Pi is None else Pi
        if self.D == 8:
            x = torch.cat([E, torch.zeros(*E.shape[:2], 5, dtype=DTYPE) if g is None else g], -1)
            y = torch.einsum("nab,cnb->cna", Pi, x)
            mu, th = y[..., :3], y[..., 3:]
            Theta = spherical_to_cartesian_quadrupole(th)
        else:
            mu, Theta = torch.einsum("nab,cnb->cna", Pi, E), None
        dq = -(phi @ self.flow_kernel().T) if self.bonds else torch.zeros(phi.shape, dtype=DTYPE)
        return dq, mu, Theta


# ---------------------------------------------------------------------------
# potentials of induced moments
# ---------------------------------------------------------------------------
def _damped(coords, b, w, points, max_rank):
    P, N = points.shape[0], coords.shape[0]
    dr = (points[:, None, :] - coords[None, :, :]).reshape(-1, 3)
    r = dr.norm(dim=-1)
    bb = b.repeat(P)
    f = slater_one_center_damp(bb * r, max_rank) if w is None else shaped_one_center_damp(bb * r, w.repeat(P, 1), max_rank)
    T = damped_interaction_tensor(dr, 1.0 - f, 1.0 / r, max_rank=max_rank)
    return T.reshape(P, N, T.shape[-2], T.shape[-1])


def kernels(coords, b, w, points, max_rank=2, w_quad=None):
    """Damped interaction tensors probe <- site (P, N, K, K): phi = T[..., 0, :] . m, E = -T[..., 1:4, :] . m.
    Charge and dipole columns use the parent shape ``w``; quadrupole columns ``w_quad`` (default ``w``)."""
    T = _damped(coords, b, w, points, max_rank)
    if max_rank == 2 and w_quad is not None and w_quad is not w:
        Tq = _damped(coords, b, w_quad, points, max_rank)
        T = torch.cat([T[..., :4], Tq[..., 4:]], -1)
    return T


def moments_potential_field(T, dq, mu, Theta=None):
    """phi (P, C), E (P, 3, C) of per-column site moments dq (C, N), mu (C, N, 3), Theta (C, N, 3, 3)."""
    C, N = dq.shape
    rank = 1 if T.shape[-1] == 4 else 2
    m = build_polytensor(dq.reshape(-1), mu.reshape(-1, 3), None if Theta is None else Theta.reshape(-1, 3, 3),
                         max_rank=rank).reshape(C, N, -1)
    phi = torch.einsum("pnk,cnk->pc", T[:, :, 0, :], m)
    E = -torch.einsum("pnak,cnk->pac", T[:, :, 1:4, :], m)
    return phi, E


def response_potential_field(coords, A, b, w, points, taper_ang=None, create_graph=True, dq=None):
    """d phi / dF_x (P, 3) and d E_a / dF_x (P, 3, 3) of site dipoles mu_i = A_i F and site charges
    dq_i = dq[i] . F (uncoupled sites), with shaped densities (exponents b, shapes w). With a taper the field is
    taken by autograd of the tapered potential."""
    coords = torch.as_tensor(coords, dtype=DTYPE)
    points = torch.as_tensor(points, dtype=DTYPE)
    n = coords.shape[0]
    q = torch.zeros(3, n, dtype=DTYPE) if dq is None else dq.T
    mu = A.permute(2, 0, 1)                                   # (3 columns, N, 3)
    if taper_ang is None:
        return moments_potential_field(kernels(coords, b, w, points, 1), q, mu)
    phis, Es = [], []
    for x in range(3):
        sp = F.SiteParams(q[x], mu[x], torch.zeros(n, 3, 3, dtype=DTYPE), torch.zeros(n, dtype=DTYPE), b, w)
        phi, E = F.potential_field(coords, sp, points, max_rank=1, create_graph=create_graph, taper_ang=taper_ang)
        phis.append(phi)
        Es.append(E)
    return torch.stack(phis, 1), torch.stack(Es, 2)


# ---------------------------------------------------------------------------
# perturbation columns
# ---------------------------------------------------------------------------
def field_columns(X):
    """Uniform unit fields along x, y, z: E (3, N, 3), g (3, N, 5), phi (3, N)."""
    n = X.shape[0]
    I = torch.eye(3, dtype=DTYPE)
    return I[:, None, :].expand(3, n, 3), torch.zeros(3, n, 5, dtype=DTYPE), -(X @ I).T


def gradient_columns(X, origin):
    """Uniform field gradients G_k = B_k about ``origin``: E (5, N, 3), g (5, N, 5), phi (5, N)."""
    d = X - torch.as_tensor(np.asarray(origin, float), dtype=DTYPE)
    E = torch.einsum("kab,nb->kna", QUAD_BASIS, d)
    g = grad_drive(QUAD_BASIS)[:, None, :].expand(5, X.shape[0], 5)
    phi = -0.5 * torch.einsum("na,kab,nb->kn", d, QUAD_BASIS, d)
    return E, g, phi


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


def _traceless_ref(rec, tag):
    """Reference traceless second moments per column (C, 3, 3), or None."""
    key = "dQ_dF" if tag == "F" else "dQ_dG"
    if key not in rec:
        return None
    Q = torch.as_tensor(np.moveaxis(np.asarray(rec[key], float), -1, 0), dtype=DTYPE)
    return Q - torch.diagonal(Q, dim1=1, dim2=2).sum(-1)[:, None, None] * torch.eye(3, dtype=DTYPE) / 3


def _column_sets(rec, X, sel, use_gradient):
    sets = [("F", field_columns(X), rec["dphi_dF"][sel], rec["dE_dF"][sel])]
    if use_gradient and "dphi_dG" in rec:
        sets.append(("G", gradient_columns(X, rec["gradient_origin"]), rec["dphi_dG"][sel], rec["dE_dG"][sel]))
    return sets


def response_sets(rec, X, cfg: PolConfig, use_gradient=True, min_scale=0.0):
    """Points (P, 3) and the perturbation column sets of a response record that carry signal:
    [(columns, phi_ref, E_ref, w_phi, w_E, Q_ref, origin)]."""
    sel = np.asarray(rec["scale"]) >= min_scale
    P = torch.as_tensor(rec["points"][sel], dtype=DTYPE)
    sets = []
    for tag, cols, phi_ref, E_ref in _column_sets(rec, X, sel, use_gradient):
        if np.sqrt(np.mean(phi_ref ** 2)) < 1e-10:        # no response in this channel (e.g. no l = 2 basis functions)
            continue
        wp, wE = _weights(rec["scale"][sel], phi_ref, E_ref)
        wt = 1.0 if tag == "F" else cfg.w_gradient_data
        Qref = _traceless_ref(rec, tag)
        sets.append((cols, torch.as_tensor(phi_ref, dtype=DTYPE), torch.as_tensor(E_ref, dtype=DTYPE), wt * wp, wt * wE,
                     Qref, rec.get("origin") if tag == "F" else rec.get("gradient_origin")))
    return P, sets


def response_loss(mod: "PolFit", sets, T):
    """Weighted squared error of the induced potential and field over the column sets, plus the molecular
    second-moment and polarizability penalties of ``mod.cfg``."""
    cfg = mod.cfg
    Pi = mod.blocks()
    L = 0.0
    for (E, g, ph), phi_ref, E_ref, wp, wE, Qref, org in sets:
        dq, mu, Th = mod.induced(E, g, ph, Pi)
        phi, Ef = moments_potential_field(T, dq, mu, Th)
        L = L + (wp[:, None] * (phi - phi_ref) ** 2).sum() + cfg.w_field * (wE[:, None, None] * (Ef - E_ref) ** 2).sum()
        if cfg.quad_weight and Qref is not None and (Qref ** 2).sum() > 1e-12:
            L = L + cfg.quad_weight * ((mod.second_moments(dq, mu, Th, org) - Qref) ** 2).sum() / (Qref ** 2).sum()
    if cfg.alpha == "penalty" or (cfg.alpha == "exact" and not mod.full_tensor_exact() and cfg.alpha_weight > 0):
        d = mod.molecular_alpha() - mod.alpha_t
        L = L + cfg.alpha_weight * (d ** 2).sum() / (mod.alpha_t ** 2).sum()
    return L


def fit_response(rec: dict, numbers, cfg: PolConfig | None = None, coords=None, min_scale=0.0, steps=300,
                 outer=10, b_starts=None, use_gradient=True, w_starts=("flat",)):
    """Fit the response parameters to a ``pol_ref.field_response`` record (field columns, and the field-gradient
    columns when present and ``use_gradient``). Multi-started over initial exponents (``b_starts``: factors on
    the element prior; default (0.6, 0.9, 1.4); one start when the exponents are tied) and initial shapes
    (``w_starts``: "flat", "inner" (r^0-heavy), "outer" (r^n-heavy)); the lowest loss is kept. Returns (PolFit, metrics)."""
    cfg = cfg or PolConfig()
    X = torch.as_tensor(rec["coords"] if coords is None else coords, dtype=DTYPE)
    P, sets = response_sets(rec, X, cfg, use_gradient, min_scale)
    a_ref = np.asarray(rec["alpha"], float)
    rank = 2 if cfg.quadrupole else 1
    tied = cfg.b_sites is not None
    best = None
    starts = [(f, ws) for f in ((1.0,) if tied else (b_starts or (0.6, 0.9, 1.4)))
              for ws in (w_starts if cfg.n_shape else ("flat",))]
    for f, ws in starts:
        b_init = {z: f * F.ZB_INIT.get(z, (0, 2.0))[1] for z in set(int(z) for z in numbers)}
        mod = PolFit(numbers, cfg, alpha_ref=a_ref, b_init=b_init, coords=X)
        if cfg.n_shape and ws != "flat":
            with torch.no_grad():
                v = torch.zeros(cfg.n_shape + 1, dtype=DTYPE)
                v[0 if ws == "inner" else -1] = 2.0
                mod.wlog.copy_(v.expand_as(mod.wlog))
                if hasattr(mod, "wlog_q"):
                    mod.wlog_q.copy_(v.expand_as(mod.wlog_q))
        T_fixed = mod.kernels(P).detach() if (tied and not cfg.n_shape) else None
        opt = torch.optim.LBFGS([p for p in mod.parameters() if p.requires_grad], lr=1, max_iter=steps,
                                tolerance_grad=1e-14, tolerance_change=1e-16, history_size=50,
                                line_search_fn="strong_wolfe")

        def loss_fn():
            return response_loss(mod, sets, T_fixed if T_fixed is not None else mod.kernels(P))

        def closure():
            opt.zero_grad()
            L = loss_fn()
            L.backward()
            return L

        last = None
        for _ in range(outer):
            saved = {k: v.detach().clone() for k, v in mod.state_dict().items()}
            L = float(opt.step(closure).detach())
            if not np.isfinite(L) or not all(torch.isfinite(p).all() for p in mod.parameters()):
                mod.load_state_dict(saved)          # a line search ran off; keep the last finite point
                break
            if last is not None and abs(last - L) <= 1e-10 * max(abs(L), 1e-30):
                break
            last = L
        mod.loss = float(loss_fn().detach())
        if not np.isfinite(mod.loss):
            continue
        if best is None or mod.loss < best.loss:
            best = mod
    return best, response_metrics(rec, best, coords, use_gradient)


fit_field_response = fit_response


def response_metrics(rec, mod: PolFit, coords=None, use_gradient=True) -> dict:
    """Per column set and vdW scale: RMS of d phi and |d E| errors relative to the reference RMS; molecular
    polarizability (and its charge-flow part); fitted site, bond and shape parameters."""
    X = torch.as_tensor(rec["coords"] if coords is None else coords, dtype=DTYPE)
    P = torch.as_tensor(rec["points"], dtype=DTYPE)
    sel = np.ones(len(P), bool)
    out = {"scales": {}, "scales_G": {}}
    with torch.no_grad():
        Pi = mod.blocks()
        T = mod.kernels(P)
        for tag, (E, g, ph), phi_ref, E_ref in _column_sets(rec, X, sel, True):
            if np.sqrt(np.mean(phi_ref ** 2)) < 1e-10:
                continue
            dq, mu, Th = mod.induced(E, g, ph, Pi)
            phi, Ef = moments_potential_field(T, dq, mu, Th)
            phi, Ef = phi.numpy(), Ef.numpy()
            key = "scales" if tag == "F" else "scales_G"
            Qref = _traceless_ref(rec, tag)
            if Qref is not None and float((Qref ** 2).sum()) > 1e-12:
                Qm = mod.second_moments(dq, mu, Th, rec["origin"] if tag == "F" else rec["gradient_origin"])
                out["quad_rel_" + tag] = float(((Qm - Qref) ** 2).sum().sqrt() / (Qref ** 2).sum().sqrt())
            for s in np.unique(rec["scale"]):
                k = rec["scale"] == s
                out[key][round(float(s), 2)] = {
                    "phi_rel": float(np.sqrt(np.mean((phi[k] - phi_ref[k]) ** 2) / np.mean(phi_ref[k] ** 2))),
                    "E_rel": float(np.sqrt(np.mean((Ef[k] - E_ref[k]) ** 2) / np.mean(E_ref[k] ** 2)))}
        Am, Acf = mod.molecular_alpha().numpy(), mod.charge_flow_alpha().numpy()
        An = Pi[:, :3, :3].numpy()
        Cn = Pi[:, 3:, 3:].numpy() if mod.D == 8 else None
        w = mod.w()
        out.update(alpha_model=Am.tolist(), alpha_ref=np.asarray(rec["alpha"]).tolist(), alpha_cf=Acf.tolist(),
                   alpha_iso_model=float(np.trace(Am) / 3), alpha_iso_ref=float(np.trace(rec["alpha"]) / 3),
                   alpha_iso_cf=float(np.trace(Acf) / 3), site_alpha=An.tolist(),
                   site_alpha_iso=(np.trace(An, axis1=1, axis2=2) / 3).tolist(),
                   site_c_iso=None if Cn is None else (np.trace(Cn, axis1=1, axis2=2) / 5).tolist(),
                   site_A_norm=None if mod.D != 8 else np.linalg.norm(Pi[:, :3, 3:].numpy(), axis=(1, 2)).tolist(),
                   bonds=[list(b) for b in mod.bonds], compliance=mod.compliance().numpy().tolist(),
                   hardness=mod.hardness().numpy().tolist() if mod.bonds else None,
                   b=mod.b().numpy().tolist(), w=None if w is None else w.numpy().tolist(),
                   w_quad=None if (w is None or mod.D != 8) else mod.w_quad().numpy().tolist(), loss=mod.loss)
    return out


# ---------------------------------------------------------------------------
# environment records: the local-response model in the environment's field, gradient and potential
# ---------------------------------------------------------------------------
def env_prediction(mod: PolFit, coords, env_field, points, env_phi=None, env_grad=None, taper_ang=None):
    """d phi, d E at ``points`` from site moments responding to the environment's field (and gradient,
    potential) at the nuclei (uncoupled sites). Returns (phi, E, mu, dq)."""
    X = torch.as_tensor(coords, dtype=DTYPE)
    P = torch.as_tensor(points, dtype=DTYPE)
    with torch.no_grad():
        E = torch.as_tensor(env_field, dtype=DTYPE)[None]
        g = None if env_grad is None else grad_drive(env_grad)[None]
        ph = torch.zeros(1, X.shape[0], dtype=DTYPE) if env_phi is None else torch.as_tensor(env_phi, dtype=DTYPE)[None]
        dq, mu, Th = mod.induced(E, g, ph)
        T = mod.kernels(P)
        phi, Ef = moments_potential_field(T, dq, mu, Th)
    return phi[:, 0].numpy(), Ef[..., 0].numpy(), mu[0].numpy(), dq[0].numpy()


def env_errors(db, tag, fits: dict, grid_kw=None, use_gradient=True):
    """For each record of ``db/pol_env/<tag>`` whose monomer has a fit (``fits[name] = PolFit``): relative RMS
    error of the model's induced potential against the full and the linear reference; induced dipoles. Rows."""
    from . import elst_db as D
    from . import pol_ref as R
    rows, load = R.load_records(db, tag)
    out = []
    for r in rows:
        if r["name"] not in fits:
            continue
        x = load(r["id"])
        g = D.monomer_grid(db, r["name"], **(grid_kw or {}))
        phi, _, mu, dq = env_prediction(fits[r["name"]], g["coords"], x["env_field"], g["points"], x["env_phi"],
                                        x["env_grad"] if use_gradient else None)
        rel = lambda ref: float(np.sqrt(np.mean((phi - ref) ** 2) / np.mean(ref ** 2)))
        out.append({"id": r["id"], "name": r["name"], "kind": r["kind"], "partner": r.get("partner"),
                    "err_full": rel(x["dphi"]), "err_lin": rel(x["dphi_lin"]),
                    "dmu_model": float(np.linalg.norm(mu.sum(0) + dq @ np.asarray(g["coords"]))),
                    "dmu_lin": float(np.linalg.norm(x["dmu_lin"])), "dmu_full": float(np.linalg.norm(x["dmu"]))})
    return out
