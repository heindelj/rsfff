"""Joint per-geometry fit of the permanent and the induced electrostatics with shared exponents.

The permanent model is :mod:`elst_fit` (point nucleus Z plus Slater 1s shells carrying [q - Z, mu, Theta]),
the induced model :mod:`pol_fit` (site response matrices, split-charge flow, shaped induced parents). They
share the density exponents: by default each site's induced parent decays with its *outermost* (most diffuse,
valence) permanent shell, ``b_ind = b_perm,outer`` -- a core shell, where there is one, is not polarizable.
``JointConfig.free_ratio`` lets b_ind / b_perm float per element (the anions prefer ~1.1-1.25).

One objective, so the exponent is decided by both data sets at once:

    L = w_perm L_perm(phi, E on the monomer grid) + w_resp L_resp(d phi/dF, dE/dF, d phi/dG, dE/dG on its grid)

each with every vdW shell weighted equally relative to its reference RMS. :func:`fit_joint` starts from the
separate fits (permanent first, then the response tied to the permanent exponents), which are reported as the
"separate" baseline, and then refines everything together.

Units: atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import elst_fit as F
from . import pol_fit as PF

DTYPE = torch.float64


@dataclass
class JointConfig:
    perm: F.FitConfig = field(default_factory=F.FitConfig)
    pol: PF.PolConfig = field(default_factory=PF.PolConfig)
    tie: str = "outer"                   # "outer" (most diffuse permanent shell) | "mean" (weight average)
    free_ratio: bool | tuple = False     # True: free b_ind/b_perm for every element; or a tuple of atomic numbers
    w_perm: float = 1.0
    w_resp: float = 1.0
    perm_weights: str = "relative"       # elst_fit grid weighting
    perm_w_field: float = 1.0
    perm_min_scale: float = 0.0
    resp_min_scale: float = 0.0
    ridge: float = 1e-5
    use_gradient: bool = True


class JointFit(torch.nn.Module):
    def __init__(self, numbers, charge, coords, cfg: JointConfig, alpha_ref=None, perm: F.MonomerFit | None = None,
                 pol: PF.PolFit | None = None):
        super().__init__()
        self.cfg, self.numbers = cfg, [int(z) for z in numbers]
        self.perm = perm or F.MonomerFit(numbers, charge, config=cfg.perm)
        self.pol = pol or PF.PolFit(numbers, cfg.pol, alpha_ref=alpha_ref, coords=coords)
        self.pol.cfg.b_sites = None
        self.elements = sorted(set(self.numbers))
        self.key = [self.elements.index(z) for z in self.numbers]
        free = cfg.free_ratio
        mask = [bool(free is True or (isinstance(free, (tuple, list)) and z in free)) for z in self.elements]
        self.register_buffer("ratio_mask", torch.tensor(mask))
        self.log_ratio = torch.nn.Parameter(torch.zeros(len(self.elements), dtype=DTYPE), requires_grad=any(mask))

    def b_induced(self, sp: F.SiteParams | None = None):
        sp = self.perm.params() if sp is None else sp
        b, sw = sp.shells()
        base = b[:, 0] if self.cfg.tie == "outer" else (b * sw).sum(1)
        r = torch.where(self.ratio_mask, self.log_ratio, torch.zeros_like(self.log_ratio)).exp()
        return base * r[self.key]

    def sync(self):
        """Point the response model at the current permanent exponents; returns the permanent SiteParams."""
        sp = self.perm.params()
        self.pol.b_ext = self.b_induced(sp)
        return sp


def permanent_loss(perm: F.MonomerFit, sp: F.SiteParams, X, P, phi_ref, E_ref, wp, wE, cfg: F.FitConfig, w_field=1.0,
                   ridge=1e-5):
    phi, E = F.potential_field(X, sp, P, cfg.max_rank, cfg.penetration, taper_ang=cfg.taper)
    L = (wp * (phi - phi_ref) ** 2).sum() + w_field * (wE * ((E - E_ref) ** 2).sum(1)).sum()
    return L + ridge * ((sp.mu ** 2).sum() + (sp.theta ** 2).sum())


def fit_joint(grid: dict, rec: dict, numbers, charge, coords, cfg: JointConfig, steps=200, outer=8,
              perm_init: F.MonomerFit | None = None, w_starts=("flat", "outer"), log=None):
    """Separate fits (permanent, then response tied to its exponents), then the joint refinement.
    Returns (JointFit, metrics) with metrics["separate"] and metrics["joint"]."""
    X = torch.as_tensor(np.asarray(coords, float), dtype=DTYPE)
    # 1. permanent alone
    if perm_init is None:
        perm_init, _ = F.fit_monomer(grid, numbers, charge, config=cfg.perm, weights=cfg.perm_weights,
                                     w_field=cfg.perm_w_field, min_scale=cfg.perm_min_scale, ridge=cfg.ridge)
    # 2. response tied to the permanent exponents
    with torch.no_grad():
        jf0 = JointFit(numbers, charge, coords, cfg, alpha_ref=rec["alpha"], perm=perm_init)
        b_tied = jf0.b_induced().numpy().copy()
    pol_cfg = PF.PolConfig(**{**vars(cfg.pol), "b_sites": b_tied})
    pol_init, resp_sep = PF.fit_response(rec, numbers, pol_cfg, coords=coords, use_gradient=cfg.use_gradient,
                                         min_scale=cfg.resp_min_scale, w_starts=w_starts, steps=steps, outer=outer)
    ratio0 = None
    if cfg.free_ratio:
        # start the free ratio from the response's own preferred exponent (the tied start is a local minimum)
        free_cfg = PF.PolConfig(**{**vars(cfg.pol), "b_sites": None})
        pol_free, resp_free = PF.fit_response(rec, numbers, free_cfg, coords=coords, use_gradient=cfg.use_gradient,
                                              min_scale=cfg.resp_min_scale, w_starts=w_starts, steps=steps, outer=outer,
                                              b_starts=(0.8, 1.2))
        if resp_free["loss"] < resp_sep["loss"]:
            ratio0 = np.asarray(resp_free["b"]) / b_tied
            pol_init = pol_free
    sep = {"perm": F.grid_metrics(grid, perm_init.params(), cfg.perm.max_rank, cfg.perm.penetration, cfg.perm.taper),
           "resp": resp_sep}
    # 3. joint
    jf = JointFit(numbers, charge, coords, cfg, alpha_ref=rec["alpha"], perm=perm_init, pol=pol_init)
    if ratio0 is not None:
        with torch.no_grad():
            for e, z in enumerate(jf.elements):
                if bool(jf.ratio_mask[e]):
                    jf.log_ratio[e] = float(np.log(np.mean([ratio0[i] for i, zz in enumerate(jf.numbers) if zz == z])))
    sel = grid["scale"] >= cfg.perm_min_scale
    Pp = torch.as_tensor(grid["points"][sel], dtype=DTYPE)
    phi_ref = torch.as_tensor(grid["phi"][sel], dtype=DTYPE)
    E_ref = torch.as_tensor(grid["field"][sel], dtype=DTYPE)
    wp, wE = F._grid_weights(grid, sel, cfg.perm_weights)
    Pr, sets = PF.response_sets(rec, X, jf.pol.cfg, cfg.use_gradient, cfg.resp_min_scale)
    params = [p for p in jf.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=steps, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=50, line_search_fn="strong_wolfe")

    def loss_fn():
        sp = jf.sync()
        Lp = permanent_loss(jf.perm, sp, X, Pp, phi_ref, E_ref, wp, wE, cfg.perm, cfg.perm_w_field, cfg.ridge)
        Lr = PF.response_loss(jf.pol, sets, jf.pol.kernels(Pr))
        return cfg.w_perm * Lp + cfg.w_resp * Lr, Lp, Lr

    def closure():
        opt.zero_grad()
        L = loss_fn()[0]
        L.backward()
        return L

    last = None
    for _ in range(outer):
        saved = {k: v.detach().clone() for k, v in jf.state_dict().items()}
        L = float(opt.step(closure).detach())
        if not np.isfinite(L) or not all(torch.isfinite(p).all() for p in params):
            jf.load_state_dict(saved)
            break
        if last is not None and abs(last - L) <= 1e-10 * max(abs(L), 1e-30):
            break
        last = L
    L, Lp, Lr = (float(x.detach()) for x in loss_fn())      # (the field is an autograd derivative)
    jf.pol.loss = Lr
    sp = jf.sync()
    joint = {"perm": F.grid_metrics(grid, sp, cfg.perm.max_rank, cfg.perm.penetration, cfg.perm.taper),
             "resp": PF.response_metrics(rec, jf.pol, coords, cfg.use_gradient),
             "b_perm": sp.shells()[0].detach().numpy().tolist(), "b_ind": jf.pol.b().detach().numpy().tolist(),
             "ratio": jf.log_ratio.detach().exp().numpy().tolist(), "loss_perm": float(Lp), "loss_resp": float(Lr)}
    return jf, {"separate": sep, "joint": joint}
