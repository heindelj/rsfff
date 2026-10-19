"""A parameter network for the permanent *and* induced electrostatics, trained on reference densities.

Replaces the one-off per-geometry fits of :mod:`rsfff.qcgen.elst_fit` / :mod:`rsfff.qcgen.pol_fit` /
:mod:`rsfff.qcgen.joint_fit` with heads that emit the same parameters from the monomer's own descriptor:

* **response solve** -- :class:`rsfff.ff.response.FragmentResponse` in the direct parameterization: SQE charges
  (chi, eta, channel compliance, baseline q0), permanent atomic dipoles and quadrupoles, the atomic dipole
  polarizability alpha_i (PSD) and the isotropic quadrupole polarizability c_i. The *same* eta and compliances
  that fix the permanent charges set the induced charge flow, so the two are tied through the solve itself;
* **shells** (:class:`ShellHeads`) -- the permanent densities: per element a nuclear charge Z (fixed or learned)
  and ``n_shells`` Slater 1s shells with ordered exponents b (per-element prior + zero-initialized environment
  correction) and positive weights;
* **induced densities** -- per element the shapes (r^0..r^n_shape weights) of the dipole and quadrupole parents;
  their exponent is the outermost permanent shell's, b_ind = b_perm,outer (x a per-element ratio, fixed at 1
  unless ``free_ratio``).

The physics that turns parameters into observables is the per-geometry fits' (rsfff.qcgen), so the loss is the
same one those fits minimized: the potential and field of the permanent model on the monomer's grid, and the
induced potential and field per unit field (and field gradient) on its response grid, each vdW shell weighted
equally relative to its reference RMS, plus the molecular polarizability.

Units at the boundary: the featurizer and the SQE positions are in Angstrom (rsfff convention); everything the
physics reads -- coordinates, grid points, multipoles, polarizabilities -- is atomic units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from bb.features import FlatLambdaSOAPFeaturizer
from ..mlip.heads import two_slot_mlp, zero_init_readout
from ..mlip.response_heads import voigt_vector_to_symmetric_matrix  # noqa: F401  (re-export for callers)
from ..mlip.sqe import PairComplianceHead
from .multipole import irrep2_to_spherical, spherical_to_cartesian_quadrupole
from .pairs import intra_fragment_channels
from .response import ElectrostaticParameterHeads, FragmentResponse, build_elec_priors

BOHR_ANG = 0.52917721092


@dataclass
class RefElecConfig:
    neighbor_types: tuple = (1, 3, 7, 8, 9, 11, 17, 19, 35)
    cutoff: float = 5.0
    n_max: int = 4
    l_max: int = 3
    density_channels: int = 8
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2
    equiv_channels: int = 32
    n_shells: dict = field(default_factory=lambda: {9: 2, 17: 2, 35: 2})
    #: fixed nuclear charges by element; elements not listed are learned (per element, from ``z_init``)
    z_fixed: dict = field(default_factory=lambda: {9: 7.0, 17: 11.0, 35: 12.0})
    z_init: dict = field(default_factory=dict)
    b_init: dict = field(default_factory=dict)        # {Z: [b per shell, diffuse first]}
    b_min: float = 1.2                                # floor on every permanent exponent
    environment_b: bool = True
    n_shape: int = 3
    free_ratio: tuple = ()                            # elements whose b_ind/b_perm is learned
    s_init: float = 0.5
    eta_floor: float = 0.05
    #: per-element site polarizability priors (a0^3): the alpha head's output multiplies prior / ln 2, so it
    #: starts there and learns a relative correction (rsfff.ff.response.alpha_scale_prior)
    alpha_prior: dict = field(default_factory=lambda: {1: 0.6, 3: 0.19, 7: 8.0, 8: 6.5, 9: 7.0, 11: 0.96,
                                                       17: 31.3, 19: 5.36, 35: 43.3})
    #: per-element isotropic quadrupole polarizability priors (theta = c g convention of rsfff.qcgen.pol_fit)
    cquad_prior: dict = field(default_factory=lambda: {1: 4.0, 3: 0.05, 7: 40.0, 8: 21.0, 9: 30.0, 11: 0.93,
                                                       17: 208.0, 19: 7.6, 35: 293.0})


def _isp(x):
    return math.log(math.expm1(max(float(x), 1e-6)))


class ShellHeads(nn.Module):
    """Per-atom permanent shells (Z, b (N, S), weights (N, S)) and induced-density shapes."""

    def __init__(self, cfg: RefElecConfig, p0: int):
        super().__init__()
        types = [int(z) for z in cfg.neighbor_types]
        n = len(types)
        self.S = max([cfg.n_shells.get(z, 1) for z in types])
        self.b_min = float(cfg.b_min)
        ns = torch.tensor([cfg.n_shells.get(z, 1) for z in types])
        self.register_buffer("mask", torch.arange(self.S)[None, :] < ns[:, None])
        zf = torch.tensor([cfg.z_fixed.get(z, float("nan")) for z in types])
        self.register_buffer("z_fixed", zf)
        from ..qcgen.elst_fit import ZB_INIT
        z0 = [cfg.z_init.get(z, ZB_INIT.get(z, (1.0, 2.0))[0]) for z in types]
        self.log_z = nn.Parameter(torch.tensor(z0, dtype=torch.float32).log())
        # ordered exponents: b_1 = b_min + softplus(r_1), b_s = b_{s-1} + softplus(r_s)
        r0 = torch.zeros(n, self.S)
        for k, z in enumerate(types):
            bs = list(cfg.b_init.get(z, [ZB_INIT.get(z, (1.0, 2.0))[1] * (1.0 + 0.8 * s) for s in range(self.S)]))
            bs = (bs + [bs[-1] * 1.6] * self.S)[: self.S]
            prev = self.b_min
            for s in range(self.S):
                r0[k, s] = _isp(max(bs[s], prev + 0.05) - prev)
                prev = max(bs[s], prev + 0.05)
        self.r_b = nn.Parameter(r0)
        self.sw_logit = nn.Parameter(torch.zeros(n, self.S))
        self.emb = nn.Embedding(n, cfg.emb_dim)
        self.b_mlp = two_slot_mlp(p0, 0, cfg.hidden, cfg.depth, 2 * self.S, p_tail=cfg.emb_dim) if cfg.environment_b else None
        if self.b_mlp is not None:
            zero_init_readout(self.b_mlp)
        K = cfg.n_shape + 1
        self.w_dip = nn.Parameter(torch.zeros(n, K))
        self.w_quad = nn.Parameter(torch.zeros(n, K))
        ratio_mask = torch.tensor([z in cfg.free_ratio for z in types])
        self.register_buffer("ratio_mask", ratio_mask)
        self.log_ratio = nn.Parameter(torch.zeros(n))

    def forward(self, feats):
        s = feats.species_idx
        r = self.r_b[s]
        logit = self.sw_logit[s]
        if self.b_mlp is not None:
            d = self.b_mlp(torch.cat((feats.inv_feats, self.emb(s)), -1))
            r, logit = r + d[:, : self.S], logit + d[:, self.S:]
        b = self.b_min + torch.cumsum(nn.functional.softplus(r), 1)
        sw = torch.softmax(logit.masked_fill(~self.mask[s], -torch.inf), 1)
        z = torch.where(torch.isnan(self.z_fixed[s]), self.log_z[s].exp(), self.z_fixed[s])
        ratio = torch.where(self.ratio_mask[s], self.log_ratio[s], torch.zeros_like(self.log_ratio[s])).exp()
        return {"Z": z, "b": b, "sw": sw, "b_ind": b[:, 0] * ratio,
                "w_dip": torch.softmax(self.w_dip[s], 1), "w_quad": torch.softmax(self.w_quad[s], 1)}


class RefElecModel(nn.Module):
    def __init__(self, cfg: RefElecConfig):
        super().__init__()
        self.cfg = cfg
        types = tuple(int(z) for z in cfg.neighbor_types)
        self.featurizer = FlatLambdaSOAPFeaturizer(
            cutoff=cfg.cutoff, n_max=cfg.n_max, l_max=cfg.l_max, neighbor_types=types,
            selected_lambdas=(0, 1, 2), backend="e3nn", density_channels=cfg.density_channels)
        p0 = self.featurizer.feature_dims[0]
        p1, p2 = self.featurizer.feature_dims.get(1), self.featurizer.feature_dims.get(2)
        from ..qcgen.elst_fit import ZB_INIT
        log_z, log_b, q0 = build_elec_priors(
            types, z_prior={z: ZB_INIT.get(z, (1.0, 2.0))[0] for z in types},
            b_prior={z: ZB_INIT.get(z, (1.0, 2.0))[1] for z in types},
            q0_prior={3: 1.0, 19: 1.0, 35: -1.0})
        i6 = self.featurizer.backend.irrep6_to_voigt()
        heads = ElectrostaticParameterHeads(
            p0, p1, p2, len(types), log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0, irrep6_to_voigt=i6,
            irrep2_to_spherical_map=irrep2_to_spherical(i6), emb_dim=cfg.emb_dim, hidden=cfg.hidden, depth=cfg.depth,
            equiv_channels=cfg.equiv_channels, max_rank=2, eta_floor=cfg.eta_floor, learn_z=False, learn_b=False,
            quadrupole_response=True, cquad_init=5.0, environment_cquad=True)
        with torch.no_grad():
            c0 = torch.tensor([cfg.cquad_prior.get(z, 5.0) for z in types])
            heads.cquad0_raw.copy_(torch.log(torch.expm1((c0 - heads.cquad_floor).clamp_min(1e-6))))
        self.response = FragmentResponse(
            heads, PairComplianceHead(p0, cutoff=cfg.cutoff, s_init=cfg.s_init), direct_multipoles=True)
        self.shells = ShellHeads(cfg, p0)
        self.register_buffer("alpha_scale", torch.tensor(
            [cfg.alpha_prior.get(z, math.log(2.0)) / math.log(2.0) for z in types]))

    def forward(self, batch):
        """Per-atom parameters for a flat batch of isolated monomers (one fragment per frame)."""
        feats = self.featurizer(batch, batch.fragment_idx)
        rp = self.response.response_parameters(batch, feats)
        out = self.response(batch, feats, with_polarizability=False)
        sh = self.shells(feats)
        bond_index, _ = intra_fragment_channels(batch.fragment_idx)
        theta = spherical_to_cartesian_quadrupole(out.quad_s)
        alpha = rp.alpha * self.alpha_scale[feats.species_idx][:, None, None]
        return {"q": out.charges, "mu": out.mu, "theta": theta, "alpha": alpha, "cquad": rp.cquad,
                "eta": rp.eta, "compliance": rp.compliance, "bond_index": bond_index, **sh}


# ---------------------------------------------------------------------------
# batches and physics
# ---------------------------------------------------------------------------
def make_batch(monomers, device="cpu", dtype=torch.float32):
    """A flat batch (rsfff conventions, positions in Angstrom) from a list of dicts with ``numbers``,
    ``coords`` (Bohr) and ``charge``."""
    pos, Z, bidx = [], [], []
    for k, m in enumerate(monomers):
        pos.append(torch.as_tensor(np.asarray(m["coords"]) * BOHR_ANG, dtype=dtype))
        Z.append(torch.as_tensor(m["numbers"], dtype=torch.long))
        bidx.append(torch.full((len(m["numbers"]),), k, dtype=torch.long))
    bidx = torch.cat(bidx)
    return SimpleNamespace(positions=torch.cat(pos).to(device), atomic_numbers=torch.cat(Z).to(device),
                           batch_idx=bidx.to(device), n_systems=len(monomers), fragment_idx=bidx.to(device),
                           n_fragments=len(monomers),
                           fragment_charge=torch.tensor([float(m["charge"]) for m in monomers], dtype=dtype, device=device),
                           fragment_to_batch=torch.arange(len(monomers), device=device))


def split_params(p, batch):
    """Per-monomer views of the per-atom parameters (and each monomer's channels, renumbered locally)."""
    out = []
    starts = torch.cumsum(torch.bincount(batch.batch_idx, minlength=batch.n_systems), 0) - \
        torch.bincount(batch.batch_idx, minlength=batch.n_systems)
    bi = p["bond_index"]
    for k in range(batch.n_systems):
        sel = batch.batch_idx == k
        bsel = sel[bi[0]]
        d = {key: v[sel] for key, v in p.items() if key not in ("bond_index", "compliance")}
        d["bond_index"] = bi[:, bsel] - starts[k]
        d["compliance"] = p["compliance"][bsel]
        out.append(d)
    return out


def flow_kernel(eta, compliance, bond_index, n):
    """K (n, n) with dq = -K phi: the SQE linear response, B S (L S + 1)^-1 B^T, L = B^T H B."""
    if bond_index.shape[1] == 0:
        return eta.new_zeros(n, n)
    B = eta.new_zeros(n, bond_index.shape[1])
    e = torch.arange(bond_index.shape[1])
    B[bond_index[0], e] = 1.0
    B[bond_index[1], e] = -1.0
    S = torch.diag(compliance)
    L = B.T @ torch.diag(eta) @ B
    return B @ S @ torch.linalg.solve(L @ S + torch.eye(len(e), dtype=eta.dtype), B.T)
