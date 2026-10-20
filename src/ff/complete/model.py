"""C2: the generic model on the field / overlap environment (complete model plan).

Everything the ion CT model (R11, direct CT form) does, with its per-ion network outputs replaced by environment
heads (:class:`EnvironmentHead`: isolated-atom parameter times a factor that is exactly 1 for an empty environment)
and two equivariant pieces that the invariant-only model cannot express:

* an anisotropic polarizability from the overlap quadrupole of the atom's own dipole parent,
      alpha_i^-1 = (c_alpha,i alpha0)^-1 expm( kappa_s S[dip]^(2)_i / s2 ),
  a symmetric positive-definite matrix with det = (c alpha0)^-3 (the traceless exponent moves polarizability between
  the bond axis and the perpendicular plane without changing its geometric mean; kappa = 0 is the isotropic model);

* the pair gap factor on the direct CT: E_ij = -qD_i qA_j f(bD_i, bA_j; r)/r exp(-(lamD_i + lamA_j) dphi_ij / s),
  dphi_ij = phibar_D(i) - phibar_A(j) the difference of the potentials (permanent + induced + external) of all the
  OTHER atoms over the donor frontier density of i and the acceptor frontier density of j. For a dimer dphi = 0 (the
  pair kernel carries the pair), so this is the three-body shift of the CT gap by the environment; lam = 0 at start.

Species enter only through the isolated-atom parameters (alpha0, C0, the CT charges qD0 / qA0 and exponents, kappa,
lam) and the pseudo-potential pair types; the heads are shared (``species_emb`` adds an embedding to their input).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..ionct import kernels as K
from ..ionct.model import CHANNEL_TYPES, IonCTModel
from ..ionct.params import IonTable
from ..ionct.ct import ChannelTable
from ..multipole import slater_two_center_damp_unequal
from .heads import EnvironmentHead

__all__ = ["CompleteModel", "solve_induction_blocks"]

GAP_EXP_MAX = 4.0     # bound on the log of the CT gap factor


def solve_induction_blocks(L, A_off, alpha_inv, C, quadrupoles=True):
    """Like :func:`rsfff.ff.ionct.kernels.solve_induction` with a 3x3 inverse polarizability per site:
    L (B, N, 8), A_off (B, 8N, 8N), alpha_inv (B, N, 3, 3) symmetric positive definite, C (B, N)."""
    B, N, _ = L.shape
    cinv = (1.0 / C) if quadrupoles else torch.full_like(C, 1e12)
    D = L.new_zeros(B, N, 8, 8)
    D[:, :, :3, :3] = alpha_inv
    D[:, :, 3:, 3:] = torch.diag_embed(cinv[..., None].expand(B, N, 5))
    ar = torch.arange(N, device=L.device)
    Ad = L.new_zeros(B, N, 8, N, 8)
    Ad[:, ar, :, ar, :] = D.permute(1, 0, 2, 3)
    A = A_off + Ad.reshape(B, 8 * N, 8 * N)
    if quadrupoles:
        return -torch.linalg.solve(A, L.reshape(B, -1, 1)).reshape(B, N, 8)
    keep = torch.zeros(N, 8, dtype=torch.bool, device=L.device)
    keep[:, :3] = True
    keep = keep.reshape(-1).nonzero().squeeze(1)
    xs = -torch.linalg.solve(A[:, keep][:, :, keep], L.reshape(B, -1)[:, keep, None])[..., 0]
    return L.new_zeros(B, N * 8).index_copy(1, keep, xs).reshape(B, N, 8)


class CompleteModel(IonCTModel):
    """``feature_names`` (from the featurizer) locate the dipole-overlap-quadrupole scale; ``env_keys`` says which
    equivariant blocks the caches carry (``aniso`` needs ``"dip:S2"``)."""

    def __init__(self, tab: IonTable, chan: ChannelTable, n_feat: int, *, feature_names=None, hidden=64, depth=2,
                 feat_scale=None, c_range=(0.05, 1.25), confine=None, pseudo=True, ps_init=0.02, tie_c=True,
                 species_emb=False, aniso=True, ct_gap=True, dphi_scale=0.05):
        super().__init__(tab, chan, n_feat, hidden=hidden, depth=depth, feat_scale=feat_scale, c_range=c_range,
                         confine=confine, pseudo=pseudo, ps_init=ps_init, tie_c=tie_c, ct_form="direct",
                         gap_form="pair", pol_form="confine")
        del self.conf, self.ctq, self.gap                  # replaced by the environment heads / not used (direct CT)
        S = len(tab.symbols)
        ns = S if species_emb else 0
        self.head_conf = EnvironmentHead(n_feat, 2, hidden, depth, ns)     # log c_alpha, log c_C
        self.head_ctq = EnvironmentHead(n_feat, 2, hidden, depth, ns)      # log qD / qD0, log qA / qA0
        self.ct_lnq0 = nn.Parameter(torch.zeros(S, 2, dtype=torch.float64))
        self.aniso, self.ct_gap, self.dphi_scale = aniso, ct_gap, dphi_scale
        names = list(feature_names or [])
        s2 = 1.0
        if aniso:
            self.kappa_aniso = nn.Parameter(torch.zeros(S, dtype=torch.float64))
            if "dip:|S2|" in names and feat_scale is not None:
                s2 = float(feat_scale[names.index("dip:|S2|")])
        self.register_buffer("s2_scale", torch.tensor(s2, dtype=torch.float64))
        if ct_gap:
            self.gap_lamD = nn.Parameter(torch.zeros(S, dtype=torch.float64))
            self.gap_lamA = nn.Parameter(torch.zeros(S, dtype=torch.float64))
        self.double()

    # -- heads --
    def scaled(self, feats):
        return feats / self.feat_scale

    def confinement_from(self, I, sp):
        """(N, 2) scales on alpha0 and C0 (c_range, tanh-saturated like the ion CT model), 1 for an isolated atom."""
        u = self.head_conf(I, sp) * self.confine_mask[sp][:, None]
        lo, hi = self.log_c_lo, self.log_c_hi
        logc = torch.where(u > 0, hi * torch.tanh(u / hi), lo * torch.tanh(u / lo))
        if self.tie_c:
            logc = torch.stack([logc[:, 0], logc[:, 0] * 5.0 / 3.0], 1)
        return logc.exp()

    def ct_charges_from(self, I, sp):
        return torch.exp(self.ct_lnq0[sp] + self.head_ctq(I, sp))

    def confinement(self, I, h0=None, species=None):
        """Same entry point as the ion CT model (the pair-scan attribution overrides it), on the scaled invariants."""
        return self.confinement_from(I, species)

    def ct_charges(self, h):
        raise NotImplementedError("CompleteModel: use ct_charges_from(I, species)")

    def alpha_inverse(self, alpha, sp, env):
        """(N, 3, 3) inverse polarizability: isotropic 1/alpha times expm(kappa S2 / s2)."""
        if not self.aniso or env is None or "dip:S2" not in env:
            return torch.eye(3, dtype=alpha.dtype)[None] / alpha[:, None, None]
        M = torch.linalg.matrix_exp(self.kappa_aniso[sp][:, None, None] * env["dip:S2"] / self.s2_scale)
        return M / alpha[:, None, None]

    @staticmethod
    def _local(bt):
        fr = bt["frame"]
        counts = torch.bincount(fr, minlength=bt["B"])
        return fr, torch.arange(fr.shape[0]) - (torch.cumsum(counts, 0) - counts)[fr]

    def dphi_pairs(self, bt, x, phi_ext):
        """(C,) donor-minus-acceptor environment potential (Eh/e) of each channel i -> j, every atom but i and j."""
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        VD, KD, VA, KA = bt["VD"], bt["KD"], bt["VA"], bt["KA"]
        phiD = VD.sum(-1) + torch.einsum("bikd,bkd->bi", KD, x)
        phiA = VA.sum(-1) + torch.einsum("bikd,bkd->bi", KA, x)
        f, li, lj = fr[i], loc[i], loc[j]
        dD = phiD[f, li] - VD[f, li, lj] - (KD[f, li, lj] * x[f, lj]).sum(-1)
        dA = phiA[f, lj] - VA[f, lj, li] - (KA[f, lj, li] * x[f, li]).sum(-1)
        return dD - dA + phi_ext[i] - phi_ext[j]

    def direct_ct_env(self, q, bt, sp, x, phi_ext):
        B = bt["B"]
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_ch = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        dphi = e_ch.new_zeros(e_ch.shape[0])
        if self.ct_gap:
            dphi = self.dphi_pairs(bt, x, phi_ext)
            lam = self.gap_lamD[sp[i]] + self.gap_lamA[sp[j]]
            u = -lam * dphi / self.dphi_scale
            e_ch = e_ch * torch.exp(GAP_EXP_MAX * torch.tanh(u / GAP_EXP_MAX))     # bounded: at most e^+-GAP_EXP_MAX
        e = e_ch.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
        nT = len(CHANNEL_TYPES)
        e_type = e_ch.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], e_ch).reshape(B, nT)
        return e, e_type, e_ch, dphi

    # -- the model --
    def forward(self, bt: dict, quadrupoles=True, field=None) -> dict:
        sp = bt["species"]
        I = self.scaled(bt["feats"])
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        c = self.confinement(I, None, sp)
        alpha_flat = self.tab.alpha[sp] * c[:, 0]
        Cq = torch.ones(B, nmax, dtype=I.dtype).masked_scatter(mask, self.tab.C[sp] * c[:, 1])
        Lflat = I.new_zeros(n, 8)
        A_ps = I.new_zeros(0)
        if self.pseudo and bt["ps_index"].shape[1]:
            h, _ = self.latents(bt["feats"], sp)
            Lps, A_ps = self.pseudo_drive(h, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], n)
            Lflat = Lflat + Lps
        phi_ext = I.new_zeros(n)
        if field is not None:
            Fa = field[bt["frame"]]
            Lflat = Lflat - torch.cat([Fa, I.new_zeros(n, 5)], 1)
            phi_ext = -(Fa * bt["pos"] / K.BOHR_ANG).sum(-1)
        L = bt["L"] + I.new_zeros(B, nmax, 8).masked_scatter(mask[..., None].expand(B, nmax, 8), Lflat)
        ainv = self.alpha_inverse(alpha_flat, sp, bt.get("env"))
        ainv_pad = torch.eye(3, dtype=I.dtype).expand(B, nmax, 3, 3).clone()
        ainv_pad = ainv_pad.masked_scatter(mask[..., None, None].expand(B, nmax, 3, 3), ainv)
        x0 = solve_induction_blocks(L, bt["A_off"], ainv_pad, Cq, quadrupoles)
        e_pol = 0.5 * (L * x0).sum((1, 2))
        q = self.ct_charges_from(I, sp)
        nC = bt["ch_index"].shape[1]
        if nC:
            e_ct, e_type, e_ch, dphi = self.direct_ct_env(q, bt, sp, x0, phi_ext)
        else:
            e_ct, e_type, e_ch, dphi = I.new_zeros(B), I.new_zeros(B, len(CHANNEL_TYPES)), I.new_zeros(0), I.new_zeros(0)
        z = I.new_zeros(nC)
        return {"e_pol": e_pol, "e_ct": e_ct, "e_ct_post": e_ct, "e_type": e_type, "c": c, "dov": z, "A_ps": A_ps,
                "W": z, "Delta": z, "delta": z, "p": z, "dq": I.new_zeros(n), "x": x0, "x_pol": x0,
                "joint_iters": 0, "q_ct": q, "e_ch": e_ch, "stark": z, "e_pauli": I.new_zeros(B), "dphi": dphi,
                "alpha_inv": ainv}

    def smoothness_penalty(self, bt, sigma=0.05):
        """Feature-noise penalty on the heads' log outputs (log c, log q) and the pseudo-potential strengths."""
        sp = bt["species"]
        f = bt["feats"]
        nz = (f.abs().sum(1, keepdim=True) > 0).to(f.dtype)
        noise = sigma * self.feat_scale * torch.randn_like(f) * nz
        I1, I2 = self.scaled(f), self.scaled(f + noise)
        pen = ((torch.log(self.confinement_from(I2, sp)) - torch.log(self.confinement_from(I1, sp))) ** 2).mean()
        pen = pen + ((self.head_ctq(I2, sp) - self.head_ctq(I1, sp)) ** 2).mean()
        if self.pseudo and bt["ps_index"].shape[1]:
            h1, _ = self.latents(f, sp)
            h2, _ = self.latents(f + noise, sp)
            A0 = self.ps_A0[bt["ps_type"]]
            a1 = self.pseudo_strength(h1, bt["ps_index"], bt["ps_r"], bt["ps_type"])
            a2 = self.pseudo_strength(h2, bt["ps_index"], bt["ps_r"], bt["ps_type"])
            pen = pen + ((torch.log1p(a2 / A0) - torch.log1p(a1 / A0)) ** 2).mean()
        return pen / sigma ** 2

    def summary(self):
        sym = self.tab.symbols
        out = {"qD0/qA0": {x: [round(v, 4) for v in self.ct_lnq0[k].exp().tolist()] for k, x in enumerate(sym)}}
        if self.aniso:
            out["kappa_aniso"] = {x: round(float(self.kappa_aniso[k]), 4) for k, x in enumerate(sym)}
        if self.ct_gap:
            out["lam_D"] = {x: round(float(self.gap_lamD[k]), 4) for k, x in enumerate(sym)}
            out["lam_A"] = {x: round(float(self.gap_lamA[k]), 4) for k, x in enumerate(sym)}
        return out
