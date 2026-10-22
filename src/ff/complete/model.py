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

Breathing (l = 0) response, when the table carries it (``a0 > 0``, from ``scripts/ion_parents.py``):

    E_0,i = -1/2 a0[s] exp(head_0(I_i)) (V[breath]_i^(0) + lam0[s] S[breath]_i^(0))^2

with V[breath]^(0) the penetration potential of all the other frozen densities over the ion's l = 0 response shape
(the ``breath`` probe of the environment) and S[breath]^(0) their overlap with it: the Coulomb drive (fitted to the
CPKS point-charge response) and the Pauli-pressure drive (exchange polarization at l = 0 -- the density contracts
to relieve overlap; lam0 learned, 0 at start). The drive is a scalar, so it adds coherently over neighbours -- the
polarization of an ion in a symmetric cage that no dipole / quadrupole response can give.

Frozen-level terms (``frozen=True``), labels ``mod_pauli`` and ``disp``, heads on the overlap invariants only
(the EDA hierarchy: no fields at the frozen level):

    E_P    = sum_{i<j} qP_i qP_j f(bP_i, bP_j; r) / r,   qP_i = qP0[s] exp(head_P(S_i))      (penetration kernel >= 0)
    E_disp = -sum_{i<j} f6(sqrt(b_i b_j) r) sqrt(C6_i C6_j) / r^6,   C6_i = C60[s] exp(head_6(S_i))   (Tang-Toennies)

Exchange-polarization drive (``pseudo=True``): the Pauli pseudo-potential of neighbour j acting on i's response is
atom-centred,

    A_ij = gamma_i qP_j S_ij / s,   gamma_i = gamma0[s] exp(head_gamma(S_i)) >= 0,

the functional derivative of the bilinear (overlap-squared) Pauli energy with respect to i's density: a damped
exclusion potential whose moment is j's own Pauli charge qP_j (the one of E_P above) and whose range is j's outer
shell, times the overlap S_ij of i's outermost shell with j's density (fixed geometry), times i's sensitivity
gamma_i from its overlap invariants. No pair head, no pair types (the CT-only model's pair form is gone).

Species enter only through the isolated-atom parameters (alpha0, C0, the CT charges qD0 / qA0 and exponents, kappa,
lam, gamma0); the heads are shared (``species_emb`` adds an embedding to their input).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..dispersion import DEFAULT_C6_PRIOR, tt_damped_c6_energy
from ..ionct import kernels as K
from ..ionct.model import PAULI_INIT, PSEUDO_CUTOFF, IonCTModel
from ..ionct.params import IonTable
from ..ionct.ct import ChannelTable
from ..multipole import cartesian_to_spherical_quadrupole, slater_two_center_damp_unequal
from .elst import permanent_electrostatics
from .heads import EnvironmentHead

__all__ = ["CompleteModel", "solve_induction_blocks"]

GAP_EXP_MAX = 4.0     # bound on the log of the CT gap factor
#: start values from a fit to the ct-scan pair labels (R >= 2.2 A), unequal-exponent penetration kernel:
#: Pauli (qP [e], bP [1/bohr]) vs mod_pauli (rel rms Na-Cl 12 %, Cl-Cl 3 %); dispersion (C6 [Ha bohr^6], b [1/bohr])
#: vs disp with geometric-mean combination (rms Cl-Cl 0.37, Na-Cl 0.84 kJ/mol -- the combination rule is the weak
#: spot, the environment head and a learned b take it from there). Other species: DEFAULT_C6_PRIOR / fallback.
PAULI_INIT_UNEQUAL = {"Na": (1.077, 1.453), "Cl": (7.035, 1.599)}
PSEUDO_GAMMA_PER_ALPHA = 0.1   # start of gamma0[s] per bohr^3 of alpha0[s]
#: pauli_form="shells": exclusion weights per permanent shell (outer first? no -- in the table's shell order) and the
#: per-species scale on the shell exponents, from a fit of sum_{t,u} w_t w_u f1(b_t s, b_u s; r)/r to the ct-scan
#: mod_pauli labels (R >= 2.2 A) with the pf2p shells: rms ln E 0.155 (Na-Cl 0.12, Cl-Cl 0.20) vs 0.42 for one
#: Slater per species -- the ion's own shells carry the two-exponent decay of the Cl- Pauli wall
PAULI_SHELL_INIT = {"Na": ([8.856], 0.871), "Cl": ([9.78, 0.302], 0.893)}
DISP_INIT = {"Na": (3.41, 3.872), "Cl": (554.7, 1.130)}
DISP_FALLBACK = (10.0, 2.0)


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
                 species_emb=False, aniso=True, ct_gap=True, dphi_scale=0.05, frozen=False, conf_form="tanh",
                 hyper=False, pauli_moments=False, pauli_form="slater", ct_stark=False):
        super().__init__(tab, chan, n_feat, hidden=hidden, depth=depth, feat_scale=feat_scale, c_range=c_range,
                         confine=confine, pseudo=False, tie_c=tie_c, ct_form="direct", gap_form="pair",
                         pol_form="confine")
        self.pseudo = pseudo
        del self.conf, self.ctq, self.gap                  # replaced by the environment heads / not used (direct CT)
        S = len(tab.symbols)
        ns = S if species_emb else 0
        self.head_conf = EnvironmentHead(n_feat, 2, hidden, depth, ns)     # log c_alpha, log c_C
        # conf_form="mono": stiffening only -- log c_alpha = -softplus(a_s + head) * S[shell]^(0) / s, so c <= 1,
        # exactly 1 at zero overlap and decreasing in it; any enhancement is the overlap drive's job (M3 analysis:
        # the two mechanisms otherwise trade off and the network switches between them)
        # hyper: field-induced enhancement of the polarizability, alpha -> alpha (1 + softplus(g_s) |V[dip]^(1)|^2 / s^2):
        # the gamma hyperpolarizability of the ion in the neighbours' field (ALMO alpha_MI of Cl- sits 8 % above the
        # free-ion value 3-5 A from a Na+); the stiffening head cannot produce c > 1 by design
        self.hyper = hyper
        if hyper:
            names_ = list(feature_names or [])
            self.hyper_col = names_.index("dip:|V1|")
            self.hyper_g = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
        self.conf_form = conf_form
        if conf_form == "mono":
            self.conf_a = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
            self.conf_s0_col = list(feature_names or []).index("shell:S0")
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
        # ct_stark: the CT state's second-order Stark lowering of the gap. The CT state (Na0 Cl0 for an ion pair) is
        # far more polarizable than the ground state, so a field at the donor / acceptor lowers the gap by
        # 1/2 dalpha |E|^2 at second order -- the one way a field perpendicular to a pair reaches the CT (the gap
        # factor's potential difference is blind to it: alpha_CT,perp = 0 without this). The fields are the
        # neighbours' permanent fields over the frontier shapes (env V^(1) of the donor / acceptor probes) plus the
        # external field; eta >= 0 per species, in units of the training-set field scale.
        self.ct_stark = ct_stark
        if ct_stark:
            names_ = list(feature_names or [])
            self.stark_sD = float(feat_scale[names_.index("donor:|V1|")]) if ("donor:|V1|" in names_ and feat_scale is not None) else 1.0
            self.stark_sA = float(feat_scale[names_.index("acceptor:|V1|")]) if ("acceptor:|V1|" in names_ and feat_scale is not None) else 1.0
            self.stark_D = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
            self.stark_A = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
        self.breath = bool((tab.a0 > 0).any()) and "breath:V0_pen" in names
        if self.breath:
            self.breath_col = names.index("breath:V0_pen")
            self.breath_s0_col = names.index("breath:S0")
            self.head_breath = EnvironmentHead(n_feat, 1, hidden, depth, ns)
            self.breath_lna = nn.Parameter(torch.zeros(S, dtype=torch.float64))     # scale on the fitted a0
            self.breath_lam = nn.Parameter(torch.zeros(S, dtype=torch.float64))     # Pauli-pressure drive per overlap
            sv = float(feat_scale[self.breath_col] / feat_scale[self.breath_s0_col]) if feat_scale is not None else 1.0
            self.register_buffer("breath_s0_scale", torch.tensor(sv, dtype=torch.float64))
        # pauli_moments: induced Pauli dipole and quadrupole from the overlap gradients of the outer shell (C1),
        # mu_P,i = -a1[s] S[shell]^(1)_i / s1, Theta_P,i = a2[s] S[shell]^(2)_i / s2, and the frozen Pauli energy becomes
        # the full polytensor contraction P_j^T K_P(bP_i, bP_j; r) P_i with the overlap-complement kernel (a1 = a2 = 0
        # reproduces the monopole form exactly; ions have no permanent Pauli moments)
        self.pauli_moments = pauli_moments
        if pauli_moments:
            if not frozen:
                raise ValueError("pauli_moments needs frozen=True")
            names_ = list(feature_names or [])
            s1 = float(feat_scale[names_.index("shell:|S1|")]) if ("shell:|S1|" in names_ and feat_scale is not None) else 1.0
            s2 = float(feat_scale[names_.index("shell:|S2|")]) if ("shell:|S2|" in names_ and feat_scale is not None) else 1.0
            self.register_buffer("pm_scale", torch.tensor([s1, s2], dtype=torch.float64))
            self.pm_a1 = nn.Parameter(torch.zeros(S, dtype=torch.float64))
            self.pm_a2 = nn.Parameter(torch.zeros(S, dtype=torch.float64))
        self.frozen = frozen
        self.pauli_form = pauli_form
        if pauli_form not in ("slater", "shells"):
            raise ValueError(f"pauli_form {pauli_form!r}")
        if frozen or pseudo:
            f64 = dict(dtype=torch.float64)
            sym = tab.symbols
            pp = [PAULI_INIT_UNEQUAL.get(x, PAULI_INIT.get(x, (1.0, 1.5))) for x in sym]
            self.pauli_lnq0 = nn.Parameter(torch.tensor([math.log(q) for q, _ in pp], **f64))
            self.pauli_lnb = nn.Parameter(torch.tensor([math.log(b) for _, b in pp], **f64))
            if pauli_form == "shells":
                # exclusion density = the ion's own permanent shells with per-shell weights w[s, t] (an absent shell
                # has weight 0) and exponents b[s, t] * exp(lnbs[s]); unknown species start with the single-Slater
                # q on the outermost shell
                T = tab.shell_n.shape[1]
                lnw = torch.full((S, T), -30.0, **f64)
                lnbs = torch.zeros(S, **f64)
                for k, x in enumerate(sym):
                    occ = [t for t in range(T) if tab.shell_n[k, t] > 0]
                    if x in PAULI_SHELL_INIT:
                        ws, bs = PAULI_SHELL_INIT[x]
                        for t, w in zip(occ, ws):
                            lnw[k, t] = math.log(w)
                        lnbs[k] = math.log(bs)
                    else:
                        lnw[k, occ[-1]] = math.log(pp[k][0])
                self.pauli_lnw = nn.Parameter(lnw)
                self.pauli_lnbs = nn.Parameter(lnbs)
                self.register_buffer("pauli_shell_mask", (tab.shell_n > 0).to(torch.float64))
                self.register_buffer("pauli_outer", torch.tensor([max(t for t in range(T) if tab.shell_n[k, t] > 0) for k in range(S)]))
            pri = [DISP_INIT.get(x, DEFAULT_C6_PRIOR.get(int(z), DISP_FALLBACK)) for x, z in zip(sym, tab.Z.tolist())]
            self.disp_lnc60 = nn.Parameter(torch.tensor([math.log(c) for c, _ in pri], **f64))
            self.disp_lnb = nn.Parameter(torch.tensor([math.log(b) for _, b in pri], **f64))
            # the frozen heads see the overlap invariants only (no fields at the frozen level)
            cols = [k for k, nm in enumerate(names) if nm.split(":")[-1] in ("S0", "|S1|", "|S2|")] or list(range(n_feat))
            self.register_buffer("frozen_cols", torch.tensor(cols, dtype=torch.long))
            self.head_frozen = EnvironmentHead(len(cols), 2, hidden, depth, ns)     # log qP / qP0, log C6 / C60
        if pseudo:
            # the drive strength carries the overlap S_ij of i's outermost shell with j's density (bilinear Pauli
            # energy ~ S^2: its derivative with respect to i's density is one overlap times the exclusion potential);
            # the drive basis alone decays like the penetration potential of j's shell, too slowly: without the factor
            # the gamma head has to supply the decay from the overlap invariants and does so badly for many-body
            names_ = list(feature_names or [])
            sref = float(feat_scale[names_.index("shell:S0")]) if ("shell:S0" in names_ and feat_scale is not None) else 1.0
            self.register_buffer("drive_s_ref", torch.tensor(sref, dtype=torch.float64))
            b_outer = torch.where(tab.shell_n > 0, tab.shell_b, torch.full_like(tab.shell_b, float("inf"))).min(1).values
            self.register_buffer("drive_b_outer", b_outer)
            self._ps_ovl = None
            # gamma_i: i's sensitivity to its neighbours' exclusion potentials, overlap invariants only. Start value
            # proportional to the ion's polarizability (a diffuse response density feels the exclusion more): with
            # PSEUDO_GAMMA_PER_ALPHA = 0.1, Cl- starts at gamma = 3 (A[Cl<Na] = 3.2, the pair-head runs end near 13)
            # and Na+ at 0.1 (A[Na<Cl] = 0.7, pair-head runs near 1.5)
            self.ps_lng0 = nn.Parameter(torch.log(PSEUDO_GAMMA_PER_ALPHA * tab.alpha.clamp(min=1e-3)).double())
            self.head_ps = EnvironmentHead(len(cols), 1, hidden, depth, ns)
        self.double()

    # -- heads --
    def scaled(self, feats):
        return feats / self.feat_scale

    def confinement_from(self, I, sp):
        """(N, 2) scales on alpha0 and C0 (c_range, tanh-saturated like the ion CT model), 1 for an isolated atom."""
        u = self.head_conf(I, sp) * self.confine_mask[sp][:, None]
        if self.conf_form == "mono":
            rate = nn.functional.softplus(self.conf_a[sp][:, None] + u)          # (N, 2) >= 0
            logc = -rate * I[:, self.conf_s0_col:self.conf_s0_col + 1].clamp(min=0) * self.confine_mask[sp][:, None]
        else:
            lo, hi = self.log_c_lo, self.log_c_hi
            logc = torch.where(u > 0, hi * torch.tanh(u / hi), lo * torch.tanh(u / lo))
        if self.tie_c:
            logc = torch.stack([logc[:, 0], logc[:, 0] * 5.0 / 3.0], 1)
        c = logc.exp()
        if self.hyper:
            v1 = I[:, self.hyper_col]                                         # scaled |V[dip]^(1)|
            enh = 1 + nn.functional.softplus(self.hyper_g[sp]) * v1 * v1 * self.confine_mask[sp]
            c = c * torch.stack([enh, enh ** (5.0 / 3.0) if self.tie_c else torch.ones_like(enh)], 1)
        return c

    def ct_charges_from(self, I, sp):
        return torch.exp(self.ct_lnq0[sp] + self.head_ctq(I, sp))

    def confinement(self, I, h0=None, species=None):
        """Same entry point as the ion CT model (the pair-scan attribution overrides it), on the scaled invariants."""
        return self.confinement_from(I, species)

    def ct_charges(self, h):
        raise NotImplementedError("CompleteModel: use ct_charges_from(I, species)")

    def pseudo_gamma(self, I, sp):
        """(N,) gamma_i >= 0: the response sensitivity of each atom to its neighbours' exclusion potentials."""
        return torch.exp(self.ps_lng0[sp] + self.head_ps(I[:, self.frozen_cols], sp)[:, 0])

    def pair_overlap(self, idx, r_ang, sp):
        """Overlap (bohr^-3) of i's outermost permanent shell with j's shells for directed pairs (i <- j)."""
        i, j = idx
        rb = r_ang / K.BOHR_ANG
        one = K._onehot0(rb.shape[0], rb)
        tab = self.tab
        S = rb.new_zeros(rb.shape[0])
        for t in range(tab.shell_n.shape[1]):
            nt = tab.shell_n[sp[j], t]
            if bool((nt > 0).any()):
                S = S + nt * K.shaped_overlap(self.drive_b_outer[sp[i]], one, tab.shell_b[sp[j], t], one, rb)
        return S

    def pseudo_strength(self, I, idx, r, ty=None):
        """A_ij,t = gamma_i qP_j,t S_ij / s >= 0 of each directed pair (i <- j) and exclusion shell t of j (P, T),
        smoothly zero at PSEUDO_CUTOFF. With the single-Slater Pauli form the one exclusion charge sits on j's
        outermost shell."""
        sp = self._sp
        qP = self.frozen_params(I, sp)[0]
        ovl = self._ps_ovl
        if ovl is None or ovl.shape[0] != idx.shape[1]:          # computed once per batch (forward sets it)
            ovl = self.pair_overlap(idx, r, sp)
        x = ((r - (PSEUDO_CUTOFF - 1.0)) / 1.0).clamp(0, 1)
        cut = 1 - x * x * (3 - 2 * x)
        A = self.pseudo_gamma(I, sp)[idx[0]] * ovl / self.drive_s_ref * cut             # (P,)
        if self.pauli_form == "shells":
            return A[:, None] * qP[idx[1]]
        T = self.tab.shell_n.shape[1]
        outer = torch.tensor([max(t for t in range(T) if self.tab.shell_n[k, t] > 0) for k in range(len(self.tab.symbols))],
                             device=I.device)
        return (A * qP[idx[1]])[:, None] * nn.functional.one_hot(outer[sp[idx[1]]], T).to(I.dtype)

    def pseudo_drive(self, I, idx, r, ty, basis, n, basis_sh=None):
        """(n, 8): sum_j sum_t A_ij,t basis_ij,t on every ion, and the total strength A (P,). A 1-D strength (an
        override) is applied to the outer-shell ``basis``."""
        A = self.pseudo_strength(I, idx, r, ty)
        if A.dim() == 1 or basis_sh is None:
            At = A if A.dim() == 1 else A.sum(1)
            return I.new_zeros(n, 8).index_add(0, idx[0], At[:, None] * basis), At
        return I.new_zeros(n, 8).index_add(0, idx[0], torch.einsum("pt,ptb->pb", A, basis_sh)), A.sum(1)

    def alpha_inverse(self, alpha, sp, env):
        """(N, 3, 3) inverse polarizability: isotropic 1/alpha times expm(kappa S2 / s2)."""
        if not self.aniso or env is None or "dip:S2" not in env:
            return torch.eye(3, dtype=alpha.dtype, device=alpha.device)[None] / alpha[:, None, None]
        M = torch.linalg.matrix_exp(self.kappa_aniso[sp][:, None, None] * env["dip:S2"] / self.s2_scale)
        return M / alpha[:, None, None]

    @staticmethod
    def _local(bt):
        fr = bt["frame"]
        counts = torch.bincount(fr, minlength=bt["B"])
        return fr, torch.arange(fr.shape[0], device=fr.device) - (torch.cumsum(counts, 0) - counts)[fr]

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

    def stark_gap(self, bt, sp, Fa):
        """(C,) >= 0: eta_D |E_D|^2 + eta_A |E_A|^2 of each channel, fields in units of the training scale."""
        i, j = bt["ch_index"]
        env = bt.get("env") or {}
        n = sp.shape[0]
        ED = env.get("donor:V1", Fa.new_zeros(n, 3)) + Fa
        EA = env.get("acceptor:V1", Fa.new_zeros(n, 3)) + Fa
        uD = nn.functional.softplus(self.stark_D[sp[i]]) * (ED[i] * ED[i]).sum(-1) / self.stark_sD ** 2
        uA = nn.functional.softplus(self.stark_A[sp[j]]) * (EA[j] * EA[j]).sum(-1) / self.stark_sA ** 2
        return uD + uA

    def direct_ct_env(self, q, bt, sp, x, phi_ext, Fa=None):
        B = bt["B"]
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_ch = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        dphi = e_ch.new_zeros(e_ch.shape[0])
        u = e_ch.new_zeros(e_ch.shape[0])
        if self.ct_gap:
            dphi = self.dphi_pairs(bt, x, phi_ext)
            lam = self.gap_lamD[sp[i]] + self.gap_lamA[sp[j]]
            u = u - lam * dphi / self.dphi_scale
        if self.ct_stark:
            u = u + self.stark_gap(bt, sp, Fa if Fa is not None else e_ch.new_zeros(sp.shape[0], 3))
        if self.ct_gap or self.ct_stark:
            e_ch = e_ch * torch.exp(GAP_EXP_MAX * torch.tanh(u / GAP_EXP_MAX))     # bounded: at most e^+-GAP_EXP_MAX
        e = e_ch.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
        nT = self.nT
        e_type = e_ch.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], e_ch).reshape(B, nT)
        return e, e_type, e_ch, dphi

    def frozen_params(self, I, sp):
        """Per atom: Pauli charge qP, exponent bP, C6, dispersion exponent (each (N,)); with the shells form qP and
        bP are (N, T) per permanent shell (weight 0 where the shell is absent)."""
        u = self.head_frozen(I[:, self.frozen_cols], sp)
        C6 = torch.exp(self.disp_lnc60[sp] + u[:, 1])
        if self.pauli_form == "shells":
            w = torch.exp(self.pauli_lnw[sp] + u[:, 0:1]) * self.pauli_shell_mask[sp]
            b = self.tab.shell_b[sp] * self.pauli_lnbs[sp].exp()[:, None]
            return w, b, C6, self.disp_lnb[sp].exp()
        qP = torch.exp(self.pauli_lnq0[sp] + u[:, 0])
        return qP, self.pauli_lnb[sp].exp(), C6, self.disp_lnb[sp].exp()

    def frozen_terms(self, I, bt, sp):
        """Frozen Pauli and dispersion energies per frame (B,), (B,) and the parameters."""
        B = bt["B"]
        qP, bP, C6, b6 = self.frozen_params(I, sp)
        pairs = K.all_pairs(bt["frame"])
        i, j = pairs
        r_ang = (bt["pos"][j] - bt["pos"][i]).norm(dim=-1)
        rb = r_ang / K.BOHR_ANG
        shells = self.pauli_form == "shells"
        qPs = qP if shells else qP[:, None]                                       # (N, T)
        bPs = bP if shells else bP[:, None]
        T_ = qPs.shape[1]
        if self.pauli_moments and bt.get("env") is not None:
            env = bt["env"]
            mu = -self.pm_a1[sp][:, None] * env["shell:S1"] / self.pm_scale[0]
            th = self.pm_a2[sp][:, None] * cartesian_to_spherical_quadrupole(env["shell:S2"]) / self.pm_scale[1]
            P = K._poly_map(I.dtype, I.device)
            polym = (torch.cat([mu, th], 1) @ P.T)[:, 1:]                        # (N, 9) induced Pauli moments
            dr = (bt["pos"][j] - bt["pos"][i]) / K.BOHR_ANG
            # the induced moments sit on the outermost shell (the overlap-gradient probe)
            outer = self.pauli_outer[sp] if shells else torch.zeros_like(sp)
            e_p = rb.new_zeros(rb.shape[0])
            for t in range(T_):
                pt = torch.cat([qPs[:, t:t + 1], polym * (outer == t).to(I.dtype)[:, None]], 1)
                for u in range(T_):
                    pu = torch.cat([qPs[:, u:u + 1], polym * (outer == u).to(I.dtype)[:, None]], 1)
                    f = slater_two_center_damp_unequal(bPs[i, t], bPs[j, u], rb, 2)   # (5, P) complements = the kernel
                    Tt = K._tensor(dr, f)                                             # (P, 10, 10) [j, i]
                    e_p = e_p + torch.einsum("pa,pab,pb->p", pu[j], Tt, pt[i])
            self._pauli_moments_out = (mu, th)
        else:
            e_p = rb.new_zeros(rb.shape[0])
            for t in range(T_):
                for u in range(T_):
                    f = slater_two_center_damp_unequal(bPs[i, t], bPs[j, u], rb, 0)[0]
                    e_p = e_p + qPs[i, t] * qPs[j, u] * f / rb
        e_d = tt_damped_c6_energy(r_ang, torch.sqrt(C6[i] * C6[j]), torch.sqrt(b6[i] * b6[j]))
        fr = bt["frame"][i]
        return I.new_zeros(B).index_add(0, fr, e_p), I.new_zeros(B).index_add(0, fr, e_d), (qP, bP, C6, b6)

    # -- the model --
    def forward(self, bt: dict, quadrupoles=True, field=None) -> dict:
        sp = bt["species"]
        I = self.scaled(bt["feats"])
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        c = self.confinement(I, None, sp)
        alpha_flat = self.tab.alpha[sp] * c[:, 0]
        Cq = torch.ones(B, nmax, dtype=I.dtype, device=I.device).masked_scatter(mask, self.tab.C[sp] * c[:, 1])
        Lflat = I.new_zeros(n, 8)
        A_ps = I.new_zeros(0)
        self._sp = sp
        if self.pseudo and bt["ps_index"].shape[1]:
            self._ps_ovl = self.pair_overlap(bt["ps_index"], bt["ps_r"], sp)
            Lps, A_ps = self.pseudo_drive(I, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], n,
                                          bt.get("ps_basis_sh"))
            Lflat = Lflat + Lps
        phi_ext = I.new_zeros(n)
        Fa = None
        if field is not None:
            Fa = field[bt["frame"]]
            Lflat = Lflat - torch.cat([Fa, I.new_zeros(n, 5)], 1)
            phi_ext = -(Fa * bt["pos"] / K.BOHR_ANG).sum(-1)
        L = bt["L"] + I.new_zeros(B, nmax, 8).masked_scatter(mask[..., None].expand(B, nmax, 8), Lflat)
        ainv = self.alpha_inverse(alpha_flat, sp, bt.get("env"))
        ainv_pad = torch.eye(3, dtype=I.dtype, device=I.device).expand(B, nmax, 3, 3).clone()
        ainv_pad = ainv_pad.masked_scatter(mask[..., None, None].expand(B, nmax, 3, 3), ainv)
        x0 = solve_induction_blocks(L, bt["A_off"], ainv_pad, Cq, quadrupoles)
        e_pol = 0.5 * (L * x0).sum((1, 2))
        e_breath_atom = I.new_zeros(n)
        if self.breath:
            v0 = bt["feats"][:, self.breath_col] + self.breath_lam[sp] * self.breath_s0_scale * bt["feats"][:, self.breath_s0_col]
            a = self.tab.a0[sp] * torch.exp(self.breath_lna[sp] + self.head_breath(I, sp)[:, 0])
            e_breath_atom = -0.5 * a * v0 * v0
            e_pol = e_pol + I.new_zeros(B).index_add(0, bt["frame"], e_breath_atom)
        q = self.ct_charges_from(I, sp)
        nC = bt["ch_index"].shape[1]
        if nC:
            e_ct, e_type, e_ch, dphi = self.direct_ct_env(q, bt, sp, x0, phi_ext, Fa)
        else:
            e_ct, e_type, e_ch, dphi = I.new_zeros(B), I.new_zeros(B, self.nT), I.new_zeros(0), I.new_zeros(0)
        z = I.new_zeros(nC)
        e_pauli, e_disp, frz = I.new_zeros(B), I.new_zeros(B), None
        if self.frozen:
            e_pauli, e_disp, frz = self.frozen_terms(I, bt, sp)
        e_elst = permanent_electrostatics(bt["pos"], sp, bt["frame"], self.tab) if self.frozen else I.new_zeros(B)
        return {"e_pol": e_pol, "e_ct": e_ct, "e_ct_post": e_ct, "e_type": e_type, "c": c, "dov": z, "A_ps": A_ps,
                "W": z, "Delta": z, "delta": z, "p": z, "dq": I.new_zeros(n), "x": x0, "x_pol": x0,
                "joint_iters": 0, "q_ct": q, "e_ch": e_ch, "stark": z, "e_pauli": e_pauli, "e_disp": e_disp,
                "frozen_params": frz, "dphi": dphi, "alpha_inv": ainv, "e_breath_atom": e_breath_atom,
                "e_elst": e_elst, "e_int": e_elst + e_pauli + e_disp + e_pol + e_ct}

    def smoothness_penalty(self, bt, sigma=0.05):
        """Feature-noise penalty on the heads' log outputs (log c, log q) and the pseudo-potential strengths."""
        sp = bt["species"]
        f = bt["feats"]
        nz = (f.abs().sum(1, keepdim=True) > 0).to(f.dtype)
        noise = sigma * self.feat_scale * torch.randn_like(f) * nz
        I1, I2 = self.scaled(f), self.scaled(f + noise)
        pen = ((torch.log(self.confinement_from(I2, sp)) - torch.log(self.confinement_from(I1, sp))) ** 2).mean()
        pen = pen + ((self.head_ctq(I2, sp) - self.head_ctq(I1, sp)) ** 2).mean()
        if self.frozen or self.pseudo:
            c_ = self.frozen_cols
            pen = pen + ((self.head_frozen(I2[:, c_], sp) - self.head_frozen(I1[:, c_], sp)) ** 2).mean()
        if self.pseudo:
            c_ = self.frozen_cols
            pen = pen + ((self.head_ps(I2[:, c_], sp) - self.head_ps(I1[:, c_], sp)) ** 2).mean()
        return pen / sigma ** 2

    @torch.no_grad()
    def summary(self):
        sym = self.tab.symbols
        out = {"qD0/qA0": {x: [round(v, 4) for v in self.ct_lnq0[k].exp().tolist()] for k, x in enumerate(sym)}}
        if self.aniso:
            out["kappa_aniso"] = {x: round(float(self.kappa_aniso[k].detach()), 4) for k, x in enumerate(sym)}
        if self.ct_gap:
            out["lam_D"] = {x: round(float(self.gap_lamD[k]), 4) for k, x in enumerate(sym)}
            out["lam_A"] = {x: round(float(self.gap_lamA[k]), 4) for k, x in enumerate(sym)}
        if self.ct_stark:
            out["ct stark softplus(eta_D/eta_A)"] = {x: [round(float(nn.functional.softplus(self.stark_D[k])), 4),
                                                        round(float(nn.functional.softplus(self.stark_A[k])), 4)] for k, x in enumerate(sym)}
        if self.hyper:
            out["hyper softplus(g)"] = {x: round(float(nn.functional.softplus(self.hyper_g[k])), 4) for k, x in enumerate(sym)}
        if self.conf_form == "mono":
            out["conf rate softplus(a)"] = {x: round(float(nn.functional.softplus(self.conf_a[k])), 4) for k, x in enumerate(sym)}
        if self.breath:
            out["breath a0 scale"] = {x: round(float(self.breath_lna[k].exp()), 4) for k, x in enumerate(sym)}
            out["breath lam0"] = {x: round(float(self.breath_lam[k]), 4) for k, x in enumerate(sym)}
        if self.pauli_moments:
            out["pauli moments a1/a2"] = {x: [round(float(self.pm_a1[k]), 4), round(float(self.pm_a2[k]), 4)] for k, x in enumerate(sym)}
        if (self.frozen or self.pseudo) and self.pauli_form == "shells":
            out["pauli shell w"] = {x: [round(float(v), 4) for v, m in zip(self.pauli_lnw[k].exp(), self.pauli_shell_mask[k]) if m > 0]
                                    for k, x in enumerate(sym)}
            out["pauli b scale"] = {x: round(float(self.pauli_lnbs[k].exp()), 4) for k, x in enumerate(sym)}
        elif self.frozen or self.pseudo:
            out["pauli qP0/bP"] = {x: [round(float(self.pauli_lnq0[k].exp()), 4), round(float(self.pauli_lnb[k].exp()), 4)]
                                  for k, x in enumerate(sym)}
        if self.pseudo:
            out["pseudo gamma0"] = {x: round(float(self.ps_lng0[k].exp()), 4) for k, x in enumerate(sym)}
        if self.frozen:
            out["disp C60/b"] = {x: [round(float(self.disp_lnc60[k].exp()), 3), round(float(self.disp_lnb[k].exp()), 4)]
                                for k, x in enumerate(sym)}
        return out
