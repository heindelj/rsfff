"""The cmm model: a polarizable, charge-transfer force field whose environment-dependent parameters come from one
learned latent per atom.

    E_int = E_elst + E_Pauli + E_disp + E_pol + E_CT                     (Hartree, relative to the isolated species)

Fixed by the species table (``IonTable``): the permanent density (nucleus + Slater 1s shells), the free-ion dipole,
quadrupole and breathing polarizabilities and response shapes, the Legendre hyperpolarizabilities (gamma, b), the
frontier exponents and the channel cutoff / taper.

  ``config.hyperpol = "saturating"`` replaces the Legendre step's polynomial field energy by a bounded one with the
  same gamma and b (:func:`rsfff.ff.cmm.precompute.saturating_hyperpolarization`): alpha(E_p) stays between
  alpha / 2 and the ceiling alpha + f (1 / u1 - alpha), f in (0, 1) learned (:meth:`hyper_fraction`), 1 / u1 the
  largest polarizability for which the induction operator is positive definite at any geometry (u1 the dipole
  shape's Coulomb self-energy).

* **E_elst**: frozen densities, penetration included (no parameters).
* **E_Pauli** = sum_{i<j} sum_{t,u} w_it w_ju f(b_it, b_ju; r) / r, w_it = w0[s,t] e^{cap(u_pauli)}, b_it = b_shell[s,t] e^{lnbs[s]}.
  ``config.pauli_form = "site"``: E_Pauli = sum_{i<j} e^{cap(u_i) + cap(u_j)} [<n_i | v_j> + <n_j | v_i>], n_i the
  permanent shells (the electrostatics' own density) and v_j(r) = sum_t a[s,t] e^{-beta r} (1 / r + beta / 2),
  beta = b_shell[s,t] e^{lnbs[s]}: the electrons of one ion in a short-ranged site potential of the other
  (:func:`site_pauli_kernel`; ``pauli_ln_w`` then holds ln a and ``pauli_ln_bscale`` the potential's exponent scale).
* **E_disp** = -sum_{i<j} f6(sqrt(b_i b_j) r) sqrt(C6_i C6_j) / r^6, C6_i = C60[s] e^{cap(u_disp)}, b_i = the Pauli range of
  the outermost shell (``pauli_form = "site"``: the outermost shell's exponent times its own learned scale).
  ``config.disp_model = "response"``: no learned coefficient and no environment factor. The free-ion C6_ab and
  C8_ab = C8dq_ab + C8dq_ba of every species pair come from the ions' response spectra (``disp_table.py``), and each
  is scaled by the confinement of the two ions' own polarizabilities, rho_l,i = alpha_l,i^eff / alpha_l^free
  (l = 1 dipole, 2 quadrupole; alpha^eff from the inverse of the ion's on-site block, the free ion stiffened by its
  exchange Gram blocks: ``disp_confine``), at fixed oscillator mass (w -> w / sqrt(rho)):
      E_disp = -sum_{i<j} [f6 C6_ij / r^6 + f8 C8_ij / r^8],     s(ra, rb; wa, wb) = ra rb (wa + wb) / (wa sqrt(rb) + wb sqrt(ra)),
      C6_ij = C6_ab s(rho1_i, rho1_j; w1_a, w1_b),
      C8_ij = C8dq_ab s(rho1_i, rho2_j; w1_a, w2_b) + C8dq_ba s(rho2_i, rho1_j; w2_a, w1_b)
  (rho^(3/2) for a like pair; the many-body dispersion of confined ions), Tang-Toennies damping with one learned
  range per species (``disp_ln_bscale``, its own, not the Pauli range).
* **E_pol** and **E_CT**: one joint quadratic in the induced moments z = (dipole, quadrupole, breathing) and the
  channel charges q,
      E(z, q) = 1/2 z A z + (L - h) z + sum_i c_i + [Gram exchange polarization: g z + 1/2 z H z]
                + sum_c [1/2 b_c q_c^2 - V_c q_c + q_c dphi_ext,c] + 1/2 Q^T H_Q Q + Q^T C^T z,   Q = G q
  with A the induction operator (free-ion on-site responses, the hyperpolarized dipole block alpha(E_p), damped
  parent-parent coupling), L the permanent drive, h / c the Legendre offset / constant, g_i = sum_j E0_ij W1_i
  d ln O_ij / dz_i the exchange drive (E0_ij the pair's Pauli energy, O_ij the frozen density overlap) and H the
  on-site Gram blocks W2_i sum_j 2 (E0_ij / O_ij) Gt(i <- j). The CT channels (every directed pair within the CT
  cutoff) have the kernel energy e_c = -qD_i qA_j f(bD_i, bA_j; r) / r (taper), compliance b_c = V0_c^2 / (2|e_c|) and
  drive V_c calibrated so that a lone pair gives exactly e_c + e_c'. E_pol is the minimum at q = 0 (ALMO's CT-free
  polarization); E_CT = E_joint - E_pol (the CT-induced repolarization is CT). One Cholesky factorization of A + H
  serves both, and the polarizability (the energy is quadratic in a uniform field: exact, three more right-hand sides).

  ``config.xpol_bound`` ("rank" / "scalar") ties the two exchange weights so that the Pauli wall survives any output:
  the rank weights K = diag(W1) act on the response functions, i.e. on both orders,
      g_i = sum_j E0_ij K_i d ln O_ij / dz_i,     H_i = theta_i sum_j 2 (E0_ij / O_ij) K_i Gt(i <- j) K_i,
  with theta_i = 2 (1 + e^{ln W2 + u}) >= 2. By Cauchy-Schwarz (Gt >= O d d^T / 4) the terms of side i of a pair are
  >= -E0_ij / theta_i for every z, so E0_ij + both sides >= E0_ij (1 - 1 / theta_i - 1 / theta_j) >= 0
  (:meth:`xpol_ratio`, :meth:`xpol_fraction`; theory/response_functional.tex). "scalar": one W1 per atom.

  Two optional terms stiffen the flows that move no net charge (ring currents: G q = 0, invisible to H_Q and to the
  induction, so held by b_c alone, which vanishes as |e_c| grows):
      + sum_c 1/2 b_min q_c^2                       ``config.ct_floor``: b_c -> b_c + b_min, every channel
      + sum_i 1/2 tau_i T_i^2,  T = |G| q           ``config.ct_relay``: tau_i = kappa eta_i on the site's throughput
  T_i = D_i + A_i with D_i the charge i donates (channels out of i) and A_i the charge it accepts (Q_i = D_i - A_i):
  per site 1/2 (eta + tau)(D^2 + A^2) - (eta - tau) D A, a donor and an acceptor slot whose mutual cancellation is
  incomplete. A relay site (D = A) is cooperative for kappa < 1 and neutral at kappa = 1. The drives are calibrated
  with both terms, so a lone pair still gives exactly e_c + e_c'; the ring enhancement is bounded by 1 + h_c / b_min.

``config.ct_model = "gap"`` replaces the flux's channels by two-state channels with an explicit gap (no pair
calibration, no v0; the charges are one-sided, so the environment can only lower them):

      W_c^2    = qD_i qA_j f(bD_i, bA_j; r) / r (taper)                the coupling
      Delta_c  = I[s_i] - A[s_j] + phi_D(i) - phi_A(j) - J_ij + kappa[s_j] P_j
      a_c      = sqrt(Delta^2 / 4 + W^2) - Delta / 2                   the lone channel's CT energy (E = -a)
      q*_c     = 1/2 (1 - Delta / sqrt(Delta^2 + 4 W^2))               its charge, = dE / dDelta
      k_c      = (Delta^2 + 4 W^2)^(3/2) / (2 W^2)                     its stiffness at q*, = 1 / (dq* / dDelta)

  phi_D(i), phi_A(j): the other ions' frozen potential (partner included) averaged over the donor frontier shape of
  i / the acceptor frontier shape of j (``bt["phi_front"]``: point charges + penetration); J_ij the slots' Coulomb;
  kappa[s_j] P_j the Pauli shift of the acceptor level by its neighbours' closed shells (:meth:`acceptor_shift`):
  by default kappa[s_j] sum_k (E0_jk / O_jk) S_A(j <- k), the acceptor probe's overlap with each neighbour times
  that pair's frozen Pauli energy per unit density overlap, kappa a learned dimensionless constant per acceptor
  species, bounded to within a factor 5 of 1 (:meth:`overlap_strength`) (``ct_gap_overlap = "probe"``: kappa times the probe's total overlap instead); A of an anion is
  I - GAP_ANION_J J_self (twice the Coulomb cost of a second electron in the same shape: anion -> anion gaps stay
  open between differently coordinated anions). The channels
  enter the joint quadratic expanded about their two-state minima,
      sum_c [-a_c + 1/2 k_c (q_c - q*_c)^2] + sum_{c < c'} (G^T H_Q G)_cc' q_c q_c' + Q^T C^T z,
  so a lone channel is exact in energy, charge and susceptibility (of the two-state form broadened to the least
  crossing width ``ct_gap_w0``, :meth:`gap_channels`) and the site hardness only couples different
  channels (a channel's own charging cost is in I - A - J). k_c - h_c may be small or negative: such channels are
  kept explicit in the cluster solve instead of being eliminated (:meth:`_flux_solve_mixed`).

  ``config.ct_gap_converge`` > 0 makes that expansion exact: the channels' two-state energies are kept in full,
      E(z, q) = [the quadratic in z] + sum_c [F_c(q_c) - 1/2 h_c q_c^2] + 1/2 Q^T H_Q Q + Q^T C^T z,
      F_c(q) = Delta_c q - 2 W_b sqrt(q (lam - q)),
  and the joint minimum is found by Newton iterations of the gap equation delta_c = (psi_i - psi_j) - h_c q_c, psi the
  potential of the joint solution's own charges and moments (:meth:`_converge_gap`; each iteration is the coupled
  solve above about the current charges, the predictor only the start). The gaps are then converged with the
  response: every charge is a two-state charge in (0, lam) whatever the coupling, and the energy is bounded below.
  One differentiable Newton step from the converged point puts its dependence on positions and parameters in the
  graph, so forces, polarizabilities and the gradients of both are exact. Under PBC the same iteration runs on the
  CG solve (:meth:`_converge_gap_pbc`), warm-started from the previous step's shifts (``bt["guess"]``).

  ``config.ct_exclusion``: the channels of a site share its capacity (a site gives at most one electron and takes
  at most one): F_c = Delta_c q_c - 2 W_b sqrt(q_c (lam_c (1 - b_c) - q_c)), b_c the charge of the other channels
  out of c's donor and into its acceptor (:meth:`_flux_excl`, :meth:`_forward_pbc_excl`). No parameters.

The environment factors u (``encoder.READOUTS``) are decoded from one latent per atom (:class:`Encoder`) whose inputs
are the overlap-weighted two- and three-body channels (``channels.py``) and the overlap-gated diffuse-probe
invariants. Everything that depends on the geometry alone (A, L, h, c, the encoder's inputs, the exchange pairs'
overlap maps and Gram blocks, the charge slots, the channels, E_elst) comes from :meth:`CmmModel.precompute`
(``precompute.py``); the forward does what depends on the parameters.

Clusters solve the joint minimum densely (Cholesky of A + H, the channels eliminated onto the atomic charges).
Periodic cells (``precompute(..., cell=...)``) solve the same minimum by matrix-free CG on (z, q) with Ewald sums for
the long-range couplings of the moments and the slot charges (:class:`rsfff.ff.cmm.periodic.CmmOperator`); every
short-range piece -- Pauli, the exchange drive and Gram blocks, the channels' kernels, compliances, drives and
curvatures h_c, the encoder's inputs -- is the cluster's, from the image pair list. Dispersion is switched off at the
cutoff with the uniform tail added. A cluster in a large cell reproduces the cluster (to the images' 1/L^5).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..damping import tang_toennies
from ..dispersion import tt_damped_c6_energy
from ..multipole import slater_two_center_damp_unequal
from . import kernels as K
from .batch import channel_types
from .channels import CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS
from .params import ChannelTable, IonTable
from .periodic import (CmmOperator, PolarizationCatastrophe, dispersion_switch, dispersion_tail_integral,
                       solve_cmm_periodic)
from .precompute import (CmmTable, batched_precompute, cmm_table, dipole_self_coulomb,
                         saturating_hyperpolarization)
from .species import SpeciesEmbedding
from .tabulate import PROBES, shape_self_coulomb
from .config import PAIR_GROUPS, CmmConfig
from .ct_overlap import load_ct_overlap, overlap_kernel, pair_switch
from .disp_table import load_disp_table, pair_scale
from .encoder import READOUTS, Encoder

__all__ = ["CmmModel", "PROBES", "GATED_NAMES"]

#: |ln(bA / b_Fukui)| bound of the floating cation acceptor exponents
CT_ACC_CAP = 0.5
GATED_NAMES = tuple(f"{p}:{k}" for p in PROBES for k in ("S0", "V0_pen", "|V1g|^2", "|V2g|^2", "V1g.S1"))
#: start values of the learned per-species constants (fits of the frozen-level labels for Na+ / Cl-)
PAULI_SHELL_INIT = {"Na": ([8.856], 0.871), "Cl": ([9.78, 0.302], 0.893)}
DISP_INIT = {"Na": (3.41, 3.872), "Cl": (554.7, 1.130)}
DISP_FALLBACK = (10.0, 2.0)
FLUX_V0_INIT = 0.025
#: ``xpol_bound``: start values of the drive weight W1 and of the stiffness ratio theta (W1^2 theta ~ 1, the unbound
#: start's Gram stiffness; each side can remove at most 1 / theta = a third of a pair's Pauli energy)
#: ``hyperpol = "saturating"``: start value of the ceiling fraction f (the free chloride's finite-field response
#: follows a ceiling of 1.34 alpha, a quarter of its room)
HYPER_F_INIT = 0.25
XPOL_BOUND_W1_INIT = 0.6
XPOL_BOUND_THETA_INIT = 3.0
#: gap model: start values of the acceptors' overlap strength ("probe": Hartree per unit probe overlap, about what
#: cancels the pair's penetration at contact; "pauli": dimensionless, on the pair Pauli energy per density overlap) and of the CT charges (W of a few 0.01 Hartree at contact), and the channel stiffness
#: k_c - h_c below which a channel is solved explicitly instead of being eliminated onto the atomic charges
GAP_KOV_INIT = {"probe": 10.0, "pauli": 1.5}
GAP_KOV_REF = {"probe": 10.0, "pauli": 1.0}     # the overlap strength stays within a factor GAP_KOV_CAP of this
GAP_KOV_CAP = 5.0
GAP_Q0_INIT = 0.5
FRONTIER_Q0_INIT = 0.1          # Hartree: W = sqrt(qD qA) S starts as 0.1 S (the size of the trained slater couplings)
GAP_B_EPS = 0.05
GAP_ANION_J = 2.0           # an anion's acceptor level: A = I - GAP_ANION_J J_self


def site_pauli_kernel(b, beta, rb):
    """<rho_b | v_beta>(R): a normalized Slater 1s density of exponent ``b`` in the unit site potential
    v(r) = e^{-beta r} (1 / r + beta / 2) centred a distance ``rb`` away (bohr; broadcast). v is 1 / r minus the
    potential of a 1s cloud of exponent beta, so the integral is the cloud-cloud overlap complement minus the
    point-cloud one: f(b, beta; R) / R - e^{-b R} (1 / R + b / 2)."""
    b, beta, rb = torch.broadcast_tensors(b, beta, rb)
    f = slater_two_center_damp_unequal(b, beta, rb, 0)[0]
    return f / rb - torch.exp(-b * rb) * (1.0 / rb + 0.5 * b)


def _saturate(x, m):
    """m tanh(x / m) per column where m > 0 (the training range's bound), x elsewhere."""
    if not bool((m > 0).any()):
        return x
    mm = torch.where(m > 0, m, torch.ones_like(m))
    return torch.where(m > 0, mm * torch.tanh(x / mm), x)


class CmmModel(nn.Module):
    """``CmmModel(tab, chan, config)``; ``forward(bt, field=None, alpha=False)`` on a batch of
    :meth:`precompute` (:func:`rsfff.ff.cmm.precompute.batched_precompute`)."""

    def __init__(self, tab: IonTable, chan: ChannelTable, config: CmmConfig | None = None):
        super().__init__()
        cfg = config or CmmConfig()
        self.config, self.tab, self.chan = cfg, tab, chan
        f64 = dict(dtype=torch.float64)
        S, T = len(tab.symbols), tab.shell_n.shape[1]
        sym = tab.symbols
        self.ctypes = list(channel_types(sym))
        self.nT = len(self.ctypes)
        # -- frozen level: Pauli (the permanent shells, per-shell weights, one exponent scale) and dispersion
        occ = tab.shell_n > 0
        outer = torch.tensor([max(t for t in range(T) if occ[k, t]) for k in range(S)])
        lnw = torch.full((S, T), -30.0, **f64)
        lnbs = torch.zeros(S, **f64)
        self.pauli_site = cfg.pauli_form == "site"
        for k, x in enumerate(sym):
            occ_k = [t for t in range(T) if occ[k, t]]
            if self.pauli_site:                 # ln a per shell and the potential's exponent scale (table: pauli_site)
                if tab.pauli_site_s is None or not math.isfinite(float(tab.pauli_site_s[k])):
                    raise ValueError(f"pauli_form = 'site': the species table has no pauli_site entry for {x}")
                lnw[k] = torch.where(occ[k], torch.log(tab.pauli_site_a[k].to(lnw).clamp(min=1e-13)), lnw[k])
                lnbs[k] = math.log(float(tab.pauli_site_s[k]))
            elif tab.pauli_s is not None and math.isfinite(float(tab.pauli_s[k])):      # the frozen fit
                lnw[k] = torch.where(occ[k], torch.log(tab.pauli_w[k].to(lnw)), lnw[k])
                lnbs[k] = math.log(float(tab.pauli_s[k]))
            elif x in PAULI_SHELL_INIT:
                ws, bs = PAULI_SHELL_INIT[x]
                for t, w in zip(occ_k, ws):
                    lnw[k, t] = math.log(w)
                lnbs[k] = math.log(bs)
            else:
                lnw[k, occ_k[-1]] = 0.0
        self.pauli_ln_w = nn.Parameter(lnw)
        self.pauli_ln_bscale = nn.Parameter(lnbs)
        self.register_buffer("shell_mask", occ.to(torch.float64), persistent=False)
        self.register_buffer("outer_shell", outer, persistent=False)
        from ..dispersion import DEFAULT_C6_PRIOR
        c6 = [DISP_INIT.get(x, DEFAULT_C6_PRIOR.get(int(z), DISP_FALLBACK))[0] for x, z in zip(sym, tab.Z.tolist())]
        self.disp_ln_c6 = nn.Parameter(torch.log(torch.tensor(c6, **f64)))
        self.disp_response = cfg.disp_model == "response"
        if self.pauli_site:           # the damping range starts from the table's disp_range (else the outermost shell's
            ob = tab.shell_b[torch.arange(S), outer].to(**f64)           # exponent): the potential's scale is not a density range
            db = tab.disp_b.to(**f64) if tab.disp_b is not None else torch.full((S,), float("nan"), **f64)
            lnbs_disp = torch.where(torch.isfinite(db), torch.log(db.clamp(min=1e-6) / ob), torch.zeros(S, **f64))
            if not self.disp_response:
                self.disp_ln_bscale = nn.Parameter(lnbs_disp.clone())
        else:
            lnbs_disp = lnbs
        if self.disp_response:
            from .io import resolve_params
            dtab = load_disp_table(resolve_params(cfg.disp_table), sym)
            self.register_buffer("disp_c6", dtab["c6"])              # persistent: a checkpoint carries its table
            self.register_buffer("disp_c8", dtab["c8"])
            self.register_buffer("disp_w", dtab["w"])
            self.disp_ln_bscale = nn.Parameter(lnbs_disp.clone())    # the damping range, per species
            Cq = tab.C.to(torch.float64)
            inv = [1.0 / tab.alpha.to(torch.float64), torch.where(Cq > 0, 1.0 / Cq.clamp(min=1e-30), torch.full_like(Cq, 1e12)),
                   torch.where(tab.a0 > 0, 1.0 / tab.a0.clamp(min=1e-30), torch.ones_like(tab.a0)).to(torch.float64)]
            self.register_buffer("disp_free_inv", torch.stack(inv, 1), persistent=False)     # (S, 3): 1 / alpha, 1 / C, 1 / a0
            self.register_buffer("disp_has_quad", Cq > 0, persistent=False)
        # -- exchange polarization: W1 per species and rank (dipole, quadrupole, breathing), W2 per species
        self.ln_w1 = nn.Parameter(torch.zeros(S, 3, **f64))          # W1 = 1 at start (the overlap model)
        self.ln_w2 = nn.Parameter(torch.zeros(S, **f64))
        if cfg.xpol_bound != "off":                                  # ln_w2 is then ln(theta / 2 - 1): :meth:`w2`
            with torch.no_grad():
                self.ln_w1.fill_(math.log(XPOL_BOUND_W1_INIT))
                self.ln_w2.fill_(math.log(XPOL_BOUND_THETA_INIT / 2 - 1))
        self.hyper_sat = cfg.hyperpol == "saturating"
        if self.hyper_sat:
            u1 = dipole_self_coulomb(tab).to(tab.alpha)
            al = tab.alpha.to(torch.float64)
            self.register_buffer("hyper_u1", u1.to(torch.float64), persistent=False)
            self.register_buffer("hyper_room", (1.0 / u1.to(torch.float64) - al).clamp(min=0.0), persistent=False)
            self.register_buffer("hyper_alpha", al.clone(), persistent=False)
            self.register_buffer("hyper_gb", torch.stack([tab.hyper_gamma.to(torch.float64).clamp(min=0.0),
                                                          tab.hyper_b.to(torch.float64)], 1), persistent=False)
            self.hyper_raw = nn.Parameter(torch.full((S if cfg.hyper_ceiling == "species" else 1,),
                                                     math.log(HYPER_F_INIT / (1 - HYPER_F_INIT)), **f64))
        self.breath = bool((tab.a0 > 0).any())
        self.n_b = 1 if self.breath else 0
        self.register_buffer("breath_mask", (tab.a0 > 0).to(torch.float64), persistent=False)
        # -- CT
        self.ct_ln_q0 = nn.Parameter(torch.zeros(S, 2, **f64))                  # ln qD0, ln qA0
        self.ct_acc_raw = nn.Parameter(torch.zeros(S, **f64))                   # cation acceptor exponent scale
        b_sh = tab.shell_b[torch.arange(S), outer].to(torch.float64)
        anion = tab.charge < 0
        self.register_buffer("ct_b_donor", b_sh.clone(), persistent=False)
        self.register_buffer("ct_b_acc0", torch.where(anion, b_sh, tab.front_b["acceptor"].to(torch.float64)),
                             persistent=False)
        self.register_buffer("ct_anion", anion, persistent=False)
        self.ct_ln_v0 = nn.Parameter(torch.full((S,), math.log(FLUX_V0_INIT), **f64))
        js = shape_self_coulomb(tab)
        self.register_buffer("ct_jself", js, persistent=False)
        eta0 = (tab.I_donor - torch.nan_to_num(tab.A_acceptor, nan=0.0) - js).clamp(min=0.01)
        self.ct_eta_raw = nn.Parameter(torch.log(torch.expm1(eta0)).to(**f64))  # softplus: on-site hardness
        self.gap = cfg.ct_model == "gap"
        self.frontier = cfg.ct_kernel == "frontier"
        self.overlap_cut = self.frontier or cfg.ct_cut == "overlap"      # each pair's own switch (ct_overlap.py)
        if self.overlap_cut:
            from .io import resolve_params
            ov = load_ct_overlap(resolve_params(cfg.ct_overlap_table), sym, cfg.ct_overlap_window)
            self.register_buffer("ctk_coef", ov["coef"])            # persistent: a checkpoint carries its table
            self.register_buffer("ctk_grid", ov["grid"])
            self.register_buffer("ctk_ron", ov["r_on"])
            self.register_buffer("ctk_roff", ov["r_off"])
            self._chan0 = chan
            self.sync_channels()
        if self.gap:
            with torch.no_grad():
                self.ct_ln_q0.fill_(math.log(FRONTIER_Q0_INIT if self.frontier else GAP_Q0_INIT))
            c, u0 = math.log(GAP_KOV_CAP), math.log(GAP_KOV_INIT[cfg.ct_gap_overlap] / GAP_KOV_REF[cfg.ct_gap_overlap])
            self.ct_ln_kov = nn.Parameter(torch.full((S,), c * math.atanh(u0 / c), **f64))   # per acceptor species, raw
            A = torch.where(torch.isnan(tab.A_acceptor), tab.I_donor - GAP_ANION_J * js, tab.A_acceptor)
            self.register_buffer("ct_gap_I", tab.I_donor.to(torch.float64).clone(), persistent=False)
            self.register_buffer("ct_gap_A", A.to(torch.float64), persistent=False)
        # -- encoder
        sel = {g for t in cfg.pairs for g in PAIR_GROUPS[t]}
        keep = [(nm in sel) if nm.startswith("S^") else (nm[2:4] in sel) for nm in CHANNEL_NAMES]
        self.register_buffer("chan_keep", torch.tensor(keep, **f64), persistent=False)
        n_ch = len(CHANNEL_NAMES)
        self.register_buffer("chan_scale", torch.ones(n_ch, **f64))
        self.register_buffer("chan_sat", torch.zeros(n_ch, **f64))
        n_g = len(GATED_NAMES) if cfg.probes else 0
        self.register_buffer("gate_scale", torch.ones(n_g, **f64))
        self.register_buffer("gate_sat", torch.zeros(n_g, **f64))
        self.species = SpeciesEmbedding(tab, dim=cfg.species_dim, kind="descriptor", hidden=cfg.species_hidden) \
            if cfg.film else None
        self.encoder = Encoder(n_ch + n_g, latent=cfg.latent, hidden=cfg.hidden, depth=cfg.depth,
                               cond_dim=cfg.species_dim if cfg.film else 0)
        self._table = None
        self.table_store = {}            # pair key -> coefficients: shipped with the checkpoint (io.load_cmm)
        self._u = None
        self.double()

    # -- geometry ----------------------------------------------------------------------------------------------
    def sync_channels(self):
        """Per-pair overlap switch (the frontier kernel, ``ct_cut = "overlap"``): the channel list from the switch
        radii the model holds (a checkpoint's own once its state is loaded): a pair is a channel inside its own
        cutoff, and only there (no taper)."""
        if not self.overlap_cut:
            return
        sym, roff, chan = self.tab.symbols, self.ctk_roff, self._chan0
        self.chan = ChannelTable(types={f"{a}>{b}": float(roff[ia, ib]) for ia, a in enumerate(sym)
                                        for ib, b in enumerate(sym) if f"{a}>{b}" in chan.types and roff[ia, ib] > 0},
                                 cutoff=float(roff.max()), taper=chan.taper,
                                 meta={**chan.meta, "kernel": self.config.ct_kernel, "cut": "overlap"})

    def table(self, device=None) -> CmmTable:
        """The precompute's radial table of this species table on ``device``. Its species pairs are filled on first
        use from :attr:`table_store` (the pairs shipped with the checkpoint), the disk cache, or built; each pair is
        keyed by the two ions' parameter values (:meth:`rsfff.ff.cmm.tabulate.RadialTable.pair_key`)."""
        if self._table is None:
            self._table = cmm_table(self.tab.to("cpu"), store=self.table_store)
        if device is not None and self._table.device != torch.device(device):
            self._table.to(device)
        return self._table

    def precompute(self, pos_ang, atomic_numbers, batch_idx, labels=None, **kw) -> dict:
        """The batch of positions (N, 3) Angstrom, atomic numbers (N,) and frame index (N,) (autograd flows from the
        positions): :func:`batched_precompute` with this model's tables, gate and switch (``cell`` etc. pass
        through)."""
        return batched_precompute(pos_ang, atomic_numbers, batch_idx, self.tab, self.table(pos_ang.device), self.chan,
                                  labels=labels, gate=self.config.env_gate, switch=self.config.env_switch,
                                  overlap=self.config.env_overlap, **kw)

    def encoder_inputs(self, X, Gv=None):
        """The encoder's input rows from the raw channels X (N, 17) and gated invariants Gv (N, 15): scaled, saturated,
        the unread channel groups zeroed; trunk order [three-body, two-body, gated]."""
        x = _saturate(X / self.chan_scale, self.chan_sat) * self.chan_keep
        parts = [x[:, list(THREE_BODY_COLS)], x[:, list(TWO_BODY_COLS)]]
        if self.config.probes:
            parts.append(_saturate(Gv / self.gate_scale, self.gate_sat))
        return torch.cat(parts, 1)

    def _inputs(self, bt):
        if tuple(bt["gate"]) != tuple(self.config.env_gate):
            raise ValueError(f"the batch's probe gate {bt['gate']} is not the model's {self.config.env_gate}")
        if bt.get("switch", "cubic") != self.config.env_switch:
            raise ValueError(f"the batch's switch {bt.get('switch', 'cubic')!r} is not the model's "
                             f"{self.config.env_switch!r}")
        if tuple(bt.get("overlap", (1e-7, 1e-6))) != tuple(self.config.env_overlap):
            raise ValueError(f"the batch's overlap window {bt.get('overlap')} is not the model's "
                             f"{self.config.env_overlap}")
        return bt["chan"], (bt["gated"] if self.config.probes else None)

    def environment(self, bt):
        """{readout: (N,)} the raw log-factors of this batch (also kept for the parameter maps and the penalties)."""
        sp = bt["species"]
        X, Gv = self._inputs(bt)
        c = self.species(sp) if self.species is not None else None
        U = self.encoder(self.encoder_inputs(X, Gv), c)
        self._u = {k: U[:, m] for m, k in enumerate(READOUTS)}
        self._S = X[:, 0]
        return self._u

    def cap(self, u):
        c = self.config.env_cap
        return c * torch.tanh(u / c)

    # -- parameter maps ----------------------------------------------------------------------------------------
    def ct_exponents(self):
        """(donor, acceptor) Slater exponents of the CT kernel per species: the outermost shell, except the cation
        acceptors (b_Fukui, scaled within e^+-CT_ACC_CAP)."""
        c = CT_ACC_CAP
        acc = torch.where(self.ct_anion, self.ct_b_acc0, self.ct_b_acc0 * torch.exp(c * torch.tanh(self.ct_acc_raw / c)))
        return self.ct_b_donor, acc

    def ct_charges(self, sp):
        """(N, 2) qD, qA = q0[s] e^{f}, f = c tanh(u / c); with ``config.ct_q_onesided`` (always in the gap model) f
        is one-sided. ``ct_q_headroom = 0``: clipped at 0, so the isolated-species charges q0 are a ceiling and the
        pair kernels of any geometry are bounded by the bare ones -- with a kink in every charge where its readout
        changes sign, a jump in the forces. ``ct_q_headroom = h > 0``: the smooth
        f = h - c tanh(softplus(-beta u) / (beta c)), beta = ln 2 / (c atanh(h / c)): f(0) = 0, f in (h - c, h), so
        the ceiling is e^h q0 and the kernels are bounded by e^{2h} times the bare ones."""
        u = torch.stack([self._u["qD"], self._u["qA"]], 1)
        c = self.config.ct_q_cap
        h = self.config.ct_q_headroom
        if (self.config.ct_q_onesided or self.gap) and h > 0:
            beta = math.log(2.0) / (c * math.atanh(h / c))
            f = h - c * torch.tanh(nn.functional.softplus(-beta * u) / (beta * c))
            return torch.exp(self.ct_ln_q0[sp] + f)
        f = c * torch.tanh(u / c)
        if self.config.ct_q_onesided or self.gap:
            f = torch.where(f > 0, torch.zeros_like(f), f)      # not clamp: its gradient is 0 at u = 0, the start
        return torch.exp(self.ct_ln_q0[sp] + f)

    def hyper_fraction(self):
        """(S,) f: the learned fraction of the room 1 / u1 - alpha that the saturating hyperpolarization's ceiling
        uses (``config.hyper_ceiling``: one value for all species or one each)."""
        return torch.sigmoid(self.hyper_raw).expand(self.hyper_room.shape[0])

    def hyperpolarized(self, bt, sp):
        """The batch with the saturating hyperpolarization in place of the precompute's Legendre step
        (``config.hyperpol = "saturating"``): the on-site dipole blocks alpha(E_p)^-1, the drive's offset h and the
        constants e_hyp, from the permanent field and gradient at every atom and the learned ceiling
        alpha + f (1 / u1 - alpha)."""
        kap = (self.hyper_fraction() * self.hyper_room)[sp]
        gb = self.hyper_gb[sp]
        ainv, h, c = saturating_hyperpolarization(bt["hyp_L"], self.hyper_alpha[sp], kap, gb[:, 0], gb[:, 1])
        d = ainv - bt["hyp_ainv"]
        B, D = bt["B"], bt["D"]
        fr, loc = self._local(bt)
        dL = torch.cat([bt["hyp_h"] - h, h.new_zeros(h.shape[0], D - 3)], 1)
        out = {**bt, "Lz": bt["Lz"].index_put((fr, loc), dL, accumulate=True), "e_hyp": c.new_zeros(B).index_add(0, fr, c)}
        if bt.get("pbc") is not None:
            on = bt["pbc"]["onsite"]
            out["pbc"] = {**bt["pbc"], "onsite": on + torch.nn.functional.pad(d, (0, D - 3, 0, D - 3))}
        else:
            ar = torch.arange(3, device=d.device)
            o = (loc * D)[:, None, None]
            out["A"] = bt["A"].index_put((fr[:, None, None], o + ar[None, :, None], o + ar[None, None, :]), d, accumulate=True)
        return out

    def disp_range(self):
        """(S,) the dispersion damping exponent per species: the outermost shell's, times the Pauli range scale
        (emitted dispersion) or its own learned scale (response dispersion)."""
        S = self.outer_shell.shape[0]
        ln = self.disp_ln_bscale if (self.disp_response or self.pauli_site) else self.pauli_ln_bscale
        return self.tab.shell_b[torch.arange(S, device=self.outer_shell.device), self.outer_shell] * ln.exp()

    def frozen_params(self, sp):
        """Per atom: Pauli shell weights and exponents (N, T), C6 (N,), dispersion exponent (N,). With
        ``pauli_form = "site"`` the weights are the site potential's amplitudes a_it e^{cap(u_i)} and the exponents
        its beta_it."""
        w = torch.exp(self.pauli_ln_w[sp] + self.cap(self._u["pauli"])[:, None]) * self.shell_mask[sp]
        b = self.tab.shell_b[sp] * self.pauli_ln_bscale.exp()[sp][:, None]
        if self.disp_response:              # the free like-pair coefficient (the pair values: response_dispersion)
            C6 = self.disp_c6[sp, sp]
        else:
            C6 = torch.exp(self.disp_ln_c6[sp] + self.cap(self._u["disp"]))
        return w, b, C6, self.disp_range()[sp]

    def w1(self, sp):
        """(N, D) the drive weight W1 of every response component (dipole x3, quadrupole x5, breathing). With
        ``config.xpol_bound = "scalar"`` the dipole row and readout serve every rank (one weight per atom)."""
        u = torch.stack([self._u["w1_dip"], self._u["w1_quad"], self._u["w1_breath"]], 1)
        k = torch.exp(self.ln_w1[sp] + self.cap(u))
        if self.config.xpol_bound == "scalar":
            k = k[:, :1].expand(-1, 3)
        parts = [k[:, :1].expand(-1, 3), k[:, 1:2].expand(-1, 5)]
        if self.n_b:
            parts.append((k[:, 2] * self.breath_mask[sp])[:, None])
        return torch.cat(parts, 1)

    def w2(self, sp):
        """(N,) the Gram weight: W2 = e^{ln_w2 + cap(u)}, or with ``config.xpol_bound`` the stiffness ratio
        theta = 2 (1 + e^{ln_w2 + cap(u)}) >= 2 that multiplies K Gt K (:meth:`exchange`)."""
        w = torch.exp(self.ln_w2[sp] + self.cap(self._u["w2"]))
        return 2 * (1 + w) if self.config.xpol_bound != "off" else w

    def xpol_ratio(self, sp):
        """(N,) with ``config.xpol_bound``: 1 / theta = 1/2 sigmoid(-(ln_w2 + cap(u))) <= 1/2, the largest fraction
        of a pair's Pauli energy that this ion's response can remove (W1^2 / W2 of a rank-independent W1)."""
        if self.config.xpol_bound == "off":
            raise ValueError("xpol_ratio: the model has no bound (config.xpol_bound = 'off'); see xpol_fraction")
        return 1.0 / self.w2(sp)

    # -- terms -------------------------------------------------------------------------------------------------
    def frozen_terms(self, bt, sp):
        """Frozen Pauli and dispersion energies per frame (B,), (B,); keeps the Pauli pair energies for xpol.
        With the response dispersion the second is None: it needs the exchange blocks
        (:meth:`response_dispersion`)."""
        B = bt["B"]
        w, b, C6, b6 = self.frozen_params(sp)
        (i, j), dr = bt["pairs"], bt["pair_dr"]
        r_ang = dr.norm(dim=-1)
        rb = r_ang / K.BOHR_ANG
        if self.pauli_site:           # e^{u_i + u_j} [<n_i | v_j> + <n_j | v_i>]: w carries a e^{u} of the potential's owner
            n, bn = self.tab.shell_n[sp], self.tab.shell_b[sp]
            env = torch.exp(self.cap(self._u["pauli"]))
            kij = site_pauli_kernel(bn[i][:, :, None], b[j][:, None, :], rb[:, None, None])      # i's shells in v_j
            kji = site_pauli_kernel(bn[j][:, :, None], b[i][:, None, :], rb[:, None, None])
            e_p = env[i] * torch.einsum("pt,ptu,pu->p", n[i], kij, w[j]) + env[j] * torch.einsum("pt,ptu,pu->p", n[j], kji, w[i])
        else:
            f = slater_two_center_damp_unequal(b[i][:, :, None], b[j][:, None, :], rb[:, None, None], 0)[0]
            e_p = torch.einsum("pt,ptu,pu->p", w[i], f, w[j]) / rb
        self._pauli_pairs = ((i, j), e_p)
        fr = bt["frame"][i]
        if self.disp_response:
            return r_ang.new_zeros(B).index_add(0, fr, e_p), None, (w, b, C6, b6)
        e_d = tt_damped_c6_energy(r_ang, torch.sqrt(C6[i] * C6[j]), torch.sqrt(b6[i] * b6[j]))
        pbc = bt.get("pbc")
        if pbc is not None:           # switched off over the cutoff's last switch_width, the uniform tail beyond
            e_d = e_d * dispersion_switch(r_ang, pbc["cutoff"], pbc["switch_width"])
        E_d = r_ang.new_zeros(B).index_add(0, fr, e_d)
        if pbc is not None and pbc.get("disp_tail") is not None:
            sq = r_ang.new_zeros(B).index_add(0, bt["frame"], torch.sqrt(C6))
            E_d = E_d - 2 * math.pi / pbc["volume"] * sq * sq * pbc["disp_tail"]
        return r_ang.new_zeros(B).index_add(0, fr, e_p), E_d, (w, b, C6, b6)

    def confinement(self, sp, gram=None, onsite=None):
        """(N, 2) rho_1, rho_2: each ion's confined dipole and quadrupole polarizability over the free ion's,
        alpha^eff the rank's block of the inverse on-site block (its trace / 3 or / 5: the other ranks relaxed).
        ``config.disp_confine``: "pauli" the free-ion block plus ``gram`` (N, D, D), the ion's summed exchange Gram
        blocks (positive semidefinite, so rho <= 1); "onsite" the solve's own block ``onsite`` (N, D, D), which also
        has the field-dependent hyperpolarized dipole response; "none" ones. An ion without quadrupole response has
        rho_2 = 1 (its quadrupole coefficients are zero)."""
        mode = self.config.disp_confine
        one = self.disp_free_inv.new_ones(sp.shape[0], 2)
        if mode == "none":
            return one
        fi = self.disp_free_inv[sp]
        if mode == "onsite":
            O = onsite
        else:
            D = gram.shape[-1]
            d = torch.cat([fi[:, :1].expand(-1, 3), fi[:, 1:2].expand(-1, 5)] + ([fi[:, 2:3]] if D > 8 else []), 1)
            O = torch.diag_embed(d) + gram
        Oi = torch.linalg.inv(O)
        a1 = torch.diagonal(Oi[:, :3, :3], dim1=1, dim2=2).sum(-1) / 3
        a2 = torch.diagonal(Oi[:, 3:8, 3:8], dim1=1, dim2=2).sum(-1) / 5
        r2 = torch.where(self.disp_has_quad[sp], a2 * fi[:, 1], one[:, 1])
        return torch.stack([a1 * fi[:, 0], r2], 1).clamp(min=1e-12)

    def response_dispersion(self, bt, sp, rho):
        """The response dispersion per frame (B,) for the confinement ratios ``rho`` (N, 2) (:meth:`confinement`):
        the module docstring's E_disp. Keeps the pair coefficients (``self._disp_pairs`` = C6_ij, C8_ij). Under PBC
        the pair terms are switched off at the cutoff and the uniform tail of both powers is added, with the scale
        factor of the missing pairs factorized as (rho_i rho_j)^(3/4) (exact for a like pair, and where the two
        confined frequencies agree)."""
        B = bt["B"]
        (i, j), dr = bt["pairs"], bt["pair_dr"]
        r_ang = dr.norm(dim=-1)
        rb = r_ang / K.BOHR_ANG
        si, sj = sp[i], sp[j]
        w, r1, r2 = self.disp_w, rho[:, 0], rho[:, 1]
        c6 = self.disp_c6[si, sj] * pair_scale(r1[i], r1[j], w[si, 0], w[sj, 0])
        c8 = (self.disp_c8[si, sj] * pair_scale(r1[i], r2[j], w[si, 0], w[sj, 1])
              + self.disp_c8[sj, si] * pair_scale(r2[i], r1[j], w[si, 1], w[sj, 0]))
        self._disp_pairs = (c6, c8)
        b6 = self.disp_range()[sp]
        x = torch.sqrt(b6[i] * b6[j]) * rb
        e_d = -(tang_toennies(x, 6) * c6 / rb.pow(6) + tang_toennies(x, 8) * c8 / rb.pow(8))
        pbc = bt.get("pbc")
        if pbc is not None:
            e_d = e_d * dispersion_switch(r_ang, pbc["cutoff"], pbc["switch_width"])
        E_d = r_ang.new_zeros(B).index_add(0, bt["frame"][i], e_d)
        if pbc is not None and pbc.get("disp_tail") is not None:
            S = self.disp_c6.shape[0]
            q = rho.pow(0.75)
            R = r_ang.new_zeros(B * S, 2).index_add(0, bt["frame"] * S + sp, q).reshape(B, S, 2)
            t6 = torch.einsum("ba,ac,bc->b", R[..., 0], self.disp_c6, R[..., 0])
            t8 = 2 * torch.einsum("ba,ac,bc->b", R[..., 0], self.disp_c8, R[..., 1])
            I8 = dispersion_tail_integral(pbc["cutoff"], pbc["switch_width"], power=6)
            E_d = E_d - 2 * math.pi / pbc["volume"] * (t6 * pbc["disp_tail"] + t8 * I8)
        return E_d

    @staticmethod
    def _local(bt):
        fr = bt["frame"]
        counts = torch.bincount(fr, minlength=bt["B"])
        return fr, torch.arange(fr.shape[0], device=fr.device) - (torch.cumsum(counts, 0) - counts)[fr]

    def exchange(self, bt, sp):
        """The exchange pairs' drive g (N, D) and on-site Gram blocks W2_i 2 A_ij Gt(i <- j), W2_j 2 A_ij Gt(j <- i):
        g_i = sum_j 2 E0_ij W1_i d ln O_ij / dz_i over the pairs within R_XPOL, E0_ij the pair's Pauli energy and
        A_ij = E0_ij / O_ij (switched on above the overlap noise). With ``config.xpol_bound`` the blocks are
        theta_i 2 A_ij K_i Gt(i <- j) K_i, K = diag(W1): the rank weights on both orders. Returns (i, j), g, Gi, Gj."""
        xs = bt["x_sel"]
        i, j = bt["pairs"][:, xs]
        e0 = self._pauli_pairs[1][xs]
        P = xs.shape[0]
        k = self.w1(sp)
        rel = bt["x_rel"]
        vj = 0.5 * k[j] * rel[:P]                                # (i <- j): j's deformation changes O_ij
        vi = 0.5 * k[i] * rel[P:]
        g = rel.new_zeros(sp.shape[0], k.shape[1]).index_add(0, i, 2 * e0[:, None] * vi).index_add(0, j, 2 * e0[:, None] * vj)
        w = (2 * e0 * bt["x_swO"])[:, None, None]
        w2 = self.w2(sp)
        Gi, Gj = w * bt["x_Gi"], w * bt["x_Gj"]
        if self.config.xpol_bound != "off":
            Gi = k[i][:, :, None] * Gi * k[i][:, None, :]
            Gj = k[j][:, :, None] * Gj * k[j][:, None, :]
        return (i, j), g, w2[i][:, None, None] * Gi, w2[j][:, None, None] * Gj

    def xpol_fraction(self, bt):
        """(2P,) for every exchange pair, sides i then j: the largest fraction of the pair's frozen Pauli energy that
        the side's exchange drive and Gram block can remove, rho = 1/2 g_p^T H_p^+ g_p / E0 with g_p, H_p the pair's
        own contributions to that ion's drive and Gram block (the minimum over z of g_p z + 1/2 z H_p z is -rho E0).
        A pair's Pauli energy plus its exchange terms is >= E0 (1 - rho_i - rho_j) for every z. With
        ``config.xpol_bound`` rho <= :meth:`xpol_ratio` <= 1/2; without it this is the diagnostic of how far a trained
        model is from that bound. Pairs with a negligible Pauli energy return 0."""
        sp = bt["species"]
        self.environment(bt)
        self.frozen_terms(bt, sp)
        xs = bt["x_sel"]
        i, j = bt["pairs"][:, xs]
        e0 = self._pauli_pairs[1][xs]
        P = xs.shape[0]
        k, w2 = self.w1(sp), self.w2(sp)
        rel = bt["x_rel"]
        w = (2 * e0 * bt["x_swO"])[:, None, None]
        out = []
        for a, r, G in ((i, rel[P:], bt["x_Gi"]), (j, rel[:P], bt["x_Gj"])):
            g = e0[:, None] * k[a] * r
            H = w * G
            if self.config.xpol_bound != "off":
                H = k[a][:, :, None] * H * k[a][:, None, :]
            H = w2[a][:, None, None] * H
            x = torch.einsum("pab,pb->pa", torch.linalg.pinv(H, hermitian=True), g)
            ok = e0 > 1e-14
            out.append(torch.where(ok, 0.5 * (g * x).sum(1) / torch.where(ok, e0, torch.ones_like(e0)),
                                   torch.zeros_like(e0)))
        return torch.cat(out)

    def slot_eta(self, sp):
        """(N,) the slots' self-energy + hardness."""
        return self.ct_jself[sp] + nn.functional.softplus(self.ct_eta_raw[sp])

    def slot_tau(self, sp):
        """(N,) the throughput hardness tau_i = kappa eta_i (``config.ct_relay``), or None when it is off."""
        k = self.config.ct_relay
        return k * self.slot_eta(sp) if k > 0 else None

    def channel_curvature(self, bt, sp):
        """(C,) h_c = (G^T H_Q G)_cc of each channel's own pair: eta_i + eta_j - 2 J_ij (the slots' Coulomb at the
        channel's distance, unscreened -- a local quantity, the same in a cluster and in a cell)."""
        i, j = bt["ch_index"]
        eta = self.slot_eta(sp)
        return eta[i] + eta[j] - 2 * bt["ch_J"]

    def channel_throughput(self, bt, sp):
        """(C,) tau_i + tau_j of each channel's pair (zeros when ``config.ct_relay`` is 0)."""
        i, j = bt["ch_index"]
        tau = self.slot_tau(sp)
        return bt["ch_J"].new_zeros(i.shape[0]) if tau is None else tau[i] + tau[j]

    def channel_drives(self, bt, sp, q, h):
        """Per channel, given h_c = (G^T H_Q G)_cc: the kernel energy e_c <= 0, the drive V_c and s_c = 1 / sqrt(b_c),
        b_c = V0_c^2 / (2|e_c|) + b_min. A lone channel's stiffness is b_c + h_c + tau_i + tau_j and V_c is set so that
        its minimum is e_c; the two directions of a pair couple by -(h_c - tau_i - tau_j) (counter-flow cancels the
        net charge, not the throughput) and share the factor t_p that makes the pair's own minimum exactly
        e_c + e_c'."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_c = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        ae = (-e_c).clamp(min=0.0)
        v0 = torch.exp(self.ct_ln_v0)
        V0 = v0[sp[i]] + v0[sp[j]]
        bf = self.config.ct_floor
        tij = self.channel_throughput(bt, sp)
        hd, ho = h + tij + bf, h - tij                  # the pair's 2x2 curvature beyond b: diagonal, -(off-diagonal)
        Vh2 = V0 * V0 + 2 * ae * hd
        cc = 2 * ae / Vh2
        rev = bt["ch_rev"]
        has = rev >= 0
        rv = rev.clamp(min=0)
        ae2 = torch.where(has, ae[rv], torch.zeros_like(ae))
        cc2 = torch.where(has, cc[rv], torch.zeros_like(cc))
        S_ = ae + ae2
        t2 = S_ * (1 - ho * ho * cc * cc2) / (S_ + 2 * ho * torch.sqrt(ae * cc * ae2 * cc2 + 1e-300) + 1e-300)
        t2 = torch.where(has, t2, torch.ones_like(t2))
        s = torch.sqrt(2 * ae + 1e-30) / (V0 if bf == 0 else torch.sqrt(V0 * V0 + 2 * ae * bf))
        return e_c, torch.sqrt(t2 * Vh2), s

    def solve(self, bt, sp, Lz, rhs_extra):
        """The minimum of 1/2 z (A + H) z + (Lz + g) z over z = (dipole, quadrupole, breathing): one Cholesky
        factorization of A + H (LU if it is not positive definite) that also solves ``rhs_extra``. Returns z
        (B, nmax, D), the electrical part 1/2 z A z + Lz z (B,), the exchange part g z + 1/2 z H z (B,) and
        (A + H)^-1 rhs_extra."""
        B, nmax, D = bt["B"], bt["nmax"], bt["D"]
        N = nmax * D
        fr, loc = self._local(bt)
        (i, j), g, Gi, Gj = self.exchange(bt, sp)
        gz = Lz.new_zeros(B, nmax, D).index_put((fr, loc), g).reshape(B, N)
        ar = torch.arange(D, device=Lz.device)
        A = bt["A"]
        oi, oj = (loc[i] * D)[:, None, None], (loc[j] * D)[:, None, None]
        M = A.index_put((fr[i][:, None, None], oi + ar[None, :, None], oi + ar[None, None, :]), Gi, accumulate=True)
        M = M.index_put((fr[j][:, None, None], oj + ar[None, :, None], oj + ar[None, None, :]), Gj, accumulate=True)
        Lf = Lz.reshape(B, N)
        Mb = M.reshape(B, nmax, D, nmax, D)
        self._onsite = torch.diagonal(Mb, dim1=1, dim2=3).permute(0, 3, 1, 2)    # (B, nmax, D, D): the gap predictor's O
        if self.disp_response:
            self._gram = Gi.new_zeros(sp.shape[0], D, D).index_add(0, i, Gi).index_add(0, j, Gj)
        self._gz = gz
        rhs = torch.cat([(Lf + gz)[..., None], rhs_extra], 2)
        Lc, info = torch.linalg.cholesky_ex(M)
        Y = torch.cholesky_solve(rhs, Lc) if not bool(info.any()) else torch.linalg.solve(M, rhs)
        z = -Y[..., 0]
        zr = z.reshape(B, nmax, D)
        zi, zj = zr[fr[i], loc[i]], zr[fr[j], loc[j]]
        zHz = torch.einsum("pa,pab,pb->p", zi, Gi, zi) + torch.einsum("pa,pab,pb->p", zj, Gj, zj)
        e_el = 0.5 * (z * (A @ z[..., None])[..., 0]).sum(1) + (Lf * z).sum(1)
        e_x = (gz * z).sum(1) + 0.5 * z.new_zeros(B).index_add(0, fr[i], zHz)
        return zr, e_el, e_x, Y[..., 1:]

    def charge_flux(self, bt, sp, Ca, Hq, z_pol, W, phi_ext, dpsi=None):
        """The joint minimum with the channels eliminated: K = G diag(s^2) G^T, u = G s^2 V, (1 + K S) Q = u - K psi0
        with S = H_Q - C^T W and psi0 = phi_ext + C^T z_pol; q_c = s_c^2 (V_c - (psi_i - psi_j)), z* = z_pol - W Q.
        With the throughput hardness the elimination is onto each site's donated and accepted charge instead
        (:meth:`_flux_solve_relay`, 2N unknowns per frame).
        Returns E_CT (B,), per channel 1/2 b~_c q_c (C,), q (C,), z* (B, nmax D), Q (B, nmax), e_c (C,) and, for
        ``dpsi`` (B, nmax, k) derivatives of psi0, the derivatives of Q (B, nmax, k) (else None)."""
        B, N = Hq.shape[0], Hq.shape[1]
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        e_c, V, s = self.channel_drives(bt, sp, self._q, self.channel_curvature(bt, sp))
        w = s * s
        psi0 = Hq.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Hq - Ca.transpose(1, 2) @ W
        tau = self.slot_tau(sp)
        if tau is not None:
            Q, dQ, qc = self._flux_solve_relay(B, N, cf, li, lj, w, V, psi0, S, Hq.new_zeros(B, N).index_put((fr, loc), tau), dpsi)
        else:
            Kl = Hq.new_zeros(B, N, N)
            Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
            Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
            wV = w * V
            u = Hq.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
            Mq = torch.eye(N, dtype=Hq.dtype, device=Hq.device) + Kl @ S
            rhs = (u - (Kl @ psi0[..., None])[..., 0])[..., None]
            if dpsi is not None:
                rhs = torch.cat([rhs, -(Kl @ dpsi)], 2)
            Ys = torch.linalg.solve(Mq, rhs)
            Q = Ys[..., 0]
            dQ = Ys[..., 1:] if dpsi is not None else None
            psi = psi0 + (S @ Q[..., None])[..., 0]
            qc = w * (V - (psi[cf, li] - psi[cf, lj]))
        e_ch = 0.5 * (-V + psi0[cf, li] - psi0[cf, lj]) * qc
        z = z_pol - (W @ Q[..., None])[..., 0]
        return Hq.new_zeros(B).index_add(0, cf, e_ch), e_ch, qc, z, Q, e_c, dQ

    @staticmethod
    def _flux_solve_relay(B, N, cf, li, lj, w, V, psi0, S, tau, dpsi=None):
        """The flux minimum with the throughput hardness, the channels eliminated onto y = (D, A): D_i the charge i
        donates (its channels out), A_i the charge it accepts (its channels in), Q = D - A, T = D + A.

            q_c = w_c (V_c - mu^D_i - mu^A_j),    mu^D = S Q + psi0 + tau T,    mu^A = -(S Q + psi0) + tau T
            (1 + K2 S2) y = u2 - K2 [psi0, -psi0],    K2 = [[diag(sum_out w), w_ij], [w_ij^T, diag(sum_in w)]],
            S2 = [[S + tau, -S + tau], [-S + tau, S + tau]],    u2 = (sum_out w V, sum_in w V).

        Returns Q (B, N), dQ (B, N, k) for ``dpsi`` (else None) and the channel charges q (C,)."""
        K2 = S.new_zeros(B, 2 * N, 2 * N)
        K2 = K2.index_put((cf, li, li), w, accumulate=True).index_put((cf, N + lj, N + lj), w, accumulate=True)
        K2 = K2.index_put((cf, li, N + lj), w, accumulate=True).index_put((cf, N + lj, li), w, accumulate=True)
        wV = w * V
        u2 = S.new_zeros(B, 2 * N).index_put((cf, li), wV, accumulate=True).index_put((cf, N + lj), wV, accumulate=True)
        Td = torch.diag_embed(tau)
        S2 = torch.cat([torch.cat([S + Td, Td - S], 2), torch.cat([Td - S, S + Td], 2)], 1)
        psi2 = torch.cat([psi0, -psi0], 1)
        M2 = torch.eye(2 * N, dtype=S.dtype, device=S.device) + K2 @ S2
        rhs = (u2 - (K2 @ psi2[..., None])[..., 0])[..., None]
        if dpsi is not None:
            rhs = torch.cat([rhs, -(K2 @ torch.cat([dpsi, -dpsi], 1))], 2)
        Ys = torch.linalg.solve(M2, rhs)
        y = Ys[..., 0]
        mu = psi2 + (S2 @ y[..., None])[..., 0]
        qc = w * (V - mu[cf, li] - mu[cf, N + lj])
        dQ = (Ys[:, :N, 1:] - Ys[:, N:, 1:]) if dpsi is not None else None
        return y[:, :N] - y[:, N:], dQ, qc

    # -- the gap model's channels -----------------------------------------------------------------------------
    def acceptor_shift(self, bt, sp):
        """(N,) the Pauli shift of each atom's acceptor level by its neighbours' closed shells.
        ``ct_gap_overlap = "pauli"`` (default): c[s_j] sum_k (E0_jk / O_jk) S_A(j <- k) over the exchange pairs --
        S_A(j <- k) the overlap of j's normalized acceptor probe with k's electron density, O_jk the overlap of the
        two ions' electron densities and E0_jk the pair's frozen Pauli energy (with its environment factors), so
        E0 / O is that pair's Pauli energy per unit overlap (the xpol anchoring, with its overlap switch) and the
        shift is the Pauli energy the pair would have with one electron in j's acceptor shape in place of j's
        density, times c. :meth:`frozen_terms` comes first.
        ``"probe"``: kappa[s_j] S_A(j), the probe's overlap with all neighbours (Hartree per unit overlap)."""
        k = self.overlap_strength()[sp]
        if self.config.ct_gap_overlap == "probe":
            return k * bt["S0_atom"][:, 1]
        xs = bt["x_sel"]
        pi, pj = bt["pairs"][:, xs]
        ratio = self._pauli_pairs[1][xs] * bt["x_swO"]
        Px = xs.shape[0]
        Sa = bt["x_Sa"]
        return k * ratio.new_zeros(sp.shape[0]).index_add(0, pi, ratio * Sa[:Px]).index_add(0, pj, ratio * Sa[Px:])

    def overlap_strength(self):
        """(S,) the acceptor shift's strength per acceptor species: GAP_KOV_REF exp(c tanh(ct_ln_kov / c)),
        c = ln GAP_KOV_CAP, so within a factor GAP_KOV_CAP of the reference whatever the fit does (free, it trades
        the shift against the CT charges until the channel is a rigid W^2 / Delta term)."""
        c = math.log(GAP_KOV_CAP)
        return GAP_KOV_REF[self.config.ct_gap_overlap] * torch.exp(c * torch.tanh(self.ct_ln_kov / c))

    def gap_channels(self, bt, sp):
        """Per channel c = i -> j of the gap model: the coupling W2 = W^2, the gap Delta, the lone channel's a = -E
        and charge qs = q* = -da/dDelta, and den = rt^3 - 2 W^2 h = 2 W^2 (k - h) with h the channel's own share of
        G^T H_Q G (so w = 2 W^2 / den is 1 / (k - h), finite as W -> 0). The crossing has the width
        W_b^2 = W^2 + W0^2 (``ct_gap_w0``) and the amplitude lam = W^2 / W_b^2:

            a = lam [sqrt(Delta^2 / 4 + W_b^2) - Delta / 2],   rt = sqrt(Delta^2 + 4 W_b^2),   k = rt^3 / (2 W^2),

        the two-state channel at W0 = 0 and W^2 / Delta at a large gap for any W0; a channel transfers at most lam
        and responds with at most W^2 / (4 W_b^3), so one whose gap closes where W is small stays inert."""
        i, j = bt["ch_index"]
        q = self._q
        if self.frontier:                # the frontier orbitals' overlap with the pair's own cutoff (ct_overlap.py)
            W2 = q[i, 0] * q[j, 1] * overlap_kernel(self.ctk_coef, self.ctk_grid, self.ctk_ron, self.ctk_roff,
                                                    sp[i], sp[j], bt["ch_r"])
        else:
            bD, bA = self.ct_exponents()
            rb = bt["ch_r"] / K.BOHR_ANG
            f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
            W2 = q[i, 0] * q[j, 1] * f / rb
            if self.overlap_cut:         # the pair's own switch in place of the table's taper
                W2 = W2 * pair_switch(self.ctk_ron, self.ctk_roff, sp[i], sp[j], bt["ch_r"]) ** 2
            else:
                W2 = W2 * torch.exp(bt["ch_ltaper"])
            W2 = W2.clamp(min=0.0)
        phi = bt["phi_front"]
        Delta = (self.ct_gap_I[sp[i]] - self.ct_gap_A[sp[j]] + phi[i, 0] - phi[j, 1] - bt["ch_J"]
                 + self.acceptor_shift(bt, sp)[j])
        h = self.channel_curvature(bt, sp)
        return {"W2": W2, "Delta": Delta, "h": h, **self.two_state(W2, Delta, h)}

    def two_state(self, W2, Delta, h):
        """The broadened two-state forms of :meth:`gap_channels` at the gap ``Delta``: {a, qs, den}."""
        w02 = self.config.ct_gap_w0 ** 2
        Wb2 = W2 + w02
        rt = torch.sqrt(Delta * Delta + 4 * Wb2 + 1e-60)
        pos = Delta > 0
        dp = torch.where(pos, rt + Delta, torch.ones_like(rt))            # the unselected branch stays finite
        lam = W2 / Wb2 if w02 > 0 else torch.ones_like(W2)
        a = torch.where(pos, 2 * W2 / dp, lam * 0.5 * (rt - Delta))
        qs = torch.where(pos, 2 * W2 / (rt * dp), lam * 0.5 * (1 - Delta / rt))
        return {"a": a, "qs": qs, "den": rt ** 3 - 2 * W2 * h}

    def expand_at(self, g, delta):
        """The gap channels ``g`` (:meth:`gap_channels`, frozen gap Delta) expanded about q0 = q*(Delta + delta): with
        F(q) = Delta q - 2 W_b sqrt(q (lam - q)) a channel's two-state energy, F(q0) = -a(Delta') - delta q0,
        F'(q0) = -delta and F''(q0) = k(Delta'). Returns g with a, qs = q0, den at Delta' = Delta + delta, ``delta``
        and ``Delta0`` = the frozen gap (``Delta`` becomes Delta')."""
        D0 = g.get("Delta0", g["Delta"])
        Dp = D0 + delta
        return {**g, **self.two_state(g["W2"], Dp, g["h"]), "Delta": Dp, "Delta0": D0, "delta": delta}

    def charge_flux_gap(self, bt, sp, Ca, Hq, z_pol, W, phi_ext, dpsi=None, pred=None):
        """The joint minimum of the gap model, solved for the channels' departures d = q - q0 from the charges q0 they
        are expanded about. With Q0 = G q0, S = H_Q - C^T W, psi0 = phi_ext + C^T z_pol and psi* = psi0 + S Q0, the
        energy is

            E* + sum_c [1/2 (k_c - h_c) d_c^2 - V_c d_c] + 1/2 dQ^T S dQ,   V_c = h_c q0_c - (psi*_i - psi*_j) + delta_c,
            E* = sum_c (-a_c - delta_c q0_c - 1/2 h_c q0_c^2) + 1/2 Q0^T S Q0 + Q0^T psi0,

        which is the flux's problem with compliance k - h, drive V and no psi0 (:meth:`_flux_solve_mixed`); a, k and
        q0 = q* are the two-state forms at the gap Delta + delta (:meth:`expand_at`). ``ct_gap_predict`` passes set
        delta_c = (psi_i - psi_j) - h_c q0_c from the previous q0, psi = H_Q Q0 + C^T z0 with z0 = -O^-1 (L + g + C Q0)
        the direct (uncoupled) moments of the field-free drive and the charges Q0 (O the on-site blocks of A + H;
        ``pred`` = (O^-1, -O^-1 (L + g))): the gap the channel sees in the polarized environment and the other
        channels' charges, so that V is small and the expansion holds where the frozen ions alone close a gap that
        polarization reopens. Only the expansion point uses the direct moments; the solve about it is the coupled
        one. delta is explicit and keeps its graph (exact forces, the same under PBC).
        Returns as :meth:`charge_flux`; e_kernel is the lone channels' -a_c."""
        B, N = Hq.shape[0], Hq.shape[1]
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        psi0 = Hq.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Hq - Ca.transpose(1, 2) @ W

        def slots(q):
            return Hq.new_zeros(B, N).index_put((cf, li), q, accumulate=True).index_put((cf, lj), -q, accumulate=True)

        g = self.gap_channels(bt, sp)
        delta = torch.zeros_like(g["Delta"])
        nconv = self.config.ct_gap_converge
        if self.config.ct_gap_predict:
            Oinv, z_perm = pred
            Dz = Oinv.shape[-1]
        with torch.set_grad_enabled(torch.is_grad_enabled() and not nconv):      # converging: the start only
            for _ in range(self.config.ct_gap_predict):
                Q0 = slots(g["qs"])
                CQ = (Ca @ Q0[..., None])[..., 0].reshape(B, -1, Dz)
                z0 = z_perm - torch.einsum("bnac,bnc->bna", Oinv, CQ)
                ps = (Hq @ Q0[..., None])[..., 0] + torch.einsum("bnk,bn->bk", Ca, z0.reshape(B, -1))
                delta = (ps[cf, li] - ps[cf, lj]) - g["h"] * g["qs"]
                g = self.expand_at(g, delta)

        def _dl(gx):                                                      # the frozen-gap expansion has no shift
            return gx["delta"] if "delta" in gx else torch.zeros_like(gx["Delta"])

        def step(gx, dps=None):
            """The coupled solve about the charges of ``gx``: the residual drive V, the channels' departures d and
            dQ = G d (with ``dps`` the extra right-hand sides), and the expansion charges' Q0, S Q0."""
            qs_, h_, W2_, den_ = gx["qs"], gx["h"], gx["W2"], gx["den"]
            Qs_ = slots(qs_)
            SQs_ = (S @ Qs_[..., None])[..., 0]
            psis = psi0 + SQs_
            V_ = h_ * qs_ - (psis[cf, li] - psis[cf, lj]) + _dl(gx)
            good = den_ > 2 * W2_ * GAP_B_EPS                             # k - h > GAP_B_EPS (W = 0: rigid, w = 0)
            w = torch.where(good, 2 * W2_ / torch.where(good, den_, torch.ones_like(den_)), torch.zeros_like(den_))
            # (the unselected side of a where must stay finite to second order: a channel at the end of its switch
            # has W^2 = 0 or 1e-200, and 1 / W^2 there turns the force loss's gradient into nan)
            b_bad = den_ / torch.where(good, torch.ones_like(den_), (2 * W2_).clamp(min=1e-300))
            Vc = V_[:, None] if dps is None else torch.cat([V_[:, None], -(dps[cf, li] - dps[cf, lj])], 1)
            dQ_, d_ = self._flux_solve_mixed(B, N, cf, li, lj, w, ~good, b_bad, Vc, S)
            return {"V": V_, "d": d_, "dQ": dQ_, "Qs": Qs_, "SQs": SQs_, "good": good}

        def newton(gx, st):
            """k_c d_c: minus the Newton change of the gap shift delta (q* moves by d to first order), finite for a
            channel without coupling (there it is the plain update delta <- (psi_i - psi_j) - h q)."""
            W2_, den_ = gx["W2"], gx["den"]
            rt3 = den_ + 2 * W2_ * gx["h"]
            SdQ = (S @ st["dQ"][..., :1])[..., 0]
            lin = st["V"] - (SdQ[cf, li] - SdQ[cf, lj])
            return torch.where(st["good"], rt3 / torch.where(st["good"], den_, torch.ones_like(den_)) * lin,
                               rt3 / torch.where(st["good"], torch.ones_like(den_), (2 * W2_).clamp(min=1e-300))
                               * st["d"][:, 0])

        def energy(gx):
            """(B,) the energy at the expansion charges themselves (the moments relaxed): E* of the docstring."""
            qs_ = gx["qs"]
            SQ = (S @ slots(qs_)[..., None])[..., 0]
            e = (-gx["a"] - _dl(gx) * qs_ - 0.5 * gx["h"] * qs_ * qs_
                 + qs_ * (0.5 * (SQ[cf, li] - SQ[cf, lj]) + psi0[cf, li] - psi0[cf, lj]))
            return Hq.new_zeros(B).index_add(0, cf, e)

        self._gap_info = None
        if self.config.ct_exclusion and cf.numel():
            return self._flux_excl(g, B, N, cf, li, lj, psi0, S, dpsi, z_pol, W)
        if nconv and cf.numel():
            delta = self._converge_gap(g, delta.detach(), step, newton, energy, B, cf)
            g0 = {k: v for k, v in g.items() if k not in ("delta", "Delta0")} | {"Delta": g.get("Delta0", g["Delta"])}
            gx = self.expand_at(g0, delta)
            nd = -newton(gx, step(gx))                    # one differentiable Newton step: the implicit derivative
            g = self.expand_at(g0, delta + (nd - nd.detach()))
        elif nconv:
            g = self.expand_at(g, delta)
        self._gap = g
        a, qs, h = g["a"], g["qs"], g["h"]
        st = step(g, dpsi)
        V, dQ, d, SQs = st["V"], st["dQ"], st["d"], st["SQs"]
        dlt = d[:, 0]
        e_ch = (-a - _dl(g) * qs - 0.5 * h * qs * qs + qs * (0.5 * (SQs[cf, li] - SQs[cf, lj]) + psi0[cf, li] - psi0[cf, lj])
                - 0.5 * V * dlt)
        Q = st["Qs"] + dQ[..., 0]
        z = z_pol - (W @ Q[..., None])[..., 0]
        return (Hq.new_zeros(B).index_add(0, cf, e_ch), e_ch, qs + dlt, z, Q, -a,
                (dQ[..., 1:] if dpsi is not None else None))

    # -- exclusion: a site's channels share its capacity ---------------------------------------------------------
    @staticmethod
    def _excl_solve(A, c, X, rhs, live, cf, p, r, B, N):
        """x = H^-1 rhs (C, k) for the exclusion functional's Hessian in its site form,

            H = diag(A) + U X U^T + diag(c) U U^T + U U^T diag(c),

        U (C, 2N per frame) the channels' incidence on [donor sites, acceptor sites] (``p``, ``r`` = N + acceptor: the
        local columns), X (B, 2N, 2N), without ever forming a channel-by-channel matrix: with y1 = U^T x and
        y2 = U^T (c x) the channels are eliminated, x = (rhs - U (X y1 + y2) - c U y1) / A, onto 4N site unknowns per
        frame. A channel whose A is (nearly) zero cannot be divided out and stays an explicit unknown (exact for any
        split; N4 + the largest number of such channels in a frame). Channels that are not ``live`` return 0. Also
        returns ``info`` (B,) != 0 where a frame's system is singular."""
        lv = live.to(rhs.dtype)
        bad = live & (A.abs() < 1e-9)
        good = live & ~bad
        w = torch.where(good, 1.0 / torch.where(good, A, torch.ones_like(A)), torch.zeros_like(A))
        N2, N4 = 2 * N, 4 * N

        def site2(v):                                   # U^T diag(v) U
            M = X.new_zeros(B, N2, N2)
            return (M.index_put((cf, p, p), v, accumulate=True).index_put((cf, r, r), v, accumulate=True)
                    .index_put((cf, p, r), v, accumulate=True).index_put((cf, r, p), v, accumulate=True))

        def Ut(v):                                      # U^T v, v (C, k)
            return X.new_zeros(B, N2, v.shape[1]).index_put((cf, p), v, accumulate=True).index_put((cf, r), v, accumulate=True)

        def Uv(y):                                      # U y, y (B, 2N, k)
            return (y[cf, p] + y[cf, r]) * lv[:, None]

        K0, K1, K2 = site2(w), site2(w * c), site2(w * c * c)
        eye = torch.eye(N2, dtype=X.dtype, device=X.device)
        T = torch.cat([torch.cat([eye + K0 @ X + K1, K0], 2), torch.cat([K1 @ X + K2, eye + K1], 2)], 1)
        wb = w[:, None] * rhs
        ry = torch.cat([Ut(wb), Ut(c[:, None] * wb)], 1)
        nb = int(bad.sum())
        if nb:
            ib = bad.nonzero().squeeze(1)
            fb, pb_, rb = cf[ib], p[ib], r[ib]
            order = torch.argsort(fb, stable=True)
            counts = torch.bincount(fb, minlength=B)
            rank = torch.empty_like(ib)
            rank[order] = torch.arange(nb, device=ib.device) - (torch.cumsum(counts, 0) - counts)[fb[order]]
            nm = int(counts.max())
            col = N4 + rank
            Te = X.new_zeros(B, N4 + nm, N4 + nm)
            Te[:, :N4, :N4] = T
            one, cb = torch.ones_like(A[ib]), c[ib]
            Te = (Te.index_put((fb, pb_, col), -one, accumulate=True).index_put((fb, rb, col), -one, accumulate=True)
                  .index_put((fb, N2 + pb_, col), -cb, accumulate=True).index_put((fb, N2 + rb, col), -cb, accumulate=True))
            rows = X.new_zeros(B, nm, N4)
            rows[fb, rank, :N2] = X[fb, pb_] + X[fb, rb]
            rows = (rows.index_put((fb, rank, pb_), cb, accumulate=True).index_put((fb, rank, rb), cb, accumulate=True)
                    .index_put((fb, rank, N2 + pb_), one, accumulate=True).index_put((fb, rank, N2 + rb), one, accumulate=True))
            diag = X.new_ones(B, nm).index_put((fb, rank), A[ib])
            Te = torch.cat([Te[:, :N4], torch.cat([rows, torch.diag_embed(diag)], 2)], 1)
            ry = torch.cat([ry, X.new_zeros(B, nm, rhs.shape[1]).index_put((fb, rank), rhs[ib])], 1)
            Y, info = torch.linalg.solve_ex(Te, ry)
        else:
            Y, info = torch.linalg.solve_ex(T, ry)
        y1, y2 = Y[:, :N2], Y[:, N2:N4]
        x = w[:, None] * (rhs - Uv(X @ y1 + y2) - c[:, None] * Uv(y1))
        if nb:
            x = x.index_put((ib,), Y[:, N4:][fb, rank])
        return x * lv[:, None], info

    @staticmethod
    def _excl_inertia(A, c, X, live, cf, p, r, B, N):
        """(B,) True where the exclusion Hessian of :meth:`_excl_solve` (diagonal ``A``) is positive definite, from
        the inertia of its site form: with K_m = U^T diag(c^m / A) U, the number of negative eigenvalues of H is
        #neg(A) + #neg(-[[K0, 1 + K1], [1 + K1, K2 - X]]) - 2N."""
        lv = live.to(X.dtype)
        Ar = torch.where(A.abs() < 1e-12, torch.full_like(A, 1e-12), A)
        w = lv / Ar
        N2 = 2 * N

        def site2(v):
            M = X.new_zeros(B, N2, N2)
            return (M.index_put((cf, p, p), v, accumulate=True).index_put((cf, r, r), v, accumulate=True)
                    .index_put((cf, p, r), v, accumulate=True).index_put((cf, r, p), v, accumulate=True))

        K0, K1, K2 = site2(w), site2(w * c), site2(w * c * c)
        eye = torch.eye(N2, dtype=X.dtype, device=X.device)
        cap = -torch.cat([torch.cat([K0, eye + K1], 2), torch.cat([eye + K1, K2 - X], 2)], 1)
        cap = 0.5 * (cap + cap.transpose(1, 2))
        fin = torch.isfinite(cap).all(-1).all(-1)
        try:
            ev = torch.linalg.eigvalsh(torch.where(fin[:, None, None], cap, torch.zeros_like(cap)))
        except RuntimeError:                    # no eigenvalues: not known to be positive definite
            return torch.zeros(B, dtype=torch.bool, device=X.device)
        nneg = (ev < 0).sum(1) + torch.zeros(B, dtype=torch.long, device=X.device).index_add(0, cf, (live & (Ar < 0)).long())
        return (nneg == N2) & fin

    def _flux_excl(self, g, B, N, cf, li, lj, psi0, S, dpsi, z_pol, W):
        """The joint minimum with the exclusion functional (``config.ct_exclusion``) in place of the independent
        two-state channels: every channel c = i -> j has the capacity its site-mates leave,

            F_c = Delta_c q_c - 2 W_b sqrt(q_c n_c),     n_c = lam_c (1 - b_c) - q_c,
            b_c = (the charge of the other channels out of i) + (that of the other channels into j),

        so a site gives at most one electron and takes at most one. sum_c F_c is jointly convex on its polytope and
        its minimizer is interior; for the channels of one site alone it is the exact ground-state energy of the
        configurations {no transfer, transfer c} (the lowest eigenvalue of the arrowhead matrix with diagonal
        (0, Delta_c) and border W_c). The electrostatics of the transferred charges is unchanged (mean field,
        1/2 Q^T S Q - 1/2 sum_c h_c q_c^2 + Q^T psi0).

        The functional is not separable, but its Hessian is a diagonal plus terms that act through the sites,
        kappa V^T V with V = diag(n + q) + diag(q lam) (U U^T - 2) and kappa = W_b / (2 (q n)^(3/2)), so every solve
        is 4N site unknowns per frame (:meth:`_excl_solve`) and everything else is sums over the channels. Newton
        from the two-state charges of the predicted gaps scaled into the polytope: in the amplitudes x = sqrt(q)
        where that Hessian is positive definite (there a weakly filled channel is quadratic, Delta x^2 - 2 W sqrt(n)
        x; in q it is a square root that Newton climbs only threefold per step), else in q with the convex part of
        the Hessian (no -h; a channel whose gap the potentials have closed has a maximum at x = 0), each step cut
        to stay inside and to lower the energy. One differentiable Newton step from the converged charges carries
        their dependence on positions and parameters (exact first to third derivatives: forces, polarizabilities,
        the losses on both). Returns as :meth:`charge_flux_gap`; ``self._gap`` has the converged charges (``qs``),
        the two-state gap they satisfy (``Delta`` = s - r: the frozen gap + the potential difference + the
        blocking by the site-mates) and the capacity left (``left``)."""
        cfg = self.config
        dt = S.dtype
        p, r = li, N + lj
        # the global site of each channel's donor and acceptor (frame * N + local)
        gi, gj = cf * N + li, cf * N + lj
        W2, D0, h = g["W2"], g.get("Delta0", g["Delta"]), g["h"]
        w02 = cfg.ct_gap_w0 ** 2
        Wb = torch.sqrt(W2 + w02)
        lam = W2 / (W2 + w02) if w02 > 0 else torch.ones_like(W2)
        live = W2.detach() > 1e-24
        lv = live.to(dt)
        gp = psi0[cf, li] - psi0[cf, lj]
        S2 = torch.cat([torch.cat([S, -S], 2), torch.cat([-S, S], 2)], 1)
        big = torch.full_like(lv, float("inf"))

        def site(v):                            # (B N,) the sums over the channels out of / into every site
            z0 = v.new_zeros(B * N)
            return z0.index_add(0, gi, v), z0.index_add(0, gj, v)

        def mates(v):                           # the sum over the other channels of c's donor and of its acceptor
            dsum, asum = site(v)
            return dsum[gi] + asum[gj] - 2 * v

        def per(v):
            return v.new_zeros(B).index_add(0, cf, v)

        def fmin(v):
            return v.new_full((B,), float("inf")).scatter_reduce(0, cf, torch.where(live, v, big), "amin")

        def terms(q, Wb_, lam_, D0_, h_, S_, gp_):
            nn = lam_ * (1 - mates(q)) - q
            qs, ns = torch.where(live, q, torch.ones_like(q)), torch.where(live, nn, torch.ones_like(nn))
            rr = Wb_ * torch.sqrt(qs / ns) * lv
            ss = Wb_ * torch.sqrt(ns / qs) * lv
            dsum, asum = site(q)
            SQ = (S_ @ (dsum - asum).reshape(B, N, 1))[..., 0]
            sq = SQ[cf, li] - SQ[cf, lj]
            e = (D0_ * q - 2 * Wb_ * torch.sqrt(qs * ns) * lv - 0.5 * h_ * q * q + q * (0.5 * sq + gp_)) * lv
            gr = (D0_ - ss + rr + mates(lam_ * rr) - h_ * q + sq + gp_) * lv
            return nn, rr, ss, e, gr

        def hess(q, nn, Wb_, lam_, h_, S2_):
            """The site form of the Hessian: the diagonal a (so A = a - h), c, the site matrix X and diag(kappa V^T V)."""
            qs, ns = torch.where(live, q, torch.ones_like(q)), torch.where(live, nn, torch.ones_like(nn))
            kap = Wb_ / (2 * (qs * ns) ** 1.5) * lv
            ql = q * lam_
            dv = nn + q - 2 * ql
            e2 = kap * ql * ql
            X = S2_.new_zeros(B, 2 * N, 2 * N)
            X = (X.index_put((cf, p, p), e2, accumulate=True).index_put((cf, r, r), e2, accumulate=True)
                 .index_put((cf, p, r), e2, accumulate=True).index_put((cf, r, p), e2, accumulate=True)) + S2_
            return kap * dv * dv, kap * dv * ql, X, kap * (nn + q) ** 2 + mates(e2)

        def solve(A, c, X, rhs):
            A = torch.where(live, A, torch.ones_like(A))
            return self._excl_solve(A, c * lv, X, rhs * lv[:, None], live, cf, p, r, B, N)

        det = lambda *v: tuple(t.detach() for t in v)                                      # noqa: E731
        Pd = det(Wb, lam, D0, h, S, gp)
        S2d = S2.detach()
        with torch.no_grad():
            q = g["qs"].detach() * lv
            crowd = (torch.where(live, q / Pd[1].clamp(min=1e-300), torch.zeros_like(q)) + mates(q)) * lv
            q = q * (0.8 / q.new_zeros(B).scatter_reduce(0, cf, crowd, "amax", include_self=True).clamp(min=0.8))[cf]
            nn, rr, ss, e, gr = terms(q, *Pd)
            E = per(e)
            it = 0
            amax = lambda v: v.new_zeros(B).scatter_reduce(0, cf, v, "amax", include_self=True)     # noqa: E731
            res = amax(gr.abs())
            while it < cfg.ct_gap_converge:
                active = res > cfg.ct_gap_tol
                if not bool(active.any()):
                    break
                it += 1
                a, c, X, dU = hess(q, nn, Pd[0], Pd[1], Pd[3], S2d)
                # Newton in x = sqrt(q): (D H D + 2 diag(g)) dx = -D g with D = 2x, i.e. u = D dx solves
                # (H + diag(g / 2q)) u = -g
                qs = torch.where(live, q, torch.ones_like(q))
                Ax = torch.where(live, a - Pd[3] + gr / (2 * qs), torch.ones_like(a))
                u, info = solve(Ax, c, X, -gr[:, None])
                u = u[:, 0]
                x = torch.sqrt(qs)
                dx = u / (2 * x)
                slope = per(gr * u)
                newt = (self._excl_inertia(Ax, c * lv, X, live, cf, p, r, B, N) & (info == 0) & (slope < 0)
                        & torch.isfinite(slope))
                dx = torch.where(newt[cf], dx, torch.zeros_like(dx))
                d = torch.zeros_like(q)
                if not bool((newt | ~active).all()):     # elsewhere: in q, the convex part of the Hessian (no -h)
                    d, info2 = solve(a, c, X, -gr[:, None])
                    d = d[:, 0]
                    sl2 = per(gr * d)
                    conv = (info2 == 0) & (sl2 < 0) & torch.isfinite(sl2)
                    d = torch.where(conv[cf], d, -gr / dU.clamp(min=1e-300))
                    slope = torch.where(newt, slope, torch.where(conv, sl2, per(-gr * gr / dU.clamp(min=1e-300))))
                t = torch.ones_like(E)
                done, moved, q_new = ~active, torch.zeros_like(active), q
                for _ in range(60):
                    tc = t[cf]
                    qt = torch.where(newt[cf], (x + tc * dx) ** 2 * lv, q + tc * d)
                    nt, _, _, et, _ = terms(qt, *Pd)
                    inside = (fmin(torch.minimum(nt - 0.1 * nn, qt - 0.01 * q)) > 0) & (fmin(x + tc * dx) > 0)
                    ok = inside & (per(et) <= E + 1e-4 * t * slope + 1e-13 * (1 + E.abs())) & ~done
                    q_new = torch.where(ok[cf], qt, q_new)
                    done, moved = done | ok, moved | ok
                    if bool(done.all()):
                        break
                    t = torch.where(done, t, 0.5 * t)
                if not bool(moved.any()):
                    break
                q = q_new
                nn, rr, ss, e, gr = terms(q, *Pd)
                E, res = per(e), amax(gr.abs())
        self._gap_info = {"iters": it, "resid": res, "converged": res <= cfg.ct_gap_tol}
        # one differentiable Newton step from the converged charges (its value added as zero)
        nn, rr, ss, e, gr = terms(q, Wb, lam, D0, h, S, gp)
        a, c, X, _ = hess(q, nn, Wb, lam, h, S2)
        nd = -solve(a - h, c, X, gr[:, None])[0][:, 0]
        nd = torch.where((res <= 1e3 * cfg.ct_gap_tol)[cf], nd, torch.zeros_like(nd))    # not where it did not converge
        q1 = q + (nd - nd.detach())
        n1, r1, s1, e1, _ = terms(q1, Wb, lam, D0, h, S, gp)
        dQ = None
        if dpsi is not None:                                  # the tangent response to the fields' potentials
            a1, c1, X1, _ = hess(q1, n1, Wb, lam, h, S2)
            dq = -solve(a1 - h, c1, X1, dpsi[cf, li] - dpsi[cf, lj])[0]
            dQ = (S.new_zeros(B * N, dq.shape[1]).index_add(0, gi, dq).index_add(0, gj, -dq)).reshape(B, N, -1)
        dsum, asum = site(q1)
        Q = (dsum - asum).reshape(B, N)
        z = z_pol - (W @ Q[..., None])[..., 0]
        gap = s1 - r1
        self._gap = {"W2": W2, "h": h, "Delta0": D0, "Delta": gap, "delta": gap - D0, "qs": q1.detach(), "left": n1}
        a0 = self.two_state(W2, D0, h)["a"]
        return per(e1), e1, q1, z, Q, -a0, dQ

    def _converge_gap(self, g, delta, step, newton, energy, B, cf):
        """The self-consistent gap shifts (C,), no graph: Newton on the gap equation V_c(delta) = 0 in the variable
        delta (so every channel's charge q*(Delta + delta) stays a two-state charge, inside (0, lam)), from the start
        ``delta``. One coupled solve per iteration (``step``); the update -k d (``newton``) is taken per frame with
        the largest step 1, 1/2, 1/4, ... that lowers the energy at the expansion charges (``energy``: no solve), and
        where it is not a descent direction (a channel softer than its coupling: k - h + G^T S G indefinite) the plain
        update delta <- (psi_i - psi_j) - h q is taken instead, which always is one. Stops when every coupled
        channel's |V_c| is below ``ct_gap_tol``, after ``ct_gap_converge`` iterations, or when no frame can lower its
        energy further; ``self._gap_info`` = {iters, resid (B,), converged (B,)}."""
        cfg = self.config
        g0 = {k: v.detach() for k, v in g.items() if k not in ("delta", "Delta0")} | {"Delta": g.get("Delta0", g["Delta"]).detach()}
        live = g0["W2"] > 0
        zc = torch.zeros_like(delta)

        def per(v):
            return delta.new_zeros(B).index_add(0, cf, v)

        with torch.no_grad():
            gx = self.expand_at(g0, delta)
            st, E = step(gx), energy(gx)
            it = 0
            while it < cfg.ct_gap_converge:
                V = st["V"]
                r = delta.new_zeros(B).scatter_reduce(0, cf, torch.where(live, V.abs(), zc), "amax", include_self=True)
                active = r > cfg.ct_gap_tol
                if not bool(active.any()):
                    break
                it += 1
                slope = per(-V * st["d"][:, 0])                                   # dE/dt along the Newton step
                desc = slope < 0
                kk = (gx["den"] + 2 * gx["W2"] * gx["h"]) / (2 * gx["W2"]).clamp(min=1e-300)
                step_d = torch.where(desc[cf], -newton(gx, st), -V)
                slope = torch.where(desc, slope, per(torch.where(live, -V * V / kk, zc)))
                t = delta.new_ones(B)
                done, moved, new_delta = ~active, torch.zeros_like(active), delta
                for _ in range(40):
                    trial = delta + t[cf] * step_d
                    Et = energy(self.expand_at(g0, trial))
                    ok = (Et <= E + 1e-4 * t * slope + 1e-13 * (1 + E.abs())) & ~done
                    new_delta = torch.where(ok[cf], trial, new_delta)
                    done, moved = done | ok, moved | ok
                    if bool(done.all()):
                        break
                    t = torch.where(done, t, 0.5 * t)
                if not bool(moved.any()):
                    break
                delta = new_delta
                gx = self.expand_at(g0, delta)
                st, E = step(gx), energy(gx)
            r = delta.new_zeros(B).scatter_reduce(0, cf, torch.where(live, st["V"].abs(), zc), "amax", include_self=True)
        self._gap_info = {"iters": it, "resid": r, "converged": r <= cfg.ct_gap_tol}
        return delta

    @staticmethod
    def _flux_solve_mixed(B, N, cf, li, lj, w, bad, b_bad, V, S):
        """The minimum of sum_c [1/2 b_c q_c^2 - V_c q_c] + 1/2 Q^T S Q (Q = G q) for right-hand sides V (C, m).
        The channels with ``bad`` False are eliminated onto the atomic charges with w = 1 / b (K = G w G^T); the
        ``bad`` ones (b = ``b_bad`` small, zero or negative, where 1 / b is not usable) stay explicit:

            (1 + K S) Q - G_b q_b = G w V,        G_b^T S Q + b_b q_b = V_b.

        Exact for any split; N + (the largest number of explicit channels in a frame) unknowns per frame.
        Returns Q (B, N, m) and q (C, m)."""
        m = V.shape[1]
        Kl = S.new_zeros(B, N, N)
        Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
        Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
        wV = w[:, None] * V
        u = S.new_zeros(B, N, m).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
        M = torch.eye(N, dtype=S.dtype, device=S.device) + Kl @ S
        nb = int(bad.sum())
        if nb == 0:
            Q = torch.linalg.solve(M, u)
            SQ = S @ Q
            return Q, wV - w[:, None] * (SQ[cf, li] - SQ[cf, lj])
        ib = bad.nonzero().squeeze(1)
        fb, lib, ljb = cf[ib], li[ib], lj[ib]
        order = torch.argsort(fb, stable=True)
        counts = torch.bincount(fb, minlength=B)
        rank = torch.empty_like(ib)
        rank[order] = torch.arange(nb, device=ib.device) - (torch.cumsum(counts, 0) - counts)[fb[order]]
        nm = int(counts.max())
        T = S.new_zeros(B, N + nm, N + nm)
        T[:, :N, :N] = M
        T = T.index_put((fb, lib, N + rank), -torch.ones_like(b_bad[ib]), accumulate=True)
        T = T.index_put((fb, ljb, N + rank), torch.ones_like(b_bad[ib]), accumulate=True)
        rows = S.new_zeros(B, nm, N).index_put((fb, rank), S[fb, lib] - S[fb, ljb])
        diag = S.new_ones(B, nm).index_put((fb, rank), b_bad[ib])
        T = torch.cat([T[:, :N], torch.cat([rows, torch.diag_embed(diag)], 2)], 1)
        rhs = torch.cat([u, S.new_zeros(B, nm, m).index_put((fb, rank), V[ib])], 1)
        Y = torch.linalg.solve(T, rhs)
        Q = Y[:, :N]
        SQ = S @ Q
        q = wV - w[:, None] * (SQ[cf, li] - SQ[cf, lj])
        return Q, q.index_put((ib,), Y[:, N:][fb, rank])

    # -- the model ---------------------------------------------------------------------------------------------
    def forward(self, bt: dict, field=None, alpha: bool = False) -> dict:
        """``bt`` from :meth:`precompute`; ``field`` (B, 3) a uniform external field per frame (atomic units).
        Returns the energy terms (B,) (Hartree) and the solution; with ``alpha`` also the polarizability per frame
        (B, 3, 3) [induced dipole, field] (bohr^3) of the joint minimum (``alpha``) and of the CT-free one
        (``alpha_pol``): the energy is quadratic in the field, so these are exact (extra right-hand sides of the
        same factorizations)."""
        if self.hyper_sat:
            bt = self.hyperpolarized(bt, bt["species"])
        if bt.get("pbc") is not None:
            if field is not None or alpha:
                raise NotImplementedError("cmm under PBC: no uniform field / polarizability (tin-foil cells)")
            return self._forward_pbc(bt)
        sp = bt["species"]
        self.environment(bt)
        B, nmax, D, mask = bt["B"], bt["nmax"], bt["D"], bt["mask"]
        if D != 8 + self.n_b:
            raise ValueError(f"the batch's response layout D = {D} is not the model's {8 + self.n_b}")
        n = sp.shape[0]
        dt, dev = bt["pos"].dtype, bt["pos"].device
        fr, loc = self._local(bt)
        Lz = bt["Lz"]
        phi_ext = torch.zeros(n, dtype=dt, device=dev)
        Lz_free = Lz
        if field is not None:
            Fp = field[:, None, :] * mask[..., None].to(dt)
            Lz = Lz - torch.cat([Fp, Fp.new_zeros(B, nmax, D - 3)], 2)
            phi_ext = -(field[fr] * bt["pos"] / K.BOHR_ANG).sum(-1)
        e_pauli, e_disp, frz = self.frozen_terms(bt, sp)
        self._q = self.ct_charges(sp)
        nC = bt["ch_index"].shape[1]
        Ca = bt["Ca"]
        Hq = bt["Jq"].index_put((fr, loc, loc), nn.functional.softplus(self.ct_eta_raw[sp]), accumulate=True)
        rhs = Ca
        if alpha:                       # unit fields along x, y, z on every dipole
            Ef = torch.zeros(B, nmax, D, 3, dtype=dt, device=dev)
            Ef[:, :, :3, :] = torch.eye(3, dtype=dt, device=dev) * mask[..., None, None].to(dt)
            rhs = torch.cat([Ca, Ef.reshape(B, nmax * D, 3)], 2)
        z_pol, e_el, e_x, Yx = self.solve(bt, sp, Lz, rhs)
        if self.disp_response:
            e_disp = self.response_dispersion(bt, sp, self.confinement(sp, self._gram, self._onsite[fr, loc]))
        W = Yx[..., :nmax]
        e_pol = e_el + e_x + bt["e_hyp"]
        Q = torch.zeros(n, dtype=dt, device=dev)
        x = z_pol[..., :8]
        out = {}
        dpsi = None
        if alpha:
            Zf = Yx[..., nmax:]                                          # dz_pol / dF (B, nmax D, 3)
            Xb = torch.zeros(B, nmax, 3, dtype=dt, device=dev).index_put((fr, loc), bt["pos"] / K.BOHR_ANG)
            a_pol = Zf.reshape(B, nmax, D, 3)[:, :, :3].sum(1)
            out["alpha_pol"] = out["alpha"] = a_pol
            dpsi = -Xb + Ca.transpose(1, 2) @ Zf
        if nC:
            if self.gap:
                # the predictor's direct moments: O^-1 on the field-free drive, so the energy stays quadratic in the
                # field (the analytic alpha is its exact second derivative)
                pred = None
                if self.config.ct_gap_predict:
                    Oinv = torch.linalg.inv(self._onsite)
                    z_perm = -torch.einsum("bnac,bnc->bna", Oinv, Lz_free + self._gz.reshape(B, nmax, D))
                    pred = (Oinv, z_perm)
                e_ct, e_ch, qc, zj, Qd, e_c, dQ = self.charge_flux_gap(bt, sp, Ca, Hq, z_pol.reshape(B, -1), W, phi_ext,
                                                                       dpsi, pred=pred)
            else:
                e_ct, e_ch, qc, zj, Qd, e_c, dQ = self.charge_flux(bt, sp, Ca, Hq, z_pol.reshape(B, -1), W, phi_ext, dpsi)
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, bt["ch_frame"] * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
            x = zj.reshape(B, nmax, -1)[..., :8]
            Q = Qd[fr, loc]
            if alpha:
                dz = (Zf - W @ dQ).reshape(B, nmax, D, 3)
                out["alpha"] = dz[:, :, :3].sum(1) + torch.einsum("bka,bkc->bca", dQ, Xb)
        else:
            e_ct, e_ch, qc, e_c = (torch.zeros(B, dtype=dt, device=dev), torch.zeros(0, dtype=dt, device=dev),
                                   torch.zeros(0, dtype=dt, device=dev), torch.zeros(0, dtype=dt, device=dev))
            e_type = torch.zeros(B, self.nT, dtype=dt, device=dev)
        e_elst = bt["e_elst"]
        out.update({"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x,
                    "e_ct": e_ct, "e_int": e_elst + e_pauli + e_disp + e_pol + e_ct, "e_ch": e_ch, "e_type": e_type,
                    "q_ch": qc, "e_kernel": e_c, "x": x, "x_pol": z_pol[..., :8], "z_pol": z_pol, "dq": Q,
                    "q_ct": self._q, "frozen_params": frz})
        if self.gap and nC:
            out.update(ct_gap=self._gap["Delta"], q_star=self._gap["qs"])
            if self._gap_info is not None:
                out.update(ct_gap_iters=self._gap_info["iters"], ct_gap_resid=self._gap_info["resid"])
        return out

    def _forward_pbc(self, bt: dict) -> dict:
        """The periodic forward: the same terms, the joint (z, q) minimum by matrix-free CG
        (:func:`rsfff.ff.cmm.periodic.solve_cmm_periodic`); the energies are the quadratic forms at the (detached)
        solutions with the live operator, so their first derivatives (forces, stress, parameter gradients) are exact.
        ``bt["guess"]``: the previous call's ``out["guess"]`` (warm start; channels matched by ``ch_key``);
        ``bt["cg_rtol"]``: the CG tolerance (default 1e-10; MD is fine with ~1e-7).
        ``bt["total_only"]``: skip the CT-free solve (one CG instead of two). E_int needs only the joint minimum
        (E_pol + E_CT = E_joint + the Legendre constants, ``e_polct``); e_pol, e_ct, e_xpol, x_pol, z_pol (and the
        flux model's e_ch / e_type) are then NaN. For MD; the energies and forces are the same."""
        sp = bt["species"]
        self.environment(bt)
        B, D, mask, fr = bt["B"], bt["D"], bt["mask"], bt["frame"]
        if D != 8 + self.n_b:
            raise ValueError(f"the batch's response layout D = {D} is not the model's {8 + self.n_b}")
        pb = bt["pbc"]
        n = sp.shape[0]
        e_pauli, e_disp, frz = self.frozen_terms(bt, sp)
        self._q = self.ct_charges(sp)
        (i, j), g, Gi, Gj = self.exchange(bt, sp)
        on = pb["onsite"].index_add(0, i, Gi).index_add(0, j, Gj)
        if self.disp_response:
            e_disp = self.response_dispersion(bt, sp, self.confinement(sp, on - pb["onsite"], on))
        Lz = bt["Lz"][mask]
        ci, cj = bt["ch_index"]
        cf = bt["ch_frame"]
        if self.gap:
            return self._forward_pbc_gap(bt, sp, on, Lz, g, Gi, Gj, i, j, e_pauli, e_disp, frz)
        h = self.channel_curvature(bt, sp)
        e_c, V, s = self.channel_drives(bt, sp, self._q, h)
        op = CmmOperator(on, self.slot_eta(sp), pb["A_pairs"], pb["blocks9"], pb["frames"], ci, cj, 1.0 / (s * s),
                         tau=self.slot_tau(sp))
        h = h + self.channel_throughput(bt, sp)          # a lone channel's curvature beyond b (b holds the floor)
        guess, info = self._guess(bt), {}
        total_only = bool(bt.get("total_only"))
        z_pol, z, q = solve_cmm_periodic(op, -(Lz + g), V, h, guess, info=info, rtol=float(bt.get("cg_rtol") or 1e-10),
                                         free=not total_only)
        per = lambda v, idx: Lz.new_zeros(B).index_add(0, idx, v)                         # noqa: E731
        # E_pol at q = 0 (unless total_only), the joint energy, both stationary
        e_quad, e_x = self._pbc_free_energy(op, z_pol, Lz, g, Gi, Gj, i, j, per, fr, B)
        oz, oq = op.joint(z, q)
        e_joint = per(0.5 * (z * oz).sum(1) + ((Lz + g) * z).sum(1), fr) + per(0.5 * q * oq - V * q, cf)
        e_pol, e_ct = self._pbc_split(e_quad, e_joint, bt)
        with torch.no_grad():
            if z_pol is None:
                e_ch = torch.full_like(q, float("nan"))
            else:
                psi0, _ = op.apply(Lz.new_zeros(n), z_pol)
                e_ch = 0.5 * (-V + psi0[ci] - psi0[cj]) * q
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, cf * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
        nmax = bt["nmax"]
        pad = lambda v: v.new_zeros(B, nmax, v.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, v.shape[1]), v)  # noqa: E731
        e_elst = bt["e_elst"]
        zp = z_pol if z_pol is not None else torch.full_like(z, float("nan"))
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_polct": bt["e_hyp"] + e_joint,
                "e_int": e_elst + e_pauli + e_disp + bt["e_hyp"] + e_joint, "e_ch": e_ch, "e_type": e_type, "q_ch": q,
                "e_kernel": e_c, "x": pad(z[:, :8]), "x_pol": pad(zp[:, :8]), "z_pol": pad(zp),
                "dq": op.charges(q), "q_ct": self._q, "frozen_params": frz, "cg_iters": info.get("cg_iters", 0),
                "guess": {"z_pol": None if z_pol is None else z_pol.detach(), "z": z.detach(), "q": q.detach(),
                          "ch_key": bt["ch_key"],
                          "qb": (q * (op.b + h)).detach(),
                          "n": n}}

    @staticmethod
    def _pbc_free_energy(op, z_pol, Lz, g, Gi, Gj, i, j, per, fr, B):
        """E_quad (the CT-free quadratic at its minimum z_pol) and its exchange part, per frame; NaN without z_pol."""
        if z_pol is None:
            nan = Lz.new_full((B,), float("nan"))
            return nan, nan
        _, oz0 = op.apply(Lz.new_zeros(z_pol.shape[0]), z_pol)
        zHz = (torch.einsum("pa,pab,pb->p", z_pol[i], Gi, z_pol[i]) + torch.einsum("pa,pab,pb->p", z_pol[j], Gj, z_pol[j]))
        e_quad = per(0.5 * (z_pol * oz0).sum(1) + ((Lz + g) * z_pol).sum(1), fr)
        e_x = per((g * z_pol).sum(1), fr) + 0.5 * per(zHz, fr[i])
        return e_quad, e_x

    @staticmethod
    def _pbc_split(e_quad, e_joint, bt):
        """(E_pol, E_CT) = (E_quad + the Legendre constants, E_joint - E_quad); NaN when E_quad is (total_only)."""
        return e_quad + bt["e_hyp"], e_joint - e_quad

    def _forward_pbc_gap(self, bt, sp, on, Lz, g, Gi, Gj, i, j, e_pauli, e_disp, frz):
        """The gap model's periodic solve: CG on (z, d), d = q - q0 (:meth:`charge_flux_gap`'s problem, the expansion
        charges' potential C Q0 added to the drive of z); the energies at the solution with the live operator. The
        predictor (``ct_gap_predict``) is the cluster's: direct moments z0 = O^-1 (rhs - C Q0) and psi = H_Q Q0 +
        C^T z0 from two operator products, explicit (no solve, the graph kept). ``bt["total_only"]``: see
        :meth:`_forward_pbc`."""
        B, mask, fr, pb = bt["B"], bt["mask"], bt["frame"], bt["pbc"]
        total_only = bool(bt.get("total_only"))
        n = sp.shape[0]
        ci, cj = bt["ch_index"]
        cf = bt["ch_frame"]
        gc = self.gap_channels(bt, sp)
        h, W2 = gc["h"], gc["W2"]
        rhs_z = -(Lz + g)
        zero_Q, zero_z = Lz.new_zeros(n), torch.zeros_like(Lz)
        eta = self.slot_eta(sp)
        delta = torch.zeros_like(W2)
        nconv = self.config.ct_gap_converge if W2.numel() else 0
        guess, info = self._guess(bt), {}
        rtol = float(bt.get("cg_rtol") or 1e-10)
        warm = nconv and guess is not None and guess.get("delta") is not None
        if warm:                                    # the previous step's converged shifts (new channels: none)
            delta = torch.nan_to_num(guess["delta"]).to(W2)
        elif self.config.ct_gap_predict:
            op0 = CmmOperator(on, eta, pb["A_pairs"], pb["blocks9"], pb["frames"], ci, cj, torch.zeros_like(W2))
            Oinv = torch.linalg.inv(on)
            z_perm = torch.einsum("nac,nc->na", Oinv, rhs_z)
            with torch.set_grad_enabled(torch.is_grad_enabled() and not nconv):
                for _ in range(self.config.ct_gap_predict):
                    Q0 = op0.charges(gc["qs"])
                    z0 = z_perm - torch.einsum("nac,nc->na", Oinv, op0.apply(Q0, zero_z)[1])
                    pQ, _ = op0.apply(Q0, z0)
                    delta = (pQ[ci] - pQ[cj]) - h * gc["qs"]
                    gc = self.expand_at(gc, delta)
        self._gap_info = None
        if self.config.ct_exclusion and W2.numel():
            return self._forward_pbc_excl(bt, sp, on, Lz, g, Gi, Gj, i, j, e_pauli, e_disp, frz, gc, guess, rtol)
        if nconv:
            g0 = {k: v for k, v in gc.items() if k not in ("delta", "Delta0")} | {"Delta": gc.get("Delta0", gc["Delta"])}
            delta, z_c = self._converge_gap_pbc(g0, delta.detach(), on, eta, pb, ci, cj, cf, fr, Lz + g, guess, rtol, B)
            gc = self.expand_at(g0, delta)          # delta without graph: the energy below is stationary in it
            guess = {**(guess or {}), "z": z_c, "q": torch.zeros_like(delta), "qb": None}
            info["cg_iters"] = self._gap_info["cg_iters"]
        self._gap = gc
        a, qs = gc["a"], gc["qs"]
        b = (gc["den"] / (2 * W2).clamp(min=1e-30)).clamp(max=1e12)        # k - h (a channel without coupling is rigid)
        op = CmmOperator(on, eta, pb["A_pairs"], pb["blocks9"], pb["frames"], ci, cj, b)
        with torch.no_grad():
            Qs_d = op.charges(qs)
            psi_s, oz_s = op.detach().apply(Qs_d, zero_z)
            rhs_q = h * qs - (psi_s[ci] - psi_s[cj]) + delta
        info0 = dict(info)
        info = {}
        z_pol, z, d = solve_cmm_periodic(op, rhs_z, rhs_q, h, guess, info=info, rtol=rtol,
                                         rhs_z_joint=rhs_z - oz_s, free=not total_only)
        info["cg_iters"] = info.get("cg_iters", 0) + info0.get("cg_iters", 0)
        per = lambda v, idx: Lz.new_zeros(B).index_add(0, idx, v)                         # noqa: E731
        e_quad, e_x = self._pbc_free_energy(op, z_pol, Lz, g, Gi, Gj, i, j, per, fr, B)
        q = qs + d
        Q = op.charges(q)
        pQ, oz = op.apply(Q, z)
        e_joint = (per(0.5 * (z * oz).sum(1) + ((Lz + g) * z).sum(1) + 0.5 * Q * pQ, fr)
                   + per(-a - delta * q - 0.5 * h * qs * qs - h * qs * d + 0.5 * b * d * d, cf))
        e_pol, e_ct = self._pbc_split(e_quad, e_joint, bt)
        with torch.no_grad():
            e_ch = -a
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, cf * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
        nmax = bt["nmax"]
        pad = lambda v: v.new_zeros(B, nmax, v.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, v.shape[1]), v)  # noqa: E731
        e_elst = bt["e_elst"]
        zp = z_pol if z_pol is not None else torch.full_like(z, float("nan"))
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_polct": bt["e_hyp"] + e_joint,
                "e_int": e_elst + e_pauli + e_disp + bt["e_hyp"] + e_joint, "e_ch": e_ch, "e_type": e_type, "q_ch": q,
                "e_kernel": -a, "x": pad(z[:, :8]), "x_pol": pad(zp[:, :8]), "z_pol": pad(zp),
                "dq": Q, "q_ct": self._q, "frozen_params": frz, "cg_iters": info.get("cg_iters", 0),
                "ct_gap": gc["Delta"], "q_star": qs,
                **({"ct_gap_iters": self._gap_info["iters"], "ct_gap_resid": self._gap_info["resid"]} if nconv else {}),
                "guess": {"z_pol": None if z_pol is None else z_pol.detach(), "z": z.detach(), "q": d.detach(),
                          "ch_key": bt["ch_key"], "delta": delta.detach() if nconv else None,
                          "qb": (d * (b + h)).detach(), "n": n}}

    def _forward_pbc_excl(self, bt, sp, on, Lz, g, Gi, Gj, i, j, e_pauli, e_disp, frz, gc, guess, rtol):
        """The periodic solve with the exclusion functional (:meth:`_flux_excl`): Newton on the charges with the
        joint (z, d) solve by CG, the functional's Hessian as a matrix-free product on the channels (sums over each
        channel's donor and acceptor: O(C)). The channel block is floored so that the model stays positive definite
        (and falls back to its convex part if CG still meets a negative direction); each step is cut to stay inside
        the polytope and to lower the exact joint energy. The energies are evaluated at the converged (z, q) with
        the live parameters: stationary, so forces and parameter gradients are exact. ``bt["guess"]`` warm-starts
        the charges and moments."""
        cfg = self.config
        B, mask, fr, pb = bt["B"], bt["mask"], bt["frame"], bt["pbc"]
        total_only = bool(bt.get("total_only"))
        n_at = sp.shape[0]
        ci, cj = bt["ch_index"]
        cf = bt["ch_frame"]
        W2, h = gc["W2"], gc["h"]
        D0 = gc.get("Delta0", gc["Delta"])
        w02 = cfg.ct_gap_w0 ** 2
        Wb = torch.sqrt(W2 + w02)
        lam = W2 / (W2 + w02) if w02 > 0 else torch.ones_like(W2)
        live = W2.detach() > 1e-24
        lv = live.to(W2.dtype)
        eta = self.slot_eta(sp)
        Lg = Lz + g
        zero_z = torch.zeros_like(Lz)

        def mates(v):                           # the sum over the other channels of c's donor and of its acceptor
            return v.new_zeros(n_at).index_add(0, ci, v)[ci] + v.new_zeros(n_at).index_add(0, cj, v)[cj] - 2 * v

        def terms(q, Wb_, lam_, D0_, h_):
            nn = lam_ * (1 - mates(q)) - q
            qs, ns = torch.where(live, q, torch.ones_like(q)), torch.where(live, nn, torch.ones_like(nn))
            r = Wb_ * torch.sqrt(qs / ns) * lv
            s_ = Wb_ * torch.sqrt(ns / qs) * lv
            u = (D0_ * q - 2 * Wb_ * torch.sqrt(qs * ns) * lv - 0.5 * h_ * q * q) * lv
            gU = (D0_ - s_ + r + mates(lam_ * r) - h_ * q) * lv
            return nn, r, s_, u, gU

        per = lambda v, idx: Lz.new_zeros(B).index_add(0, idx, v)                         # noqa: E731
        amax = lambda v: Lz.new_zeros(B).scatter_reduce(0, cf, v, "amax", include_self=True)   # noqa: E731
        Wd, ld, Dd, hd = Wb.detach(), lam.detach(), D0.detach(), h.detach()
        on_d, eta_d, Lgd = on.detach(), eta.detach(), Lg.detach()
        frames = [f.detach() for f in pb["frames"]]
        B9 = pb["blocks9"].detach()
        mk = lambda b, **kw: CmmOperator(on_d, eta_d, pb["A_pairs"], B9, frames, ci, cj, b, **kw)   # noqa: E731
        op_e = mk(torch.zeros_like(Wd))

        def energy(z, q):
            Q = op_e.charges(q)
            pQ, oz = op_e.apply(Q, z)
            return per(0.5 * (z * oz).sum(1) + (Lgd * z).sum(1) + 0.5 * Q * pQ, fr) + per(terms(q, Wd, ld, Dd, hd)[3], cf)

        cg, it = 0, 0
        with torch.no_grad():
            q = gc["qs"].detach() * lv
            gq = (guess or {}).get("q") if (guess or {}).get("excl") else None
            if gq is not None:                  # the previous step's charges (a new channel: its two-state charge)
                q = torch.where(live & (gq > 0), gq.to(q), q)
            crowd = (torch.where(live, q / ld.clamp(min=1e-300), torch.zeros_like(q)) + mates(q)) * lv
            q = q * (0.8 / amax(crowd).clamp(min=0.8))[cf]
            z = (guess or {}).get("z")
            res = Lz.new_full((B,), float("inf"))
            while True:
                nn, r, s_, u, gU = terms(q, Wd, ld, Dd, hd)
                qs, ns = torch.where(live, q, torch.ones_like(q)), torch.where(live, nn, torch.ones_like(nn))
                kap = Wd / (2 * (qs * ns) ** 1.5) * lv
                vd, ql = nn + q, q * ld
                dU = kap * vd * vd + mates(kap * ql * ql)

                def hv(v, kap=kap, vd=vd, ql=ql):
                    w = kap * (vd * v + ql * mates(v))
                    return (vd * w + mates(ql * w)) * lv

                psi_s, oz_s = op_e.apply(op_e.charges(q), zero_z)
                rhs_q = (-gU - (psi_s[ci] - psi_s[cj])) * lv
                z_new = d = None
                for floor in (torch.where(live, (GAP_B_EPS + hd - dU).clamp(min=0.0) - hd, torch.full_like(hd, 1e12)),
                              torch.where(live, torch.full_like(hd, GAP_B_EPS), torch.full_like(hd, 1e12))):
                    op = mk(floor, ch_hess=hv, ch_diag=dU)
                    inf_ = {}
                    try:
                        _, z_new, d = solve_cmm_periodic(op, -Lgd, rhs_q, hd, {"z": z, "q": torch.zeros_like(q)} if z is not None
                                                         else None, info=inf_, rtol=rtol, rhs_z_joint=-Lgd - oz_s, free=False)
                        cg += inf_.get("cg_iters", 0)
                        break
                    except PolarizationCatastrophe:
                        continue
                if d is None:
                    raise PolarizationCatastrophe("exclusion CT: the joint operator is not positive definite even with "
                                                  "the convex channel block")
                z, d = z_new, d * lv
                res = amax((d * (op.b + hd + dU)).abs() * lv)
                active = res > cfg.ct_gap_tol
                if not bool(active.any()) or it >= cfg.ct_gap_converge:
                    break
                it += 1
                E0 = energy(z, q)
                t = Lz.new_ones(B)
                done, moved, q_new = ~active, torch.zeros_like(active), q
                big = torch.full_like(q, float("inf"))
                for _ in range(60):
                    qt = q + t[cf] * d
                    nt = terms(qt, Wd, ld, Dd, hd)[0]
                    inside = Lz.new_full((B,), float("inf")).scatter_reduce(
                        0, cf, torch.where(live, torch.minimum(nt - 0.1 * nn, qt - 0.1 * q), big), "amin") > 0
                    ok = inside & ~done
                    if bool(ok.any()):
                        ok = ok & (energy(z, torch.where(ok[cf], qt, q)) <= E0 + 1e-13 * (1 + E0.abs()))
                    q_new = torch.where(ok[cf], qt, q_new)
                    done, moved = done | ok, moved | ok
                    if bool(done.all()):
                        break
                    t = torch.where(done, t, 0.5 * t)
                if not bool(moved.any()):
                    break
                q = q_new
        self._gap_info = {"iters": it, "resid": res, "converged": res <= cfg.ct_gap_tol, "cg_iters": cg}
        # the CT-free minimum (unless total_only) and the joint moments once more from the converged point
        info = {}
        g2 = {"z_pol": (guess or {}).get("z_pol"), "z": z, "q": torch.zeros_like(q)}
        z_pol, z, _ = solve_cmm_periodic(op, -Lg, rhs_q, h, g2, info=info, rtol=rtol, rhs_z_joint=-Lg.detach() - oz_s,
                                         free=not total_only)
        cg += info.get("cg_iters", 0)
        opl = CmmOperator(on, eta, pb["A_pairs"], pb["blocks9"], pb["frames"], ci, cj, torch.zeros_like(W2))
        e_quad, e_x = self._pbc_free_energy(opl, z_pol, Lz, g, Gi, Gj, i, j, per, fr, B)
        nn, r, s_, u, _ = terms(q, Wb, lam, D0, h)
        Q = opl.charges(q)
        pQ, oz = opl.apply(Q, z)
        e_joint = per(0.5 * (z * oz).sum(1) + (Lg * z).sum(1) + 0.5 * Q * pQ, fr) + per(u, cf)
        e_pol, e_ct = self._pbc_split(e_quad, e_joint, bt)
        a0 = self.two_state(W2, D0, h)["a"]
        gap = s_ - r
        self._gap = {"W2": W2, "h": h, "Delta0": D0, "Delta": gap, "delta": gap - D0, "qs": q, "left": nn}
        with torch.no_grad():
            e_ch = u + q * (0.5 * (pQ[ci] - pQ[cj]))
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, cf * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
        nmax = bt["nmax"]
        pad = lambda v: v.new_zeros(B, nmax, v.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, v.shape[1]), v)  # noqa: E731
        e_elst = bt["e_elst"]
        zp = z_pol if z_pol is not None else torch.full_like(z, float("nan"))
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_polct": bt["e_hyp"] + e_joint,
                "e_int": e_elst + e_pauli + e_disp + bt["e_hyp"] + e_joint, "e_ch": e_ch, "e_type": e_type, "q_ch": q,
                "e_kernel": -a0, "x": pad(z[:, :8]), "x_pol": pad(zp[:, :8]), "z_pol": pad(zp),
                "dq": Q, "q_ct": self._q, "frozen_params": frz, "cg_iters": cg,
                "ct_gap": gap, "q_star": q, "ct_gap_iters": it, "ct_gap_resid": res,
                "guess": {"z_pol": None if z_pol is None else z_pol.detach(), "z": z.detach(), "q": q.detach(),
                          "ch_key": bt["ch_key"], "excl": True, "qb": None, "delta": None, "n": n_at}}

    def _converge_gap_pbc(self, g0, delta, on, eta, pb, ci, cj, cf, fr, Lg, guess, rtol, B):
        """The periodic self-consistent gap shifts (C,) and the moments at them (N, D), no graph: the iteration of
        :meth:`_converge_gap` with the joint (z, d) solve by CG. Each iteration expands the channels about the current
        two-state charges and solves the joint quadratic with the channel stiffness k - h floored at GAP_B_EPS (a
        positive definite model whatever the channels' softness: CG always runs and -k d is a descent direction; the
        fixed point, d = 0, is the same), then takes the largest step 1, 1/2, ... of delta <- delta - k d that lowers
        the exact joint energy at the solved moments. Stops when every coupled channel's |k d| (the change of its gap)
        is below ``ct_gap_tol``. ``self._gap_info`` = {iters, resid (B,), converged (B,), cg_iters}."""
        cfg = self.config
        g0 = {k: v.detach() for k, v in g0.items()}
        on, eta, Lg = on.detach(), eta.detach(), Lg.detach()
        h, W2 = g0["h"], g0["W2"]
        live = W2 > 0
        zc = torch.zeros_like(delta)
        n = on.shape[0]
        zero_z = torch.zeros_like(Lg)
        frames = [f.detach() for f in pb["frames"]]
        B9 = pb["blocks9"].detach()
        mk = lambda b: CmmOperator(on, eta, pb["A_pairs"], B9, frames, ci, cj, b)        # noqa: E731
        op_e = mk(zc)
        per = lambda v, idx: Lg.new_zeros(B).index_add(0, idx, v)                         # noqa: E731

        def energy(z, dl):
            gt = self.expand_at(g0, dl)
            qs = gt["qs"]
            Q = op_e.charges(qs)
            pQ, oz = op_e.apply(Q, z)
            return (per(0.5 * (z * oz).sum(1) + (Lg * z).sum(1) + 0.5 * Q * pQ, fr)
                    + per(-gt["a"] - dl * qs - 0.5 * h * qs * qs, cf))

        gz = (guess or {}).get("z")
        cg, it = 0, 0
        r = delta.new_full((B,), float("inf"))
        with torch.no_grad():
            z = gz
            while True:
                gx = self.expand_at(g0, delta)
                qs = gx["qs"]
                bt_ = (gx["den"] / (2 * W2).clamp(min=1e-30)).clamp(max=1e12)
                op = mk(bt_.clamp(min=GAP_B_EPS))
                psi_s, oz_s = op.apply(op.charges(qs), zero_z)
                rhs_q = h * qs - (psi_s[ci] - psi_s[cj]) + delta
                inf = {}
                _, z, d = solve_cmm_periodic(op, -Lg, rhs_q, h, {"z": z, "q": zc} if z is not None else None, info=inf,
                                             rtol=rtol, rhs_z_joint=-Lg - oz_s, free=False)
                cg += inf.get("cg_iters", 0)
                kd = (bt_ + h) * d
                r = delta.new_zeros(B).scatter_reduce(0, cf, torch.where(live, kd.abs(), zc), "amax", include_self=True)
                active = r > cfg.ct_gap_tol
                if not bool(active.any()) or it >= cfg.ct_gap_converge:
                    break
                it += 1
                E0 = energy(z, delta)
                t = delta.new_ones(B)
                done, moved, new_delta = ~active, torch.zeros_like(active), delta
                for _ in range(40):
                    trial = delta - t[cf] * kd
                    ok = (energy(z, trial) <= E0 + 1e-13 * (1 + E0.abs())) & ~done
                    new_delta = torch.where(ok[cf], trial, new_delta)
                    done, moved = done | ok, moved | ok
                    if bool(done.all()):
                        break
                    t = torch.where(done, t, 0.5 * t)
                if not bool(moved.any()):
                    break
                delta = new_delta
        self._gap_info = {"iters": it, "resid": r, "converged": r <= cfg.ct_gap_tol, "cg_iters": cg}
        return delta, z

    @staticmethod
    def _guess(bt):
        """The warm start of a periodic solve from ``bt["guess"]`` (same atoms; channels matched by key)."""
        gs = bt.get("guess")
        if not gs or gs.get("n") != bt["species"].shape[0]:
            return None
        out = {"z_pol": gs.get("z_pol"), "z": gs["z"], "excl": gs.get("excl")}
        key, okey = bt["ch_key"], gs["ch_key"]
        if okey.numel():
            srt, order = torch.sort(okey)
            pos = torch.searchsorted(srt, key).clamp(max=okey.shape[0] - 1)
            hit = srt[pos] == key
            out["q"] = torch.where(hit, gs["q"][order[pos]], torch.zeros_like(key, dtype=gs["q"].dtype))
            if gs.get("qb") is not None:
                out["qb"] = torch.where(hit, gs["qb"][order[pos]], torch.full_like(key, float("nan"), dtype=gs["qb"].dtype))
            if gs.get("delta") is not None:
                out["delta"] = torch.where(hit, gs["delta"][order[pos]], torch.zeros_like(key, dtype=gs["delta"].dtype))
        return out

    # -- regularizers ------------------------------------------------------------------------------------------
    def env_mid_penalty(self, s_c=5e-4):
        """Mean over the last batch's atoms of sum_p m_i cap(u_p,i)^2, m_i = 1 / (1 + (S_i / s_c)^2) with S_i the
        atom's total overlap: holds the factors at the isolated values where the energies barely constrain them."""
        if self._u is None:
            return torch.zeros((), dtype=torch.float64)
        m = 1.0 / (1.0 + (self._S.clamp(min=0) / s_c) ** 2)
        u = torch.stack([self.cap(v) for v in self._u.values()], 1)
        return (m[:, None] * u ** 2).sum(1).mean()

    def smoothness_penalty(self, bt, sigma=0.05):
        """Sum over readouts of the mean squared change under noise of sigma x the channels' training scales."""
        sp = bt["species"]
        X, Gv = self._inputs(bt)
        c = self.species(sp) if self.species is not None else None
        nzc = (X.abs().sum(1, keepdim=True) > 0).to(X.dtype)
        u1 = self.encoder(self.encoder_inputs(X, Gv), c)
        # noise on the channels only, as the complete-model runs trained it (the probe invariants enter unperturbed)
        u2 = self.encoder(self.encoder_inputs(X + sigma * self.chan_scale * torch.randn_like(X) * nzc, Gv), c)
        return ((u2 - u1) ** 2).mean(0).sum() / sigma ** 2

    @torch.no_grad()
    def summary(self) -> str:
        sym = self.tab.symbols
        w1 = self.ln_w1.exp()
        bound = self.config.xpol_bound != "off"
        g2 = [f"theta {2 * (1 + self.ln_w2[k].exp()):.3f}" if bound else f"W2 {self.ln_w2[k].exp():.3f}"
              for k in range(len(sym))]
        rows = [f"{x:>3}: W1 {w1[k, 0]:.3f}/{w1[k, 1]:.3f}/{w1[k, 2]:.3f}  {g2[k]}  "
                f"qD0 {self.ct_ln_q0[k, 0].exp():.3f} qA0 {self.ct_ln_q0[k, 1].exp():.3f}  "
                + (f"site potential scale {self.pauli_ln_bscale[k].exp():.3f}  " if self.pauli_site else "")
                + (f"disp range x{self.disp_ln_bscale[k].exp():.3f}" if self.disp_response
                   else f"C6 {self.disp_ln_c6[k].exp():.1f}") + f"  v0 {self.ct_ln_v0[k].exp():.4f}  "
                f"eta {nn.functional.softplus(self.ct_eta_raw[k]):.3f}"
                + (f"  kov {self.overlap_strength()[k]:.2f}" if self.gap else "")
                + (f"  alpha ceiling x{1 + float(self.hyper_fraction()[k] * self.hyper_room[k] / self.hyper_alpha[k]):.3f}"
                   f" (bound x{1 / float(self.hyper_alpha[k] * self.hyper_u1[k]):.2f})" if self.hyper_sat else "")
                for k, x in enumerate(sym)]
        if self.overlap_cut:
            so, sf = self.config.ct_overlap_window
            rows.append(f"CT kernel: {'frontier overlap' if self.frontier else 'slater, overlap switch'}, "
                        f"|S| {so:g} -> {sf:g}; anion -> cation cutoffs (A) "
                        + ", ".join(f"{sym[a]}>{sym[b]} {self.ctk_ron[a, b]:.1f}-{self.ctk_roff[a, b]:.1f}"
                                    for a in range(len(sym)) for b in range(len(sym))
                                    if self.ct_anion[a] and not self.ct_anion[b]))
        return "\n".join(rows)
