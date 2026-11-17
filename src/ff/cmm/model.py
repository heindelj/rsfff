"""The cmm model: a polarizable, charge-transfer force field whose environment-dependent parameters come from one
learned latent per atom.

    E_int = E_elst + E_Pauli + E_disp + E_pol + E_CT                     (Hartree, relative to the isolated species)

Fixed by the species table (``IonTable``): the permanent density (nucleus + Slater 1s shells), the free-ion dipole,
quadrupole and breathing polarizabilities and response shapes, the Legendre hyperpolarizabilities (gamma, b), the
frontier exponents and the channel cutoff / taper.

* **E_elst**: frozen densities, penetration included (no parameters).
* **E_Pauli** = sum_{i<j} sum_{t,u} w_it w_ju f(b_it, b_ju; r) / r, w_it = w0[s,t] e^{cap(u_pauli)}, b_it = b_shell[s,t] e^{lnbs[s]}.
* **E_disp** = -sum_{i<j} f6(sqrt(b_i b_j) r) sqrt(C6_i C6_j) / r^6, C6_i = C60[s] e^{cap(u_disp)}, b_i = the Pauli range of
  the outermost shell.
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

from ..dispersion import tt_damped_c6_energy
from ..multipole import slater_two_center_damp_unequal
from . import kernels as K
from .batch import channel_types
from .channels import CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS
from .params import ChannelTable, IonTable
from .periodic import CmmOperator, dispersion_switch, solve_cmm_periodic
from .precompute import CmmTable, batched_precompute, cmm_table
from .species import SpeciesEmbedding
from .tabulate import PROBES, shape_self_coulomb
from .config import PAIR_GROUPS, CmmConfig
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
#: gap model: start values of the acceptors' overlap strength ("probe": Hartree per unit probe overlap, about what
#: cancels the pair's penetration at contact; "pauli": dimensionless, on the pair Pauli energy per density overlap) and of the CT charges (W of a few 0.01 Hartree at contact), and the channel stiffness
#: k_c - h_c below which a channel is solved explicitly instead of being eliminated onto the atomic charges
GAP_KOV_INIT = {"probe": 10.0, "pauli": 1.5}
GAP_KOV_REF = {"probe": 10.0, "pauli": 1.0}     # the overlap strength stays within a factor GAP_KOV_CAP of this
GAP_KOV_CAP = 5.0
GAP_Q0_INIT = 0.5
GAP_B_EPS = 0.05
GAP_ANION_J = 2.0           # an anion's acceptor level: A = I - GAP_ANION_J J_self


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
        for k, x in enumerate(sym):
            occ_k = [t for t in range(T) if occ[k, t]]
            if tab.pauli_s is not None and math.isfinite(float(tab.pauli_s[k])):      # the frozen fit
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
        # -- exchange polarization: W1 per species and rank (dipole, quadrupole, breathing), W2 per species
        self.ln_w1 = nn.Parameter(torch.zeros(S, 3, **f64))          # W1 = 1 at start (the overlap model)
        self.ln_w2 = nn.Parameter(torch.zeros(S, **f64))
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
        if self.gap:
            with torch.no_grad():
                self.ct_ln_q0.fill_(math.log(GAP_Q0_INIT))
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
        positions): :func:`batched_precompute` with this model's tables and gate (``cell`` etc. pass through)."""
        return batched_precompute(pos_ang, atomic_numbers, batch_idx, self.tab, self.table(pos_ang.device), self.chan,
                                  labels=labels, gate=self.config.env_gate, **kw)

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
        """(N, 2) qD, qA = q0[s] e^{f}, f = c tanh(u / c); with ``config.ct_q_onesided`` f is clipped at 0, so the
        isolated-species charges q0 are a ceiling and the pair kernels of any geometry are bounded by the bare ones."""
        u = torch.stack([self._u["qD"], self._u["qA"]], 1)
        c = self.config.ct_q_cap
        f = c * torch.tanh(u / c)
        if self.config.ct_q_onesided or self.gap:
            f = torch.where(f > 0, torch.zeros_like(f), f)      # not clamp: its gradient is 0 at u = 0, the start
        return torch.exp(self.ct_ln_q0[sp] + f)

    def disp_range(self):
        S = self.outer_shell.shape[0]
        return self.tab.shell_b[torch.arange(S, device=self.outer_shell.device), self.outer_shell] * self.pauli_ln_bscale.exp()

    def frozen_params(self, sp):
        """Per atom: Pauli shell weights and exponents (N, T), C6 (N,), dispersion exponent (N,)."""
        w = torch.exp(self.pauli_ln_w[sp] + self.cap(self._u["pauli"])[:, None]) * self.shell_mask[sp]
        b = self.tab.shell_b[sp] * self.pauli_ln_bscale.exp()[sp][:, None]
        C6 = torch.exp(self.disp_ln_c6[sp] + self.cap(self._u["disp"]))
        return w, b, C6, self.disp_range()[sp]

    def w1(self, sp):
        """(N, D) the drive weight W1 of every response component (dipole x3, quadrupole x5, breathing)."""
        u = torch.stack([self._u["w1_dip"], self._u["w1_quad"], self._u["w1_breath"]], 1)
        k = torch.exp(self.ln_w1[sp] + self.cap(u))
        parts = [k[:, :1].expand(-1, 3), k[:, 1:2].expand(-1, 5)]
        if self.n_b:
            parts.append((k[:, 2] * self.breath_mask[sp])[:, None])
        return torch.cat(parts, 1)

    def w2(self, sp):
        return torch.exp(self.ln_w2[sp] + self.cap(self._u["w2"]))

    # -- terms -------------------------------------------------------------------------------------------------
    def frozen_terms(self, bt, sp):
        """Frozen Pauli and dispersion energies per frame (B,), (B,); keeps the Pauli pair energies for xpol."""
        B = bt["B"]
        w, b, C6, b6 = self.frozen_params(sp)
        (i, j), dr = bt["pairs"], bt["pair_dr"]
        r_ang = dr.norm(dim=-1)
        rb = r_ang / K.BOHR_ANG
        f = slater_two_center_damp_unequal(b[i][:, :, None], b[j][:, None, :], rb[:, None, None], 0)[0]
        e_p = torch.einsum("pt,ptu,pu->p", w[i], f, w[j]) / rb
        self._pauli_pairs = ((i, j), e_p)
        e_d = tt_damped_c6_energy(r_ang, torch.sqrt(C6[i] * C6[j]), torch.sqrt(b6[i] * b6[j]))
        pbc = bt.get("pbc")
        if pbc is not None:           # switched off over the cutoff's last switch_width, the uniform tail beyond
            e_d = e_d * dispersion_switch(r_ang, pbc["cutoff"], pbc["switch_width"])
        fr = bt["frame"][i]
        E_d = r_ang.new_zeros(B).index_add(0, fr, e_d)
        if pbc is not None and pbc.get("disp_tail") is not None:
            sq = r_ang.new_zeros(B).index_add(0, bt["frame"], torch.sqrt(C6))
            E_d = E_d - 2 * math.pi / pbc["volume"] * sq * sq * pbc["disp_tail"]
        return r_ang.new_zeros(B).index_add(0, fr, e_p), E_d, (w, b, C6, b6)

    @staticmethod
    def _local(bt):
        fr = bt["frame"]
        counts = torch.bincount(fr, minlength=bt["B"])
        return fr, torch.arange(fr.shape[0], device=fr.device) - (torch.cumsum(counts, 0) - counts)[fr]

    def exchange(self, bt, sp):
        """The exchange pairs' drive g (N, D) and on-site Gram blocks W2_i 2 A_ij Gt(i <- j), W2_j 2 A_ij Gt(j <- i):
        g_i = sum_j 2 E0_ij W1_i d ln O_ij / dz_i over the pairs within R_XPOL, E0_ij the pair's Pauli energy and
        A_ij = E0_ij / O_ij (switched on above the overlap noise). Returns (i, j), g, Gi, Gj."""
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
        return (i, j), g, w2[i][:, None, None] * (w * bt["x_Gi"]), w2[j][:, None, None] * (w * bt["x_Gj"])

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
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        W2 = (q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])).clamp(min=0.0)
        phi = bt["phi_front"]
        Delta = (self.ct_gap_I[sp[i]] - self.ct_gap_A[sp[j]] + phi[i, 0] - phi[j, 1] - bt["ch_J"]
                 + self.acceptor_shift(bt, sp)[j])
        w02 = self.config.ct_gap_w0 ** 2
        Wb2 = W2 + w02
        rt = torch.sqrt(Delta * Delta + 4 * Wb2 + 1e-60)
        pos = Delta > 0
        dp = torch.where(pos, rt + Delta, torch.ones_like(rt))            # the unselected branch stays finite
        lam = W2 / Wb2 if w02 > 0 else torch.ones_like(W2)
        a = torch.where(pos, 2 * W2 / dp, lam * 0.5 * (rt - Delta))
        qs = torch.where(pos, 2 * W2 / (rt * dp), lam * 0.5 * (1 - Delta / rt))
        h = self.channel_curvature(bt, sp)
        return {"W2": W2, "Delta": Delta, "a": a, "qs": qs, "h": h, "den": rt ** 3 - 2 * W2 * h}

    def charge_flux_gap(self, bt, sp, Ca, Hq, z_pol, W, phi_ext, dpsi=None):
        """The joint minimum of the gap model, solved for the channels' departures d = q - q* from their two-state
        charges. With Q* = G q*, S = H_Q - C^T W, psi0 = phi_ext + C^T z_pol and psi* = psi0 + S Q*, the energy is

            E* + sum_c [1/2 (k_c - h_c) d_c^2 - V_c d_c] + 1/2 dQ^T S dQ,     V_c = h_c q*_c - (psi*_i - psi*_j),
            E* = sum_c (-a_c - 1/2 h_c q*_c^2) + 1/2 Q*^T S Q* + Q*^T psi0,

        which is the flux's problem with compliance k - h, drive V and no psi0 (:meth:`_flux_solve_mixed`).
        Returns as :meth:`charge_flux`; e_kernel is the lone channels' -a_c."""
        B, N = Hq.shape[0], Hq.shape[1]
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        g = self.gap_channels(bt, sp)
        self._gap = g
        a, qs, h, W2, den = g["a"], g["qs"], g["h"], g["W2"], g["den"]
        psi0 = Hq.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Hq - Ca.transpose(1, 2) @ W
        Qs = Hq.new_zeros(B, N).index_put((cf, li), qs, accumulate=True).index_put((cf, lj), -qs, accumulate=True)
        SQs = (S @ Qs[..., None])[..., 0]
        psis = psi0 + SQs
        V = h * qs - (psis[cf, li] - psis[cf, lj])
        good = den > 2 * W2 * GAP_B_EPS                                   # k - h > GAP_B_EPS (W = 0: rigid, w = 0)
        w = torch.where(good, 2 * W2 / torch.where(good, den, torch.ones_like(den)), torch.zeros_like(den))
        b_bad = den / (2 * W2).clamp(min=1e-300)
        Vc = V[:, None] if dpsi is None else torch.cat([V[:, None], -(dpsi[cf, li] - dpsi[cf, lj])], 1)
        dQ, d = self._flux_solve_mixed(B, N, cf, li, lj, w, ~good, b_bad, Vc, S)
        dlt = d[:, 0]
        e_ch = (-a - 0.5 * h * qs * qs + qs * (0.5 * (SQs[cf, li] - SQs[cf, lj]) + psi0[cf, li] - psi0[cf, lj])
                - 0.5 * V * dlt)
        Q = Qs + dQ[..., 0]
        z = z_pol - (W @ Q[..., None])[..., 0]
        return (Hq.new_zeros(B).index_add(0, cf, e_ch), e_ch, qs + dlt, z, Q, -a,
                (dQ[..., 1:] if dpsi is not None else None))

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
            solve_ct = self.charge_flux_gap if self.gap else self.charge_flux
            e_ct, e_ch, qc, zj, Qd, e_c, dQ = solve_ct(bt, sp, Ca, Hq, z_pol.reshape(B, -1), W, phi_ext, dpsi)
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
        return out

    def _forward_pbc(self, bt: dict) -> dict:
        """The periodic forward: the same terms, the joint (z, q) minimum by matrix-free CG
        (:func:`rsfff.ff.cmm.periodic.solve_cmm_periodic`); the energies are the quadratic forms at the (detached)
        solutions with the live operator, so their first derivatives (forces, stress, parameter gradients) are exact.
        ``bt["guess"]``: the previous call's ``out["guess"]`` (warm start; channels matched by ``ch_key``);
        ``bt["cg_rtol"]``: the CG tolerance (default 1e-10; MD is fine with ~1e-7)."""
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
        z_pol, z, q = solve_cmm_periodic(op, -(Lz + g), V, h, guess, info=info, rtol=float(bt.get("cg_rtol") or 1e-10))
        zero = Lz.new_zeros(n)
        # E_pol at q = 0, the joint energy, both stationary
        psi0, oz0 = op.apply(zero, z_pol)
        zHz = (torch.einsum("pa,pab,pb->p", z_pol[i], Gi, z_pol[i]) + torch.einsum("pa,pab,pb->p", z_pol[j], Gj, z_pol[j]))
        per = lambda v, idx: Lz.new_zeros(B).index_add(0, idx, v)                         # noqa: E731
        e_quad = per(0.5 * (z_pol * oz0).sum(1) + ((Lz + g) * z_pol).sum(1), fr)
        e_x = per((g * z_pol).sum(1), fr) + 0.5 * per(zHz, fr[i])
        e_pol = e_quad + bt["e_hyp"]
        oz, oq = op.joint(z, q)
        e_joint = per(0.5 * (z * oz).sum(1) + ((Lz + g) * z).sum(1), fr) + per(0.5 * q * oq - V * q, cf)
        e_ct = e_joint - e_quad
        with torch.no_grad():
            e_ch = 0.5 * (-V + psi0[ci] - psi0[cj]) * q
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, cf * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
        nmax = bt["nmax"]
        pad = lambda v: v.new_zeros(B, nmax, v.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, v.shape[1]), v)  # noqa: E731
        e_elst = bt["e_elst"]
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_int": e_elst + e_pauli + e_disp + e_pol + e_ct, "e_ch": e_ch, "e_type": e_type, "q_ch": q,
                "e_kernel": e_c, "x": pad(z[:, :8]), "x_pol": pad(z_pol[:, :8]), "z_pol": pad(z_pol),
                "dq": op.charges(q), "q_ct": self._q, "frozen_params": frz, "cg_iters": info.get("cg_iters", 0),
                "guess": {"z_pol": z_pol.detach(), "z": z.detach(), "q": q.detach(), "ch_key": bt["ch_key"],
                          "qb": (q * (op.b + h)).detach(),
                          "n": n}}

    def _forward_pbc_gap(self, bt, sp, on, Lz, g, Gi, Gj, i, j, e_pauli, e_disp, frz):
        """The gap model's periodic solve: CG on (z, d), d = q - q* (:meth:`charge_flux_gap`'s problem, the two-state
        charges' potential C Q* added to the drive of z); the energies at the solution with the live operator."""
        B, mask, fr, pb = bt["B"], bt["mask"], bt["frame"], bt["pbc"]
        n = sp.shape[0]
        ci, cj = bt["ch_index"]
        cf = bt["ch_frame"]
        gc = self.gap_channels(bt, sp)
        self._gap = gc
        a, qs, h, W2 = gc["a"], gc["qs"], gc["h"], gc["W2"]
        b = (gc["den"] / (2 * W2).clamp(min=1e-30)).clamp(max=1e12)        # k - h (a channel without coupling is rigid)
        op = CmmOperator(on, self.slot_eta(sp), pb["A_pairs"], pb["blocks9"], pb["frames"], ci, cj, b)
        zero_Q, zero_z = Lz.new_zeros(n), torch.zeros_like(Lz)
        with torch.no_grad():
            Qs_d = op.charges(qs)
            psi_s, oz_s = op.detach().apply(Qs_d, zero_z)
            rhs_q = h * qs - (psi_s[ci] - psi_s[cj])
        guess, info = self._guess(bt), {}
        z_pol, z, d = solve_cmm_periodic(op, -(Lz + g), rhs_q, h, guess, info=info, rtol=float(bt.get("cg_rtol") or 1e-10),
                                         rhs_z_joint=-(Lz + g) - oz_s)
        per = lambda v, idx: Lz.new_zeros(B).index_add(0, idx, v)                         # noqa: E731
        _, oz0 = op.apply(zero_Q, z_pol)
        zHz = (torch.einsum("pa,pab,pb->p", z_pol[i], Gi, z_pol[i]) + torch.einsum("pa,pab,pb->p", z_pol[j], Gj, z_pol[j]))
        e_quad = per(0.5 * (z_pol * oz0).sum(1) + ((Lz + g) * z_pol).sum(1), fr)
        e_x = per((g * z_pol).sum(1), fr) + 0.5 * per(zHz, fr[i])
        e_pol = e_quad + bt["e_hyp"]
        q = qs + d
        Q = op.charges(q)
        pQ, oz = op.apply(Q, z)
        e_joint = (per(0.5 * (z * oz).sum(1) + ((Lz + g) * z).sum(1) + 0.5 * Q * pQ, fr)
                   + per(-a - 0.5 * h * qs * qs - h * qs * d + 0.5 * b * d * d, cf))
        e_ct = e_joint - e_quad
        with torch.no_grad():
            e_ch = -a
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, cf * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
        nmax = bt["nmax"]
        pad = lambda v: v.new_zeros(B, nmax, v.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, v.shape[1]), v)  # noqa: E731
        e_elst = bt["e_elst"]
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_int": e_elst + e_pauli + e_disp + e_pol + e_ct, "e_ch": e_ch, "e_type": e_type, "q_ch": q,
                "e_kernel": -a, "x": pad(z[:, :8]), "x_pol": pad(z_pol[:, :8]), "z_pol": pad(z_pol),
                "dq": Q, "q_ct": self._q, "frozen_params": frz, "cg_iters": info.get("cg_iters", 0),
                "ct_gap": gc["Delta"], "q_star": qs,
                "guess": {"z_pol": z_pol.detach(), "z": z.detach(), "q": d.detach(), "ch_key": bt["ch_key"],
                          "qb": (d * (b + h)).detach(), "n": n}}

    @staticmethod
    def _guess(bt):
        """The warm start of a periodic solve from ``bt["guess"]`` (same atoms; channels matched by key)."""
        gs = bt.get("guess")
        if not gs or gs.get("n") != bt["species"].shape[0]:
            return None
        out = {"z_pol": gs["z_pol"], "z": gs["z"]}
        key, okey = bt["ch_key"], gs["ch_key"]
        if okey.numel():
            srt, order = torch.sort(okey)
            pos = torch.searchsorted(srt, key).clamp(max=okey.shape[0] - 1)
            hit = srt[pos] == key
            out["q"] = torch.where(hit, gs["q"][order[pos]], torch.zeros_like(key, dtype=gs["q"].dtype))
            if gs.get("qb") is not None:
                out["qb"] = torch.where(hit, gs["qb"][order[pos]], torch.full_like(key, float("nan"), dtype=gs["qb"].dtype))
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
        rows = [f"{x:>3}: W1 {w1[k, 0]:.3f}/{w1[k, 1]:.3f}/{w1[k, 2]:.3f}  W2 {self.ln_w2[k].exp():.3f}  "
                f"qD0 {self.ct_ln_q0[k, 0].exp():.3f} qA0 {self.ct_ln_q0[k, 1].exp():.3f}  "
                f"C6 {self.disp_ln_c6[k].exp():.1f}  v0 {self.ct_ln_v0[k].exp():.4f}  "
                f"eta {nn.functional.softplus(self.ct_eta_raw[k]):.3f}"
                + (f"  kov {self.overlap_strength()[k]:.2f}" if self.gap else "") for k, x in enumerate(sym)]
        return "\n".join(rows)
