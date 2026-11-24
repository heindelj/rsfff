"""The architecture choices of a cmm model, the only ones it has.

Everything else that the complete model's exploration made optional is fixed in :class:`rsfff.ff.cmm.CmmModel`:
frozen Pauli (permanent shells, one environment factor per atom) and Tang-Toennies C6 dispersion with the Pauli range;
induction with the free-ion dipole, quadrupole and breathing (pinned a0) responses and the Legendre hyperpolarization
(gamma and the field-gradient b); Gram exchange polarization with the learned drive and on-site weights W1, W2
(``xpol_bound`` ties them so that a pair's Pauli energy plus its exchange polarization cannot go negative);
charge-flux CT with the pinned exponents (cation acceptors floating within e^+-0.5 of Fukui), solved jointly with the
induction, with three optional stabilizers (all off by default): CT charges that the environment can only lower
(``ct_q_onesided``), a stiffness floor on every channel (``ct_floor``) and a hardness on each site's throughput
(``ct_relay``). ``ct_model = "gap"`` replaces the flux by two-state channels with an explicit gap (the model module's
docstring); the three stabilizers belong to the flux and are not used there (its charges are always one-sided); the encoder: one latent from the overlap-weighted channels and the gated diffuse probes, decoded linearly
into every environment factor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["CmmConfig", "PAIR_GROUPS"]

#: channel groups the encoder can read: the two-body sums S^w (plain overlap weights) and S^s (signed by the pair's
#: frozen electrostatics), and the three-body families T^ww, T^ws, T^ss (L = 0..4). Family tokens are unions.
PAIR_GROUPS = {"ww": ("S^w", "ww"), "ws": ("S^s", "ws"), "ss": ("S^s", "ss"), "Sw": ("S^w",), "Ss": ("S^s",),
               "Tww": ("ww",), "Tws": ("ws",), "Tss": ("ss",)}


@dataclass
class CmmConfig:
    hidden: int = 64                  # encoder trunk width
    depth: int = 2                    # encoder trunk hidden layers
    latent: int = 6                   # width of the shared latent
    pairs: tuple = ("ww",)            # channel groups read (PAIR_GROUPS)
    probes: bool = True               # the gated diffuse-probe invariants (donor, acceptor, breath) as encoder inputs
    film: bool = True                 # FiLM conditioning of the trunk on the species embedding
    species_dim: int = 8              # species embedding width
    species_hidden: int = 16          # species embedding MLP width (descriptor -> embedding)
    env_cap: float = 3.0              # |ln factor| bound of the Pauli, C6, W1, W2 environment factors
    ct_q_cap: float = 1.5             # |ln factor| bound of the CT charges' environment factors
    pauli_form: str = "shells"        # the frozen Pauli energy. "shells": the pair kernel between Pauli charges on the
                                      # permanent shells, sum_tu w_it w_ju f(b_it s_i, b_ju s_j; r) / r (table: pauli).
                                      # "site": each ion's electrons in a short-ranged potential of the other,
                                      # E_ij = e^{u_i + u_j} [<n_i | v_j> + <n_j | v_i>], n the permanent shells,
                                      # v_j(r) = sum_t a_jt e^{-beta r} (1 / r + beta / 2), beta = s_j b_jt (table:
                                      # pauli_site). One-body in the partner's density: the form a QM region can take
                                      # (claude/qm_embedding_pauli.md)
    pauli_env: str = "product"        # pauli_form = "site": how the environment factors scale a pair. "product":
                                      # e^{u_i} e^{u_j} (the cross term of i's and j's environments is a 4-body energy).
                                      # "mean": (e^{u_i} + e^{u_j}) / 2, each term one centre's environment (the same
                                      # factor in the site xpol and the site CT shifts). "linear": (f_i + f_j) / 2 with
                                      # f_i = 1 + sum_k c[s_i, s_k] sw O_ik / O_REF over i's exchange pairs, c learned
                                      # per species pair (start 0, |c| < PAULI_LIN_CAP): the Pauli energy is exactly
                                      # 2- plus 3-body (the encoder's Pauli readout is not used). A smooth floor keeps
                                      # the factor above 0.05. claude/cmm_manybody_partition.md. "global": the
                                      # linear form with one learned constant on the normalized overlap,
                                      # f_i = 1 + c sum_k sw O_ik / (sqrt(O_ii O_kk) S_REF), O_ii the ion's density
                                      # self-overlap (no species-pair parameter). "ion": the linear form with one
                                      # constant per centre species, f_i = 1 + c_{s_i} sum_k sw O_ik / O_REF. "edge": the product
                                      # form with edge-additive log-factors, f_ij = e^{u_i + u_j},
                                      # u_i = sum_k u(s_i, s_k; s_ik), s_ik = sw O_ik / O_REF: a shared network on
                                      # species embeddings emits for each ordered species pair a bounded, monotone
                                      # function of the overlap (PauliEdge): a neighbour adds to the log-factor, so the
                                      # 4-body Pauli is the exponential's second order, a product of two edges
    xpol_form: str = "overlap"        # what exchange polarization perturbs. "overlap": the pair's Pauli energy per
                                      # density overlap times the overlap of the perturbed densities (drive
                                      # A_ij <dn_i | rho_j>, Gram A_ij int dphi_i^2 rho_j, A_ij = E0_ij / O_ij).
                                      # "site" (needs pauli_form = "site"): the site-potential energy itself with the
                                      # perturbed amplitude, e^{u_i + u_j} <(phi_i + dphi_i)^2 | v_j>: drive
                                      # <dn_i | v_j>, Gram int dphi_i^2 v_j, no overlap ratio. With unit weights a
                                      # complete square, so with xpol_bound theta = 1 + e^{ln W2 + u} >= 1 (each
                                      # ion's response can remove at most 1 / theta of its own half of the pair's
                                      # Pauli energy). The potential's exponent scale is then fixed at the table's
                                      # (the tabulated integrals depend on it). claude/qm_embedding_xpol_site.md
    xpol_bound: str = "off"           # exchange polarization, the bound that keeps a pair's Pauli wall
                                      # (theory/response_functional.tex, the Pauli-wall proposition). "off": W1 per
                                      # rank on the drive, W2 on the Gram blocks, independent. "rank": the rank
                                      # weights act on the response functions, so on both orders (drive W1 d, Gram
                                      # theta W1 Gt W1), and theta = 2 (1 + e^{ln W2 + u}) >= 2: one ion's response
                                      # removes at most 1 / theta = 1/2 sigmoid(-(ln W2 + u)) <= 1/2 of a pair's Pauli
                                      # energy, so Pauli + exchange polarization >= 0 for every pair, any output.
                                      # "scalar": the same with one drive weight per atom (the dipole row and readout)
    disp_model: str = "emitted"       # dispersion. "emitted": -f6 sqrt(C6_i C6_j) / r^6 with a learned C6 per species and
                                      # an environment factor per atom. "response": the free-ion C6 and C8 of every
                                      # species pair from response theory (``disp_table``), scaled by the confinement
                                      # of the two ions' polarizabilities (:meth:`CmmModel.response_dispersion`); the
                                      # only learned dispersion parameter is one damping range per species
    disp_confine: str = "pauli"       # response dispersion, the confined polarizability alpha_eff = (on-site block)^-1:
                                      # "pauli" the free ion stiffened by its exchange Gram blocks (alpha_eff <=
                                      # alpha_free: the coefficients can only shrink), "onsite" the solve's on-site
                                      # block (with the field-dependent hyperpolarized dipole), "none" the free ion
    disp_table: str = "disp_wb97mv_tzvpd.json"    # response dispersion, the table (rsfff.ff.cmm.disp_table; params/)
    hyperpol: str = "legendre"        # the dipole hyperpolarization about the permanent field. "legendre": the free
                                      # ion's polynomial field energy (gamma, b): alpha(E) grows without bound with
                                      # the field and one eigenvalue falls with the gradient. "saturating": a bounded
                                      # field energy with the same gamma and b at small field (precompute.
                                      # saturating_hyperpolarization): every eigenvalue of alpha(E) stays in
                                      # (alpha / 2, alpha + f (1 / u1 - alpha)), 1 / u1 the largest polarizability the Coulomb
                                      # Gram certificate allows (u1 the dipole shape's self-energy), f in (0, 1)
                                      # learned. An ion with alpha u1 >= 1 has no room and responds linearly
    hyper_ceiling: str = "global"     # saturating: one learned fraction f for all species ("global") or one each
                                      # ("species")
    ct_model: str = "flux"            # "flux": the pair-calibrated charge flux; "gap": two-state channels with an explicit gap
    ct_gap_overlap: str = "pauli"     # gap model, the acceptor level's overlap shift: "pauli" (c x the acceptor probe's
                                      # overlap with each neighbour x that pair's Pauli energy per unit density
                                      # overlap) or "probe" (kappa x the acceptor probe's total overlap) or "site"
                                      # (needs pauli_form = "site"): kappa x the transferred electron in the
                                      # neighbours' site potentials, sum_k e^{u_j + u_k} <A_j | v_k>, A_j the
                                      # normalized acceptor shape; kappa = 1 is the electron's own Pauli energy
    ct_gap_kov_fixed: bool = False    # gap model: kappa held at its start value (1 for "site") instead of learned
    ct_gap_donor: bool = False        # gap model (needs pauli_form = "site"): the donor's relief, the gap minus
                                      # sum_k e^{u_i + u_k} <D_i | v_k>: the electron that leaves i's donor shape no
                                      # longer sits in the neighbours' site potentials. No parameter
    ct_gap_centred: str = "off"       # gap model: the donor and acceptor levels evaluated at points moved along the
                                      # channel (n donor -> acceptor), X_i + s_D n and X_j - s_A n; the partner's own
                                      # field is left out, so a dimer's gap is unchanged. "species": first order,
                                      # gap + s_D n.grad phi3(i) + s_A n.grad phi3(j), s per species and role learned
                                      # in (0, CENTRE_DMAX). "overlap": s_D = lam t_D, s_A = lam (R - t_D), t_D the
                                      # distance of the donor-acceptor density overlap's centroid from the donor (the
                                      # CT kernel's exponents), lam in (0, 1) one learned constant; the other ions'
                                      # full frozen potentials (nucleus plus shells) averaged over the donor's /
                                      # acceptor's frontier shape at the moved point, in closed form; periodic: what
                                      # lies beyond the pair list (Ewald reciprocal) to first order. lam = 1 puts
                                      # both levels at the centroid (only the shapes' difference is left in the
                                      # gap). Only at contact (ct_centre_window) and for opposite-charge channels
                                      # (X > X: second neighbours, often bridged by a cation the centroid would sit
                                      # on). True / False read as "species" / "off"
    ct_centre_window: tuple = (0.7, 0.5)  # ct_gap_centred = "overlap": the displacement is full while the channel's
                                      # frontier overlap |S| is above 0.7 of the pair's largest and off below 0.5
                                      # (C-infinity step; the ct_overlap_table curves): contacts only, not channels
                                      # through a bridging ion
    ct_alpha_radical: str = "off"     # gap model: a site's dipole polarizability moves with its transferred charge
                                      # toward its CT radical's, alpha_i = alpha_ion + Q_i^D dalpha_D + Q_i^A dalpha_A
                                      # (Q^D / Q^A the charge i gives / takes), inside the joint minimum: outer passes
                                      # rebuild the on-site blocks from the previous charges and add d E / d alpha
                                      # dalpha (= -1/2 dalpha |F|^2 for a linear site) to each channel's gap, to a
                                      # fixed point (claude/ct_gap_dipole.md). dalpha_D: halogen radical - anion;
                                      # dalpha_A: alkali atom - cation, capped at CT_ALPHA_ROOM of the cation parent's
                                      # room 1 / u1 - alpha. "fixed": those values; "learned": times lam in (0, 1) per
                                      # species and role. Clusters, saturating hyperpolarization only
    ct_alpha_solve: str = "joint"     # ct_alpha_radical: "joint": first order in the step about the ion model, one
                                      # factorization: the ion model's joint minimum, then t_c from its dipoles (in
                                      # the graph) added to the gaps and a second CT solve; E = min E_ion + sum q t.
                                      # Within 0.01 kJ/mol rms of "outer" on the trained nocd (capped acceptor steps).
                                      # The polarizability is the field Hessian by autograd. "outer": the rigorous
                                      # fixed point (refactorizes every pass, ~10x slower)
    ct_alpha_outer: int = 10          # ct_alpha_solve = "outer": at most this many passes (stop at |dQ| < 1e-8)
    ct_cation_donor: bool = True      # CT channels out of cations (M>X, M>M). Their donor levels (the cations' second
                                      # ionization, 23 - 76 eV) leave them inert; False drops them (about half the
                                      # channels, the same energies)
    ct_gap_w0: float = 0.005          # gap model, the least width W0 (Hartree) of a channel's crossing: the two-state form
                                      # with W_b^2 = W^2 + W0^2 and amplitude W^2 / W_b^2 (same W^2 / Delta at a large
                                      # gap; response <= W^2 / (4 W_b^3), transfer <= W^2 / W_b^2). 0: the exact form
    ct_gap_predict: int = 1           # gap model, predictor passes: each channel is expanded about the two-state charge of
                                      # its gap shifted by the potential of the direct (uncoupled) induced moments and of
                                      # the other channels' two-state charges, Delta' = Delta + (psi_i - psi_j) - h q*,
                                      # instead of about that of the frozen gap (the coupled solve about it is unchanged);
                                      # 0: the frozen gap (fails where polarization reopens a gap the frozen ions close)
    ct_gap_converge: int = 0          # gap model, the self-consistent solve: at most this many Newton iterations of the
                                      # gap equation delta_c = (psi_i - psi_j) - h_c q_c with the *coupled* moments and
                                      # charges (each one coupled solve about the current charges, safeguarded by the
                                      # energy), started from the predictor. The energy is then the minimum of the
                                      # full two-state functional, not of its expansion about a predicted point;
                                      # forces, polarizabilities and parameter gradients stay exact (one more
                                      # differentiable solve). 0: off (the expansion about the predictor)
    ct_gap_tol: float = 1e-9          # ... converged when every channel's residual drive |V_c| is below this (Hartree)
    ct_exclusion: bool = False        # gap model, converged: a site gives at most one electron and takes at most one.
                                      # The channels out of one donor (and those into one acceptor) share its
                                      # capacity: channel c has what the others leave, lam_c (1 - b_c) with b_c the
                                      # charge of the other channels out of its donor and into its acceptor, so
                                      # F_c = Delta_c q_c - 2 W_b sqrt(q_c (lam_c (1 - b_c) - q_c)) (jointly convex;
                                      # for one site's channels alone, the exact ground state of its configurations).
                                      # No parameters. Needs ct_gap_converge > 0
    ct_q_onesided: bool = False       # the CT charges' environment factors can only lower them: ln factor <= 0
    ct_floor: float = 0.0             # channel stiffness floor b_min (Hartree): b_c = V0_c^2 / (2|e_c|) + b_min
    ct_relay: float = 0.0             # throughput hardness as a fraction kappa of the slot hardness: tau_i = kappa eta_i
    ct_kernel: str = "slater"         # the coupling's radial form (gap model). "slater": W^2 = qD qA f(bD, bA; r) / r,
                                      # one exponential per site, the cation's acceptor exponent learned, the channel
                                      # table's taper. "frontier": W^2 = qD qA [S(R) g(R)]^2, S the overlap of the free
                                      # ions' frontier orbitals from ``ct_overlap_table``, g each pair's own cutoff
                                      # (rsfff.ff.cmm.ct_overlap); no learned exponent, no taper. "table":
                                      # W^2 = 2 [W_ab(R) g(R)]^2, W_ab the tabulated one-electron coupling of the free
                                      # ions' frontier orbitals in the frozen dimer (``ct_coupling_table``), g the
                                      # same cutoff: no CT charges, no exponent, nothing learned or emitted
    ct_cut: str = "taper"             # where the channels end (gap model). "taper": the channel table's cutoff and
                                      # taper, one distance for all pairs. "overlap": each pair's own C-infinity switch
                                      # between the distances at which the overlap of the free ions' frontier orbitals
                                      # is ``ct_overlap_window`` (rsfff.ff.cmm.ct_overlap) -- with the "slater" kernel
                                      # W^2 = qD qA f / r g(R)^2: the learned exponent keeps the short-range shape,
                                      # the switch ends the tail. The "frontier" kernel always ends this way
    ct_overlap_table: str = "ct_overlap_wb97mv_tzvpd.json"   # the overlap table (scripts/cmm_frontier_shapes.py
                                      # --table; params/) of ct_kernel "frontier" / ct_cut "overlap". A checkpoint
                                      # carries its own
    ct_coupling_table: str = "ct_coupling_wb97mv_tzvpd.json"   # the coupling table of ct_kernel "table"
                                      # (scripts/cmm_ct_coupling.py; params/). A checkpoint carries its own
    ct_breath: bool = False           # ct_kernel "table": the coupling follows the ions' breathing,
                                      # W = W_ab + dW_donor z_i + dW_acceptor z_j, z the ions' CT-free breathing
                                      # amplitudes and dW the table's derivatives along the same breathing
                                      # functions (scripts/cmm_ct_coupling.py --breath; no parameter). Clusters only
    ct_closed: str = "off"            # the closed-channel (core relaxation) part of short-range CT. "pair": each
                                      # ion's electrons in a neighbour's site potential, the halves of the site
                                      # Pauli energy, times -f[neighbour species], f = ct_closed_max sigmoid(raw)
                                      # learned per species: an attractive, charge-free fraction of the Pauli
                                      # energy, reported with CT (claude/ct_closed_channel_form.md). Needs
                                      # pauli_form = "site". Clusters only
    ct_closed_max: float = 0.5        # the largest fraction f
    ct_overlap_window: tuple = (0.06, 0.02)   # a pair's coupling is switched off between the outermost distances at
                                      # which its |S| is the first and the second value
    ct_q_headroom: float = 0.0        # one-sided CT charges (the gap model; ct_q_onesided): 0 clips the ln factor at
                                      # 0, a kink wherever the readout changes sign (a jump in the forces). h > 0:
                                      # the smooth f(u) = h - c tanh(softplus(-beta u) / (beta c)), c = ct_q_cap,
                                      # beta = ln 2 / (c atanh(h / c)): f(0) = 0 (the free ion), slope
                                      # (1 - (h / c)^2) / 2 there, f < h (the ceiling is e^h times the free-ion
                                      # charge) and f > h - c. Needs 0 <= h < ct_q_cap
    env_gate: tuple = (1e-7, 1e-5)    # probe overlap at which a neighbour's point-charge field starts / fully counts
    env_overlap: tuple = (1e-7, 1e-6)  # density overlap at which an exchange pair's overlap ratios (E0 / O: the xpol
                                      # drive and Gram weight, the acceptor shift) and its channel weights start /
                                      # fully count; the start cannot go below the table's noise floor, 1e-7
    env_switch: str = "cubic"         # the step of the overlap windows (the probe gate, the exchange pairs' overlap
                                      # switch): "cubic" t^2 (3 - 2 t) is C^1, so the curvature of the energy jumps
                                      # where a window opens or closes; "quintic" is C^2, "smooth" C^infinity
                                      # (channels.switch_step)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pairs = tuple(self.pairs)
        if self.ct_alpha_radical not in ("off", "fixed", "learned"):
            raise ValueError(f"ct_alpha_radical {self.ct_alpha_radical!r}: 'off', 'fixed' or 'learned'")
        if self.ct_alpha_solve not in ("joint", "outer"):
            raise ValueError(f"ct_alpha_solve {self.ct_alpha_solve!r}: 'joint' or 'outer'")
        if self.ct_alpha_radical != "off" and (self.ct_model != "gap" or self.hyperpol != "saturating"):
            raise ValueError("ct_alpha_radical needs ct_model = 'gap' and hyperpol = 'saturating'")
        if isinstance(self.ct_gap_centred, bool):
            self.ct_gap_centred = "species" if self.ct_gap_centred else "off"
        if self.ct_gap_centred not in ("off", "species", "overlap"):
            raise ValueError(f"ct_gap_centred {self.ct_gap_centred!r}: 'off', 'species' or 'overlap'")
        self.ct_centre_window = tuple(float(x) for x in self.ct_centre_window)
        if len(self.ct_centre_window) != 2 or not 0 < self.ct_centre_window[1] < self.ct_centre_window[0] <= 1:
            raise ValueError(f"ct_centre_window {self.ct_centre_window}: needs 0 < off < on <= 1")
        self.env_gate = tuple(float(x) for x in self.env_gate)
        self.env_overlap = tuple(float(x) for x in self.env_overlap)
        if len(self.env_overlap) != 2 or not 1e-7 <= self.env_overlap[0] < self.env_overlap[1]:
            raise ValueError(f"env_overlap {self.env_overlap}: needs 1e-7 <= on < full")
        bad = set(self.pairs) - set(PAIR_GROUPS)
        if bad or not self.pairs:
            raise ValueError(f"pairs {self.pairs!r}: a non-empty subset of {sorted(PAIR_GROUPS)}")
        if not 0 < self.env_gate[0] < self.env_gate[1]:
            raise ValueError(f"env_gate {self.env_gate}: needs 0 < on < full")
        if self.ct_model not in ("flux", "gap"):
            raise ValueError(f"ct_model {self.ct_model!r}: 'flux' or 'gap'")
        if self.ct_gap_overlap not in ("probe", "pauli", "site"):
            raise ValueError(f"ct_gap_overlap {self.ct_gap_overlap!r}: 'probe', 'pauli' or 'site'")
        if (self.ct_gap_overlap == "site" or self.ct_gap_donor) and self.pauli_form != "site":
            raise ValueError("ct_gap_overlap = 'site' and ct_gap_donor need pauli_form = 'site'")
        if self.pauli_form not in ("shells", "site"):
            raise ValueError(f"pauli_form {self.pauli_form!r}: 'shells' or 'site'")
        if self.pauli_env not in ("product", "mean", "linear", "global", "ion", "edge"):
            raise ValueError(f"pauli_env {self.pauli_env!r}: 'product', 'mean', 'linear', 'global', 'ion' or 'edge'")
        if self.pauli_env != "product" and self.pauli_form != "site":
            raise ValueError(f"pauli_env = {self.pauli_env!r} needs pauli_form = 'site'")
        if self.xpol_form not in ("overlap", "site"):
            raise ValueError(f"xpol_form {self.xpol_form!r}: 'overlap' or 'site'")
        if self.xpol_form == "site" and self.pauli_form != "site":
            raise ValueError("xpol_form = 'site' needs pauli_form = 'site'")
        if self.xpol_bound not in ("off", "rank", "scalar"):
            raise ValueError(f"xpol_bound {self.xpol_bound!r}: 'off', 'rank' or 'scalar'")
        if self.disp_model not in ("emitted", "response"):
            raise ValueError(f"disp_model {self.disp_model!r}: 'emitted' or 'response'")
        if self.disp_confine not in ("pauli", "onsite", "none"):
            raise ValueError(f"disp_confine {self.disp_confine!r}: 'pauli', 'onsite' or 'none'")
        if self.ct_gap_converge < 0:
            raise ValueError("ct_gap_converge: a number of iterations >= 0")
        if self.ct_exclusion and not (self.ct_model == "gap" and self.ct_gap_converge > 0):
            raise ValueError("ct_exclusion needs ct_model = 'gap' and ct_gap_converge > 0 (it has no one-step form)")
        if self.ct_kernel not in ("slater", "frontier", "table"):
            raise ValueError(f"ct_kernel {self.ct_kernel!r}: 'slater', 'frontier' or 'table'")
        if self.ct_kernel in ("frontier", "table") and self.ct_model != "gap":
            raise ValueError(f"ct_kernel = {self.ct_kernel!r} needs ct_model = 'gap'")
        if self.ct_breath and self.ct_kernel != "table":
            raise ValueError("ct_breath needs ct_kernel = 'table'")
        if self.ct_closed not in ("off", "pair"):
            raise ValueError(f"ct_closed {self.ct_closed!r}: 'off' or 'pair'")
        if self.ct_closed != "off" and self.pauli_form != "site":
            raise ValueError("ct_closed needs pauli_form = 'site'")
        if not 0 < float(self.ct_closed_max) < 1:
            raise ValueError(f"ct_closed_max {self.ct_closed_max}: a fraction in (0, 1)")
        if self.ct_cut not in ("taper", "overlap"):
            raise ValueError(f"ct_cut {self.ct_cut!r}: 'taper' or 'overlap'")
        if self.ct_cut == "overlap" and self.ct_model != "gap":
            raise ValueError("ct_cut = 'overlap' needs ct_model = 'gap'")
        self.ct_overlap_window = tuple(float(x) for x in self.ct_overlap_window)
        if len(self.ct_overlap_window) != 2 or not 0 < self.ct_overlap_window[1] < self.ct_overlap_window[0]:
            raise ValueError(f"ct_overlap_window {self.ct_overlap_window}: needs s_on > s_off > 0")
        if self.hyperpol not in ("legendre", "saturating"):
            raise ValueError(f"hyperpol {self.hyperpol!r}: 'legendre' or 'saturating'")
        if self.env_switch not in ("cubic", "quintic", "smooth"):
            raise ValueError(f"env_switch {self.env_switch!r}: 'cubic', 'quintic' or 'smooth'")
        self.ct_q_headroom = float(self.ct_q_headroom)
        if not 0 <= self.ct_q_headroom < self.ct_q_cap:
            raise ValueError(f"ct_q_headroom {self.ct_q_headroom}: needs 0 <= h < ct_q_cap = {self.ct_q_cap}")
        if self.hyper_ceiling not in ("global", "species"):
            raise ValueError(f"hyper_ceiling {self.hyper_ceiling!r}: 'global' or 'species'")
        self.ct_floor, self.ct_relay, self.ct_gap_w0 = float(self.ct_floor), float(self.ct_relay), float(self.ct_gap_w0)
        self.ct_gap_predict = int(self.ct_gap_predict)
        if self.ct_gap_predict < 0:
            raise ValueError(f"ct_gap_predict {self.ct_gap_predict} must be >= 0")
        if self.ct_floor < 0 or self.ct_relay < 0 or self.ct_gap_w0 < 0:
            raise ValueError(f"ct_floor {self.ct_floor}, ct_relay {self.ct_relay} and ct_gap_w0 {self.ct_gap_w0} must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"], d["env_gate"], d["env_overlap"] = list(self.pairs), list(self.env_gate), list(self.env_overlap)
        d["ct_overlap_window"] = list(self.ct_overlap_window)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CmmConfig":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)
