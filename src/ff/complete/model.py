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

    E_0,i = -1/2 a0[s] e^lna[s] (V[breath]_i^(0) + lam0[s] S[breath]_i^(0))^2

with V[breath]^(0) the penetration potential of all the other frozen densities over the ion's l = 0 response shape
(the ``breath`` probe of the environment) and S[breath]^(0) their overlap with it: the Coulomb drive (fitted to the
CPKS point-charge response) and the Pauli-pressure drive (exchange polarization at l = 0 -- the density contracts
to relieve overlap; lam0 learned, 0 at start). The drive is a scalar, so it adds coherently over neighbours -- the
polarization of an ion in a symmetric cage that no dipole / quadrupole response can give. The breathing
polarizability a0 e^lna is a per-species constant (no environment head): with xpol its environment dependence
is the drive and the exchange coupling below.

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

Quadratic exchange polarization (``xpol=True``, with ``frozen``; the default of ``train_ionct --model complete
--frozen``, in place of the pseudo-potential drive and the separate breathing term, with the confinement head kept
for the many-body stiffening -- NaCl m29_xpol): the frozen Pauli energy of every pair, E_ij^0 (the penetration
kernel of mod_pauli), is made a function of the induced moments x and breathing amplitudes b through one overlap
amplitude per pair, the relative change of the two density overlaps,

    u_ij = kappa_j (d_ij . x_j + e_ij b_j) / O_ij + kappa_i (d_ji . x_i + e_ji b_i) / O_ji,
    E_P(x) = sum_{i<j} E_ij^0 [1 + 2 u_ij + lam_ij u_ij^2],      lam_ij = 1 + (l_i + l_j) / 2 >= 1

with O_ij = <rho_i | rho_j> the density-density overlap of the two ions' electrons, d_ij its change per unit induced
moment of j and e_ij per unit breathing amplitude of j (``XpolTable``; the partner's full density, not its outer shell
alone: for Cl- partners the compact inner shell doubles the relative change at contact, which the Cl-Cl pair
polarization needs). The linear term is the exclusion drive (pseudo-potential)
with the pair's own Pauli energy as its strength; the quadratic term is positive semi-definite by construction and holds
the on-site stiffening (confinement, along the bond) and the inter-site coupling of the same amplitude; lam >= 1 keeps
every pair's E_P >= 0 for any x (lam = 1: the linearised overlap-squared limit). kappa_i = kappa0[s, l]
exp(head(S_i)) per component (dipole, quadrupole, breathing). The head's inputs (default ``xpol_kappa_env =
"scalefree"``) are three scale-free coordination descriptors of i's frozen pair overlaps (``xpol_scalefree``:
n_eff - 1, eta1, eta2), all exactly 0 for a single neighbour at any distance, so kappa = kappa0 on every pair and the
exchange term's R dependence is E0_ij and the tabulated d / O alone; ``"overlap"`` uses the frozen overlap invariants
(R dependent, as in checkpoints from before the scale-free inputs). Either way the inputs are frozen, so the
functional stays quadratic and its minimum is one linear solve,

    (A + H) z = -(L + g),   z = (x, b),   E_pol = 1/2 z A z + L z + [g z + 1/2 z H z]

with the breathing amplitude's own stiffness 1 / (a0 e^lna) and Coulomb drive V[breath]^(0) in A and L. E_P(z) - E_P(0)
is exchange polarization and is booked in e_pol (``e_xpol``); e_pauli stays E_P(0). Periodic frames: the same
functional over every pair and image of the Pauli pair list, solved by matrix-free CG with the Ewald frames.

Gap-form CT (``ct_model="gap"``, with ``frozen``): fixed couplings, the environment only in an additive gap,

    W_c^2 = qD0[s_i] qA0[s_j] f(bD_i, bA_j; r) / r          (per-species constants, no environment)
    D_c   = I_D(i) - A_A(j) + phiD(i) - phiA(j) + delta_A(j) - delta_D(i) [- Stark_c]
    E_c   = (D_c - sqrt(D_c^2 + 4 W_c^2)) / 2              (or the per-ion saturation, ct_sat_ion)

phiD(i) / phiA(j) the full potential of every other atom (permanent shells, induced moments, external field, the
solvent's reaction potential) averaged over i's donor / j's acceptor frontier density: the orbital-energy gap of a
perturbative transfer (no electron-hole term). delta_X(k) = a_X[s_k] S_k exp(h_X(I_k)) >= 0, S_k the ion's outer-shell
overlap invariant and h a head on the overlap invariants only: the overlap raising the frontier levels, zero with the
overlap (``gap_shift="acceptor"``, the default: delta_D = 0). The Stark term (``ct_stark``) is additive here, 1/2 dalpha_D |E_D|^2 + 1/2 dalpha_A |E_A|^2. The donor /
acceptor probe blocks are left out of every head's input (``head_frontier=False``): the frontier shapes are operators
of the gap, not features.

Perpendicular exchange stiffening (``xpol_perp``, on by default with xpol; ``--no-xpol-perp``; NaCl m30_xperp): lam u^2 is rank
one per pair -- it stiffens only the deformation that changes O_ij (for two spherical ions: the dipole along the bond,
the m = 0 quadrupole, breathing), so a pair's polarizability across the bond feels no exchange at all. The term

    E0_ij k_perp,ij sum_blocks ( |v_i,b|^2 |z_i,b|^2 - (v_i,b . z_i,b)^2 ) + (i <-> j),   k_perp = (p_i + p_j)/2 >= 0

(blocks: dipole, quadrupole) adds the missing on-site stiffness across the overlap direction with the same Pauli weight
and the same sensitivity (Lagrange identity: |v|^2|z|^2 = (v.z)^2 + |v x z|^2). It is non-negative and on-site only,
so positivity and the single linear solve are unchanged; with lam it gives each ion the exchange stiffness
2 E0 [lam v v^T + k_perp (|v|^2 1 - v v^T)] per pair: parallel and perpendicular rates, in place of a confinement head.

Charge-flux CT (``ct_model="flux"``, with xpol; NaCl 2026-11): every direct channel c = i -> j moves an explicit
charge q_c (an electron from i to j: Delta Q_i = +q_c, Delta Q_j = -q_c), placed in each ion's dipole-parent shape (an
l = 0 charge slot next to the induced dipole, so its couplings are the induced-moment kernels' rank-0 rows; no frontier
shapes). The direct-CT kernel e_c = -qD_i qA_j f(bD, bA; r) / r <= 0 (environment heads as before, no gap factor, no
saturation) sets the channel's compliance, SQE style:

    E(z, q) = E_xpol(z) + sum_c [ 1/2 b_c q_c^2 - V_c q_c + q_c (phi_ext(i) - phi_ext(j)) ]
              + 1/2 Q^T H Q + Q^T C^T z,                 Q = G q (atomic charge changes)
    b_c = V0_c^2 / (2 |e_c|),   V0_c = v0[s_i] + v0[s_j],   V_c = t_p sqrt(V0_c^2 + 2 |e_c| h_c),   h_c = (G^T H G)_cc

H the Coulomb matrix of the charge slots (shape self-energies on the diagonal) plus a learned on-site hardness
eta[s] >= 0 (started at max(I - A - J_self, 0.01): the ions' IP / EA hardness), C the coupling of the charges to the
induced moments (dipole / quadrupole parents). H is positive definite and b_c > 0, so the charge block is PD. The
drive is tied to h_c: a lone channel (t = 1) at zero field gives exactly E = -V_c^2 / (2 (b_c + h_c)) = e_c, q* =
2 |e_c| / V_c and alpha_CT,par = 2 |e_c| R^2 / V_c^2 = R^2 / (b_c + h_c). The two directions of a pair have opposite
charge patterns (coupled by -h: counter-flow cancels the Coulomb cost), so their drives share the factor t_p <= 1 that
makes the pair's own 2x2 minimum exactly e_c + e_c' (a symmetric pair moves no net charge and gets V = sqrt(2 |e| b)).
The compliance 1 / b_c vanishes with e_c (no long-range CT).
Everything else -- the field, other channels (shared atoms through H), the induced moments and through them xpol --
enters linearly, and the joint minimum is one solve. The channels couple to each other and to everything else only
through the atomic charge changes Q = G q (the only per-channel pieces are b_c and V_c), so they are eliminated exactly
(``flux_solver="atomic"``, the default): stationarity in q gives

    q = w (V - G^T psi),   w_c = 1 / b_c = s_c^2,   psi = H Q + C^T z + phi_ext (the potential on the charges)
    Q = u - K psi,         K = G diag(w) G^T (an N x N weighted graph Laplacian),   u = G w V

and with A = M + H_xpol, W = A^-1 C (N extra right-hand sides of the xpol solve, C the atomic charges' coupling) and
S = H - C^T W (N x N) the joint minimum is

    (1 + K S) Q = u - K (phi_ext + C^T z_pol),   z* = z_pol - W Q,   E_CT = E_joint - E_pol = 1/2 b~ . q*,
    b~ = -V + G^T (phi_ext + C^T z_pol)

1 + K S is nonsingular (K PSD, S PD) and a vanishing channel (w_c -> 0) drops out of K and u, so the cost is the xpol
LU, N right-hand sides and an N x N solve, plus O(C) scatters: independent of the channel count. ``flux_solver="channel"``
keeps the original channel-space form for reference (identical to ~1e-14): W = A^-1 G-mapped couplings with one
right-hand side per channel and the Schur complement S = diag(b) + G^T H G - C^T W over the channels (solved in
q_c / s_c with s_c = sqrt(2|e_c|) / V0_c), q* = -S^-1 b~ -- (D N)^2 C + C^2 D N + C^3, with C = N (N - 1) for a cluster
inside the 8 A CT cutoff (5434 channels for 125 ions: 3.0 s vs 0.39 s per CPU forward).

(ALMO bookkeeping: the CT-induced repolarization is CT). Clusters only; no solvent; no permanent third-body potential
on the charges (the heads see it; the induced moments' potential acts through C, the pair's own included: for Na-Cl it
takes the dimer E_CT to ~0.65-0.75 of e_c + e_c' at the untrained start).

Species enter only through the isolated-atom parameters (alpha0, C0, the CT charges qD0 / qA0 and exponents, kappa,
lam, gamma0); the heads are shared (``species_emb`` adds an embedding to their input).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..dispersion import DEFAULT_C6_PRIOR, tt_damped_c6_energy
from ..ionct import kernels as K
from ..ionct.model import A_UNBOUND_INIT, PAULI_INIT, PSEUDO_CUTOFF, IonCTModel
from ..ionct.params import IonTable
from ..ionct.ct import ChannelTable
from ..multipole import cartesian_to_spherical_quadrupole, slater_two_center_damp_unequal
from .elst import permanent_electrostatics
from .heads import EnvironmentHead, SharedEnvironment
from .periodic import _matvec, _perp_quad, dispersion_switch, recip_apply, solve_induction_periodic, solve_xpol_periodic
from .tabulate import FluxTable, XpolTable, flux_pair_terms, shape_self_coulomb, xpol_overlap_map
from .environment import inv_form_of


def _mag_col(names, probe, key):
    """Column of the magnitude invariant ``key`` (V1, V2, S1, S2) of ``probe`` and whether it is squared:
    "<probe>:|<key>|" (inv_form norm) or "<probe>:|<key>|^2" (inv_form sq); (None, False) if absent."""
    for nm, sq in ((f"{probe}:|{key}|", False), (f"{probe}:|{key}|^2", True)):
        if nm in names:
            return names.index(nm), sq
    return None, False


def _mag_scale(names, feat_scale, probe, key):
    """The training-set scale of the magnitude |<key>| of ``probe`` (field, gradient, overlap gradient ...) in its
    own units: the feature scale of |x|, or the square root of that of |x|^2; 1 if unknown."""
    col, sq = _mag_col(list(names or []), probe, key)
    if col is None or feat_scale is None:
        return 1.0
    v = float(feat_scale[col])
    return math.sqrt(v) if sq else v

__all__ = ["CompleteModel", "induction_matrix_blocks", "solve_induction_blocks"]

GAP_EXP_MAX = 4.0     # bound on the log of the CT gap factor
GAP_DELTA_INIT = 0.05  # Hartree per unit scaled overlap: start of the gap form's overlap shifts (delta_D, delta_A)
STARK_DALPHA = 100.0   # bohr^3 per unit softplus(stark_D/A) in the gap form's additive Stark term
CT_GAP_MIN = 0.05     # Hartree: floor of the CT gap (self-consistent CT), a softplus of width CT_GAP_WIDTH above it
CT_GAP_WIDTH = 0.02


def _gap_floor(D):
    return CT_GAP_MIN + CT_GAP_WIDTH * nn.functional.softplus((D - CT_GAP_MIN) / CT_GAP_WIDTH)
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
XPOL_R_MAX = 10.0     # Angstrom: end of the exchange-polarization overlap table (the overlaps are < 1e-9 there)
#: the overlap window over which the xpol ratios d ln O switch on (above the table's quadrature noise, ~1e-8 for Na+-Na+)
XPOL_O_ON, XPOL_O_FULL = 1e-7, 1e-6
#: xpol: start of kappa0 per species and component (dipole, quadrupole, breathing) and of lam - 1, from a fit of
#: these constants alone (heads off, no confinement, no pseudo drive) to the NaCl EDA pol labels: train pol MAE
#: 0.58 kJ/mol per ion (the free-ion model: ~10)
XPOL_KAPPA_INIT = {"Na": (0.07, 0.013, 0.1), "Cl": (0.28, 0.25, 0.34)}
XPOL_KAPPA_FALLBACK = (0.3, 0.3, 0.3)
XPOL_LAM_INIT = {"Na": 1.8, "Cl": 3.2}     # lam - 1 at start
XPOL_LAM_FALLBACK = 2.0
#: xpol_perp: p (k_perp,ij = (p_i + p_j)/2) at start, softplus of one parameter per species. Pair E_pol does not see
#: k_perp, so it is learned from alpha and asymmetric clusters only and moves slowly: start near the NaCl m30 values
#: (xperp_sf: Na 5.67, Cl 0.079) rather than small
XPOL_KPERP_INIT = {"Na": 6.0, "Cl": 0.1}
XPOL_KPERP_FALLBACK = 0.1
#: the component blocks the perpendicular stiffening acts in (dipole, quadrupole); breathing is a scalar
XPOL_PERP_BLOCKS = ((0, 3), (3, 8))
#: xpol_perp_weight="sqrt_overlap": W_ij = XPOL_PERP_E_S sqrt(O_ij / XPOL_PERP_O_S) in place of E0_ij -- half the decay
#: rate (E0_ij is ~5 O_ij for these shells, so O_ij itself would decay like E0); equal to E0 at Na-Cl 2.6 A
XPOL_PERP_O_S = 4.0e-3
XPOL_PERP_E_S = 0.023
#: xpol_kappa_env="scalefree": the kappa head's inputs, exactly constant along any pair (see xpol_scalefree)
XPOL_SCALEFREE_NAMES = ("n_eff - 1", "eta1", "eta2")
# charge-flux CT: start of v0[s] (Hartree), V0_c = v0_i + v0_j. NaCl at 3 A: ALMO alpha_CT,par = 16 bohr^3 means a
# total stiffness b + h = R^2 / alpha ~ 2 Eh; with h ~ 1.5 (IP hardness, slot Coulomb) b ~ 0.5 and V0^2 = 2 |e_c| b ~ 2e-3
FLUX_V0_INIT = 0.025


def induction_matrix_blocks(A_off, alpha_inv, C, quadrupoles=True):
    """(B, 8N, 8N) the induction matrix A = A_off + the on-site blocks (alpha_inv (B, N, 3, 3), C (B, N))."""
    B, N = C.shape
    cinv = torch.where(C > 0, 1.0 / C.clamp(min=1e-30), torch.full_like(C, 1e12)) if quadrupoles else torch.full_like(C, 1e12)
    D = alpha_inv.new_zeros(B, N, 8, 8)
    D[:, :, :3, :3] = alpha_inv
    D[:, :, 3:, 3:] = torch.diag_embed(cinv[..., None].expand(B, N, 5))
    ar = torch.arange(N, device=C.device)
    Ad = alpha_inv.new_zeros(B, N, 8, N, 8)
    Ad[:, ar, :, ar, :] = D.permute(1, 0, 2, 3)
    return A_off + Ad.reshape(B, 8 * N, 8 * N)


def solve_induction_blocks(L, A_off, alpha_inv, C, quadrupoles=True):
    """Like :func:`rsfff.ff.ionct.kernels.solve_induction` with a 3x3 inverse polarizability per site:
    L (B, N, 8), A_off (B, 8N, 8N), alpha_inv (B, N, 3, 3) symmetric positive definite, C (B, N)."""
    B, N, _ = L.shape
    cinv = torch.where(C > 0, 1.0 / C.clamp(min=1e-30), torch.full_like(C, 1e12)) if quadrupoles else torch.full_like(C, 1e12)    # C = 0: no quadrupole response (Li+)
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

    PEN_PROBES = ("donor", "acceptor", "breath")
    PEN_NAMES = tuple(f"{p}:{k}" for p in PEN_PROBES for k in ("S0", "V0_pen", "|V1_pen|^2", "|V2_pen|^2", "V1_pen.S1"))

    def __init__(self, tab: IonTable, chan: ChannelTable, n_feat: int, *, feature_names=None, hidden=64, depth=2,
                 feat_scale=None, c_range=(0.05, 1.25), confine=None, pseudo=True, ps_init=0.02, tie_c=True,
                 species_emb=False, aniso=True, ct_gap=True, dphi_scale=0.05, frozen=False, conf_form="tanh",
                 hyper=False, pauli_moments=False, pauli_form="slater", ct_stark=False, pauli_ii=False, conf_head=True, ct_sc=0,
                 env_cap=0.0, env_log=0.0, ct_sat=False, ct_q_cap=0.0, gap_cap=GAP_EXP_MAX, ct_sat_ion=False,
                 ct_w_fixed=False, xpol=False, ct_model="direct", head_frontier=None, gap_shift="acceptor",
                 xpol_perp=None, xpol_kperp_init=None, xpol_kappa_head=True,
                 xpol_perp_weight="pauli", xpol_kappa_env="scalefree", flux_v0_init=FLUX_V0_INIT,
                 flux_solver="atomic", env_channels=False, env_latent=4, env_mode="shared",
                 env_two_body=("pauli", "disp", "ct"), ct_q_head=True, pin_a0=False, env_extra=(),
                 env_extra_terms=("ct",), env_field=False, ct_legacy_head=False, env_ct_pen=False,
                 env_extra_capacity="full", ct_outer=1, ct_outer_tol=1e-10, ct_gap_min=0.03,
                 ct_gap_learn=False):
        super().__init__(tab, chan, n_feat, hidden=hidden, depth=depth, feat_scale=feat_scale, c_range=c_range,
                         confine=confine, pseudo=False, tie_c=tie_c, ct_form="direct", gap_form="pair",
                         pol_form="confine")
        self.pseudo = pseudo
        # env_cap > 0: the log environment factors of the CT charges, Pauli weights, C6 and exclusion strengths
        # are cap * tanh(h / cap), so each parameter stays within e^+-cap of its isolated value
        # however far the invariants extrapolate (an unbounded exp(h) took the NaCl dimer CT to -1e8 kJ/mol at 1.8 A)
        self.env_cap = float(env_cap)
        self.env_log = float(env_log)          # > 0: the heads read the signed log of the invariants (heads.py)
        # ct_sat: each direct CT channel is a two-state problem, E_c = (D_c - sqrt(D_c^2 + 4 W_c^2)) / 2, with the
        # coupling W_c^2 = -e_c D_c0 from the pair kernel e_c and the pair gap D_c0, and the environment (gap, Stark)
        # lowering the gap, D_c = D_c0 exp(-u), in place of scaling the energy by exp(u). The same -W^2/D_c = e_c e^u
        # while 2 W << D; but |E_c| <= W_c whatever the gap does, and W_c is finite at r -> 0 (the Slater kernel
        # is), so the channel saturates at short range instead of running away with the environment.
        self.ct_sat = bool(ct_sat)
        # ct_sat_ion: saturate per ion, not per channel. A donor has one frontier pair to give and an acceptor one
        # hole; each channel bounded on its own still lets ~2400 of them sum without limit in a compressed crystal.
        # Every donor (and every acceptor) is a two-state problem coupled to all of its channels at once,
        # E_g = (D_g - sqrt(D_g^2 + 4 S_g)) / 2 with S_g = sum_c W_c^2 and the effective gap D_g = S_g / sum_c W_c^2 / D_c
        # (so E_g = sum_c -W_c^2 / D_c while 2 W << D): |E_g| <= sqrt(S_g), the square root of the channel count.
        # Each channel keeps its perturbative energy times the smooth minimum of its donor's and acceptor's
        # saturation factors E_g / sum_c (-W_c^2 / D_c); a lone channel is exactly the per-channel two-state form.
        self.ct_sat_ion = bool(ct_sat_ion)
        if self.ct_sat_ion and not self.ct_sat:
            raise ValueError("ct_sat_ion needs ct_sat (the two-state channels)")
        # ct_w_fixed: the coupling W_c from the isolated ions' CT charges only (W_c^2 = -e_c0 D_c0, e_c0 the pair
        # kernel at q0), and the environment heads' charge factors lower the gap instead, D_c = D_c0 exp(-(u + g_c)),
        # g_c = ln(qD_i / qD0_i) + ln(qA_j / qA0_j). The same -W^2/D = e_c0 e^(u + g) while 2 W << D, but every bound
        # (|E_c| <= W_c, per ion sqrt(sum W_c^2)) is now fixed by the pair geometry, whatever the environment does
        self.ct_w_fixed = bool(ct_w_fixed)
        if self.ct_w_fixed and not self.ct_sat:
            raise ValueError("ct_w_fixed needs ct_sat (the two-state channels)")
        # output bounds from the training range: the CT charges' log factors within +-ct_q_cap (0: env_cap), the
        # gap / Stark exponent within +-gap_cap. Inside the data the heads stay near |h| ~ 1; an MD that is free to
        # look for the heads' maxima found them at the caps (Cl- qD x 24, Na+ qA x 13 at ordinary 2.7 A contacts)
        self.ct_q_cap, self.gap_cap = float(ct_q_cap), float(gap_cap)
        # feat_sat > 0 (per invariant, set by the trainer from the training range): the heads read
        # m tanh(I / m) of the scaled invariants, so past the data every parameter tends to a constant and each term
        # keeps its functional form's own short-range limit (the overlap and penetration invariants grow as e^(-bR))
        self.register_buffer("feat_sat", torch.zeros(n_feat, dtype=torch.float64))
        self._s0sq = None
        # the environment invariants' form (squared norms or magnitudes), read off the featurizer's names; callers
        # of batched_precompute pass it through so the tabulated features match the ones the model was trained on
        self.inv_form = inv_form_of(feature_names)
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
            self.hyper_col, self.hyper_sq = _mag_col(names_, "dip", "V1")
            if self.hyper_col is None:
                raise ValueError("hyper needs the dip:|V1| (or |V1|^2) invariant")
            self.hyper_g = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
        self.conf_form = conf_form
        self.conf_head = conf_head          # False: the stiffening rate is the per-species constant alone (no head)
        if conf_form == "mono":
            self.conf_a = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
            self.conf_s0_col = list(feature_names or []).index("shell:S0")
        if conf_form == "sq":
            # overlap-squared stiffening, no head: alpha_i^-1 -> alpha_i^-1 (1 + sp(h[s]) sum_j S_ij^2 / s^2), S_ij the
            # overlap of i's dipole parent with j's density -- the second-order (orbital-overlap, (dS)^2) Pauli
            # energy of the induced density, in the solve; exactly 1 for an isolated atom
            self.conf_h = nn.Parameter(torch.full((S,), -1.0, dtype=torch.float64))
            names_ = list(feature_names or [])
            ssq = float(feat_scale[names_.index("dip:S0")]) ** 2 if ("dip:S0" in names_ and feat_scale is not None) else 1.0
            self.register_buffer("conf_ssq", torch.tensor(ssq, dtype=torch.float64))
        # pauli_ii: the induced-induced Pauli kernel in the solve, A -> A + kappa_i kappa_j A_pp (the penetration part
        # of the parent-parent coupling, cached); kappa per species, free sign, small nonzero start (the gradient of
        # a product vanishes at zero)
        # ct_sc: self-consistent (variational) CT, ct_sc fixed-point iterations. Every channel is a two-state
        # problem with population p_c: E_c = Delta_c p_c - 2 W_c sqrt(p_c (1 - p_c)), whose minimum over p is
        # 1/2 (Delta_c - sqrt(Delta_c^2 + 4 W_c^2)) (= -W^2/Delta for small p, the direct form); the coupling W_c is
        # set by the direct-form pair kernel, W_c^2 = -e_c0 Delta_c0, and the gap Delta_c = Delta_c0 + g_c dphi_c(x)
        # - Stark_c carries the environment's potential difference (permanent + induced + external) over the frontier
        # shapes, in place of the exponential gap factor. Joint minimum over the induced moments x and the
        # populations: A x + L + sum_c g_c p_c d(dphi_c)/dx = 0 -- the populations, as holes in the donor and
        # electrons in the acceptor frontier shapes, drive the induction through the frontier operators -- and
        # p_c from Delta_c(x). ALMO bookkeeping: E_pol is the CT-free solve, E_CT = E_total - E_pol (the CT-induced
        # repolarization is CT).
        self.ct_sc = int(ct_sc)
        self.ct_model = str(ct_model)
        if self.ct_model not in ("direct", "gap", "flux", "twostate"):
            raise ValueError(f"ct_model {ct_model!r}")
        self.flux_solver = str(flux_solver)       # flux CT: "atomic" (N x N, default) or "channel" (reference)
        if self.flux_solver not in ("atomic", "channel"):
            raise ValueError(f"flux_solver {flux_solver!r}")
        if self.ct_model in ("flux", "twostate"):
            if ct_sc or ct_sat or ct_stark or ct_w_fixed:
                raise ValueError(f"ct_model={self.ct_model!r}: no ct_sc, ct_sat, ct_stark, ct_w_fixed (the joint solve replaces them)")
            if not xpol:
                raise ValueError(f"ct_model={self.ct_model!r} needs xpol=True (the charges join the xpol solve)")
            ct_gap = False                  # no multiplicative gap factor: the field acts on the charges directly
        if self.ct_model == "gap" and self.ct_sc:
            raise ValueError("ct_model='gap': no ct_sc (self-consistent CT not implemented for it yet)")
        if (self.ct_sc or self.ct_sat or self.ct_model == "gap") and not hasattr(self, "A_fixed"):     # the pair gap
            unbound = torch.isnan(tab.A_acceptor)
            self.register_buffer("A_fixed", torch.nan_to_num(tab.A_acceptor))
            self.register_buffer("A_unbound", unbound)
            self.A_free = nn.Parameter(torch.full((S,), A_UNBOUND_INIT, dtype=torch.float64))
        if self.ct_sc:
            self.sc_gD = nn.Parameter(torch.zeros(S, dtype=torch.float64))     # log of the gap's response to dphi
            self.sc_gA = nn.Parameter(torch.zeros(S, dtype=torch.float64))
        self.pauli_ii = pauli_ii
        if pauli_ii:
            self.pp_kappa = nn.Parameter(torch.full((S,), 0.1, dtype=torch.float64))
        self.head_ctq = EnvironmentHead(n_feat, 2, hidden, depth, ns)      # log qD / qD0, log qA / qA0
        self.ct_lnq0 = nn.Parameter(torch.zeros(S, 2, dtype=torch.float64))
        self.aniso, self.ct_gap, self.dphi_scale = aniso, ct_gap, dphi_scale
        names = list(feature_names or [])
        s2 = 1.0
        if aniso:
            self.kappa_aniso = nn.Parameter(torch.zeros(S, dtype=torch.float64))
            s2 = _mag_scale(names, feat_scale, "dip", "S2")
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
            self.stark_sD = _mag_scale(names_, feat_scale, "donor", "V1")
            self.stark_sA = _mag_scale(names_, feat_scale, "acceptor", "V1")
            self.stark_D = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
            self.stark_A = nn.Parameter(torch.full((S,), -3.0, dtype=torch.float64))
        self.breath = bool((tab.a0 > 0).any()) and "breath:V0_pen" in names
        if self.breath:
            self.breath_col = names.index("breath:V0_pen")
            self.breath_s0_col = names.index("breath:S0")
            # a_b = a0 exp(breath_lna): a per-species constant. Its environment dependence is physical: the drive V0
            # and, with xpol, the exchange coupling (E0, k_par, the breathing kappa) -- no environment head
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
            s1 = _mag_scale(names_, feat_scale, "shell", "S1")
            s2 = _mag_scale(names_, feat_scale, "shell", "S2")
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
                # the frozen fit's (w, s) from the ion table, where it has them, override the starts and are the
                # reference values of pauli_prior (the training pulls toward the fit rather than freezing it)
                fit = torch.zeros(S, dtype=torch.bool)
                if tab.pauli_s is not None:
                    for k in range(S):
                        if math.isfinite(float(tab.pauli_s[k])):
                            occ = tab.shell_n[k] > 0
                            lnw[k] = torch.where(occ, torch.log(tab.pauli_w[k].to(lnw)), lnw[k])
                            lnbs[k] = math.log(float(tab.pauli_s[k]))
                            fit[k] = True
                self.register_buffer("pauli_lnw0", lnw.clone())
                self.register_buffer("pauli_lnbs0", lnbs.clone())
                self.register_buffer("pauli_fit_mask", fit.to(torch.float64))
                self.pauli_lnw = nn.Parameter(lnw)
                self.pauli_lnbs = nn.Parameter(lnbs)
                self.register_buffer("pauli_shell_mask", (tab.shell_n > 0).to(torch.float64))
                self.register_buffer("pauli_outer", torch.tensor([max(t for t in range(T) if tab.shell_n[k, t] > 0) for k in range(S)]))
            pri = [DISP_INIT.get(x, DEFAULT_C6_PRIOR.get(int(z), DISP_FALLBACK)) for x, z in zip(sym, tab.Z.tolist())]
            self.disp_lnc60 = nn.Parameter(torch.tensor([math.log(c) for c, _ in pri], **f64))
            self.disp_lnb = nn.Parameter(torch.tensor([math.log(b) for _, b in pri], **f64))
            # the frozen heads see the overlap invariants only (no fields at the frozen level)
            cols = [k for k, nm in enumerate(names)
                    if nm.split(":")[-1] in ("S0", "|S1|", "|S2|", "|S1|^2", "|S2|^2")] or list(range(n_feat))
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
        self.xpol = bool(xpol)
        self.xpol_perp = False
        self.xpol_kappa_env, self.xpol_perp_weight = "overlap", "pauli"
        if xpol_perp is None:           # default: the perpendicular stiffening comes with xpol
            xpol_perp = bool(xpol)
        if xpol_perp and not xpol:
            raise ValueError("xpol_perp needs xpol=True")
        if self.xpol:
            if not frozen:
                raise ValueError("xpol needs frozen=True (the pair Pauli energies)")
            if pseudo:
                raise ValueError("xpol replaces the pseudo-potential drive (pseudo=False)")
            if pauli_ii or ct_sc:
                raise ValueError("xpol: no pauli_ii, no ct_sc")
            ka = [XPOL_KAPPA_INIT.get(x, XPOL_KAPPA_FALLBACK) for x in tab.symbols]
            self.xpol_lnk = nn.Parameter(torch.log(torch.tensor(ka, dtype=torch.float64)))         # (S, 3)
            self.xpol_lam = nn.Parameter(torch.tensor([math.log(math.expm1(XPOL_LAM_INIT.get(x, XPOL_LAM_FALLBACK)))
                                                       for x in tab.symbols], dtype=torch.float64))
            # xpol_kappa_head=False: kappa = kappa0[s, l] only (per species and component, no environment head); the
            # R dependence of the exchange term is then E0_ij and the tabulated d_ij / O_ij alone
            if xpol_kappa_env not in ("overlap", "scalefree"):
                raise ValueError(f"xpol_kappa_env {xpol_kappa_env!r}: overlap or scalefree")
            if xpol_perp_weight not in ("pauli", "sqrt_overlap"):
                raise ValueError(f"xpol_perp_weight {xpol_perp_weight!r}: pauli or sqrt_overlap")
            self.xpol_kappa_env, self.xpol_perp_weight = xpol_kappa_env, xpol_perp_weight
            n_in = len(XPOL_SCALEFREE_NAMES) if xpol_kappa_env == "scalefree" else len(self.frozen_cols)
            self.head_xpol = EnvironmentHead(n_in, 1, hidden, depth, ns) if xpol_kappa_head else None
            self.xpol_perp = bool(xpol_perp)
            if self.xpol_perp:
                kp0 = [{**XPOL_KPERP_INIT, **(xpol_kperp_init or {})}.get(x, XPOL_KPERP_FALLBACK) for x in tab.symbols]
                self.xpol_kperp = nn.Parameter(torch.tensor([math.log(math.expm1(max(v, 1e-6))) for v in kp0],
                                                            dtype=torch.float64))   # p_s: k_perp,ij = (p_i + p_j)/2
            self.register_buffer("xpol_breath_mask", (tab.a0 > 0).to(torch.float64))
            self._xpol_table = None
        if self.ct_model == "flux":
            f64 = dict(dtype=torch.float64)
            self.flux_lnv = nn.Parameter(torch.full((S,), math.log(flux_v0_init), **f64))     # v0[s]: V0_c = v0_i + v0_j
            js = shape_self_coulomb(tab)
            self.register_buffer("flux_jself", js)
            eta_ip = tab.I_donor - torch.nan_to_num(tab.A_acceptor, nan=0.0)
            e0 = (eta_ip - js).clamp(min=0.01)
            self.flux_eta = nn.Parameter(torch.log(torch.expm1(e0)).to(**f64))              # softplus: on-site hardness
            self._flux_table = None
        if self.ct_model == "twostate":
            # two-state channel CT (twostate.py): the gap Delta_c = I_i - A_j + phi_perm(i) - phi_perm(j) - H_ij at
            # unit charge, additive per species (table I, A plus learned shifts); an unbound affinity (A = NaN, e.g.
            # Cl-) gets the on-site estimate A = I - J_self of its slot shape. ct_outer: 1 = linear response about the
            # isolated two-state minimum, > 1 = Newton steps to the exact minimum. ct_gap_min: the gap's smooth floor
            # (twostate.floor_gap). ct_gap_learn: train the per-species shifts dI, dA (default: frozen at 0, the table
            # values -- learned shifts drift onto the floor and move the dimer's Na + Cl crossing in to ~4 A).
            f64 = dict(dtype=torch.float64)
            js = shape_self_coulomb(tab)
            self.register_buffer("flux_jself", js)
            self.register_buffer("ts_I0", tab.I_donor.to(**f64).clone())
            self.register_buffer("ts_A0", torch.where(torch.isnan(tab.A_acceptor), tab.I_donor - js,
                                                      tab.A_acceptor).to(**f64))
            self.ts_dI = nn.Parameter(torch.zeros(S, **f64), requires_grad=bool(ct_gap_learn))
            self.ts_dA = nn.Parameter(torch.zeros(S, **f64), requires_grad=bool(ct_gap_learn))
            self.ct_gap_learn = bool(ct_gap_learn)
            self.ct_outer, self.ct_outer_tol, self.ct_gap_min = int(ct_outer), float(ct_outer_tol), float(ct_gap_min)
            self._flux_table = None
            self._ts_info = {}
        if self.ct_model == "gap":
            if not frozen:
                raise ValueError("ct_model='gap' needs frozen=True (the overlap invariants' columns)")
            # delta_D, delta_A: amplitude per species (Hartree per unit scaled overlap), the head's log factor
            self.gap_lna = nn.Parameter(torch.full((S, 2), math.log(GAP_DELTA_INIT), dtype=torch.float64))
            self.head_gap = EnvironmentHead(len(self.frozen_cols), 2, hidden, depth, ns)
            names_ = list(feature_names or [])
            self.gap_s0_col = names_.index("shell:S0") if "shell:S0" in names_ else 0
            # which frontier levels the overlap shifts: "acceptor" (delta_A >= 0 only: the gap can only open with
            # overlap) or "both" (delta_D lowers it; with no label to stop it the donor shift closed the Na-Cl dimer
            # gap at 2.4-3 A in the first run)
            if gap_shift not in ("acceptor", "both"):
                raise ValueError(f"gap_shift {gap_shift!r}")
            self.register_buffer("gap_shift_mask", torch.tensor([float(gap_shift == "both"), 1.0], dtype=torch.float64),
                                 persistent=False)
        # the donor / acceptor probe blocks as head inputs: kept by default for the direct CT (as trained), left out
        # with the gap form (the frontier shapes are its operators, not features)
        hf = (self.ct_model != "gap") if head_frontier is None else bool(head_frontier)
        keep = torch.ones(n_feat, dtype=torch.float64)
        if not hf:
            for k, nm in enumerate(feature_names or []):
                if nm.split(":")[0] in ("donor", "acceptor"):
                    keep[k] = 0.0
        self.register_buffer("feat_keep", keep, persistent=False)
        for h in self.modules():
            if isinstance(h, EnvironmentHead):
                h.log_k = self.env_log
        # m33: the shared environment on the two- and three-body channels (channels.py) replaces head_frozen, head_xpol
        # and head_ctq as the source of every environment log-factor; the old heads stay built but unused
        self.env_channels = bool(env_channels)
        self.ct_q_head = bool(ct_q_head)
        # ct_legacy_head: the CT strengths keep m32's head_ctq on the density invariants (all 42) while the other
        # parameters move to the shared environment (the hybrid rung of the ladder)
        self.ct_legacy_head = bool(ct_legacy_head)
        self.env = None
        if self.env_channels:
            if not (self.xpol and self.frozen):
                raise ValueError("env_channels needs xpol=True and frozen=True (the channel overlaps are the xpol table's)")
            from .channels import CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS
            tb = set(env_two_body or ())
            unknown = tb - {"pauli", "disp", "kappa", "ct"}
            if unknown:
                raise ValueError(f"env_two_body: unknown terms {sorted(unknown)}")
            # env_extra: invariants of the density environment (feature names, e.g. the diffuse probes' l = 0
            # overlaps "acceptor:S0") given to the readouts of env_extra_terms as an additional branch
            names_all = list(feature_names or [])
            missing = [x for x in (env_extra or ()) if x not in names_all]
            if missing:
                raise ValueError(f"env_extra: not among the feature names: {missing}")
            self.register_buffer("env_extra_cols", torch.tensor([names_all.index(x) for x in (env_extra or ())],
                                                                dtype=torch.long), persistent=False)
            self.env_extra_names = list(env_extra or ())
            # env_field: the permanent electrostatic field and field gradient at each site, |E|^2 and |grad E|^2 of
            # the induction drive L (the frozen environment's field over the ion's dipole / quadrupole parents, the
            # same field the polarization sees; Ewald-summed in periodic frames) -- for the readouts of env_extra_terms
            self.env_field = bool(env_field)
            self.register_buffer("field_scale", torch.ones(2, dtype=torch.float64))
            self.register_buffer("field_sat", torch.zeros(2, dtype=torch.float64))
            # env_ct_pen: the diffuse probes' (donor, acceptor, breath) invariants up to l = 2 with penetration-only
            # Coulomb blocks (the neighbours' point charges subtracted from V1, V2 as from V0): S0, V0_pen,
            # |V1_pen|^2, |V2_pen|^2, V1_pen . S1 per probe -- the densities CT populates (frontier / response shapes)
            self.env_ct_pen = bool(env_ct_pen)
            self.register_buffer("pen_scale", torch.ones(len(self.PEN_NAMES), dtype=torch.float64))
            self.register_buffer("pen_sat", torch.zeros(len(self.PEN_NAMES), dtype=torch.float64))
            n_extra = len(self.env_extra_names) + (2 if self.env_field else 0) + (len(self.PEN_NAMES) if self.env_ct_pen else 0)
            et = set(env_extra_terms or ())
            ro = [("pauli", "pauli", "pauli" in tb, "pauli" in et), ("disp", "disp", "disp" in tb, "disp" in et)]
            if self.head_xpol is not None:            # xpol_kappa_head=False: kappa = kappa0 (the ablation)
                ro += [(f"kappa_{c}", "kappa", "kappa" in tb, "kappa" in et) for c in ("dip", "quad", "breath")]
            if self.ct_q_head and not self.ct_legacy_head:   # ct_q_head=False: q = q0 per species (the ablation)
                ro += [("qD", "ct", "ct" in tb, "ct" in et), ("qA", "ct", "ct" in tb, "ct" in et)]
            self.env = SharedEnvironment(len(CHANNEL_NAMES), THREE_BODY_COLS, TWO_BODY_COLS, ro, latent=env_latent,
                                         hidden=hidden, depth=depth, mode=env_mode, n_extra=n_extra,
                                         extra_capacity=env_extra_capacity)
            self.channel_names = list(CHANNEL_NAMES)
            self.register_buffer("chan_scale", torch.ones(len(CHANNEL_NAMES), dtype=torch.float64))
            self.register_buffer("chan_sat", torch.zeros(len(CHANNEL_NAMES), dtype=torch.float64))
        self._env_u = None
        self.pin_a0 = bool(pin_a0)
        if self.pin_a0 and self.breath:                # breathing stiffness = the monomer value (no learned scale)
            self.breath_lna.requires_grad_(False)
        self.double()

    # -- heads --
    def _load_from_state_dict(self, state_dict, prefix, *args, **kw):
        # checkpoints from before feat_sat existed: no saturation (zeros), as they were trained
        state_dict.setdefault(prefix + "feat_sat", torch.zeros_like(self.feat_sat))
        # m33 checkpoints from before the field / penetration branches: their (unused) scales
        for k in ("field_scale", "field_sat", "pen_scale", "pen_sat"):
            if hasattr(self, k):
                state_dict.setdefault(prefix + k, getattr(self, k).clone())
        # which species may be confined is this model's argument (``confine``), not the checkpoint's
        if prefix + "confine_mask" in state_dict:
            state_dict[prefix + "confine_mask"] = self.confine_mask.clone()
        # checkpoints from before the breathing environment head was removed (it was unused with xpol)
        for k in [k for k in state_dict if k.startswith(prefix + "head_breath.")]:
            del state_dict[k]
        super()._load_from_state_dict(state_dict, prefix, *args, **kw)

    def scaled(self, feats):
        I = feats / self.feat_scale * self.feat_keep
        m = self.feat_sat
        if bool((m > 0).any()):
            mm = torch.where(m > 0, m, torch.ones_like(m))
            I = torch.where(m > 0, mm * torch.tanh(I / mm), I)
        return I

    def confinement_from(self, I, sp):
        """(N, 2) scales on alpha0 and C0 (c_range, tanh-saturated like the ion CT model), 1 for an isolated atom."""
        u = self.head_conf(I, sp) * self.confine_mask[sp][:, None]
        if not self.conf_head:
            u = u * 0.0
        if self.conf_form == "sq":
            ssq = self._s0sq if (self._s0sq is not None and self._s0sq.shape[0] == sp.shape[0]) else I.new_zeros(sp.shape[0])
            rate = nn.functional.softplus(self.conf_h[sp]) * self.confine_mask[sp]
            logc = -torch.log1p(rate * ssq / self.conf_ssq)[:, None].expand(-1, 2)
        elif self.conf_form == "mono":
            rate = nn.functional.softplus(self.conf_a[sp][:, None] + u)          # (N, 2) >= 0
            logc = -rate * I[:, self.conf_s0_col:self.conf_s0_col + 1].clamp(min=0) * self.confine_mask[sp][:, None]
        else:
            lo, hi = self.log_c_lo, self.log_c_hi
            logc = torch.where(u > 0, hi * torch.tanh(u / hi), lo * torch.tanh(u / lo))
        if self.tie_c:
            logc = torch.stack([logc[:, 0], logc[:, 0] * 5.0 / 3.0], 1)
        c = logc.exp()
        if self.hyper:
            v = I[:, self.hyper_col]                          # scaled |V[dip]^(1)|^2 (sq) or |V[dip]^(1)| (norm)
            v2 = v if self.hyper_sq else v * v
            enh = 1 + nn.functional.softplus(self.hyper_g[sp]) * v2 * self.confine_mask[sp]
            c = c * torch.stack([enh, enh ** (5.0 / 3.0) if self.tie_c else torch.ones_like(enh)], 1)
        return c

    def capped(self, u):
        """A log environment factor, bounded to +-env_cap (unchanged when env_cap = 0)."""
        return self.env_cap * torch.tanh(u / self.env_cap) if self.env_cap > 0 else u

    def ct_charges_from(self, I, sp):
        if self.env is not None and not self.ct_legacy_head:
            if not self.ct_q_head:
                return torch.exp(self.ct_lnq0[sp])
            eu = self._env_out(sp)
            h = torch.stack([eu["qD"], eu["qA"]], 1)
            c = self.ct_q_cap
            return torch.exp(self.ct_lnq0[sp] + (c * torch.tanh(h / c) if c > 0 else self.capped(h)))
        h = self.head_ctq(I, sp)
        c = self.ct_q_cap
        return torch.exp(self.ct_lnq0[sp] + (c * torch.tanh(h / c) if c > 0 else self.capped(h)))

    def confinement(self, I, h0=None, species=None):
        """Same entry point as the ion CT model (the pair-scan attribution overrides it), on the scaled invariants."""
        return self.confinement_from(I, species)

    def ct_charges(self, h):
        raise NotImplementedError("CompleteModel: use ct_charges_from(I, species)")

    def parent_overlap_sq(self, bt, sp):
        """(N,) sum_j S_ij^2, S_ij the overlap (bohr^-3) of i's dipole parent with j's shells, over the pseudo pairs."""
        n = sp.shape[0]
        idx, r_ang = bt["ps_index"], bt["ps_r"]
        if idx.shape[1] == 0:
            return bt["feats"].new_zeros(n)
        i, j = idx
        rb = r_ang / K.BOHR_ANG
        one = K._onehot0(rb.shape[0], rb)
        tab = self.tab
        S = rb.new_zeros(rb.shape[0])
        for t in range(tab.shell_n.shape[1]):
            nt = tab.shell_n[sp[j], t]
            if bool((nt > 0).any()):
                S = S + nt * K.shaped_overlap(tab.b_dip[sp[i]], tab.w_dip[sp[i]], tab.shell_b[sp[j], t], one, rb)
        return rb.new_zeros(n).index_add(0, i, S * S)

    def pseudo_gamma(self, I, sp):
        """(N,) gamma_i >= 0: the response sensitivity of each atom to its neighbours' exclusion potentials."""
        return torch.exp(self.ps_lng0[sp] + self.capped(self.head_ps(I[:, self.frozen_cols], sp)[:, 0]))

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
        i, j = bt["ch_index"]
        pbc = bt.get("pbc")
        if pbc is not None:
            # per-atom sums over every pair and image (real space, screened) plus the induced moments' reciprocal
            # part; the channel's own partner is removed bare (ch_v / ch_k, that image only)
            xf = x[bt["mask"]]
            phi_x = recip_apply(pbc["frames"], xf @ K._poly_map(xf.dtype, xf.device).T)[:, 0]
            a_, b_ = pbc["dp"]
            n = xf.shape[0]
            phiD = pbc["phiD0"] + phi_x + xf.new_zeros(n).index_add(0, a_, (pbc["kD"] * xf[b_]).sum(-1))
            phiA = pbc["phiA0"] + phi_x + xf.new_zeros(n).index_add(0, a_, (pbc["kA"] * xf[b_]).sum(-1))
            dD = phiD[i] - pbc["ch_vD"] - (pbc["ch_kD"] * xf[j]).sum(-1)
            dA = phiA[j] - pbc["ch_vA"] - (pbc["ch_kA"] * xf[i]).sum(-1)
            return dD - dA + phi_ext[i] - phi_ext[j]
        fr, loc = self._local(bt)
        VD, KD, VA, KA = bt["VD"], bt["KD"], bt["VA"], bt["KA"]
        phiD = VD.sum(-1) + torch.einsum("bikd,bkd->bi", KD, x)
        phiA = VA.sum(-1) + torch.einsum("bikd,bkd->bi", KA, x)
        f, li, lj = fr[i], loc[i], loc[j]
        dD = phiD[f, li] - VD[f, li, lj] - (KD[f, li, lj] * x[f, lj]).sum(-1)
        dA = phiA[f, lj] - VA[f, lj, li] - (KA[f, lj, li] * x[f, li]).sum(-1)
        return dD - dA + phi_ext[i] - phi_ext[j]

    def frontier_potentials(self, bt, x, phi_ext):
        """(n,), (n,): the full potential (Eh/e) of every other atom -- permanent densities, induced moments x
        (B, nmax, 8) -- averaged over each atom's donor / acceptor frontier density, plus ``phi_ext`` (n,)."""
        pbc = bt.get("pbc")
        if pbc is not None:
            xf = x[bt["mask"]]
            phi_x = recip_apply(pbc["frames"], xf @ K._poly_map(xf.dtype, xf.device).T)[:, 0]
            a_, b_ = pbc["dp"]
            n = xf.shape[0]
            phiD = pbc["phiD0"] + phi_x + xf.new_zeros(n).index_add(0, a_, (pbc["kD"] * xf[b_]).sum(-1))
            phiA = pbc["phiA0"] + phi_x + xf.new_zeros(n).index_add(0, a_, (pbc["kA"] * xf[b_]).sum(-1))
            return phiD + phi_ext, phiA + phi_ext
        fr, loc = self._local(bt)
        phiD = bt["VD"].sum(-1) + torch.einsum("bikd,bkd->bi", bt["KD"], x)
        phiA = bt["VA"].sum(-1) + torch.einsum("bikd,bkd->bi", bt["KA"], x)
        return phiD[fr, loc] + phi_ext, phiA[fr, loc] + phi_ext

    def gap_deltas(self, I, sp):
        """(n, 2) >= 0: the overlap shifts (delta_D, delta_A) of each atom's frontier levels, a[s] S exp(h(I))."""
        u = self.capped(self.head_gap(I[:, self.frozen_cols], sp))
        s0 = I[:, self.gap_s0_col].clamp(min=0)[:, None]
        return torch.exp(self.gap_lna[sp] + u) * s0 * self.gap_shift_mask

    def gap_ct(self, bt, sp, I, x, phi_ext, Fa=None):
        """Gap-form CT: per-frame E_CT (B,), per-type sums (B, nT), per-channel energies (C,), and the channel gaps
        D (C,), couplings^2 W2 (C,) and the electrostatic part of the gap phiD(i) - phiA(j) (C,)."""
        B = bt["B"]
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        q0 = torch.exp(self.ct_lnq0[sp])
        W2 = q0[i, 0] * q0[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        phiD, phiA = self.frontier_potentials(bt, x, phi_ext)
        dphi = phiD[i] - phiA[j]
        dl = self.gap_deltas(I, sp)
        D = self.tab.I_donor[sp[i]] - self.acceptor_affinity()[sp[j]] + dphi + dl[j, 1] - dl[i, 0]
        if self.ct_stark:
            n = sp.shape[0]
            env = bt.get("env") or {}
            Fz = Fa if Fa is not None else I.new_zeros(n, 3)
            ED = env.get("donor:V1", I.new_zeros(n, 3)) + Fz
            EA = env.get("acceptor:V1", I.new_zeros(n, 3)) + Fz
            aD = STARK_DALPHA * nn.functional.softplus(self.stark_D[sp[i]])
            aA = STARK_DALPHA * nn.functional.softplus(self.stark_A[sp[j]])
            D = D - 0.5 * aD * (ED[i] * ED[i]).sum(-1) - 0.5 * aA * (EA[j] * EA[j]).sum(-1)
        D = _gap_floor(D)
        if self.ct_sat_ion:
            e_ch = -W2 / D * self.ion_saturation(W2, D, i, j, sp.shape[0])
        else:
            e_ch = 0.5 * (D - torch.sqrt(D * D + 4 * W2))
        e = e_ch.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
        nT = self.nT
        e_type = e_ch.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], e_ch).reshape(B, nT)
        return e, e_type, e_ch, D, W2, dphi

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

    def ct_pair_gap(self, bt, sp):
        """(C,) Delta_c0 = I_D - A_A + phi_D^pair - phi_A^pair (Hartree) of every channel i -> j, floored smoothly."""
        i, j = bt["ch_index"]
        if bt.get("pbc") is not None:
            vD, vA = bt["pbc"]["ch_vD"], bt["pbc"]["ch_vA"]
        else:
            fr, loc = self._local(bt)
            f, li, lj = fr[i], loc[i], loc[j]
            vD, vA = bt["VD"][f, li, lj], bt["VA"][f, lj, li]
        gap = self.tab.I_donor[sp[i]] - self.acceptor_affinity()[sp[j]] + vD - vA
        return _gap_floor(gap)

    def ct_drive(self, w, bt, sp):
        """(B, nmax, 8) induction drive of the channel populations: w_c = g_c p_c as a hole in the donor frontier
        shape of i and an electron in the acceptor shape of j, acting on every atom but the channel's own two
        (dphi excludes the partner), through the frontier operators (potential of l's induced moments over k's
        frontier shape = the drive on l of a unit charge in that shape)."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        f, li, lj = fr[i], loc[i], loc[j]
        KD, KA = bt["KD"], bt["KA"]
        QD = w.new_zeros(B, nmax).index_put((f, li), w, accumulate=True)
        QA = w.new_zeros(B, nmax).index_put((f, lj), w, accumulate=True)
        L = torch.einsum("bk,bkld->bld", QD, KD) - torch.einsum("bk,bkld->bld", QA, KA)
        # remove the partner terms channel by channel
        L = L.index_put((f, lj), -w[:, None] * KD[f, li, lj], accumulate=True)
        L = L.index_put((f, li), w[:, None] * KA[f, lj, li], accumulate=True)
        return L

    def ct_two_state(self, q, bt, sp, x, phi_ext, Fa):
        """Per channel at the induced moments x: E_c (Hartree), population p_c, gap Delta_c, coupling W_c, weight
        g_c p_c for the drive, and dphi_c."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e0 = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])          # the pair kernel, <= 0
        D0 = self.ct_pair_gap(bt, sp)
        W = torch.sqrt((-e0 * D0).clamp(min=1e-300))
        dphi = self.dphi_pairs(bt, x, phi_ext)
        g = torch.exp(self.sc_gD[sp[i]] + self.sc_gA[sp[j]])
        if self.ct_stark:       # the Stark lowering as a factor on the pair gap: bounded, never closes the gap
            D0 = D0 * torch.exp(-self.stark_gap(bt, sp, Fa if Fa is not None else e0.new_zeros(sp.shape[0], 3)))
        D = _gap_floor(D0 + g * dphi)
        root = torch.sqrt(D * D + 4 * W * W)
        e = 0.5 * (D - root)
        p = 0.5 * (1 - D / root)
        return e, p, D, W, g * p, dphi

    @staticmethod
    def ion_saturation(W2, D, i, j, n, k=4.0):
        """(C,) in (0, 1]: each channel's factor on its perturbative energy -W_c^2 / D_c, the power-mean (exponent
        -k, a smooth minimum equal to both when they agree) of the saturation of its donor i and its acceptor j.
        Per ion g: S = sum W^2, P = sum W^2 / D, effective gap S / P, and f = 2 / (1 + sqrt(1 + 4 P^2 / S))."""
        def group(g):
            S = W2.new_zeros(n).index_add(0, g, W2)
            P = W2.new_zeros(n).index_add(0, g, W2 / D)
            return (2 / (1 + torch.sqrt(1 + 4 * P * P / (S + 1e-30))))[g]
        fD, fA = group(i), group(j)
        return ((fD ** -k + fA ** -k) / 2) ** (-1 / k)

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
        if self.ct_sat:
            D0 = self.ct_pair_gap(bt, sp)
            if self.ct_w_fixed:
                q0 = torch.exp(self.ct_lnq0[sp])
                u = u + torch.log(q[i, 0] / q0[i, 0]) + torch.log(q[j, 1] / q0[j, 1])
                e_ch = -q0[i, 0] * q0[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
            D = _gap_floor(D0 * torch.exp(-self.gap_cap * torch.tanh(u / self.gap_cap)))
            if self.ct_sat_ion:
                W2 = (-e_ch * D0).clamp(min=0)
                e_ch = -W2 / D * self.ion_saturation(W2, D, i, j, sp.shape[0])
            else:
                e_ch = 0.5 * (D - torch.sqrt(D * D - 4 * e_ch * D0))                 # -e_ch D0 = W^2 >= 0
        elif self.ct_gap or self.ct_stark:
            e_ch = e_ch * torch.exp(self.gap_cap * torch.tanh(u / self.gap_cap))     # bounded: at most e^+-gap_cap
        e = e_ch.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
        nT = self.nT
        e_type = e_ch.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], e_ch).reshape(B, nT)
        return e, e_type, e_ch, dphi

    def frozen_params(self, I, sp):
        """Per atom: Pauli charge qP, exponent bP, C6, dispersion exponent (each (N,)); with the shells form qP and
        bP are (N, T) per permanent shell (weight 0 where the shell is absent)."""
        if self.env is not None:
            eu = self._env_out(sp)
            u = self.capped(torch.stack([eu["pauli"], eu["disp"]], 1))
        else:
            u = self.capped(self.head_frozen(I[:, self.frozen_cols], sp))
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
        pbc = bt.get("pbc")
        if pbc is not None:                 # every pair and lattice image within the cutoff
            pairs, dr_ang = bt["pairs"], bt["pair_dr"]
        else:
            pairs = K.all_pairs(bt["frame"])
            dr_ang = bt["pos"][pairs[1]] - bt["pos"][pairs[0]]
        i, j = pairs
        r_ang = dr_ang.norm(dim=-1)
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
            dr = dr_ang / K.BOHR_ANG
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
        self._pauli_pairs = (pairs, e_p)
        e_d = tt_damped_c6_energy(r_ang, torch.sqrt(C6[i] * C6[j]), torch.sqrt(b6[i] * b6[j]))
        fr = bt["frame"][i]
        e_disp = I.new_zeros(B)
        if pbc is not None:                 # switched off over the last switch_width; the tail only if asked for
            e_d = e_d * dispersion_switch(r_ang, pbc["cutoff"], pbc["switch_width"])
            if pbc["disp_tail"] is not None:
                sc = I.new_zeros(B).index_add(0, bt["frame"], torch.sqrt(C6))
                e_disp = e_disp - 2 * math.pi / pbc["volume"] * sc * sc * pbc["disp_tail"]
        e_disp = e_disp.index_add(0, fr, e_d)
        return I.new_zeros(B).index_add(0, fr, e_p), e_disp, (qP, bP, C6, b6)

    # -- m33 environment channels --
    def _xpol_tbl(self, device):
        if self._xpol_table is None:
            self._xpol_table = XpolTable(self.tab, r_max=XPOL_R_MAX)
        if self._xpol_table.device != device:
            self._xpol_table.to(device)
        return self._xpol_table

    def env_raw(self, pos_ang, sp, frame, pairs=None, pair_dr=None):
        """(N, 17) raw channels (channels.py) of every atom from its pairs within XPOL_R_MAX: the cluster's all-pairs
        list, or ``pairs`` (2, P) with separations ``pair_dr`` (P, 3) Angstrom (lattice images included)."""
        from .channels import channels_from_pairs, overlap_switch, pair_electrostatics
        if pairs is None:
            pairs = K.all_pairs(frame)
            pair_dr = pos_ang[pairs[1]] - pos_ang[pairs[0]]
        i, j = pairs
        r = pair_dr.norm(dim=-1)
        keep = r < XPOL_R_MAX
        i, j, dr, r = i[keep], j[keep], pair_dr[keep], r[keep]
        a = torch.cat([i, j])                       # directed (a <- b), both ways
        b = torch.cat([j, i])
        drd = torch.cat([dr, -dr])
        rr = torch.cat([r, r])
        tbl = self._xpol_tbl(pos_ang.device)
        y = tbl(sp[a], sp[b], rr)
        O = y[:, tbl.cols(("xp:ovl",), y.device)[0]]
        w = overlap_switch(O) * O.clamp(min=0.0)
        E = pair_electrostatics(sp[a], sp[b], rr / K.BOHR_ANG, self.tab)
        return channels_from_pairs(sp.shape[0], a, drd, w, E * w)

    def env_raw_batch(self, bt):
        pbc = bt.get("pbc")
        if pbc is not None:
            return self.env_raw(bt["pos"], bt["species"], bt["frame"], bt["pairs"], bt["pair_dr"])
        return self.env_raw(bt["pos"], bt["species"], bt["frame"])

    def scaled_channels(self, X):
        I = X / self.chan_scale
        m = self.chan_sat
        if bool((m > 0).any()):
            mm = torch.where(m > 0, m, torch.ones_like(m))
            I = torch.where(m > 0, mm * torch.tanh(I / mm), I)
        return I

    def env_prepare(self, bt, sp):
        """Evaluate the shared environment once for this batch; the parameter maps read it through _env_out."""
        X = self.env_raw_batch(bt)
        self._env_u = self.env(self.scaled_channels(X), self._env_extra(bt["feats"], bt))
        self._env_n = sp.shape[0]
        return X

    @staticmethod
    def field_invariants(L):
        """(N, 2) |E|^2 and |grad E|^2 of the permanent drive L (N, 8): [-E (3), grad E (5 spherical)]."""
        return torch.stack([(L[:, :3] ** 2).sum(1), (L[:, 3:8] ** 2).sum(1)], 1)

    def diffuse_pen_invariants(self, bt):
        """(N, 15) the diffuse probes' invariants with penetration-only Coulomb blocks (``env_ct_pen``), from the
        batch's equivariant environment blocks and the neighbours' point charges (clusters: direct sum)."""
        env = bt.get("env")
        if not env:
            raise ValueError("env_ct_pen needs the equivariant environment blocks in the batch (bt['env'])")
        if bt.get("pbc") is not None:
            raise NotImplementedError("env_ct_pen: clusters only (the point-charge fields are a direct sum)")
        sp, pos, fr = bt["species"], bt["pos"], bt["frame"]
        X = pos / K.BOHR_ANG
        pairs = K.all_pairs(fr)
        i = torch.cat([pairs[0], pairs[1]])
        j = torch.cat([pairs[1], pairs[0]])
        dr = X[j] - X[i]
        T0 = K._tensor(dr, torch.ones(5, dr.shape[0], dtype=X.dtype, device=X.device))[:, 0, :]     # (P, 10)
        Dp = (self.tab.charge[sp[j]][:, None] * T0) @ K._poly_map(X.dtype, X.device)                # (P, 8)
        n = sp.shape[0]
        v1pt = X.new_zeros(n, 3).index_add(0, i, -Dp[:, :3])
        v2pt = X.new_zeros(n, 5).index_add(0, i, Dp[:, 3:])
        cols = []
        for p in self.PEN_PROBES:
            has = (self.tab.a0[sp] > 0).to(X.dtype) if p == "breath" else torch.ones(n, dtype=X.dtype, device=X.device)
            V1 = (env[f"{p}:V1"] - v1pt) * has[:, None]
            V2 = (env[f"{p}:V2"] - v2pt) * has[:, None]
            S1 = env[f"{p}:S1"]
            cols += [env[f"{p}:S0"], env[f"{p}:V0"], (V1 * V1).sum(1), (V2 * V2).sum(1), (V1 * S1).sum(1)]
        return torch.stack(cols, 1)

    def _env_extra(self, feats, bt=None, I=None):
        parts = []
        if self.env_extra_names:
            parts.append((self.scaled(feats) if I is None else I)[:, self.env_extra_cols])
        if self.env_field:
            Fv = self.field_invariants(bt["L"][bt["mask"]]) / self.field_scale
            m = self.field_sat
            if bool((m > 0).any()):
                mm = torch.where(m > 0, m, torch.ones_like(m))
                Fv = torch.where(m > 0, mm * torch.tanh(Fv / mm), Fv)
            parts.append(Fv)
        if self.env_ct_pen:
            Pv = self.diffuse_pen_invariants(bt) / self.pen_scale
            m = self.pen_sat
            if bool((m > 0).any()):
                mm = torch.where(m > 0, m, torch.ones_like(m))
                Pv = torch.where(m > 0, mm * torch.tanh(Pv / mm), Pv)
            parts.append(Pv)
        return torch.cat(parts, 1) if parts else None

    def _env_out(self, sp):
        if self._env_u is None or self._env_n != sp.shape[0]:
            raise RuntimeError("env_channels: the shared environment has not been evaluated for this batch "
                               "(call env_prepare(bt, species) first; forward does)")
        return self._env_u

    # -- quadratic exchange polarization --
    @staticmethod
    def xpol_scalefree(n, i, j, dr, O):
        """(n, 3) scale-free coordination descriptors of every atom from its exchange pairs (overlap O_ij, bond dr):
        with S = sum_j O_ij, V = sum_j O_ij r_ij^, Q = sum_j O_ij (r^ r^T - 1/3),

            n_eff - 1 = S^2 / sum_j O_ij^2 - 1,   eta1 = 1 - |V|^2 / S^2,   eta2 = 1 - (3/2) |Q|_F^2 / S^2

        all exactly 0 for one neighbour at any distance (a pair scan sees a constant kappa), n_eff = number of equal
        neighbours, eta1 = 1 when the overlap pull cancels (a180, sq4, Td, Oh), eta2 = 1 when its rank-2 part does
        (Td, Oh). Squared norms: smooth at the symmetric points. An atom without pairs gets 0."""
        rh = dr / dr.norm(dim=-1, keepdim=True)
        idx = torch.cat([i, j])
        w = torch.cat([O, O])
        r2 = torch.cat([rh, -rh])
        S = O.new_zeros(n).index_add(0, idx, w)
        S2 = O.new_zeros(n).index_add(0, idx, w * w)
        V = O.new_zeros(n, 3).index_add(0, idx, w[:, None] * r2)
        Qp = r2[:, :, None] * r2[:, None, :] - torch.eye(3, dtype=O.dtype, device=O.device) / 3
        Q = O.new_zeros(n, 3, 3).index_add(0, idx, w[:, None, None] * Qp)
        ok = S2 > 1e-280
        Ss, S2s = torch.where(ok, S, torch.ones_like(S)), torch.where(ok, S2, torch.ones_like(S2))
        ne = Ss * Ss / S2s - 1
        e1 = 1 - (V * V).sum(1) / (Ss * Ss)
        e2 = 1 - 1.5 * (Q * Q).sum((1, 2)) / (Ss * Ss)
        out = torch.stack([ne, e1, e2], 1)
        return torch.where(ok[:, None], out, torch.zeros_like(out))

    def xpol_kappa(self, I, sp, pairs=None):
        """(N, D) the per-component scale of each atom's overlap amplitude: kappa0[s] exp(head(S_i)) on the dipole (3),
        quadrupole (5) and, with breathing, the l = 0 amplitude (zero for species without one)."""
        if self.env is not None:
            if self.head_xpol is None:
                k = torch.exp(self.xpol_lnk[sp])
            else:
                eu = self._env_out(sp)
                k = torch.exp(self.xpol_lnk[sp] + self.capped(torch.stack([eu["kappa_dip"], eu["kappa_quad"],
                                                                           eu["kappa_breath"]], 1)))
            parts = [k[:, :1].expand(-1, 3), k[:, 1:2].expand(-1, 5)]
            if self.breath:
                parts.append((k[:, 2] * self.xpol_breath_mask[sp])[:, None])
            return torch.cat(parts, 1)
        if self.head_xpol is None:
            h = I.new_zeros(I.shape[0])
        elif self.xpol_kappa_env == "scalefree":
            if pairs is None:
                raise ValueError("xpol_kappa_env='scalefree' needs the exchange pairs (i, j, dr, O)")
            h = self.capped(self.head_xpol(self.xpol_scalefree(I.shape[0], *pairs), sp)[:, 0])
        else:
            h = self.capped(self.head_xpol(I[:, self.frozen_cols], sp)[:, 0])
        k = torch.exp(self.xpol_lnk[sp] + h[:, None])                                   # (N, 3)
        parts = [k[:, :1].expand(-1, 3), k[:, 1:2].expand(-1, 5)]
        if self.breath:
            parts.append((k[:, 2] * self.xpol_breath_mask[sp])[:, None])
        return torch.cat(parts, 1)

    def xpol_pairs(self, bt, sp, I):
        """Undirected pairs (i, j) within XPOL_R_MAX: their frozen Pauli energies E0 (P,) (from ``frozen_terms``,
        which must have run), lam (P,) and the vectors v_i, v_j (P, D) with u_ij = v_i . z_i + v_j . z_j."""
        if self._xpol_table is None:
            self._xpol_table = XpolTable(self.tab, r_max=XPOL_R_MAX)
        tbl = self._xpol_table
        if tbl.device != bt["pos"].device:
            tbl.to(bt["pos"].device)
        (i, j), e0 = self._pauli_pairs
        if bt.get("pbc") is not None:       # every pair and image of the Pauli pair list
            dr = bt["pair_dr"] / K.BOHR_ANG
        else:
            dr = (bt["pos"][j] - bt["pos"][i]) / K.BOHR_ANG
        keep = dr.norm(dim=-1) * K.BOHR_ANG < XPOL_R_MAX
        i, j, dr, e0 = i[keep], j[keep], dr[keep], e0[keep]
        P = i.shape[0]
        dp = torch.stack([torch.cat([i, j]), torch.cat([j, i])])
        O, d, e = xpol_overlap_map(tbl, sp, dp, torch.cat([dr, -dr]))
        # the ratios are taken only where the overlap is above the quadrature noise of the table: switched on
        # smoothly between O = XPOL_O_ON and XPOL_O_FULL (a negative O -- noise -- counts as 0). The noise floor is
        # ~1e-8 for Na+-Na+ (O stops decreasing at 3.25 A, O = 2e-7, and turns negative at 3.65 A), so a lower window
        # takes d/O of noise: E_pol jumped ~1 kJ/mol over 0.0005 A at the B1 second-neighbour Na-Na distance (3.8 A).
        lo, hi = math.log10(XPOL_O_ON), math.log10(XPOL_O_FULL)
        t = ((torch.log10(O.clamp(min=1e-300)) - lo) / (hi - lo)).clamp(0.0, 1.0)
        sw = t * t * (3 - 2 * t)
        rel = torch.cat([d, e[:, None]], 1) * (sw / O.clamp(min=XPOL_O_ON))[:, None]   # (2P, 9): d ln O_ab per unit of b's z
        # the scale-free descriptors see each pair's overlap through the same switch (no quadrature noise)
        kap = self.xpol_kappa(I, sp, (i, j, dr, sw[:P] * O[:P].clamp(min=0.0)))
        self._xpol_O = O[:P]                          # O_ij of the kept pairs (the sqrt_overlap perpendicular weight)
        D = kap.shape[1]
        vj = kap[j] * rel[:P, :D]                     # (i <- j): j's deformation changes O_ij
        vi = kap[i] * rel[P:, :D]                     # (j <- i): i's deformation changes O_ji
        sl = nn.functional.softplus(self.xpol_lam)
        lam = 1 + 0.5 * (sl[sp[i]] + sl[sp[j]])
        return i, j, e0, lam, vi, vj

    def xpol_perp_weights(self, sp, i, j, e0):
        """(P,) w_perp = 2 W_ij k_perp,ij of each exchange pair, W = E0_ij or (sqrt_overlap) E_s sqrt(O_ij / O_s) (None without ``xpol_perp``): the on-site Hessian of the
        perpendicular stiffening on side i is w_perp (|v_i|^2 1 - v_i v_i^T) within each component block."""
        if not self.xpol_perp:
            return None
        kp = nn.functional.softplus(self.xpol_kperp)
        if self.xpol_perp_weight == "sqrt_overlap":
            W = XPOL_PERP_E_S * torch.sqrt(self._xpol_O.clamp(min=0.0) / XPOL_PERP_O_S + 1e-30)
        else:
            W = e0
        return 2 * W * 0.5 * (kp[sp[i]] + kp[sp[j]])

    @staticmethod
    def xpol_perp_blocks(v):
        """(P, D, D) the per-pair on-site matrix sum_blocks (|v_b|^2 1_b - v_b v_b^T) of the perpendicular stiffening."""
        P_, D = v.shape
        M = v.new_zeros(P_, D, D)
        for a, b in XPOL_PERP_BLOCKS:
            if b > D:
                continue
            vb = v[:, a:b]
            n2 = (vb * vb).sum(1)
            eye = torch.eye(b - a, dtype=v.dtype, device=v.device)
            M[:, a:b, a:b] = n2[:, None, None] * eye - vb[:, :, None] * vb[:, None, :]
        return M

    def xpol_solve(self, bt, sp, I, L, A_off, ainv_pad, Cq, quadrupoles=True, rhs_extra=None):
        """The minimum of 1/2 z A z + L z + E_P(z) - E_P(0) over z = (x, b) for clusters: one dense solve.
        Returns z (B, nmax, D), the electrical part 1/2 z A z + L z (B,), the exchange part g z + 1/2 z H z (B,) and
        the breathing atoms' own energy 1/2 b^2 / a - V0 b (n,). ``rhs_extra`` (B, nmax D, k): more right-hand sides
        of the same operator, solved with it; then (A + H)^-1 rhs_extra (B, nmax D, k) is returned fifth."""
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        fr, loc = self._local(bt)
        A = induction_matrix_blocks(A_off, ainv_pad, Cq, quadrupoles).reshape(B, nmax, 8, nmax, 8)
        D = 9 if self.breath else 8
        ar = torch.arange(nmax, device=A.device)
        M = A.new_zeros(B, nmax, D, nmax, D)
        M[:, :, :8, :, :8] = A
        Lz = L.new_zeros(B, nmax, D)
        Lz[..., :8] = L
        if D == 9:
            a_b = self.tab.a0[sp] * torch.exp(self.breath_lna[sp])
            has = a_b > 0
            ainv_b = torch.where(has, 1.0 / a_b.clamp(min=1e-30), torch.ones_like(a_b))
            V0 = bt["feats"][:, self.breath_col] * has.to(a_b.dtype)
            diag = torch.ones(B, nmax, dtype=A.dtype, device=A.device).index_put((fr, loc), ainv_b)
            M[:, ar, 8, ar, 8] = diag
            Lz = Lz.index_put((fr, loc, torch.full_like(fr, 8)), -V0, accumulate=True)
        i, j, e0, lam, vi, vj = self.xpol_pairs(bt, sp, I)
        if not quadrupoles:
            vi, vj = vi.clone(), vj.clone()
            vi[:, 3:8], vj[:, 3:8] = 0.0, 0.0
        g = L.new_zeros(n, D).index_add(0, i, 2 * e0[:, None] * vi).index_add(0, j, 2 * e0[:, None] * vj)
        gz = L.new_zeros(B, nmax, D).index_put((fr, loc), g)
        w = (2 * lam * e0)[:, None, None]
        f, li, lj = fr[i][:, None, None], loc[i][:, None, None], loc[j][:, None, None]
        a_ = torch.arange(D, device=A.device)[None, :, None]
        b_ = torch.arange(D, device=A.device)[None, None, :]
        H = A.new_zeros(B, nmax, D, nmax, D)
        H = H.index_put((f, li, a_, li, b_), w * vi[:, :, None] * vi[:, None, :], accumulate=True)
        H = H.index_put((f, lj, a_, lj, b_), w * vj[:, :, None] * vj[:, None, :], accumulate=True)
        H = H.index_put((f, li, a_, lj, b_), w * vi[:, :, None] * vj[:, None, :], accumulate=True)
        H = H.index_put((f, lj, a_, li, b_), w * vj[:, :, None] * vi[:, None, :], accumulate=True)
        wp = self.xpol_perp_weights(sp, i, j, e0)
        if wp is not None:              # E0 k_perp (|v_i|^2 |z_i|^2 - (v_i.z_i)^2) per block, both sides: on-site only
            H = H.index_put((f, li, a_, li, b_), wp[:, None, None] * self.xpol_perp_blocks(vi), accumulate=True)
            H = H.index_put((f, lj, a_, lj, b_), wp[:, None, None] * self.xpol_perp_blocks(vj), accumulate=True)
        N = nmax * D
        Mf, Hf = M.reshape(B, N, N), H.reshape(B, N, N)
        rhs = (Lz + gz).reshape(B, N, 1)
        if rhs_extra is not None:
            rhs = torch.cat([rhs, rhs_extra], 2)
        Y = torch.linalg.solve(Mf + Hf, rhs)
        z = -Y[..., 0]
        Lf = Lz.reshape(B, N)
        e_el = 0.5 * (z * (Mf @ z[..., None])[..., 0]).sum(1) + (Lf * z).sum(1)
        e_x = (gz.reshape(B, N) * z).sum(1) + 0.5 * (z * (Hf @ z[..., None])[..., 0]).sum(1)
        z = z.reshape(B, nmax, D)
        e_b = L.new_zeros(n)
        if D == 9:
            bz = z[fr, loc, 8]
            e_b = torch.where(has, 0.5 * bz * bz * ainv_b - V0 * bz, torch.zeros_like(bz))
        if rhs_extra is not None:
            return z, e_el, e_x, e_b, Y[..., 1:]
        return z, e_el, e_x, e_b

    # -- charge-flux CT --
    def flux_operator(self, bt, sp, q, D, quadrupoles=True):
        """The charge block of the flux CT for clusters, in the form ``flux_solver`` selects: a tuple
        (C, H, s, V, G, ch, e_c) whose first element is the extra right-hand sides of the xpol solve and which
        :meth:`flux_solve` takes back. See :meth:`flux_operator_atomic` and :meth:`flux_operator_channel`."""
        if self.flux_solver == "channel":
            return self.flux_operator_channel(bt, sp, q, D, quadrupoles)
        return self.flux_operator_atomic(bt, sp, q, D, quadrupoles)

    def flux_solve(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):
        """The joint minimum, given the xpol solution z_pol (B, nmax D) and W = (A + H)^-1 C for the operator of
        :meth:`flux_operator`: per frame E_CT (B,), per channel its share 1/2 b~_c q_c (C,) and charge q_c (C,), the
        joint moments z (B, nmax D) and the atomic charge changes (B, nmax)."""
        if self.flux_solver == "channel":
            return self.flux_solve_channel(bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext)
        return self.flux_solve_atomic(bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext)

    def _flux_atoms(self, bt, sp, D, quadrupoles=True):
        """Atomic charge slots: C (B, nmax, D, nmax) the coupling of a unit charge at k to i's site variables
        ([l, a, k]: energy Q_k C[l, a, k] z_l,a) and H (B, nmax, nmax) the slots' Coulomb with the on-site hardness
        (shape self-energy + eta) on the diagonal; padded atoms have zero rows."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        if self._flux_table is None:
            self._flux_table = FluxTable(self.tab)
        tbl = self._flux_table
        if tbl.device != bt["pos"].device:
            tbl.to(bt["pos"].device)
        X = bt["pos"] / K.BOHR_ANG
        pairs = K.all_pairs(bt["frame"])
        dp = torch.cat([pairs, pairs.flip(0)], 1)
        a_, b_ = dp
        Kp, Jp = flux_pair_terms(tbl, sp, dp, X[b_] - X[a_])
        Kd = X.new_zeros(B, nmax, nmax, 8).index_put((fr[a_], loc[a_], loc[b_]), Kp)    # [l, k]: on l of a charge at k
        eta = self.flux_jself[sp]
        if self.ct_model == "flux":           # twostate: the full Coulomb matrix (self-energies only on the diagonal)
            eta = eta + nn.functional.softplus(self.flux_eta[sp])
        Jd = X.new_zeros(B, nmax, nmax).index_put((fr[a_], loc[a_], loc[b_]), Jp).index_put((fr, loc, loc), eta)
        Ca = Kd.permute(0, 1, 3, 2)                                                     # (B, nmax, 8, nmax)
        if not quadrupoles:
            Ca = torch.cat([Ca[:, :, :3], torch.zeros_like(Ca[:, :, 3:])], 2)
        if D > 8:
            Ca = torch.cat([Ca, Ca.new_zeros(B, nmax, D - 8, nmax)], 2)
        return Ca, Jd

    def ct_kernel_energy(self, bt, sp, q):
        """(C,) the direct-CT energy of every channel, e_c = -qD_i qA_j f(bD, bA; r) / r times the taper (<= 0)."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        return -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])

    def slot_potentials(self, bt, sp):
        """(n,) the permanent potential (Hartree / e) of every other atom's frozen density averaged over each atom's
        charge slot (its dipole-parent shape): the penetration part (environment block dip:V0) plus the point charges
        (clusters)."""
        if bt.get("pbc") is not None:
            raise NotImplementedError("slot potentials: clusters only (the point-charge part needs the Ewald sum)")
        env = bt.get("env")
        if env is None or "dip:V0" not in env:
            raise ValueError("ct_model='twostate' needs the dip probe's environment blocks (bt['env']['dip:V0'])")
        X = bt["pos"] / K.BOHR_ANG
        i, j = K.all_pairs(bt["frame"])
        r = (X[j] - X[i]).norm(dim=-1)
        qn = self.tab.charge[sp].to(X.dtype)
        phi = X.new_zeros(sp.shape[0]).index_add(0, i, qn[j] / r).index_add(0, j, qn[i] / r)
        return env["dip:V0"] + phi

    def twostate_operator(self, bt, sp, q, D, quadrupoles=True):
        """The two-state CT block: C (B, nmax D, nmax) the slot charges' coupling to the site variables (the extra
        right-hand sides of the xpol solve), H (B, nmax, nmax) the slots' full Coulomb matrix, the channels' (frame,
        donor slot, acceptor slot), their direct energies e_c (C,) and floored gaps Delta_c (C,) (twostate.py)."""
        from . import twostate as TS
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        Ca, Hd = self._flux_atoms(bt, sp, D, quadrupoles)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        e_c = self.ct_kernel_energy(bt, sp, q)
        phi = self.slot_potentials(bt, sp)
        Delta = ((self.ts_I0 + self.ts_dI)[sp[i]] - (self.ts_A0 + self.ts_dA)[sp[j]] + phi[i] - phi[j]
                 - Hd[cf, li, lj])
        return Ca.reshape(B, nmax * D, nmax), Hd, (cf, li, lj), e_c, TS.floor_gap(Delta, self.ct_gap_min), Delta

    def _flux_drives(self, bt, sp, q, h):
        """Per channel, given h_c = (G^T H G)_cc (C,): the kernel energy e_c <= 0, the drive V_c (with the pair
        factor t_p) and the variable scale s_c = sqrt(2 |e_c|) / V0_c = 1 / sqrt(b_c)."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_c = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])          # <= 0
        ae = (-e_c).clamp(min=0.0)
        v0 = torch.exp(self.flux_lnv)
        V0 = v0[sp[i]] + v0[sp[j]]
        # the pair's two directions c = i -> j and c' = j -> i see the same h and are coupled by -h (opposite charge
        # patterns): their drives V_c = t_p sqrt(V0_c^2 + 2 |e_c| h) with the one factor t_p that makes the pair's own
        # 2x2 minimum exactly e_c + e_c'. In the compliances c = 1 / (b + h):
        #     t^2 = S (1 - h^2 c c') / (S + 2 h sqrt(|e| c |e'| c')),   S = |e| + |e'|
        Vh2 = V0 * V0 + 2 * ae * h
        cc = 2 * ae / Vh2
        rev = self.flux_reverse(i, j, sp.shape[0])
        has = rev >= 0
        rv = rev.clamp(min=0)
        ae2 = torch.where(has, ae[rv], torch.zeros_like(ae))
        cc2 = torch.where(has, cc[rv], torch.zeros_like(cc))
        S_ = ae + ae2
        t2 = S_ * (1 - h * h * cc * cc2) / (S_ + 2 * h * torch.sqrt(ae * cc * ae2 * cc2 + 1e-300) + 1e-300)
        t2 = torch.where(has, t2, torch.ones_like(t2))
        Vc = torch.sqrt(t2 * Vh2)
        sc = torch.sqrt(2 * ae + 1e-30) / V0
        return e_c, Vc, sc

    def flux_operator_atomic(self, bt, sp, q, D, quadrupoles=True):
        """The flux block in atomic-charge space. Returns C (B, nmax D, nmax) the atomic charges' coupling to the
        induced moments, H (B, nmax, nmax), s (C,) and V (C,) per channel, None (no charge-pattern matrix), the
        channels' (frame, donor slot, acceptor slot) and their kernel energies e_c (C,)."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        Ca, Jd = self._flux_atoms(bt, sp, D, quadrupoles)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        h = Jd[cf, li, li] + Jd[cf, lj, lj] - Jd[cf, li, lj] - Jd[cf, lj, li]     # (G^T H G)_cc by gather
        e_c, Vc, sc = self._flux_drives(bt, sp, q, h)
        return Ca.reshape(B, nmax * D, nmax), Jd, sc, Vc, None, (cf, li, lj), e_c

    def flux_solve_atomic(self, bt, Ca, Ha, s, V, _G, ch, z_pol, W, phi_ext):
        """The joint minimum with the channels eliminated (module docstring): K = G diag(s^2) G^T, u = G s^2 V,
        (1 + K S) Q = u - K psi0 with S = H - C^T W, psi0 = phi_ext + C^T z_pol; q = s^2 (V - G^T psi)."""
        cf, li, lj = ch
        B, N = Ha.shape[0], Ha.shape[1]
        fr, loc = self._local(bt)
        w = s * s                                                      # compliance 1 / b_c
        Kl = Ha.new_zeros(B, N, N)
        Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
        Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
        wV = w * V
        u = Ha.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
        psi0 = Ha.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Ha - Ca.transpose(1, 2) @ W                                # (B, N, N)
        M = torch.eye(N, dtype=Ha.dtype, device=Ha.device) + Kl @ S
        Q = torch.linalg.solve(M, (u - (Kl @ psi0[..., None])[..., 0])[..., None])[..., 0]
        psi = psi0 + (S @ Q[..., None])[..., 0]
        qc = w * (V - (psi[cf, li] - psi[cf, lj]))
        e_ch = 0.5 * (-V + psi0[cf, li] - psi0[cf, lj]) * qc          # 1/2 b~_c q_c
        z = z_pol - (W @ Q[..., None])[..., 0]
        return Ha.new_zeros(B).index_add(0, cf, e_ch), e_ch, qc, z, Q

    def flux_operator_channel(self, bt, sp, q, D, quadrupoles=True):
        """The charge block of the flux CT in channel space (reference form, ``flux_solver="channel"``). Returns C
        (B, nmax D, Cmax) the charges' coupling to the induced moments (energy q^T C^T z), Hq = G^T H G (B, Cmax,
        Cmax), s (B, Cmax) the variable scales sqrt(2 |e_c|) / V0_c, V (B, Cmax) the drives, G (B, nmax, Cmax) the
        charge pattern, the channels' (frame, slot) and their kernel energies e_c (C,); padded channel slots have
        s = 0, V = 0."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        Ca, Jd = self._flux_atoms(bt, sp, D, quadrupoles)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        nC = i.shape[0]
        cnt = torch.bincount(cf, minlength=B)
        Cmax = int(cnt.max()) if nC else 0
        order = torch.argsort(cf, stable=True)
        slot = torch.empty_like(cf)
        slot[order] = torch.arange(nC, device=cf.device) - (torch.cumsum(cnt, 0) - cnt)[cf[order]]
        one = Jd.new_ones(nC)
        G = Jd.new_zeros(B, nmax, Cmax).index_put((cf, loc[i], slot), one, accumulate=True)
        G = G.index_put((cf, loc[j], slot), -one, accumulate=True)
        Hq = torch.einsum("bkc,bkl,bld->bcd", G, Jd, G)
        Cq = torch.einsum("blak,bkc->blac", Ca, G)                         # (B, nmax, D, Cmax)
        e_c, Vc, sc = self._flux_drives(bt, sp, q, Hq[cf, slot, slot])
        s_pad = Jd.new_zeros(B, Cmax).index_put((cf, slot), sc)
        V_pad = Jd.new_zeros(B, Cmax).index_put((cf, slot), Vc)
        return Cq.reshape(B, nmax * D, Cmax), Hq, s_pad, V_pad, G, (cf, slot), e_c

    @staticmethod
    def flux_reverse(i, j, n):
        """(C,) the index of each channel's reverse channel j -> i, -1 where there is none."""
        key, rkey = i * n + j, j * n + i
        srt, order = torch.sort(key)
        pos = torch.searchsorted(srt, rkey).clamp(max=max(key.shape[0] - 1, 0))
        hit = srt[pos] == rkey if key.shape[0] else torch.zeros_like(key, dtype=torch.bool)
        return torch.where(hit, order[pos], torch.full_like(key, -1))

    def flux_solve_channel(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):
        """Reference form (``flux_solver="channel"``): the joint minimum given the xpol solution z_pol (B, nmax D) and W = (A + H)^-1 C: per frame E_CT (B,),
        per channel its share 1/2 b~_c q_c (C,) and charge q_c (C,), the joint moments z (B, nmax D) and the
        atomic charge changes (B, nmax)."""
        cf, slot = ch
        B, Cmax = s.shape
        i, j = bt["ch_index"]
        dphi = phi_ext[i] - phi_ext[j]
        bq = -V + V.new_zeros(B, Cmax).index_put((cf, slot), dphi)
        bt_ = bq + torch.einsum("bnc,bn->bc", Cz, z_pol)
        Sm = torch.eye(Cmax, dtype=s.dtype, device=s.device) + s[:, :, None] * (Hq - Cz.transpose(1, 2) @ W) * s[:, None, :]
        bp = s * bt_
        p = -torch.linalg.solve(Sm, bp[..., None])[..., 0]
        qc = s * p
        e_slot = 0.5 * bp * p
        z = z_pol - (W @ qc[..., None])[..., 0]
        Q = (G @ qc[..., None])[..., 0]
        return e_slot.sum(1), e_slot[cf, slot], qc[cf, slot], z, Q

    def xpol_solve_periodic(self, bt, sp, I, Lf, ainv, Cs, quadrupoles=True, info=None):
        """The periodic counterpart of :meth:`xpol_solve` (matrix-free CG with the Ewald frames, the exchange pairs
        over every image of the Pauli pair list). Lf (n, 8) the flat drive, ainv (n, 3, 3), Cs (n,). Returns z (n, D),
        the electrical part (B,), the exchange part (B,) and the breathing atoms' own energy (n,)."""
        pbc, fr = bt["pbc"], bt["frame"]
        B, n = bt["B"], sp.shape[0]
        i, j, e0, lam, vi, vj = self.xpol_pairs(bt, sp, I)
        if not quadrupoles:
            vi, vj = vi.clone(), vj.clone()
            vi[:, 3:8], vj[:, 3:8] = 0.0, 0.0
        D = vi.shape[1]
        Lz, ainv_b, V0 = Lf, Lf.new_zeros(n), None
        if D == 9:
            a_b = self.tab.a0[sp] * torch.exp(self.breath_lna[sp])
            has = a_b > 0
            ainv_b = torch.where(has, 1.0 / a_b.clamp(min=1e-30), torch.ones_like(a_b))
            V0 = bt["feats"][:, self.breath_col] * has.to(a_b.dtype)
            Lz = torch.cat([Lf, -V0[:, None]], 1)
        g = Lf.new_zeros(n, D).index_add(0, i, 2 * e0[:, None] * vi).index_add(0, j, 2 * e0[:, None] * vj)
        w = 2 * lam * e0
        wp = self.xpol_perp_weights(sp, i, j, e0)
        hp = (i, j, w, vi, vj) if wp is None else (i, j, w, vi, vj, wp)
        xg = bt.get("x_guess")
        if xg is not None and xg.shape[0] == n:
            xg = torch.cat([xg, xg.new_zeros(n, D - xg.shape[1])], 1) if xg.shape[1] < D else xg[:, :D]
        else:
            xg = None
        z = solve_xpol_periodic(Lz + g, ainv, Cs, ainv_b, pbc["A_pairs"], pbc["A_blocks"], pbc["frames"],
                                hp, quadrupoles, x0=xg, info=info)
        P = K._poly_map(Lf.dtype, Lf.device)
        cinv = torch.where(Cs > 0, 1.0 / Cs.clamp(min=1e-30), torch.full_like(Cs, 1e12))
        dof = Lf.new_ones(8)
        if not quadrupoles:
            dof[3:] = 0
        Az = _matvec(z[:, :8], ainv, cinv, pbc["A_pairs"], pbc["A_blocks"], pbc["frames"], P, dof)
        e_el = Lf.new_zeros(B).index_add(0, fr, 0.5 * (z[:, :8] * Az).sum(1) + (Lf * z[:, :8]).sum(1))
        e_b = Lf.new_zeros(n)
        if D == 9:
            bz = z[:, 8]
            e_b = torch.where(has, 0.5 * bz * bz * ainv_b - V0 * bz, torch.zeros_like(bz))
            e_el = e_el + Lf.new_zeros(B).index_add(0, fr, e_b)
        u = (vi * z[i]).sum(1) + (vj * z[j]).sum(1)
        e_x = Lf.new_zeros(B).index_add(0, fr, (g * z).sum(1)).index_add(0, fr[i], 0.5 * w * u * u)
        if wp is not None:
            e_x = e_x.index_add(0, fr[i], 0.5 * wp * (_perp_quad(vi, z[i]) + _perp_quad(vj, z[j])))
        return z, e_el, e_x, e_b

    # -- the model --
    def forward(self, bt: dict, quadrupoles=True, field=None, solvent=None) -> dict:
        """``solvent``: a :class:`rsfff.ff.complete.solvent.SolventTerms` for this batch (C-PCM; None = gas phase):
        the induction solve is screened by the equilibrated surface charges, the CT gap sees the reaction potential,
        and the frozen electrostatics gain the screened cross term (in ``e_elst``) and the desolvation ``e_cav``."""
        sp = bt["species"]
        I = self.scaled(bt["feats"])
        if self.env is not None:
            self.env_prepare(bt, sp)
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        self._s0sq = self.parent_overlap_sq(bt, sp) if self.conf_form == "sq" else None
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
        if solvent is not None:
            L = L + solvent.dL
        ainv = self.alpha_inverse(alpha_flat, sp, bt.get("env"))
        ainv_pad = torch.eye(3, dtype=I.dtype, device=I.device).expand(B, nmax, 3, 3).clone()
        ainv_pad = ainv_pad.masked_scatter(mask[..., None, None].expand(B, nmax, 3, 3), ainv)
        pbc = bt.get("pbc")
        cg = {}
        e_pauli, e_disp, frz = I.new_zeros(B), I.new_zeros(B), None
        if self.frozen:
            e_pauli, e_disp, frz = self.frozen_terms(I, bt, sp)
        e_xpol, z_x = I.new_zeros(B), None
        nC = bt["ch_index"].shape[1]
        flux = None
        if self.ct_model in ("flux", "twostate") and (pbc is not None or solvent is not None):
            raise NotImplementedError(f"ct_model={self.ct_model!r}: clusters in the gas phase only (no periodic frames, no solvent)")
        if self.xpol and pbc is not None:
            if solvent is not None:
                raise NotImplementedError("periodic frames: no solvent")
            zf, e_pol, e_xpol, e_breath_atom = self.xpol_solve_periodic(bt, sp, I, L[mask], ainv,
                                                                        self.tab.C[sp] * c[:, 1], quadrupoles, cg)
            z_x = I.new_zeros(B, nmax, zf.shape[1]).masked_scatter(mask[..., None].expand(B, nmax, zf.shape[1]), zf)
            x0 = z_x[..., :8]
        elif self.xpol:
            A_off = bt["A_off"] if solvent is None else bt["A_off"] + solvent.dA
            if self.ct_model in ("flux", "twostate") and nC:
                op = self.twostate_operator if self.ct_model == "twostate" else self.flux_operator
                flux = op(bt, sp, self.ct_charges_from(I, sp), 9 if self.breath else 8, quadrupoles)
                z_x, e_pol, e_xpol, e_breath_atom, W_flux = self.xpol_solve(bt, sp, I, L, A_off, ainv_pad, Cq,
                                                                            quadrupoles, rhs_extra=flux[0])
            else:
                z_x, e_pol, e_xpol, e_breath_atom = self.xpol_solve(bt, sp, I, L, A_off, ainv_pad, Cq, quadrupoles)
            x0 = z_x[..., :8]
        elif pbc is not None:
            # matrix-free CG on the real-space pair blocks + the Ewald frames (periodic.py)
            if solvent is not None or self.ct_sc:
                raise NotImplementedError("periodic frames: no solvent, no self-consistent CT")
            Bij = pbc["A_blocks"]
            if self.pauli_ii:
                a_, b_ = pbc["A_pairs"]
                Bij = Bij + (self.pp_kappa[sp[a_]] * self.pp_kappa[sp[b_]])[:, None, None] * pbc["App_blocks"]
            xg = bt.get("x_guess")
            xf = solve_induction_periodic(L[mask], ainv, self.tab.C[sp] * c[:, 1], pbc["A_pairs"], Bij, pbc["frames"],
                                          quadrupoles, x0=xg if xg is not None and xg.shape[0] == n else None, info=cg)
            x0 = I.new_zeros(B, nmax, 8).masked_scatter(mask[..., None].expand(B, nmax, 8), xf)
        else:
            A_off = bt["A_off"]
            if solvent is not None:
                A_off = A_off + solvent.dA
            if self.pauli_ii:
                kk = torch.zeros(B, nmax, dtype=I.dtype, device=I.device).masked_scatter(mask, self.pp_kappa[sp])
                kk = kk.repeat_interleave(8, dim=1)
                A_off = A_off + bt["A_pp"] * kk[:, :, None] * kk[:, None, :]
            x0 = solve_induction_blocks(L, A_off, ainv_pad, Cq, quadrupoles)
        if self.xpol:
            e_pol = e_pol + e_xpol         # 1/2 z A z + L z (breathing included) + g z + 1/2 z H z
        else:
            e_pol = 0.5 * (L * x0).sum((1, 2))
            e_breath_atom = I.new_zeros(n)
        if self.breath and not self.xpol:
            v0 = bt["feats"][:, self.breath_col] + self.breath_lam[sp] * self.breath_s0_scale * bt["feats"][:, self.breath_s0_col]
            a = self.tab.a0[sp] * torch.exp(self.breath_lna[sp])
            e_breath_atom = -0.5 * a * v0 * v0
            e_pol = e_pol + I.new_zeros(B).index_add(0, bt["frame"], e_breath_atom)
        q = self.ct_charges_from(I, sp)
        x_sc, dq, e_ct_post, sc_iters, Dc, Wc = x0, I.new_zeros(nC), None, 0, I.new_zeros(nC), I.new_zeros(nC)
        phi_rf = (lambda x: solvent.phi_rf(bt, x)) if solvent is not None else (lambda x: 0.0)
        Q_flux = I.new_zeros(n)
        if nC and self.ct_model == "twostate":
            from . import twostate as TS
            Cz, Hd, ch_, e_kern, Dl, Dl_raw = flux
            self._ts_info = {}
            e_ct, e_ch, dq, zj, Qd = TS.solve(Cz, Hd, W_flux, z_x.reshape(B, -1),
                                             I.new_zeros(B, nmax).index_put(self._local(bt), phi_ext), ch_, e_kern,
                                             Dl, n_outer=self.ct_outer, tol=self.ct_outer_tol, info=self._ts_info)
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, bt["ch_frame"] * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
            Wc, Dc, dphi = e_kern, Dl, I.new_zeros(nC)
            x_sc = zj.reshape(B, nmax, -1)[..., :8]
            fr_, loc_ = self._local(bt)
            Q_flux = Qd[fr_, loc_]
        elif nC and self.ct_model == "flux":
            Cz, Hq, s_, V_, G_, ch_, e_kern = flux
            e_ct, e_ch, dq, zj, Qd = self.flux_solve(bt, Cz, Hq, s_, V_, G_, ch_, z_x.reshape(B, -1), W_flux, phi_ext)
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, bt["ch_frame"] * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
            Wc, dphi = e_kern, I.new_zeros(nC)          # "W": the channels' kernel energies e_c
            x_sc = zj.reshape(B, nmax, -1)[..., :8]
            fr_, loc_ = self._local(bt)
            Q_flux = Qd[fr_, loc_]
        elif nC and self.ct_model == "gap":
            e_ct, e_type, e_ch, Dc, Wc, dphi = self.gap_ct(bt, sp, I, x0, phi_ext + phi_rf(x0), Fa)
        elif nC:
            e_ct, e_type, e_ch, dphi = self.direct_ct_env(q, bt, sp, x0, phi_ext + phi_rf(x0), Fa)
            if self.ct_sc:
                x = x0
                e_ch, dq, Dc, Wc, w, dphi = self.ct_two_state(q, bt, sp, x, phi_ext + phi_rf(x), Fa)
                for sc_iters in range(1, self.ct_sc + 1):
                    L_ct = self.ct_drive(w, bt, sp)
                    x = solve_induction_blocks(L + L_ct, A_off, ainv_pad, Cq, quadrupoles)
                    e_ch, dq, Dc, Wc, w, dphi = self.ct_two_state(q, bt, sp, x, phi_ext + phi_rf(x), Fa)
                e_ct_sc = e_ch.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
                e_type = e_ch.new_zeros(B * self.nT).index_add(0, bt["ch_frame"] * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
                # E_total = 1/2 x A x + L x + sum_c E_c with A x = -(L + L_ct); E_CT = E_total - E_pol(x0)
                e_pol_sc = 0.5 * (L * x).sum((1, 2)) - 0.5 * (L_ct * x).sum((1, 2))
                e_ct_post = e_ct_sc + e_pol_sc - 0.5 * (L * x0).sum((1, 2))
                x_sc = x
        else:
            e_ct, e_type, e_ch, dphi = I.new_zeros(B), I.new_zeros(B, self.nT), I.new_zeros(0), I.new_zeros(0)
        if e_ct_post is None:
            e_ct_post = e_ct
        z = I.new_zeros(nC)
        if not self.frozen:
            e_elst = I.new_zeros(B)
        elif pbc is not None:
            e_elst = permanent_electrostatics(bt["pos"], sp, bt["frame"], self.tab, bt["pairs"], bt["pair_dr"], pbc)
        else:
            e_elst = permanent_electrostatics(bt["pos"], sp, bt["frame"], self.tab)
        e_cav, e_scr = I.new_zeros(B), I.new_zeros(B)
        if solvent is not None:
            e_scr, e_cav = solvent.e_scr, solvent.e_cav
            e_elst = e_elst + e_scr
        return {"e_pol": e_pol, "e_ct": e_ct_post, "e_ct_post": e_ct_post, "e_ct_0": e_ct, "e_type": e_type, "c": c,
                "e_cav": e_cav, "e_scr": e_scr, "phi_rf": phi_rf(x_sc) if solvent is not None else I.new_zeros(n),
                "dov": z, "A_ps": A_ps, "W": Wc, "Delta": Dc, "delta": z, "p": dq, "dq": Q_flux, "x": x_sc,
                "x_pol": x0, "joint_iters": sc_iters, "q_ct": q, "e_ch": e_ch, "stark": z, "e_pauli": e_pauli,
                "e_disp": e_disp, "frozen_params": frz, "dphi": dphi, "alpha_inv": ainv,
                "e_breath_atom": e_breath_atom, "e_elst": e_elst, "cg_iters": cg.get("cg_iters", 0),
                "e_xpol": e_xpol, "z_xpol": z_x,
                "gap_deltas": self.gap_deltas(I, sp) if self.ct_model == "gap" else None,
                "e_int": e_elst + e_cav + e_pauli + e_disp + e_pol + e_ct_post}

    def smoothness_penalty(self, bt, sigma=0.05):
        """Feature-noise penalty on the heads' log outputs (log c, log q) and the pseudo-potential strengths."""
        sp = bt["species"]
        f = bt["feats"]
        nz = (f.abs().sum(1, keepdim=True) > 0).to(f.dtype)
        noise = sigma * self.feat_scale * torch.randn_like(f) * nz
        I1, I2 = self.scaled(f), self.scaled(f + noise)
        if self.conf_form == "sq":
            self._s0sq = self.parent_overlap_sq(bt, sp)
        pen = ((torch.log(self.confinement_from(I2, sp)) - torch.log(self.confinement_from(I1, sp))) ** 2).mean()
        if self.env is not None:            # m33: the shared environment's readouts under channel noise
            X = self.env_raw_batch(bt)
            nzc = (X.abs().sum(1, keepdim=True) > 0).to(X.dtype)
            E1, E2 = self._env_extra(f, bt, I1), self._env_extra(f, bt, I2)
            u1 = self.env(self.scaled_channels(X), E1)
            u2 = self.env(self.scaled_channels(X + sigma * self.chan_scale * torch.randn_like(X) * nzc), E2)
            pen = pen + sum(((u2[k] - u1[k]) ** 2).mean() for k in u1)
            if self.ct_legacy_head:
                pen = pen + ((self.head_ctq(I2, sp) - self.head_ctq(I1, sp)) ** 2).mean()
            return pen / sigma ** 2
        if self.ct_model == "gap":
            c_ = self.frozen_cols
            pen = pen + ((self.head_gap(I2[:, c_], sp) - self.head_gap(I1[:, c_], sp)) ** 2).mean()
        else:
            pen = pen + ((self.head_ctq(I2, sp) - self.head_ctq(I1, sp)) ** 2).mean()
        if self.frozen or self.pseudo:
            c_ = self.frozen_cols
            pen = pen + ((self.head_frozen(I2[:, c_], sp) - self.head_frozen(I1[:, c_], sp)) ** 2).mean()
        if self.pseudo:
            c_ = self.frozen_cols
            pen = pen + ((self.head_ps(I2[:, c_], sp) - self.head_ps(I1[:, c_], sp)) ** 2).mean()
        return pen / sigma ** 2

    def pauli_prior(self):
        """Squared log deviation of the shells-form Pauli charges and exponent scales from the frozen fit's values
        (species the ion table has a fit for; 0 otherwise)."""
        if not hasattr(self, "pauli_fit_mask"):
            return next(self.parameters()).new_zeros(())
        m = self.pauli_fit_mask
        dw = ((self.pauli_lnw - self.pauli_lnw0) ** 2 * self.pauli_shell_mask).sum(1)
        ds = (self.pauli_lnbs - self.pauli_lnbs0) ** 2
        return ((dw + ds) * m).sum()

    @torch.no_grad()
    def summary(self):
        sym = self.tab.symbols
        out = {"qD0/qA0": {x: [round(v, 4) for v in self.ct_lnq0[k].exp().tolist()] for k, x in enumerate(sym)}}
        if self.env is not None:
            out["env"] = {"mode": self.env.mode, "latent": self.env.latent, "readouts": self.env.names,
                          "two_body": [p for p in self.env.names if self.env.two_body[p]], "pin_a0": self.pin_a0,
                          "extra": self.env_extra_names + (["|E|^2", "|gradE|^2"] if self.env_field else [])
                                   + (list(self.PEN_NAMES) if self.env_ct_pen else []),
                          "extra_readouts": [p for p in self.env.names if self.env.extra[p]]}
        bD, bA = self.ct_exponents()
        out["ct bD/bA"] = {x: [round(float(bD[k]), 4), round(float(bA[k]), 4)] for k, x in enumerate(sym)}
        if self.ct_model == "gap":
            out["gap delta amp D/A"] = {x: [round(v, 4) for v in self.gap_lna[k].exp().tolist()] for k, x in enumerate(sym)}
            out["A_acceptor"] = {x: round(float(self.acceptor_affinity()[k]), 4) for k, x in enumerate(sym)}
        if self.ct_model == "twostate":
            out["twostate I, A (table + shift)"] = {x: [round(float(self.ts_I0[k] + self.ts_dI[k]), 4),
                                                        round(float(self.ts_A0[k] + self.ts_dA[k]), 4)]
                                                    for k, x in enumerate(sym)}
            out["twostate outer / gap floor / gap learned"] = [self.ct_outer, self.ct_gap_min, self.ct_gap_learn]
        if self.ct_model == "flux":
            out["flux v0"] = {x: round(float(self.flux_lnv[k].exp()), 4) for k, x in enumerate(sym)}
            out["flux eta (+ J_self)"] = {x: [round(float(nn.functional.softplus(self.flux_eta[k])), 4),
                                              round(float(self.flux_jself[k]), 4)] for k, x in enumerate(sym)}
        if self.xpol:
            out["xpol kappa0 (dip, quad, breath)"] = {x: [round(v, 4) for v in self.xpol_lnk[k].exp().tolist()]
                                                      for k, x in enumerate(sym)}
            out["xpol lam"] = {x: round(1 + float(nn.functional.softplus(self.xpol_lam[k])), 4) for k, x in enumerate(sym)}
            if self.xpol_perp:
                out["xpol k_perp"] = {x: round(float(nn.functional.softplus(self.xpol_kperp[k])), 4) for k, x in enumerate(sym)}
        if self.aniso:
            out["kappa_aniso"] = {x: round(float(self.kappa_aniso[k].detach()), 4) for k, x in enumerate(sym)}
        if self.ct_gap:
            out["lam_D"] = {x: round(float(self.gap_lamD[k]), 4) for k, x in enumerate(sym)}
            out["lam_A"] = {x: round(float(self.gap_lamA[k]), 4) for k, x in enumerate(sym)}
        if self.ct_sc:
            out["ct_sc gap response exp(gD/gA)"] = {x: [round(float(self.sc_gD[k].exp()), 4), round(float(self.sc_gA[k].exp()), 4)]
                                                   for k, x in enumerate(sym)}
            out["ct_sc A_free (unbound acceptors)"] = {x: round(float(self.A_free[k]), 4) for k, x in enumerate(sym) if bool(self.A_unbound[k])}
        if self.ct_stark:
            out["ct stark softplus(eta_D/eta_A)"] = {x: [round(float(nn.functional.softplus(self.stark_D[k])), 4),
                                                        round(float(nn.functional.softplus(self.stark_A[k])), 4)] for k, x in enumerate(sym)}
        if self.hyper:
            out["hyper softplus(g)"] = {x: round(float(nn.functional.softplus(self.hyper_g[k])), 4) for k, x in enumerate(sym)}
        if self.conf_form == "sq":
            out["conf sq softplus(h)"] = {x: round(float(nn.functional.softplus(self.conf_h[k])), 4) for k, x in enumerate(sym)}
        if self.pauli_ii:
            out["pauli_ii kappa"] = {x: round(float(self.pp_kappa[k]), 4) for k, x in enumerate(sym)}
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
