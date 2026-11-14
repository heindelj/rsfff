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
  polarization); E_CT = E_joint - E_pol (the CT-induced repolarization is CT). One factorization of A + H serves both.

The environment factors u (``encoder.READOUTS``) are decoded from one latent per atom (:class:`Encoder`) whose inputs
are the overlap-weighted two- and three-body channels (``channels.py``) and the overlap-gated diffuse-probe
invariants. Clusters only for now: the periodic Gram blocks and the periodic flux solve are the condensed-phase work.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..dispersion import tt_damped_c6_energy
from ..ionct import kernels as K
from ..ionct.params import IonTable
from ..ionct.ct import ChannelTable
from ..multipole import slater_two_center_damp_unequal
from ..complete.channels import CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS
from ..complete.elst import permanent_electrostatics
from ..complete.environment import invariant_names
from ..complete.species import SpeciesEmbedding
from ..complete.tabulate import FluxTable, XpolTable, cached_table, flux_pair_terms, shape_self_coulomb, xpol_overlap_map
from .config import PAIR_GROUPS, CmmConfig
from .encoder import READOUTS, Encoder

__all__ = ["CmmModel", "induction_matrix_blocks", "PROBES", "GATED_NAMES"]

#: exchange pairs and environment channels end here (the frozen overlaps are < 1e-9 beyond)
R_MAX = 10.0
#: the overlap window over which a pair's exchange ratios and channel weights switch on (above the tables' noise)
O_ON, O_FULL = 1e-7, 1e-6
#: |ln(bA / b_Fukui)| bound of the floating cation acceptor exponents
CT_ACC_CAP = 0.5
#: the diffuse probes whose gated invariants the encoder reads
PROBES = ("donor", "acceptor", "breath")
GATED_NAMES = tuple(f"{p}:{k}" for p in PROBES for k in ("S0", "V0_pen", "|V1g|^2", "|V2g|^2", "V1g.S1"))
#: start values of the learned per-species constants (fits of the frozen-level labels for Na+ / Cl-)
PAULI_SHELL_INIT = {"Na": ([8.856], 0.871), "Cl": ([9.78, 0.302], 0.893)}
DISP_INIT = {"Na": (3.41, 3.872), "Cl": (554.7, 1.130)}
DISP_FALLBACK = (10.0, 2.0)
FLUX_V0_INIT = 0.025


def induction_matrix_blocks(A_off, alpha_inv, C):
    """(B, 8N, 8N) the induction matrix: A_off plus the on-site blocks (alpha_inv (B, N, 3, 3), 1/C (B, N) on the
    quadrupoles; C = 0: no quadrupole response)."""
    B, N = C.shape
    cinv = torch.where(C > 0, 1.0 / C.clamp(min=1e-30), torch.full_like(C, 1e12))
    D = alpha_inv.new_zeros(B, N, 8, 8)
    D[:, :, :3, :3] = alpha_inv
    D[:, :, 3:, 3:] = torch.diag_embed(cinv[..., None].expand(B, N, 5))
    ar = torch.arange(N, device=C.device)
    Ad = alpha_inv.new_zeros(B, N, 8, N, 8)
    Ad[:, ar, :, ar, :] = D.permute(1, 0, 2, 3)
    return A_off + Ad.reshape(B, 8 * N, 8 * N)


def _smoothstep_log(O, on, full):
    lo, hi = math.log10(on), math.log10(full)
    t = ((torch.log10(O.clamp(min=1e-300)) - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


def _saturate(x, m):
    """m tanh(x / m) per column where m > 0 (the training range's bound), x elsewhere."""
    if not bool((m > 0).any()):
        return x
    mm = torch.where(m > 0, m, torch.ones_like(m))
    return torch.where(m > 0, mm * torch.tanh(x / mm), x)


class CmmModel(nn.Module):
    """``CmmModel(tab, chan, config)``; ``forward(bt, field=None)`` on a batch of :func:`batched_precompute`."""

    def __init__(self, tab: IonTable, chan: ChannelTable, config: CmmConfig | None = None):
        super().__init__()
        cfg = config or CmmConfig()
        self.config, self.tab, self.chan = cfg, tab, chan
        f64 = dict(dtype=torch.float64)
        S, T = len(tab.symbols), tab.shell_n.shape[1]
        sym = tab.symbols
        self.ctypes = [f"{a}>{b}" for a in sym for b in sym]
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
        self.breath_col = list(invariant_names("sq")).index("V0_pen")       # within the breath probe's block
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
        self._tables = {}
        self._u = None
        self.double()

    # -- tables ------------------------------------------------------------------------------------------------
    def _table(self, name, device):
        t = self._tables.get(name)
        if t is None:
            if name == "xpol":
                t = cached_table(XpolTable, self.tab, r_max=R_MAX)
            elif name == "gram":
                from ..complete.gram import GramTable
                t = cached_table(GramTable, self.tab, r_max=R_MAX, breath="shape")
            elif name == "flux":
                t = cached_table(FluxTable, self.tab)
            else:
                from ..complete.tabulate import ProbeOverlapTable
                t = cached_table(ProbeOverlapTable, self.tab)
            self._tables[name] = t
        if t.device != device:
            t.to(device)
        return t

    # -- encoder inputs ----------------------------------------------------------------------------------------
    @staticmethod
    def _pairs(bt):
        """(i, j) and dr = X_j - X_i (Angstrom): the batch's pair list (periodic: with images) or all pairs."""
        if bt.get("pbc") is not None:
            return bt["pairs"], bt["pair_dr"]
        p = K.all_pairs(bt["frame"])
        return p, bt["pos"][p[1]] - bt["pos"][p[0]]

    def channels_raw(self, bt):
        """(N, 17) the raw two- and three-body channels of every atom (``channels.channels_from_pairs``)."""
        from ..complete.channels import channels_from_pairs, overlap_switch, pair_electrostatics
        sp = bt["species"]
        (i, j), dr = self._pairs(bt)
        r = dr.norm(dim=-1)
        keep = r < R_MAX
        i, j, dr, r = i[keep], j[keep], dr[keep], r[keep]
        a, b = torch.cat([i, j]), torch.cat([j, i])
        rr = torch.cat([r, r])
        tbl = self._table("xpol", dr.device)
        y = tbl(sp[a], sp[b], rr)
        O = y[:, tbl.cols(("xp:ovl",), y.device)[0]]
        w = overlap_switch(O) * O.clamp(min=0.0)
        E = pair_electrostatics(sp[a], sp[b], rr / K.BOHR_ANG, self.tab)
        return channels_from_pairs(sp.shape[0], a, torch.cat([dr, -dr]), w, E * w)

    def gate_reach(self):
        """(A) the largest distance at which any probe's overlap with any species is above the gate's on-value."""
        t = self._table("probe", torch.device("cpu"))
        on = t.coef[..., 0] > self.config.env_gate[0]
        hit = on.any(0).any(0).any(0)
        return float(t.r[hit].max()) if bool(hit.any()) else 0.0

    def gated_raw(self, bt):
        """(N, 15) the diffuse probes' invariants S0, V0_pen, |V1g|^2, |V2g|^2, V1g . S1 with the neighbours'
        point-charge field and gradient switched off with the probe's overlap with each neighbour (``env_gate``)."""
        env = bt.get("env")
        if not env:
            raise ValueError("the gated probe invariants need the equivariant environment blocks (bt['env'])")
        sp, X = bt["species"], bt["pos"] / K.BOHR_ANG
        n = sp.shape[0]
        pbc = bt.get("pbc")
        if pbc is not None:
            if pbc["cutoff"] < self.gate_reach():
                raise ValueError(f"the pair cutoff {pbc['cutoff']} A is inside the gate's reach {self.gate_reach():.2f} A")
            pr, du = bt["pairs"], bt["pair_dr"] / K.BOHR_ANG
        else:
            pr = K.all_pairs(bt["frame"])
            du = X[pr[1]] - X[pr[0]]
        i, j = torch.cat([pr[0], pr[1]]), torch.cat([pr[1], pr[0]])
        dr = torch.cat([du, -du])
        T0 = K._tensor(dr, torch.ones(5, dr.shape[0], dtype=X.dtype, device=X.device))[:, 0, :]
        Dp = (self.tab.charge[sp[j]].to(X.dtype)[:, None] * T0) @ K._poly_map(X.dtype, X.device)      # (P, 8)
        tbl = self._table("probe", X.device)
        ov = tbl(sp[i], sp[j], dr.norm(dim=-1) * K.BOHR_ANG)
        G = len(PROBES)
        c = [tbl.index[f"ov:{p}"] for p in PROBES]
        o = ov[:, c[0]:c[0] + G] if c == list(range(c[0], c[0] + G)) else ov[:, c]
        sig = _smoothstep_log(o, *self.config.env_gate)
        SD = torch.cat([-Dp[:, :3], Dp[:, 3:]], 1)
        Vg = X.new_zeros(n, G, 8).index_add(0, i, sig[:, :, None] * SD[:, None, :])
        if all(f"{p}:V1pen" in env for p in PROBES):
            Vpen = torch.stack([torch.cat([env[f"{p}:V1pen"], env[f"{p}:V2pen"]], 1) for p in PROBES], 1)
        elif pbc is None:
            Vpen = (torch.stack([torch.cat([env[f"{p}:V1"], env[f"{p}:V2"]], 1) for p in PROBES], 1)
                    - X.new_zeros(n, 8).index_add(0, i, SD)[:, None, :])
        else:
            raise ValueError("periodic frames need the penetration blocks (batched_precompute)")
        has = torch.stack([self.breath_mask[sp] if p == "breath" else torch.ones(n, dtype=X.dtype, device=X.device)
                           for p in PROBES], 1)
        V = (Vpen + Vg) * has[:, :, None]
        S1 = torch.stack([env[f"{p}:S1"] for p in PROBES], 1)
        V1 = V[..., :3]
        cols = torch.stack([torch.stack([env[f"{p}:S0"] for p in PROBES], 1),
                            torch.stack([env[f"{p}:V0"] for p in PROBES], 1),
                            (V1 * V1).sum(-1), (V[..., 3:] ** 2).sum(-1), (V1 * S1).sum(-1)], -1)
        return cols.reshape(n, G * 5)

    def encoder_inputs(self, X, Gv=None):
        """The encoder's input rows from the raw channels X (N, 17) and gated invariants Gv (N, 15): scaled, saturated,
        the unread channel groups zeroed; trunk order [three-body, two-body, gated]."""
        x = _saturate(X / self.chan_scale, self.chan_sat) * self.chan_keep
        parts = [x[:, list(THREE_BODY_COLS)], x[:, list(TWO_BODY_COLS)]]
        if self.config.probes:
            parts.append(_saturate(Gv / self.gate_scale, self.gate_sat))
        return torch.cat(parts, 1)

    def environment(self, bt):
        """{readout: (N,)} the raw log-factors of this batch (also kept for the parameter maps and the penalties)."""
        sp = bt["species"]
        X = self.channels_raw(bt)
        Gv = self.gated_raw(bt) if self.config.probes else None
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
        u = torch.stack([self._u["qD"], self._u["qA"]], 1)
        c = self.config.ct_q_cap
        return torch.exp(self.ct_ln_q0[sp] + c * torch.tanh(u / c))

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
        (i, j), dr = self._pairs(bt)
        r_ang = dr.norm(dim=-1)
        rb = r_ang / K.BOHR_ANG
        f = slater_two_center_damp_unequal(b[i][:, :, None], b[j][:, None, :], rb[:, None, None], 0)[0]
        e_p = torch.einsum("pt,ptu,pu->p", w[i], f, w[j]) / rb
        self._pauli_pairs = ((i, j), e_p)
        e_d = tt_damped_c6_energy(r_ang, torch.sqrt(C6[i] * C6[j]), torch.sqrt(b6[i] * b6[j]))
        fr = bt["frame"][i]
        return (r_ang.new_zeros(B).index_add(0, fr, e_p), r_ang.new_zeros(B).index_add(0, fr, e_d), (w, b, C6, b6))

    @staticmethod
    def _local(bt):
        fr = bt["frame"]
        counts = torch.bincount(fr, minlength=bt["B"])
        return fr, torch.arange(fr.shape[0], device=fr.device) - (torch.cumsum(counts, 0) - counts)[fr]

    @staticmethod
    def _hyper_ET(Lp):
        """E (n, 3) and the traceless gradient tensor T (n, 3, 3) of the permanent drive Lp (n, 8)."""
        E = -Lp[:, :3]
        q20, q21c, q21s, q22c, q22s = Lp[:, 3:8].unbind(1)
        h3 = 0.5 * 3.0 ** 0.5
        txx = 0.5 * (-q20 + 2 * h3 * q22c)
        tyy = 0.5 * (-q20 - 2 * h3 * q22c)
        txy, txz, tyz = h3 * q22s, h3 * q21c, h3 * q21s
        T = torch.stack([torch.stack([txx, txy, txz], -1), torch.stack([txy, tyy, tyz], -1),
                         torch.stack([txz, tyz, q20], -1)], -2)
        return E, T

    def hyperpolarization(self, Lp, sp):
        """The Legendre hyperpolarization: with the free-ion field energy W(E) = -1/2 a|E|^2 - gamma |E|^4 / 24 - b E T E
        and its polarized dipole mu_p, the ion's internal energy expanded to second order about mu_p is the solve's
        quadratic form with alpha(E) = a + gamma/6 (|E|^2 1 + 2 E E^T) + 2 b T on site (returned inverted), a field
        offset h = alpha(E)^-1 mu_p - E and a constant c = W + 1/2 mu_p alpha(E)^-1 mu_p."""
        E, T = self._hyper_ET(Lp)
        a = self.tab.alpha[sp][:, None, None] * torch.eye(3, dtype=E.dtype, device=E.device)
        g, bb = self.tab.hyper_gamma[sp], self.tab.hyper_b[sp]
        e2 = (E * E).sum(-1)
        eye = torch.eye(3, dtype=E.dtype, device=E.device)
        TE = (T @ E[:, :, None])[..., 0]
        aE = a + (g / 6)[:, None, None] * (e2[:, None, None] * eye + 2 * E[:, :, None] * E[:, None, :]) \
            + (2 * bb)[:, None, None] * T
        mu = (a @ E[:, :, None])[..., 0] + (g / 6 * e2)[:, None] * E + (2 * bb)[:, None] * TE
        W = -0.5 * (E * (a @ E[:, :, None])[..., 0]).sum(-1) - g * e2 * e2 / 24 - bb * (E * TE).sum(-1)
        ainvE = torch.linalg.inv(aE)
        aimu = (ainvE @ mu[:, :, None])[..., 0]
        return ainvE, aimu - E, W + 0.5 * (mu * aimu).sum(-1)

    def exchange_pairs(self, bt, sp):
        """Exchange pairs (i, j) within R_MAX: their Pauli energies E0 (P,), the drive vectors W1 d ln O / dz of each
        side (P, D) and the Gram geometry (dr bohr, A_ij = E0_ij / O_ij)."""
        tbl = self._table("xpol", bt["pos"].device)
        (i, j), e0 = self._pauli_pairs
        dr = (bt["pair_dr"] if bt.get("pbc") is not None else bt["pos"][j] - bt["pos"][i]) / K.BOHR_ANG
        keep = dr.norm(dim=-1) * K.BOHR_ANG < R_MAX
        i, j, dr, e0 = i[keep], j[keep], dr[keep], e0[keep]
        P = i.shape[0]
        dp = torch.stack([torch.cat([i, j]), torch.cat([j, i])])
        O, d, e = xpol_overlap_map(tbl, sp, dp, torch.cat([dr, -dr]))
        sw = _smoothstep_log(O, O_ON, O_FULL)
        e = e - O / self.tab.shell_n.sum(1)[sp[dp[1]]]          # the breathing density has zero monopole
        rel = torch.cat([d, e[:, None]], 1) * (sw / O.clamp(min=O_ON))[:, None]
        k = self.w1(sp)
        D = k.shape[1]
        vj = 0.5 * k[j] * rel[:P, :D]                            # (i <- j): j's deformation changes O_ij
        vi = 0.5 * k[i] * rel[P:, :D]
        return i, j, e0, vi, vj, (dr, e0 * sw[:P] / O[:P].clamp(min=O_ON))

    def gram_blocks(self, sp, i, j, geo, D):
        """(P, D, D) the on-site Hessians W2_i 2 A_ij Gt(i <- j) and W2_j 2 A_ij Gt(j <- i)."""
        from ..complete.gram import gram_blocks
        dr, Aij = geo
        tbl = self._table("gram", dr.device)
        w = (2 * Aij)[:, None, None]
        Gi = w * gram_blocks(tbl, sp, (i, j), dr)[:, :D, :D]
        Gj = w * gram_blocks(tbl, sp, (j, i), -dr)[:, :D, :D]
        w2 = self.w2(sp)
        return w2[i][:, None, None] * Gi, w2[j][:, None, None] * Gj

    def charge_slots(self, bt, sp, D):
        """C (B, nmax D, nmax) the coupling of a unit charge in each atom's slot (its dipole-parent shape) to the site
        variables, and H_Q (B, nmax, nmax) the slots' Coulomb with the self-energy + hardness on the diagonal."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        tbl = self._table("flux", bt["pos"].device)
        X = bt["pos"] / K.BOHR_ANG
        pairs = K.all_pairs(bt["frame"])
        dp = torch.cat([pairs, pairs.flip(0)], 1)
        a_, b_ = dp
        Kp, Jp = flux_pair_terms(tbl, sp, dp, X[b_] - X[a_])
        Kd = X.new_zeros(B, nmax, nmax, 8).index_put((fr[a_], loc[a_], loc[b_]), Kp)
        eta = self.ct_jself[sp] + nn.functional.softplus(self.ct_eta_raw[sp])
        Hq = X.new_zeros(B, nmax, nmax).index_put((fr[a_], loc[a_], loc[b_]), Jp).index_put((fr, loc, loc), eta)
        Ca = Kd.permute(0, 1, 3, 2)
        if D > 8:
            Ca = torch.cat([Ca, Ca.new_zeros(B, nmax, D - 8, nmax)], 2)
        return Ca.reshape(B, nmax * D, nmax), Hq

    @staticmethod
    def _reverse(i, j, n):
        """(C,) the index of each channel's reverse channel j -> i, -1 where there is none."""
        key, rkey = i * n + j, j * n + i
        srt, order = torch.sort(key)
        pos = torch.searchsorted(srt, rkey).clamp(max=max(key.shape[0] - 1, 0))
        hit = srt[pos] == rkey if key.shape[0] else torch.zeros_like(key, dtype=torch.bool)
        return torch.where(hit, order[pos], torch.full_like(key, -1))

    def channel_drives(self, bt, sp, q, h):
        """Per channel, given h_c = (G^T H_Q G)_cc: the kernel energy e_c <= 0, the drive V_c and s_c = 1 / sqrt(b_c).
        The two directions of a pair share the factor t_p that makes the pair's own minimum exactly e_c + e_c'."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_c = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        ae = (-e_c).clamp(min=0.0)
        v0 = torch.exp(self.ct_ln_v0)
        V0 = v0[sp[i]] + v0[sp[j]]
        Vh2 = V0 * V0 + 2 * ae * h
        cc = 2 * ae / Vh2
        rev = self._reverse(i, j, sp.shape[0])
        has = rev >= 0
        rv = rev.clamp(min=0)
        ae2 = torch.where(has, ae[rv], torch.zeros_like(ae))
        cc2 = torch.where(has, cc[rv], torch.zeros_like(cc))
        S_ = ae + ae2
        t2 = S_ * (1 - h * h * cc * cc2) / (S_ + 2 * h * torch.sqrt(ae * cc * ae2 * cc2 + 1e-300) + 1e-300)
        t2 = torch.where(has, t2, torch.ones_like(t2))
        return e_c, torch.sqrt(t2 * Vh2), torch.sqrt(2 * ae + 1e-30) / V0

    def solve(self, bt, sp, L, ainv_pad, Cq, rhs_extra):
        """The minimum of 1/2 z (A + H) z + (L + g) z over z = (dipole, quadrupole, breathing) for clusters, one dense
        factorization that also solves ``rhs_extra``. Returns z (B, nmax, D), the electrical part 1/2 z A z + L z (B,),
        the exchange part g z + 1/2 z H z (B,) and (A + H)^-1 rhs_extra."""
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        fr, loc = self._local(bt)
        A = induction_matrix_blocks(bt["A_off"], ainv_pad, Cq).reshape(B, nmax, 8, nmax, 8)
        D = 8 + self.n_b
        M = A.new_zeros(B, nmax, D, nmax, D)
        M[:, :, :8, :, :8] = A
        Lz = L.new_zeros(B, nmax, D)
        Lz[..., :8] = L
        if self.n_b:
            a_b = self.tab.a0[sp]
            has = a_b > 0
            ainv_b = torch.where(has, 1.0 / a_b.clamp(min=1e-30), torch.ones_like(a_b))
            V0 = bt["feats"][:, self._breath_feature] * has.to(a_b.dtype)
            ar = torch.arange(nmax, device=A.device)
            M[:, ar, 8, ar, 8] = torch.ones(B, nmax, dtype=A.dtype, device=A.device).index_put((fr, loc), ainv_b)
            Lz = Lz.index_put((fr, loc, torch.full_like(fr, 8)), -V0, accumulate=True)
        i, j, e0, vi, vj, geo = self.exchange_pairs(bt, sp)
        g = L.new_zeros(n, D).index_add(0, i, 2 * e0[:, None] * vi).index_add(0, j, 2 * e0[:, None] * vj)
        gz = L.new_zeros(B, nmax, D).index_put((fr, loc), g)
        f, li, lj = fr[i][:, None, None], loc[i][:, None, None], loc[j][:, None, None]
        a_ = torch.arange(D, device=A.device)[None, :, None]
        b_ = torch.arange(D, device=A.device)[None, None, :]
        Gi, Gj = self.gram_blocks(sp, i, j, geo, D)
        H = A.new_zeros(B, nmax, D, nmax, D)
        H = H.index_put((f, li, a_, li, b_), Gi, accumulate=True).index_put((f, lj, a_, lj, b_), Gj, accumulate=True)
        N = nmax * D
        Mf, Hf = M.reshape(B, N, N), H.reshape(B, N, N)
        rhs = torch.cat([(Lz + gz).reshape(B, N, 1), rhs_extra], 2)
        Y = torch.linalg.solve(Mf + Hf, rhs)
        z = -Y[..., 0]
        Lf = Lz.reshape(B, N)
        e_el = 0.5 * (z * (Mf @ z[..., None])[..., 0]).sum(1) + (Lf * z).sum(1)
        e_x = (gz.reshape(B, N) * z).sum(1) + 0.5 * (z * (Hf @ z[..., None])[..., 0]).sum(1)
        return z.reshape(B, nmax, D), e_el, e_x, Y[..., 1:]

    def charge_flux(self, bt, sp, Ca, Hq, z_pol, W, phi_ext):
        """The joint minimum with the channels eliminated: K = G diag(s^2) G^T, u = G s^2 V, (1 + K S) Q = u - K psi0
        with S = H_Q - C^T W and psi0 = phi_ext + C^T z_pol; q_c = s_c^2 (V_c - (psi_i - psi_j)), z* = z_pol - W Q.
        Returns E_CT (B,), per channel 1/2 b~_c q_c (C,), q (C,), z* (B, nmax D), Q (B, nmax), e_c (C,)."""
        B, N = Hq.shape[0], Hq.shape[1]
        fr, loc = self._local(bt)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        h = Hq[cf, li, li] + Hq[cf, lj, lj] - Hq[cf, li, lj] - Hq[cf, lj, li]
        e_c, V, s = self.channel_drives(bt, sp, self._q, h)
        w = s * s
        Kl = Hq.new_zeros(B, N, N)
        Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
        Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
        wV = w * V
        u = Hq.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
        psi0 = Hq.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Hq - Ca.transpose(1, 2) @ W
        Mq = torch.eye(N, dtype=Hq.dtype, device=Hq.device) + Kl @ S
        Q = torch.linalg.solve(Mq, (u - (Kl @ psi0[..., None])[..., 0])[..., None])[..., 0]
        psi = psi0 + (S @ Q[..., None])[..., 0]
        qc = w * (V - (psi[cf, li] - psi[cf, lj]))
        e_ch = 0.5 * (-V + psi0[cf, li] - psi0[cf, lj]) * qc
        z = z_pol - (W @ Q[..., None])[..., 0]
        return Hq.new_zeros(B).index_add(0, cf, e_ch), e_ch, qc, z, Q, e_c

    # -- the model ---------------------------------------------------------------------------------------------
    @property
    def _breath_feature(self):
        """The column of breath:V0_pen in the batch's environment invariants (the breathing drive)."""
        from ..complete.environment import PROBE_NAMES
        names = invariant_names("sq")
        return PROBE_NAMES.index("breath") * len(names) + self.breath_col

    def forward(self, bt: dict, field=None) -> dict:
        """``bt`` from :func:`batched_precompute`; ``field`` (B, 3) a uniform external field per frame (atomic units).
        Returns the energy terms (B,) (Hartree) and the solution."""
        if bt.get("pbc") is not None:
            raise NotImplementedError("cmm: clusters only so far (periodic Gram blocks and flux solve not implemented)")
        sp = bt["species"]
        self.environment(bt)
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        dt, dev = bt["pos"].dtype, bt["pos"].device
        Cq = torch.ones(B, nmax, dtype=dt, device=dev).masked_scatter(mask, self.tab.C[sp])
        m8 = mask[..., None].expand(B, nmax, 8)
        phi_ext = torch.zeros(n, dtype=dt, device=dev)
        L = bt["L"]
        if field is not None:
            Fa = field[bt["frame"]]
            L = L - torch.zeros(B, nmax, 8, dtype=dt, device=dev).masked_scatter(m8, torch.cat([Fa, Fa.new_zeros(n, 5)], 1))
            phi_ext = -(Fa * bt["pos"] / K.BOHR_ANG).sum(-1)
        ainv, hyp_h, hyp_c = self.hyperpolarization(bt["L"][mask], sp)
        L = L - torch.zeros(B, nmax, 8, dtype=dt, device=dev).masked_scatter(m8, torch.cat([hyp_h, hyp_h.new_zeros(n, 5)], 1))
        ainv_pad = torch.eye(3, dtype=dt, device=dev).expand(B, nmax, 3, 3).clone()
        ainv_pad = ainv_pad.masked_scatter(mask[..., None, None].expand(B, nmax, 3, 3), ainv)
        e_pauli, e_disp, frz = self.frozen_terms(bt, sp)
        D = 8 + self.n_b
        self._q = self.ct_charges(sp)
        nC = bt["ch_index"].shape[1]
        Ca, Hq = self.charge_slots(bt, sp, D)
        z_pol, e_el, e_x, W = self.solve(bt, sp, L, ainv_pad, Cq, Ca)
        e_pol = e_el + e_x + torch.zeros(B, dtype=dt, device=dev).index_add(0, bt["frame"], hyp_c)
        Q = torch.zeros(n, dtype=dt, device=dev)
        x = z_pol[..., :8]
        if nC:
            e_ct, e_ch, qc, zj, Qd, e_c = self.charge_flux(bt, sp, Ca, Hq, z_pol.reshape(B, -1), W, phi_ext)
            e_type = e_ch.new_zeros(B * self.nT).index_add(0, bt["ch_frame"] * self.nT + bt["ch_type"], e_ch).reshape(B, self.nT)
            x = zj.reshape(B, nmax, -1)[..., :8]
            fr, loc = self._local(bt)
            Q = Qd[fr, loc]
        else:
            e_ct, e_ch, qc, e_c = (torch.zeros(B, dtype=dt, device=dev), torch.zeros(0, dtype=dt, device=dev),
                                   torch.zeros(0, dtype=dt, device=dev), torch.zeros(0, dtype=dt, device=dev))
            e_type = torch.zeros(B, self.nT, dtype=dt, device=dev)
        e_elst = permanent_electrostatics(bt["pos"], sp, bt["frame"], self.tab)
        return {"e_elst": e_elst, "e_pauli": e_pauli, "e_disp": e_disp, "e_pol": e_pol, "e_xpol": e_x, "e_ct": e_ct,
                "e_int": e_elst + e_pauli + e_disp + e_pol + e_ct, "e_ch": e_ch, "e_type": e_type, "q_ch": qc,
                "e_kernel": e_c, "x": x, "x_pol": z_pol[..., :8], "z_pol": z_pol, "dq": Q, "q_ct": self._q,
                "frozen_params": frz}

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
        X = self.channels_raw(bt)
        Gv = self.gated_raw(bt) if self.config.probes else None
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
                f"eta {nn.functional.softplus(self.ct_eta_raw[k]):.3f}" for k, x in enumerate(sym)]
        return "\n".join(rows)
