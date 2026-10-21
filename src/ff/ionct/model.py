"""The ion CT force field with its two learned quantities (ion ct ff plan, R4/R5).

Everything physical is fixed (``IonTable``, ``ChannelTable``): permanent shaped densities, free-ion alpha and C,
induced-density shapes, frontier densities, intrinsic gaps and the pair couplings W_T(R). The network emits

    c_alpha,i, c_C,i   confinement of the free-ion polarizabilities (exactly 1 for an isolated ion)
    Delta^ov_ij >= 0   the whole overlap gap of channel i -> j, partner included (the fully learned form)
    A_ij >= 0          (pseudo=True) strength of the Pauli pseudo-potential of ion j acting on ion i's response: a
                       short-ranged, chargeless drive (kernels.pseudo_drive_basis) added to the induction drive. It
                       lowers E_pol without touching the response to a uniform field, which the confinement alone sets

and the model does the rest: mutual induction with the confined polarizabilities (E_pol), the frontier potentials
from permanent + induced moments, the gaps Delta = g_T + s dphi + Delta^ov, the WKB coupling response and the
per-donor arrowhead (E_CT). No CT feedback into the electrostatics.

Training on fixed geometries (EDA energies) only needs the network to change, so every geometry-dependent tensor is
precomputed once per frame (:func:`precompute`) and frames are evaluated as a padded batch (:func:`collate`,
:meth:`IonCTModel.forward`). Units: atomic (energies Hartree).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import ct as CT
from . import kernels as K
from .params import IonTable
from ..multipole import slater_two_center_damp_unequal

__all__ = ["FrameCache", "precompute", "collate", "IonCTModel", "CHANNEL_TYPES", "PSEUDO_TYPES", "channel_types", "pseudo_type_names"]

CHANNEL_TYPES = ("Cl>Na", "Na>Cl", "Cl>Cl", "Na>Na")       # the NaCl legacy order (cached labels and runs)


def channel_types(symbols):
    """Directed channel type names 'D>A' for a species list: the legacy NaCl order for {Na, Cl} (the cached labels
    and the trained runs index it), every ordered pair otherwise."""
    if set(symbols) == {"Na", "Cl"}:
        return CHANNEL_TYPES
    return tuple(f"{a}>{b}" for a in symbols for b in symbols)


def pseudo_type_names(symbols):
    """Pseudo-potential drive types 'i<j' for a species list (same ordering rule as :func:`channel_types`)."""
    if set(symbols) == {"Na", "Cl"}:
        return PSEUDO_TYPES
    return tuple(f"{a}<{b}" for a in symbols for b in symbols)
#: Pauli pseudo-potential drive types, "i<j": the response of i to the pseudo-potential of j
PSEUDO_TYPES = ("Cl<Na", "Na<Cl", "Cl<Cl", "Na<Na")
#: scale of the pseudo-potential strength per type, so the network's softplus output is O(1): A = A0 softplus(z).
#: Roughly the strength at which the pseudo drive equals the permanent drive at contact (Na+ is compact, so its
#: pseudo-potential reaches the Cl- response density weakly)
PSEUDO_A0 = {"Cl<Na": 40.0, "Na<Cl": 5.0, "Cl<Cl": 5.0, "Na<Na": 5.0}
PSEUDO_A0_DEFAULT = 5.0
PSEUDO_CUTOFF = 6.0
#: saturation (Hartree) of the CT-state Stark shift: Delta -= S tanh(u / S), u = 1/2 a*_A |E_A|^2 + 1/2 a*_D |E_D|^2
STARK_SAT = 0.3
#: initial excess polarizabilities of the CT state (bohr^3), per species: as acceptor (the transferred electron;
#: neutral-atom estimate alpha(X0) - alpha(X+) for Na, a diffuse-anion guess for Cl) and as donor (the hole:
#: alpha(X0) - alpha(X-) for Cl, ~0 for Na)
#: R9 density coupling W = w_T O(b_D,eff, b_A; R)^gamma_T: initial ln w_T, gamma_T per channel type (a power-law
#: fit of the free-shape frontier overlap to the old pair-inverted couplings; both are trained), the barrier
#: response lambda_T, and the saturation of the barrier argument
COUPLING_INIT = {"Cl>Na": (-0.49, 0.803), "Na>Cl": (2.88, 1.328), "Cl>Cl": (-3.68, 0.466), "Na>Na": (-6.11, 0.365)}
LAMBDA_INIT = 0.5
BARRIER_SAT = 2.0
#: initial effective electron affinity (Hartree) of acceptors without a bound extra-electron state (Cl- -> Cl2-)
A_UNBOUND_INIT = -0.3
#: polarizable frontier densities (coupling_resp="frontier"): initial orbital polarizabilities (bohr^3) of the
#: acceptor (virtual) and donor (occupied) frontier orbitals per species, and the saturation (bohr) of their
#: centroid displacement
AORB_INIT = {"acceptor": {"Na": 60.0, "Cl": 150.0}, "donor": {"Na": 1.0, "Cl": 20.0}}
DISP_MAX = 3.0
#: pol_form="pauli": CMM Slater Pauli kernel on permanent + induced moments. Initial (qP [e], bP [1/bohr]) per
#: species from a fit of the monopole kernel qP_i qP_j f_1(sqrt(b_i b_j) r)/r to ALMO mod_pauli of the Na/Cl pairs
#: (> 0.5 kJ/mol; rms 10% in ln E), the induced Pauli-moment scales kappa (dipole, quadrupole slots) and the
#: second-order stiffening [s0d, s2d, s0q] (set so a Cl- 2.8 A from a Na+ has its response cut by ~25%)
PAULI_INIT = {"Na": (0.978, 1.360), "Cl": (6.575, 1.577)}
PAULI_KAPPA_INIT = {"Na": (1.0, 1.0), "Cl": (1.0, 1.0)}
PAULI_STIFF_INIT = {"Na": (5.0, 0.0, 5.0), "Cl": (4.4, 0.0, 0.66)}
ASTAR_INIT = {"acceptor": {"Na": 162.0, "Cl": 300.0}, "donor": {"Na": 0.0, "Cl": -16.0}}


@dataclass
class FrameCache:
    species: torch.Tensor        # (N,)
    pos: torch.Tensor            # (N, 3) A
    L: torch.Tensor              # (N, 8) induction drive
    A_off: torch.Tensor          # (8N, 8N)
    VD: torch.Tensor             # (N, N) donor-frontier potentials, permanent part
    KD: torch.Tensor             # (N, N, 8) induced kernel
    VA: torch.Tensor
    KA: torch.Tensor
    ch_index: torch.Tensor       # (2, C)
    ch_type: torch.Tensor        # (C,) index into CHANNEL_TYPES
    ch_r: torch.Tensor           # (C,) A
    ch_g: torch.Tensor
    ch_lnW: torch.Tensor
    ch_dphi_pair: torch.Tensor
    feats: torch.Tensor          # (N, F) environment invariants (zero for an isolated ion)
    ps_index: torch.Tensor = None   # (2, P) directed pairs (i <- j) of the Pauli pseudo-potential drive
    ps_type: torch.Tensor = None    # (P,) index into PSEUDO_TYPES
    ps_r: torch.Tensor = None       # (P,) A
    ps_basis: torch.Tensor = None   # (P, 8) drive on i per unit pseudo-potential strength on j
    EVD: torch.Tensor = None        # (N, N, 3) field of j's permanent density over i's donor frontier
    EKD: torch.Tensor = None        # (N, N, 3, 8) ... of j's induced moments
    EVA: torch.Tensor = None        # acceptor-frontier versions
    EKA: torch.Tensor = None
    ch_EP: torch.Tensor = None      # (C, 3) field of the CT hole at the acceptor (= of the electron at the donor)
    ch_J: torch.Tensor = None       # (C,) electron-hole Coulomb of the donor / acceptor frontier densities
    ch_ltaper: torch.Tensor = None  # (C,) log of the channel cutoff taper
    ch_mid: torch.Tensor = None     # (C, 3) A, bond midpoint (barrier point)
    ch_Vmid: torch.Tensor = None    # (C, N) potential at the midpoint of every ion but the donor, permanent part
    ch_Kmid: torch.Tensor = None    # (C, N, 8) ... induced kernel
    labels: dict = field(default_factory=dict)   # 'pol', 'ct' (Hartree), masks
    meta: dict = field(default_factory=dict)
    env: dict = None                # equivariant environment blocks, "<probe>:<V0|V1|V2|S0|S1|S2>" -> (N, ...)
    n_types: int = len(CHANNEL_TYPES)   # number of channel types of the table (the width of the per-type sums)


def precompute(symbols, pos_ang, tab: IonTable, chan: CT.ChannelTable, featurizer, labels=None, meta=None,
               differentiable=False, lean=False, frontier=False) -> FrameCache:
    """Geometry-dependent tensors of one frame. With ``differentiable`` they keep the graph to ``pos_ang`` (forces
    through the cached forward); otherwise they are computed without autograd. ``lean`` computes only what the
    direct CT form and the confinement / pseudo-potential polarization use (induction operator, channel list,
    pseudo drive, features): the frontier potentials, fields, midpoint potentials and electron-hole Coulomb of the
    channel CT model are left out (zeros / None), which is most of the cost. ``frontier`` keeps the frontier
    potentials (VD, KD, VA, KA; cheap) in the lean cache -- the complete model's CT gap factor uses them. When the
    featurizer returns equivariant blocks (:class:`rsfff.ff.complete.EnvFeatures`), they are kept on ``env``."""
    with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
        return _precompute(symbols, pos_ang, tab, chan, featurizer, labels, meta, lean, frontier)


def _precompute(symbols, pos_ang, tab, chan, featurizer, labels, meta, lean=False, frontier=False) -> FrameCache:
    Zmap = {"Na": 11, "Cl": 17}
    pos = torch.as_tensor(pos_ang, dtype=torch.float64)
    Z = torch.tensor([Zmap[s] for s in symbols])
    s = tab.species_index(Z)
    b = torch.zeros(len(symbols), dtype=torch.long)
    L, A_off = K.induction_operator(pos, s, b, tab)
    geom = CT.channel_geometry(pos, s, b, tab, chan)
    ctypes = channel_types(tab.symbols)
    ch_type = torch.tensor([ctypes.index(n) for n in geom["names"]], dtype=torch.long)
    ef = featurizer(positions=pos, atomic_numbers=Z, batch_idx=b)
    feats = ef.inv_feats.to(torch.float64)
    env = None
    if type(ef).__name__ == "EnvFeatures" and ef.blocks:        # rsfff.ff.complete.EnvFeatures (no import cycle)
        env = {f"{p}:{k}": v.to(torch.float64) for p, blk in ef.blocks.items() for k, v in blk.items()}
    ps_index, ps_r, ps_basis = K.pseudo_drive_basis(pos, s, b, tab, PSEUDO_CUTOFF)
    ps_type = pseudo_types(s, ps_index, tab)
    ch_ltaper = torch.log(CT._taper(geom["r"], chan.cutoff, chan.taper).clamp(min=1e-300))
    if lean:
        N = pos.shape[0]
        if frontier:
            VD, KD = K.frontier_operator(pos, s, b, tab, "donor")
            VA, KA = K.frontier_operator(pos, s, b, tab, "acceptor")
        else:
            VD, KD = pos.new_zeros(N, N), pos.new_zeros(N, N, K.D_SITE)
            VA, KA = VD.clone(), KD.clone()
        fc = FrameCache(s, pos, L, A_off, VD, KD, VA, KA, geom["index"], ch_type, geom["r"],
                        geom["g"], geom["lnW"], geom["dphi_pair"], feats, ps_index, ps_type, ps_r, ps_basis,
                        labels=dict(labels or {}), meta=dict(meta or {}), env=env)
        fc.ch_ltaper = ch_ltaper
        fc.n_types = len(ctypes)
        fc.meta["lean"] = True
        fc.meta["frontier"] = bool(frontier)
        return fc
    VD, KD = K.frontier_operator(pos, s, b, tab, "donor")
    VA, KA = K.frontier_operator(pos, s, b, tab, "acceptor")
    EVD, EKD = K.frontier_field_operator(pos, s, b, tab, "donor")
    EVA, EKA = K.frontier_field_operator(pos, s, b, tab, "acceptor")
    ch_EP = K.frontier_hole_field(pos, s, tab, geom["index"])
    i, j = geom["index"]
    ch_J = K.frontier_pair_coulomb(pos, s, tab, geom["index"])
    ch_mid = 0.5 * (pos[i] + pos[j])
    ch_Vmid, ch_Kmid = K.point_potential_operator(pos, s, tab, ch_mid, i)
    return FrameCache(s, pos, L, A_off, VD, KD, VA, KA, geom["index"], ch_type, geom["r"], geom["g"], geom["lnW"],
                      geom["dphi_pair"], feats, ps_index, ps_type, ps_r, ps_basis, EVD, EKD, EVA, EKA, ch_EP,
                      ch_J, ch_ltaper, ch_mid, ch_Vmid, ch_Kmid, dict(labels or {}), dict(meta or {}), env, len(ctypes))


def pseudo_types(species, idx, tab: IonTable):
    sym = tab.symbols
    ptypes = pseudo_type_names(tab.symbols)
    return torch.tensor([ptypes.index(f"{sym[species[i]]}<{sym[species[j]]}") for i, j in idx.T.tolist()],
                        dtype=torch.long)


def collate(frames: list[FrameCache]) -> dict:
    """Pad the per-frame matrices to the largest frame; flatten atoms and channels with offsets."""
    B = len(frames)
    nmax = max(f.species.shape[0] for f in frames)
    dt = frames[0].L.dtype
    L = torch.zeros(B, nmax, 8, dtype=dt)
    A = torch.zeros(B, 8 * nmax, 8 * nmax, dtype=dt)
    VD = torch.zeros(B, nmax, nmax, dtype=dt)
    VA = torch.zeros(B, nmax, nmax, dtype=dt)
    KD = torch.zeros(B, nmax, nmax, 8, dtype=dt)
    KA = torch.zeros(B, nmax, nmax, 8, dtype=dt)
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    species, feats, frame_of, idx, ty, r, g, lnW, dpp, ch_frame = [], [], [], [], [], [], [], [], [], []
    pos, ps_idx, ps_ty, ps_r, ps_basis = [], [], [], [], []
    has_E = all(f.EVD is not None for f in frames)
    EVD = torch.zeros(B, nmax, nmax, 3, dtype=dt) if has_E else None
    EVA = torch.zeros(B, nmax, nmax, 3, dtype=dt) if has_E else None
    EKD = torch.zeros(B, nmax, nmax, 3, 8, dtype=dt) if has_E else None
    EKA = torch.zeros(B, nmax, nmax, 3, 8, dtype=dt) if has_E else None
    ch_EP = []
    has_D = all(f.ch_J is not None for f in frames)
    has_T = all(f.ch_ltaper is not None for f in frames)
    cJ, cT, cM, cV, cK = [], [], [], [], []
    off = 0
    for k, f in enumerate(frames):
        n = f.species.shape[0]
        L[k, :n] = f.L
        A[k, :8 * n, :8 * n] = f.A_off
        VD[k, :n, :n], VA[k, :n, :n], KD[k, :n, :n], KA[k, :n, :n] = f.VD, f.VA, f.KD, f.KA
        mask[k, :n] = True
        species.append(f.species)
        feats.append(f.feats)
        frame_of.append(torch.full((n,), k, dtype=torch.long))
        idx.append(f.ch_index + off)
        ty.append(f.ch_type)
        r.append(f.ch_r)
        g.append(f.ch_g)
        lnW.append(f.ch_lnW)
        dpp.append(f.ch_dphi_pair)
        ch_frame.append(torch.full((f.ch_r.shape[0],), k, dtype=torch.long))
        pos.append(f.pos)
        if has_E:
            EVD[k, :n, :n], EVA[k, :n, :n], EKD[k, :n, :n], EKA[k, :n, :n] = f.EVD, f.EVA, f.EKD, f.EKA
            ch_EP.append(f.ch_EP)
        if has_T:
            cT.append(f.ch_ltaper)
        if has_D:
            C = f.ch_J.shape[0]
            cJ.append(f.ch_J)
            cM.append(f.ch_mid)
            cV.append(torch.cat([f.ch_Vmid, f.ch_Vmid.new_zeros(C, nmax - n)], 1))
            cK.append(torch.cat([f.ch_Kmid, f.ch_Kmid.new_zeros(C, nmax - n, 8)], 1))
        if f.ps_index is not None:
            ps_idx.append(f.ps_index + off)
            ps_ty.append(f.ps_type)
            ps_r.append(f.ps_r)
            ps_basis.append(f.ps_basis)
        off += n
    env = None
    if all(f.env for f in frames):
        env = {k: torch.cat([f.env[k] for f in frames]) for k in frames[0].env}
    lab = {key: torch.tensor([f.labels.get(key, float("nan")) for f in frames], dtype=dt)
           for key in ("pol", "ct", "pauli", "disp", "elec", "int")}
    lab["ct_mask"] = torch.tensor([bool(f.labels.get("ct_mask", True)) for f in frames])
    nT = getattr(frames[0], "n_types", len(CHANNEL_TYPES))
    lab["chan"] = torch.tensor([f.labels.get("chan") or [float("nan")] * nT for f in frames], dtype=dt)
    lab["n"] = torch.tensor([f.species.shape[0] for f in frames], dtype=dt)
    return {"B": B, "nmax": nmax, "L": L, "A_off": A, "VD": VD, "VA": VA, "KD": KD, "KA": KA, "mask": mask,
            "species": torch.cat(species), "feats": torch.cat(feats), "frame": torch.cat(frame_of),
            "ch_index": torch.cat(idx, 1), "ch_type": torch.cat(ty), "ch_r": torch.cat(r), "ch_g": torch.cat(g),
            "ch_lnW": torch.cat(lnW), "ch_dphi_pair": torch.cat(dpp), "ch_frame": torch.cat(ch_frame), "labels": lab,
            "pos": torch.cat(pos),
            "ps_index": torch.cat(ps_idx, 1) if ps_idx else torch.zeros(2, 0, dtype=torch.long),
            "ps_type": torch.cat(ps_ty) if ps_ty else torch.zeros(0, dtype=torch.long),
            "ps_r": torch.cat(ps_r) if ps_r else torch.zeros(0, dtype=dt),
            "ps_basis": torch.cat(ps_basis) if ps_basis else torch.zeros(0, 8, dtype=dt),
            "EVD": EVD, "EVA": EVA, "EKD": EKD, "EKA": EKA,
            "ch_EP": torch.cat(ch_EP) if has_E else None,
            "ch_J": torch.cat(cJ) if has_D else None, "ch_ltaper": torch.cat(cT) if has_T else None,
            "ch_mid": torch.cat(cM) if has_D else None, "ch_Vmid": torch.cat(cV) if has_D else None,
            "ch_Kmid": torch.cat(cK) if has_D else None, "env": env}


def _mlp(i, h, o, depth):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class IonCTModel(nn.Module):
    """Confinement and overlap-gap heads on per-ion environment invariants, plus the fixed physics."""

    def __init__(self, tab: IonTable, chan: CT.ChannelTable, n_feat: int, *, hidden=64, depth=2, emb=8, n_rbf=16,
                 r_rbf=(1.5, 8.0), c_range=(0.05, 1.25), ov_init=0.15, wkb="smooth+floor", feat_scale=None,
                 confine=None, pseudo=False, ps_init=0.02, joint=False, joint_tol=1e-9, joint_max_iter=30,
                 ct_pol=False, gap_form="pair", tie_c=False, coupling_resp="barrier", ct_form="channels",
                 pol_form="confine", pauli_nn=False):
        super().__init__()
        self.tab, self.chan, self.wkb = tab, chan, wkb
        self.ctypes, self.ptypes = channel_types(tab.symbols), pseudo_type_names(tab.symbols)
        self.nT = len(self.ctypes)
        self.joint, self.joint_tol, self.joint_max_iter = joint, joint_tol, joint_max_iter
        S = len(tab.symbols)
        self.register_buffer("feat_scale", torch.ones(n_feat) if feat_scale is None else feat_scale.clone())
        self.emb = nn.Embedding(S, emb)
        self.atom = _mlp(n_feat + emb, hidden, hidden, depth)
        self.conf = _mlp(hidden, hidden, 2, 1)
        nn.init.zeros_(self.conf[-1].weight)
        nn.init.zeros_(self.conf[-1].bias)
        self.log_c_lo, self.log_c_hi = math.log(c_range[0]), math.log(c_range[1])
        # species whose polarizabilities the network may confine (default all); the others stay free-ion
        confine = tab.symbols if confine is None else tuple(confine)
        self.register_buffer("confine_mask", torch.tensor([float(x in confine) for x in tab.symbols], dtype=torch.float64))
        self.register_buffer("rbf_mu", torch.linspace(r_rbf[0], r_rbf[1], n_rbf))
        self.rbf_w = (r_rbf[1] - r_rbf[0]) / (n_rbf - 1)
        self.gap = _mlp(2 * hidden + n_rbf + self.nT, hidden, 1, depth)
        nn.init.zeros_(self.gap[-1].weight)
        nn.init.constant_(self.gap[-1].bias, math.log(math.expm1(ov_init)))
        # Pauli pseudo-potential strengths A_ij >= 0 (directed pairs within PSEUDO_CUTOFF)
        self.pseudo = pseudo
        if pseudo:
            self.ps = _mlp(2 * hidden + n_rbf + len(self.ptypes), hidden, 1, depth)
            nn.init.zeros_(self.ps[-1].weight)
            nn.init.constant_(self.ps[-1].bias, math.log(math.expm1(ps_init)))
            self.register_buffer("ps_A0", torch.tensor([PSEUDO_A0.get(t, PSEUDO_A0_DEFAULT) for t in self.ptypes], dtype=torch.float64))
        # excess polarizability of the CT state (bohr^3 / 100): the transferred electron (acceptor, >= 0) and the
        # hole (donor, either sign), one per species
        self.ct_pol = ct_pol
        # R9: gap and coupling from densities (no pair-calibrated g_T, W_T(R), dphi_pair)
        self.gap_form, self.tie_c = gap_form, tie_c
        self.coupling_resp = coupling_resp
        # CMM-style direct CT (baseline): per-ion donor / acceptor CT charges from the environment and a short-range,
        # purely attractive Slater kernel -- E = -sum_{i != j} qD_i qA_j f(bD_i, bA_j; r) / r, where f / r is the
        # Coulomb of two Slater 1s densities with the point-charge 1/r subtracted (penetration only)
        self.ct_form = ct_form
        if ct_form == "direct":
            self.ctq = _mlp(hidden, hidden, 2, depth)
            nn.init.zeros_(self.ctq[-1].weight)
            nn.init.zeros_(self.ctq[-1].bias)                       # q = softplus(0) / ln 2 = 1 at start
            b0 = lambda role: torch.log(torch.expm1(tab.front_b[role].clone()))
            self.ct_bD_raw = nn.Parameter(b0("donor"))
            self.ct_bA_raw = nn.Parameter(b0("acceptor"))
        # pol_form="pauli": no confinement network, no pseudo-potential network -- the response is shaped by one Pauli
        # kernel acting on the permanent and induced moments (kernels.pauli_kernel), per-species constants; with
        # pauli_nn the network rescales each ion's Pauli monopole (exactly 1 for an isolated ion)
        self.pol_form, self.pauli_nn = pol_form, pauli_nn
        if pol_form == "pauli":
            if pseudo or joint:
                raise ValueError("pol_form='pauli' replaces the pseudo-potential and does not support the joint solve")
            sym = tab.symbols
            f64 = dict(dtype=torch.float64)
            isp = lambda v: torch.log(torch.expm1(torch.tensor(v, **f64)))
            self.pauli_lnq = nn.Parameter(torch.tensor([math.log(PAULI_INIT[x][0]) for x in sym], **f64))
            self.pauli_lnb = nn.Parameter(torch.tensor([math.log(PAULI_INIT[x][1]) for x in sym], **f64))
            self.pauli_kappa_raw = nn.Parameter(isp([PAULI_KAPPA_INIT[x] for x in sym]))            # (S, 2)
            self.pauli_s0_raw = nn.Parameter(isp([(PAULI_STIFF_INIT[x][0], PAULI_STIFF_INIT[x][2]) for x in sym]))
            # s2d = s0d (-1.5 + 4.5 sigmoid(z)) keeps H_d positive definite (eigenvalues s0d + 2/3 s2d, s0d - s2d/3)
            z0 = [math.log(1 / 3 + PAULI_STIFF_INIT[x][1] / (4.5 * PAULI_STIFF_INIT[x][0])) -
                  math.log(2 / 3 - PAULI_STIFF_INIT[x][1] / (4.5 * PAULI_STIFF_INIT[x][0])) for x in sym]
            self.pauli_s2_raw = nn.Parameter(torch.tensor(z0, **f64))
            if pauli_nn:
                self.pauli_q = _mlp(hidden, hidden, 1, depth)
                nn.init.zeros_(self.pauli_q[-1].weight)
                nn.init.zeros_(self.pauli_q[-1].bias)
        if coupling_resp == "frontier":
            if not (ct_pol and gap_form == "density"):
                raise ValueError("coupling_resp='frontier' needs gap_form='density' and ct_pol=True")
            sp_ = lambda v: torch.log(torch.expm1(torch.tensor(v / 100, dtype=torch.float64)))
            self.aorb_acc_raw = nn.Parameter(torch.stack([sp_(AORB_INIT["acceptor"][x]) for x in tab.symbols]))
            self.aorb_don_raw = nn.Parameter(torch.stack([sp_(AORB_INIT["donor"][x]) for x in tab.symbols]))
        if gap_form == "density":
            self.lnw = nn.Parameter(torch.tensor([COUPLING_INIT.get(t, (0.0, 1.0))[0] for t in self.ctypes], dtype=torch.float64))
            self.gamma = nn.Parameter(torch.tensor([COUPLING_INIT.get(t, (0.0, 1.0))[1] for t in self.ctypes], dtype=torch.float64))
            self.lam = nn.Parameter(torch.full((self.nT,), LAMBDA_INIT, dtype=torch.float64))
            unbound = torch.isnan(tab.A_acceptor)
            self.register_buffer("A_fixed", torch.nan_to_num(tab.A_acceptor))
            self.register_buffer("A_unbound", unbound)
            self.A_free = nn.Parameter(torch.full((len(tab.symbols),), A_UNBOUND_INIT, dtype=torch.float64))
        if ct_pol:
            a0 = torch.tensor([ASTAR_INIT["acceptor"][x] / 100 for x in tab.symbols], dtype=torch.float64)
            self.astar_acc_raw = nn.Parameter(torch.log(torch.expm1(a0)))
            self.astar_don = nn.Parameter(torch.tensor([ASTAR_INIT["donor"][x] / 100 for x in tab.symbols],
                                                       dtype=torch.float64))
        self.double()

    # -- network outputs --
    def latents(self, feats, species):
        e = self.emb(species)
        h = self.atom(torch.cat([feats / self.feat_scale, e], -1))
        h0 = self.atom(torch.cat([torch.zeros_like(feats), e], -1))
        return h, h0

    def confinement(self, h, h0, species=None):
        """(N, 2) scales on the free-ion alpha and C, in c_range, exactly 1 for an isolated ion (u = 0); the map
        u -> log c is smooth with unit slope at 0 and saturates at log c_min / log c_max."""
        u = self.conf(h) - self.conf(h0)
        if species is not None:
            u = u * self.confine_mask[species][:, None]
        lo, hi = self.log_c_lo, self.log_c_hi
        logc = torch.where(u > 0, hi * torch.tanh(u / hi), lo * torch.tanh(u / lo))
        if self.tie_c:      # a uniformly scaled ion: alpha ~ a^3, C ~ a^5, so c_C = c_alpha^(5/3)
            logc = torch.stack([logc[:, 0], logc[:, 0] * 5.0 / 3.0], 1)
        return logc.exp()

    def overlap_gap(self, h, idx, r, ty):
        i, j = idx
        rbf = torch.exp(-((r[:, None] - self.rbf_mu) / self.rbf_w) ** 2)
        onehot = nn.functional.one_hot(ty, self.nT).to(h.dtype)
        z = torch.cat([h[i], h[j], rbf, onehot], -1)
        x = ((r - (self.chan.cutoff - 1.0)) / 1.0).clamp(0, 1)
        cut = 1 - x ** 3 * (10 - 15 * x + 6 * x * x)
        return nn.functional.softplus(self.gap(z)[:, 0]) * cut

    def ct_charges(self, h):
        """(N, 2) donor and acceptor CT charges >= 0 (direct form)."""
        return nn.functional.softplus(self.ctq(h)) / math.log(2.0)

    def ct_exponents(self):
        """(donor, acceptor) Slater exponents (1/bohr) of the direct CT kernel, per species."""
        return nn.functional.softplus(self.ct_bD_raw), nn.functional.softplus(self.ct_bA_raw)

    def direct_ct(self, h, bt, sp):
        """CMM-style direct CT: per-frame energy (B,), per-type sums (B, 4) and the charges (N, 2)."""
        B = bt["B"]
        i, j = bt["ch_index"]
        q = self.ct_charges(h)
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_ch = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])
        e = h.new_zeros(B).index_add(0, bt["ch_frame"], e_ch)
        nT = self.nT
        e_type = h.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], e_ch).reshape(B, nT)
        return e, e_type, e_ch, q

    def pauli_params(self, h, h0, sp):
        """Per ion: Pauli monopole qP (N,), exponent bP (N,), induced Pauli-moment scales kappa (N, 8) and the
        stiffening [s0d, s2d, s0q] (N, 3)."""
        lnq = self.pauli_lnq[sp]
        if self.pauli_nn:
            lnq = lnq + (self.pauli_q(h) - self.pauli_q(h0))[:, 0]
        kap = nn.functional.softplus(self.pauli_kappa_raw)[sp]                 # (N, 2)
        kappa = torch.cat([kap[:, :1].expand(-1, 3), kap[:, 1:].expand(-1, 5)], 1)
        s0 = nn.functional.softplus(self.pauli_s0_raw)[sp]
        s2 = s0[:, 0] * (-1.5 + 4.5 * torch.sigmoid(self.pauli_s2_raw[sp]))
        return lnq.exp(), self.pauli_lnb.exp()[sp], kappa, torch.stack([s0[:, 0], s2, s0[:, 1]], 1)

    def pauli(self, h, h0, bt):
        """The Pauli kernel on a padded batch: frozen energy per frame (B,), the drive g (N, 8) and the matrix the
        kernel adds to the induction Hessian (B, 8 nmax, 8 nmax)."""
        sp, B, nmax, fr = bt["species"], bt["B"], bt["nmax"], bt["frame"]
        qP, bP, kappa, stiff = self.pauli_params(h, h0, sp)
        pairs, e0, g, Bij, Hon = K.pauli_kernel(bt["pos"], fr, qP, bP, kappa, stiff)
        counts = torch.bincount(fr, minlength=B)
        loc = torch.arange(fr.shape[0]) - (torch.cumsum(counts, 0) - counts)[fr]
        ar = torch.arange(8)
        i, j = pairs
        A = h.new_zeros(B, nmax, 8, nmax, 8)
        bi = fr[i][:, None, None]
        A = A.index_put((bi, loc[j][:, None, None], ar[None, :, None], loc[i][:, None, None], ar[None, None, :]),
                        Bij, accumulate=True)
        A = A.index_put((bi, loc[i][:, None, None], ar[None, :, None], loc[j][:, None, None], ar[None, None, :]),
                        Bij.transpose(1, 2), accumulate=True)
        fa = fr[:, None, None]
        la = loc[:, None, None]
        A = A.index_put((fa, la, ar[None, :, None], la, ar[None, None, :]), Hon, accumulate=True)
        e = h.new_zeros(B).index_add(0, fr[i], e0)
        return e, g, A.reshape(B, 8 * nmax, 8 * nmax)

    def pseudo_strength(self, h, idx, r, ty):
        """A_ij >= 0 of each directed pair (i <- j), smoothly zero at PSEUDO_CUTOFF."""
        i, j = idx
        rbf = torch.exp(-((r[:, None] - self.rbf_mu) / self.rbf_w) ** 2)
        onehot = nn.functional.one_hot(ty, len(self.ptypes)).to(h.dtype)
        z = torch.cat([h[i], h[j], rbf, onehot], -1)
        x = ((r - (PSEUDO_CUTOFF - 1.0)) / 1.0).clamp(0, 1)
        cut = 1 - x ** 3 * (10 - 15 * x + 6 * x * x)
        return self.ps_A0[ty] * nn.functional.softplus(self.ps(z)[:, 0]) * cut

    def pseudo_drive(self, h, idx, r, ty, basis, n):
        """(n, 8): sum_j A_ij basis_ij on every ion, and A (P,)."""
        A = self.pseudo_strength(h, idx, r, ty)
        return h.new_zeros(n, 8).index_add(0, idx[0], A[:, None] * basis), A

    # -- the model --
    def forward(self, bt: dict, quadrupoles=True, field=None) -> dict:
        """``field`` (B, 3) a uniform field per frame (au), applied with the network outputs of the field-free
        geometry: it shifts the drive (dipole slots) and every frontier potential by -F . x."""
        sp = bt["species"]
        h, h0 = self.latents(bt["feats"], sp)
        B, nmax, mask = bt["B"], bt["nmax"], bt["mask"]
        n = sp.shape[0]
        A_off, e_pauli = bt["A_off"], h.new_zeros(B)
        Lflat = h.new_zeros(n, 8)
        if self.pol_form == "pauli":
            c = h.new_ones(n, 2)
            e_pauli, g_P, A_P = self.pauli(h, h0, bt)
            Lflat = Lflat + g_P
            A_off = A_off + A_P
        else:
            c = self.confinement(h, h0, sp)
        alpha = torch.ones(B, nmax, dtype=h.dtype).masked_scatter(mask, self.tab.alpha[sp] * c[:, 0])
        Cq = torch.ones(B, nmax, dtype=h.dtype).masked_scatter(mask, self.tab.C[sp] * c[:, 1])
        A_ps = h.new_zeros(0)
        if self.pseudo and bt["ps_index"].shape[1]:
            Lps, A_ps = self.pseudo_drive(h, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], n)
            Lflat = Lflat + Lps
        phi_ext = h.new_zeros(n)
        Fa = h.new_zeros(n, 3)
        if field is not None:
            Fa = field[bt["frame"]]
            Lflat = Lflat - torch.cat([Fa, h.new_zeros(n, 5)], 1)
            phi_ext = -(Fa * bt["pos"] / K.BOHR_ANG).sum(-1)
        L = bt["L"] + h.new_zeros(B, nmax, 8).masked_scatter(mask[..., None].expand(B, nmax, 8), Lflat)
        x0 = K.solve_induction(L, A_off, alpha, Cq, quadrupoles)                # (B, nmax, 8)
        e_pol = 0.5 * (L * x0).sum((1, 2))
        if self.ct_form == "direct":
            nC = bt["ch_index"].shape[1]
            if nC:
                e_ct, e_type, e_ch, q = self.direct_ct(h, bt, sp)
            else:
                e_ct, e_type, e_ch, q = h.new_zeros(B), h.new_zeros(B, self.nT), h.new_zeros(0), h.new_zeros(n, 2)
            z = h.new_zeros(nC)
            return {"e_pol": e_pol, "e_ct": e_ct, "e_ct_post": e_ct, "e_type": e_type, "c": c, "dov": z, "A_ps": A_ps,
                    "W": z, "Delta": z, "delta": z, "p": z, "dq": h.new_zeros(n), "x": x0, "x_pol": x0,
                    "joint_iters": 0, "q_ct": q, "e_ch": e_ch, "stark": z, "e_pauli": e_pauli}
        dov = self.overlap_gap(h, bt["ch_index"], bt["ch_r"], bt["ch_type"])
        if not bt["ch_index"].shape[1]:
            z = h.new_zeros(B)
            return {"e_pol": e_pol, "e_ct": z, "e_ct_post": z, "e_type": h.new_zeros(B, self.nT), "c": c,
                    "dov": dov, "A_ps": A_ps, "W": dov, "Delta": dov, "delta": dov, "p": dov, "dq": h.new_zeros(n),
                    "x": x0, "x_pol": x0, "joint_iters": 0, "e_pauli": e_pauli}
        ct0 = self._ct(x0, bt, sp, mask, phi_ext, dov, Fa)
        out = {"e_pol": e_pol, "c": c, "dov": dov, "A_ps": A_ps, "x_pol": x0, "e_ct_post": ct0["e_ct"],
               "e_pauli": e_pauli}
        if not self.joint:
            return {**out, **ct0, "x": x0, "joint_iters": 0}
        # joint pol + CT: x* = argmin 1/2 x^T A x + L.x + E_CT(x). E_CT depends on x through the frontier potentials
        # (gaps, WKB): its gradient is the field of the CT populations at the induced sites. Fixed point
        # x <- -A^-1 (L + dE_CT/dx) on detached copies; the energy at x* is then differentiated only explicitly
        # (envelope theorem: x* is stationary), so parameter gradients and forces need no implicit term.
        A = K.induction_matrix(bt["A_off"], alpha, Cq).detach()
        lu = torch.linalg.lu_factor(A)
        Ld, dovd = L.detach(), dov.detach()
        x, it = x0.detach(), 0
        for it in range(1, self.joint_max_iter + 1):
            with torch.enable_grad():
                xg = x.clone().requires_grad_(True)
                g = torch.autograd.grad(self._ct(xg, bt, sp, mask, phi_ext.detach(), dovd, Fa.detach())["e_ct"].sum(), xg)[0]
            xn = -torch.linalg.lu_solve(*lu, (Ld + g).reshape(B, -1, 1)).reshape(B, nmax, 8)
            dx = (xn - x).abs().max().item()
            x = xn.detach()
            if dx < self.joint_tol:
                break
        Ax = torch.bmm(K.induction_matrix(bt["A_off"], alpha, Cq), x.reshape(B, -1, 1)).reshape(B, nmax, 8)
        e_quad = (0.5 * Ax + L).mul(x).sum((1, 2))
        ct = self._ct(x, bt, sp, mask, phi_ext, dov, Fa)
        return {**out, **ct, "e_ct": e_quad + ct["e_ct"] - e_pol, "e_ct_chan": ct["e_ct"], "x": x, "joint_iters": it}

    def _ct_density(self, x, bt, sp, mask, phi_ext, dov, Fa):
        """R9 channels, everything from densities and a few global constants:

            Delta = I_D(i) - A_A(j) + s [phibar_D(i) - phibar_A(j)] - J_ij - Stark_ij + Delta^ov_ij
            W     = w_T O(b_D,eff, b_A; R)^gamma_T taper(R)
            ln b_D,eff = ln b_D + lambda_T a/2 tanh(u / a),  u = s [phibar_D(i) - phi(midpoint)] / I_D(i)   ("barrier")
            or, coupling_resp="frontier": O at |R + d_A - d_D|, d = -alpha_orb E (polarizable frontier densities)

        J is the electron-hole Coulomb of the frontier densities; O their overlap. u is the donor level's binding
        relative to the potential at the bond midpoint (every ion but the donor, induced moments and the field: gauge
        invariant), the barrier the electron crosses; the donor density's decay follows it (WKB, sqrt(1 + u) to
        first order, saturating). A uniform field shifts u by F.(R_D - midpoint): the coupling's field response."""
        B, n = bt["B"], sp.shape[0]
        i, j = bt["ch_index"]
        ty = bt["ch_type"]
        s_ = self.chan.s
        phiD = (bt["VD"].sum(-1) + torch.einsum("bijk,bjk->bi", bt["KD"], x))[mask] + phi_ext
        phiA = (bt["VA"].sum(-1) + torch.einsum("bijk,bjk->bi", bt["KA"], x))[mask] + phi_ext
        I_D = self.tab.I_donor[sp[i]]
        A_A = self.acceptor_affinity()[sp[j]]
        Delta = I_D - A_A + s_ * (phiD[i] - phiA[j]) - bt["ch_J"] + dov
        stark = x.new_zeros(Delta.shape)
        if self.ct_pol:
            stark, EA, ED = self.stark_shift(x, bt, sp, mask, Fa)
            Delta = Delta - stark
        if self.coupling_resp == "frontier":
            # polarizable frontier densities: the field (of the CT state, as for the Stark shift) moves the donor and
            # acceptor frontier densities by -alpha E; the coupling follows their overlap at the displaced separation
            # Only the shift along the bond changes the (sigma) overlap: a rigid sideways shift would lower it, while a
            # real perpendicular deformation opens pi channels -- that part is the CT state's Stark lowering.
            R0 = bt["ch_r"] / K.BOHR_ANG
            ehat = (bt["pos"][j] - bt["pos"][i]) / K.BOHR_ANG / R0[:, None]
            dA, dD = self.frontier_displacements(EA, ED, sp, i, j, ehat)
            Reff = R0 + dA - dD
            Reff = 0.5 * (Reff + torch.sqrt(Reff * Reff + 0.01))                      # stays > 0 (smoothly)
            bD = self.tab.front_b["donor"][sp[i]]
            u = Reff - bt["ch_r"] / K.BOHR_ANG                                       # diagnostic: dR (bohr)
        else:
            xc = x[bt["ch_frame"]]                                                    # (C, nmax, 8)
            phi_mid = bt["ch_Vmid"].sum(1) + (bt["ch_Kmid"] * xc).sum((1, 2)) \
                - (Fa[i] * bt["ch_mid"] / K.BOHR_ANG).sum(-1)
            u = s_ * (phiD[i] - phi_mid) / I_D
            a = BARRIER_SAT
            bD = self.tab.front_b["donor"][sp[i]] * torch.exp(self.lam[ty] * 0.5 * a * torch.tanh(u / a))
            Reff = bt["ch_r"] / K.BOHR_ANG
        O = K.shaped_overlap(bD, self.tab.front_w["donor"][sp[i]], self.tab.front_b["acceptor"][sp[j]],
                             self.tab.front_w["acceptor"][sp[j]], Reff)
        W = torch.exp(self.lnw[ty] + self.gamma[ty] * torch.log(O.clamp(min=1e-300)) + bt["ch_ltaper"])
        eps, p, dq = CT.arrowhead(W, Delta, bt["ch_index"], n)
        e_ct = x.new_zeros(B).index_add(0, bt["frame"], eps)
        nT = self.nT
        e_type = x.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + ty, -W / Delta).reshape(B, nT)
        return {"e_ct": e_ct, "e_type": e_type, "W": W, "Delta": Delta, "delta": u, "p": p, "dq": dq,
                "stark": stark, "bD": bD}

    def smoothness_penalty(self, bt, sigma=0.05):
        """Mean squared change of the network outputs (log c, Delta^ov in Eh, log(1 + A/A0)) under a random
        perturbation of the environment features of size sigma (in units of the feature spread), divided by
        sigma^2: a finite-difference input-gradient penalty that keeps the outputs smooth functions of the
        geometry (no spikes at isolated training geometries). Isolated ions (zero features) are left alone."""
        sp = bt["species"]
        f = bt["feats"]
        nz = (f.abs().sum(1, keepdim=True) > 0).to(f.dtype)
        noise = sigma * self.feat_scale * torch.randn_like(f) * nz
        h, h0 = self.latents(f, sp)
        h2, _ = self.latents(f + noise, sp)
        if self.pol_form == "pauli":
            pen = h.new_zeros(())
            if self.pauli_nn:
                pen = ((self.pauli_q(h2) - self.pauli_q(h)) ** 2).mean()
        else:
            pen = ((torch.log(self.confinement(h2, h0, sp)) - torch.log(self.confinement(h, h0, sp))) ** 2).mean()
        if self.ct_form == "direct":
            pen = pen + ((torch.log(self.ct_charges(h2)) - torch.log(self.ct_charges(h))) ** 2).mean()
        elif bt["ch_index"].shape[1]:
            d1 = self.overlap_gap(h, bt["ch_index"], bt["ch_r"], bt["ch_type"])
            d2 = self.overlap_gap(h2, bt["ch_index"], bt["ch_r"], bt["ch_type"])
            pen = pen + ((d2 - d1) ** 2).mean()
        if self.pseudo and bt["ps_index"].shape[1]:
            A0 = self.ps_A0[bt["ps_type"]]
            a1 = self.pseudo_strength(h, bt["ps_index"], bt["ps_r"], bt["ps_type"])
            a2 = self.pseudo_strength(h2, bt["ps_index"], bt["ps_r"], bt["ps_type"])
            pen = pen + ((torch.log1p(a2 / A0) - torch.log1p(a1 / A0)) ** 2).mean()
        return pen / sigma ** 2

    def orbital_polarizabilities(self):
        """(acceptor, donor) frontier-orbital polarizabilities per species (bohr^3), both >= 0 (frontier mode)."""
        return 100 * nn.functional.softplus(self.aorb_acc_raw), 100 * nn.functional.softplus(self.aorb_don_raw)

    def astar(self):
        """(acceptor, donor) coefficients of 1/2 |E|^2 in the CT state's Stark lowering, per species (bohr^3).
        Frontier mode: the transferred electron gains its acceptor orbital's polarizability and loses its donor
        orbital's, so (alpha_A, -alpha_D) -- the same constants that displace the frontier densities."""
        if self.coupling_resp == "frontier":
            a_acc, a_don = self.orbital_polarizabilities()
            return a_acc, -a_don
        return 100 * nn.functional.softplus(self.astar_acc_raw), 100 * self.astar_don

    def frontier_displacements(self, EA, ED, sp, i, j, ehat):
        """Shifts (C,) bohr along the bond (donor -> acceptor unit vector ehat) of the acceptor and donor frontier
        densities in the CT-state fields: an electron density moves against the field, d = -alpha E.ehat, saturating
        smoothly at DISP_MAX. Only the along-bond field enters, so a perpendicular field leaves the overlap alone."""
        a_acc, a_don = self.orbital_polarizabilities()
        vA = -a_acc[sp[j]] * (EA * ehat).sum(-1)
        vD = -a_don[sp[i]] * (ED * ehat).sum(-1)
        sat = lambda v: v / torch.sqrt(1 + (v / DISP_MAX) ** 2)
        return sat(vA), sat(vD)

    def stark_shift(self, x, bt, sp, mask, Fa):
        """The CT state's second-order Stark lowering of every gap (C,), and the fields (C, 3) at the acceptor and
        donor. The fields are those of the CT state's own charge distribution, averaged over the frontier densities
        where the transferred electron and the hole sit: every other ion (permanent density + induced moments), the
        partner in its CT form (its ionic density plus the hole / electron as a frontier density) and the external
        field. No pair is singled out: an isolated Na-Cl pair (both partners neutral in the CT state) feels only
        penetration and the external field."""
        i, j = bt["ch_index"]
        FD = (bt["EVD"].sum(2) + torch.einsum("bijck,bjk->bic", bt["EKD"], x))[mask] + Fa
        FA = (bt["EVA"].sum(2) + torch.einsum("bijck,bjk->bic", bt["EKA"], x))[mask] + Fa
        EA = FA[j] + bt["ch_EP"]
        ED = FD[i] + bt["ch_EP"]
        a_acc, a_don = self.astar()
        u = 0.5 * a_acc[sp[j]] * (EA * EA).sum(-1) + 0.5 * a_don[sp[i]] * (ED * ED).sum(-1)
        return STARK_SAT * torch.tanh(u / STARK_SAT), EA, ED

    def acceptor_affinity(self):
        """A_A per species (Hartree): the free ion's, or the trained effective one where none is bound."""
        return torch.where(self.A_unbound, self.A_free, self.A_fixed)

    def _ct(self, x, bt, sp, mask, phi_ext, dov, Fa):
        """CT channels on the induced moments x (B, nmax, 8): per-frame E_CT (arrowhead roots) and diagnostics."""
        if self.gap_form == "density":
            return self._ct_density(x, bt, sp, mask, phi_ext, dov, Fa)
        B, n = bt["B"], sp.shape[0]
        phiD = (bt["VD"].sum(-1) + torch.einsum("bijk,bjk->bi", bt["KD"], x))[mask] + phi_ext
        phiA = (bt["VA"].sum(-1) + torch.einsum("bijk,bjk->bi", bt["KA"], x))[mask] + phi_ext
        geom = {"index": bt["ch_index"], "r": bt["ch_r"], "g": bt["ch_g"], "lnW": bt["ch_lnW"],
                "dphi_pair": bt["ch_dphi_pair"], "dov_pair": torch.zeros_like(bt["ch_r"])}
        W, Delta, delta = CT.channel_gaps(geom, phiD, phiA, self.tab.I_donor[sp], self.chan.s, ov_add=dov,
                                          pair_overlap=False, wkb=self.wkb)
        stark = x.new_zeros(Delta.shape)
        if self.ct_pol:
            stark = self.stark_shift(x, bt, sp, mask, Fa)[0]
            Delta = Delta - stark
        eps, p, dq = CT.arrowhead(W, Delta, bt["ch_index"], n)
        e_ct = x.new_zeros(B).index_add(0, bt["frame"], eps)
        nT = self.nT
        e_type = x.new_zeros(B * nT).index_add(0, bt["ch_frame"] * nT + bt["ch_type"], -W / Delta).reshape(B, nT)
        return {"e_ct": e_ct, "e_type": e_type, "W": W, "Delta": Delta, "delta": delta, "p": p, "dq": dq,
                "stark": stark}
