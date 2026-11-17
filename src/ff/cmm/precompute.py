"""The cmm precompute: everything of a batch that depends on the geometry (and the fixed species table) alone, from
positions, in one pass over the pairs and one lookup of one table.

:class:`rsfff.ff.cmm.table.CmmTable` is the one radial table: per ordered species pair, the precompute rows of
:class:`rsfff.ff.cmm.tabulate.RadialTable` (permanent drive and induced-induced complements, the diffuse probes'
drive complements and shell overlaps), the exchange-polarization rows of :class:`XpolTable` (frozen overlap and its
derivative maps) and the Gram rows of :class:`rsfff.ff.cmm.gram.GramTable`, on one grid (0.3 .. 12 A, h = 0.01 A).

:func:`batched_precompute` returns the batch dict :class:`rsfff.ff.cmm.CmmModel` takes:

    A (B, nmax D, nmax D)  the induction operator: damped parent-parent couplings plus the on-site blocks (the
                           hyperpolarized dipole response alpha(E_p)^-1, 1 / C, 1 / a0), identity on padding
    phi_front (N, 2)       the other ions' frozen potential at each atom averaged over its donor / acceptor frontier
                           shape (point charges + penetration; the Ewald sum in a cell): the CT gap's electrostatics
    Lz (B, nmax, D)        the drive: the permanent field (and gradient) minus the Legendre offset h, the breathing
                           drive -V0_pen of the breath probe; e_hyp (B,) the Legendre constants
    chan (N, 17)           the encoder's raw two- and three-body channels
    gated (N, 15)          the raw gated diffuse-probe invariants (S0, V0_pen, |V1g|^2, |V2g|^2, V1g . S1 per probe)
    x_sel, x_rel, x_swO,   the exchange pairs (within R_XPOL): their index into ``pairs``, the overlap-relative
    x_Gi, x_Gj             derivative rows d ln O / dz of both sides (2P, D), sw / O, the Gram blocks Gt(i <- j), Gt(j <- i)
    Ca (B, nmax D, nmax)   the coupling of a unit charge in each atom's slot to the site variables; Jq (B, nmax, nmax)
                           the slots' Coulomb with the shape self-energy on the diagonal
    ch_*                   the CT channels (directed pairs within the channel cutoff) and their taper
    e_elst (B,)            the frozen electrostatics
    pairs, pair_dr         every pair (clusters) or every pair and image within ``cutoff`` (periodic)

Pairs beyond the table's reach (clusters) enter the drive, the operator and the slots as bare multipoles and are
skipped everywhere else. Periodic cells (``cell``): the pairs are every pair and image within the cutoff, and the
operator, slots and Ewald frames go to ``bt["pbc"]`` for the matrix-free solve (``batched_precompute``). Everything is differentiable in the positions (forces, stress).
"""
from __future__ import annotations

import math

import torch

from . import kernels as K
from .batch import channel_types
from .channels import channels_from_pairs, overlap_switch, pair_electrostatics
from .elst import permanent_electrostatics
from .gram import gram_blocks_from_rows, gram_layout
from .params import ChannelTable, IonTable, channel_taper
from .periodic import (dispersion_tail_integral, erfc_screening, ewald_alpha, make_ewald_frames, periodic_pairs,
                       recip_apply, smooth_switch)
from .table import CmmTable, cmm_table
from .tabulate import BOHR, GATE, NORD, O_ON, PROBES, R_XPOL, _basis_tensors

__all__ = ["CmmTable", "batched_precompute", "default_cutoff", "hyperpolarization", "cmm_table", "smoothstep_log",
           "short_range_reach"]


def smoothstep_log(O, on, full):
    """Smoothstep in log10 O from 0 at ``on`` to 1 at ``full`` (O <= 0 counts as 0)."""
    lo, hi = math.log10(on), math.log10(full)
    t = ((torch.log10(O.clamp(min=1e-300)) - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


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


def hyperpolarization(Lp, sp, tab: IonTable):
    """The Legendre hyperpolarization: with the free-ion field energy W(E) = -1/2 a|E|^2 - gamma |E|^4 / 24 - b E T E
    and its polarized dipole mu_p, the ion's internal energy expanded to second order about mu_p is the solve's
    quadratic form with alpha(E) = a + gamma/6 (|E|^2 1 + 2 E E^T) + 2 b T on site (returned inverted), a field
    offset h = alpha(E)^-1 mu_p - E and a constant c = W + 1/2 mu_p alpha(E)^-1 mu_p."""
    E, T = _hyper_ET(Lp)
    eye = torch.eye(3, dtype=E.dtype, device=E.device)
    a = tab.alpha[sp][:, None, None] * eye
    g, bb = tab.hyper_gamma[sp], tab.hyper_b[sp]
    e2 = (E * E).sum(-1)
    TE = (T @ E[:, :, None])[..., 0]
    aE = a + (g / 6)[:, None, None] * (e2[:, None, None] * eye + 2 * E[:, :, None] * E[:, None, :]) \
        + (2 * bb)[:, None, None] * T
    mu = (a @ E[:, :, None])[..., 0] + (g / 6 * e2)[:, None] * E + (2 * bb)[:, None] * TE
    W = -0.5 * (E * (a @ E[:, :, None])[..., 0]).sum(-1) - g * e2 * e2 / 24 - bb * (E * TE).sum(-1)
    ainvE = torch.linalg.inv(aE)
    aimu = (ainvE @ mu[:, :, None])[..., 0]
    return ainvE, aimu - E, W + 0.5 * (mu * aimu).sum(-1)


def _memo(table, name, dev, make):
    memo = table.__dict__.setdefault("_memo", {})
    key = (name, str(dev))
    if key not in memo:
        memo[key] = make()
    return memo[key]


def _channel_lut(tab, chan):
    """(S, S) channel type index of each ordered species pair (-1 where the pair has no channel)."""
    S = len(tab.symbols)
    ctypes = channel_types(tab.symbols)
    ch = torch.full((S, S), -1, dtype=torch.long)
    for a in range(S):
        for b in range(S):
            nm = f"{tab.symbols[a]}>{tab.symbols[b]}"
            if nm in chan.types and nm in ctypes:
                ch[a, b] = ctypes.index(nm)
    return ch


def _species(table, species):
    S = table.built.shape[0]
    sp = range(S) if species is None else sorted({int(x) for x in torch.as_tensor(species).reshape(-1).tolist()})
    table.ensure_species(sp)
    return torch.tensor(list(sp), dtype=torch.long)


def _reach(table, names, thr, species=None) -> float:
    sp = _species(table, species)
    cols = [table.index[k] for k in names]
    on = table.coef[sp.to(table.device)][:, sp.to(table.device)][:, :, cols, :, 0].cpu() > thr
    hit = on.any(0).any(0).any(0)
    return float(table.r.cpu()[hit].max()) if bool(hit.any()) else 0.0


def gate_reach(table: CmmTable, gate=GATE, species=None) -> float:
    """(A) the largest distance at which any probe's overlap with any of the species (indices; None: all) reaches
    the gate's on-value."""
    return _reach(table, [f"envS:{p}:0" for p in PROBES], gate[0], species)


def short_range_reach(table: CmmTable, chan: ChannelTable, gate=GATE, species=None) -> float:
    """(A) the largest distance at which a short-range term of the model is on among the species (indices; None:
    all): the CT channels, the probe gates and the density overlap of the channels and the exchange pairs (above its
    switch-on value)."""
    return max(chan.cutoff, gate_reach(table, gate, species), _reach(table, ["xp:ovl"], O_ON, species))


def default_cutoff(table: CmmTable, chan: ChannelTable, gate=GATE, switch_width=1.0, species=None) -> float:
    """The periodic pair cutoff (A) for the species (indices; None: all): the short-range reach plus the switch
    width, rounded up to 0.5 A, so the switch-off of the table's complements leaves every short-range term exactly
    as in a cluster."""
    return math.ceil(2 * (short_range_reach(table, chan, gate, species) + switch_width)) / 2


def batched_precompute(pos_ang, atomic_numbers, batch_idx, tab: IonTable, table: CmmTable, chan: ChannelTable,
                       labels: dict | None = None, cell=None, cutoff: float | None = None, switch_width: float = 1.0,
                       ewald_tol: float = 1e-10, dispersion_tail: bool = False, gate=GATE) -> dict:
    """The batch of :class:`rsfff.ff.cmm.CmmModel` from positions (N, 3) Angstrom (autograd flows through), atomic
    numbers (N,) and the frame index (N,) sorted by frame (module docstring). ``labels``: per-frame tensors (B,) to
    attach. Runs on the device of ``pos_ang``; ``tab`` and ``table`` must live there too.

    ``cell`` (3, 3) or (B, 3, 3) Angstrom, rows the lattice vectors, makes the frames periodic: the pairs are every
    pair and image within ``cutoff`` (default :func:`default_cutoff`), the table's short-range complements are switched
    off over the last
    ``switch_width`` and the bare Coulomb part of the drive is the Ewald sum (:mod:`rsfff.ff.cmm.periodic`). The
    operator is then not assembled: the on-site blocks, the real-space pair blocks of [slot charge, dipole,
    quadrupole] and the Ewald frames go into ``bt["pbc"]`` for the matrix-free joint solve
    (:class:`rsfff.ff.cmm.periodic.CmmOperator`). ``gate``: the probe overlap window of the gated invariants (the
    model's ``env_gate``)."""
    if tuple(table.probes) != PROBES:
        raise ValueError(f"the table's probes {table.probes} are not the cmm probes {PROBES}")
    X = pos_ang / BOHR
    dev, dt = X.device, X.dtype
    N = X.shape[0]
    b_idx = torch.as_tensor(batch_idx, device=dev)
    B = int(b_idx.max()) + 1 if N else 0
    s = tab.species_index(torch.as_tensor(atomic_numbers, device=dev))
    counts = torch.bincount(b_idx, minlength=B)
    loc = torch.arange(N, device=dev) - (torch.cumsum(counts, 0) - counts)[b_idx]
    nmax = int(counts.max()) if N else 0
    P = K._poly_map(dt, dev)
    has_b = tab.a0[s] > 0
    n_b = 1 if bool((tab.a0 > 0).any()) else 0
    D = 8 + n_b
    periodic = cell is not None
    if periodic:
        cell_ang = torch.as_tensor(cell, dtype=dt, device=dev)
        cell_ang = cell_ang.expand(B, 3, 3) if cell_ang.dim() == 2 else cell_ang
        spk = tuple(sorted(set(s.tolist())))
        reach = _memo(table, ("reach", tuple(gate), chan.cutoff, spk), "cpu",
                      lambda: short_range_reach(table, chan, gate, spk))
        rc = float(cutoff) if cutoff is not None else math.ceil(2 * (reach + switch_width)) / 2
        if rc - switch_width < reach:
            raise ValueError(f"cutoff - switch_width = {rc - switch_width} A is inside the short-range terms' reach "
                             f"{reach:.2f} A (channels, probe gates, overlaps)")
        pairs, shifts = periodic_pairs(pos_ang, cell_ang, b_idx, rc)
        Cb = cell_ang / BOHR
        dr_u = X[pairs[1]] - X[pairs[0]] + torch.einsum("pk,pkl->pl", shifts, Cb[b_idx[pairs[0]]])
    else:
        pairs = K.all_pairs(b_idx)
        dr_u = X[pairs[1]] - X[pairs[0]]
    r_u = dr_u.norm(dim=-1)
    if periodic:                       # every image pair within the cutoff is inside the table
        pn = torch.arange(pairs.shape[1], device=dev)
        pf = pn[:0]
    else:
        near = r_u * BOHR < table.r_max
        pn, pf = near.nonzero().squeeze(1), (~near).nonzero().squeeze(1)
    nP = pn.shape[0]
    # -- the pairs inside the table: directed (i <- j), dr = X_j - X_i ---------------------------------------------
    dp = torch.cat([pairs[:, pn], pairs[:, pn].flip(0)], 1)
    i, j = dp
    dr = torch.cat([dr_u[pn], -dr_u[pn]])
    r = torch.cat([r_u[pn], r_u[pn]])
    r_ang = r * BOHR
    rhat = dr / r[:, None]
    y = table(s[i], s[j], r_ang)                                     # (Pd, F): the one lookup
    qj = tab.charge[s[j]]
    if periodic:
        y = y * smooth_switch(r_ang, rc - switch_width, rc)[:, None]
        alpha = ewald_alpha(rc / BOHR, ewald_tol)
        g = erfc_screening(r, alpha)                                 # (NORD, Pd): the real-space Ewald screening
        frames = make_ewald_frames(X, Cb, b_idx, alpha, 2.2 * alpha * math.sqrt(-math.log(ewald_tol)))
        Mq = X.new_zeros(N, 10)
        Mq[:, 0] = tab.charge[s]
        out_q = recip_apply(frames, Mq)                              # (N, 10): reciprocal + self + background
        rq = out_q @ P
    else:
        g = torch.ones(1, r.shape[0], dtype=dt, device=dev)
    qg = qj[None] * g
    Tb = _basis_tensors(dr)                                          # (NORD, Pd, 10, 10)
    T0 = Tb[:, :, 0, :]                                              # row 0: the source as a charge
    T0u = T0.sum(0)                                                  # the bare tensor (all damping factors 1)
    # the pairs beyond the table (clusters): bare multipoles
    fi, fj = pairs[:, pf]
    di, dj = torch.cat([fi, fj]), torch.cat([fj, fi])
    drf = torch.cat([dr_u[pf], -dr_u[pf]])
    Tf = K._tensor(drf, torch.ones(NORD, drf.shape[0], dtype=dt, device=dev)) if pf.numel() else None
    # -- permanent drive (+ probes' drive complements) -------------------------------------------------------------
    rows = table.rows(y, ["drive:dip", "drive:quad"] + [f"envV:{p}" for p in PROBES])   # (5, NORD, Pd)
    R0 = torch.einsum("knp,npb->kpb", qg[None] - rows, T0)
    Rp = R0 @ P                                                      # (5, Pd, 8)
    Lflat = X.new_zeros(N, 8).index_add(0, i, torch.cat([Rp[0, :, :3], Rp[1, :, 3:]], 1))
    if Tf is not None:
        Lflat = Lflat.index_add(0, di, (tab.charge[s[dj]][:, None] * Tf[:, 0, :]) @ P)
    if periodic:
        Lflat = Lflat + rq
    # -- induced-induced coupling (undirected; rows the source jj, columns the receiver ii, aoff:{ii parent}{jj parent})
    arows = table.rows(y[:nP], ["aoff:dd", "aoff:dq", "aoff:qd", "aoff:qq"])
    if periodic:                      # [slot charge, dipole, quadrupole]: the charge lives in the dipole parent
        from .periodic import poly_map9
        P9 = poly_map9(dt, dev)
        PTP = torch.einsum("ab,npbc,cd->npad", P9.T, Tb[:, :nP], P9)                       # (NORD, nP, 9, 9)
        blocks = torch.einsum("knp,npab->kpab", g[:, :nP] - arows, PTP)
        Bij = torch.cat([torch.cat([blocks[0][:, :4, :4], blocks[2][:, :4, 4:]], 2),
                         torch.cat([blocks[1][:, 4:, :4], blocks[3][:, 4:, 4:]], 2)], 1)   # (nP, 9, 9)
    else:
        PTP = torch.einsum("ab,npbc,cd->npad", P.T, Tb[:, :nP], P)       # (NORD, nP, 8, 8)
        blocks = torch.einsum("knp,npab->kpab", g[:, :nP] - arows, PTP)  # (4, nP, 8, 8): dd, dq, qd, qq
        Bij = torch.cat([torch.cat([blocks[0][:, :3, :3], blocks[2][:, :3, 3:]], 2),
                         torch.cat([blocks[1][:, 3:, :3], blocks[3][:, 3:, 3:]], 2)], 1)
    ii, jj = pairs[:, pn]
    if Tf is not None:
        Bij = torch.cat([Bij, P.T @ Tf[:pf.shape[0]] @ P])
        ii, jj = torch.cat([ii, fi]), torch.cat([jj, fj])
    # -- on-site: hyperpolarized dipole, quadrupole, breathing; the drive's offset and the breathing drive ----------
    ainv, hyp_h, hyp_c = hyperpolarization(Lflat, s, tab)
    Cs = tab.C[s]
    on = X.new_zeros(N, D, D)
    on[:, :3, :3] = ainv
    on[:, 3:8, 3:8] = torch.diag_embed(torch.where(Cs > 0, 1.0 / Cs.clamp(min=1e-30), torch.full_like(Cs, 1e12))[:, None]
                                       .expand(N, 5))
    # -- environment: the probes' penetration potentials and fields, overlaps; the gated invariants ---------------
    G_ = len(PROBES)
    v0 = (R0[2:, :, 0] - (qg[0] / r)[None]).T                        # (Pd, 3): the penetration potential
    Dpen = torch.einsum("knp,npb->kpb", -rows[2:], T0) @ P          # (3, Pd, 8)
    Vpen = X.new_zeros(N, G_, 8).index_add(0, i, torch.cat([-Dpen[..., :3], Dpen[..., 3:]], 2).permute(1, 0, 2))
    S0p = y[:, table.cols(tuple(f"envS:{p}:0" for p in PROBES), dev)]                  # (Pd, 3)
    S1p = y[:, table.cols(tuple(f"envS:{p}:1" for p in PROBES), dev)]
    V0 = X.new_zeros(N, G_).index_add(0, i, v0)
    # the other ions' frozen potential at every atom, averaged over its donor / acceptor frontier shape: the point
    # charges (bare in a cluster, the Ewald sum in a cell) plus the probes' penetration potentials (the CT gap)
    phi_pt = X.new_zeros(N).index_add(0, i, qg[0] / r)
    if pf.numel():
        phi_pt = phi_pt.index_add(0, di, tab.charge[s[dj]] / drf.norm(dim=-1))
    if periodic:
        phi_pt = phi_pt + out_q[:, 0]
    phi_front = phi_pt[:, None] + V0[:, [PROBES.index("donor"), PROBES.index("acceptor")]]
    S0 = X.new_zeros(N, G_).index_add(0, i, S0p)
    S1 = X.new_zeros(N, G_, 3).index_add(0, i, -S1p[:, :, None] * rhat[:, None, :])
    Dq = (qj[:, None] * T0u) @ P                                     # the neighbour's bare point-charge drive
    sig = smoothstep_log(S0p, *gate)                                 # (Pd, 3)
    Vg = X.new_zeros(N, G_, 8).index_add(0, i, sig[:, :, None] * torch.cat([-Dq[:, :3], Dq[:, 3:]], 1)[:, None, :])
    has = torch.ones(N, G_, dtype=dt, device=dev)
    has[:, PROBES.index("breath")] = has_b.to(dt)
    V0, S0, S1 = V0 * has, S0 * has, S1 * has[:, :, None]
    V = (Vpen + Vg) * has[:, :, None]
    V1 = V[..., :3]
    gated = torch.stack([S0, V0, (V1 * V1).sum(-1), (V[..., 3:] ** 2).sum(-1), (V1 * S1).sum(-1)], -1).reshape(N, G_ * 5)
    # -- the drive in the solve's layout -------------------------------------------------------------------------
    Lfull = torch.cat([Lflat[:, :3] - hyp_h, Lflat[:, 3:]], 1)
    if n_b:
        Lfull = torch.cat([Lfull, -V0[:, PROBES.index("breath"), None]], 1)
        on[:, 8, 8] = torch.where(has_b, 1.0 / tab.a0[s].clamp(min=1e-30), torch.ones_like(tab.a0[s]))
    Lz = X.new_zeros(B, nmax, D).index_put((b_idx, loc), Lfull)
    e_hyp = X.new_zeros(B).index_add(0, b_idx, hyp_c)
    # -- exchange pairs: overlap derivative maps, Gram blocks; the encoder's channels -----------------------------
    xs = (r_u[pn] * BOHR < R_XPOL).nonzero().squeeze(1)              # into the table pairs
    Px = xs.shape[0]
    dsel = torch.cat([xs, xs + nP])
    yx = y[dsel]
    O = yx[:, table.index["xp:ovl"]]
    lam = table.rows(yx, ["xp:dip", "xp:quad"])                      # (2, NORD, 2Px)
    Tx = Tb[:, dsel]
    Td = torch.einsum("np,npa->pa", lam[0], Tx[:, :, 1:4, 0])
    Tq = torch.einsum("np,npa->pa", lam[1], Tx[:, :, 4:10, 0]) @ P[4:10, 3:8]
    sw = overlap_switch(O)
    eb = yx[:, table.index["xp:breath"]] - O / tab.shell_n.sum(1)[s[j[dsel]]]   # the breathing density has no monopole
    swO = sw / O.clamp(min=O_ON)
    x_rel = (torch.cat([-Td, -Tq, eb[:, None]], 1)[:, :D]) * swO[:, None]
    gcols = table.cols(tuple(f"gram:{a}{b}{L}" for a, b, L in gram_layout("shape")[1]), dev)
    x_Gi = gram_blocks_from_rows(yx[:Px, gcols], dr[xs])[:, :D, :D]
    x_Gj = gram_blocks_from_rows(yx[Px:, gcols], -dr[xs])[:, :D, :D]
    w = sw * O.clamp(min=0.0)
    E = pair_electrostatics(s[i[dsel]], s[j[dsel]], r[dsel], tab)
    chan_raw = channels_from_pairs(N, i[dsel], dr[dsel], w, E * w)
    # -- CT channels -------------------------------------------------------------------------------------------
    lut = _memo(table, ("lut", tuple(chan.types)), dev, lambda: _channel_lut(tab, chan).to(dev))
    ct = lut[s[i], s[j]]
    ck = (ct >= 0) & (r_ang < chan.cutoff)
    ch_index = dp[:, ck]
    ch_r = r_ang[ck]
    kk = ck.nonzero().squeeze(1)
    pos_of = torch.full((2 * nP,), -1, dtype=torch.long, device=dev).index_put((kk,), torch.arange(kk.shape[0], device=dev))
    ch_rev = pos_of[(kk + nP) % max(2 * nP, 1)]                     # the reverse channel j -> i (same image), or -1
    ch_J = (1 - y[ck, table.index["aoff:dd:0"]]) / r[ck]             # the slots' Coulomb at the channel's distance
    sh = torch.cat([shifts[pn], -shifts[pn]]) if periodic else torch.zeros(2 * nP, 3, dtype=dt, device=dev)
    code = ((sh[ck] + 64).long() * torch.tensor([16384, 128, 1], device=dev)).sum(1)
    ch_key = (ch_index[0] * N + ch_index[1]) * (1 << 21) + code       # the channel's identity across MD steps
    lab = dict(labels or {})
    lab.setdefault("n", counts.to(dt))
    lab.setdefault("ct_mask", torch.ones(B, dtype=torch.bool, device=dev))
    for key in ("pol", "ct", "pauli", "disp", "elec", "int"):
        lab.setdefault(key, torch.full((B,), float("nan"), dtype=dt, device=dev))
    mask = torch.zeros(B, nmax, dtype=torch.bool, device=dev).index_put((b_idx, loc), torch.ones(N, dtype=torch.bool, device=dev))
    bt = {"B": B, "nmax": nmax, "D": D, "mask": mask, "species": s, "frame": b_idx, "pos": pos_ang, "pairs": pairs,
          "pair_dr": dr_u * BOHR, "Lz": Lz, "e_hyp": e_hyp, "chan": chan_raw, "gated": gated, "gate": tuple(gate),
          "x_sel": pn[xs], "x_rel": x_rel, "x_swO": swO[:Px], "x_Gi": x_Gi, "x_Gj": x_Gj,
          # the exchange pairs' acceptor-probe overlaps: [:Px] i's probe with j, [Px:] j's probe with i
          "x_Sa": S0p[dsel, PROBES.index("acceptor")],
          "ch_index": ch_index, "ch_type": ct[ck], "ch_r": ch_r, "ch_frame": b_idx[ch_index[0]], "ch_rev": ch_rev,
          "ch_J": ch_J, "ch_key": ch_key, "ch_rhat": dr[ck] / r[ck][:, None],     # unit vector donor -> acceptor
          # the channel partner's share of the donor (i) / acceptor (j) probe overlaps, and the atoms' totals: the
          # confinement of a channel by everything but its own partner is S0_atom[i, 0] - ch_Sd (donor side) and
          # S0_atom[j, 1] - ch_Sa (acceptor side)
          "ch_Sd": S0p[kk, PROBES.index("donor")], "ch_Sa": S0p[(kk + nP) % max(2 * nP, 1), PROBES.index("acceptor")],
          "S0_atom": S0[:, [PROBES.index("donor"), PROBES.index("acceptor")]], "phi_front": phi_front,
          "ch_ltaper": torch.log(channel_taper(ch_r, chan.cutoff, chan.taper).clamp(min=1e-300)), "labels": lab}
    if periodic:
        bt.update(A=None, Ca=None, Jq=None, pbc={
            "cell": cell_ang, "volume": torch.linalg.det(Cb).abs(), "cutoff": rc, "switch_width": switch_width,
            "alpha": alpha, "frames": frames, "A_pairs": pairs, "blocks9": Bij, "onsite": on, "jself": table.jself[s],
            "e_elst_recip": 0.5 * X.new_zeros(B).index_add(0, b_idx, Mq[:, 0] * out_q[:, 0]), "g0_pairs": g[0, :nP],
            "disp_tail": dispersion_tail_integral(rc, switch_width) if dispersion_tail else None})
        bt["e_elst"] = permanent_electrostatics(pos_ang, s, b_idx, tab, pairs, dr_u * BOHR, bt["pbc"])
        return bt
    # -- the induction operator, assembled in the solve's layout -----------------------------------------------------
    arD = torch.arange(D, device=dev)
    ar8 = arD[:8]
    A = X.new_zeros(B, nmax, D, nmax, D)
    fb = b_idx[ii][:, None, None]
    A = A.index_put((fb, loc[jj][:, None, None], ar8[None, :, None], loc[ii][:, None, None], ar8[None, None, :]), Bij,
                    accumulate=True)
    A = A.index_put((fb, loc[ii][:, None, None], ar8[None, :, None], loc[jj][:, None, None], ar8[None, None, :]),
                    Bij.transpose(1, 2), accumulate=True)
    pb, pl = (~mask).nonzero(as_tuple=True)
    on_all = torch.cat([on, torch.eye(D, dtype=dt, device=dev).expand(pb.shape[0], D, D)])
    ob, ol = torch.cat([b_idx, pb]), torch.cat([loc, pl])
    A = A.index_put((ob[:, None, None], ol[:, None, None], arD[None, :, None], ol[:, None, None], arD[None, None, :]),
                    on_all, accumulate=True)
    # -- the charge slots: unit charges in the dipole-parent shapes ------------------------------------------------
    srows = 1 - table.rows(y, ["aoff:dd", "aoff:qd"])                 # (2, NORD, Pd)
    Rs = torch.einsum("knp,npb->kpb", srows, T0)
    Rsp = Rs @ P
    Kp, Jp, a_, b_ = torch.cat([Rsp[0, :, :3], Rsp[1, :, 3:]], 1), Rs[0, :, 0], i, j
    if Tf is not None:
        Kp = torch.cat([Kp, Tf[:, 0, :] @ P])
        Jp = torch.cat([Jp, Tf[:, 0, 0]])
        a_, b_ = torch.cat([a_, di]), torch.cat([b_, dj])
    Kd = X.new_zeros(B, nmax, nmax, 8).index_put((b_idx[a_], loc[a_], loc[b_]), Kp)
    Ca = Kd.permute(0, 1, 3, 2)
    if D > 8:
        Ca = torch.cat([Ca, Ca.new_zeros(B, nmax, D - 8, nmax)], 2)
    Jq = X.new_zeros(B, nmax, nmax).index_put((b_idx[a_], loc[a_], loc[b_]), Jp).index_put(
        (b_idx, loc, loc), table.jself[s])
    bt.update(A=A.reshape(B, nmax * D, nmax * D), Ca=Ca.reshape(B, nmax * D, nmax), Jq=Jq, pbc=None,
              e_elst=permanent_electrostatics(pos_ang, s, b_idx, tab, pairs, dr_u * BOHR))
    return bt
