"""Per-frame geometry caches of the cmm trainer and their collation into a batch.

:func:`precompute` evaluates, once per labelled frame and without autograd, everything of a frame the model reads
that depends on the geometry only: the induction drive and operator, the CT channel list with its taper, the
environment invariants and their equivariant blocks. :func:`collate` pads a list of them into the batch dict
:class:`rsfff.ff.cmm.CmmModel` takes -- the same keys :func:`rsfff.ff.cmm.tabulate.batched_precompute` produces
(force frames, MD), which computes the same quantities from the tabulated radial functions, differentiably.

(Moved from ``rsfff.ff.cmm.model``; the channel CT model's frontier potentials, fields, midpoint potentials,
pair functions and the Pauli pseudo-potential drive are gone.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import kernels as K
from .params import ATOMIC_NUMBER, ChannelTable, IonTable, channel_taper

__all__ = ["FrameCache", "precompute", "collate", "channel_types", "channel_list", "CHANNEL_TYPES"]

CHANNEL_TYPES = ("Cl>Na", "Na>Cl", "Cl>Cl", "Na>Na")       # the NaCl legacy order (cached labels and runs)


def channel_types(symbols):
    """Directed channel type names 'D>A' for a species list: the legacy NaCl order for {Na, Cl}, every ordered pair
    otherwise."""
    if set(symbols) == {"Na", "Cl"}:
        return CHANNEL_TYPES
    return tuple(f"{a}>{b}" for a in symbols for b in symbols)


def channel_list(pos_ang, species, batch_idx, tab: IonTable, chan: ChannelTable):
    """Directed CT channels (i -> j) of every frame inside the cutoff: index (2, C), r (C,) A, type (C,), log taper."""
    pairs = K.all_pairs(batch_idx)
    dp = torch.cat([pairs, pairs.flip(0)], dim=1)
    r = (pos_ang[dp[1]] - pos_ang[dp[0]]).norm(dim=-1)
    ctypes = channel_types(tab.symbols)
    S = len(tab.symbols)
    lut = torch.full((S, S), -1, dtype=torch.long)
    for a in range(S):
        for b in range(S):
            nm = f"{tab.symbols[a]}>{tab.symbols[b]}"
            if nm in chan.types and nm in ctypes:
                lut[a, b] = ctypes.index(nm)
    ty = lut.to(dp.device)[species[dp[0]], species[dp[1]]]
    keep = (ty >= 0) & (r < chan.cutoff)
    r = r[keep]
    return dp[:, keep], r, ty[keep], torch.log(channel_taper(r, chan.cutoff, chan.taper).clamp(min=1e-300))


@dataclass
class FrameCache:
    species: torch.Tensor        # (N,)
    pos: torch.Tensor            # (N, 3) A
    L: torch.Tensor              # (N, 8) induction drive
    A_off: torch.Tensor          # (8N, 8N) induced-induced coupling
    ch_index: torch.Tensor       # (2, C)
    ch_type: torch.Tensor        # (C,) index into channel_types(symbols)
    ch_r: torch.Tensor           # (C,) A
    ch_ltaper: torch.Tensor      # (C,) log of the channel cutoff taper
    feats: torch.Tensor          # (N, F) environment invariants (zero for an isolated ion)
    env: dict = None             # equivariant environment blocks, "<probe>:<V0|V1|V2|S0|S1|S2>" -> (N, ...)
    labels: dict = field(default_factory=dict)   # 'pol', 'ct', ... (Hartree), masks
    meta: dict = field(default_factory=dict)


def precompute(symbols, pos_ang, tab: IonTable, chan: ChannelTable, featurizer, labels=None, meta=None,
               differentiable=False) -> FrameCache:
    """The geometry-dependent tensors of one frame (no graph unless ``differentiable``). ``featurizer``: a
    :class:`rsfff.ff.cmm.environment.DensityEnvironment`; its equivariant blocks are kept on ``env``."""
    with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
        pos = torch.as_tensor(pos_ang, dtype=torch.float64)
        Z = torch.tensor([ATOMIC_NUMBER[s] for s in symbols])
        s = tab.species_index(Z)
        b = torch.zeros(len(symbols), dtype=torch.long)
        L, A_off = K.induction_operator(pos, s, b, tab)
        idx, r, ty, lt = channel_list(pos, s, b, tab, chan)
        ef = featurizer(positions=pos, atomic_numbers=Z, batch_idx=b)
        env = {f"{p}:{k}": v.to(torch.float64) for p, blk in ef.blocks.items() for k, v in blk.items()} if ef.blocks else None
        return FrameCache(s, pos, L, A_off, idx, ty, r, lt, ef.inv_feats.to(torch.float64), env,
                          labels=dict(labels or {}), meta=dict(meta or {}))


LABEL_KEYS = ("pol", "ct", "pauli", "disp", "elec", "int")


def collate(frames: list[FrameCache]) -> dict:
    """Pad the per-frame matrices to the largest frame; flatten atoms and channels with offsets."""
    B = len(frames)
    nmax = max(f.species.shape[0] for f in frames)
    dt = frames[0].L.dtype
    L = torch.zeros(B, nmax, 8, dtype=dt)
    A = torch.zeros(B, 8 * nmax, 8 * nmax, dtype=dt)
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    species, feats, frame_of, pos, idx, ty, r, lt, ch_frame = [], [], [], [], [], [], [], [], []
    off = 0
    for k, f in enumerate(frames):
        n = f.species.shape[0]
        L[k, :n] = f.L
        A[k, :8 * n, :8 * n] = f.A_off
        mask[k, :n] = True
        species.append(f.species)
        feats.append(f.feats)
        pos.append(f.pos)
        frame_of.append(torch.full((n,), k, dtype=torch.long))
        idx.append(f.ch_index + off)
        ty.append(f.ch_type)
        r.append(f.ch_r)
        lt.append(f.ch_ltaper)
        ch_frame.append(torch.full((f.ch_r.shape[0],), k, dtype=torch.long))
        off += n
    env = {k: torch.cat([f.env[k] for f in frames]) for k in frames[0].env} if all(f.env for f in frames) else None
    lab = {key: torch.tensor([f.labels.get(key, float("nan")) for f in frames], dtype=dt) for key in LABEL_KEYS}
    lab["ct_mask"] = torch.tensor([bool(f.labels.get("ct_mask", True)) for f in frames])
    lab["n"] = torch.tensor([f.species.shape[0] for f in frames], dtype=dt)
    return {"B": B, "nmax": nmax, "L": L, "A_off": A, "mask": mask, "species": torch.cat(species),
            "feats": torch.cat(feats), "frame": torch.cat(frame_of), "pos": torch.cat(pos),
            "ch_index": torch.cat(idx, 1), "ch_type": torch.cat(ty), "ch_r": torch.cat(r),
            "ch_ltaper": torch.cat(lt), "ch_frame": torch.cat(ch_frame), "labels": lab, "env": env}
