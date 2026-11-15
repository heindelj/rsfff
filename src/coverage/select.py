"""Candidate pools against a :class:`~rsfff.coverage.hull.FeatureHull`: features, a physical guard, scoring and greedy
hull extension.

Greedy hull extension picks, one frame at a time, the candidate whose farthest atom is farthest outside the current
hull, then adds that frame's atoms to the hull before the next pick, so near-duplicates of a picked frame drop out.
Distances only shrink as the hull grows, so a stale score is an upper bound and the selection is lazy (CELF): only
the frame on top of the heap is re-scored. A pure hull criterion prefers broken geometries (collapsed or detached
atoms have enormous invariants), so candidates pass :class:`Guard` first.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .hull import INSIDE_TOL, FeatureHull, hull_distance

__all__ = ["IONIC_RADII", "Guard", "featurizer_for_params", "frame_features", "score_frames", "greedy_extend"]

#: Shannon ionic radii (Angstrom, CN 6) for the guard's contact scale; other elements fall back to ASE covalent radii
IONIC_RADII = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}


def _radius(sym):
    if sym in IONIC_RADII:
        return IONIC_RADII[sym]
    from ase.data import atomic_numbers, covalent_radii
    return float(covalent_radii[atomic_numbers[sym]])


@dataclass
class Guard:
    """Rejects geometries no label should be spent on: like atoms closer than ``like`` x (r_i + r_j), unlike atoms
    closer than ``unlike`` x (r_i + r_j), or an atom farther than ``detached`` Angstrom from every other atom."""
    like: float = 0.8
    unlike: float = 0.6
    detached: float = 6.0

    def __call__(self, symbols, pos) -> bool:
        return self.reason(symbols, pos) is None

    def reason(self, symbols, pos):
        """None if the frame passes, else a short reason."""
        pos = np.asarray(pos, float)
        n = len(pos)
        if n < 2:
            return None
        R = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        np.fill_diagonal(R, np.inf)
        r = np.array([_radius(s) for s in symbols])
        rr = r[:, None] + r[None]
        same = np.array(symbols)[:, None] == np.array(symbols)[None]
        if (R[same] < self.like * rr[same]).any():
            return "like contact"
        if (R[~same] < self.unlike * rr[~same]).any():
            return "unlike contact"
        if R.min(1).max() > self.detached:
            return "detached atom"
        return None


def featurizer_for_params(params, inv_form="sq", space="density"):
    """The features a hull lives in, for a species table: ``space="env_channels"`` (cmm runs' hull.pt) the 17 raw
    encoder channels, else the density environment's invariants (before the feature scale)."""
    from ..ff.cmm.params import IonTable
    tab = IonTable.load(params)
    if space == "env_channels":
        return _ChannelFeaturizer(tab)
    from ..ff.cmm.environment import DensityEnvironment
    return DensityEnvironment(tab, inv_form=inv_form)


class _ChannelFeaturizer:
    """The cmm encoder's raw channels of a frame (``rsfff.ff.cmm.channels.raw_channels``)."""

    def __init__(self, tab):
        from ..ff.cmm.channels import R_MAX
        from ..ff.cmm.tabulate import XpolTable
        self.tab, self.table = tab, XpolTable(tab, r_max=R_MAX)

    def __call__(self, positions, atomic_numbers, batch_idx):
        from types import SimpleNamespace
        from ..ff.cmm.channels import raw_channels
        sp = self.tab.species_index(atomic_numbers)
        return SimpleNamespace(inv_feats=raw_channels(self.tab, self.table, positions.double(), sp, batch_idx))


def frame_features(featurizer, numbers, pos):
    """Raw invariants (N, F) of one frame (positions in Angstrom)."""
    import torch
    numbers = np.asarray(numbers)
    out = featurizer(torch.as_tensor(np.asarray(pos, float)), torch.as_tensor(numbers),
                     torch.zeros(len(numbers), dtype=torch.long))
    return out.inv_feats.detach().cpu().numpy()


def score_frames(hull: FeatureHull, feats_list):
    """Per frame: (max distance, mean distance, number of atoms outside) against the hull."""
    out = []
    for F in feats_list:
        d = hull.distance(F)
        out.append((float(d.max()), float(d.mean()), int((d > INSIDE_TOL).sum())))
    return np.array(out, dtype=float).reshape(-1, 3)


def greedy_extend(hull: FeatureHull, feats_list, n, score="max", copy=True, verbose=False):
    """Pick up to ``n`` frames by greedy hull extension.

    ``feats_list``: raw invariants per candidate frame. ``score``: "max" (farthest atom; default) or "sum" (total
    distance of the frame's atoms, favouring frames that bring many new environments). Stops early when every
    remaining candidate is inside. Returns (picked indices, their scores at the time of picking, the extended hull;
    a copy unless ``copy`` is False)."""
    h = FeatureHull.from_dict(hull.to_dict()) if copy else hull
    Z = [h.whiten(F) for F in feats_list]
    agg = np.max if score == "max" else np.sum

    def s(i):
        return float(agg(hull_distance(h.points, Z[i])))

    heap = [(-s(i), i) for i in range(len(Z))]
    heapq.heapify(heap)
    picked, scores = [], []
    while heap and len(picked) < n:
        neg, i = heapq.heappop(heap)
        cur = s(i)                                    # re-score against the current hull
        if heap and cur < -heap[0][0] - 1e-12:        # no longer the best: push back with its fresh score
            heapq.heappush(heap, (-cur, i))
            continue
        if cur <= INSIDE_TOL:
            break
        picked.append(i)
        scores.append(cur)
        for z in Z[i][np.argsort(-np.linalg.norm(Z[i], axis=1))]:
            if hull_distance(h.points, z)[0] > INSIDE_TOL:
                h.points = np.vstack([h.points, z])
        if verbose:
            print(f"  pick {len(picked):4d}: frame {i} score {cur:.3g} (support {len(h.points)})", flush=True)
    return picked, scores, h
