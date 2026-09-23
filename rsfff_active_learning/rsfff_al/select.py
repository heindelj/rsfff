"""From recorded trajectories to the frames worth a Q-Chem calculation.

Two steps, because they answer different questions:

``pool``    per trajectory, the most uncertain frame in every ``window_fs`` of time. Frames
            10 fs apart are the same question asked twice; one per window decorrelates the
            candidates and bounds the size of what the sample stage writes.
``choose``  per cluster size, the ``k`` most uncertain candidates, at most ``max_per_traj``
            from any one trajectory, and of those the ``n_eda`` that also get an EDA.

Scores. ``sigma_energy`` (total, Hartree) and ``sigma_forces`` (DeePMD max-atom deviation,
Hartree/Angstrom) fail differently -- an energy can agree by cancellation while the forces do
not -- so the default ``"both"`` divides each by its median over the size's pool and takes the
larger. Ranking is always within one size, so the growth of either number with cluster size
cannot swallow the budget.

The EDA subset is spread evenly down the ranking rather than taken from its top: EDA labels
anchor the decomposition, and the few frames the committee finds most alarming are the least
typical of what the terms have to describe.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["pool_indices", "scores", "choose"]


def pool_indices(replica, time_fs, score, *, window_fs: float = 100.0, phase=None,
                 include_warmup: bool = True) -> np.ndarray:
    """Row indices: the best-scoring row per (replica, time window)."""
    replica = np.asarray(replica)
    time_fs = np.asarray(time_fs, float)
    score = np.asarray(score, float)
    keep = np.isfinite(score)
    if phase is not None and not include_warmup:
        keep &= np.asarray(phase) == 1
    best: dict[tuple[int, int], int] = {}
    for j in np.flatnonzero(keep):
        key = (int(replica[j]), int(time_fs[j] // window_fs))
        if key not in best or score[j] > score[best[key]]:
            best[key] = int(j)
    return np.array(sorted(best.values()), dtype=int)


def scores(sigma_energy, sigma_forces, mode: str = "both") -> np.ndarray:
    se = np.asarray(sigma_energy, float)
    sf = np.asarray(sigma_forces, float)
    if mode == "energy":
        return se
    if mode == "forces":
        return sf
    if mode != "both":
        raise ValueError(f"score mode {mode!r}: both | energy | forces")
    me = float(np.nanmedian(se)) or 1.0
    mf = float(np.nanmedian(sf)) or 1.0
    return np.maximum(se / me, sf / mf)


def choose(score, trajectory, k: int, *, n_eda: int = 0, max_per_traj: int | None = None,
           exclude=None) -> tuple[np.ndarray, np.ndarray]:
    """``(chosen, eda)``: indices of the ``k`` best candidates (ranked, best first) and the
    subset of ``n_eda`` of them that also get an EDA."""
    score = np.asarray(score, float)
    traj = np.asarray(trajectory)
    ok = np.isfinite(score)
    if exclude is not None:
        ok &= ~np.asarray(exclude, bool)
    order = [int(j) for j in np.argsort(-score, kind="stable") if ok[j]]
    if max_per_traj is None:
        n_traj = max(len(set(traj[ok].tolist())), 1)
        max_per_traj = max(1, math.ceil(2.0 * k / n_traj))
    taken: dict = {}
    chosen = []
    for j in order:
        if len(chosen) >= k:
            break
        t = traj[j]
        if taken.get(t, 0) >= max_per_traj:
            continue
        taken[t] = taken.get(t, 0) + 1
        chosen.append(j)
    if len(chosen) < k:                    # the per-trajectory cap was too tight: relax it
        rest = [j for j in order if j not in set(chosen)]
        chosen += rest[:k - len(chosen)]
    chosen = np.array(chosen, dtype=int)
    n_eda = min(int(n_eda), len(chosen))
    if n_eda <= 0:
        return chosen, np.array([], dtype=int)
    picks = np.unique(np.linspace(0, len(chosen) - 1, n_eda).round().astype(int))
    return chosen, chosen[picks]
