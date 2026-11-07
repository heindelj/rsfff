"""Convex hull of the environment invariants: coverage, extrapolation and greedy hull extension.

The complete model's heads read, per atom, I = feats / feat_scale (no species embedding), so one hull in that space
covers every species. The hull is taken in the leading k components of a PCA fitted on the training atoms and
whitened (each component scaled to unit variance), so the k directions count equally and the hull is low enough in
dimension to be informative (in 42 dimensions almost every new point is outside the hull of any finite set).

Distance to the hull is a non-negative least-squares problem with the simplex constraint as one stiff extra row:

    d(x) = min_{l >= 0} || P^T l - x ||   with   M (1^T l - 1) = 0 enforced by the row M 1^T  (M = 1e3),

solved by the Lawson-Hanson active set (scipy nnls). The solution has at most k + 1 non-zeros, so a query against
the hull's support points costs ~0.5 ms at k = 6. The projection p* = P^T l also gives the gradient of the distance
(Danskin): grad_x d = (x - p*) / d, which is what a hull-distance bias in MD needs.

Only the hull's support points are kept: :func:`hull_support` returns a subset S of the rows with conv(S) = conv(P)
(all the vertices, possibly a few non-vertices), found by inserting the points from the outside in and keeping only
those outside the current support's hull.

:class:`FeatureHull` holds the whitener, the support points and the metadata a checkpoint needs (feature names,
feat_scale, k); it saves to and loads from a plain dict (``torch.save``-able). Greedy hull extension, the guard and
the frame features for candidate pools are in :mod:`rsfff.coverage.select`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

__all__ = ["Whitener", "hull_distance", "hull_support", "FeatureHull", "INSIDE_TOL", "maxvol", "maxvol_grade",
           "knn_distance", "twonn_dimension", "information_imbalance", "greedy_feature_selection"]

#: distances below this (whitened units, i.e. fractions of a standard deviation) count as inside the hull
INSIDE_TOL = 1e-5


# ----------------------------------------------------------------------------------------------- whitening
@dataclass
class Whitener:
    """PCA whitening: z = (x - mu) V / sd, with the leading k components of the fitting set."""
    mu: np.ndarray
    V: np.ndarray            # (F, k)
    sd: np.ndarray           # (k,)
    explained: np.ndarray    # (F,) variance fraction of every component of the fitting set

    @classmethod
    def fit(cls, X, k=None, var=0.9):
        """``k`` components, or (k None) the fewest that explain the variance fraction ``var``."""
        X = np.asarray(X, float)
        mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        ev = s ** 2 / max(len(X) - 1, 1)
        frac = ev / ev.sum()
        if k is None:
            k = int(np.searchsorted(np.cumsum(frac), var) + 1)
        k = int(min(k, Vt.shape[0]))
        return cls(mu, Vt[:k].T.copy(), np.sqrt(ev[:k]).clip(min=1e-300), frac)

    @property
    def k(self):
        return self.V.shape[1]

    def __call__(self, X):
        return (np.asarray(X, float) - self.mu) @ self.V / self.sd

    def jacobian(self):
        """dz/dx (F, k): constant, for chaining a hull-distance gradient back to the invariants."""
        return self.V / self.sd


# ----------------------------------------------------------------------------------------------- distance
def hull_distance(P, Q, M=1e3, return_projection=False):
    """Euclidean distance of every row of ``Q`` to conv(rows of ``P``) (same, whitened, coordinates).

    With ``return_projection`` also the nearest hull points p* (rows), so grad d = (q - p*) / d."""
    P, Q = np.atleast_2d(np.asarray(P, float)), np.atleast_2d(np.asarray(Q, float))
    if len(P) == 0:
        raise ValueError("empty hull")
    A = np.vstack([P.T, np.full(len(P), M)])
    d = np.empty(len(Q))
    proj = np.empty_like(Q) if return_projection else None
    maxiter = 50 * A.shape[1]
    for i, q in enumerate(Q):
        lam, _ = nnls(A, np.append(q, M), maxiter=maxiter)
        p = P.T @ lam
        d[i] = np.linalg.norm(p - q)
        if return_projection:
            proj[i] = p
    return (d, proj) if return_projection else d


def hull_support(P, tol=INSIDE_TOL, prune=False):
    """Indices of a subset S of the rows of ``P`` with conv(P[S]) = conv(P).

    Points are inserted from the farthest (from the centroid) inwards; a point joins S only if it lies outside
    conv(S). Every vertex of conv(P) ends up in S. With ``prune`` the few non-vertices that entered early are removed
    by testing each member of S against the others."""
    P = np.asarray(P, float)
    if len(P) <= P.shape[1] + 1:
        return np.arange(len(P))
    order = np.argsort(-np.linalg.norm(P - P.mean(0), axis=1))
    S = list(order[: P.shape[1] + 1])
    for i in order[P.shape[1] + 1:]:
        if hull_distance(P[S], P[i])[0] > tol:
            S.append(i)
    S = np.array(S)
    if prune:
        keep = [j for j, i in enumerate(S) if hull_distance(np.delete(P[S], j, 0), P[i])[0] > tol]
        S = S[keep]
    return np.sort(S)


# ----------------------------------------------------------------------------------------------- the hull object
@dataclass
class FeatureHull:
    """The training set's hull in whitened invariant space.

    ``points`` are the support points (whitened); ``feat_scale`` maps raw invariants to the network's input
    (I = feats / feat_scale). ``meta`` is free-form provenance (run, cache, number of training atoms, ...)."""
    whitener: Whitener
    points: np.ndarray
    feat_scale: np.ndarray
    feature_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- construction
    @classmethod
    def fit(cls, feats, feat_scale, k=6, var=None, feature_names=None, meta=None, verbose=False):
        """From raw training invariants ``feats`` (N, F). Atoms with an all-zero environment (isolated ions) are kept:
        the origin is a legitimate training point."""
        import time
        t0 = time.time()
        feat_scale = np.asarray(feat_scale, float)
        I = np.asarray(feats, float) / feat_scale
        if not np.isfinite(I).all():
            raise ValueError("non-finite scaled invariants (a zero feat_scale? train_ionct clamps it at 1e-6)")
        W = Whitener.fit(I, k=None if var else k, var=var or 0.9)
        Z = np.unique(W(I).round(10), axis=0)
        S = hull_support(Z)
        h = cls(W, Z[S], feat_scale, list(feature_names or []),
                dict(meta or {}, n_train_atoms=int(len(I)), n_unique=int(len(Z))))
        if verbose:
            print(f"hull: {len(I)} atoms -> {len(Z)} unique, k = {W.k} "
                  f"({100 * W.explained[:W.k].sum():.1f} % of the variance), {len(S)} support points "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        return h

    def to_dict(self):
        w = self.whitener
        return {"mu": w.mu, "V": w.V, "sd": w.sd, "explained": w.explained, "points": self.points,
                "feat_scale": self.feat_scale, "feature_names": list(self.feature_names), "meta": dict(self.meta),
                "version": 1}

    @classmethod
    def from_dict(cls, d):
        W = Whitener(np.asarray(d["mu"]), np.asarray(d["V"]), np.asarray(d["sd"]), np.asarray(d["explained"]))
        return cls(W, np.asarray(d["points"]), np.asarray(d["feat_scale"]), list(d.get("feature_names") or []),
                   dict(d.get("meta") or {}))

    def save(self, path):
        import torch
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path):
        import torch
        return cls.from_dict(torch.load(path, weights_only=False))

    # -- queries
    @property
    def k(self):
        return self.whitener.k

    def whiten(self, feats):
        """Raw invariants (N, F) -> whitened hull coordinates (N, k)."""
        return self.whitener(np.asarray(feats, float) / self.feat_scale)

    def distance(self, feats, return_projection=False):
        """Distance of each atom (raw invariants, (N, F)) to the hull; 0 inside."""
        return hull_distance(self.points, self.whiten(feats), return_projection=return_projection)

    def gradient(self, feats):
        """(d, dd/dfeats): the distance and its gradient with respect to the raw invariants (zero inside)."""
        Z = self.whiten(feats)
        d, p = hull_distance(self.points, Z, return_projection=True)
        g = np.zeros_like(Z)
        out = d > INSIDE_TOL
        g[out] = (Z[out] - p[out]) / d[out, None]
        return d, (g @ self.whitener.jacobian().T) / self.feat_scale

    def inside(self, feats, tol=INSIDE_TOL):
        return self.distance(feats) <= tol

    # -- growth
    def extend(self, feats, tol=INSIDE_TOL):
        """Add atoms (raw invariants) to the hull in place; only those outside it join the support. Returns how
        many joined. The whitener is kept (the hull's coordinates do not move between rounds)."""
        Z = self.whiten(feats)
        added = 0
        for z in Z[np.argsort(-np.linalg.norm(Z, axis=1))]:
            if hull_distance(self.points, z)[0] > tol:
                self.points = np.vstack([self.points, z])
                added += 1
        return added

    def prune(self):
        """Drop support points that are not vertices (after many :meth:`extend` calls)."""
        self.points = self.points[hull_support(self.points, prune=True)]
        return self


# ----------------------------------------------------------------------------------------------- related measures
def maxvol(A, iters=500, tol=1.01):
    """Rows of a (near-)maximum-volume square submatrix of A (n x d), by greedy row swaps (Goreinov et al.). With
    whitened features (+ a constant column) this is the D-optimal active set of MTP/ACE active learning."""
    from scipy.linalg import qr
    A = np.asarray(A, float)
    n, d = A.shape
    _, _, piv = qr(A.T, pivoting=True, mode="economic")
    idx = piv[:d].copy()
    for _ in range(iters):
        B = A @ np.linalg.inv(A[idx])
        i, j = np.unravel_index(np.abs(B).argmax(), B.shape)
        if abs(B[i, j]) <= tol:
            break
        idx[j] = i
    return idx


def maxvol_grade(A_active, X):
    """Extrapolation grade gamma(x) = max_j |(x A^-1)_j| (> 1: outside the active set's polytope)."""
    return np.abs(np.asarray(X, float) @ np.linalg.inv(A_active)).max(1)


def knn_distance(P, Q, k=5):
    """Mean distance of each row of Q to its k nearest rows of P (holes inside the hull)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(Q, k=k)
    return d.mean(1) if k > 1 else d


def twonn_dimension(X, frac=0.9):
    """TwoNN intrinsic dimension (Facco et al. 2017). Deduplicate X first (repeated points bias it low)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(X, k=3)
    ok = d[:, 1] > 1e-12
    mu = np.sort(d[ok, 2] / d[ok, 1])
    n, m = len(mu), int(frac * len(mu))
    x, y = np.log(mu[:m]), -np.log(1 - np.arange(1, m + 1) / n)
    return float((x @ y) / (x @ x))


def _imbalance(DA, DB, ties):
    """2/N <rank in B of the nearest neighbour in A>; exact ties in A broken at random."""
    N = len(DA)
    nn = (DA + 1e-9 * ties).argmin(1)
    r = (DB < DB[np.arange(N), nn][:, None]).sum(1) + 1
    return 2.0 * r.mean() / N


def information_imbalance(XA, XB, n=3000, seed=0):
    """Delta(A -> B) (Glielmo, Zeni, Laio 2022): ~0 if A's neighbourhoods predict B's, ~1 if A says nothing of B."""
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(XA), min(n, len(XA)), replace=False)
    A, B = np.asarray(XA, float)[sel], np.asarray(XB, float)[sel]
    DA, DB = cdist(A, A), cdist(B, B)
    np.fill_diagonal(DA, np.inf)
    np.fill_diagonal(DB, np.inf)
    return _imbalance(DA, DB, rng.random(DA.shape))


def greedy_feature_selection(Z, n_max=20, groups=None, seed=0):
    """Forward selection of columns (or named column groups) of Z minimizing Delta(selected -> all columns).
    Standardize Z first. Returns (selected keys, Delta after each step)."""
    from scipy.spatial.distance import cdist
    Z = np.asarray(Z, float)
    DB = cdist(Z, Z)
    np.fill_diagonal(DB, np.inf)
    ties = np.random.default_rng(seed).random(DB.shape)
    groups = groups or {i: [i] for i in range(Z.shape[1])}
    sel, curve = [], []
    for _ in range(min(n_max, len(groups))):
        best = None
        for g, cols in groups.items():
            if g in sel:
                continue
            c = sum((list(groups[s]) for s in sel), []) + list(cols)
            DA = cdist(Z[:, c], Z[:, c])
            np.fill_diagonal(DA, np.inf)
            v = _imbalance(DA, DB, ties)
            if best is None or v < best[1]:
                best = (g, v)
        sel.append(best[0])
        curve.append(best[1])
    return sel, curve
