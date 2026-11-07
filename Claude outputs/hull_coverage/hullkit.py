"""Convex-hull coverage of the complete model's environment invariants (per ion, in the network's input space)."""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.optimize import nnls

SYMS = ['Li', 'Na', 'K', 'Rb', 'Cs', 'F', 'Cl', 'Br', 'I']
PROBES = ("shell", "dip", "quad", "donor", "acceptor", "breath")
INV = ("V0_pen", "|V1|^2", "|V2|^2", "S0", "|S1|^2", "|S2|^2", "V1.S1")
NAMES = [f"{p}:{q}" for p in PROBES for q in INV]


def sq_feats(env):
    """(N, 42) squared-norm invariants from a frame's equivariant blocks (same as rsfff env_invariants, form sq)."""
    cols = []
    for p in PROBES:
        V0, V1, V2, S0, S1, S2 = (env[f"{p}:{k}"] for k in ("V0", "V1", "V2", "S0", "S1", "S2"))
        cols.append(torch.stack([V0, (V1 * V1).sum(-1), (V2 * V2).sum(-1), S0, (S1 * S1).sum(-1),
                                 (S2.flatten(-2) ** 2).sum(-1), (V1 * S1).sum(-1)], -1))
    return torch.cat(cols, -1).numpy()


def ion_table(frames, preds=None):
    """One row per ion: frame index, species, n ions, workflow/parent/order, split (from predictions.csv)."""
    rows, F = [], []
    for k, f in enumerate(frames):
        X = sq_feats(f["env"])
        F.append(X)
        m = f["meta"]
        for a, s in enumerate(f["species"].tolist()):
            rows.append(dict(frame=k, atom=a, sym=SYMS[s], n=len(f["species"]), workflow=m.get("workflow"),
                             parent=m.get("parent"), order=m.get("order"), full=bool(m.get("full")),
                             ho=float(m.get("ho_ratio", np.nan))))
    df = pd.DataFrame(rows)
    return df, np.concatenate(F)


def match_predictions(frames, preds: pd.DataFrame):
    """Frame index -> predictions row, matched on (workflow, parent, order, ho_ratio)."""
    key = lambda w, p, o, h: (w, p, int(o), round(float(h), 8))
    lut = {key(r.workflow, r.parent, r.order, r.ho_ratio): i for i, r in preds.iterrows()}
    idx = []
    for f in frames:
        m = f["meta"]
        idx.append(lut.get(key(m["workflow"], m["parent"], m["order"], m["ho_ratio"]), -1))
    return np.array(idx)


class Whitener:
    """PCA whitening fitted on the training rows; keeps the leading k components (or a variance fraction)."""

    def __init__(self, X, k=None, var=0.999):
        self.mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        ev = s ** 2 / (len(X) - 1)
        self.ev = ev
        if k is None:
            k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), var) + 1)
        self.k, self.V, self.sd = k, Vt[:k].T, np.sqrt(ev[:k])

    def __call__(self, X):
        return (X - self.mu) @ self.V / self.sd


def hull_distance(P, Q, M=1e3, return_weights=False):
    """Euclidean distance of each row of Q to conv(rows of P), by NNLS with the simplex constraint as a stiff
    extra row: min ||P^T l - q||^2 + M^2 (1^T l - 1)^2, l >= 0. Active-set, so cost ~ (k+1) nonzeros per query."""
    A = np.vstack([P.T, M * np.ones(len(P))])
    out, W = np.empty(len(Q)), []
    for i, q in enumerate(Q):
        l, _ = nnls(A, np.append(q, M), maxiter=50 * A.shape[1])
        out[i] = np.linalg.norm(P.T @ l - q)
        if return_weights:
            W.append(l)
    return (out, W) if return_weights else out


def extreme_points(P, tol=1e-9):
    """Indices of the rows of P that are vertices of conv(P) (each tested against the others)."""
    keep = []
    for i in range(len(P)):
        others = np.delete(P, i, 0)
        if hull_distance(others, P[i:i + 1])[0] > tol:
            keep.append(i)
    return np.array(keep)


def maxvol(A, iters=200, tol=1.01):
    """Row indices of a (near-)maximal-volume d x d submatrix of A (n x d), Goreinov et al. greedy swaps."""
    n, d = A.shape
    from scipy.linalg import qr
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
    """Extrapolation grade gamma = max_j |(x A^-1)_j| (> 1: outside the active set's polytope)."""
    return np.abs(X @ np.linalg.inv(A_active)).max(1)


def knn_dist(P, Q, k=5):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(Q, k=k)
    return d.mean(1) if k > 1 else d


def twonn_id(X, frac=0.9):
    """TwoNN intrinsic dimension (Facco et al. 2017)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    m = int(frac * n)
    x = np.log(mu[:m])
    y = -np.log(1 - np.arange(1, m + 1) / n)
    return float((x @ y) / (x @ x))


def information_imbalance(XA, XB, n=3000, seed=0):
    """Delta(A -> B) (Glielmo et al. 2022): mean rank in B of each point's nearest neighbour in A, x 2/N.
    ~0: A predicts B's neighbourhoods; ~1: A carries no information about B."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(XA), min(n, len(XA)), replace=False)
    A, B = XA[sel], XB[sel]
    from scipy.spatial.distance import cdist
    DA, DB = cdist(A, A), cdist(B, B)
    np.fill_diagonal(DA, np.inf)
    np.fill_diagonal(DB, np.inf)
    nnA = DA.argmin(1)
    rB = DB.argsort(1).argsort(1)                 # rank 0 = nearest
    return 2.0 * (rB[np.arange(len(A)), nnA] + 1).mean() / len(A)


def greedy_ii_selection(Z, n_max=20, groups=None):
    """Forward selection of columns (or column groups) of Z minimizing Delta(selected -> all columns)."""
    from scipy.spatial.distance import cdist
    N = len(Z)
    DB = cdist(Z, Z); np.fill_diagonal(DB, np.inf)
    groups = groups or {i: [i] for i in range(Z.shape[1])}
    sel, curve = [], []
    rows = np.arange(N)
    TB = np.random.default_rng(0).random((N, N))
    for _ in range(min(n_max, len(groups))):
        best = None
        for g, cols in groups.items():
            if g in sel:
                continue
            c = sum((groups[s] for s in sel), []) + cols
            DA = cdist(Z[:, c], Z[:, c]); np.fill_diagonal(DA, np.inf)
            # exact ties in A (e.g. a probe that is identically zero for some species) are broken at random
            nn = (DA + 1e-9 * TB).argmin(1)
            r = (DB < DB[rows, nn][:, None]).sum(1) + 1
            d = 2.0 * r.mean() / N
            if best is None or d < best[1]:
                best = (g, d)
        sel.append(best[0]); curve.append(best[1])
    return sel, curve
