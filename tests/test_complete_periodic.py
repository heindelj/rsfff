"""The periodic complete model: Ewald sums against lattice sums, the cluster limit, and forces / stress as exact
derivatives of the periodic energy (pair list with lattice images, matrix-free induction solve)."""
import itertools
import math
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete.periodic import (erfc_screening, make_ewald_frames, periodic_pairs, recip_apply,
                                        solve_induction_periodic)
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.multipole import damped_interaction_tensor

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
ZMAP = {"Na": 11, "Cl": 17}


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(5)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)
    m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=True, frozen=True,
                      pauli_ii=True)
    with torch.no_grad():
        for p in (m.kappa_aniso, m.gap_lamD, m.gap_lamA, m.breath_lam):
            p.fill_(0.3)
    return tab, chan, table, m


def _ewald_out(X, C, M, alpha, rc, kcut):
    """sum_j M_j T(X_j - X_i) over images, real space (screened, every image within rc) + reciprocal (bohr)."""
    b = torch.zeros(X.shape[0], dtype=torch.long)
    (i, j), s = periodic_pairs(X, C[None], b, rc, use_torchff=False)
    dr = X[j] - X[i] + s @ C
    g = erfc_screening(dr.norm(dim=-1), alpha)
    out = torch.zeros_like(M)
    out = out.index_add(0, i, torch.einsum("pa,pab->pb", M[j], damped_interaction_tensor(dr, g, max_rank=2)))
    out = out.index_add(0, j, torch.einsum("pa,pab->pb", M[i], damped_interaction_tensor(-dr, g, max_rank=2)))
    return out + recip_apply(make_ewald_frames(X, C[None], b, alpha, kcut), M)


def test_ewald_is_independent_of_the_splitting():
    """Charges, dipoles and quadrupoles in a triclinic, charged cell: real + reciprocal + self + background does not
    depend on alpha (the self tensor, the background and the reciprocal convention are all in this)."""
    torch.manual_seed(0)
    C = torch.tensor([[14.0, 0, 0], [1.5, 13.3, 0], [0.7, -1.1, 14.7]], dtype=torch.float64)
    X = torch.rand(12, 3, dtype=torch.float64) @ C
    M = torch.randn(12, 10, dtype=torch.float64)
    M[:, 0] += 0.3 - M[:, 0].mean()
    o1 = _ewald_out(X, C, M, 0.32, 20.0, 2 * 0.32 * 6.5)
    o2 = _ewald_out(X, C, M, 0.45, 20.0, 2 * 0.45 * 6.5)
    assert (o1 - o2).abs().max() < 1e-11


def test_madelung_constant():
    base = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float64)
    X, q = [], []
    for c in itertools.product(range(2), repeat=3):
        for b in base:
            X += [2 * torch.tensor(c, dtype=torch.float64) + b, 2 * torch.tensor(c, dtype=torch.float64) + b + torch.tensor([1.0, 0, 0])]
            q += [1.0, -1.0]
    X, q = torch.stack(X), torch.tensor(q, dtype=torch.float64)
    M = torch.zeros(len(q), 10, dtype=torch.float64)
    M[:, 0] = q
    out = _ewald_out(X, torch.eye(3, dtype=torch.float64) * 4, M, 1.2, 3.98, 2 * 1.2 * 6)
    assert abs(-0.5 * (q * out[:, 0]).sum() / (len(q) / 2) - 1.747564594633) < 1e-9


def _rocksalt(n, a=5.64, jitter=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=torch.float64)
    X, Z = [], []
    for c in itertools.product(range(n), repeat=3):
        for b in base:
            o = (torch.tensor(c, dtype=torch.float64) + b) * a
            X += [o, o + torch.tensor([a / 2, 0, 0], dtype=torch.float64)]
            Z += [11, 17]
    X = torch.stack(X)
    X = X + jitter * torch.randn(X.shape, generator=g, dtype=torch.float64)
    return X, torch.tensor(Z), torch.eye(3, dtype=torch.float64) * a * n


def _energy(m, tab, table, chan, X, Z, cell, **kw):
    fr = torch.zeros(len(Z), dtype=torch.long)
    return m(batched_precompute(X, Z, fr, tab, table, chan, cell=cell, **kw))


def test_model_is_independent_of_the_splitting(setup):
    """The whole model (drive, features, induction operator, frontier potentials, electrostatics) at two Ewald
    tolerances, i.e. two alphas at the same real-space cutoff."""
    tab, chan, table, m = setup
    X, Z, cell = _rocksalt(1, jitter=0.25, seed=1)
    with torch.no_grad():
        o1 = _energy(m, tab, table, chan, X, Z, cell, cutoff=11.0, ewald_tol=1e-9)
        o2 = _energy(m, tab, table, chan, X, Z, cell, cutoff=11.0, ewald_tol=1e-13)
    for k in ("e_elst", "e_pol", "e_ct", "e_int"):
        assert abs(float(o1[k] - o2[k])) < 2e-8, k
    assert o1["x"].abs().max() > 1e-3                     # the moments are not trivially zero


def test_huge_box_is_the_cluster(setup):
    """Na4Cl4 rocksalt cube in a large box, pair list past the table: the cluster energy. The cube has no dipole and
    no quadrupole; what is left is its octupole's interaction with its images, ~L^-7 (2e-6 Eh at L = 30 A, scaling
    exactly so: 30 -> 40 A divides every term by 7.5), so at L = 50 A every term is the cluster's to < 1e-7."""
    tab, chan, table, m = setup
    corners = list(itertools.product(range(2), repeat=3))
    X = torch.tensor(corners, dtype=torch.float64) * 2.7 + 25.0
    Z = torch.tensor([11 if sum(c) % 2 == 0 else 17 for c in corners])
    with torch.no_grad():
        ref = m(batched_precompute(X, Z, torch.zeros(8, dtype=torch.long), tab, table, chan))
        out = _energy(m, tab, table, chan, X, Z, torch.eye(3, dtype=torch.float64) * 50.0, cutoff=13.0,
                      ewald_tol=1e-10)
    assert abs(float(ref["e_elst"])) > 0.5 and abs(float(ref["e_pol"])) > 1e-3 and abs(float(ref["e_ct"])) > 1e-4
    for k in ("e_elst", "e_pauli", "e_disp", "e_pol", "e_ct"):
        assert abs(float(ref[k] - out[k])) < 1e-7, (k, float(ref[k]), float(out[k]))


def test_periodic_forces_and_stress(setup):
    """A small triclinic cell (cutoff past half the cell: several images of each pair, and self images): forces are
    -dE/dx and the stress dE/d(strain) / V, both against central differences of the periodic energy."""
    tab, chan, table, m = setup
    X, Z, cell = _rocksalt(1, a=5.7, jitter=0.2, seed=3)
    cell = cell + torch.tensor([[0, 0, 0], [0.4, 0, 0], [0.2, -0.3, 0]], dtype=torch.float64)
    kw = dict(cutoff=9.5, dispersion_tail=True)

    def E(pos, eps):
        F = torch.eye(3, dtype=torch.float64) + eps
        return _energy(m, tab, table, chan, pos @ F, Z, cell @ F, **kw)["e_int"].sum()

    pos = X.clone().requires_grad_(True)
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    g, ge = torch.autograd.grad(E(pos, eps), (pos, eps))
    h = 1e-4
    with torch.no_grad():
        for a, b in ((0, 0), (3, 2), (6, 1)):
            pp, pm = X.clone(), X.clone()
            pp[a, b] += h
            pm[a, b] -= h
            fd = (E(pp, torch.zeros(3, 3, dtype=torch.float64)) - E(pm, torch.zeros(3, 3, dtype=torch.float64))) / (2 * h)
            assert abs(float(fd - g[a, b])) < 1e-7, (a, b, float(fd), float(g[a, b]))
        for a, b in ((0, 0), (1, 2), (2, 1)):
            ep, em = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, 3, dtype=torch.float64)
            ep[a, b], em[a, b] = h, -h
            fd = (E(X, ep) - E(X, em)) / (2 * h)
            assert abs(float(fd - ge[a, b])) < 1e-7, (a, b, float(fd), float(ge[a, b]))


def test_cg_matches_dense_solve():
    """The matrix-free CG against a dense solve of the same operator (assembled column by column)."""
    torch.manual_seed(2)
    n = 6
    C = torch.eye(3, dtype=torch.float64) * 9.0
    X = torch.rand(n, 3, dtype=torch.float64) * 9.0
    b = torch.zeros(n, dtype=torch.long)
    pairs, s = periodic_pairs(X, C[None], b, 8.0, use_torchff=False)
    Bij = 0.01 * torch.randn(pairs.shape[1], 8, 8, dtype=torch.float64)
    frames = make_ewald_frames(X, C[None], b, 0.4, 3.0)
    ainv = torch.eye(3, dtype=torch.float64).expand(n, 3, 3) / 2.0
    Cq = torch.full((n,), 5.0, dtype=torch.float64)
    L = torch.randn(n, 8, dtype=torch.float64) * 0.1
    x = solve_induction_periodic(L, ainv, Cq, pairs, Bij, frames)
    from rsfff.ff.complete.periodic import _matvec
    from rsfff.ff.ionct import kernels as K
    P, dof = K._poly_map(torch.float64, "cpu"), torch.ones(8, dtype=torch.float64)
    A = torch.stack([_matvec(e.reshape(n, 8), ainv, 1 / Cq, pairs, Bij, frames, P, dof).reshape(-1)
                     for e in torch.eye(8 * n, dtype=torch.float64)], 1)
    assert torch.allclose(A, A.T, atol=1e-12)
    assert torch.allclose(x.reshape(-1), -torch.linalg.solve(A, L.reshape(-1)), atol=1e-10)
