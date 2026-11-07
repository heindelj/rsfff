"""Overlap-gated diffuse-probe invariants (``CompleteModel(env_ct_gated=True)``): the fields of the donor, acceptor
and breath probes with each neighbour's point charge switched off with the probe's own overlap with it.

Checks: the penetration blocks of batched_precompute plus the point charges are the full fields; with every
neighbour inside the gate the invariants are the full-field ones; an ion beyond the gate's reach changes nothing;
a periodic cell gives the same invariants as a cluster cut from its lattice (no lattice sums); derivatives are
finite differences."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
f64 = torch.float64
PROBES = ("donor", "acceptor", "breath")


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)
    m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                      pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="twostate",
                      env_channels=True, env_ct_gated=True)
    return tab, chan, table, feat, m


def _bt(table, tab, chan, X, Z, m, **kw):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form,
                              **kw)


def _full(bt, feat):
    names = list(feat.feature_names)
    return torch.stack([bt["feats"][:, names.index(f"{p}:{k}")] for p in PROBES
                        for k in ("S0", "V0_pen", "|V1|^2", "|V2|^2", "V1.S1")], 1)


def test_penetration_blocks_plus_point_charges_are_the_full_fields(setup):
    tab, chan, table, feat, m = setup
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=f64)
    X = g * 2.8 + 0.1 * torch.randn(8, 3, generator=torch.Generator().manual_seed(0), dtype=f64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    bt = _bt(table, tab, chan, X, Z, m)
    env, sp, Xb = bt["env"], bt["species"], bt["pos"] / K.BOHR_ANG
    pr = K.all_pairs(bt["frame"])
    i, j = torch.cat([pr[0], pr[1]]), torch.cat([pr[1], pr[0]])
    T0 = K._tensor(Xb[j] - Xb[i], torch.ones(5, i.shape[0], dtype=f64))[:, 0, :]
    Dp = (tab.charge[sp[j]][:, None] * T0) @ K._poly_map(f64, Xb.device)
    v1 = Xb.new_zeros(8, 3).index_add(0, i, -Dp[:, :3])
    v2 = Xb.new_zeros(8, 5).index_add(0, i, Dp[:, 3:])
    for p in PROBES:
        has = (tab.a0[sp] > 0).to(f64)[:, None] if p == "breath" else 1.0
        assert torch.allclose(env[f"{p}:V1"], env[f"{p}:V1pen"] + v1 * has, atol=1e-14)
        assert torch.allclose(env[f"{p}:V2"], env[f"{p}:V2pen"] + v2 * has, atol=1e-14)


def test_inside_the_gate_the_invariants_are_the_full_ones(setup):
    """NaCl at 2.8 A: every probe's overlap with the partner is above the gate's full value."""
    tab, chan, table, feat, m = setup
    bt = _bt(table, tab, chan, torch.tensor([[0.0, 0, 0], [2.8, 0, 0]], dtype=f64), torch.tensor([11, 17]), m)
    G = m.diffuse_gated_invariants(bt)
    assert torch.allclose(G, _full(bt, feat), rtol=1e-12, atol=1e-16)
    assert float(G.abs().max()) > 0


def test_an_ion_beyond_the_reach_changes_nothing(setup):
    """A Cl- 3 A beyond the gate's reach: the gated invariants of the others are unchanged (the full-field ones are
    not), and its own are those of an isolated ion."""
    tab, chan, table, feat, m = setup
    reach = m.gate_reach()
    assert 6.0 < reach < 9.0
    X = torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [1.4, 2.6, 0.3]], dtype=f64)
    Z = torch.tensor([11, 17, 11])
    far = torch.tensor([[0.0, 0.0, 2.8 + reach + 3.0]], dtype=f64)
    b1 = _bt(table, tab, chan, X, Z, m)
    b2 = _bt(table, tab, chan, torch.cat([X, far]), torch.cat([Z, torch.tensor([17])]), m)
    G1, G2 = m.diffuse_gated_invariants(b1), m.diffuse_gated_invariants(b2)
    assert torch.allclose(G2[:3], G1, rtol=1e-12, atol=1e-18)
    assert float((_full(b2, feat)[:3] - _full(b1, feat)).abs().max()) > 1e-6
    assert float(G2[3].abs().max()) == 0.0


def test_periodic_cell_equals_the_cluster_cut_from_its_lattice(setup):
    """A displaced 8-ion rock-salt cell: the gated invariants under PBC equal those of the same ions in a cluster of
    every lattice image within the gate's reach (+ the penetration range) -- no lattice sums enter, while the
    full-field invariants (Ewald) differ from the cluster's."""
    tab, chan, table, feat, m = setup
    a = 5.64
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=f64)
    X = g * a / 2 + 0.12 * torch.randn(8, 3, generator=torch.Generator().manual_seed(4), dtype=f64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    cell = torch.eye(3, dtype=f64) * a
    bp = _bt(table, tab, chan, X, Z, m, cell=cell, cutoff=10.0)
    Gp = m.diffuse_gated_invariants(bp)
    n = 3
    sh = torch.tensor([(i, j, k) for i in range(-n, n + 1) for j in range(-n, n + 1) for k in range(-n, n + 1)], dtype=f64)
    Xs = (X[None] + (sh * a)[:, None]).reshape(-1, 3)
    Zs = Z.repeat(sh.shape[0])
    d = (Xs[:, None] - X[None]).norm(dim=-1).min(1).values
    keep = d < m.gate_reach() + 2.0
    s0 = int(torch.nonzero((sh == 0).all(1))[0])
    central = list(range(8 * s0, 8 * s0 + 8))
    sel = central + [k for k in torch.nonzero(keep).flatten().tolist() if k not in central]
    bc = _bt(table, tab, chan, Xs[sel], Zs[sel], m)
    Gc = m.diffuse_gated_invariants(bc)[:8]

    def rel(A, B):
        return float(((A - B).abs().max(0).values / B.abs().max(0).values.clamp(min=1e-30)).max())
    # the periodic short-range rows are switched off over the last 1 A of the 10 A cutoff (as for every environment
    # block): ~1e-5 relative; the full-field (Ewald) invariants differ at the percent level and more
    assert rel(Gp, Gc) < 2e-4
    assert rel(_full(bp, feat), _full(bc, feat)[:8]) > 1e-2


def test_gated_invariants_are_differentiable(setup):
    tab, chan, table, feat, m = setup
    X = torch.tensor([[0.0, 0, 0], [2.7, 0.2, 0], [1.3, 2.5, 0.4], [4.0, 3.1, -0.5]], dtype=f64)
    Z = torch.tensor([11, 17, 17, 11])
    w = torch.randn(4, 15, generator=torch.Generator().manual_seed(2), dtype=f64)

    with torch.no_grad():
        scale = m.diffuse_gated_invariants(_bt(table, tab, chan, X, Z, m)).abs().amax(0).clamp(min=1e-30)

    def f(Xv):
        return (m.diffuse_gated_invariants(_bt(table, tab, chan, Xv, Z, m)) / scale * w).sum()
    Xr = X.clone().requires_grad_(True)
    g, = torch.autograd.grad(f(Xr), Xr)
    h = 1e-5
    for a, k in ((0, 0), (2, 1), (3, 2)):
        Xp, Xm = X.clone(), X.clone()
        Xp[a, k] += h
        Xm[a, k] -= h
        with torch.no_grad():
            fd = (f(Xp) - f(Xm)) / (2 * h)
        assert float(g[a, k]) == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_model_forward_with_gated_branch(setup):
    """The CT readouts' extra branch takes the 15 gated invariants; a forward pass and its gradient are finite."""
    tab, chan, table, feat, m = setup
    assert m.env.n_extra == 15
    X = torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [1.4, 2.6, 0.3], [4.1, 2.7, 0.2]], dtype=f64).requires_grad_(True)
    o = m(_bt(table, tab, chan, X, torch.tensor([11, 17, 17, 11]), m))
    g, = torch.autograd.grad(o["e_int"].sum(), X)
    assert torch.isfinite(g).all() and torch.isfinite(o["e_ct"]).all()
