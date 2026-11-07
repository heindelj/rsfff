"""m33 environment channels (``channels.py``) and the shared environment (``heads.SharedEnvironment``,
``CompleteModel(env_channels=True)``): the three-body identity against a brute-force triple loop, dimer and isolated
limits, invariances, continuity as a neighbour leaves, readout masks, the pinned a0, and forces as derivatives of
the energy through the channels."""
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import legval

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete.channels import (CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS, channels_from_pairs,
                                        pair_electrostatics, real_harmonics)
from rsfff.ff.complete.heads import SharedEnvironment
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
f64 = torch.float64


def test_harmonics_addition_theorem():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(7, 3, generator=g, dtype=f64)
    b = torch.randn(7, 3, generator=g, dtype=f64)
    a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    for L in range(1, 5):
        lhs = (real_harmonics(a, L) * real_harmonics(b, L)).sum(1).numpy()
        np.testing.assert_allclose(lhs, legval((a * b).sum(1).numpy(), [0] * L + [1]), atol=1e-13)


def test_channels_match_the_triple_loop():
    g = torch.Generator().manual_seed(1)
    dr = torch.randn(5, 3, generator=g, dtype=f64)
    w = torch.rand(5, generator=g, dtype=f64)
    s = torch.randn(5, generator=g, dtype=f64)
    X = channels_from_pairs(1, torch.zeros(5, dtype=torch.long), dr, w, s)[0].numpy()
    rh = (dr / dr.norm(dim=1, keepdim=True)).numpy()
    wn, sn = w.numpy(), s.numpy()
    ref = [wn.sum(), sn.sum()]
    for A, B in ((wn, wn), (wn, sn), (sn, sn)):
        for L in range(5):
            t = sum(0.5 * (A[j] * B[k] + A[k] * B[j]) * (1.0 if L == 0 else 1 - legval(rh[j] @ rh[k], [0] * L + [1]))
                    for j in range(5) for k in range(j + 1, 5))
            ref.append(t / wn.sum())
    np.testing.assert_allclose(X, ref, rtol=1e-12, atol=1e-15)
    assert len(ref) == len(CHANNEL_NAMES)


def test_single_neighbour_has_no_three_body_and_no_pairs_gives_zero():
    dr = torch.tensor([[0.0, 0.0, 2.8]], dtype=f64)
    X = channels_from_pairs(2, torch.tensor([0]), dr, torch.tensor([3e-3], dtype=f64), torch.tensor([-1e-3], dtype=f64))
    assert float(X[0, list(THREE_BODY_COLS)].abs().max()) < 1e-18
    assert float(X[0, 0]) == pytest.approx(3e-3) and float(X[0, 1]) == pytest.approx(-1e-3)
    assert float(X[1].abs().max()) == 0.0                           # atom 1 is the centre of no pair


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = DensityEnvironment(tab)
    table = RadialTable(tab, h=0.05)
    cache = {}

    def make(**kw):
        torch.manual_seed(5)
        args = dict(feature_names=feat.feature_names, pseudo=False, frozen=True, pauli_form="shells", confine=(),
                    env_cap=3.0, xpol=True, ct_model="flux", env_channels=True, aniso=False)
        args.update(kw)
        m = CompleteModel(tab, chan, feat.n_feat, **args)
        if m.env is not None:            # random readouts, so every environment path is exercised
            with torch.no_grad():
                for p in m.env.parameters():
                    p.add_(0.3 * torch.randn_like(p))
        if "flux" in cache:
            m._flux_table = cache["flux"]
        if "xpol" in cache:
            m._xpol_table = cache["xpol"]
        return m

    def keep(m):
        if getattr(m, "_flux_table", None) is not None:
            cache.setdefault("flux", m._flux_table)
        if getattr(m, "_xpol_table", None) is not None:
            cache.setdefault("xpol", m._xpol_table)
    return tab, chan, table, feat, make, keep


def _cluster(a=2.8, seed=1):
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=f64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.1 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(seed), dtype=f64)
    return X, Z


def test_pair_electrostatics_matches_the_frame_sum(setup):
    from rsfff.ff.complete.elst import permanent_electrostatics
    from rsfff.ff.ionct import kernels as K
    tab, *_ = setup
    X, Z = _cluster()
    sp = tab.species_index(Z)
    fr = torch.zeros(len(Z), dtype=torch.long)
    i, j = K.all_pairs(fr)
    e = pair_electrostatics(sp[i], sp[j], (X[j] - X[i]).norm(dim=1) / K.BOHR_ANG, tab).sum()
    assert float(e) == pytest.approx(float(permanent_electrostatics(X, sp, fr, tab)[0]), rel=1e-12)


def test_env_channels_invariance_and_signs(setup):
    tab, chan, table, feat, make, keep = setup
    m = make()
    X, Z = _cluster()
    sp = tab.species_index(Z)
    fr = torch.zeros(len(Z), dtype=torch.long)
    C0 = m.env_raw(X, sp, fr)
    keep(m)
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(2), dtype=f64))[0]
    C1 = m.env_raw(X @ Q.T + 1.7, sp, fr)
    torch.testing.assert_close(C0, C1, rtol=1e-9, atol=1e-14)
    perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(3))
    C2 = m.env_raw(X[perm], sp[perm], fr)
    torch.testing.assert_close(C0[perm], C2, rtol=1e-9, atol=1e-14)
    # rock-salt cube: every first neighbour is unlike, so S^s < 0 (unlike charges: E_elst < 0); like pairs give T^ss > 0
    assert bool((C0[:, 1] < 0).all())
    i_tss0 = CHANNEL_NAMES.index("T^ss_0/S")
    assert bool((C0[:, i_tss0] > 0).all())


def test_channels_continuous_as_a_neighbour_leaves(setup):
    tab, chan, table, feat, make, keep = setup
    m = make()
    keep(m)
    sp = tab.species_index(torch.tensor([17, 11, 11]))
    fr = torch.zeros(3, dtype=torch.long)
    R = torch.linspace(2.6, 9.9, 400, dtype=f64)
    rows = []
    for r in R:
        X = torch.tensor([[0, 0, 0], [2.8, 0, 0], [0, float(r), 0]], dtype=f64)
        rows.append(m.env_raw(X, sp, fr)[0])
    C = torch.stack(rows)
    step = (C[1:] - C[:-1]).abs().max(0).values
    span = C.abs().max(0).values.clamp(min=1e-30)
    assert float((step / span).max()) < 0.05                       # no jumps anywhere on a 0.018 A grid
    assert float(C[-1, list(THREE_BODY_COLS)].abs().max()) < 1e-6 * float(span[list(THREE_BODY_COLS)].max())


def test_shared_environment_limits_and_masks():
    ro = [("a", "t1", True), ("b", "t1", False), ("c", "t2", False)]
    for mode in ("shared", "per_term"):
        env = SharedEnvironment(len(CHANNEL_NAMES), THREE_BODY_COLS, TWO_BODY_COLS, ro, latent=3, hidden=8, mode=mode).double()
        with torch.no_grad():
            for p in env.parameters():
                p.add_(0.5 * torch.randn_like(p))
        X = torch.zeros(3, len(CHANNEL_NAMES), dtype=f64)
        X[1, :2] = torch.tensor([0.7, -0.4])                         # a dimer: two-body channels only
        X[2] = torch.randn(len(CHANNEL_NAMES), dtype=f64)
        u = env(X)
        assert all(float(u[k][0].abs()) == 0.0 for k in u)            # isolated atom
        assert float(u["b"][1].abs()) == 0.0 and float(u["c"][1].abs()) == 0.0   # no two-body input: dimer-exact
        assert float(u["a"][1].abs()) > 0.0
        assert len(env.trunks) == (1 if mode == "shared" else 2)
        env.zero_()
        assert all(float(v.abs().max()) == 0.0 for v in env(X).values())


def test_model_readouts_ablations_and_pinned_a0(setup):
    tab, chan, table, feat, make, keep = setup
    m = make()
    assert m.env.names == ["pauli", "disp", "kappa_dip", "kappa_quad", "kappa_breath", "qD", "qA"]
    m2 = make(xpol_kappa_head=False, ct_q_head=False, env_two_body=("pauli",), env_mode="per_term", pin_a0=True)
    assert m2.env.names == ["pauli", "disp"] and m2.env.two_body == {"pauli": True, "disp": False}
    assert set(m2.env.trunks) == {"pauli", "disp"}
    if m2.breath:
        assert not m2.breath_lna.requires_grad


def test_dimer_kappa_is_kappa0_without_two_body(setup):
    tab, chan, table, feat, make, keep = setup
    m = make(env_two_body=("pauli", "disp", "ct"))
    keep(m)
    X = torch.tensor([[0, 0, 0], [2.7, 0, 0]], dtype=f64)
    Z = torch.tensor([11, 17])
    bt = batched_precompute(X, Z, torch.zeros(2, dtype=torch.long), tab, table, chan, inv_form=m.inv_form)
    sp = bt["species"]
    m.env_prepare(bt, sp)
    k = m.xpol_kappa(m.scaled(bt["feats"]), sp)
    torch.testing.assert_close(k[:, 0], torch.exp(m.xpol_lnk[sp, 0]))
    u = m._env_out(sp)
    assert float(u["pauli"].abs().max()) > 0.0                      # the two-body readout does act on the dimer


def test_forces_through_the_channels(setup):
    tab, chan, table, feat, make, keep = setup
    m = make()
    X, Z = _cluster(seed=4)
    fr = torch.zeros(len(Z), dtype=torch.long)

    def energy(x):
        bt = batched_precompute(x, Z, fr, tab, table, chan, inv_form=m.inv_form)
        return m(bt)["e_int"].sum()

    x = X.clone().requires_grad_(True)
    e = energy(x)
    (g,) = torch.autograd.grad(e, x)
    keep(m)
    h = 1e-4
    for (a, c) in ((0, 0), (3, 2), (6, 1)):
        xp, xm = X.clone(), X.clone()
        xp[a, c] += h
        xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(xp) - energy(xm)) / (2 * h)
        assert float(g[a, c]) == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_field_branch_is_consistent_and_differentiable(setup):
    """--env-field: the CT readouts see |E|^2, |grad E|^2 of the permanent drive; forces stay derivatives."""
    tab, chan, table, feat, make, keep = setup
    m = make(env_field=True)
    assert m.env.extra == {"pauli": False, "disp": False, "kappa_dip": False, "kappa_quad": False,
                           "kappa_breath": False, "qD": True, "qA": True}
    X, Z = _cluster(seed=6)
    fr = torch.zeros(len(Z), dtype=torch.long)

    def energy(x):
        bt = batched_precompute(x, Z, fr, tab, table, chan, inv_form=m.inv_form)
        return m(bt)["e_int"].sum()

    x = X.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(energy(x), x)
    keep(m)
    h = 1e-4
    xp, xm = X.clone(), X.clone()
    xp[2, 1] += h
    xm[2, 1] -= h
    with torch.no_grad():
        fd = (energy(xp) - energy(xm)) / (2 * h)
    assert float(g[2, 1]) == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_ct_legacy_head_hybrid(setup):
    """--ct-legacy-head: the CT strengths come from head_ctq on the density invariants; no CT readouts."""
    tab, chan, table, feat, make, keep = setup
    m = make(ct_legacy_head=True)
    assert "qD" not in m.env.names and "qA" not in m.env.names
    X, Z = _cluster(seed=7)
    bt = batched_precompute(X, Z, torch.zeros(len(Z), dtype=torch.long), tab, table, chan, inv_form=m.inv_form)
    sp = bt["species"]
    I = m.scaled(bt["feats"])
    m.env_prepare(bt, sp)
    h = m.head_ctq(I, sp)
    c = m.ct_q_cap
    ref = torch.exp(m.ct_lnq0[sp] + (c * torch.tanh(h / c) if c > 0 else m.capped(h)))
    torch.testing.assert_close(m.ct_charges_from(I, sp), ref)


def test_diffuse_penetration_branch_and_curvature(setup):
    """--env-ct-pen: point charges removed from V1 / V2 (the far field of a dimer leaves nothing), the CT readouts
    read the 15 diffuse-probe invariants, and the curvature prior is finite and differentiable."""
    from rsfff.train.train_ionct import curvature_penalty
    from rsfff.ff.ionct.model import precompute
    tab, chan, table, feat, make, keep = setup
    m = make(env_ct_pen=True)
    assert m.env.extra["qD"] and m.env.extra["qA"] and not m.env.extra["pauli"]
    far = torch.tensor([[0.0, 0, 0], [9.0, 0, 0]], dtype=f64)
    bt = batched_precompute(far, torch.tensor([11, 17]), torch.zeros(2, dtype=torch.long), tab, table, chan,
                            inv_form=m.inv_form)
    pv = m.diffuse_pen_invariants(bt)
    full = (bt["env"]["acceptor:V1"] ** 2).sum(1)
    assert float(pv[:, 7].abs().max()) < 1e-6 * float(full.max())            # acceptor |V1_pen|^2 ~ 0 far away
    X, Z = _cluster(seed=8)
    sym = [tab.symbols[int(s)] for s in tab.species_index(Z)]
    c = precompute(sym, X.numpy(), tab, chan, feat, {"pol": 0.0, "ct": 0.0}, {}, lean=True, frontier=True)
    c.env = c.env or {}
    for p in m.parameters():
        p.grad = None
    cp = curvature_penalty(m, [c], tab, table, chan, 0.02)
    cp.backward()
    keep(m)
    assert torch.isfinite(cp) and float(cp) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.env.parameters())
