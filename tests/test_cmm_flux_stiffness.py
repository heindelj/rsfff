"""The charge flux's stabilizers. ``CmmConfig.ct_q_onesided``: the environment can only lower the CT charges.
``CmmConfig.ct_floor``, ``CmmConfig.ct_relay``: a lone pair's CT is still exactly
its kernel energy; the returned channel charges are stationary for the energy with both terms; the 2N (donated,
accepted) elimination is the dense channel-space minimum; forces and the polarizability stay exact; the periodic CG
solve is the cluster's; and the floor bounds the gain of a ring current over the pair kernels."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
VARIANTS = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.5), (0.15, 0.4), (0.05, 1.3)]      # (ct_floor, ct_relay)

RING = (["Cl", "Na", "Cl", "Na", "Cl", "Li"],
        torch.tensor([[2.9, 0.1, 0.0], [1.4, 2.5, 0.1], [-1.5, 2.4, -0.1], [-2.8, 0.0, 0.2], [-1.4, -2.5, 0.0],
                      [1.5, -2.2, -0.2]], dtype=f64))
PAIR = (["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0.1, 0]], dtype=f64))


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, floor=0.0, relay=0.0, q_boost=0.0):
    torch.manual_seed(5)
    m = CmmModel(*tables, CmmConfig(film=False, ct_floor=floor, ct_relay=relay))
    with torch.no_grad():                       # an environment that does something: random readouts and constants
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw):
            p.add_(0.1 * torch.randn_like(p))
        m.ct_ln_q0.add_(q_boost)
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def _incidence(m, bt):
    """G (signed) and |G| (N, C) of a one-frame batch."""
    i, j = bt["ch_index"]
    C, n = i.shape[0], bt["species"].shape[0]
    ar = torch.arange(C)
    out = torch.zeros(n, C, dtype=f64).index_put((i, ar), torch.ones(C, dtype=f64))
    inn = torch.zeros(n, C, dtype=f64).index_put((j, ar), torch.ones(C, dtype=f64))
    return out - inn, out + inn


@pytest.mark.parametrize("floor,relay", VARIANTS)
def test_lone_pair_ct_is_still_the_kernel(tables, floor, relay):
    m = _model(tables, floor, relay)
    bt = _batch(m, [PAIR])
    with torch.no_grad():
        bare = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
    assert bare["e_kernel"].shape[0] == 2
    assert float(bare["e_ct"][0]) == pytest.approx(float(bare["e_kernel"].sum()), rel=1e-10)


@pytest.mark.parametrize("floor,relay", VARIANTS)
def test_channel_charges_are_stationary_and_the_dense_minimum(tables, floor, relay):
    """grad_q of [sum_c 1/2 b_c q_c^2 - V_c q_c] + 1/2 Q H_Q Q + Q C^T z + 1/2 T tau T vanishes at the returned (z, q);
    and with the moments decoupled (Ca = 0) q is the solution of the dense channel-space system."""
    m = _model(tables, floor, relay)
    bt = _batch(m, [RING])
    with torch.no_grad():
        out = m(bt)
        sp = bt["species"]
        n = sp.shape[0]
        G, aG = _incidence(m, bt)
        Hq = bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))
        tau = m.slot_tau(sp)
        tau = torch.zeros(n, dtype=f64) if tau is None else tau
        e_c, V, s = m.channel_drives(bt, sp, m._q, m.channel_curvature(bt, sp))
        q = out["q_ch"]
        b = 1 / (s * s)
        zf = torch.cat([out["x"], torch.zeros(1, bt["nmax"], bt["D"] - 8, dtype=f64)], -1).reshape(1, -1)
        psi_z = torch.einsum("bnk,bn->bk", bt["Ca"], zf)[0, :n]          # C^T z (the breathing rows of Ca are zero)
        grad = b * q - V + G.T @ (Hq @ (G @ q) + psi_z) + aG.T @ (tau * (aG @ q))
        assert float(grad.abs().max()) < 1e-9 * float(V.abs().max())
        assert torch.allclose(out["dq"], G @ q, atol=1e-13)
        if floor > 0:
            e_min = (e_c.abs() / (m.ct_ln_v0.exp()[sp[bt["ch_index"][0]]] + m.ct_ln_v0.exp()[sp[bt["ch_index"][1]]]) ** 2)
            assert float((b - 0.5 / e_min.clamp(min=1e-300)).min()) >= floor * (1 - 1e-9)      # b = V0^2 / 2|e| + floor
        bare = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
        Hc = torch.diag(b) + G.T @ Hq @ G + aG.T @ torch.diag(tau) @ aG
        q_dense = torch.linalg.solve(Hc, V)
        assert torch.allclose(bare["q_ch"], q_dense, rtol=1e-9, atol=1e-13)
        assert float(bare["e_ct"][0]) == pytest.approx(float(-0.5 * (V * q_dense).sum()), rel=1e-10)


def test_forces_and_alpha_with_both_terms(tables):
    m = _model(tables, 0.15, 0.4)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    out = m(_batch(m, [RING], pos))
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, k in ((0, 0), (3, 1), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[a, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_int"]))
            assert float(F[a, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-9)
        bt = _batch(m, [RING])
        o = m(bt, alpha=True)
        hf, E = 1e-3, {}
        for a in range(3):
            for sg in (1, -1):
                Fd = torch.zeros(1, 3, dtype=f64)
                Fd[:, a] = sg * hf
                r = m(bt, field=Fd)
                E[a, sg] = r["e_pol"] + r["e_ct"]
        r0 = m(bt)
        fd = torch.stack([-(E[a, 1] + E[a, -1] - 2 * (r0["e_pol"] + r0["e_ct"])) / hf ** 2 for a in range(3)], 1)
        assert torch.allclose(torch.diagonal(o["alpha"], dim1=1, dim2=2), fd, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("floor,relay", [(0.15, 0.0), (0.0, 0.5), (0.15, 0.4)])
def test_periodic_solve_is_the_cluster(tables, floor, relay):
    """An inversion-symmetric neutral cluster in cubes of 30 and 40 A extrapolates (1/L^5) to the cluster."""
    m = _model(tables, floor, relay)
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym = ["Na", "Cl", "Li", "F"] * 2
    X = torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym, X)]))
        p = {L: m(_batch(m, [(sym, X + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False))
             for L in (30., 40.)}
    w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
    for k in ("e_pol", "e_ct", "e_int"):
        ext = p[30.][k] + w * (p[40.][k] - p[30.][k])
        assert float((ext - c[k]).abs()) < 2e-7, k
    assert float(c["e_ct"].abs()) > 1e-4


def test_floor_bounds_the_ring_gain(tables):
    """Strong kernels (the CT charges x e^1.5) on a six-ring, moments decoupled: without the floor the flux gains far
    more than the pair kernels; with it the gain is below sum_c |e_c| (1 + (h_c + tau_c) / b_min)."""
    gains = {}
    for floor in (0.0, 0.2):
        m = _model(tables, floor, 0.0, q_boost=1.5)
        bt = _batch(m, [RING])
        with torch.no_grad():
            o = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
            sp = bt["species"]
            h = m.channel_curvature(bt, sp)
            e = o["e_kernel"].abs()
            gains[floor] = float(-o["e_ct"][0] / e.sum())
            if floor:
                assert float(-o["e_ct"][0]) <= float((e * (1 + h / floor)).sum())
    assert gains[0.2] < gains[0.0]


def test_onesided_charges_never_exceed_the_isolated_ones(tables):
    """With ``ct_q_onesided`` every CT charge is at most its isolated-species value (so every kernel is at most the
    bare pair kernel), the factors that were already below 1 are untouched, and the gradient still flows at u = 0."""
    m = _model(tables)
    bt = _batch(m, [RING, PAIR])
    with torch.no_grad():
        free = m(bt)["q_ct"]
        m.config.ct_q_onesided = True
        one = m(bt)["q_ct"]
    q0 = m.ct_ln_q0.exp()[bt["species"]]
    assert bool((free > q0 * (1 + 1e-9)).any())                          # the two-sided factors do raise some charges
    assert bool((one <= q0 * (1 + 1e-12)).all())
    low = free < q0
    assert torch.allclose(one[low], free[low], rtol=1e-12)
    m._u = {k: torch.zeros(3, dtype=f64, requires_grad=True) for k in ("qD", "qA")}
    q = m.ct_charges(torch.zeros(3, dtype=torch.long))
    g = torch.autograd.grad(q.sum(), m._u["qD"])[0]
    assert bool((g > 0).all())
    m.config.ct_q_onesided = False
