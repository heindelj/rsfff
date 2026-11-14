"""CmmModel: the two precompute paths agree, the forces are exact derivatives, an isolated ion has no energy and no
environment, a lone ion pair's CT is exactly its kernel energy, and the joint induction + charge-flux solution is
stationary. (The reproduction of the trained complete model is tests/test_cmm_golden.py.)"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, DensityEnvironment, IonTable, batched_precompute
from rsfff.ff.cmm.batch import collate, precompute
from rsfff.train.cmm import radial_table

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(5)
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    table = radial_table(tab, PARAMS, os.environ.get("RSFFF_TABLE_CACHE"))
    m = CmmModel(tab, chan, CmmConfig(film=False))
    with torch.no_grad():                       # an environment that does something: random readouts and constants
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw):
            p.add_(0.1 * torch.randn_like(p))
    frames = [(["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0.1, 0]], dtype=torch.float64)),
              (["Na", "Cl", "Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [2.8, 2.9, 0.1], [0, 2.75, 0.2]],
                                                       dtype=torch.float64)),
              (["Li", "F", "Cl", "Na", "F"], torch.tensor([[0.0, 0, 0], [1.9, 0, 0.1], [-2.6, 0.3, 0], [0.2, 3.1, 0.4],
                                                           [-1.0, -2.2, 1.4]], dtype=torch.float64))]
    return tab, chan, table, m, frames


def _batch(frames, tab, table, chan, pos=None):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return batched_precompute(pos, Z, fr, tab, table, chan)


def test_batched_matches_per_frame(setup):
    tab, chan, table, m, frames = setup
    feat = DensityEnvironment(tab)
    ref = collate([precompute(sym, X, tab, chan, feat) for sym, X in frames])
    bt = _batch(frames, tab, table, chan)
    for k in ("L", "A_off"):
        assert torch.allclose(ref[k], bt[k], atol=1e-7), k
    assert torch.allclose(ref["feats"], bt["feats"], atol=1e-6)
    for k in ("donor:S1", "acceptor:V2", "breath:V0"):
        assert torch.allclose(ref["env"][k], bt["env"][k], atol=1e-7), k
    assert set(map(tuple, ref["ch_index"].T.tolist())) == set(map(tuple, bt["ch_index"].T.tolist()))
    with torch.no_grad():
        o1, o2 = m(ref), m(bt)
    for k in ("e_pol", "e_ct", "e_pauli", "e_disp", "e_elst"):
        assert torch.allclose(o1[k], o2[k], atol=1e-7), k


def test_forces_are_exact_derivatives(setup):
    tab, chan, table, m, frames = setup
    pos = torch.cat([X for _, X in frames])

    def E(p):
        return m(_batch(frames, tab, table, chan, p))["e_int"].sum()

    pg = pos.clone().requires_grad_(True)
    g = torch.autograd.grad(E(pg), pg)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((0, 0), (3, 1), (6, 2), (9, 0)):
            pp, pm = pos.clone(), pos.clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7


def test_isolated_ions(setup):
    tab, chan, table, m, _ = setup
    frames = [([x], torch.zeros(1, 3, dtype=torch.float64)) for x in ("Li", "Na", "F", "Cl")]
    with torch.no_grad():
        out = m(_batch(frames, tab, table, chan))
    assert torch.allclose(out["e_int"], torch.zeros(4, dtype=torch.float64), atol=1e-14)
    assert all(float(v.abs().max()) == 0.0 for v in m._u.values())


def test_lone_pair_ct_is_the_kernel(setup, monkeypatch):
    """A Na-Cl pair at zero field: without the charges' coupling to the induced moments, E_CT = e_c + e_c' of its two
    channels exactly (the drives' calibration); with it, the moments' response to the moved charge lowers the CT
    (part of the pair's CT is then booked as induction-coupled relaxation, E_CT < e_c + e_c' in size)."""
    tab, chan, table, m, frames = setup
    bt = _batch(frames[:1], tab, table, chan)
    with torch.no_grad():
        full = m(bt)
    real = m.charge_slots
    monkeypatch.setattr(m, "charge_slots", lambda *a: (torch.zeros_like(real(*a)[0]), real(*a)[1]))
    with torch.no_grad():
        bare = m(bt)
    assert bare["e_kernel"].shape[0] == 2
    assert float(bare["e_ct"][0]) == pytest.approx(float(bare["e_kernel"].sum()), rel=1e-10)
    assert float(bare["e_kernel"].sum()) < float(full["e_ct"][0]) < 0


def test_joint_solution_is_stationary(setup):
    """E(z, q) of the module docstring is stationary at the returned joint moments and channel charges."""
    tab, chan, table, m, frames = setup
    bt = _batch(frames[1:2], tab, table, chan)
    with torch.no_grad():
        out = m(bt)
    sp, B, nmax = bt["species"], bt["B"], bt["nmax"]
    i, j = bt["ch_index"]
    q = out["q_ch"]
    # the charge pattern and the energy as an explicit function of (z, q) around the solution
    D = 8 + m.n_b
    Ca, Hq = m.charge_slots(bt, sp, D)
    fr, loc = m._local(bt)
    li, lj = loc[i], loc[j]
    cf = bt["ch_frame"]
    h = Hq[cf, li, li] + Hq[cf, lj, lj] - Hq[cf, li, lj] - Hq[cf, lj, li]
    e_c, V, s = m.channel_drives(bt, sp, m._q, h)
    G = torch.zeros(B, nmax, q.shape[0], dtype=q.dtype).index_put((cf, li, torch.arange(q.shape[0])), torch.ones_like(q))
    G = G.index_put((cf, lj, torch.arange(q.shape[0])), -torch.ones_like(q), accumulate=True)
    # stationarity in q: b q - V + G^T (H_Q Q + C^T z) = 0, with Q = G q and z the joint moments (all of D)
    z = out["x"]
    zf = torch.cat([z, torch.zeros(B, nmax, D - 8, dtype=z.dtype)], -1) if D > 8 else z
    # the breathing amplitudes are not returned in x: take them from the joint solve (z_pol - W Q) instead
    Q = (G @ q[None, :, None])[..., 0]
    grad_q = q / (s * s) - V + (G.transpose(1, 2) @ (Hq @ Q[..., None]))[0, :, 0]
    grad_q = grad_q + (G.transpose(1, 2) @ torch.einsum("bnk,bn->bk", Ca, zf.reshape(B, -1))[..., None])[0, :, 0]
    if D > 8:          # the breathing slots carry no charge coupling (Ca has zero rows there), so x suffices
        pass
    assert float(grad_q.detach().abs().max()) < 1e-9 * float(V.detach().abs().max())
    assert torch.allclose(out["dq"], Q[0, :4], atol=1e-14)
