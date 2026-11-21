"""The gap model's per-frame polarized gap (``ct_gap_min``: the frozen gap plus the CT-free potentials, the lowest
over a frame's channels) and the training switch ``skip_negative_gap``: frames with a negative gap are not iterated
and are flagged (``ct_gap_dropped``); the others are untouched, and nothing changes with the switch off. The
exclusion solve iterates only the frames still iterating, with the same result as iterating all."""
from __future__ import annotations

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

from test_cmm_ct_site import SITE

f64 = torch.float64


def _model():
    tab = IonTable.load(SITE, symbols=("Li", "Na", "F", "Cl"))
    torch.manual_seed(0)
    cfg = CmmConfig(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
                    ct_gap_converge=40, ct_exclusion=True)
    return CmmModel(tab, ChannelTable.direct(("Li", "Na", "F", "Cl")), cfg)


def _batch(m, compress):
    """Three frames: Na2Cl2, a Li2F+ trimer compressed by ``compress`` (closes its gap), NaCl."""
    frames = [(["Na", "Cl", "Na", "Cl"], [[0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]]),
              (["Li", "Li", "F"], [[-1.87 * compress, -0.06, -1.17 * compress], [-1.81 * compress, 0.07, 1.08 * compress],
                                   [1.82 * compress, -1.2 * compress, -0.06]]),
              (["Na", "Cl"], [[0, 0, 0], [2.4, 0, 0]])]
    X = torch.tensor([x for _, xs in frames for x in xs], dtype=f64)
    Z = torch.tensor([ATOMIC_NUMBER[a] for s, _ in frames for a in s])
    fr = torch.cat([torch.full((len(s),), k) for k, (s, _) in enumerate(frames)])
    return X, Z, fr


@pytest.fixture(scope="module")
def model():
    return _model()


def test_gap_min_and_switch_off_unchanged(model):
    m = model
    X, Z, fr = _batch(m, 0.8)
    o = m(m.precompute(X, Z, fr))
    assert o["ct_gap_min"].shape == (3,) and not bool(o["ct_gap_dropped"].any())
    g = m._gap_frames["gap_min"]
    assert torch.equal(g, o["ct_gap_min"])


def test_negative_gap_frames_are_skipped_only(model):
    m = model
    X, Z, fr = _batch(m, 0.8)
    kF = list(m.tab.symbols).index("F")
    I0 = m.ct_gap_I.clone()
    o0 = m(m.precompute(X, Z, fr))
    shift = float(o0["ct_gap_min"][1]) + 0.02           # close the trimer's gap (F is its only donor)
    with torch.no_grad():
        m.ct_gap_I[kF] -= shift
    try:
        o0 = m(m.precompute(X, Z, fr))
        neg = o0["ct_gap_min"] < 0
        assert neg.tolist() == [False, True, False]
        m.skip_negative_gap = True
        Xg = X.clone().requires_grad_(True)
        o1 = m(m.precompute(Xg, Z, fr))
        assert torch.equal(o1["ct_gap_dropped"], neg)
        keep = ~neg
        assert torch.allclose(o1["e_int"][keep], o0["e_int"][keep], rtol=0, atol=1e-12)
        assert bool(torch.isfinite(o1["e_int"]).all())
        g = torch.autograd.grad((o1["e_int"] * keep).sum(), Xg)[0]
        assert bool(torch.isfinite(g).all())
    finally:
        m.skip_negative_gap = False
        with torch.no_grad():
            m.ct_gap_I.copy_(I0)


def test_subset_iteration_matches_full(model, monkeypatch):
    """Iterating every frame each time (the old loop: the frames of a finished one are never changed) and only the
    active ones give the same charges."""
    m = model
    X, Z, fr = _batch(m, 1.0)
    bt = m.precompute(X, Z, fr)
    o1 = m(bt)
    calls = []
    orig = CmmModel._excl_ops

    def spy(self, cf, li, lj, live, B, N, dt):
        calls.append(B)
        return orig(self, cf, li, lj, live, B, N, dt)

    monkeypatch.setattr(CmmModel, "_excl_ops", spy)
    o2 = m(bt)
    assert torch.allclose(o1["e_ct"], o2["e_ct"], rtol=0, atol=1e-14)
    assert calls[0] == 3 and (len(calls) == 1 or min(calls[1:]) <= 3)
