"""The frontier-overlap CT kernel (``CmmConfig.ct_kernel = "frontier"``, rsfff.ff.cmm.ct_overlap): W^2 = qD qA (S g)^2
with S the tabulated overlap of the free ions' frontier orbitals and g each pair's own C-infinity cutoff between the
distances at which |S| is 0.06 and 0.02."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable, load_cmm, save_cmm
from rsfff.ff.cmm.ct_overlap import load_ct_overlap, overlap_kernel, switch_radii
from rsfff.ff.cmm.io import resolve_params

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
TABLE = "ct_overlap_wb97mv_tzvpd.json"
f64 = torch.float64
SYM = ["Li", "Na", "F", "Cl"]


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


@pytest.fixture(scope="module")
def ov():
    return load_ct_overlap(resolve_params(TABLE), SYM)


def _model(tables, **kw):
    torch.manual_seed(7)
    cfg = dict(film=False, ct_model="gap", ct_gap_overlap="pauli", ct_gap_w0=0.005, ct_gap_converge=40,
               ct_gap_tol=1e-12, ct_kernel="frontier", ct_q_headroom=0.2, env_switch="smooth")
    cfg.update(kw)
    m = CmmModel(*tables, CmmConfig(**cfg))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def _dimers(a, b, R):
    return [([a, b], torch.tensor([[0.0, 0, 0], [0, 0, float(r)]], dtype=f64)) for r in R]


def _rel_jump(y, k, h):
    """The largest change of the k-th difference quotient between neighbouring points over its largest size."""
    D = torch.diff(y, n=k) / h ** k
    return float(torch.diff(D).abs().max() / D.abs().max())


# -- the table -----------------------------------------------------------------------------------------------------
def test_table_and_switch_radii(ov):
    d = json.load(open(resolve_params(TABLE)))
    r = d["r0"] + d["h"] * np.arange(d["n"])
    iF, iCl, iNa, iLi = SYM.index("F"), SYM.index("Cl"), SYM.index("Na"), SYM.index("Li")
    S = np.array(d["S"]["Cl>Na"])
    on, off = float(ov["r_on"][iCl, iNa]), float(ov["r_off"][iCl, iNa])
    assert 5.0 < on < 5.4 and 6.3 < off < 6.7                           # the anion -> cation pairs end at 5.7 - 6.5 A
    assert abs(np.interp(on, r, np.abs(S)) - 0.06) < 1e-4 and abs(np.interp(off, r, np.abs(S)) - 0.02) < 1e-4
    assert bool((np.abs(S[r > off + d["h"]]) < 0.02).all())             # the outermost crossing
    assert float(ov["r_off"].max()) < 7.0 and float(ov["r_off"][iF, iLi]) < float(ov["r_off"][iCl, iLi])
    # the spline goes through the nodes; S^2 inside the switch
    for a, b in ((iCl, iNa), (iLi, iF), (iCl, iCl)):
        y = torch.tensor(d["S"][f"{SYM[a]}>{SYM[b]}"], dtype=f64)
        k = torch.arange(20, 200, 7)
        rr = torch.tensor(r, dtype=f64)[k]
        K = overlap_kernel(ov["coef"], ov["grid"], ov["r_on"], ov["r_off"], torch.full_like(k, a), torch.full_like(k, b), rr)
        inside = rr <= ov["r_on"][a, b]
        assert torch.allclose(K[inside], y[k][inside] ** 2, rtol=1e-12, atol=1e-16)
        assert bool((K[rr >= ov["r_off"][a, b]] == 0).all()) and bool((K <= y[k] ** 2 + 1e-16).all())
    # the rule itself
    x = np.linspace(1.0, 9.0, 801)
    assert switch_radii(0.3 * np.exp(-0.8 * (x - 1.0)), x, 0.06, 0.02) == pytest.approx(
        (1.0 + np.log(5.0) / 0.8, 1.0 + np.log(15.0) / 0.8), abs=2e-4)
    assert switch_radii(0.01 * np.exp(-x), x, 0.06, 0.02) == (0.0, 0.0)                      # never a channel
    on2, off2 = switch_radii(0.05 * np.exp(-0.5 * (x - 3.0) ** 2), x, 0.06, 0.02)
    assert on2 == pytest.approx(3.0, abs=0.011) and off2 > 4.3                                # from its maximum
    with pytest.raises(ValueError):
        switch_radii(np.full_like(x, 0.03), x, 0.06, 0.02)                                   # the grid ends too early
    with pytest.raises(ValueError):
        load_ct_overlap(resolve_params(TABLE), SYM, window=(0.02, 0.06))
    with pytest.raises(KeyError):
        load_ct_overlap(resolve_params(TABLE), SYM + ["Xe"])


def test_kernel_is_smooth_and_compact(ov):
    a, b = SYM.index("Cl"), SYM.index("Na")
    h = 0.004
    r = torch.arange(3.0, 7.5, h, dtype=f64, requires_grad=True)
    si, sj = torch.full((len(r),), a), torch.full((len(r),), b)
    K = overlap_kernel(ov["coef"], ov["grid"], ov["r_on"], ov["r_off"], si, sj, r)
    g1, = torch.autograd.grad(K.sum(), r, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), r)
    out = r.detach() >= ov["r_off"][a, b]
    assert bool((K[out] == 0).all()) and bool((g1[out] == 0).all()) and bool((g2[out] == 0).all())
    assert bool(torch.isfinite(g2).all()) and bool((torch.diff(K.detach()) <= 1e-18).all())      # decays monotonically
    # no jump of the value, slope or curvature anywhere (the switch's ends included)
    y = K.detach()
    assert _rel_jump(y, 1, h) < 0.05 and _rel_jump(y, 2, h) < 0.05
    assert torch.allclose(g1.detach()[1:-1], (y[2:] - y[:-2]) / (2 * h), atol=2e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        CmmConfig(ct_kernel="frontier")                               # needs the gap model
    with pytest.raises(ValueError):
        CmmConfig(ct_model="gap", ct_kernel="sto")
    with pytest.raises(ValueError):
        CmmConfig(ct_model="gap", ct_kernel="frontier", ct_overlap_window=(0.02, 0.02))
    c = CmmConfig()
    assert (c.ct_kernel, c.ct_overlap_window) == ("slater", (0.06, 0.02))
    d = CmmConfig(ct_model="gap", ct_kernel="frontier", ct_overlap_window=[0.08, 0.03]).to_dict()
    assert CmmConfig.from_dict(json.loads(json.dumps(d))).ct_overlap_window == (0.08, 0.03)


# -- in the model --------------------------------------------------------------------------------------------------
def test_channels_end_at_their_pair_cutoff(tables):
    m, m0 = _model(tables), _model(tables, ct_kernel="slater")
    assert m.frontier and not m0.frontier and "ctk_coef" in m.state_dict() and "ctk_coef" not in m0.state_dict()
    assert m.chan.cutoff == pytest.approx(float(m.ctk_roff.max())) and m0.chan.cutoff == 8.0
    g = torch.Generator().manual_seed(2)
    sym = ["Na", "Cl", "Li", "F"] * 4
    X = 9.0 * torch.rand(16, 3, generator=g, dtype=f64)
    bt, bt0 = _batch(m, [(sym, X)]), _batch(m0, [(sym, X)])
    i, j = bt["ch_index"]
    sp = bt["species"]
    assert bool((bt["ch_r"] < m.ctk_roff[sp[i], sp[j]]).all())                       # only pairs inside their cutoff
    d = (X[:, None] - X[None]).norm(dim=2)
    n = int(((d < m.ctk_roff[sp[:, None], sp[None]]) & ~torch.eye(16, dtype=torch.bool)).sum())
    assert i.numel() == n and n < bt0["ch_index"].shape[1]                           # fewer than the 8 A list
    with torch.no_grad():
        o = m(bt)
        W2 = m.gap_channels(bt, sp)["W2"]
    assert bool((W2 >= 0).all()) and float(o["ct_gap_resid"].max()) < 1e-9 and bool(torch.isfinite(o["e_int"]).all())
    # the coupling: qD qA S^2 inside the switch
    k = bt["ch_r"] < m.ctk_ron[sp[i], sp[j]]
    S2 = overlap_kernel(m.ctk_coef, m.ctk_grid, m.ctk_ron, m.ctk_roff, sp[i], sp[j], bt["ch_r"])
    assert bool(k.any()) and torch.allclose(W2, m._q[i, 0] * m._q[j, 1] * S2, rtol=1e-13)


def test_pair_dissociates_smoothly_with_no_ct_beyond_its_cutoff(tables):
    """Na-Cl through its switch (5.2 - 6.5 A): E_CT and its first two derivatives are continuous, E_CT is exactly zero
    beyond the cutoff, and the forces are the derivative of the energy inside the window."""
    m = _model(tables)
    h = 0.004
    R = torch.arange(4.4, 7.4, h, dtype=f64)
    with torch.no_grad():
        o = m(_batch(m, _dimers("Na", "Cl", R)))
    e = o["e_ct"]
    off = float(m.ctk_roff[m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")])
    assert bool((e[R >= off] == 0).all()) and float(e[R < off - 0.5].max()) < -1e-6
    assert _rel_jump(e, 1, h) < 0.05 and _rel_jump(e, 2, h) < 0.08
    for r0 in (5.6, 6.2, off - 0.02, off + 0.05):
        d = 1e-4
        Rs = torch.tensor([r0 - d, r0, r0 + d], dtype=f64)
        pos = torch.cat([X for _, X in _dimers("Na", "Cl", Rs)]).requires_grad_(True)
        ei = m(_batch(m, _dimers("Na", "Cl", Rs), pos))["e_int"]
        g, = torch.autograd.grad(ei[1], pos)
        fd = float((ei[2] - ei[0]) / (2 * d))
        assert abs(float(g[3, 2]) - fd) < 1e-6 * max(1e-3, abs(fd)), r0


def test_cluster_forces_exclusion_and_periodic(tables):
    m = _model(tables, ct_exclusion=True)
    g = torch.Generator().manual_seed(4)
    sym = ["Na", "Cl", "Li", "F", "Cl", "Na"]
    X = torch.tensor([[0, 0, 0], [2.7, 0.2, 0], [0.3, 2.4, 0.4], [2.9, 2.6, -0.2], [-2.6, 0.3, 0.5], [5.9, 0.4, 0.3]], dtype=f64)
    pos = X.clone().requires_grad_(True)
    o = m(_batch(m, [(sym, X)], pos))
    assert float(o["ct_gap_resid"].max()) < 1e-9
    F, = torch.autograd.grad(o["e_int"].sum(), pos)
    with torch.no_grad():
        for at, k in ((0, 0), (5, 1)):
            e = []
            for s in (1, -1):
                Xp = X.clone()
                Xp[at, k] += s * 1e-4
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_int"]))
            assert float(F[at, k]) == pytest.approx((e[0] - e[1]) / 2e-4, rel=2e-6, abs=1e-9)
    # a cell: the same channel rule on the images
    a = 5.6
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    Xc = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)]) + 0.12 * torch.randn(8, 3, generator=g, dtype=f64)
    symc, cell = ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a
    posc = Xc.clone().requires_grad_(True)
    bt = _batch(m, [(symc, Xc)], posc, cell=cell)
    i, j = bt["ch_index"]
    assert bool((bt["ch_r"] < m.ctk_roff[bt["species"][i], bt["species"][j]]).all())
    oc = m(bt)
    assert float(oc["ct_gap_resid"].max()) < 1e-8 and float(oc["e_ct"]) < 0
    Fc, = torch.autograd.grad(oc["e_int"].sum(), posc)
    with torch.no_grad():
        e = []
        for s in (1, -1):
            Xp = Xc.clone()
            Xp[5, 2] += s * 1e-4
            e.append(float(m(_batch(m, [(symc, Xp)], cell=cell))["e_int"]))
    assert float(Fc[5, 2]) == pytest.approx((e[0] - e[1]) / 2e-4, rel=5e-6, abs=1e-8)


def test_checkpoint_carries_the_table(tables, tmp_path):
    m = _model(tables)
    with torch.no_grad():
        m.ctk_roff.mul_(0.97)                    # not the file's: the checkpoint's own must come back
    bt = _batch(m, _dimers("Na", "Cl", [2.8, 5.9]))
    with torch.no_grad():
        e = m(bt)["e_ct"]
    save_cmm(tmp_path / "m.pt", m, params=PARAMS)
    m2 = load_cmm(tmp_path / "m.pt")
    assert m2.frontier and torch.equal(m2.ctk_roff, m.ctk_roff) and torch.equal(m2.ctk_coef, m.ctk_coef)
    assert m2.chan.cutoff == pytest.approx(float(m.ctk_roff.max())) and m2.chan.cutoff < m.chan.cutoff
    with torch.no_grad():
        assert torch.allclose(m2(_batch(m2, _dimers("Na", "Cl", [2.8, 5.9])))["e_ct"], e, rtol=1e-12, atol=1e-16)
    assert "frontier" in m.summary() and m.ct_acc_raw.grad is None


# -- the trainer's frame weights -------------------------------------------------------------------------------------
def test_pair_weights(tables):
    """Two-ion frames with an anion count fully while their pair's overlap is above 1.5 s_on, not at all from where
    its switch starts; cation pairs, larger frames and models without the kernel are left alone."""
    from rsfff.train import cmm_data as D
    from rsfff.train.cmm import summarize, wmean
    m, m0 = _model(tables), _model(tables, ct_kernel="slater")
    iCl, iNa = m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")
    on, off = float(m.ctk_ron[iCl, iNa]), float(m.ctk_roff[iCl, iNa])
    assert float(m.ctk_ron[iNa, iCl]) < on                              # the anion -> cation direction decides
    d = json.load(open(resolve_params(TABLE)))
    r = d["r0"] + d["h"] * np.arange(d["n"])
    rw = switch_radii(np.array(d["S"]["Cl>Na"]), r, 0.09, 0.06)[0]
    assert 0.2 < on - rw < 1.0
    Rs = [2.8, rw - 0.01, 0.5 * (rw + on), on - 0.3 * (on - rw), on + 0.01, off + 0.5]
    mk = lambda: [D.as_frame(sym, X, {"int": -0.1, "pol": -0.01, "ct": -0.001}, {"k": k})       # noqa: E731
                  for k, (sym, X) in enumerate(_dimers("Na", "Cl", Rs) + _dimers("Cl", "Na", [on + 0.01]))] + [
        D.as_frame(["Na", "Cl", "Na"], [[0, 0, 0], [0, 0, on + 1.0], [0, 0, 2 * on + 2.0]], {"int": -0.1}, {"k": 9}),
        D.as_frame(["Na", "Na"], [[0, 0, 0], [0, 0, 3.0]], {"int": 0.1}, {"k": 10})]      # no anion: never weighted
    fr = mk()
    n = D.set_pair_weights(m, fr, 1.5)
    w = [f.labels.get("weight", 1.0) for f in fr]
    assert n == {"pairs": 7, "reduced": 2, "dropped": 3}
    assert w[0] == w[1] == 1.0 and w[2] == pytest.approx(0.5) and 0 < w[3] < 0.2 and w[4] == w[5] == w[6] == 0.0
    assert w[7] == 1.0 and w[8] == 1.0 and float(m.ctk_ron[iNa, iNa]) < 3.0
    fr0 = mk()
    assert D.set_pair_weights(m0, fr0, 1.5)["pairs"] == 0 and D.set_pair_weights(m, mk(), 0.0)["pairs"] == 0
    assert all("weight" not in f.labels for f in fr0)
    with pytest.raises(ValueError):
        D.set_pair_weights(m, mk(), 0.9)
    _, bt = D.batch_of(fr[:7], m)
    assert torch.equal(bt["labels"]["weight"], torch.tensor(w[:7], dtype=f64))
    assert torch.equal(D.batch_of(fr0[:2], m0)[1]["labels"]["weight"], torch.ones(2, dtype=f64))
    x = torch.arange(7, dtype=f64)
    assert float(wmean(x, bt["labels"]["weight"])) == pytest.approx(float((x * torch.tensor(w[:7])).sum() / 7))
    assert float(wmean(x, torch.ones(7, dtype=f64))) == float(x.mean())
    # the reported MAEs leave the frames of weight 0 out
    rows = [dict(pol=1.0, pol_ref=0.0, ct=0.0, ct_ref=0.0, ct_mask=True, etot=e, etot_ref=0.0, full=True, n=2, weight=wk)
            for e, wk in ((1.0, 1.0), (3.0, 0.5), (50.0, 0.0))]
    s = summarize(rows, cap=20.0)["all"]
    assert s["etot_mae"] == pytest.approx((1.0 + 1.5) / 1.5) and s["etot_mae_raw"] == pytest.approx(18.0)
    assert s["n"] == 3 and s["n_w"] == 1.5 and D._mae(np.array([1.0, 30.0]), 20.0) == 10.5
    assert D._mae(np.array([1.0, 2.0]), 0.0, np.zeros(2)) == 0.0


@pytest.mark.parametrize("excl", [False, True])
def test_force_loss_gradient_is_finite_at_the_end_of_a_switch(tables, excl):
    """A channel at the end of its switch has W^2 = 0 exactly or of order 1e-200: the gradient of a loss on the forces
    (second order in the positions) must stay finite. (Without exclusion the unselected side of a where carried
    1 / W^2: every force step of a training run was non-finite.)"""
    m = _model(tables, ct_exclusion=excl).train()
    iCl, iNa = m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")
    off = float(m.ctk_roff[iCl, iNa])
    seen_zero = seen_tiny = False
    for R in (off - 0.3, off - 0.05, off - 0.01, off - 2e-3, off - 1e-4, off - 1e-6):
        frames = _dimers("Na", "Cl", [R]) + [(["Na", "Cl", "Na", "Cl"], torch.tensor(
            [[0, 0, 0], [0, 0, 2.7], [0, 2.9, 0.2], [0.3, 0.1, R]], dtype=f64))]
        pos = torch.cat([X for _, X in frames]).requires_grad_(True)
        bt = _batch(m, frames, pos)
        o = m(bt)
        W2 = m.gap_channels(bt, bt["species"])["W2"].detach()
        seen_zero |= bool((W2 == 0).any())
        seen_tiny |= bool(((W2 > 0) & (W2 < 1e-30)).any())
        F, = torch.autograd.grad(o["e_int"].sum(), pos, create_graph=True)
        m.zero_grad()
        ((F ** 2).sum() + (o["e_int"] ** 2).sum()).backward()
        assert bool(torch.isfinite(F).all()), R
        for n, p in m.named_parameters():
            assert p.grad is None or bool(torch.isfinite(p.grad).all()), (R, n)
    assert seen_zero and seen_tiny


# -- the per-pair overlap switch on the learned-exponent kernel (ct_cut = "overlap") -------------------------------

def _slater_cut(tables, **kw):
    return _model(tables, ct_kernel="slater", ct_cut="overlap", **kw)


def test_overlap_cut_config():
    assert CmmConfig().ct_cut == "taper"
    with pytest.raises(ValueError, match="ct_cut"):
        CmmConfig(ct_cut="pair")
    with pytest.raises(ValueError, match="gap"):
        CmmConfig(ct_cut="overlap")
    assert CmmConfig.from_dict(CmmConfig(ct_model="gap", ct_cut="overlap").to_dict()).ct_cut == "overlap"


def test_slater_kernel_with_the_overlap_switch(tables):
    """The learned-exponent coupling qD qA f / r inside each pair's R_on, times g^2 through its switch, no channel
    beyond R_off; the table's taper is not applied; the frontier kernel's own cutoffs and pair weights."""
    from rsfff.ff.cmm.ct_overlap import pair_switch
    m, mt, mf = _slater_cut(tables), _model(tables, ct_kernel="slater"), _model(tables)
    assert m.overlap_cut and not m.frontier and not mt.overlap_cut and mf.overlap_cut and "ctk_roff" in m.state_dict()
    assert torch.equal(m.ctk_roff, mf.ctk_roff) and m.chan.types == mf.chan.types
    mt.load_state_dict(m.state_dict(), strict=False)                    # the same parameters with the 8 A taper
    g = torch.Generator().manual_seed(3)
    sym = ["Na", "Cl", "Li", "F"] * 4
    X = 9.0 * torch.rand(16, 3, generator=g, dtype=f64)
    bt, btt = _batch(m, [(sym, X)]), _batch(mt, [(sym, X)])
    sp = bt["species"]
    i, j = bt["ch_index"]
    assert bool((bt["ch_r"] < m.ctk_roff[sp[i], sp[j]]).all()) and i.numel() < btt["ch_index"].shape[1]
    with torch.no_grad():
        o, ot = m(bt), mt(btt)
        W2, W2t = m.gap_channels(bt, sp)["W2"], mt.gap_channels(btt, sp)["W2"]
    assert float(o["ct_gap_resid"].max()) < 1e-9 and bool(torch.isfinite(o["e_int"]).all())
    # match the channels of the two lists: W2 = (slater, no taper) x g^2
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(*btt["ch_index"]))}
    kt = torch.tensor([key[int(a), int(b)] for a, b in zip(i, j)])
    bare = W2t[kt] / torch.exp(btt["ch_ltaper"][kt])
    gsw = pair_switch(m.ctk_ron, m.ctk_roff, sp[i], sp[j], bt["ch_r"])
    inside = bt["ch_r"] < m.ctk_ron[sp[i], sp[j]]
    assert bool(inside.any()) and bool((~inside).any())
    assert torch.allclose(W2, bare * gsw ** 2, rtol=1e-12, atol=1e-30) and torch.allclose(W2[inside], bare[inside], rtol=1e-12)
    assert "slater, overlap switch" in m.summary()


def test_slater_cut_pair_dissociates_smoothly(tables):
    m = _slater_cut(tables)
    h = 0.004
    R = torch.arange(4.4, 7.4, h, dtype=f64)
    with torch.no_grad():
        e = m(_batch(m, _dimers("Na", "Cl", R)))["e_ct"]
    off = float(m.ctk_roff[m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")])
    assert bool((e[R >= off] == 0).all()) and float(e[R < off - 0.5].max()) < -1e-7
    assert _rel_jump(e, 1, h) < 0.05 and _rel_jump(e, 2, h) < 0.08
    for r0 in (3.0, 5.6, off - 0.02, off + 0.05):
        d = 1e-4
        Rs = torch.tensor([r0 - d, r0, r0 + d], dtype=f64)
        pos = torch.cat([X for _, X in _dimers("Na", "Cl", Rs)]).requires_grad_(True)
        ei = m(_batch(m, _dimers("Na", "Cl", Rs), pos))["e_int"]
        gr, = torch.autograd.grad(ei[1], pos)
        fd = float((ei[2] - ei[0]) / (2 * d))
        assert abs(float(gr[3, 2]) - fd) < 1e-6 * max(1e-3, abs(fd)), r0


@pytest.mark.parametrize("excl", [False, True])
def test_slater_cut_force_loss_gradient_is_finite_at_the_end_of_a_switch(tables, excl):
    m = _slater_cut(tables, ct_exclusion=excl).train()
    off = float(m.ctk_roff[m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")])
    for R in (off - 0.3, off - 0.01, off - 1e-4, off - 1e-6):
        frames = _dimers("Na", "Cl", [R]) + [(["Na", "Cl", "Na", "Cl"], torch.tensor(
            [[0, 0, 0], [0, 0, 2.7], [0, 2.9, 0.2], [0.3, 0.1, R]], dtype=f64))]
        pos = torch.cat([X for _, X in frames]).requires_grad_(True)
        o = m(_batch(m, frames, pos))
        F, = torch.autograd.grad(o["e_int"].sum(), pos, create_graph=True)
        m.zero_grad()
        ((F ** 2).sum() + (o["e_int"] ** 2).sum()).backward()
        assert bool(torch.isfinite(F).all()), R
        for n, p in m.named_parameters():
            assert p.grad is None or bool(torch.isfinite(p.grad).all()), (R, n)
    assert m.ct_acc_raw.grad is not None and float(m.ct_acc_raw.grad.abs().max()) > 0     # the exponent is trained


def test_slater_cut_checkpoint_and_pair_weights(tables, tmp_path):
    from rsfff.train import cmm_data as D
    m = _slater_cut(tables)
    with torch.no_grad():
        m.ctk_roff.mul_(0.9)
        m.ctk_ron.mul_(0.9)
    m.sync_channels()
    save_cmm(tmp_path / "m.pt", m, params=PARAMS)
    m2 = load_cmm(tmp_path / "m.pt")
    assert m2.overlap_cut and not m2.frontier and torch.equal(m2.ctk_roff, m.ctk_roff) and m2.chan.types == m.chan.types
    fr = _dimers("Na", "Cl", [3.0, 6.0])
    with torch.no_grad():
        assert torch.allclose(m(_batch(m, fr))["e_int"], m2(_batch(m2, fr))["e_int"], rtol=0, atol=1e-13)
    m3 = _slater_cut(tables)
    on = float(m3.ctk_ron[m3.tab.symbols.index("Cl"), m3.tab.symbols.index("Na")])
    frames = D.as_frames([(["Na", "Cl"], np.array([[0, 0, 0], [0, 0, r]]), {}, {}) for r in (3.0, on + 0.2)])
    pw = D.set_pair_weights(m3, frames, 1.5)
    assert pw["pairs"] == 2 and pw["dropped"] == 1 and frames[0].labels["weight"] == 1.0 and frames[1].labels["weight"] == 0.0
    assert D.set_pair_weights(_model(tables, ct_kernel="slater"), frames, 1.5)["pairs"] == 0
