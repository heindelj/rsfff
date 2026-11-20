"""Dipole moments as labels of the cmm trainer (``loss.w_dip``): the record convention (``cmm_data.dipole_label``), the
induced dipole a frame's label gives (``cmm_data.induced_dipole``), the model's -dE/dF (``cmm.model_dipole``) and the
loss on it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.train import cmm as T
from rsfff.train import cmm_data as D

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
SYM = ["Na", "Cl", "Li", "F", "Cl", "Na"]
X = np.array([[0, 0, 0], [2.7, 0.2, 0], [0.3, 2.4, 0.4], [2.9, 2.6, -0.2], [-2.6, 0.3, 0.5], [5.9, 0.4, 0.3]], float)


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, **kw):
    torch.manual_seed(11)
    cfg = dict(film=False, ct_model="gap", ct_gap_overlap="pauli", ct_gap_w0=0.005, ct_gap_converge=40,
               ct_gap_tol=1e-13, ct_kernel="slater", ct_cut="overlap", ct_q_headroom=0.2, env_switch="smooth")
    cfg.update(kw)
    m = CmmModel(*tables, CmmConfig(**cfg))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def _frames(dip=True):
    rows = [(SYM, X, {"pol": 0.0, "ct": 0.0}, {}),
            (["Na", "Cl"], np.array([[0.5, 0.2, -0.1], [0.5, 0.2, 2.6]]), {"pol": 0.0, "ct": 0.0}, {}),
            (["Li", "F", "Li"], np.array([[0, 0, 0], [1.9, 0.3, 0], [3.6, -0.4, 0.5]]), {"pol": 0.0, "ct": 0.0}, {})]
    fr = D.as_frames(rows)
    if dip:
        for k, f in enumerate(fr[:2]):                                   # the third frame has no dipole
            f.labels.update(D.dipole_label([0.3 * k + 1.0, -0.4, 2.0]))
    return fr


@pytest.mark.parametrize("kw", [dict(), dict(ct_exclusion=True), dict(ct_kernel="frontier", ct_cut="taper"),
                                dict(ct_cut="taper", ct_gap_converge=0)])
def test_model_dipole_is_the_field_derivative(tables, kw):
    m = _model(tables, **kw)
    _, bt = D.batch_of(_frames(False), m)
    with torch.no_grad():
        o0 = m(bt)
    o, mu = T.model_dipole(m, bt)
    assert mu.shape == (3, 3) and not mu.requires_grad
    assert torch.allclose(o["e_int"], o0["e_int"], rtol=0, atol=1e-13) and torch.allclose(o["e_ct"], o0["e_ct"], rtol=0, atol=1e-13)
    h = 1e-4
    with torch.no_grad():
        for a in range(3):
            F = torch.zeros(3, 3, dtype=f64)
            F[:, a] = h
            fd = -(m(bt, field=F)["e_int"] - m(bt, field=-F)["e_int"]) / (2 * h)
            assert torch.allclose(mu[:, a], fd, rtol=1e-6, atol=1e-8), (kw, a)
    # a pair's induced dipole lies on its axis and opposes the ion pair's own dipole (cation at the origin side)
    assert float(mu[1, :2].abs().max()) < 1e-10 and float(mu[1, 2]) > 0.05
    # its change with a small field is the model's polarizability
    oa = m(bt, alpha=True)
    F = torch.tensor([[2e-5, -1e-5, 1.5e-5]] * 3, dtype=f64, requires_grad=True)
    muF = -torch.autograd.grad(m(bt, field=F)["e_int"].sum(), F)[0]
    assert torch.allclose(muF - mu, torch.einsum("bca,ba->bc", oa["alpha"], F.detach()), rtol=1e-3, atol=1e-9)


@pytest.mark.parametrize("kw", [dict(), dict(ct_exclusion=True)])
def test_dipole_loss_gradient_is_exact(tables, kw):
    """d(loss)/d(parameter) of a loss on the dipoles (a second derivative of the energy) against finite differences,
    with channels inside, in and at the end of their switches."""
    m = _model(tables, **kw).train()
    off = float(m.ctk_roff[m.tab.symbols.index("Cl"), m.tab.symbols.index("Na")])
    fr = _frames() + D.as_frames([(["Na", "Cl"], np.array([[0, 0, 0], [0, 0, r]]), {"pol": 0.0, "ct": 0.0}, {})
                                  for r in (off - 0.6, off - 1e-4)])
    for f in fr[3:]:
        f.labels.update(D.dipole_label([0.0, 0.0, 1.0]))
    _, bt = D.batch_of(fr, m)

    def loss():
        _, mu = T.model_dipole(m, bt, create_graph=True)
        return (T.dipole_error(mu, bt) ** 2).sum()
    L = loss()
    m.zero_grad()
    L.backward()
    for n, p in m.named_parameters():
        assert p.grad is None or bool(torch.isfinite(p.grad).all()), n
    for p, idx in ((m.ct_ln_q0, (3, 0)), (m.ct_ln_q0, (1, 1)), (m.ct_acc_raw, (1,)), (m.ln_w1, (3, 0)),
                   (m.encoder.readout.weight, (0, 0))):
        g = float(p.grad[idx])
        d = 1e-5
        with torch.no_grad():
            p[idx] += d
            lp = float(loss())
            p[idx] -= 2 * d
            lm = float(loss())
            p[idx] += d
        fd = (lp - lm) / (2 * d)
        assert g == pytest.approx(fd, rel=2e-4, abs=1e-9), (kw, tuple(p.shape), idx)


def test_dipole_label_units_and_charge():
    """``unit="au"`` (the extxyz frames, e bohr) is taken as is; the net charge falls back to the formal charges."""
    assert np.allclose(D.dipole_label([0.0, 0.0, 1.0], unit="au")["dipole"], [0, 0, 1.0])
    assert np.allclose(D.dipole_label([0.0, 0.0, D.DEBYE])["dipole"], [0, 0, 1.0])
    with pytest.raises(ValueError):
        D.dipole_label([0.0, 0.0, 1.0], unit="atomic")
    assert D._net_charge(["Na", "Cl"]) == 0 and D._net_charge(["Li", "F", "Li"]) == 1 and D._net_charge(["Na"], 0) == 0
    assert D._net_charge(["O", "H", "H"]) is None


def test_dipole_label_convention(tables):
    tab = tables[0]
    assert D.dipole_label(None) == {} and D.dipole_label(None, [0, 0, 1]) == {}
    with pytest.raises(ValueError):
        D.dipole_label([1.0, 2.0])
    lab = D.dipole_label([D.DEBYE, 0.0, -2 * D.DEBYE])
    assert np.allclose(lab["dipole"], [1, 0, -2]) and np.allclose(lab["dipole_origin"], 0)
    # Na+ at z = 1, Cl- at z = 3.6 A: point charges alone give (1 - 3.6) / bohr about the origin
    f = D.as_frame(["Na", "Cl"], [[0, 0, 1.0], [0, 0, 3.6]])
    assert np.isnan(D.induced_dipole(f, tab)).all()
    pc = (1.0 - 3.6) / D.BOHR
    f.labels.update(D.dipole_label([0, 0, (pc + 0.7) * D.DEBYE]))
    assert np.allclose(D.induced_dipole(f, tab), [0, 0, 0.7])
    # a charged frame: the total dipole depends on the origin, the induced one does not
    g = D.as_frame(["Na", "Cl", "Na"], [[0, 0, 0], [0, 0.2, 2.6], [0.4, 0, 5.3]])
    Q = np.array([1.0, -1.0, 1.0])
    ind = np.array([0.1, -0.2, 0.3])
    out = []
    for o in ([0, 0, 0], [1.0, -2.0, 0.5]):
        mu = ind + (Q[:, None] * (g.pos.numpy() - np.array(o))).sum(0) / D.BOHR
        g.labels.update(D.dipole_label(mu * D.DEBYE, o))
        out.append(D.induced_dipole(g, tab))
    assert np.allclose(out[0], ind) and np.allclose(out[1], ind)
    # the report: an off-axis component of a pair shows a dipole given in another frame or about another origin
    f.labels.update(D.dipole_label([0.3, 0, (pc + 0.7) * D.DEBYE]))
    r = D.dipole_report([f, g, D.as_frame(["Na", "Cl"], [[0, 0, 0], [0, 0, 3]])], tab)
    assert r["n"] == 2 and r["pairs"] == 1 and r["off_axis"] == pytest.approx(0.3) and r["per_ion"] > 0.5


def test_dipoles_from_the_sets_to_the_metrics(tables, tmp_path):
    """Records with ``dipole`` (Debye) -> frames -> batch labels -> evaluate / summarize; a frame without one is
    left out of the dipole loss and metrics; the weights apply."""
    m = _model(tables)
    d = tmp_path / "ct_sets"
    d.mkdir()
    geoms = [(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.6]]), (["Li", "F", "Li"], [[0, 0, 0], [1.9, 0.3, 0], [3.6, -0.4, 0.5]]),
             (["Na", "Cl"], [[0, 0, 0], [0, 0, 3.4]])]
    recs = []
    for k, (sym, xyz) in enumerate(geoms):
        r = {"stage": "label", "symbols": sym, "coords": xyz, "tags": {"source": f"s{k}"},
             "eda": {"pol": -1.0, "ct": -2.0, "mod_pauli": 3.0, "disp": -0.5, "cls_elec": -400.0, "int": -400.5}}
        recs.append(r)
    # the model's own dipoles as the labels of the first two records (total = induced + point charges, in Debye)
    fr = D.as_frames([(s, np.array(x, float), {"pol": 0.0, "ct": 0.0}, {}) for s, x in geoms])
    _, bt = D.batch_of(fr, m)
    _, mu = T.model_dipole(m, bt)
    for k in (0, 1):
        Q = m.tab.charge[m.tab.species_index(fr[k].Z)].numpy()
        o = np.array([0.3, -0.1, 0.8])
        tot = mu[k].numpy() + (Q[:, None] * (fr[k].pos.numpy() - o)).sum(0) / D.BOHR
        recs[k]["dipole"] = (tot * D.DEBYE).tolist()
        recs[k]["dipole_origin"] = o.tolist()
    (d / "ct-mbe.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    # dipole_charged=False drops a charged frame's dipole (the Li2F+ record); by default it is kept
    kept = D.load_frames(str(d), species=m.tab.symbols, max_ho_ratio=float("inf"), dipole_charged=False)
    assert ["dipole" in f[2] for f in kept] == [True, False, False]
    assert [bool(f[3].get("dipole_dropped")) for f in kept] == [False, True, False]
    frames = D.as_frames(D.load_frames(str(d), species=m.tab.symbols, max_ho_ratio=float("inf")))
    assert ["dipole" in f.labels for f in frames] == [True, True, False]
    _, bt = D.batch_of(frames, m)
    assert bt["labels"]["dip"].shape == (3, 3) and bool(torch.isnan(bt["labels"]["dip"][2]).all())
    assert torch.allclose(bt["labels"]["dip"][:2], mu[:2], atol=1e-10)
    _, mu2 = T.model_dipole(m, bt)
    assert float(T.dipole_error(mu2, bt).abs().max()) < 1e-9                  # its own labels: no error, none for frame 3
    assert "dip" not in D.batch_of(frames[2:], m)[1]["labels"]
    rows = T.evaluate(m, frames, bs=2)
    assert [np.isfinite(r["dip_err"]) for r in rows] == [True, True, False] and max(r["dip_err"] for r in rows[:2]) < 1e-8
    assert rows[0]["dipz"] == pytest.approx(float(mu[0, 2]) * D.DEBYE) and rows[0]["dipz_ref"] == pytest.approx(rows[0]["dipz"])
    assert all(set(r) == set(rows[0]) for r in rows)
    # shifted labels: the error, per frame in the metrics and per ion in the loss; weights
    frames[0].labels["dipole"] = frames[0].labels["dipole"] + np.array([0, 0, 0.5 / D.DEBYE])
    frames[1].labels["weight"] = 0.0
    rows = T.evaluate(m, frames)
    sv = T.summarize(rows)["all"]
    assert rows[0]["dip_err"] == pytest.approx(0.5) and sv["dip_n"] == 2 and sv["dip_mae"] == pytest.approx(0.5)
    assert sv["dip_mae_raw"] == pytest.approx(0.25, abs=1e-6)
    _, bt = D.batch_of(frames, m)
    _, mu3 = T.model_dipole(m, bt)
    e = T.dipole_error(mu3, bt)
    assert e[0, 2] == pytest.approx(-0.25) and float(e[1:].abs().max()) < 1e-9
    lc = T.LossConfig(w_pol=0, w_ct=0, w_pauli=0, w_disp=0, w_frz=1, w_int=1, w_dip=1)
    assert T.val_score({"frz_mae": 1.0, "etot_mae": 2.0, "dip_mae": 0.3}, lc) == pytest.approx(3.0 + T.DIP_SCORE * 0.3)
    lc.w_dip = 0
    assert T.val_score({"frz_mae": 1.0, "etot_mae": 2.0, "dip_mae": 0.3}, lc) == pytest.approx(3.0)
    assert T.LossConfig().w_dip == 0 and T.RunConfig().dip_every == 1


def test_a_few_steps_on_the_dipole_loss_lower_it(tables):
    m = _model(tables).train()
    fr = _frames()
    for f in fr:
        f.labels.update({"pauli": float("nan"), "disp": float("nan")})
    _, bt = D.batch_of(fr, m)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    hist = []
    for _ in range(12):
        _, mu = T.model_dipole(m, bt, create_graph=True)
        loss = T.wmean((T.dipole_error(mu, bt) ** 2).sum(1), bt["labels"]["weight"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    assert all(np.isfinite(hist)) and all(y < x for x, y in zip(hist, hist[1:])) and hist[-1] < 0.95 * hist[0]


def test_dipoles_of_the_crossing_region_are_left_out(tables):
    m = _model(tables)
    fr = _frames()
    fr[2].labels.update(D.dipole_label([0.1, 0.2, 0.3]))
    for f, g in zip(fr, (0.3, 0.01, float("inf"))):
        f.meta["vacuum_gap"] = g
    assert D.mask_dipoles(fr, 0.0) == 0 and D.dipole_report(fr, m.tab)["n"] == 3
    assert D.mask_dipoles(fr, 0.04) == 1 and [f.labels["dipole_off"] for f in fr] == [False, True, False]
    assert D.dipole_report(fr, m.tab)["n"] == 2 and np.isnan(D.induced_dipole(fr[1], m.tab)).all()
    _, bt = D.batch_of(fr, m)
    y = bt["labels"]["dip"]
    assert bool(torch.isnan(y[1]).all()) and bool(torch.isfinite(y[[0, 2]]).all())
    _, mu = T.model_dipole(m, bt)
    e = T.dipole_error(mu, bt)
    assert float(e[1].abs().max()) == 0.0 and float(e[0].abs().max()) > 0 and float(e[2].abs().max()) > 0
    assert "dip" not in D.batch_of(fr[1:2], m)[1]["labels"]
    assert D.mask_dipoles(fr, 0.0) == 0 and D.dipole_report(fr, m.tab)["n"] == 3 and T.DataConfig().dip_min_gap == 0.04
