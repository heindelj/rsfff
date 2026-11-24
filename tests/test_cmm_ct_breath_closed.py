"""The breathing-dependent tabulated coupling (``CmmConfig.ct_breath``) and the closed-channel pair term
(``CmmConfig.ct_closed = "pair"``).

ct_breath: W = W_ab(R) + dW_donor(R) z_i + dW_acceptor(R) z_j, z the ions' CT-free breathing amplitudes (> 0 an
expansion) and dW the derivatives of the coupling along the same breathing functions, tabulated beside W_ab.
ct_closed: -f[s_j] <n_i | v_j> - f[s_i] <n_j | v_i> on the site Pauli halves, f a bounded fraction per species."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable, load_cmm, save_cmm
from rsfff.ff.cmm.ct_overlap import coupling_kernel, load_ct_coupling, load_ct_overlap, pair_switch
from rsfff.ff.cmm.io import resolve_params

SITE = Path(__file__).resolve().parents[1] / "params" / "ions_salts8_site.json"
TABLE = "ct_coupling_wb97mv_tzvpd.json"
f64 = torch.float64


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(SITE), ChannelTable.load(SITE)


def _model(tables, **kw):
    torch.manual_seed(7)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", ct_model="gap", ct_gap_overlap="site", ct_gap_w0=0.005,
               ct_gap_converge=40, ct_gap_tol=1e-12, ct_kernel="table", ct_cation_donor=False, env_switch="smooth")
    cfg.update(kw)
    m = CmmModel(*tables, CmmConfig(**cfg))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.2 * torch.randn_like(m.encoder.readout.weight))
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


FRAMES = [(["Na", "Cl"], torch.tensor([[0.0, 0, 0], [0, 0, 2.5]], dtype=f64)),
          (["Cl", "Na", "Cl"], torch.tensor([[0.0, 0, 0], [0.1, 0, 2.7], [0, 0.4, 5.3]], dtype=f64)),
          (["F", "K", "I"], torch.tensor([[0.0, 0, 0], [0.2, 0, 2.6], [0, 0.5, 6.0]], dtype=f64))]


def _forces_match(m, frames, key="e_int"):
    pos = torch.cat([X for _, X in frames]).clone().requires_grad_(True)
    e = m(_batch(m, frames, pos))[key].sum()
    g, = torch.autograd.grad(e, pos)
    h = 1e-4
    for a, c in ((0, 2), (3, 0), (6, 1)):
        d = torch.zeros_like(pos)
        d[a, c] = h
        with torch.no_grad():
            ep = m(_batch(m, frames, pos.detach() + d))[key].sum()
            em = m(_batch(m, frames, pos.detach() - d))[key].sum()
        assert abs(float((ep - em) / (2 * h)) - float(g[a, c])) < 1e-7 + 1e-5 * abs(float(g[a, c]))


def test_table_has_the_breathing_derivatives_on_the_coarse_grid():
    d = json.load(open(resolve_params(TABLE)))
    sym = d["symbols"]
    cw = load_ct_coupling(resolve_params(TABLE), sym)
    kb = d["breath_every"]
    assert cw["has_d"].equal(cw["has"]) and abs(float(cw["dgrid"][1]) - kb * d["h"]) < 1e-12
    ov = load_ct_overlap(resolve_params("ct_overlap_wb97mv_tzvpd.json"), sym)
    si, sj = torch.tensor([sym.index("Cl")]), torch.tensor([sym.index("Na")])
    k = 4 * kb                                                         # a knot of both grids
    r = torch.tensor([d["r0"] + d["h"] * k], dtype=f64)
    lam = torch.tensor([-0.08], dtype=f64)
    W2 = coupling_kernel(cw["coef"], cw["grid"], ov["r_on"], ov["r_off"], si, sj, r, cw["dcoef"], cw["dgrid"], lam, 0 * lam)
    w = d["W"]["Cl>Na"][k] - 0.08 * d["dW_donor"]["Cl>Na"][k // kb]
    assert abs(float(W2) - 2 * (w * float(pair_switch(ov["r_on"], ov["r_off"], si, sj, r))) ** 2) < 1e-14
    # a contraction of the anion (z < 0) raises its coupling to a cation at contact: W and dW of opposite sign
    for key in ("Cl>Na", "F>Li", "I>Cs", "Br>K"):
        kc = int(np.argmax(np.abs(d["W"][key][::kb][4:11])) + 4)
        assert d["W"][key][kc * kb] * d["dW_donor"][key][kc] < 0
    assert not np.any(d["dW_acceptor"]["Cl>Na"]) and np.any(d["dW_acceptor"]["Cl>K"])     # Na+ has no breathing, K+ has


def test_breathing_amplitudes_are_the_ct_free_ones(tables):
    m = _model(tables, ct_breath=True)
    o = m(_batch(m, FRAMES))
    z = o["ct_zb"]
    assert z.shape == (8,) and float(z[0]) == 0                                    # Na+ does not breathe
    assert float(z[1]) < 0 and float(z[2]) < 0                                     # anions against a cation: contracted
    assert torch.allclose(z, torch.cat([o["z_pol"][k, :n, 8] for k, n in enumerate((2, 3, 3))]))


def test_breathing_changes_the_ct_energy_and_the_forces_are_exact(tables):
    m0, m1 = _model(tables), _model(tables, ct_breath=True)
    m1.load_state_dict(m0.state_dict(), strict=False)
    e0, e1 = m0(_batch(m0, FRAMES)), m1(_batch(m1, FRAMES))
    assert float((e0["e_ct"] - e1["e_ct"]).abs().min()) > 1e-7
    assert float((e1["e_ct"] - e0["e_ct"])[0]) < 0                                  # contracted Cl-: more CT to Na+
    _forces_match(m1, FRAMES)


def test_breath_needs_the_table_kernel():
    with pytest.raises(ValueError, match="ct_breath"):
        CmmConfig(ct_model="gap", ct_kernel="slater", ct_breath=True)


def test_closed_channel_is_the_fraction_of_the_pauli_halves(tables):
    m = _model(tables, ct_closed="pair")
    m0 = _model(tables)
    m0.load_state_dict(m.state_dict(), strict=False)
    with torch.no_grad():
        m.ct_cc_raw.copy_(torch.linspace(-3, 1, 8, dtype=f64))
    f = m.closed_fraction()
    assert float(f.max()) < m.config.ct_closed_max and float(f.min()) > 0
    bt = _batch(m, FRAMES)
    o, o0 = m(bt), m0(_batch(m0, FRAMES))
    assert torch.allclose(o["e_ct"] - o["e_cc"], o0["e_ct"], atol=1e-12)            # nothing else moves
    assert torch.allclose(o["e_int"] - o["e_cc"], o0["e_int"], atol=1e-12)
    assert torch.allclose(o["e_pauli"], o0["e_pauli"], atol=1e-14)
    assert (o["e_cc"] < 0).all() and (o["e_cc"].abs() < m.config.ct_closed_max * o["e_pauli"]).all()
    # one species' fraction for every species: exactly -f E_Pauli
    with torch.no_grad():
        m.ct_cc_raw.fill_(0.3)
    o = m(_batch(m, FRAMES))
    assert torch.allclose(o["e_cc"], -m.closed_fraction()[0] * o["e_pauli"], rtol=1e-12)
    _forces_match(m, FRAMES)
    g, = torch.autograd.grad(m(_batch(m, FRAMES))["e_int"].sum(), m.ct_cc_raw)
    assert float(g.abs().max()) > 0


def test_closed_channel_needs_site_pauli():
    with pytest.raises(ValueError, match="ct_closed"):
        CmmConfig(ct_closed="pair")
    with pytest.raises(ValueError, match="ct_closed"):
        CmmConfig(pauli_form="site", ct_closed="contact")


def test_checkpoint_round_trip(tables, tmp_path):
    m = _model(tables, ct_breath=True, ct_closed="pair")
    save_cmm(tmp_path / "m.pt", m, params=str(SITE))
    m2 = load_cmm(tmp_path / "m.pt", params=str(SITE)).eval()
    a, b = m(_batch(m, FRAMES)), m2(_batch(m2, FRAMES))
    assert torch.allclose(a["e_int"], b["e_int"], atol=1e-12) and m2.ct_breath and m2.ct_closed
