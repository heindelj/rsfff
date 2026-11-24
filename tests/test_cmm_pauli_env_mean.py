"""``pauli_env = "mean"``: the site Pauli pair energy carries (e^{u_i} + e^{u_j}) / 2 instead of e^{u_i} e^{u_j}, the
same factor in the site xpol amplitudes; no term mixes the two centres' environments. ``ct_cation_donor = False``
drops the channels out of cations: the same as a cation donor amplitude of zero."""
from __future__ import annotations

import math

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True)
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _run(m, sym, X):
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m(m.precompute(X, Z, torch.zeros(len(sym), dtype=torch.long)))


def test_mean_factor_on_a_pair():
    m = _model(pauli_env="mean")
    sym, X = ["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.6, 0, 0]], dtype=f64)
    e0 = float(_run(m, sym, X)["e_pauli"])
    torch.manual_seed(1)
    with torch.no_grad():
        m.encoder.readout.weight.copy_(50.0 * torch.randn_like(m.encoder.readout.weight))
    e1 = float(_run(m, sym, X)["e_pauli"])
    u = m.cap(m._u["pauli"])
    assert float((u[0] - u[1]).abs()) > 1e-3
    assert e1 == pytest.approx(e0 * 0.5 * float(u.exp().sum()), rel=1e-10)


def test_mean_factor_has_no_cross_centre_term():
    """d^2 E_ij / du_i du_j = 0: shifting u_i and u_j together changes E by the sum of the separate changes."""
    m = _model(pauli_env="mean")
    bt = m.precompute(X0, torch.tensor([ATOMIC_NUMBER[a] for a in SYM]), torch.zeros(4, dtype=torch.long))
    m.environment(bt)
    sp = bt["species"]
    u0 = {k: v.clone() for k, v in m._u.items()}

    def E(di, dj):
        m._u = {k: v.clone() for k, v in u0.items()}
        m._u["pauli"] = m._u["pauli"] + torch.tensor([di, dj, 0, 0], dtype=f64)
        return float(m.frozen_terms(bt, sp)[0].sum())
    d = 0.3
    cross = E(d, d) - E(d, 0) - E(0, d) + E(0, 0)
    assert abs(cross) < 1e-14
    mp = _model(pauli_env="product")
    mp.load_state_dict(m.state_dict())
    mp.environment(bt)
    u0 = {k: v.clone() for k, v in mp._u.items()}
    m = mp
    assert abs(E(d, d) - E(d, 0) - E(0, d) + E(0, 0)) > 1e-6         # the product form has one


def test_mean_factor_forces_and_xpol_amplitudes():
    m = _model(pauli_env="mean")
    torch.manual_seed(3)
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
    X = X0.clone().requires_grad_(True)
    out = _run(m, SYM, X)
    F = -torch.autograd.grad(out["e_int"].sum(), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (1, 1), (3, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        num = -(_run(m, SYM, Xp)["e_int"].sum() - _run(m, SYM, Xm)["e_int"].sum()) / (2 * h)
        assert float(F[a, c]) == pytest.approx(float(num), abs=2e-7)
    bt = m.precompute(X0, torch.tensor([ATOMIC_NUMBER[a] for a in SYM]), torch.zeros(4, dtype=torch.long))
    m.environment(bt)
    sp = bt["species"]
    i, j = bt["pairs"][:, bt["x_sel"]]
    ci, cj = m._site_amplitudes(sp, i, j)
    e = m.cap(m._u["pauli"]).exp()
    a = m._site_bare(sp)
    assert torch.allclose(ci, 0.5 * (e[i] + e[j])[:, None] * a[j], rtol=1e-13)


def test_cation_donor_channels_dropped():
    on, off = _model(), _model(ct_cation_donor=False)
    assert all(k.split(">")[0] == "Cl" for k in off.chan.types) and len(off.chan.types) == 2
    bt_on = on.precompute(X0, torch.tensor([ATOMIC_NUMBER[a] for a in SYM]), torch.zeros(4, dtype=torch.long))
    bt_off = off.precompute(X0, torch.tensor([ATOMIC_NUMBER[a] for a in SYM]), torch.zeros(4, dtype=torch.long))
    sp = bt_off["species"]
    don = sp[bt_off["ch_index"][0]]
    assert bool((off.tab.charge[don] < 0).all())
    assert bt_off["ch_index"].shape[1] < bt_on["ch_index"].shape[1]
    with torch.no_grad():                   # the same as a cation donor amplitude of zero
        on.ct_ln_q0[on.tab.charge > 0, 0] = -60.0
    e_on, e_off = on(bt_on)["e_ct"], off(bt_off)["e_ct"]
    assert torch.allclose(e_off, e_on, rtol=1e-9, atol=1e-13)


def test_config_checked():
    with pytest.raises(ValueError, match="pauli_env"):
        CmmConfig(pauli_env="sum")
    with pytest.raises(ValueError, match="pauli_form"):
        CmmConfig(pauli_env="mean")
