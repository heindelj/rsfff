"""The tabulated CT coupling (``CmmConfig.ct_kernel = "table"``, rsfff.ff.cmm.ct_overlap): W^2 = 2 [W_ab(R) g(R)]^2 with
W_ab the one-electron coupling of the free ions' frontier orbitals in the frozen dimer (scripts/cmm_ct_coupling.py)
and g each pair's own C-infinity cutoff. No CT charges enter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable, load_cmm, save_cmm
from rsfff.ff.cmm.ct_overlap import coupling_kernel, load_ct_coupling, load_ct_overlap, pair_switch
from rsfff.ff.cmm.io import resolve_params

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
TABLE = "ct_coupling_wb97mv_tzvpd.json"
f64 = torch.float64
SYM = ["Li", "Na", "F", "Cl"]


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, **kw):
    torch.manual_seed(7)
    cfg = dict(film=False, ct_model="gap", ct_gap_overlap="pauli", ct_gap_w0=0.005, ct_gap_converge=40,
               ct_gap_tol=1e-12, ct_kernel="table", ct_cation_donor=False, env_switch="smooth")
    cfg.update(kw)
    m = CmmModel(*tables, CmmConfig(**cfg))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def _dimers(a, b, R):
    return [([a, b], torch.tensor([[0.0, 0, 0], [0, 0, float(r)]], dtype=f64)) for r in R]


def test_table_holds_the_anion_donors_and_the_spline_goes_through_it():
    d = json.load(open(resolve_params(TABLE)))
    cw = load_ct_coupling(resolve_params(TABLE), SYM)
    ov = load_ct_overlap(resolve_params("ct_overlap_wb97mv_tzvpd.json"), SYM)
    assert cw["has"][2:].all() and not cw["has"][:2].any()            # F, Cl donate; Li, Na do not
    r = d["r0"] + d["h"] * np.arange(d["n"])
    k = np.arange(5, 30, 4)
    si, sj = torch.full((len(k),), 3), torch.full((len(k),), 1)       # Cl > Na
    W2 = coupling_kernel(cw["coef"], cw["grid"], ov["r_on"], ov["r_off"], si, sj, torch.tensor(r[k], dtype=f64))
    g = pair_switch(ov["r_on"], ov["r_off"], si, sj, torch.tensor(r[k], dtype=f64))
    ref = 2 * (torch.tensor(np.asarray(d["W"]["Cl>Na"])[k], dtype=f64) * g) ** 2
    assert torch.allclose(W2, ref, rtol=1e-10, atol=1e-16)
    assert float(W2.max()) > 1e-5                                     # a coupling of the size of the trained ones


def test_coupling_is_the_table_and_no_charge_enters(tables):
    m = _model(tables)
    frames = _dimers("Na", "Cl", [2.6, 3.2, 4.4])
    bt = _batch(m, frames)
    out = m(bt)
    e0 = out["e_ct"].detach().clone()
    assert (e0 < 0).all()
    with torch.no_grad():
        m.ct_ln_q0.add_(1.0)                                          # the CT charges are not part of this kernel
    assert torch.allclose(m(_batch(m, frames))["e_ct"], e0, rtol=0, atol=1e-14)


def test_zero_beyond_the_cutoff_and_smooth_across_it(tables):
    m = _model(tables)
    off = float(m.ctk_roff[3, 1])
    R = torch.linspace(off - 0.6, off + 0.3, 46, dtype=f64)
    e = m(_batch(m, _dimers("Na", "Cl", R.tolist())))["e_ct"].detach()
    assert (e[R >= off] == 0).all() and (e[R < off - 0.05] < 0).all()
    d2 = torch.diff(e, n=2)
    assert float(d2.abs().max()) < 50 * float(d2.abs().mean()) + 1e-12


def test_forces_match_finite_differences(tables):
    m = _model(tables)
    frames = [(["Na", "Cl", "Li", "F"], torch.tensor([[0.0, 0, 0], [2.7, 0.2, 0], [0.3, 2.9, 0.4], [3.0, 3.1, 0.1]],
                                                    dtype=f64))]
    pos = frames[0][1].clone().requires_grad_(True)
    e = m(_batch(m, frames, pos=pos))["e_int"].sum()
    F = -torch.autograd.grad(e, pos)[0]
    h = 1e-4
    for (a, c) in ((0, 0), (1, 1), (3, 2)):
        p1, p2 = frames[0][1].clone(), frames[0][1].clone()
        p1[a, c] += h
        p2[a, c] -= h
        fd = -(m(_batch(m, frames, pos=p1))["e_int"].sum() - m(_batch(m, frames, pos=p2))["e_int"].sum()) / (2 * h)
        assert abs(float(F[a, c]) - float(fd.detach())) < 2e-6 * max(1.0, abs(float(fd.detach())))


def test_cation_donor_channels_need_their_own_table(tables):
    with pytest.raises(ValueError, match="no coupling for the channels"):
        _model(tables, ct_cation_donor=True)


def test_needs_the_gap_model():
    with pytest.raises(ValueError, match="needs ct_model = 'gap'"):
        CmmConfig(ct_kernel="table", ct_model="flux")


def test_checkpoint_carries_the_table(tables, tmp_path):
    m = _model(tables)
    save_cmm(tmp_path / "m.pt", m, params=str(PARAMS))
    m2 = load_cmm(tmp_path / "m.pt", params=str(PARAMS)).eval()
    assert m2.table_kernel and torch.equal(m2.ctw_coef, m.ctw_coef)
    frames = _dimers("Li", "F", [1.9, 2.6])
    assert torch.allclose(m2(_batch(m2, frames))["e_ct"], m(_batch(m, frames))["e_ct"], rtol=0, atol=1e-13)
