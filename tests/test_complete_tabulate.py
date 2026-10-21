"""The tabulated, batched precompute reproduces the per-frame path, and its forces are exact derivatives."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import collate, precompute
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute, permanent_electrostatics

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(3)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)            # coarse grid keeps the test fast; the training default is 0.01 A
    feat = DensityEnvironment(tab)
    frames = []
    for n in (2, 4, 6):
        sym = (["Na", "Cl"] * 3)[:n]
        while True:
            X = torch.rand(n, 3, dtype=torch.float64) * 5.0
            if (torch.cdist(X, X) + torch.eye(n) * 10).min() > 2.0:
                break
        frames.append((sym, X))
    return tab, chan, table, feat, frames


def _flat(frames):
    pos = torch.cat([X for _, X in frames])
    Z = torch.tensor([{"Na": 11, "Cl": 17}[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return pos, Z, fr


def test_batched_matches_per_frame(setup):
    tab, chan, table, feat, frames = setup
    ref = collate([precompute(sym, X, tab, chan, feat, lean=True, frontier=True) for sym, X in frames])
    bt = batched_precompute(*_flat(frames), tab, table, chan)
    for k in ("L", "A_off", "VD", "KD", "VA", "KA"):
        assert torch.allclose(ref[k], bt[k], atol=1e-7), k
    assert torch.allclose(ref["feats"], bt["feats"], atol=1e-6)
    assert torch.allclose(ref["env"]["dip:S2"], bt["env"]["dip:S2"], atol=1e-7)
    assert set(map(tuple, ref["ch_index"].T.tolist())) == set(map(tuple, bt["ch_index"].T.tolist()))
    assert set(map(tuple, ref["ps_index"].T.tolist())) == set(map(tuple, bt["ps_index"].T.tolist()))
    m = CompleteModel(tab, chan, ref["feats"].shape[1], feature_names=feat.feature_names,
                      feat_scale=ref["feats"].std(0).clamp(min=1e-6), pseudo=True, frozen=True)
    with torch.no_grad():
        for p in (m.kappa_aniso, m.gap_lamD, m.gap_lamA, m.breath_lam):
            p.fill_(0.3)
    o1, o2 = m(ref), m(bt)
    for k in ("e_pol", "e_ct", "e_pauli", "e_disp"):
        assert torch.allclose(o1[k], o2[k], atol=1e-7), k


def test_forces_are_exact_derivatives(setup):
    tab, chan, table, feat, frames = setup
    pos, Z, fr = _flat(frames)
    m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=True, frozen=True)
    with torch.no_grad():
        for p in (m.kappa_aniso, m.gap_lamD, m.gap_lamA, m.breath_lam):
            p.fill_(0.3)

    def E(p):
        return m(batched_precompute(p, Z, fr, tab, table, chan))["e_int"].sum()

    pg = pos.clone().requires_grad_(True)
    g = torch.autograd.grad(E(pg), pg)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((0, 2), (3, 1), (7, 0), (11, 2)):
            pp, pm = pos.clone(), pos.clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7


def test_fused_basis_matches_transcription():
    from rsfff.ff.complete.tabulate import _basis_tensors_fused, _basis_tensors_ref
    dr = (torch.randn(200, 3, dtype=torch.float64) * 4).requires_grad_(True)
    a, b = _basis_tensors_ref(dr), _basis_tensors_fused(dr)
    with torch.no_grad():
        assert torch.allclose(b, a, rtol=1e-12, atol=1e-14 * float(a.abs().max()))
    w = torch.randn_like(a)
    ga = torch.autograd.grad((a * w).sum(), dr, create_graph=True)[0]
    gb = torch.autograd.grad((b * w).sum(), dr, create_graph=True)[0]
    assert torch.allclose(gb, ga, rtol=1e-12, atol=1e-14 * float(ga.abs().max()))
    v = torch.randn_like(ga)        # the double backward force training takes
    ha, hb = (torch.autograd.grad((g * v).sum(), dr)[0] for g in (ga, gb))
    assert torch.allclose(hb, ha, rtol=1e-11, atol=1e-13 * float(ha.abs().max()))


def test_electrostatics_matches_pairwise(setup):
    from rsfff.ff.ionct import kernels as K
    tab, chan, table, feat, frames = setup
    pos, Z, fr = _flat(frames)
    s = tab.species_index(Z)
    pairs = K.all_pairs(fr)
    ref = torch.zeros(len(frames), dtype=torch.float64).index_add(0, fr[pairs[0]],
                                                                  K.permanent_pair_energy(pos, s, pairs, tab))
    assert torch.allclose(permanent_electrostatics(pos, s, fr, tab), ref, rtol=1e-12, atol=1e-12)


def test_electrostatics_pair_limit(setup):
    tab, chan, table, feat, frames = setup
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], dtype=torch.float64)
    s = tab.species_index(torch.tensor([11, 17]))
    e = permanent_electrostatics(X, s, torch.zeros(2, dtype=torch.long), tab)[0]
    assert abs(e + 1.0 / (20.0 / 0.529177210903)) < 1e-9          # two point charges at 20 A
