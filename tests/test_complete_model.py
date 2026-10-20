"""C2: the complete model -- isolated limits, isotropic equivalence, the dimer gap factor, forces."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.model import collate, precompute
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.complete import CompleteModel, DensityEnvironment, EnvironmentHead

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2.json"
SYM = ["Na", "Cl", "Na", "Cl"]
X = [[0, 0, 0], [0, 0, 2.6], [2.7, 0.2, 0.1], [2.6, 0.1, 2.7]]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = DensityEnvironment(tab)
    c = precompute(SYM, X, tab, chan, feat, lean=True, frontier=True)
    scale = c.feats.std(0).clamp(min=1e-6)
    m = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=scale, pseudo=True)
    return tab, chan, feat, c, m


def test_head_zero_at_empty_environment():
    h = EnvironmentHead(7, 2, n_species=2)
    for p in h.parameters():
        p.data.normal_()
    I = torch.zeros(3, 7)
    assert torch.allclose(h(I, torch.tensor([0, 1, 1])), torch.zeros(3, 2))


def test_isolated_atom_and_isotropic_equivalence(setup):
    tab, chan, feat, c, m = setup
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([c, iso])
    o = m(bt)
    assert o["e_pol"][1] == 0 and o["e_ct"][1] == 0
    assert torch.allclose(o["c"][-1], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(o["q_ct"][-1], torch.ones(2, dtype=torch.float64))
    # kappa = 0: the block solve equals the scalar solve with the same drive
    sp = bt["species"]
    alpha = torch.ones(2, bt["nmax"], dtype=torch.float64).masked_scatter(bt["mask"], tab.alpha[sp] * o["c"][:, 0])
    Cq = torch.ones(2, bt["nmax"], dtype=torch.float64).masked_scatter(bt["mask"], tab.C[sp] * o["c"][:, 1])
    h, _ = m.latents(bt["feats"], sp)
    Lps, _ = m.pseudo_drive(h, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], sp.shape[0])
    L = bt["L"] + torch.zeros_like(bt["L"]).masked_scatter(bt["mask"][..., None].expand_as(bt["L"]), Lps)
    assert torch.allclose(K.solve_induction(L, bt["A_off"], alpha, Cq), o["x"], atol=1e-12)


def test_gap_factor_is_three_body(setup):
    tab, chan, feat, c, m = setup
    dimer = collate([precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.6]], tab, chan, feat, lean=True, frontier=True)])
    assert m(dimer)["dphi"].abs().max() < 1e-12
    assert m(collate([c]))["dphi"].abs().max() > 1e-3


def test_anisotropy_keeps_the_mean(setup):
    tab, chan, feat, c, m = setup
    with torch.no_grad():
        m.kappa_aniso.fill_(0.5)
    o = m(collate([c]))
    ai = o["alpha_inv"]
    assert (torch.linalg.eigvalsh(ai) > 0).all()
    det = torch.linalg.det(ai) * (tab.alpha[c.species] * o["c"][:, 0]) ** 3
    assert torch.allclose(det, torch.ones_like(det))
    with torch.no_grad():
        m.kappa_aniso.zero_()


def test_forces_match_finite_differences(setup):
    tab, chan, feat, c, m = setup
    with torch.no_grad():
        m.kappa_aniso.fill_(0.3)
        m.gap_lamD.fill_(0.3)
        m.gap_lamA.fill_(-0.2)

    def E(p):
        cc = precompute(SYM, p, tab, chan, feat, lean=True, frontier=True, differentiable=True)
        o = m(collate([cc]))
        return o["e_pol"][0] + o["e_ct"][0]

    pos = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(E(pos), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((1, 2), (2, 0), (3, 1)):
            pp, pm = pos.detach().clone(), pos.detach().clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7
    with torch.no_grad():
        m.kappa_aniso.zero_()
        m.gap_lamD.zero_()
        m.gap_lamA.zero_()


def test_frozen_terms_signs_and_isolated_limit(setup):
    tab, chan, feat, c, m = setup
    mf = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=m.feat_scale,
                       pseudo=True, frozen=True)
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    o = mf(collate([c, iso]))
    assert o["e_pauli"][0] > 0 and o["e_disp"][0] < 0
    assert o["e_pauli"][1] == 0 and o["e_disp"][1] == 0
    qP, bP, C6, b6 = o["frozen_params"]
    assert torch.allclose(qP[-1], mf.pauli_lnq0[c.species.new_tensor(tab.symbols.index("Cl"))].exp())
    # the frozen heads only see overlaps
    names = [feat.feature_names[k] for k in mf.frozen_cols.tolist()]
    assert all(n.split(":")[-1] in ("S0", "|S1|", "|S2|") for n in names)
