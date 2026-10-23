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
    I = m.scaled(bt["feats"])
    Lps, A = m.pseudo_drive(I, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], sp.shape[0])
    # atom-centred strength: A_ij = gamma_i qP_j S_ij / s (times the cut-off)
    i, j = bt["ps_index"]
    S = m.pair_overlap(bt["ps_index"], bt["ps_r"], sp) / m.drive_s_ref
    expect = m.pseudo_gamma(I, sp)[i] * m.frozen_params(I, sp)[0][j] * S
    assert torch.allclose(A[bt["ps_r"] < 5.0], expect[bt["ps_r"] < 5.0])
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


def test_pauli_moments_reduce_to_monopole_at_zero(setup):
    tab, chan, feat, c, m = setup
    kw = dict(feature_names=feat.feature_names, feat_scale=m.feat_scale, pseudo=True, frozen=True)
    m0 = CompleteModel(tab, chan, c.feats.shape[1], **kw)
    m1 = CompleteModel(tab, chan, c.feats.shape[1], pauli_moments=True, **kw)
    m1.load_state_dict(m0.state_dict(), strict=False)
    bt = collate([c])
    assert torch.allclose(m0(bt)["e_pauli"], m1(bt)["e_pauli"])
    with torch.no_grad():
        m1.pm_a1.fill_(0.3)
        m1.pm_a2.fill_(0.3)
    assert not torch.allclose(m0(bt)["e_pauli"], m1(bt)["e_pauli"])


def test_pauli_shells_form(setup):
    tab, chan, feat, c, m = setup
    kw = dict(feature_names=feat.feature_names, feat_scale=m.feat_scale, pseudo=True, frozen=True)
    ms = CompleteModel(tab, chan, c.feats.shape[1], pauli_form="shells", **kw)
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([c, iso])
    o = ms(bt)
    assert o["e_pauli"][0] > 0 and o["e_pauli"][1] == 0
    w, b, _, _ = o["frozen_params"]
    assert w.shape == (5, 2) and (w[:, 1][bt["species"] == tab.symbols.index("Na")] == 0).all()
    # the per-shell drive: strengths (P, T), summed total reported; Na has one exclusion shell
    I = ms.scaled(bt["feats"])
    A = ms.pseudo_strength(I, bt["ps_index"], bt["ps_r"])
    assert A.shape[1] == 2 and (A[bt["species"][bt["ps_index"][1]] == tab.symbols.index("Na"), 1] == 0).all()
    assert torch.allclose(o["A_ps"], A.sum(1))
    # the shells reduce to the single form when only the outer shell carries weight at the single exponent
    m1 = CompleteModel(tab, chan, c.feats.shape[1], **kw)
    with torch.no_grad():
        ms.pauli_lnw.fill_(-30.0)
        for k in range(len(tab.symbols)):
            t = int(ms.pauli_outer[k])
            ms.pauli_lnw[k, t] = m1.pauli_lnq0[k]
            ms.pauli_lnbs[k] = m1.pauli_lnb[k] - torch.log(tab.shell_b[k, t])
        ms.ps_lng0.copy_(m1.ps_lng0)
    ms.head_frozen.load_state_dict(m1.head_frozen.state_dict())
    ms.head_ps.load_state_dict(m1.head_ps.state_dict())
    assert torch.allclose(ms(bt)["e_pauli"], m1(bt)["e_pauli"])
    assert torch.allclose(ms(bt)["e_pol"], m1(bt)["e_pol"])


def test_pauli_shells_forces(setup):
    tab, chan, feat, c, m = setup
    ms = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=m.feat_scale,
                       pseudo=True, frozen=True, pauli_form="shells")

    def E(p):
        cc = precompute(SYM, p, tab, chan, feat, lean=True, frontier=True, differentiable=True)
        o = ms(collate([cc]))
        return o["e_pol"][0] + o["e_pauli"][0]

    pos = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(E(pos), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((1, 2), (2, 0)):
            pp, pm = pos.detach().clone(), pos.detach().clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7


def test_ct_stark_gives_perpendicular_ct_polarizability(setup):
    tab, chan, feat, c, m = setup
    kw = dict(feature_names=feat.feature_names, feat_scale=m.feat_scale, pseudo=True, frozen=True)
    m0 = CompleteModel(tab, chan, c.feats.shape[1], **kw)
    m1 = CompleteModel(tab, chan, c.feats.shape[1], ct_stark=True, **kw)
    m1.load_state_dict(m0.state_dict(), strict=False)
    dimer = collate([precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.6]], tab, chan, feat, lean=True, frontier=True)])
    F = 0.002

    def curv(model, axis):
        e = []
        for sgn in (1.0, 0.0, -1.0):
            f = torch.zeros(1, 3, dtype=torch.float64)
            f[0, axis] = sgn * F
            e.append(model(dimer, field=f)["e_ct"][0])
        return (e[0] - 2 * e[1] + e[2]) / F ** 2

    assert abs(curv(m0, 0)) < 1e-9                 # perpendicular: nothing without the Stark term
    assert abs(curv(m1, 0)) > 1e-3                 # with it: a perpendicular CT polarizability
    assert curv(m1, 0) < 0                         # the field lowers the gap: more CT, lower energy
    with torch.no_grad():
        m0.gap_lamD.fill_(0.3)
    assert abs(curv(m0, 2)) > 1e-3                 # parallel: the gap factor already responds
    assert abs(curv(m0, 0)) < 1e-9                 # ... but not perpendicular


def test_induced_pauli_block_and_sq_confinement(setup):
    tab, chan, feat, c, m = setup
    kw = dict(feature_names=feat.feature_names, feat_scale=m.feat_scale, pseudo=True, frozen=True, pauli_form="shells")
    m0 = CompleteModel(tab, chan, c.feats.shape[1], **kw)
    m1 = CompleteModel(tab, chan, c.feats.shape[1], pauli_ii=True, conf_form="sq", **kw)
    m1.load_state_dict(m0.state_dict(), strict=False)
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([c, iso])
    # the cached block is symmetric and zero for the isolated atom
    App = bt["A_pp"]
    assert torch.allclose(App, App.transpose(1, 2)) and App[1].abs().max() == 0
    o0, o1 = m0(bt), m1(bt)
    assert torch.allclose(o1["c"][-1], torch.ones(2, dtype=torch.float64))        # isolated: no stiffening
    assert (o1["c"][:-1, 0] < 1).all()                                            # cluster: stiffened
    assert o1["e_pol"][1] == 0 and not torch.allclose(o0["e_pol"][0], o1["e_pol"][0])
    with torch.no_grad():                                                         # kappa = 0, h -> 0: back to m0
        m1.pp_kappa.zero_()
        m1.conf_h.fill_(-40.0)
    assert torch.allclose(m1(bt)["e_pol"], o0["e_pol"])


def test_induced_pauli_and_sq_forces(setup):
    tab, chan, feat, c, m = setup
    ms = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=m.feat_scale,
                       pseudo=True, frozen=True, pauli_form="shells", pauli_ii=True, conf_form="sq")
    with torch.no_grad():
        ms.pp_kappa.fill_(0.3)
        ms.conf_h.fill_(1.0)

    def E(p):
        cc = precompute(SYM, p, tab, chan, feat, lean=True, frontier=True, differentiable=True)
        return ms(collate([cc]))["e_pol"][0]

    pos = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(E(pos), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((1, 2), (3, 0)):
            pp, pm = pos.detach().clone(), pos.detach().clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7


def test_self_consistent_ct(setup):
    tab, chan, feat, c, m = setup
    kw = dict(feature_names=feat.feature_names, feat_scale=m.feat_scale, pseudo=True, frozen=True, pauli_form="shells")
    m0 = CompleteModel(tab, chan, c.feats.shape[1], **kw)
    m2 = CompleteModel(tab, chan, c.feats.shape[1], ct_sc=2, **kw)
    m2.load_state_dict(m0.state_dict(), strict=False)
    with torch.no_grad():
        for mm in (m0, m2):
            mm.gap_lamD.fill_(0.3)
            mm.gap_lamA.fill_(0.2)
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([c, iso])
    o0, o2 = m0(bt), m2(bt)
    assert torch.allclose(o0["e_pol"], o2["e_pol"])                      # ALMO bookkeeping: E_pol is CT-free
    assert torch.allclose(o0["e_ct"], o2["e_ct_0"])
    assert o2["e_ct"][1] == 0 and o2["e_ct"][0] != o0["e_ct"][0]
    assert (o2["p"] > 0).all() and o2["p"].max() < 0.1                   # populations: small, positive
    assert o2["joint_iters"] == 2


def test_self_consistent_ct_forces(setup):
    tab, chan, feat, c, m = setup
    ms = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=m.feat_scale,
                       pseudo=True, frozen=True, pauli_form="shells", ct_sc=2)
    with torch.no_grad():
        ms.gap_lamD.fill_(0.3)

    def E(p):
        cc = precompute(SYM, p, tab, chan, feat, lean=True, frontier=True, differentiable=True)
        o = ms(collate([cc]))
        return o["e_pol"][0] + o["e_ct"][0]

    pos = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(E(pos), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((1, 2), (3, 0)):
            pp, pm = pos.detach().clone(), pos.detach().clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-6      # the gradient is ~3 Eh/A here
