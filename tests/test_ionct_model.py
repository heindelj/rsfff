"""rsfff.ff.ionct.model: the cached, batched forward equals the direct kernels; isolated ions stay free."""
import json
import warnings
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.model import IonCTModel, collate, precompute

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"
HAS = PARAMS.exists() and "channels" in json.load(open(PARAMS))
pytestmark = pytest.mark.skipif(not HAS, reason="params with a 'channels' section not built")
DT = torch.float64


@pytest.fixture(scope="module")
def setup():
    warnings.filterwarnings("ignore")
    from bb.features import SOAPFeaturizer
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = SOAPFeaturizer(cutoff=6.0, n_max=6, l_max=3, neighbor_types=(11, 17), selected_lambdas=(0,)).double()
    return tab, chan, feat


FRAMES = [
    (["Na", "Cl"], [[0, 0, 0], [0, 0, 2.5]]),
    (["Cl", "Na", "Cl"], [[0, 0, 0], [2.7, 0.2, 0], [4.9, 1.8, 0.3]]),
    (["Na", "Cl", "Na", "Cl"], [[0, 0, 0], [2.6, 0, 0], [2.7, 2.6, 0.1], [0.1, 2.7, -0.2]]),
    (["Cl"], [[0, 0, 0]]),
]


def test_cached_forward_matches_the_direct_model(setup):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1])
    with torch.no_grad():
        for p in model.conf.parameters():            # a nontrivial confinement
            p.normal_(0, 0.05)
        out = model(collate(caches))
    off = 0
    for q, (sym, X) in enumerate(FRAMES):
        n = len(sym)
        Xt = torch.tensor(X, dtype=DT)
        sp = tab.species_index(torch.tensor([{"Na": 11, "Cl": 17}[a] for a in sym]))
        b = torch.zeros(n, dtype=torch.long)
        c = out["c"][off:off + n]
        ind = K.induction(Xt, sp, b, tab, alpha_scale=c[:, 0], c_scale=c[:, 1])
        assert out["e_pol"][q].item() == pytest.approx(ind.energy.item(), rel=1e-10, abs=1e-14)
        if n > 1:
            m = torch.as_tensor([int(f) == q for f in collate(caches)["ch_frame"]])
            res = CT.ct_energy(Xt, sp, b, tab, chan, x_ind=ind.x, ov_add=out["dov"][m], pair_overlap=False,
                               wkb=model.wkb)
            assert out["e_ct"][q].item() == pytest.approx(res.energy.item(), rel=1e-10)
        off += n


def test_isolated_ion_is_the_free_ion(setup):
    tab, chan, feat = setup
    caches = [precompute(["Cl"], [[0, 0, 0]], tab, chan, feat)]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1])
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
        out = model(collate(caches))
    assert torch.allclose(out["c"], torch.ones_like(out["c"]), atol=1e-14)
    assert out["e_ct"].item() == 0.0 and out["e_pol"].item() == 0.0


@pytest.mark.parametrize("with_field", [False, True])
def test_pseudo_and_field_forward_match_the_direct_kernels(setup, with_field):
    """The cached batch with the Pauli pseudo-potential drive and a uniform field equals the direct kernels."""
    from rsfff.ff.ionct.model import PSEUDO_CUTOFF, pseudo_types
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1], pseudo=True, wkb="smooth")
    torch.manual_seed(1)
    with torch.no_grad():
        for p in list(model.conf.parameters()) + list(model.ps.parameters()):
            p.normal_(0, 0.05)
        bt = collate(caches)
        Fb = torch.tensor([[0.0, 0.0, 0.003], [0.002, -0.001, 0.0], [0.0, 0.004, 0.0], [0.001, 0.0, 0.0]], dtype=DT)
        out = model(bt, field=Fb if with_field else None)
    assert (out["A_ps"] > 0).all()
    off = 0
    for q, (sym, X) in enumerate(FRAMES):
        n = len(sym)
        Xt = torch.tensor(X, dtype=DT)
        sp = tab.species_index(torch.tensor([{"Na": 11, "Cl": 17}[a] for a in sym]))
        b = torch.zeros(n, dtype=torch.long)
        F = Fb[q:q + 1] if with_field else None
        with torch.no_grad():
            h, h0 = model.latents(caches[q].feats, sp)
            idx, r, basis = K.pseudo_drive_basis(Xt, sp, b, tab, PSEUDO_CUTOFF)
            d = model.pseudo_drive(h, idx, r, pseudo_types(sp, idx, tab), basis, n)[0] if idx.shape[1] else None
            c = out["c"][off:off + n]
            ind = K.induction(Xt, sp, b, tab, field=F, alpha_scale=c[:, 0], c_scale=c[:, 1], drive_add=d)
        assert out["e_pol"][q].item() == pytest.approx(ind.energy.item(), rel=1e-10, abs=1e-14)
        if n > 1:
            m = torch.as_tensor([int(f) == q for f in bt["ch_frame"]])
            res = CT.ct_energy(Xt, sp, b, tab, chan, x_ind=ind.x, field=F, ov_add=out["dov"][m], pair_overlap=False,
                               wkb=model.wkb)
            assert out["e_ct"][q].item() == pytest.approx(res.energy.item(), rel=1e-10)
        off += n


def test_pseudo_drive_leaves_the_field_response_unchanged(setup):
    """The pseudo-potential is a constant drive: it lowers |E_pol| of a Na-Cl pair but not d2E/dF2."""
    tab, chan, _ = setup
    X = torch.tensor([[0, 0, 0], [0, 0, 2.8]], dtype=DT)
    sp = tab.species_index(torch.tensor([11, 17]))
    b = torch.zeros(2, dtype=torch.long)
    idx, r, basis = K.pseudo_drive_basis(X, sp, b, tab)
    d = torch.zeros(2, 8, dtype=DT).index_add(0, idx[0], 30.0 * basis)
    e = lambda Fz, dd: K.induction(X, sp, b, tab, field=torch.tensor([[0, 0, Fz]], dtype=DT), drive_add=dd).energy.item()
    assert abs(e(0, d)) < abs(e(0, None))
    h = 2e-3
    a0 = -(e(h, None) + e(-h, None) - 2 * e(0, None)) / h ** 2
    a1 = -(e(h, d) + e(-h, d) - 2 * e(0, d)) / h ** 2
    assert a1 == pytest.approx(a0, rel=1e-6)


def _joint_model(setup, seed=2):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1], pseudo=True, joint=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in list(model.conf.parameters()) + list(model.ps.parameters()):
            p.normal_(0, 0.05)
        model.gap[-1].bias.fill_(-1.0)           # small overlap gaps: strong CT, a visible joint effect
    return model, collate(caches)


def test_joint_solve_is_stationary_and_variational(setup):
    """x* makes 1/2 x^T A x + L.x + E_CT(x) stationary, and its E_CT lies at or below the post-solve one."""
    model, bt = _joint_model(setup)
    F = torch.tensor([[0.0, 0.0, 0.002], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DT)
    with torch.no_grad():
        out = model(bt, field=F)
    assert (out["e_ct"] <= out["e_ct_post"] + 1e-12).all()
    assert (out["e_ct"][:3] < out["e_ct_post"][:3]).all()
    # stationarity: rebuild the joint functional and differentiate at x*
    sp, mask, B, nmax = bt["species"], bt["mask"], bt["B"], bt["nmax"]
    with torch.no_grad():
        h, h0 = model.latents(bt["feats"], sp)
        c = model.confinement(h, h0, sp)
        alpha = torch.ones(B, nmax, dtype=DT).masked_scatter(mask, model.tab.alpha[sp] * c[:, 0])
        Cq = torch.ones(B, nmax, dtype=DT).masked_scatter(mask, model.tab.C[sp] * c[:, 1])
        n = sp.shape[0]
        Lflat = model.pseudo_drive(h, bt["ps_index"], bt["ps_r"], bt["ps_type"], bt["ps_basis"], n)[0]
        Fa = F[bt["frame"]]
        Lflat = Lflat - torch.cat([Fa, torch.zeros(n, 5, dtype=DT)], 1)
        L = bt["L"] + torch.zeros(B, nmax, 8, dtype=DT).masked_scatter(mask[..., None].expand(B, nmax, 8), Lflat)
        phi_ext = -(Fa * bt["pos"] / K.BOHR_ANG).sum(-1)
        A = K.induction_matrix(bt["A_off"], alpha, Cq)
    x = out["x"].clone().requires_grad_(True)
    E = 0.5 * (torch.bmm(A, x.reshape(B, -1, 1)).reshape(B, nmax, 8) * x).sum() + (L * x).sum() \
        + model._ct(x, bt, sp, mask, phi_ext, out["dov"], Fa)["e_ct"].sum()
    g = torch.autograd.grad(E, x)[0]
    assert g.abs().max().item() < 1e-8


def test_joint_energy_gradient_is_the_explicit_one(setup):
    """Envelope theorem: d E_CT / d theta through the converged joint solve equals a finite difference."""
    model, bt = _joint_model(setup, seed=3)
    model.joint_tol = 1e-13
    p = model.gap[-1].bias
    e = model(bt)["e_ct"].sum()
    g = torch.autograd.grad(e, p)[0].item()
    h = 1e-5
    with torch.no_grad():
        p += h
        ep = model(bt)["e_ct"].sum().item()
        p -= 2 * h
        em = model(bt)["e_ct"].sum().item()
        p += h
    assert g == pytest.approx((ep - em) / (2 * h), rel=1e-5)


def test_ct_state_polarizability_gives_perpendicular_alpha_ct(setup):
    """With the CT-state Stark shift, an isolated Na-Cl pair has alpha_CT > 0 perpendicular to the bond, close to
    p a* for the small fields. At zero field the Cl -> Na CT-state fields are only those of the neutral partners
    (penetration of the diffuse frontier densities), so the shift is small."""
    from rsfff.train.train_ionct import F_SET, polar_alpha, polar_batch
    tab, chan, feat = setup
    c = precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.8]], tab, chan, feat)
    out = {}
    for flag in (False, True):
        model = IonCTModel(tab, chan, c.feats.shape[1], ct_pol=flag)
        with torch.no_grad():
            out[flag] = polar_alpha(model, polar_batch([c]))[0]
    assert out[False][0, 1].abs() < 1e-6                       # x: perpendicular, no response without a*
    assert out[True][0, 1] > 0.1 and out[True][1, 1] > 0.1
    # z (along the bond) also changes, but either way: the zero-field CT-state field E0 (penetration of the
    # neutral partner) adds -a* E0 . F to the gap's linear response -- the transferred electron polarizing back
    # toward its hole shortens the CT dipole -- on top of the +a* curvature
    with torch.no_grad():
        bt = collate([c])
        o = model(bt)
        st, EA, ED = model.stark_shift(o["x"], bt, bt["species"], bt["mask"], torch.zeros(2, 3, dtype=DT))
    assert EA[:, :2].abs().max() < 1e-12 and ED[:, :2].abs().max() < 1e-12   # on-axis only
    clna = bt["species"][bt["ch_index"][0]] == tab.symbols.index("Cl")
    assert (st[clna].abs() < 0.05).all()          # Cl -> Na: both partners neutral in the CT state


def _density_model(setup, **kw):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1], gap_form="density", pseudo=True, ct_pol=True, **kw)
    return model, caches


def test_density_ct_is_translation_invariant_in_a_field(setup):
    """Gap (phibar differences), barrier (phibar_D - phi_midpoint) and Stark fields are all gauge invariant."""
    import numpy as np
    tab, chan, feat = setup
    model, _ = _density_model(setup)
    X0 = np.array([[0, 0, 0], [2.6, 0, 0], [2.7, 2.6, 0.1], [0.1, 2.7, -0.2]])
    cs = [precompute(["Na", "Cl", "Na", "Cl"], X0 + d, tab, chan, feat) for d in (0.0, np.array([3.0, -2.0, 5.0]))]
    F = torch.tensor([[0.002, -0.001, 0.003]] * 2, dtype=DT)
    with torch.no_grad():
        o = model(collate(cs), field=F)
    assert o["e_ct"][0].item() == pytest.approx(o["e_ct"][1].item(), rel=1e-10)
    assert o["e_ct"][0].item() < 0


def test_density_model_isolated_ion_and_tied_confinement(setup):
    model, caches = _density_model(setup, tie_c=True)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.conf.parameters():
            p.normal_(0, 0.1)
        out = model(collate(caches))
    assert torch.allclose(out["c"][:, 1], out["c"][:, 0] ** (5.0 / 3.0))
    assert out["e_ct"][3].item() == 0.0 and out["e_pol"][3].item() == 0.0      # the isolated Cl-
    assert torch.isfinite(out["e_ct"]).all()


def test_smoothness_penalty_vanishes_for_a_constant_network(setup):
    model, caches = _density_model(setup)
    bt = collate(caches)
    with torch.no_grad():
        for net in (model.conf, model.gap, model.ps):
            for p in net.parameters():
                p.zero_()
        assert model.smoothness_penalty(bt).item() == pytest.approx(0.0, abs=1e-20)
        for p in model.gap.parameters():
            p.normal_(0, 0.3)
        assert model.smoothness_penalty(bt).item() > 0


def test_frontier_coupling_is_gauge_invariant_and_blind_to_perpendicular_fields(setup):
    """Polarizable frontier densities: E_CT invariant under translation in a field; for a pair, a field perpendicular
    to the bond changes the gap (Stark) but not the coupling (only the along-bond shift enters the overlap)."""
    import numpy as np
    tab, chan, feat = setup
    _, caches = _density_model(setup)
    model = IonCTModel(tab, chan, caches[0].feats.shape[1], gap_form="density", pseudo=True, ct_pol=True,
                       tie_c=True, coupling_resp="frontier")
    X0 = np.array([[0, 0, 0], [2.6, 0, 0], [2.7, 2.6, 0.1], [0.1, 2.7, -0.2]])
    cs = [precompute(["Na", "Cl", "Na", "Cl"], X0 + d, tab, chan, feat) for d in (0.0, np.array([3.0, -2.0, 5.0]))]
    F = torch.tensor([[0.002, -0.001, 0.003]] * 2, dtype=DT)
    with torch.no_grad():
        o = model(collate(cs), field=F)
    assert o["e_ct"][0].item() == pytest.approx(o["e_ct"][1].item(), rel=1e-10)
    c = precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.8]], tab, chan, feat)
    with torch.no_grad():
        o = model(collate([c, c]), field=torch.tensor([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]], dtype=DT))
    W = o["W"].reshape(2, -1)
    assert torch.allclose(W[0], W[1], rtol=1e-12)
    assert not torch.allclose(o["stark"].reshape(2, -1)[0], o["stark"].reshape(2, -1)[1])


def test_direct_ct_is_short_range_attractive_and_pair_additive(setup):
    """CMM-style direct CT: every term -qD qA f(r)/r <= 0, f/r = 1/r - J_slater (penetration only) decays
    exponentially; a far-separated pair contributes nothing; no field dependence (alpha_CT = 0)."""
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    model = IonCTModel(tab, chan, caches[0].feats.shape[1], pseudo=True, gap_form="density", tie_c=True,
                       ct_form="direct")
    with torch.no_grad():
        out = model(collate(caches))
        assert (out["e_ch"] <= 0).all() and (out["e_ct"][:3] < 0).all() and out["e_ct"][3] == 0
        far = precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 7.0]], tab, chan, feat)
        near = precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.8]], tab, chan, feat)
        o = model(collate([far, near]))
        assert abs(o["e_ct"][0].item()) < 1e-4 * abs(o["e_ct"][1].item())
        F = torch.tensor([[0, 0, 0.003], [0.002, 0, 0]], dtype=DT)
        o2 = model(collate([far, near]), field=F)
        assert torch.allclose(o["e_ct"], o2["e_ct"])
