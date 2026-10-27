"""C-PCM inside the complete model: gas-phase limit, isolated ion, forces, and the QM bookkeeping mirrored."""
from pathlib import Path

import pytest
import torch

from rsfff.ff import cpcm
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import collate, precompute
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.complete import CompleteModel, DensityEnvironment, solvent_terms

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2.json"
SYM = ["Na", "Cl", "Na", "Cl"]
X = [[0, 0, 0], [0, 0, 2.6], [2.7, 0.2, 0.1], [2.6, 0.1, 2.7]]
SPEC = cpcm.CPCMSpec(eps=78.3553, ng=110)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = DensityEnvironment(tab)
    c = precompute(SYM, X, tab, chan, feat, lean=True, frontier=True)
    scale = c.feats.std(0).clamp(min=1e-6)
    m = CompleteModel(tab, chan, c.feats.shape[1], feature_names=feat.feature_names, feat_scale=scale, pseudo=False,
                      frozen=True, ct_stark=False)
    return tab, chan, feat, c, m


def test_gas_phase_limit(setup):
    tab, chan, feat, c, m = setup
    bt = collate([c])
    assert solvent_terms(bt, tab, cpcm.CPCMSpec(eps=1.0)) is None
    assert solvent_terms(bt, tab, SPEC, eps=[1.0]) is None
    o0, o1 = m(bt), m(bt, solvent=None)
    assert torch.allclose(o0["e_int"], o1["e_int"])


def test_continuum_screens_and_desolvates(setup):
    tab, chan, feat, c, m = setup
    bt = collate([c])
    st = solvent_terms(bt, tab, SPEC)
    o0, o = m(bt), m(bt, solvent=st)
    assert float(st.G_frz) < 0 and (st.G_own < 0).all()
    assert float(o["e_cav"]) > 0                                   # desolvation on joining the cavities
    assert float(o["e_scr"]) > 0 and float(o["e_elst"]) > float(o0["e_elst"])   # screened electrostatics
    assert abs(float(o["e_pol"])) < abs(float(o0["e_pol"]))        # screened induction (no pseudo drive here)
    assert o["phi_rf"][bt["species"] == tab.symbols.index("Cl")].min() > 0   # positive at the anions
    assert o["phi_rf"][bt["species"] == tab.symbols.index("Na")].max() < 0
    # the reaction potential raises the CT gap once the gap factor is on
    with torch.no_grad():
        m.gap_lamD.fill_(0.3)
        m.gap_lamA.fill_(0.3)
    o0, o = m(bt), m(bt, solvent=st)
    assert abs(float(o["e_ct"])) < abs(float(o0["e_ct"]))
    with torch.no_grad():
        m.gap_lamD.zero_()
        m.gap_lamA.zero_()
    assert torch.allclose(o["e_int"], o["e_elst"] + o["e_cav"] + o["e_pauli"] + o["e_disp"] + o["e_pol"] + o["e_ct"])


def test_isolated_ion_has_no_interaction_terms(setup):
    tab, chan, feat, c, m = setup
    iso = precompute(["Cl"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([iso])
    st = solvent_terms(bt, tab, SPEC)
    o = m(bt, solvent=st)
    assert abs(float(st.e_cav)) < 1e-12 and abs(float(st.e_scr)) < 1e-12
    assert abs(float(o["e_pol"])) < 1e-10 and float(o["e_ct"]) == 0
    # Born: G = -f/(2R) for a charge inside its sphere; Cl- has 0.33 e of shell density beyond its 1.2 x Bondi
    # sphere (as the QM anion does in vacuum), which weakens its self-solvation by ~11 %
    R = cpcm.sphere_radii(["Cl"], SPEC)[0]
    born = -0.5 * SPEC.f / float(R)
    assert born < float(st.G_own[0]) < 0.85 * born
    na = precompute(["Na"], [[0, 0, 0]], tab, chan, feat, lean=True, frontier=True)
    stn = solvent_terms(collate([na]), tab, SPEC)
    assert float(stn.G_own[0]) == pytest.approx(-0.5 * SPEC.f / float(cpcm.sphere_radii(["Na"], SPEC)[0]), rel=0.01)


def test_mixed_batch_per_frame_epsilon(setup):
    tab, chan, feat, c, m = setup
    iso = precompute(["Na", "Cl"], [[0, 0, 0], [0, 0, 2.8]], tab, chan, feat, lean=True, frontier=True)
    bt = collate([c, iso])
    st = solvent_terms(bt, tab, SPEC, eps=[1.0, 4.0])
    o = m(bt, solvent=st)
    o_gas = m(collate([c]))
    o_solv = m(collate([iso]), solvent=solvent_terms(collate([iso]), tab, cpcm.CPCMSpec(eps=4.0, ng=110)))
    assert torch.allclose(o["e_int"][0], o_gas["e_int"][0], atol=1e-10)
    assert torch.allclose(o["e_int"][1], o_solv["e_int"][0], atol=1e-10)


def test_forces_match_finite_differences(setup):
    tab, chan, feat, c, m = setup
    with torch.no_grad():
        m.gap_lamD.fill_(0.3)
        m.gap_lamA.fill_(-0.2)

    def E(p):
        cc = precompute(SYM, p, tab, chan, feat, lean=True, frontier=True, differentiable=True)
        bt = collate([cc])
        o = m(bt, solvent=solvent_terms(bt, tab, SPEC))
        return o["e_int"][0]

    pos = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(E(pos), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((1, 2), (2, 0), (3, 1)):
            pp, pm = pos.detach().clone(), pos.detach().clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-6
    with torch.no_grad():
        m.gap_lamD.zero_()
        m.gap_lamA.zero_()
