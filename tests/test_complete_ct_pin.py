"""The direct-CT exponent pins (``CompleteModel(ct_b_pin=...)``): every pinned mode keeps the cation acceptors at their
Fukui acceptor exponents and the anion acceptors at their outer electrostatic shell; ``donor-shell`` starts each anion
donor at its outer shell (1.01 x b_shell) and can only tighten it (factor 1 + softplus >= 1), with the gradient reaching
the factor; ``donor-scale`` starts at the Fukui donor ratio."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    return tab, chan, DensityEnvironment(tab)


def model(setup, pin):
    tab, chan, feat = setup
    return CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                         pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="flux",
                         env_channels=True, ct_b_pin=pin).double()


def outer_shell(tab):
    S = len(tab.symbols)
    outer = [max(t for t in range(tab.shell_n.shape[1]) if tab.shell_n[k, t] > 0) for k in range(S)]
    return torch.stack([tab.shell_b[k, outer[k]] for k in range(S)]).double()


@pytest.mark.parametrize("pin", ["all", "donor-scale", "donor-shell"])
def test_acceptors_pinned(setup, pin):
    tab = setup[0]
    _, bA = model(setup, pin).ct_exponents()
    an = tab.charge < 0
    want = torch.where(an, outer_shell(tab), tab.front_b["acceptor"].double())
    assert torch.allclose(bA, want)


def test_donor_shell_starts_at_the_shell_and_learns(setup):
    tab = setup[0]
    m = model(setup, "donor-shell")
    bD, _ = m.ct_exponents()
    bs, an = outer_shell(tab), tab.charge < 0
    assert torch.allclose(bD[an], 1.01 * bs[an])
    assert torch.allclose(bD[~an], bs[~an])                 # cation donors: their shell, nothing learned
    assert m.ct_fD_raw.requires_grad
    bD.sum().backward()
    g = m.ct_fD_raw.grad
    assert g is not None and (g[an] > 0).all() and (g[~an] == 0).all()
    with torch.no_grad():                                   # the factor never drops below 1: range <= the shell's
        m.ct_fD_raw.fill_(-50.0)
    assert (m.ct_exponents()[0][an] >= bs[an]).all()


def test_donor_scale_starts_at_fukui(setup):
    tab = setup[0]
    bD, _ = model(setup, "donor-scale").ct_exponents()
    bs, an = outer_shell(tab), tab.charge < 0
    want = bs * (1 + (tab.front_b["donor"].double() / bs - 1).clamp(min=0.05))
    assert torch.allclose(bD[an], want[an])


def test_acceptor_float(setup):
    """acceptor-float: donors and anion acceptors as "all"; cation acceptors start at Fukui, learn, stay within
    exp(+-CT_FA_CAP) of it."""
    tab = setup[0]
    m = model(setup, "acceptor-float")
    ref = model(setup, "all")
    bD, bA = m.ct_exponents()
    bD0, bA0 = ref.ct_exponents()
    an = tab.charge < 0
    assert torch.allclose(bD, bD0) and torch.allclose(bA, bA0)          # starts exactly at the "all" pin
    assert torch.allclose(bA[~an], tab.front_b["acceptor"].double()[~an])
    bA.sum().backward()
    g = m.ct_fA_raw.grad
    assert g is not None and (g[~an] != 0).all() and (g[an] == 0).all()
    c = m.CT_FA_CAP
    for v in (50.0, -50.0):
        with torch.no_grad():
            m.ct_fA_raw.fill_(v)
        r = m.ct_exponents()[1][~an] / tab.front_b["acceptor"].double()[~an]
        assert torch.allclose(r, torch.full_like(r, float(torch.tensor(c * (1 if v > 0 else -1)).exp())), rtol=1e-6)
        assert torch.allclose(m.ct_exponents()[1][an], bA0[an])         # anion acceptors untouched
