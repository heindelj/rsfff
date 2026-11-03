"""Gap-form CT (``CompleteModel(ct_model="gap")``): the dimer gap from its parts, gauge invariance and a nonzero
field response, the frontier blocks out of the heads, forces as derivatives of the energy (cluster and periodic)."""
import itertools
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)

    def make(**kw):
        torch.manual_seed(3)
        m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                          pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="gap",
                          ct_stark=True, ct_sat=True, ct_sat_ion=True, gap_shift="both", **kw)
        with torch.no_grad():
            for p in m.head_gap.parameters():
                p.add_(0.3 * torch.randn_like(p))
        return m
    return tab, chan, table, feat, make


def _cluster(a=2.6, n=2):
    g = torch.tensor([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.05 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return X, Z


def _batch(table, tab, chan, X, Z, m, **kw):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form, **kw)


def test_dimer_gap_from_its_parts(setup):
    """Na+ Cl-: the Cl -> Na channel's gap is I_D(Cl) - A_A(Na) + <donor_Cl|phi_Na> - <acceptor_Na|phi_Cl>
    + delta_A(Na) - delta_D(Cl) - Stark (no Stark without fields; the induced moments' potentials included)."""
    tab, chan, table, feat, make = setup
    m = make()
    X = torch.tensor([[0.0, 0, 0], [2.8, 0, 0]], dtype=torch.float64)
    Z = torch.tensor([11, 17])
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        o = m(bt)
        sp = bt["species"]
        I = m.scaled(bt["feats"])
        phiD, phiA = m.frontier_potentials(bt, o["x_pol"], torch.zeros(2, dtype=torch.float64))
        dl = m.gap_deltas(I, sp)
        i, j = bt["ch_index"]
        c = int(((i == 1) & (j == 0)).nonzero()[0])
        D = tab.I_donor[sp[1]] - m.acceptor_affinity()[sp[0]] + phiD[1] - phiA[0] + dl[0, 1] - dl[1, 0]
        ED = bt["env"]["donor:V1"][1]
        EA = bt["env"]["acceptor:V1"][0]
        aD = 100.0 * torch.nn.functional.softplus(m.stark_D[sp[1]])
        aA = 100.0 * torch.nn.functional.softplus(m.stark_A[sp[0]])
        D = D - 0.5 * aD * (ED * ED).sum() - 0.5 * aA * (EA * EA).sum()
    from rsfff.ff.complete.model import _gap_floor
    assert abs(float(_gap_floor(D) - o["Delta"][c])) < 1e-12
    assert float(dl.min()) >= 0.0


def test_gauge_invariance_and_field_response(setup):
    """A constant potential shift changes nothing; a uniform field changes E_CT at second order (alpha_CT != 0)."""
    tab, chan, table, feat, make = setup
    m = make()
    X, Z = _cluster()
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        o0 = m(bt)
        n = X.shape[0]
        phiD, phiA = m.frontier_potentials(bt, o0["x_pol"], torch.full((n,), 0.37, dtype=torch.float64))
        phiD0, phiA0 = m.frontier_potentials(bt, o0["x_pol"], torch.zeros(n, dtype=torch.float64))
        i, j = bt["ch_index"]
        assert ((phiD[i] - phiA[j]) - (phiD0[i] - phiA0[j])).abs().max() < 1e-14
        F = torch.tensor([[0.0, 0.0, 0.004]], dtype=torch.float64)
        ep, em = m(bt, field=F)["e_ct"], m(bt, field=-F)["e_ct"]
        assert abs(float(ep + em - 2 * o0["e_ct"])) > 1e-9


def test_frontier_blocks_are_not_head_inputs(setup):
    """Changing the donor / acceptor invariants changes no head output (only the gap operators use those shapes)."""
    tab, chan, table, feat, make = setup
    m = make()
    X, Z = _cluster()
    bt = _batch(table, tab, chan, X, Z, m)
    cols = [k for k, nm in enumerate(feat.feature_names) if nm.split(":")[0] in ("donor", "acceptor")]
    f2 = bt["feats"].clone()
    f2[:, cols] += 0.5 * bt["feats"][:, cols].abs().max()
    sp = bt["species"]
    with torch.no_grad():
        a, b = m.scaled(bt["feats"]), m.scaled(f2)
        assert (m.gap_deltas(a, sp) - m.gap_deltas(b, sp)).abs().max() == 0
        assert (m.confinement_from(a, sp) - m.confinement_from(b, sp)).abs().max() == 0
        assert (m.xpol_kappa(a, sp) - m.xpol_kappa(b, sp)).abs().max() == 0


@pytest.mark.parametrize("key", ["e_ct", "e_int"])
def test_forces_are_derivatives_of_the_energy(setup, key):
    tab, chan, table, feat, make = setup
    X0, Z = _cluster(a=2.5)
    m = make()

    def energy(X):
        return m(_batch(table, tab, chan, X, Z, m))[key].sum()
    X = X0.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(X), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (3, 1), (6, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(Xp) - energy(Xm)) / (2 * h)
        assert abs(float(g[a, c] - fd)) < 1e-7 + 1e-5 * abs(float(fd)), (key, a, c, float(g[a, c]), float(fd))


def test_periodic_huge_box_is_the_cluster(setup):
    tab, chan, table, feat, make = setup
    m = make()
    corners = list(itertools.product(range(2), repeat=3))
    X = torch.tensor(corners, dtype=torch.float64) * 2.7 + 25.0
    Z = torch.tensor([11 if sum(c) % 2 == 0 else 17 for c in corners])
    with torch.no_grad():
        ref = m(_batch(table, tab, chan, X, Z, m))
        out = m(_batch(table, tab, chan, X, Z, m, cell=torch.eye(3, dtype=torch.float64) * 50.0, cutoff=13.0,
                       ewald_tol=1e-10))
    assert abs(float(ref["e_ct"])) > 1e-4
    for k in ("e_ct", "e_pol", "e_int"):
        assert abs(float(ref[k] - out[k])) < 1e-7, (k, float(ref[k]), float(out[k]))


def test_acceptor_only_shift_is_the_default(setup):
    """gap_shift="acceptor" (default): delta_D = 0, so the overlap can only open the gap."""
    tab, chan, table, feat, make = setup
    m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                      pauli_form="shells", confine=("Cl",), xpol=True, ct_model="gap")
    X, Z = _cluster()
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        d = m.gap_deltas(m.scaled(bt["feats"]), bt["species"])
    assert float(d[:, 0].abs().max()) == 0.0 and float(d[:, 1].min()) >= 0.0 and float(d[:, 1].max()) > 0.0
