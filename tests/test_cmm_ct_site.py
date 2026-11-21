"""The gap model's Pauli terms on the site potential: ``ct_gap_overlap = "site"`` shifts the acceptor level by the
transferred electron's energy in the neighbours' site potentials, kappa sum_k e^{u_j + u_k} <A_j | v_k>;
``ct_gap_donor`` lowers the gap by the same for the donor shape; ``ct_gap_kov_fixed`` holds kappa. The table rows are
the two-centre integrals; forces are exact derivatives."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.model import site_pauli_kernel

SITE = Path(__file__).resolve().parents[1] / "params" / "ions_salts8_site.json"
f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)
Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
FR = torch.zeros(4, dtype=torch.long)
BASE = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap", ct_gap_converge=40)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**BASE, **kw))


@pytest.fixture(scope="module")
def both():
    return _model(ct_gap_overlap="site", ct_gap_donor=True)


def test_frontier_rows_are_the_two_centre_integrals(both):
    """A pair: the 1s components of the frontier shapes against the closed-form kernel (Na+ donor, Cl- acceptor)."""
    m = both
    tab = m.tab
    R = 2.8
    bt = m.precompute(torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=f64), Z[:2], FR[:2])
    rb = torch.tensor(R / K.BOHR_ANG, dtype=f64)
    s = bt["species"]
    for key, role, rows in (("xs_D", "donor", bt["xs_D"]), ("xs_A", "acceptor", bt["xs_A"])):
        for side, (a, b) in enumerate(((int(s[0]), int(s[1])), (int(s[1]), int(s[0])))):      # a's shape in v_b
            if not (float(tab.front_w[role][a][0]) == 1.0 and tab.front_b[role][a].dim() == 0):
                continue
            for t in range(tab.shell_n.shape[1]):
                if float(tab.shell_n[b, t]) <= 0:
                    continue
                beta = tab.pauli_site_s[b] * tab.shell_b[b, t]
                ref = float(site_pauli_kernel(tab.front_b[role][a], beta, rb))
                assert float(rows[side, t]) == pytest.approx(ref, rel=1e-6, abs=1e-12)
    assert float(bt["xs_D"].min()) >= 0 and float(bt["xs_A"].min()) >= 0
    assert float(bt["xs_D"].max()) > 1e-4 and float(bt["xs_A"].max()) > 1e-4


def test_shift_and_relief_enter_the_gap(both):
    m = both
    bt = m.precompute(X0, Z, FR)
    sp = bt["species"]
    m.environment(bt)
    m.frozen_terms(bt, sp)
    m._q = m.ct_charges(sp)
    i, j = bt["ch_index"]
    shift, relief = m.acceptor_shift(bt, sp), m.donor_relief(bt, sp)
    assert float(shift.min()) > 0 and float(relief.min()) > 0
    xs = bt["x_sel"]
    pi, pj = bt["pairs"][:, xs]
    P = xs.shape[0]
    ci, cj = m._site_amplitudes(sp, pi, pj)
    ref = torch.zeros(4, dtype=f64).index_add(0, pi, (ci * bt["xs_A"][:P]).sum(1)).index_add(0, pj, (cj * bt["xs_A"][P:]).sum(1))
    assert torch.allclose(shift, m.overlap_strength()[sp] * ref, rtol=1e-12)
    assert torch.allclose(m.overlap_strength(), torch.ones_like(m.ct_ln_kov), rtol=1e-12)      # starts at 1
    phi = bt["phi_front"]
    d_ref = m.ct_gap_I[sp[i]] - m.ct_gap_A[sp[j]] + phi[i, 0] - phi[j, 1] - bt["ch_J"] + shift[j] - relief[i]
    assert torch.allclose(m.gap_channels(bt, sp)["Delta"], d_ref, rtol=1e-12)


def test_variants_and_forces():
    acc, fixed, don = _model(ct_gap_overlap="site"), _model(ct_gap_overlap="site", ct_gap_kov_fixed=True), \
        _model(ct_gap_overlap="site", ct_gap_donor=True)
    assert acc.ct_ln_kov.requires_grad and not fixed.ct_ln_kov.requires_grad
    assert not acc.pauli_ln_bscale.requires_grad
    e = [float(m(m.precompute(X0, Z, FR))["e_ct"].sum()) for m in (acc, fixed, don)]
    assert e[0] == pytest.approx(e[1], rel=1e-12)
    assert e[2] < e[0] < 0                                           # the relief lowers the gaps: more CT
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(don(don.precompute(X, Z, FR))["e_int"].sum(), X)[0]
    h = 1e-4
    for a, c in ((1, 0), (2, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        num = -(don(don.precompute(Xp, Z, FR))["e_int"].sum() - don(don.precompute(Xm, Z, FR))["e_int"].sum()) / (2 * h)
        assert float(F[a, c]) == pytest.approx(float(num), abs=2e-7)


def test_config_checked():
    with pytest.raises(ValueError, match="pauli_form"):
        CmmConfig(ct_model="gap", ct_gap_overlap="site")
    with pytest.raises(ValueError, match="pauli_form"):
        CmmConfig(ct_model="gap", ct_gap_donor=True)
