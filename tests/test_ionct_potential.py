"""The complete ion potential: dispersion and frozen Pauli modules, forces through the whole model."""
import json
import warnings
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.dispersion import IonDispersion
from rsfff.ff.ionct.frozen_pauli import IonPauli
from rsfff.ff.ionct.model import IonCTModel
from rsfff.ff.ionct.potential import IonPotential, soap_featurizer

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"
HAS = PARAMS.exists() and "channels" in json.load(open(PARAMS))
pytestmark = pytest.mark.skipif(not HAS, reason="params with a 'channels' section not built")
DT = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0, 0, 0], [2.6, 0.1, 0], [2.7, 2.6, 0.1], [0.1, 2.7, -0.2]], dtype=DT)


@pytest.fixture(scope="module")
def potential():
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = soap_featurizer()
    nf = feat(positions=X0, atomic_numbers=torch.tensor([11, 17, 11, 17]),
              batch_idx=torch.zeros(4, dtype=torch.long)).inv_feats.shape[1]
    m = IonCTModel(tab, chan, nf, pol_form="pauli", ct_form="direct", pauli_nn=True)
    d = IonDispersion.initial(("Na", "Cl"), n_feat=nf)
    p = IonPauli.initial(("Na", "Cl"), n_feat=nf)
    with torch.no_grad():                                 # nontrivial networks everywhere
        for net in (m.pauli_q, m.ctq, d.net, p.net):
            for w in net.parameters():
                w.normal_(0, 0.05)
    return IonPotential(m.eval(), d, p, feat)


def test_isolated_ions_have_no_pair_energy(potential):
    t = potential.terms(["Na"], torch.zeros(1, 3, dtype=DT))
    assert all(abs(v.item()) < 1e-14 for v in t.values())
    # the environment scales are exactly 1 for an isolated ion
    sp = torch.tensor([0, 1])
    z = torch.zeros(2, potential.disp.n_feat, dtype=DT)
    assert potential.disp.env_scale(sp, z).abs().max() == 0 and potential.pauli.env_scale(sp, z).abs().max() == 0


def test_dispersion_is_attractive_and_c6_at_long_range():
    d = IonDispersion.initial(("Na", "Cl"))
    sp = torch.tensor([1, 1])
    pos = torch.tensor([[0, 0, 0], [0, 0, 20.0]], dtype=DT)
    r = 20.0 / 0.529177210903
    e = d.pair_energy(pos, sp, torch.tensor([[0], [1]]))
    c6, c8 = d.ln_c6.exp()[1], d.ln_c8.exp()[1]
    assert e.item() < 0 and e.item() == pytest.approx(-(c6 / r ** 6 + c8 / r ** 8).item(), rel=1e-6)


def test_forces_are_the_energy_gradient(potential):
    E, F, t = potential.energy_forces(SYM, X0)
    assert t["int"].item() == pytest.approx(sum(t[k].item() for k in ("elec", "pauli", "disp", "pol", "ct")))
    h = 1e-4
    for a, k in ((1, 0), (1, 1), (2, 2), (3, 0)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, k] += h
        Xm[a, k] -= h
        with torch.no_grad():
            fd = -(potential.terms(SYM, Xp)["int"] - potential.terms(SYM, Xm)["int"]) / (2 * h) * 0.529177210903
        assert F[a, k].item() == pytest.approx(fd.item(), rel=1e-5, abs=1e-8)
