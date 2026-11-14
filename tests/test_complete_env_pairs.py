"""Channel selection of the shared environment (``CompleteModel(env_pairs=...)``): the family tokens keep their meaning
(ww = S^w + T^ww, ws = S^s + T^ws, ss = S^s + T^ss) and the exact groups Sw, Ss, Tww, Tws, Tss keep only their own
columns; the columns left out reach the trunk as zeros."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment
from rsfff.ff.complete.channels import CHANNEL_NAMES
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    return tab, chan, DensityEnvironment(tab)


def model(setup, pairs):
    tab, chan, feat = setup
    return CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                         pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="flux",
                         env_channels=True, env_pairs=pairs).double()


def kept(m):
    return {nm for nm, k in zip(CHANNEL_NAMES, m.chan_keep.tolist()) if k}


def fam(ab):
    return {nm for nm in CHANNEL_NAMES if nm[2:4] == ab and nm.startswith("T^")}


@pytest.mark.parametrize("pairs,want", [
    (("ww", "ws", "ss"), set(CHANNEL_NAMES)),
    (("ww",), {"S^w"} | fam("ww")),
    (("ws", "ss"), {"S^s"} | fam("ws") | fam("ss")),
    (("Sw", "Tww"), {"S^w"} | fam("ww")),
    (("Ss", "Tws"), {"S^s"} | fam("ws")),
    (("Sw", "Tws"), {"S^w"} | fam("ws")),
    (("Sw", "Tss"), {"S^w"} | fam("ss")),
    (("Ss", "Tss"), {"S^s"} | fam("ss")),
    (("Sw",), {"S^w"}),
])
def test_selection(setup, pairs, want):
    assert kept(model(setup, pairs)) == want


def test_dropped_columns_reach_the_trunk_as_zero(setup):
    m = model(setup, ("Ss", "Tws"))
    X = torch.rand(5, len(CHANNEL_NAMES), dtype=torch.float64) + 0.1
    I = m.scaled_channels(X)
    for k, nm in enumerate(CHANNEL_NAMES):
        assert (I[:, k] == 0).all() == (nm not in kept(m)), nm


def test_unknown_token(setup):
    with pytest.raises(ValueError):
        model(setup, ("Sx",))
