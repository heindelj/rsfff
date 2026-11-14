"""Species conditioning of the cmm encoder (FiLM on the free-ion descriptors).

Checks: the descriptors of an ion do not depend on which other ions the table carries; without conditioning a
heteronuclear dimer gives both atoms the same log-factors (the symmetric pair overlap is all the encoder sees, the
probes aside); a fresh FiLM encoder equals the unconditioned one with the same trunk weights (the FiLM generators start
at zero)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsfff.ff.cmm import ChannelTable, CmmConfig, CmmModel, IonTable, batched_precompute, cmm_table
from rsfff.ff.cmm.species import species_descriptors

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"
f64 = torch.float64
DIMER = ([[0.0, 0, 0], [2.6, 0, 0]], [11, 17])


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    return tab, chan, cmm_table(tab)


def env_u(setup, m, X, Z):
    tab, chan, table = setup
    bt = batched_precompute(torch.tensor(X, dtype=f64), torch.tensor(Z), torch.zeros(len(Z), dtype=torch.long),
                            tab, table, chan)
    with torch.no_grad():
        return dict(m.environment(bt))


def test_descriptors_are_per_ion(setup):
    """Each row depends only on that ion's own entries: the table in reverse species order gives the rows reversed
    (no standardisation over the loaded species)."""
    tab = setup[0]
    d = species_descriptors(tab)
    assert d.shape[0] == len(tab.symbols) and torch.isfinite(d).all()
    r = torch.arange(len(tab.symbols) - 1, -1, -1)
    rev = SimpleNamespace(**{k: getattr(tab, k)[r] for k in ("charge", "alpha", "C", "shell_n", "shell_b",
                                                             "I_donor", "A_acceptor", "a0")},
                          front_b={k: v[r] for k, v in tab.front_b.items()})
    assert torch.allclose(species_descriptors(rev), d[r])


def test_unconditioned_dimer_is_species_blind(setup):
    m = CmmModel(setup[0], setup[1], CmmConfig(film=False, probes=False))
    with torch.no_grad():
        m.encoder.readout.weight.normal_()
    u = env_u(setup, m, *DIMER)
    for k, v in u.items():
        assert torch.allclose(v[0], v[1], atol=1e-14), k
        assert float(v.abs().max()) > 0, k                  # the dimer's latent is live (S^w is an input)


def _has_bb():
    try:
        from bb.nn.conditioning import ConditionedTrunk
        ConditionedTrunk(2, 2, 1, d_c=2, mode="film")
        return True
    except Exception:  # noqa: BLE001 -- bb missing (or a stand-in)
        return False


@pytest.mark.skipif(not _has_bb(), reason="needs bb (FiLM trunk)")
def test_film_starts_as_unconditioned(setup):
    torch.manual_seed(2)
    tab, chan, _ = setup
    m0 = CmmModel(tab, chan, CmmConfig(film=False))
    m1 = CmmModel(tab, chan, CmmConfig(film=True))
    with torch.no_grad():
        m0.encoder.readout.weight.normal_()
        m1.encoder.readout.weight.copy_(m0.encoder.readout.weight)
        src = [l for l in m0.encoder.trunk.net if isinstance(l, torch.nn.Linear)]
        dst = [l.linear for l in m1.encoder.trunk.trunk.layers] + [m1.encoder.trunk.out]
        for a, b in zip(src, dst):
            b.weight.copy_(a.weight)
            b.bias.copy_(a.bias)
    X = [[0.0, 0, 0], [2.8, 0, 0], [1.4, 2.6, 0.3], [4.1, 2.7, 0.2]]
    u0, u1 = env_u(setup, m0, X, [11, 17, 17, 11]), env_u(setup, m1, X, [11, 17, 17, 11])
    for k in u0:
        assert torch.allclose(u0[k], u1[k], atol=1e-12), k
