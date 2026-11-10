"""Species-conditioned shared environment (``CompleteModel(env_species=...)``, m35).

Checks: without conditioning a heteronuclear dimer gives both atoms the same log-factors (the symmetric pair overlap is
all the encoders see); with FiLM / concatenation on either embedding the two differ; every readout is exactly zero for
an isolated ion; the descriptors of an ion do not depend on which other ions the table carries; a fresh conditioned
model equals the unconditioned one with the same trunk weights (the FiLM generators start at zero)."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete.species import species_descriptors
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
f64 = torch.float64


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    return tab, chan, RadialTable(tab, h=0.05), DensityEnvironment(tab)


def model(setup, **kw):
    tab, chan, _, feat = setup
    return CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                         pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="flux",
                         env_channels=True, **kw)


def env_u(setup, m, X, Z):
    tab, chan, table, _ = setup
    bt = batched_precompute(torch.tensor(X, dtype=f64), torch.tensor(Z), torch.zeros(len(Z), dtype=torch.long),
                            tab, table, chan, inv_form=m.inv_form)
    m(bt)
    return m._env_u


def perturb(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mods = [m.env] + ([m.species_emb] if m.species_emb is not None else [])
        for mod in mods:
            for p in mod.parameters():
                p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))


DIMER = ([[0.0, 0, 0], [2.6, 0, 0]], [11, 17])


def test_unconditioned_dimer_is_species_blind(setup):
    m = model(setup)
    perturb(m)
    u = env_u(setup, m, *DIMER)
    for k, v in u.items():
        assert torch.allclose(v[0], v[1]), k


@pytest.mark.parametrize("mode,emb", [("film", "descriptor"), ("film", "table"), ("concatenate", "descriptor")])
def test_conditioned_dimer_and_isolated(setup, mode, emb):
    torch.manual_seed(1)
    m = model(setup, env_species=mode, env_species_emb=emb)
    perturb(m)
    u = env_u(setup, m, *DIMER)
    assert not torch.allclose(u["qD"][0], u["qD"][1])
    assert not torch.allclose(u["pauli"][0], u["pauli"][1])
    ui = env_u(setup, m, [[0.0, 0, 0]], [17])
    assert all(float(v.abs().max()) == 0.0 for v in ui.values())


def test_film_starts_as_unconditioned(setup):
    torch.manual_seed(2)
    m0 = model(setup)
    m1 = model(setup, env_species="film")
    perturb(m0)
    with torch.no_grad():                   # copy the unconditioned encoder weights into the FiLM layers
        for k in m0.env.trunks:
            src = [l for l in m0.env.trunks[k] if isinstance(l, torch.nn.Linear)]
            dst = [l.linear for l in m1.env.trunks[k].trunk.layers] + [m1.env.trunks[k].out]
            for a, b in zip(src, dst):
                b.weight.copy_(a.weight); b.bias.copy_(a.bias)
        for p in m0.env.g2:
            src = [l for l in m0.env.g2[p] if isinstance(l, torch.nn.Linear)]
            dst = [l.linear for l in m1.env.g2[p].trunk.layers] + [m1.env.g2[p].out]
            for a, b in zip(src, dst):
                b.weight.copy_(a.weight); b.bias.copy_(a.bias)
        for name in ("W", "lin3", "lin2", "line", "ge"):
            getattr(m1.env, name).load_state_dict(getattr(m0.env, name).state_dict())
    X = [[0.0, 0, 0], [2.8, 0, 0], [1.4, 2.6, 0.3], [4.1, 2.7, 0.2]]
    u0, u1 = env_u(setup, m0, X, [11, 17, 17, 11]), env_u(setup, m1, X, [11, 17, 17, 11])
    for k in u0:
        assert torch.allclose(u0[k], u1[k], atol=1e-12), k


def test_descriptors_are_per_ion(setup):
    """Each row depends only on that ion's own entries: the table in reverse species order gives the rows reversed
    (no standardisation over the loaded species)."""
    from types import SimpleNamespace
    tab = setup[0]
    d = species_descriptors(tab)
    assert d.shape[0] == len(tab.symbols) and torch.isfinite(d).all()
    r = torch.arange(len(tab.symbols) - 1, -1, -1)
    rev = SimpleNamespace(**{k: getattr(tab, k)[r] for k in ("charge", "alpha", "C", "shell_n", "shell_b",
                                                             "I_donor", "A_acceptor", "a0")},
                          front_b={k: v[r] for k, v in tab.front_b.items()})
    assert torch.allclose(species_descriptors(rev), d[r])


def test_extra_latent(setup):
    """--env-extra-latent: the gated frontier invariants feed the shared trunk; no side branches; every readout exactly
    zero for an isolated ion; on a dimer the latent (and so kappa, which has no two-body branch) is live."""
    torch.manual_seed(3)
    m = model(setup, env_species="film", env_ct_gated=True, env_extra_latent=True)
    assert m.env.extra_latent and not any(m.env.extra.values()) and len(m.env.ge) == 0
    perturb(m)
    ui = env_u(setup, m, [[0.0, 0, 0]], [17])
    assert all(float(v.abs().max()) == 0.0 for v in ui.values())
    u = env_u(setup, m, *DIMER)
    assert float(u["kappa_dip"].abs().max()) > 0
