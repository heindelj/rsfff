"""The radial table's pairs: keyed by the ions' parameter values (not names or order), shipped with a model and used
only for exactly the ions they were computed for; the five-point derivatives against autograd."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ChannelTable, CmmConfig, CmmModel, IonTable, load_cmm, save_cmm
from rsfff.ff.cmm.tabulate import RadialTable

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl


def test_key_is_the_parameter_values():
    a = IonTable.load(PARAMS)
    b = IonTable.load(PARAMS, symbols=["Cl", "Na"])                     # another order and subset
    ta, tb = RadialTable(a, h=0.05), RadialTable(b, h=0.05)
    ia = {s: k for k, s in enumerate(a.symbols)}
    assert ta.pair_key(ia["Na"], ia["Cl"]) == tb.pair_key(1, 0)
    assert ta.pair_key(ia["Na"], ia["Cl"]) != tb.pair_key(0, 1)          # ordered pairs
    c = IonTable.load(PARAMS)
    c.shell_b[ia["Cl"], 0] *= 1.0 + 1e-12                                # any change of a value is a new pair
    assert RadialTable(c, h=0.05).pair_key(ia["Na"], ia["Cl"]) != ta.pair_key(ia["Na"], ia["Cl"])
    c = IonTable.load(PARAMS)
    c.alpha[ia["Cl"]] *= 2.0                                             # not an input of the integrals
    assert RadialTable(c, h=0.05).pair_key(ia["Na"], ia["Cl"]) == ta.pair_key(ia["Na"], ia["Cl"])


def test_derivatives_match_autograd():
    tab = IonTable.load(PARAMS)
    t = RadialTable(tab, h=0.05)
    si, sj = tab.symbols.index("Li"), tab.symbols.index("Cl")
    fd = t._functions(si, sj)
    r = t.r.clone().requires_grad_(True)
    vals = t._radial_values(si, sj, r)
    for k, f in vals.items():
        d1 = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), r, retain_graph=True)[0]
        for c, ref in ((0, f), (1, d1), (2, d2)):
            scale = float(ref.detach().abs().max()) + 1e-300
            tol = (1e-14, 1e-8, 1e-6)[c]
            assert float((fd[k][:, c] - ref.detach()).abs().max()) <= tol * scale, (k, c)


def test_shipped_pairs_are_used_and_only_for_their_ions(tmp_path, monkeypatch):
    monkeypatch.setenv("RSFFF_TABLE_CACHE", "off")
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    m = CmmModel(tab, chan, CmmConfig(film=False))
    na, cl = tab.symbols.index("Na"), tab.symbols.index("Cl")
    m.table().ensure_species([na, cl])
    save_cmm(tmp_path / "model.pt", m, params=PARAMS, table=True)

    def no_build(self, a, b):
        raise AssertionError(f"built pair {a, b}")

    m2 = load_cmm(tmp_path / "model.pt", params=PARAMS)
    monkeypatch.setattr(type(m2.table()), "_compute", no_build)
    m2.table().ensure_species([na, cl])
    assert torch.equal(m2.table().coef[[na, cl]][:, [na, cl]], m.table().coef[[na, cl]][:, [na, cl]])
    with pytest.raises(AssertionError, match="built pair"):
        m2.table().ensure_species([tab.symbols.index("Li")])           # not shipped: would be built
    # the same checkpoint on a species table with a changed Cl: its shipped pairs do not apply
    other = IonTable.load(PARAMS)
    other.shell_b[cl, 0] *= 1.001
    m3 = load_cmm(tmp_path / "model.pt", tab=other, chan=chan)
    with pytest.raises(AssertionError, match="built pair"):
        m3.table().ensure_species([cl])
