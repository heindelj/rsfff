"""rsfff.ff.bonded.BondedConfig: the example file loads, round-trips, and bad term sets are refused."""
from pathlib import Path

import pytest

from rsfff.ff.bonded import BONDED_TERMS, BondedConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "bonded_h2o_ions.toml"


def test_example_is_the_default_model():
    cfg = BondedConfig.load(EXAMPLE)
    ref = BondedConfig(table_init="params/bonded_monomers_wb97mv_tzvpd.json")
    assert {k: v for k, v in cfg.__dict__.items() if k != "extra"} == {k: v for k, v in ref.__dict__.items() if k != "extra"}
    assert set(cfg.extra) == {"data", "loss", "train"} and cfg.extra["loss"]["forces"] == 1.0
    assert cfg.has("opposite_bond_angle") and not cfg.has("improper")
    assert cfg.charge_flow_response and cfg.responds


def test_round_trip():
    cfg = BondedConfig.load(EXAMPLE)
    again = BondedConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_dependencies_and_unknowns():
    assert set(BondedConfig().bonded) <= set(BONDED_TERMS)
    with pytest.raises(ValueError, match="built on"):
        BondedConfig(bonded=("morse", "bond_angle"))
    with pytest.raises(ValueError, match="unknown"):
        BondedConfig(bonded=("morse", "urey_bradley"))
    with pytest.raises(ValueError, match="unknown keys"):
        BondedConfig.from_dict({"encoder": {"latnet": 8}})
    with pytest.raises(ValueError):
        BondedConfig(charges="eem")
    with pytest.raises(KeyError):
        BondedConfig().has("stretch")
    flow = BondedConfig(charges="flow", polarizability="none", quadrupole_polarizability=False)
    assert not flow.charge_flow_response and not flow.responds
