"""The cmm trainer's configuration: defaults, overrides, the worker's config."""
from __future__ import annotations

import json

import pytest

from rsfff.train.cmm import TrainConfig, load_config, parse_value
from rsfff.train.cmm_worker import make_config


def test_parse_value():
    assert parse_value("3") == 3 and parse_value("0.05") == 0.05 and parse_value("true") is True
    assert parse_value('["a", "b"]') == ["a", "b"]
    assert parse_value("[ww, Tws]") == ["ww", "Tws"]            # a shell-stripped list
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("ct/al.jsonl") == "ct/al.jsonl" and parse_value("") == ""


def test_defaults_are_the_converged_run():
    cfg, _ = load_config(None)
    m, l, t = cfg.model, cfg.loss, cfg.train
    assert (m.latent, m.pairs, m.probes, m.film, m.env_cap, m.ct_q_cap) == (6, ("ww",), True, True, 3.0, 1.5)
    assert (l.w_pol, l.w_ct, l.w_pauli, l.w_disp, l.w_alpha, l.w_force, l.w_smooth, l.w_curv, l.w_env_mid) == \
        (1.0, 1.0, 1.0, 1.0, 0.2, 0.1, 0.003, 2e-3, 0.1)
    assert (t.batch, t.force_batch, t.lr, t.force_every, t.alpha_every, t.env_sat) == (128, 64, 1e-3, 5, 4, 1.2)
    assert not hasattr(l, "w_chan")                               # the ALMO-CTA channel sums are not labels


def test_overrides_and_roundtrip(tmp_path):
    cfg, base = load_config(None, ["model.pairs=[Sw, Tws]", "train.epochs=3", "data.ct_dimers=all"])
    assert cfg.model.pairs == ("Sw", "Tws") and cfg.train.epochs == 3 and cfg.data.ct_dimers == "all"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg.to_dict()))
    again, base2 = load_config(p)
    assert again.to_dict() == cfg.to_dict() and base2 == tmp_path
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"loss": {"w_chan": 0.3}})


def test_worker_config(tmp_path):
    c = make_config(tmp_path, tmp_path / "run", {"latent": "8", "pairs": "ww/ws", "probes": "false",
                                                 "overrides": "loss.w_curv=0 data.salts=[lif,kbr]"})
    cfg = TrainConfig.from_dict(c)
    assert cfg.model.latent == 8 and cfg.model.pairs == ("ww", "ws") and not cfg.model.probes
    assert cfg.loss.w_curv == 0 and cfg.data.salts == ("lif", "kbr") and cfg.data.root == str(tmp_path.resolve())
