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
    assert cfg.data.params == "params/ions_salts8_cpksb_dipresp.json"     # dipole parents from the response density
    assert cfg.data.pair_weight == 1.5 and m.ct_kernel == "slater"          # the weights act with the frontier kernel only


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


def test_vacuum_gap_mask(tmp_path):
    """The geometric ct mask: a compact Na-Cl-Na keeps its label; two Cl- with the Na+ 6.6 A away (the ionic state is
    not the vacuum ground state) loses it, with no ALMO-CTA in the record."""
    from pathlib import Path
    from rsfff.ff.cmm import IonTable
    from rsfff.train import cmm_data as D
    tab = IonTable.load(Path(__file__).parent / "cmm_golden" / "ions_golden.json")
    ions = D.gap_ions(tab)
    near = (["Na", "Cl", "Na"], [[-2.6, 0, 0], [0, 0, 0], [2.6, 0, 0]])
    far = (["Cl", "Cl", "Na"], [[0, 0, 0], [3.8, 0, 0], [1.9, 6.4, 0]])
    assert D.vacuum_gap(*near, ions) > 0.15 and D.vacuum_gap(*far, ions) < 0.0
    assert D.vacuum_gap(["Cl", "Cl"], [[0, 0, 0], [3, 0, 0]], ions) == float("inf")
    d = tmp_path / "ct_sets"
    d.mkdir()
    rec = lambda k, f: json.dumps({"stage": "label", "symbols": f[0], "coords": f[1], "tags": {"source": f"s{k}"},
                                   "eda": {"pol": -1.0, "ct": -2.0}})
    (d / "ct-mbe.jsonl").write_text("\n".join(rec(k, f) for k, f in enumerate((near, far))) + "\n")
    kw = dict(species=tab.symbols, max_ho_ratio=float("inf"))
    assert [f[2]["ct_mask"] for f in D.load_frames(str(d), **kw)] == [True, True]
    out = D.load_frames(str(d), ct_min_gap=0.04, ions=ions, **kw)
    assert [f[2]["ct_mask"] for f in out] == [True, False] and out[1][3]["vacuum_gap"] < 0
    cfg, _ = load_config(None)
    assert cfg.data.ct_min_gap == 0.0 and cfg.model.ct_model == "flux"


def test_frozen_and_total_losses():
    """The EDA-free targets: E_Pauli + E_disp against the frozen energy minus electrostatics (the model's own or the
    EDA's), the total with the model's own electrostatics; the score follows the fitted terms."""
    from pathlib import Path
    import torch
    from rsfff.ff.cmm import ChannelTable, CmmConfig, CmmModel, IonTable
    from rsfff.train import cmm_data as D
    from rsfff.train.cmm import LossConfig, evaluate, frozen_error, summarize, term_error, val_score
    P = Path(__file__).parent / "cmm_golden" / "ions_golden.json"
    m = CmmModel(IonTable.load(P), ChannelTable.load(P), CmmConfig(film=False, disp_model="response")).eval()
    sym = ["Na", "Cl", "Na", "Cl"]
    X = [[0.0, 0, 0], [2.7, 0.1, 0], [2.8, 2.6, 0.2], [0.1, 2.7, -0.1]]
    with torch.no_grad():
        _, bt0 = D.batch_of([D.as_frame(sym, X)], m)
        o = m(bt0)
    d_el = 3e-4                                             # the EDA's cls_elec differs from the model's electrostatics
    lab = {"pol": float(o["e_pol"]), "ct": float(o["e_ct"]), "ct_mask": True, "pauli": float(o["e_pauli"]) + 1e-3 - d_el,
           "disp": float(o["e_disp"]), "elec": float(o["e_elst"]) + d_el,
           "int": float(o["e_int"]) + 2e-3}
    fr = D.as_frame(sym, X, lab, {"full": True, "n": 4})
    with torch.no_grad():
        _, bt = D.batch_of([fr], m)
        o = m(bt)
        KJ = D.KJ
        assert float(frozen_error(o, bt, "model")) == pytest.approx(-1e-3 * KJ / 4, rel=1e-9)     # elec + pauli + disp - e_elst
        assert float(frozen_error(o, bt, "eda")) == pytest.approx(-(1e-3 - d_el) * KJ / 4, rel=1e-9)
        assert float(term_error(o["e_int"], bt, "int")) == pytest.approx(-2e-3 * KJ / 4, rel=1e-9)
        with pytest.raises(ValueError):
            frozen_error(o, bt, "none")
    sv = summarize(evaluate(m, [fr], frz_elst="model"))["all"]
    assert sv["frz_mae"] == pytest.approx(1e-3 * KJ, rel=1e-6) and sv["etot_mae"] == pytest.approx(2e-3 * KJ, rel=1e-6)
    assert summarize(evaluate(m, [fr], frz_elst="eda"))["all"]["frz_mae"] == pytest.approx((1e-3 - d_el) * KJ, rel=1e-6)
    free = LossConfig(w_pol=0, w_ct=0, w_pauli=0, w_disp=0, w_frz=1.0, w_int=1.0)
    assert val_score(sv, free) == pytest.approx(sv["frz_mae"] + sv["etot_mae"])
    assert val_score(sv, LossConfig()) == pytest.approx(sv["pol_mae"] + sv["ct_mae"] + sv["pauli_mae"] + sv["disp_mae"])
    cfg, _ = load_config(None, ["loss.w_frz=1", "loss.w_int=1", "loss.frz_elst=eda", "model.disp_model=response"])
    assert cfg.loss.w_frz == 1 and cfg.loss.frz_elst == "eda" and cfg.model.disp_model == "response"
