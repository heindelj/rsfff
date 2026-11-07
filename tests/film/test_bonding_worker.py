"""rsfff.train.bonding_worker (the ChemLab train.bonding job) and rsfff.train.bonding_eval's stretch split."""

from __future__ import annotations

import json

import numpy as np
import yaml

from test_bonding import acetaldehyde, methanol


def _scan_frames(rng):
    """Two stretch scans of methanol (r / r_e from 0.9 to 1.5) and a few Wigner frames."""
    from test_bonding_train import _frames

    out = []
    for bond in ("0 1", "1 2"):
        for k, ratio in enumerate(np.linspace(0.9, 1.5, 7)):
            (a,) = _frames(methanol(), 1, rng)
            a.info.update({"name": "methanol", "kind": "scan_stretch", "scan_bond": bond,
                           "scan_value": 1.0 * ratio})
            a.calc.results["energy"] = -115.0 + (ratio - 1.0) ** 2      # minimum at ratio 1
            out.append(a)
    for a in _frames(methanol(), 4, rng):
        a.info.update({"name": "methanol", "kind": "wigner"})
        out.append(a)
    return out


def test_split_stretch(tmp_path):
    from ase.io import write
    from rsfff.train.bonding_eval import headers, split_stretch

    write(str(tmp_path / "v.xyz"), _scan_frames(np.random.default_rng(0)), format="extxyz")
    n_train, n_held = split_stretch(tmp_path / "v.xyz", tmp_path / "t.xyz", tmp_path / "h.xyz", 1.3)
    # per scan, ratios 0.9 .. 1.5 in steps of 0.1: 1.4 and 1.5 are beyond 1.3
    assert (n_train, n_held) == (2 * 5 + 4, 2 * 2)
    assert all(kind == "scan_stretch" for _, kind in headers(tmp_path / "h.xyz"))


def test_write_config_points_streams_at_the_inputs(tmp_path):
    from rsfff.train.bonding_worker import write_config

    cfg = {"data": {"path": ["x.xyz"], "reference_energies": "r.json"},
           "bonding": {"streams": [{"name": "probe", "path": "p.xyz", "weight": 2.0, "batch_size": 8},
                                   {"name": "field", "path": "nowhere.xyz"}]},
           "train": {"epochs": 50, "batch_size": 4}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    probe = tmp_path / "probe.xyz"
    out = write_config(tmp_path / "c.yaml", tmp_path / "o.yaml", train=tmp_path / "t.xyz",
                       references=tmp_path / "r.json", probe=probe, run_dir=tmp_path, epochs=3, seed=7,
                       log=lambda *a: None)
    assert out["data"]["path"] == [str(tmp_path / "t.xyz")]
    assert out["bonding"]["streams"] == [{"name": "probe", "path": str(probe), "weight": 2.0, "batch_size": 8}]
    assert out["train"]["epochs"] == 3 and out["train"]["seed"] == 7
    assert out["checkpoint_root"] == str(tmp_path / "ckpt") and out["run_name"] == "model"
    none = write_config(tmp_path / "c.yaml", tmp_path / "o2.yaml", train=tmp_path / "t.xyz",
                        references=tmp_path / "r.json", probe=None, run_dir=tmp_path, log=lambda *a: None)
    assert none["bonding"]["streams"] == [] and none["train"]["epochs"] == 50


def test_worker_job_end_to_end(tmp_path):
    """A workdir as the manager lays it out -> exit 0, the outputs by role, a report."""
    from ase.io import write
    from rsfff.train.bonding_worker import main
    from test_bonding_train import _frames

    rng = np.random.default_rng(0)
    wd = tmp_path / "wd"
    (wd / "in").mkdir(parents=True)
    write(str(wd / "in" / "vacuum"), _frames(methanol(), 6, rng) + _frames(acetaldehyde(), 6, rng),
          format="extxyz")
    write(str(wd / "in" / "probe"), _frames(methanol(), 4, rng, probes=True), format="extxyz")
    (wd / "in" / "references").write_text(json.dumps({"energies": {"H": -0.5, "C": -37.8, "O": -75.0}}))
    cfg = {"dtype": "float64", "data": {"holdout_fraction": 0.25, "seed": 0},
           "features": {"cutoff": 5.0, "n_max": 3, "l_max": 2, "selected_lambdas": [0, 1, 2],
                        "backend": "e3nn", "density_channels": 4},
           "film": {"nonbonded": "exclusions", "exclude_through": 4, "impropers": True,
                    "atom_typing": "degree", "torsions": True, "couplings": True,
                    "bond_features": "ece", "electrical_features": "ece", "ece_n_max": 3,
                    "ece_l_max": 3, "ece_channels": 4, "ece_dim": 8, "elec_hidden": 16, "elec_depth": 1,
                    "hidden": 24, "block_dim": 16, "head_hidden": 16, "head_depth": 1,
                    "equiv_channels": 4, "bonded_hidden": 16, "term_hidden": 8},
           "train": {"epochs": 5, "batch_size": 4, "learning_rate": 1.0e-3, "eval_every": 1}}
    (wd / "in" / "config").write_text(yaml.safe_dump(cfg))
    (wd / "in" / "settings").write_text(json.dumps(
        {"epochs": 1, "holdout_stretch": 0.0, "device": "cpu", "parity": True, "parity_frames": 4, "gpus": 0}))
    assert main(["run", str(wd)]) == 0
    for name in ("model.tar.gz", "eval.json", "summary.txt", "parity.png", "parity_data.npz", "train.log",
                 "config.yaml"):
        assert (wd / "out" / name).exists(), name
    report = json.loads((wd / "report.json").read_text())["data"]
    assert report["train.epochs"] == 1 and report["train.size"]["probe"] == 4
    assert np.isfinite(report["rsfff.bonding.val.f_mae"])
    assert "rsfff.bonding.probe_val.de_int" in report or "rsfff.bonding.probe_val.e_mae" in report
    ev = json.loads((wd / "out" / "eval.json").read_text())
    assert set(ev) == {"val", "probe_val"}


def test_worker_reports_failure(tmp_path):
    from rsfff.train.bonding_worker import main

    (tmp_path / "wd" / "in").mkdir(parents=True)
    assert main(["run", str(tmp_path / "wd")]) == 1
