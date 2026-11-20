"""rsfff.train.bonded: the loop runs end to end on a few synthetic frames and writes its outputs."""
import json
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.bonded import read_extxyz
from rsfff.train.bonded import load_config, main

ROOT = Path(__file__).resolve().parents[1]


def write(path, pos0, charge, n=12, seed=0, freq=None, electrical=False):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        pos = pos0 + 0.04 * rng.normal(size=pos0.shape)
        f = 0.01 * rng.normal(size=pos0.shape)
        extra = "" if freq is None else f' frequencies="{freq}"'
        if electrical:
            nine = lambda a: " ".join(f"{v:.6f}" for v in (a + a.T).reshape(-1))
            extra += (f' dipole="{" ".join(f"{v:.6f}" for v in rng.normal(size=3))}" quadrupole="{nine(rng.normal(size=(3, 3)))}"'
                      f' polarizability="{nine(np.eye(3) * 5 + 0.1 * rng.normal(size=(3, 3)))}"')
        lines += [str(len(pos)), f"Properties=species:S:1:pos:R:3:forces:R:3 energy={-76.0 + 0.001 * rng.normal():.8f} charge={charge}{extra}"]
        lines += [f"{s} " + " ".join(f"{v:.8f}" for v in (*p, *g)) for s, p, g in zip(["O"] + ["H"] * (len(pos) - 1), pos, f)]
    Path(path).write_text("\n".join(lines) + "\n")


def test_train_smoke(tmp_path):
    water = np.array([[0.0, 0.0, 0.117], [-0.763, 0.0, -0.468], [0.763, 0.0, -0.468]])
    oh = np.array([[0.0, 0.0, -0.107], [0.0, 0.0, 0.858]])
    write(tmp_path / "h2o.xyz", water, 0, electrical=True)
    write(tmp_path / "oh-.xyz", oh, -1, seed=1)
    write(tmp_path / "h2o_opt.xyz", water, 0, n=1, freq="1600 3800 3900")
    mols = read_extxyz(tmp_path / "oh-.xyz")
    assert mols[0].charge == -1 and mols[0].bonds.tolist() == [[0, 1]] and mols[0].forces.shape == (2, 3)
    w = read_extxyz(tmp_path / "h2o.xyz")[0]
    assert mols[0].dipole is None and w.dipole.shape == (3,) and w.polarizability.shape == (3, 3)
    assert np.allclose(w.second_moment, w.second_moment.T)
    cfg = tmp_path / "run.toml"
    cfg.write_text(f'''
[tables]
init = "{ROOT / "params" / "bonded_monomers_wb97mv_tzvpd.json"}"
[encoder]
latent = 3
hidden = 8
n_max = 3
l_max = 2
channels = 4
[data]
root = "{tmp_path}"
train = ["h2o.xyz", "oh-.xyz"]
reference = ["h2o_opt.xyz"]
val_fraction = 0.25
[train]
out = "{tmp_path / "out"}"
epochs = 2
batch = 8
''')
    report = main([str(cfg), "--set", "train.log_every=1"])
    assert set(report["validation"]) == {"h2o", "oh-"} and len(report["minima"]["h2o_opt"]["frequencies"]) == 3
    assert all(np.isfinite(v["f_rmse"]) for v in report["validation"].values())
    assert np.isfinite(report["validation"]["h2o"]["polarizability_rmse"]) and "dipole_rmse" not in report["validation"]["oh-"]
    saved = torch.load(tmp_path / "out" / "model.pt")
    assert saved["atom_types"] == [(1, 1), (8, 1), (8, 2)] and saved["config"]["encoder"]["latent"] == 3
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["constant"]["h2o"]["n"] == 3
    assert load_config(cfg, ["encoder.latent=5", "train.constant_only=true"])[0].latent == 5
