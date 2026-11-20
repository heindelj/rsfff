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


def test_derivative_residuals_vanish_on_the_models_own_derivatives():
    """Labels built from the model's own Jacobians and Hessian: every residual is zero, in the labels' layout and
    units (per bohr); a changed model has non-zero residuals."""
    import dataclasses

    from rsfff.ff.bonded import collate
    from rsfff.ff.units import BOHR_ANG
    from rsfff.train.bonded import derivative_residuals

    from test_bonded_electrical import _electrical_model

    m, b = _electrical_model()
    n = b.z.shape[0]
    pos = b.positions.clone().requires_grad_(True)
    out = m(b, pos)
    (g,) = torch.autograd.grad(out["energy"].sum(), pos, create_graph=True)
    dmu = torch.stack([torch.autograd.grad(out["dipole"][:, k].sum(), pos, retain_graph=True)[0] for k in range(3)], -1) * BOHR_ANG
    dal = torch.stack([torch.autograd.grad(out["polarizability"][:, j, k].sum(), pos, retain_graph=True)[0]
                       for j in range(3) for k in range(3)], -1) * BOHR_ANG
    hess = torch.stack([torch.autograd.grad(g.reshape(-1)[i], pos, retain_graph=True)[0].reshape(-1) for i in range(3 * n)]) * BOHR_ANG ** 2
    off = np.cumsum([0] + np.bincount(b.mol_idx.numpy()).tolist())
    hs = tuple(hess[3 * off[i]:3 * off[i + 1], 3 * off[i]:3 * off[i + 1]].detach() if i != 1 else None for i in range(b.n_mol))
    dmu = dmu.detach().clone()
    dmu[b.mol_idx == 3] = float("nan")                           # the last molecule has no dipole derivatives
    lab = dataclasses.replace(b, dipole_derivatives=dmu, polarizability_derivatives=dal.detach(), hessians=hs)
    assert lab.labelled("dipole_derivatives").tolist() == [True, True, True, False]
    assert lab.labelled("hessians").tolist() == [True, False, True, True]

    def residuals(model):
        p = lab.positions.clone().requires_grad_(True)
        o = model(lab, p)
        (gr,) = torch.autograd.grad(o["energy"].sum(), p, create_graph=True)
        return derivative_residuals(model, lab, o, p, gr, 3, torch.Generator().manual_seed(0), True)

    res = residuals(m)
    assert set(res) == {"dipole_derivatives", "polarizability_derivatives", "hessian"}
    assert res["dipole_derivatives"].shape[0] == int((b.mol_idx != 3).sum()) and res["hessian"].shape[0] == int((b.mol_idx != 1).sum())
    for r in res.values():
        assert float(r.abs().max()) < 1e-9
    sum(float((r ** 2).sum()) for r in res.values())
    (sum((r ** 2).sum() for r in res.values())).backward()        # the loss trains through the derivatives
    with torch.no_grad():
        for p in m.dec.values():
            p.weight.mul_(1.5)
    assert all(float(r.abs().max()) > 1e-6 for r in residuals(m).values())


def test_holdout_by_config_type(tmp_path):
    water = np.array([[0.0, 0.0, 0.117], [-0.763, 0.0, -0.468], [0.763, 0.0, -0.468]])
    write(tmp_path / "a.xyz", water, 0, n=10)
    text = (tmp_path / "a.xyz").read_text().splitlines()
    for i in range(0, len(text), 5):                              # frames 0-3 become scan frames
        text[i + 1] += " config_type=" + ("scan_stretch_0_1" if i // 5 < 4 else "wigner")
    (tmp_path / "a.xyz").write_text("\n".join(text) + "\n")
    cfg = tmp_path / "run.toml"
    cfg.write_text(f'''
[terms]
charges = "none"
dipoles = false
quadrupoles = false
polarizability = "none"
quadrupole_polarizability = false
[encoder]
latent = 2
hidden = 8
n_max = 3
l_max = 2
channels = 4
[data]
root = "{tmp_path}"
train = ["a.xyz"]
holdout = "scan"
val_fraction = 0.5
[train]
out = "{tmp_path / "out"}"
epochs = 1
''')
    report = main([str(cfg)])
    assert report["holdout"]["a"]["n"] == 4 and report["validation"]["a"]["n"] == 3
    rows = (tmp_path / "out" / "holdout.csv").read_text().splitlines()
    assert len(rows) == 5 and rows[1].split(",")[1] == "scan_stretch_0_1"
