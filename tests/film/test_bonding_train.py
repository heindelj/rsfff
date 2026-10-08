"""End-to-end smoke test of ``rsfff.train.train_bonding`` on synthetic monomer files.

Vacuum frames (energy, forces, dipole, second moments, polarizability) and probe frames
(``ext_charges``) for methanol and acetaldehyde, with explicit ``bonds``; the four model
variants (couplings x field features) each train two epochs and must stay finite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from test_bonding import acetaldehyde, methanol


def _frames(mol, n, rng, *, probes=False):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    pos0, z, bonds = mol
    out = []
    for _ in range(n):
        pos = pos0.numpy() + 0.03 * rng.standard_normal(pos0.shape)
        a = Atoms(numbers=z, positions=pos)
        a.arrays["fragment_idx"] = np.zeros(len(z), dtype=int)
        m2 = rng.standard_normal((3, 3))
        m2 = m2 + m2.T
        a.info.update({
            "bonds": np.array(bonds).reshape(-1),
            "fragment_charges": np.zeros(1), "fragment_multiplicities": np.ones(1),
            "fragment_dipoles": 0.5 * rng.standard_normal(3),
            "fragment_second_moments": m2[np.triu_indices(3)][[0, 1, 3, 2, 4, 5]],
            "polarizability": (20.0 * np.eye(3) + rng.standard_normal((3, 3))).reshape(-1),
        })
        if probes:
            p = rng.standard_normal((3, 3))
            p = 4.0 * p / np.linalg.norm(p, axis=1, keepdims=True)
            a.info["ext_charges"] = np.concatenate((p, rng.uniform(-0.5, 0.5, (3, 1))), 1).reshape(-1)
        e = -115.0 - 0.01 * rng.random()
        a.calc = SinglePointCalculator(
            a, energy=e, forces=0.01 * rng.standard_normal((len(z), 3)),
            dipole=0.5 * rng.standard_normal(3),
        )
        out.append(a)
    return out


@pytest.mark.parametrize("couplings,fields", [(False, False), (True, False), (False, True), (True, True)])
def test_train_bonding_smoke(tmp_path, couplings, fields):
    from ase.io import write
    from rsfff.train.train_bonding import train

    rng = np.random.default_rng(0)
    write(str(tmp_path / "vac.xyz"),
          _frames(methanol(), 6, rng) + _frames(acetaldehyde(), 6, rng), format="extxyz")
    write(str(tmp_path / "probe.xyz"),
          _frames(methanol(), 4, rng, probes=True) + _frames(acetaldehyde(), 4, rng, probes=True),
          format="extxyz")
    (tmp_path / "ref.json").write_text(json.dumps({"energies": {"H": -0.5, "C": -37.8, "O": -75.0}}))
    cfg = f"""
run_name: smoke
device: cpu
dtype: float64
checkpoint_root: {tmp_path}/ckpt
data:
  path: [{tmp_path}/vac.xyz]
  reference_energies: {tmp_path}/ref.json
  holdout_fraction: 0.25
  seed: 0
features: {{cutoff: 5.0, n_max: 3, l_max: 2, selected_lambdas: [0, 1, 2], backend: e3nn, density_channels: 4}}
film:
  nonbonded: exclusions
  exclude_through: 4
  impropers: true
  atom_typing: degree
  torsions: true
  couplings: {str(couplings).lower()}
  field_features: {str(fields).lower()}
  hidden: 24
  block_dim: 16
  head_hidden: 16
  head_depth: 1
  equiv_channels: 4
  bonded_hidden: 16
  term_hidden: 8
  induction: true
bonding:
  probe_path: [{tmp_path}/probe.xyz]
  probe_batch_size: 4
train: {{epochs: 2, batch_size: 4, learning_rate: 1.0e-3, eval_every: 1}}
"""
    (tmp_path / "cfg.yaml").write_text(cfg)
    train(tmp_path / "cfg.yaml")
    assert (tmp_path / "ckpt" / "smoke" / "best.pt").exists()
