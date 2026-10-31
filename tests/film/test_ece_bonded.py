"""ECE-fed bonded and term heads (``film.bond_features: ece``, :mod:`rsfff.ff.film.ece`).

What has to hold:

* a fresh ECE model *is* the prior tables (every head deviation is zero-initialized), so
  switching the features on warm-starts from the latent-only model;
* once trained away from zero: rotation, reflection and atom-relabeling invariance (the
  per-bond merge is symmetric in the bond direction, every term input in the term's own
  reversal, and the pseudoscalars only enter squared), and conservative forces;
* the features see the bond-frame geometry a latent pooling cannot: rotating a methyl
  changes the central C-C parameters, and a mirror image does not;
* ``ece_latents: false`` makes the bonded/term parameters independent of the trunk;
* the isolated / env split is untouched: theta == theta_0 for an isolated molecule.
"""

from __future__ import annotations

import math

import pytest
import torch

from rsfff.ff.film.terms import term_energies
from rsfff.ff.units import BOHR_ANG

from test_bonding import acetaldehyde, bonding_model, make_batch, methanol

ECE = dict(bond_features="ece", ece_n_max=3, ece_l_max=3, ece_channels=4, ece_dim=8,
           ece_hidden=16)


def _rot(th, axis=2):
    c, s = math.cos(th), math.sin(th)
    r = torch.eye(3, dtype=torch.float64)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def test_fresh_ece_model_is_the_prior():
    m = bonding_model(**ECE).double()
    ref = bonding_model().double()
    b = make_batch(acetaldehyde())
    out, out_ref = m(b), ref(make_batch(acetaldehyde()))
    assert torch.equal(out.parameters.bonded0.delta_iso,
                       torch.zeros_like(out.parameters.bonded0.delta_iso))
    assert torch.allclose(out.energy_bonded, out_ref.energy_bonded, atol=1e-12)
    for e, _ in term_energies(b.positions, out.topology, out.parameters.bonded0,
                              out.parameters.terms0):
        assert torch.equal(e, torch.zeros_like(e))


def test_default_config_has_no_bond_features():
    m = bonding_model()
    assert not hasattr(m.network, "bond_features")
    assert not any("bond_features" in k for k in m.state_dict())
    assert m.network.bonded_head.bond_feat_dim == 0


@pytest.mark.parametrize("latents", [True, False])
def test_ece_invariance(latents):
    m = bonding_model(randomize=True, ece_latents=latents, **ECE).double()
    pos, z, bonds = acetaldehyde()
    pos = pos.double()
    e0 = float(m(make_batch((pos, z, bonds))).energy)
    e_rot = float(m(make_batch((pos @ _rot(0.7).T @ _rot(1.1, 0).T + 1.3, z, bonds))).energy)
    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    e_mir = float(m(make_batch((pos @ mirror.T, z, bonds))).energy)
    perm = [1, 0, 2, 3, 6, 4, 5]
    inv = {old: new for new, old in enumerate(perm)}
    e_perm = float(m(make_batch((pos[perm], [z[i] for i in perm],
                                 [(inv[a], inv[b]) for a, b in bonds]))).energy)
    for e in (e_rot, e_mir, e_perm):
        assert abs(e - e0) < 1e-10


def test_ece_conservative_forces():
    m = bonding_model(randomize=True, ece_latents=False, **ECE).double()
    b = make_batch(acetaldehyde())
    b.positions.requires_grad_(True)
    (g,) = torch.autograd.grad(m(b).energy.sum(), b.positions)
    h = 1e-5
    for atom, comp in ((0, 0), (2, 1), (5, 2)):
        es = []
        for sgn in (1, -1):
            bb = make_batch(acetaldehyde())
            with torch.no_grad():
                bb.positions[atom, comp] += sgn * h
            es.append(float(m(bb).energy.sum()))
        fd = (es[0] - es[1]) / (2 * h)
        assert abs(fd - float(g[atom, comp])) < 1e-6 * max(1.0, abs(fd))


def _methyl_rotated(th):
    """Acetaldehyde with the methyl H's rotated by ``th`` about the C-C axis."""
    pos, z, bonds = acetaldehyde()
    pos = pos.double().clone()
    axis = pos[1] - pos[0]
    axis = axis / axis.norm()
    k = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=torch.float64)
    rot = torch.eye(3, dtype=torch.float64) + math.sin(th) * k + (1 - math.cos(th)) * k @ k
    for h in (4, 5, 6):
        pos[h] = pos[0] + rot @ (pos[h] - pos[0])
    return pos, z, bonds


def test_ece_sees_the_torsion():
    m = bonding_model(randomize=True, ece_latents=False, **ECE).double()

    def cc_params(mol):
        out = m(make_batch(mol))
        t = out.topology
        col = [n for n, p in enumerate(t.bond_index.t().tolist()) if sorted(p) == [0, 1]][0]
        p = out.parameters.bonded0
        return torch.stack((p.r_eq[col], p.d[col], p.k[col]))

    p0 = cc_params(_methyl_rotated(0.0))
    p1 = cc_params(_methyl_rotated(0.5))
    assert (p1 - p0).abs().max() > 1e-6          # the C-C parameters depend on the dihedral


def test_no_latents_means_trunk_independent():
    m = bonding_model(randomize=True, ece_latents=False, **ECE).double()
    b = make_batch(methanol())
    p0 = m(b).parameters.bonded0
    with torch.no_grad():
        for prm in m.network.trunk.parameters():
            prm.add_(0.3)
    p1 = m(make_batch(methanol())).parameters.bonded0
    for name in ("r_eq", "d", "k", "cos_theta_eq", "k_theta"):
        assert torch.allclose(getattr(p0, name), getattr(p1, name), atol=1e-14)


def test_isolated_theta_equals_theta0():
    m = bonding_model(randomize=True, **ECE).double()
    p = m(make_batch(acetaldehyde())).parameters
    for name in ("r_eq", "d", "k", "cos_theta_eq", "k_theta"):
        assert torch.allclose(getattr(p.bonded, name), getattr(p.bonded0, name), atol=1e-14)


def test_builder_rejects_field_shift_without_latents():
    with pytest.raises(ValueError, match="ece_latents"):
        bonding_model(ece_latents=False, field_features=True, **ECE)


@pytest.mark.parametrize("latents", [True, False])
def test_train_bonding_smoke_with_ece(tmp_path, latents):
    import json

    import numpy as np
    from ase.io import write
    from rsfff.train.train_bonding import train
    from test_bonding_train import _frames

    rng = np.random.default_rng(0)
    write(str(tmp_path / "vac.xyz"),
          _frames(methanol(), 6, rng) + _frames(acetaldehyde(), 6, rng), format="extxyz")
    (tmp_path / "ref.json").write_text(json.dumps({"energies": {"H": -0.5, "C": -37.8, "O": -75.0}}))
    cfg = f"""
run_name: smoke_ece
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
  couplings: true
  bond_features: ece
  ece_latents: {str(latents).lower()}
  ece_n_max: 3
  ece_l_max: 3
  ece_channels: 4
  ece_dim: 8
  hidden: 24
  block_dim: 16
  head_hidden: 16
  head_depth: 1
  equiv_channels: 4
  bonded_hidden: 16
  term_hidden: 8
train: {{epochs: 2, batch_size: 4, learning_rate: 1.0e-3, eval_every: 1}}
"""
    (tmp_path / "cfg.yaml").write_text(cfg)
    train(tmp_path / "cfg.yaml")
    assert (tmp_path / "ckpt" / "smoke_ece" / "best.pt").exists()


def test_hessian_labels_and_hvp_loss(tmp_path):
    """Exact model Hessians written as labels (a.u., file order) come back as h_rel ~ 0."""
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    from rsfff.train.data import load_datasets
    from rsfff.train.train_bonding import BondingWeights, hessian_vector_loss

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = bonding_model(randomize=True, ece_latents=False, **ECE).double()
        frames = []
        for mol in (methanol(), acetaldehyde()):
            pos, z, bonds = mol
            pos = pos.double()

            def energy(x, z=z, bonds=bonds):
                return m(make_batch((x.reshape(-1, 3), z, bonds))).energy.sum()

            h = torch.autograd.functional.hessian(energy, pos.reshape(-1))   # Ha/Ang^2
            a = Atoms(numbers=z, positions=pos.numpy())
            a.arrays["fragment_idx"] = np.zeros(len(z), dtype=int)
            a.info.update({"bonds": np.array(bonds).reshape(-1),
                           "fragment_charges": np.zeros(1), "fragment_multiplicities": np.ones(1),
                           "hessian": (h * BOHR_ANG ** 2).reshape(-1).numpy()})
            a.calc = SinglePointCalculator(a, energy=0.0, forces=np.zeros((len(z), 3)))
            frames.append(a)
        write(str(tmp_path / "h.xyz"), frames, format="extxyz")
        ds = load_datasets([tmp_path / "h.xyz"], dtype=torch.float64)
        b = ds.flat_batch([0, 1])
        b.positions.requires_grad_(True)
        out = m(b)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=True)
        loss, rel = hessian_vector_loss(-g, b, BondingWeights(hessian_probes=3), training=False)
        assert rel < 1e-9
    finally:
        torch.set_default_dtype(prev)
