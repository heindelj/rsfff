"""ECE electrical heads (``film.electrical_features: ece``, :mod:`rsfff.ff.film.electrical`).

* a fresh model: charges = diabatic charges + Pauling bond flows (summing exactly to the
  fragment charge), no atomic dipoles / quadrupoles, free-atom polarizabilities, compliance only
  on covalent bonds;
* once trained away from zero: charges and alpha eigenvalues invariant, dipoles / quadrupoles /
  alpha covariant under rotations and reflections, alpha positive definite;
* **dissociation on a fixed diabatic topology**: pull an atom out to 40 Ang along its bond and its
  charge goes to its diabatic charge, its dipole and quadrupole to zero, its polarizability to the
  free atom / ion value, and its bond's compliance to zero; pull a whole molecule apart and every
  atom does; and the remaining fragment's moments become those of that fragment alone;
* the diabatic Lewis guess (:mod:`rsfff.ff.film.diabats`).
"""

from __future__ import annotations

import math

import pytest
import torch

from rsfff.ff.film.diabats import guess_diabatic_charges
from rsfff.ff.film.electrical import free_atom_alpha
from rsfff.ff.multipole import spherical_to_cartesian_quadrupole

from test_bonding import acetaldehyde, bonding_model, make_batch, methanol

ELEC = dict(bond_features="ece", ece_n_max=3, ece_l_max=3, ece_channels=4, ece_dim=8,
            ece_hidden=16, electrical_features="ece", elec_hidden=16, elec_depth=1)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _params(m, mol):
    return m(make_batch(mol)).parameters


def _rot(th, axis):
    c, s = math.cos(th), math.sin(th)
    r = torch.eye(3)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def test_fresh_model_is_the_free_atom_plus_pauling_start():
    m = bonding_model(**ELEC).double()
    pos, z, bonds = acetaldehyde()
    p = _params(m, (pos.double(), z, bonds))
    assert abs(float(p.q_perm.sum())) < 1e-12
    assert float(p.q_perm[2]) < 0 and float(p.q_perm[3]) > 0          # O negative, H positive
    assert torch.equal(p.mu_perm, torch.zeros_like(p.mu_perm))
    assert torch.equal(p.quad_perm, torch.zeros_like(p.quad_perm))
    iso = torch.tensor([free_atom_alpha(zz, 0.0) for zz in z])
    assert torch.allclose(p.response0.alpha, iso[:, None, None] * torch.eye(3), atol=1e-12)


def test_charged_fragment_keeps_its_charge():
    m = bonding_model(randomize=True, **ELEC).double()
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
    b = make_batch((pos, [8, 1], [(0, 1)]))
    b.fragment_charge = torch.tensor([-1.0])
    p = m(b).parameters
    assert abs(float(p.q_perm.sum()) + 1.0) < 1e-12


@pytest.mark.parametrize("latents", [True, False])
def test_equivariance(latents):
    m = bonding_model(randomize=True, elec_latents=latents, **ELEC).double()
    pos, z, bonds = acetaldehyde()
    pos = pos.double()
    p0 = _params(m, (pos, z, bonds))
    for R in (_rot(0.7, 2) @ _rot(1.3, 0), torch.diag(torch.tensor([1.0, -1.0, 1.0]))):
        p1 = _params(m, (pos @ R.T + 0.4, z, bonds))
        assert torch.allclose(p1.q_perm, p0.q_perm, atol=1e-10)
        assert torch.allclose(p1.mu_perm, p0.mu_perm @ R.T, atol=1e-10)
        q0, q1 = (spherical_to_cartesian_quadrupole(p.quad_perm) for p in (p0, p1))
        assert torch.allclose(q1, R @ q0 @ R.T, atol=1e-10)
        a0, a1 = p0.response0.alpha, p1.response0.alpha
        assert torch.allclose(a1, R @ a0 @ R.T, atol=1e-10)
    assert p0.mu_perm.abs().max() > 1e-6 and p0.quad_perm.abs().max() > 1e-6
    assert (torch.linalg.eigvalsh(p0.response0.alpha) > 0).all()


def _pull(mol, atom, partner, dist):
    """Move ``atom`` (and nothing else) out to ``dist`` Ang from ``partner`` along their bond."""
    pos, z, bonds = mol
    pos = pos.double().clone()
    d = pos[atom] - pos[partner]
    pos[atom] = pos[partner] + dist * d / d.norm()
    return pos, z, bonds


def test_single_bond_dissociation_limits():
    m = bonding_model(randomize=True, **ELEC).double()
    pos, z, bonds = _pull(methanol(), 2, 1, 40.0)          # the hydroxyl H
    out = m(make_batch((pos, z, bonds)))
    p = out.parameters
    assert abs(float(p.q_perm[2])) < 1e-10                  # diabatic charge 0
    assert p.mu_perm[2].abs().max() < 1e-10
    assert p.quad_perm[2].abs().max() < 1e-10
    assert torch.allclose(p.response0.alpha[2], free_atom_alpha(1, 0.0) * torch.eye(3), atol=1e-10)
    # every electrical bond order of the O-H bond (flow ... compliance) has vanished
    t = out.topology
    heads = m.network.electrical_heads
    sp = m.projector.species_index(torch.tensor(z))
    env = heads.envelope(pos, m.network.bonded_head.type_index(sp, t), t.bond_index)
    col = [k for k, pr in enumerate(t.bond_index.t().tolist()) if sorted(pr) == [1, 2]][0]
    assert env[col].max() < 1e-10


def test_atomization_limits():
    """Every atom far from every other on the same topology: all bond terms vanish."""
    m = bonding_model(randomize=True, **ELEC).double()
    pos, z, bonds = acetaldehyde()
    far = 40.0 * torch.tensor([[k, (k * 7) % 5, (k * 3) % 4] for k in range(len(z))], dtype=torch.float64)
    p = _params(m, (far, z, bonds))
    assert p.q_perm.abs().max() < 1e-10
    assert p.mu_perm.abs().max() < 1e-10 and p.quad_perm.abs().max() < 1e-10
    iso = torch.tensor([free_atom_alpha(zz, 0.0) for zz in z])
    assert torch.allclose(p.response0.alpha, iso[:, None, None] * torch.eye(3), atol=1e-10)
    assert p.response0.compliance.abs().max() < 1e-10


def test_remaining_fragment_becomes_the_isolated_fragment():
    """Methanol with the hydroxyl H at 40 Ang vs the methoxy fragment on its own (element typing,
    no latents: the ECE heads then see only the covalent graph)."""
    m = bonding_model(randomize=True, elec_latents=False, atom_typing="element", **ELEC).double()
    pos, z, bonds = _pull(methanol(), 2, 1, 40.0)
    p_far = _params(m, (pos, z, bonds))
    keep = [0, 1, 3, 4, 5]
    remap = {a: k for k, a in enumerate(keep)}
    sub = [(remap[a], remap[b]) for a, b in bonds if a in remap and b in remap]
    p_frag = _params(m, (pos[keep], [z[k] for k in keep], sub))
    assert torch.allclose(p_far.q_perm[keep], p_frag.q_perm, atol=1e-10)
    assert torch.allclose(p_far.mu_perm[keep], p_frag.mu_perm, atol=1e-10)
    assert torch.allclose(p_far.response0.alpha[keep], p_frag.response0.alpha, atol=1e-10)


def test_conservative_forces_with_electrical_heads():
    m = bonding_model(randomize=True, **ELEC).double()
    b = make_batch(acetaldehyde())
    b.positions.requires_grad_(True)
    (g,) = torch.autograd.grad(m(b).energy.sum(), b.positions)
    h = 1e-5
    for atom, comp in ((0, 0), (2, 1)):
        es = []
        for sgn in (1, -1):
            bb = make_batch(acetaldehyde())
            with torch.no_grad():
                bb.positions[atom, comp] += sgn * h
            es.append(float(m(bb).energy.sum()))
        fd = (es[0] - es[1]) / (2 * h)
        assert abs(fd - float(g[atom, comp])) < 1e-6 * max(1.0, abs(fd))


def test_response_properties_run():
    m = bonding_model(randomize=True, **ELEC).double()
    b = make_batch(acetaldehyde())
    _, mu, th, al = m.response_properties(b, polarizability=True, quadrupole=True, create_graph=False)
    assert torch.isfinite(mu).all() and torch.isfinite(al).all()
    assert (torch.linalg.eigvalsh(0.5 * (al + al.transpose(-1, -2))) > 0).all()


@pytest.mark.parametrize("z,bonds,charge,want", [
    ([8, 1, 1], [(0, 1), (0, 2)], 0, [0, 0, 0]),
    ([6, 6, 8, 8, 1, 1, 1], [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)], -1,
     [0, 0, -0.5, -0.5, 0, 0, 0]),
    ([7, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)], 1, [1, 0, 0, 0, 0]),
    ([8, 1, 1, 1], [(0, 1), (0, 2), (0, 3)], 1, [1, 0, 0, 0]),
    ([8, 1], [(0, 1)], -1, [-1, 0]),
    ([8, 1], [(0, 1)], 0, [0, 0]),
    ([15, 8, 8, 8, 8, 1], [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)], -1,
     [0, -1 / 3, -1 / 3, -1 / 3, 0, 0]),
])
def test_diabat_guess(z, bonds, charge, want):
    q, ok = guess_diabatic_charges(z, bonds, charge)
    assert ok
    assert torch.allclose(torch.tensor(q), torch.tensor(want, dtype=torch.float64), atol=1e-12)


def test_train_bonding_smoke_with_electrical_heads(tmp_path):
    import json

    import numpy as np
    from ase.io import write
    from rsfff.train.train_bonding import train
    from test_bonding_train import _frames

    rng = np.random.default_rng(0)
    write(str(tmp_path / "vac.xyz"),
          _frames(methanol(), 6, rng) + _frames(acetaldehyde(), 6, rng), format="extxyz")
    write(str(tmp_path / "probe.xyz"),
          _frames(methanol(), 4, rng, probes=True), format="extxyz")
    (tmp_path / "ref.json").write_text(json.dumps({"energies": {"H": -0.5, "C": -37.8, "O": -75.0}}))
    cfg = f"""
run_name: smoke_elec
device: cpu
dtype: float64
checkpoint_root: {tmp_path}/ckpt
data:
  path: [{tmp_path}/vac.xyz]
  reference_energies: {tmp_path}/ref.json
  holdout_fraction: 0.25
  seed: 0
bonding:
  streams:
    - {{name: probe, path: {tmp_path}/probe.xyz, weight: 1.0, batch_size: 4}}
features: {{cutoff: 5.0, n_max: 3, l_max: 2, selected_lambdas: [0, 1, 2], backend: e3nn, density_channels: 4}}
film:
  nonbonded: exclusions
  exclude_through: 4
  impropers: true
  atom_typing: degree
  torsions: true
  couplings: true
  field_features: true
  bond_features: ece
  electrical_features: ece
  ece_n_max: 3
  ece_l_max: 3
  ece_channels: 4
  ece_dim: 8
  elec_hidden: 16
  elec_depth: 1
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
    assert (tmp_path / "ckpt" / "smoke_elec" / "best.pt").exists()


# --- per-bond dissociation channels -------------------------------------------------------------

def _cation(name):
    """(positions, Z, bonds, charge): NH4+, CH3NH3+, acetate."""
    if name == "nh4":
        a = 0.59
        pos = torch.tensor([[0.0, 0.0, 0.0], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]])
        return pos, [7, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)], 1.0
    if name == "ch3nh3":
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.50, 0.0, 0.0],
                            [-0.36, 1.03, 0.0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
                            [1.87, 0.96, 0.0], [1.87, -0.48, 0.83], [1.87, -0.48, -0.83]])
        return pos, [6, 7, 1, 1, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)], 1.0
    pos, z, bonds = acetaldehyde()             # acetate: the H(C=O) replaced by an O
    pos = pos.clone()
    pos[3] = torch.tensor([2.05, -1.00, -0.01])
    z = [6, 6, 8, 8, 1, 1, 1]
    return pos, z, bonds, -1.0


def _charged_batch(pos, z, bonds, charge):
    b = make_batch((pos.double(), z, bonds))
    b.fragment_charge = torch.tensor([charge])
    return b


def _chan_model(**over):
    """A randomized model over H, C, N, O (bonding_model's species have no N)."""
    import types

    from rsfff.train.build_film import build_film_model

    torch.manual_seed(0)
    features = types.SimpleNamespace(cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
                                     backend="e3nn", density_channels=4)
    cfg = dict(hidden=32, block_dim=24, head_hidden=24, head_depth=1, equiv_channels=6,
               bonded_hidden=24, nonbonded="exclusions", exclude_through=4, impropers=True,
               atom_typing="element", torsions=True, couplings=True, term_hidden=16,
               elec_latents=False, **ELEC)
    cfg.update(over)
    ref = torch.tensor([-0.5013, -37.8450, -54.5840, -75.0093])
    model = build_film_model(features, types.SimpleNamespace(**cfg), [1, 6, 7, 8], ref).double()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _sub(pos, z, bonds, keep, charge):
    remap = {a: k for k, a in enumerate(keep)}
    sub = [(remap[a], remap[b]) for a, b in bonds if a in remap and b in remap]
    return _charged_batch(pos[keep], [z[k] for k in keep], sub, charge)


def test_proton_leaves_nh4_as_h_plus():
    m = _chan_model()
    pos, z, bonds, q = _cation("nh4")
    pos = pos.double().clone()
    pos[1] = pos[1] / pos[1].norm() * 40.0
    p = m(_charged_batch(pos, z, bonds, q)).parameters
    assert abs(float(p.q_perm[1]) - 1.0) < 1e-10
    assert p.response0.alpha[1].abs().max() < 1e-10                # H+ has no electrons
    assert p.mu_perm[1].abs().max() < 1e-10
    # the rest is NH3 on its own
    p3 = m(_sub(pos, z, bonds, [0, 2, 3, 4], 0.0)).parameters
    assert torch.allclose(p.q_perm[[0, 2, 3, 4]], p3.q_perm, atol=1e-10)
    assert torch.allclose(p.mu_perm[[0, 2, 3, 4]], p3.mu_perm, atol=1e-10)
    assert torch.allclose(p.response0.alpha[[0, 2, 3, 4]], p3.response0.alpha, atol=1e-10)


def test_two_protons_do_not_both_take_the_charge():
    m = _chan_model()
    pos, z, bonds, q = _cation("nh4")
    pos = pos.double().clone()
    for k, d in ((1, 40.0), (2, 45.0)):
        pos[k] = pos[k] / pos[k].norm() * d
    p = m(_charged_batch(pos, z, bonds, q)).parameters
    assert abs(float(p.q_perm.sum()) - 1.0) < 1e-12
    assert float(p.q_perm[[1, 2]].abs().max()) < 1e-10               # one exclusive +1: spent on neither
    assert abs(float(p.q_perm[[0, 3, 4]].sum()) - 1.0) < 1e-10       # NH2+ keeps it


def test_methylammonium_separates_into_ch3_plus_and_nh3():
    m = _chan_model()
    pos, z, bonds, q = _cation("ch3nh3")
    pos = pos.double().clone()
    shift = torch.tensor([-40.0, 0.0, 0.0], dtype=torch.float64)
    for k in (0, 2, 3, 4):                                          # the methyl, rigidly
        pos[k] = pos[k] + shift
    p = m(_charged_batch(pos, z, bonds, q)).parameters
    ch3, nh3 = [0, 2, 3, 4], [1, 5, 6, 7]
    assert abs(float(p.q_perm[ch3].sum()) - 1.0) < 1e-10
    assert abs(float(p.q_perm[nh3].sum())) < 1e-10
    pc = m(_sub(pos, z, bonds, ch3, 1.0)).parameters
    pn = m(_sub(pos, z, bonds, nh3, 0.0)).parameters
    for part, ref in ((ch3, pc), (nh3, pn)):
        assert torch.allclose(p.q_perm[part], ref.q_perm, atol=1e-10)
        assert torch.allclose(p.mu_perm[part], ref.mu_perm, atol=1e-10)
        assert torch.allclose(p.response0.alpha[part], ref.response0.alpha, atol=1e-10)


def test_acetate_oxygen_leaves_as_o_minus():
    m = _chan_model()
    pos, z, bonds, q = _cation("acetate")
    pos = pos.double().clone()
    d = pos[3] - pos[1]
    pos[3] = pos[1] + 40.0 * d / d.norm()
    p = m(_charged_batch(pos, z, bonds, q)).parameters
    assert abs(float(p.q_perm[3]) + 1.0) < 1e-10
    assert abs(float(p.q_perm[[0, 1, 2, 4, 5, 6]].sum())) < 1e-10
    assert torch.allclose(p.response0.alpha[3], free_atom_alpha(8, -1.0) * torch.eye(3), atol=1e-10)


def test_batch_channels_follow_bond_orientation():
    """Flows supplied on reversed covalent rows give the same parameters as the guess."""
    from rsfff.ff.film.diabats import guess_diabat

    m = _chan_model()
    pos, z, bonds, q = _cation("ch3nh3")
    b0 = _charged_batch(pos, z, bonds, q)
    rev = [(b, a) for a, b in bonds]
    d = guess_diabat(z, rev, q)
    b1 = _charged_batch(pos, z, rev, q)
    b1.diabat_charge = torch.tensor(d.q)
    b1.diabat_flow = torch.tensor(d.flow)
    b1.diabat_group = torch.tensor(d.group)
    p0, p1 = m(b0).parameters, m(b1).parameters
    assert torch.allclose(p0.q_perm, p1.q_perm, atol=1e-12)
    assert torch.allclose(p0.response0.alpha, p1.response0.alpha, atol=1e-12)


@pytest.mark.parametrize("name,z,bonds,charge,flow,group", [
    ("ammonium", [7, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (4, 0)], 1, [1, 1, 1, -1], [0, 0, 0, 0]),
    ("methylammonium", [6, 7, 1, 1, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)],
     1, [-1, 0, 0, 0, 1, 1, 1], [1, -1, -1, -1, 1, 1, 1]),
    ("acetate", [6, 6, 8, 8, 1, 1, 1], [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)], -1,
     [0, -0.5, -0.5, 0, 0, 0], [-1, 2, 2, -1, -1, -1]),
    ("water", [8, 1, 1], [(0, 1), (0, 2)], 0, [0, 0], [-1, -1]),
    ("cyclopropane", [6, 6, 6], [(0, 1), (1, 2), (2, 0)], 0, [0, 0, 0], [-1, -1, -1]),
])
def test_channel_guess(name, z, bonds, charge, flow, group):
    from rsfff.ff.film.diabats import guess_diabat

    d = guess_diabat(z, bonds, charge)
    assert torch.allclose(torch.tensor(d.flow), torch.tensor(flow, dtype=torch.float64), atol=1e-12)
    assert d.group.tolist() == group


def test_channel_charges_are_smooth_through_the_onset():
    """gradcheck of the charges and alpha along an N-H stretch inside the channel ramp."""
    m = _chan_model()
    pos, z, bonds, q = _cation("nh4")
    pos = pos.double().clone()
    for r in (1.012 + 0.5 + 0.1, 1.012 + 0.5 + 0.6):            # in the cubic and the linear part
        pos[1] = pos[1] / pos[1].norm() * r

        def f(x, pos=pos):
            pp = pos.clone()
            pp[1] = x
            out = m(_charged_batch(pp, z, bonds, q)).parameters
            return out.q_perm, out.response0.alpha

        assert torch.autograd.gradcheck(f, (pos[1].clone().requires_grad_(True),), atol=1e-6)


def test_channels_off_is_homolytic():
    m = _chan_model(elec_channels=False)
    pos, z, bonds, q = _cation("nh4")
    pos = pos.double().clone()
    pos[1] = pos[1] / pos[1].norm() * 40.0
    p = m(_charged_batch(pos, z, bonds, q)).parameters
    assert abs(float(p.q_perm[1])) < 1e-10
    assert torch.allclose(p.response0.alpha[1], free_atom_alpha(1, 0.0) * torch.eye(3), atol=1e-10)
