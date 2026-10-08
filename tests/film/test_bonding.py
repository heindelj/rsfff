"""The monomer-bonded study (``bonding`` branch): explicit topology, typed priors, torsions and
couplings, external sources, derivative-defined response properties, O(F^2) field features.

What has to hold:

* **topology** from an explicit covalent graph: the right counts for the paper's molecules,
  graph separations, and the legacy rule unchanged for water;
* **terms**: zero at initialization (switching a family on reproduces the model without it),
  conservative forces, rotation and atom-relabeling invariance once trained away from zero;
* **external sources**: an all-zero source changes nothing; the derivative dipole equals the
  summed multipoles (Hellmann-Feynman through the variational solve); alpha = d mu/dF matches
  finite fields; forces stay conservative with probe charges present;
* **field features**: identically inert in vacuum, and **second order**: they leave the
  zero-field dipole untouched while changing the polarizability; their linear invariants are
  rotation invariant and blind to a constant potential.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from rsfff.ff.external import ExternalSources, point_fields
from rsfff.ff.film import StateDescriptor
from rsfff.ff.film.bonded import BondedTopology
from rsfff.ff.film.terms import term_energies
from rsfff.ff.units import BOHR_ANG
from rsfff.train.build_film import build_film_model
from rsfff.train.data import Batch

SPECIES = [1, 6, 8]
REF = torch.tensor([-0.5013, -37.8450, -75.0093])


def methanol() -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
    pos = torch.tensor([
        [0.000, 0.000, 0.000],     # C
        [1.430, 0.000, 0.000],     # O
        [1.750, 0.900, 0.000],     # H(O)
        [-0.360, 1.030, 0.000],
        [-0.360, -0.510, 0.890],
        [-0.360, -0.510, -0.890],
    ])
    return pos, [6, 8, 1, 1, 1, 1], [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)]


def acetaldehyde():
    pos = torch.tensor([
        [0.000, 0.000, 0.000],     # C methyl
        [1.500, 0.000, 0.000],     # C carbonyl
        [2.110, 1.050, 0.020],     # O
        [2.050, -0.950, -0.010],   # H(C=O)
        [-0.370, 1.020, 0.050],
        [-0.380, -0.540, 0.870],
        [-0.360, -0.490, -0.900],
    ])
    return pos, [6, 6, 8, 1, 1, 1, 1], [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)]


def make_batch(mol, *, ext=None, shift=None) -> Batch:
    pos, z, bonds = mol
    pos = pos.to(torch.get_default_dtype())
    if shift is not None:
        pos = pos + shift
    n = pos.shape[0]
    return Batch(
        positions=pos.clone(),
        atomic_numbers=torch.tensor(z),
        batch_idx=torch.zeros(n, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.zeros(n, dtype=torch.long),
        fragment_charge=torch.zeros(1),
        fragment_two_s=torch.zeros(1),
        fragment_to_batch=torch.zeros(1, dtype=torch.long),
        n_fragments=1,
        covalent_bonds=torch.tensor(bonds).t(),
        external=ext,
    )


def bonding_model(seed=0, *, randomize=False, **over):
    torch.manual_seed(seed)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    cfg = dict(
        hidden=32, block_dim=24, head_hidden=24, head_depth=1, equiv_channels=6,
        bonded_hidden=24, nonbonded="exclusions", exclude_through=4, impropers=True,
        atom_typing="degree", torsions=True, couplings=True, term_hidden=16,
    )
    cfg.update(over)
    model = build_film_model(features, types.SimpleNamespace(**cfg), SPECIES, REF)
    if randomize:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def probes(n=4, radius=4.0, seed=3):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g)
    pos = radius * v / v.norm(dim=-1, keepdim=True) + torch.tensor([0.7, 0.2, 0.0])
    q = torch.tensor([0.5, -0.4, 0.3, -0.6])[:n]
    return ExternalSources(charge_positions=pos, charges=q,
                           charge_batch=torch.zeros(n, dtype=torch.long))


# --- topology ----------------------------------------------------------------------------------

def _topo(z, bonds):
    n = len(z)
    b = make_batch((torch.zeros(n, 3), z, bonds))
    st = StateDescriptor.from_batch(b, torch.zeros(n, dtype=torch.long), 1)
    return BondedTopology.from_state(st, b.atomic_numbers, b.covalent_bonds)


@pytest.mark.parametrize("name,z,bonds,counts", [
    # (bonds, angles, torsions, angle pairs, impropers) -- the RDKit inventory of the plan
    ("ethane", [6, 6, 1, 1, 1, 1, 1, 1],
     [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)], (7, 12, 9, 24, 0)),
    ("methanol", [6, 8, 1, 1, 1, 1], [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)], (5, 7, 3, 12, 0)),
    ("acetate", [6, 6, 8, 8, 1, 1, 1],
     [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)], (6, 9, 6, 15, 3)),
    ("formamide", [6, 7, 8, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)], (5, 6, 4, 6, 6)),
])
def test_topology_counts(name, z, bonds, counts):
    t = _topo(z, bonds)
    got = (t.bond_index.shape[1], t.angle_index.shape[1], t.n_torsions, t.n_angle_pairs,
           t.n_impropers)
    assert got == counts


def test_separation_and_degree():
    t = _topo([6, 6, 1, 1, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)])
    pairs = torch.tensor([[0, 2, 2, 2], [1, 3, 5, 0]])
    assert t.separation(pairs, 8).tolist() == [1, 2, 3, 1]
    assert t.degree.tolist() == [4, 4, 1, 1, 1, 1, 1, 1]
    # the legacy rule (no bonds) for water is unchanged: 2 bonds, 1 angle, nothing else
    n = 3
    b = make_batch((torch.zeros(3, 3), [8, 1, 1], [(0, 1), (0, 2)]))
    st = StateDescriptor.from_batch(b, torch.zeros(n, dtype=torch.long), 1)
    legacy = BondedTopology.from_state(st, b.atomic_numbers, None)
    explicit = BondedTopology.from_state(st, b.atomic_numbers, b.covalent_bonds)
    assert torch.equal(legacy.bond_index, explicit.bond_index)
    assert legacy.n_torsions == 0 and legacy.n_angle_pairs == 0


def test_typed_priors_separate_bond_orders():
    m = bonding_model()
    head = m.network.bonded_head
    t = _topo(*acetaldehyde()[1:])
    sp = m.projector.species_index(torch.tensor(acetaldehyde()[1]))
    tidx = head.type_index(sp, t)
    r_eq = head.bond_table[tidx[t.bond_index[0]], tidx[t.bond_index[1]], 0].exp() * BOHR_ANG
    # C-C 1.53, C=O 1.215: typed by degree, not by element pair alone
    lookup = {tuple(p): float(r) for p, r in zip(t.bond_index.t().tolist(), r_eq)}
    assert abs(lookup[(0, 1)] - 1.53) < 1e-6
    assert abs(lookup[(1, 2)] - 1.215) < 1e-6


# --- torsions and couplings --------------------------------------------------------------------

def test_terms_zero_at_init():
    m = bonding_model()
    out = m(make_batch(acetaldehyde()))
    for e, _ in term_energies(out.parameters and make_batch(acetaldehyde()).positions,
                              out.topology, out.parameters.bonded0, out.parameters.terms0):
        assert torch.equal(e, torch.zeros_like(e))


def _energy_and_forces(model, batch):
    batch.positions.requires_grad_(True)
    e = model(batch).energy.sum()
    (g,) = torch.autograd.grad(e, batch.positions)
    return e.detach(), -g


def test_terms_conservative_forces():
    m = bonding_model(randomize=True).double()
    b = make_batch(acetaldehyde())
    out = m(b)
    assert any(e.abs().max() > 0 for e, _ in term_energies(
        b.positions, out.topology, out.parameters.bonded0, out.parameters.terms0))
    _, f = _energy_and_forces(m, make_batch(acetaldehyde()))
    h = 1e-5
    for atom, comp in ((0, 0), (2, 1), (5, 2)):
        ep, em = [], []
        for sgn, store in ((1, ep), (-1, em)):
            bb = make_batch(acetaldehyde())
            with torch.no_grad():
                bb.positions[atom, comp] += sgn * h
            store.append(float(m(bb).energy.sum()))
        fd = -(ep[0] - em[0]) / (2 * h)
        assert abs(fd - float(f[atom, comp])) < 1e-6 * max(1.0, abs(fd))


def test_terms_invariance():
    m = bonding_model(randomize=True).double()
    e0 = float(m(make_batch(acetaldehyde())).energy)
    # rotation + translation
    th = 0.7
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0],
                        [0.0, 0.0, 1.0]], dtype=torch.float64)
    pos, z, bonds = acetaldehyde()
    e1 = float(m(make_batch((pos.double() @ rot.T + 1.3, z, bonds))).energy)
    assert abs(e1 - e0) < 1e-10
    # relabel the atoms (reverses the stored torsion orientation for some instances)
    perm = [1, 0, 2, 3, 6, 4, 5]
    inv = {old: new for new, old in enumerate(perm)}
    e2 = float(m(make_batch((pos[perm], [z[i] for i in perm],
                             [(inv[a], inv[b]) for a, b in bonds]))).energy)
    assert abs(e2 - e0) < 1e-10


# --- external sources and response properties -------------------------------------------------

def test_zero_external_changes_nothing():
    m = bonding_model(randomize=True).double()
    e_none = m(make_batch(methanol())).energy
    zero = ExternalSources(field=torch.zeros(1, 3, dtype=torch.float64))
    e_zero = m(make_batch(methanol(), ext=zero)).energy
    assert torch.allclose(e_none, e_zero, atol=1e-12)


def test_derivative_dipole_is_hellmann_feynman():
    m = bonding_model(randomize=True).double()
    b = make_batch(methanol())
    out, mu, theta, alpha = m.response_properties(b)
    p = out.parameters
    r = b.positions / BOHR_ANG
    mu_sum = (p.q_perm[:, None] * r).sum(0) + p.mu_perm.sum(0)
    assert torch.allclose(mu[0], mu_sum, atol=1e-8)
    assert torch.allclose(alpha, alpha.transpose(-1, -2), atol=1e-7)
    assert torch.allclose(theta.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(1, dtype=torch.float64), atol=1e-10)


def test_alpha_matches_finite_field():
    m = bonding_model(randomize=True, field_features=True).double()
    _, _, _, alpha = m.response_properties(make_batch(methanol()), quadrupole=False)
    h = 2e-4
    cols = []
    for j in range(3):
        mus = []
        for sgn in (1, -1):
            f = torch.zeros(1, 3, dtype=torch.float64)
            f[0, j] = sgn * h
            _, mu, _, _ = m.response_properties(
                make_batch(methanol(), ext=ExternalSources(field=f)),
                polarizability=False, quadrupole=False, create_graph=False,
            )
            mus.append(mu[0])
        cols.append((mus[0] - mus[1]) / (2 * h))
    fd = torch.stack(cols, dim=1)
    assert torch.allclose(alpha[0], fd, atol=1e-5 * float(fd.abs().max()) + 1e-8)


def test_forces_with_probes_and_field_features():
    m = bonding_model(randomize=True, field_features=True).double()
    ext = probes()
    ext = ExternalSources(ext.charge_positions.double(), ext.charges.double(), ext.charge_batch)
    _, f = _energy_and_forces(m, make_batch(methanol(), ext=ext))
    h = 1e-5
    for atom, comp in ((0, 1), (2, 0)):
        es = []
        for sgn in (1, -1):
            bb = make_batch(methanol(), ext=ext)
            with torch.no_grad():
                bb.positions[atom, comp] += sgn * h
            es.append(float(m(bb).energy.sum()))
        fd = -(es[0] - es[1]) / (2 * h)
        assert abs(fd - float(f[atom, comp])) < 1e-6 * max(1.0, abs(fd))


# --- field features: inert in vacuum, second order ---------------------------------------------

def _set_w2(model, value):
    with torch.no_grad():
        model.network.field_features.W2.weight.copy_(value)


def test_field_features_inert_in_vacuum_and_second_order():
    m = bonding_model(randomize=True, field_features=True).double()
    w2 = m.network.field_features.W2.weight
    rand = 0.3 * torch.randn(w2.shape, generator=torch.Generator().manual_seed(5)).double()

    _set_w2(m, torch.zeros_like(w2))
    e_a = m(make_batch(methanol())).energy
    _, mu_a, th_a, al_a = m.response_properties(make_batch(methanol()))
    _set_w2(m, rand)
    e_c = m(make_batch(methanol())).energy
    _, mu_c, th_c, al_c = m.response_properties(make_batch(methanol()))

    assert torch.equal(e_a, e_c)                                  # C == A in vacuum, bitwise
    assert torch.allclose(mu_a, mu_c, atol=1e-12)                 # no first-order term
    assert torch.allclose(th_a, th_c, atol=1e-12)
    assert (al_c - al_a).abs().max() > 1e-6                       # ... but alpha moves


def test_field_invariants_rotation_and_gauge():
    m = bonding_model(randomize=True, field_features=True).double()
    ff = m.network.field_features
    ext = probes()
    ext = ExternalSources(ext.charge_positions.double(), ext.charges.double(), ext.charge_batch)

    def invariants(pos, ext, phi_shift=0.0):
        b = make_batch((pos, methanol()[1], methanol()[2]), ext=ext)
        pf = m.projector(b, StateDescriptor.from_batch(
            b, m.projector.species_index(b.atomic_numbers), m.projector.featurizer.n_species))
        phi, e, g = point_fields(ext, b.positions, b.batch_idx)
        return ff.linear_invariants(phi + phi_shift, e, g, b.fragment_idx, 1, pf.x_in)

    pos = methanol()[0].double()
    x0 = invariants(pos, ext)
    assert torch.allclose(x0, invariants(pos, ext, phi_shift=0.37), atol=1e-12)
    th = 1.1
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)],
                        [0.0, math.sin(th), math.cos(th)]], dtype=torch.float64)
    ext_r = ExternalSources(ext.charge_positions @ rot.T, ext.charges, ext.charge_batch)
    assert torch.allclose(x0, invariants(pos @ rot.T, ext_r), atol=1e-9)


# --- data: the covalent graph and external sources through the loader ---------------------------

def _write_extxyz(path, frames):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    out = []
    for pos, z, frag, info in frames:
        a = Atoms(numbers=z, positions=pos)
        a.arrays["fragment_idx"] = frag
        a.info.update(info)
        a.calc = SinglePointCalculator(a, energy=-1.0, forces=[[0.0, 0.0, 0.0]] * len(z))
        out.append(a)
    write(str(path), out, format="extxyz")


def test_loader_bonds_follow_the_fragment_sort(tmp_path):
    import numpy as np
    from rsfff.train.data import fragment_view, load_extxyz

    # two methanols written interleaved (fragment order 1, 0, 1, 0, ...) so the loader's
    # fragment sort permutes the atoms; the bonds are given in the file's order
    pos, z, bonds = methanol()
    pos = pos.numpy()
    a = [(pos[i], z[i], 0, i) for i in range(6)]
    b = [(pos[i] + np.array([5.0, 0, 0]), z[i], 1, i) for i in range(6)]
    order = [x for pair in zip(b, a) for x in pair]          # b0 a0 b1 a1 ...
    file_index = {(f, k): n for n, (_, _, f, k) in enumerate(order)}
    flat = []
    for f in (0, 1):
        for i, j in bonds:
            flat += [file_index[(f, i)], file_index[(f, j)]]
    _write_extxyz(tmp_path / "m.xyz", [(
        np.stack([o[0] for o in order]), [o[1] for o in order],
        np.array([o[2] for o in order]),
        {"bonds": np.array(flat), "fragment_charges": np.zeros(2),
         "fragment_multiplicities": np.ones(2), "fragment_energies": np.array([-1.0, -1.0]),
         "ext_charges": np.array([9.0, 0, 0, 0.5, -9.0, 0, 0, -0.5]),
         "ext_field": np.array([0.0, 0.001, 0.0])},
    )])
    ds = load_extxyz(tmp_path / "m.xyz", dtype=torch.float64)
    batch = ds.flat_batch([0])
    cb = batch.covalent_bonds
    assert cb.shape == (2, 10)
    # every bond joins atoms ~bonded apart, in the same fragment
    d = (batch.positions[cb[0]] - batch.positions[cb[1]]).norm(dim=-1)
    assert float(d.max()) < 1.5
    assert torch.equal(batch.fragment_idx[cb[0]], batch.fragment_idx[cb[1]])
    assert batch.external.charges.tolist() == [0.5, -0.5]
    assert torch.allclose(batch.external.field, torch.tensor([[0.0, 0.001, 0.0]], dtype=torch.float64))
    # fragment views carry the bonds into the exploded frames
    fv = fragment_view(ds).flat_batch([0, 1])
    assert fv.covalent_bonds.shape == (2, 10)
    d = (fv.positions[fv.covalent_bonds[0]] - fv.positions[fv.covalent_bonds[1]]).norm(dim=-1)
    assert float(d.max()) < 1.5


def test_loader_refuses_multi_heavy_fragment_without_bonds(tmp_path):
    import numpy as np
    from rsfff.train.data import load_extxyz

    pos, z, _ = methanol()
    _write_extxyz(tmp_path / "m.xyz", [(pos.numpy(), z, np.zeros(6, dtype=int),
                                        {"fragment_charges": np.zeros(1)})])
    with pytest.raises(ValueError, match="bonds"):
        load_extxyz(tmp_path / "m.xyz")


def test_derivative_alpha_equals_closed_form_without_field_features():
    """Model A: -d2E/dF2 through the external-source path == charge flow + on-site alpha."""
    m = bonding_model(randomize=True).double()
    b = make_batch(methanol())
    closed = m(b, with_polarizability=True).polarizability[0] / BOHR_ANG ** 2
    _, _, _, alpha = m.response_properties(make_batch(methanol()), quadrupole=False)
    assert torch.allclose(alpha[0], closed, rtol=1e-6, atol=1e-8)
