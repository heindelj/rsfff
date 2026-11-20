"""rsfff.ff.bonded.BondedModel: a fresh model is its tables; a trained one is invariant, conservative and bounded."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.bonded import BondedConfig, BondedModel, Molecule, atom_types_of, collate
from rsfff.ff.bonded.tables import type_key
from rsfff.ff.units import BOHR_ANG

ROOT = Path(__file__).resolve().parents[1]
TABLES = "params/bonded_monomers_wb97mv_tzvpd.json"


def molecules(seed=0, noise=0.06):
    rng = np.random.default_rng(seed)
    water = np.array([[0.0, 0.0, 0.117], [-0.763, 0.0, -0.468], [0.763, 0.0, -0.468]])
    h3o = np.array([[0.0, 0.0, 0.075], [0.941, 0.0, -0.201], [-0.470, 0.815, -0.201], [-0.470, -0.815, -0.201]])
    oh = np.array([[0.0, 0.0, -0.107], [0.0, 0.0, 0.858]])
    out = []
    for pos, q in ((water, 0), (h3o, 1), (oh, -1), (water, 0)):
        n = len(pos)
        out.append(Molecule(z=np.array([8] + [1] * (n - 1)), pos=pos + noise * rng.normal(size=pos.shape),
                            bonds=np.array([[0, i] for i in range(1, n)]), charge=q, energy=-76.0 + rng.normal(),
                            forces=0.01 * rng.normal(size=pos.shape)))
    return out


def build(randomize=True, seed=0, **over):
    torch.manual_seed(seed)
    cfg = BondedConfig(table_init=TABLES, n_max=3, l_max=2, channels=4, hidden=16, latent=4, **over)
    b = collate(molecules())
    m = BondedModel(cfg, atom_types_of(b), table_root=str(ROOT))
    m.set_statistics(b)
    if randomize:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if not n.startswith("tables."):
                    p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return m, b


def test_types_and_tables_from_the_fit():
    m, b = build(randomize=False)
    assert m.atom_types == [(1, 1), (8, 1), (8, 2), (8, 3)]
    assert set(m.tables.from_file["bond"]) == {"1.1-8.1", "1.1-8.2", "1.1-8.3"} and type_key((8, 2), (1, 1)) == "1.1-8.2"
    bp, tp = m.parameters_of(b, torch.zeros(b.graph.n_edges, 4, dtype=torch.float64))
    water = b.topo.bond_frag == 0
    assert torch.allclose(bp.r_eq[water], torch.full((2,), 1.8174151282568622)) and tp.bo_rho.shape == (3,)
    with pytest.raises(ValueError, match="not in the model"):
        BondedModel(m.config, [(1, 1), (8, 2)], table_root=str(ROOT)).tables.type_index(b.z, b.topo.degree)


def test_fresh_model_is_the_constant_model():
    m, b = build(randomize=False)
    e, f = m.energy_forces(b)
    e0, f0 = m.energy_forces(b, constant=True)
    assert torch.equal(e, e0) and torch.equal(f, f0)
    m2, _ = build(randomize=True)
    assert (m2.energy_forces(b)[0] - m2.energy_forces(b, constant=True)[0]).abs().max() > 1e-6


def test_invariance_and_permutation():
    m, b = build()
    e = m(b)["energy"]
    th = 0.9
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    for op in (rot, -rot):
        assert torch.allclose(m(b, b.positions @ op.T + 0.4)["energy"], e, atol=1e-10)
    mols = molecules()
    perm = [2, 0, 3, 1]                                    # relabel the hydronium's atoms
    inv = {old: new for new, old in enumerate(perm)}
    h = mols[1]
    mols[1] = Molecule(z=h.z[perm], pos=h.pos[perm], bonds=np.array([[inv[a], inv[c]] for a, c in h.bonds]), charge=1)
    assert torch.allclose(m(collate(mols))["energy"], e, atol=1e-10)
    assert torch.allclose(m(collate(molecules()[::-1]))["energy"], e.flip(0), atol=1e-10)   # batch order


def test_forces_are_conservative_and_in_hartree_per_bohr():
    m, b = build()
    _, f = m.energy_forces(b)
    h = 1e-5
    for atom, comp in ((0, 0), (4, 2), (8, 1)):
        e = []
        for s in (1, -1):
            x = b.positions.clone()
            x[atom, comp] += s * h
            e.append(float(m(b, x)["energy"].sum()))
        fd = -(e[0] - e[1]) / (2 * h) * BOHR_ANG
        assert abs(fd - float(f[atom, comp])) < 1e-7 * max(1.0, abs(fd))


def test_parameters_stay_inside_their_caps():
    m, b = build(param_cap=0.5, r_cap=0.1)
    with torch.no_grad():
        for d in m.dec.values():
            d.weight.mul_(1e3)                              # a wild readout
    bp, tp = m.parameters_of(b, m.latents(b))
    bp0, _ = m.parameters_of(b, torch.zeros(b.graph.n_edges, 4, dtype=torch.float64))
    assert ((bp.r_eq / bp0.r_eq).log().abs() <= 0.1 + 1e-12).all()
    assert ((bp.d / bp0.d).log().abs() <= 0.5 + 1e-12).all() and ((bp.k_theta / bp0.k_theta).log().abs() <= 0.5 + 1e-12).all()
    assert (bp.cos_theta_eq.abs() < 1).all() and (tp.ba_rho.abs() <= 0.95).all() and (tp.bo_rho.abs() <= 0.95).all()
    assert torch.isfinite(m(b)["energy"]).all()


def test_term_switches_and_reference_fit():
    m, b = build(bonded=("morse", "angle"))
    assert {k for k in m.dec if k in ("bond", "angle", "bb", "ba", "pair", "improper")} == {"bond", "angle"}
    out = m(b)
    assert out["terms"].bb_rho is None and out["terms"].bo_rho is None
    m.fit_reference(b)
    resid = m(b)["energy"] - b.energy
    comp = torch.zeros(b.n_mol, 4, dtype=torch.float64).index_put(
        (b.mol_idx, m.tables.type_index(b.z, b.topo.degree)), torch.ones(len(b.z), dtype=torch.float64), accumulate=True)
    assert (comp.T @ resid).abs().max() < 1e-8            # least squares: residual orthogonal to the composition
    with pytest.raises(NotImplementedError):
        BondedModel(BondedConfig(bonded=("morse", "torsion")), [(1, 1), (8, 2)])
    mi, _ = build(bonded=("morse", "angle", "improper"))
    assert mi(b)["bonded"].c_chi.shape == (3,)
