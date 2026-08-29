"""Phase-3 tests: bonded topology, functional forms against pyCMM, and the parameter head."""

from __future__ import annotations

import math

import torch

from rsfff.ff.film import StateDescriptor
from rsfff.ff.film.bonded import (
    BondedParameterHead,
    BondedTopology,
    DEFAULT_ANGLE_PRIOR,
    DEFAULT_BOND_PRIOR,
    cosine_angle_energy,
    morse_energy,
)
from rsfff.ff.units import BOHR_ANG

from film_helpers import make_projector, make_state, water_cluster_batch


# pyCMM water.xml, atomic units.
R_EQ, D_E, K_B = DEFAULT_BOND_PRIOR[(1, 8)]
THETA_EQ, K_THETA = DEFAULT_ANGLE_PRIOR[8]


def test_morse_matches_pycmm_minus_d():
    """Our well-referenced Morse is pyCMM's ``d(1-exp(-a(r-req)))^2`` minus ``d``."""
    r = torch.linspace(1.2, 6.0, 50)
    r_eq = torch.full_like(r, R_EQ)
    d = torch.full_like(r, D_E)
    k = torch.full_like(r, K_B)
    beta = math.sqrt(K_B / (2.0 * D_E))
    pycmm = D_E * (1.0 - torch.exp(-beta * (r - R_EQ))) ** 2
    assert torch.allclose(morse_energy(r, r_eq, d, k), pycmm - D_E, atol=1e-14)
    # limits: -D at the minimum, 0 at dissociation
    at_min = morse_energy(torch.tensor([R_EQ]), r_eq[:1], d[:1], k[:1])
    assert torch.allclose(at_min, torch.tensor([-D_E]), atol=1e-14)
    far = morse_energy(torch.tensor([50.0]), r_eq[:1], d[:1], k[:1])
    assert far.abs().item() < 1e-12


def test_cosine_angle_matches_pycmm():
    theta = torch.linspace(0.5, math.pi - 0.1, 40)
    expected = 0.5 * K_THETA * (torch.cos(theta) - math.cos(THETA_EQ)) ** 2
    got = cosine_angle_energy(
        torch.cos(theta), torch.tensor(THETA_EQ).cos().expand_as(theta),
        torch.full_like(theta, K_THETA),
    )
    assert torch.allclose(got, expected, atol=1e-14)


def test_topology_water_cluster():
    """w3: 6 O-H bonds, 3 H-O-H angles, correct fragments, no H-H bond, weight one."""
    batch = water_cluster_batch(3)
    proj = make_projector()
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)

    assert topo.bond_index.shape[1] == 6
    assert topo.angle_index.shape[1] == 3
    z = batch.atomic_numbers
    # every bond touches the O (atom 0 of each water); H-H excluded
    assert bool(((z[topo.bond_index[0]] == 8) | (z[topo.bond_index[1]] == 8)).all())
    # every angle apex is an O
    assert bool((z[topo.angle_index[1]] == 8).all())
    assert torch.equal(topo.bond_frag, torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.equal(topo.angle_frag, torch.tensor([0, 1, 2]))
    assert torch.allclose(topo.bond_weight, torch.ones(6), atol=1e-15)
    assert torch.allclose(topo.angle_weight, torch.ones(3), atol=1e-15)


def test_head_starts_at_pycmm_water():
    """Zero-init readouts: a fresh head returns exactly the prior table values."""
    batch = water_cluster_batch(2)
    proj = make_projector()
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    species_idx = proj.species_index(batch.atomic_numbers)

    head = BondedParameterHead(8, [1, 8])
    z = torch.randn(6, 8)
    params = head(z, None, None, species_idx, topo)

    assert torch.allclose(params.r_eq, torch.full((4,), R_EQ), atol=1e-12)
    assert torch.allclose(params.d, torch.full((4,), D_E), atol=1e-12)
    assert torch.allclose(params.k, torch.full((4,), K_B), atol=1e-12)
    assert torch.allclose(
        params.cos_theta_eq, torch.full((2,), math.cos(THETA_EQ)), atol=1e-12
    )
    assert torch.allclose(params.k_theta, torch.full((2,), K_THETA), atol=1e-12)
    assert torch.count_nonzero(params.delta_iso) == 0


def test_env_delta_gated_to_zero():
    """With gate = 0 the env-dressed parameters equal theta_0 bitwise, whatever the weights."""
    batch = water_cluster_batch(2)
    proj = make_projector()
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    species_idx = proj.species_index(batch.atomic_numbers)

    head = BondedParameterHead(8, [1, 8])
    with torch.no_grad():  # activate every branch
        for p in head.parameters():
            p.add_(0.05 * torch.randn(p.shape))
    z_iso = torch.randn(6, 8)
    z_joined = torch.randn(6, 8)

    theta0 = head(z_iso, None, None, species_idx, topo)
    gated_off = head(z_iso, z_joined, torch.zeros(6), species_idx, topo)
    for name in ("r_eq", "d", "k", "cos_theta_eq", "k_theta"):
        assert torch.equal(getattr(theta0, name), getattr(gated_off, name)), name

    dressed = head(z_iso, z_joined, torch.ones(6), species_idx, topo)
    assert not torch.allclose(dressed.r_eq, theta0.r_eq)


def test_h_permutation_invariance():
    """Swapping the two H latents of a water leaves the parameters unchanged."""
    batch = water_cluster_batch(1)
    proj = make_projector()
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    species_idx = proj.species_index(batch.atomic_numbers)

    head = BondedParameterHead(8, [1, 8])
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.05 * torch.randn(p.shape))
    z = torch.randn(3, 8)
    z_swapped = z[torch.tensor([0, 2, 1])]

    a = head(z, None, None, species_idx, topo)
    b = head(z_swapped, None, None, species_idx, topo)
    # bonds swap order under the H swap; compare as sorted multisets via sum/product
    assert torch.allclose(a.r_eq.sort().values, b.r_eq.sort().values, atol=1e-12)
    assert torch.allclose(a.cos_theta_eq, b.cos_theta_eq, atol=1e-12)
    assert torch.allclose(a.k_theta, b.k_theta, atol=1e-12)


def test_energy_at_equilibrium_geometry():
    """A monomer near equilibrium: bond energy ~ -2D, angle energy tiny, forces ~ 0 slope."""
    # exact prior equilibrium geometry, in Angstrom
    r0 = R_EQ * BOHR_ANG
    half = THETA_EQ / 2.0
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [r0 * math.sin(half), 0.0, r0 * math.cos(half)],
            [-r0 * math.sin(half), 0.0, r0 * math.cos(half)],
        ]
    )
    batch = water_cluster_batch(1)
    batch.positions = positions
    proj = make_projector()
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    species_idx = proj.species_index(batch.atomic_numbers)
    head = BondedParameterHead(8, [1, 8])
    params = head(torch.randn(3, 8), None, None, species_idx, topo)

    r, cos_t = topo.geometry(positions)
    e_bond, e_angle = params.energy(r, cos_t, topo)
    assert torch.allclose(e_bond.sum(), torch.tensor(-2.0 * D_E), atol=1e-10)
    assert e_angle.abs().max() < 1e-12
