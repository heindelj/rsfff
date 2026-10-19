"""New fragment types for the film model: NH3 (impropers), HF, and monatomic Na+/Cl- ions.

What has to hold:

* **improper topology** -- one row per leg of every bond-degree-3 center (NH3's N), none for
  water's O or HF, read off the covalent graph like everything else;
* **improper form** -- even under inversion through the center plane and under ``j <-> k``,
  zero at planarity, ``-k c^2/2`` at ``sin^2 chi = c``, and ``sin^2 chi`` the true Wilson angle;
* **opt-in** -- a head without impropers has exactly the old parameter set (water
  checkpoints load), and with them a fresh head returns the prior table;
* the NH3 prior reproduces its fit (HNH 106.7, umbrella below the E bend);
* **full model** on NH3 / HF / Na+Cl- clusters: forward, forces (double backward), the
  isolated-fragment vertex (zero induction), and a monatomic ion's fragment energy being its
  reference energy alone.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from rsfff.ff.film.bonded import (
    BondedParameterHead,
    BondedTopology,
    DEFAULT_IMPROPER_PRIOR,
    IMPROPER_ANGLE_PRIOR,
    improper_energy,
    wilson_improper_energy,
)
from rsfff.ff.film import StateDescriptor
from rsfff.train.build_film import build_film_model
from rsfff.train.data import Batch

from film_helpers import make_projector, make_state, water_cluster_batch


# --- geometry builders -------------------------------------------------------------------------

def nh3(r: float = 1.012, hnh_deg: float = 106.7) -> torch.Tensor:
    """C3v NH3 (Angstrom), N at the origin, C3 axis along z, H below."""
    c = math.cos(math.radians(hnh_deg))
    cb2 = (c + 0.5) / 1.5
    cb, sb = math.sqrt(cb2), math.sqrt(1.0 - cb2)
    h = [[r * sb * math.cos(2 * math.pi * k / 3), r * sb * math.sin(2 * math.pi * k / 3), -r * cb]
         for k in range(3)]
    return torch.tensor([[0.0, 0.0, 0.0]] + h)


def _batch(frames, numbers, charges, frag_sizes) -> Batch:
    positions = torch.cat(frames).to(torch.get_default_dtype())
    n = positions.shape[0]
    n_frag = len(frag_sizes)
    return Batch(
        positions=positions,
        atomic_numbers=torch.tensor(numbers),
        batch_idx=torch.zeros(n, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.repeat_interleave(torch.arange(n_frag), torch.tensor(frag_sizes)),
        fragment_charge=torch.tensor(charges, dtype=torch.get_default_dtype()),
        fragment_two_s=torch.zeros(n_frag),
        fragment_to_batch=torch.zeros(n_frag, dtype=torch.long),
        n_fragments=n_frag,
    )


def nh3_dimer(jitter: float = 0.03, seed: int = 3) -> Batch:
    g = torch.Generator().manual_seed(seed)
    a = nh3() + jitter * torch.randn(4, 3, generator=g)
    b = nh3() + torch.tensor([3.3, 0.0, 0.0]) + jitter * torch.randn(4, 3, generator=g)
    return _batch([a, b], [7, 1, 1, 1] * 2, [0.0, 0.0], [4, 4])


def hf_dimer() -> Batch:
    a = torch.tensor([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]])
    b = torch.tensor([[2.75, 0.0, 0.0], [3.2, 0.8, 0.0]])
    return _batch([a, b], [9, 1, 9, 1], [0.0, 0.0], [2, 2])


def nacl_pair(r: float = 2.4) -> Batch:
    return _batch([torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])], [11, 17], [1.0, -1.0], [1, 1])


def nacl_dimer() -> Batch:
    """(NaCl)2 rhombus: four monatomic fragments."""
    x = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [0.0, 2.5, 0.0]])
    return _batch([x], [11, 17, 11, 17], [1.0, -1.0, 1.0, -1.0], [1, 1, 1, 1])


def _topology(batch: Batch, neighbor_types) -> BondedTopology:
    proj = make_projector(neighbor_types=list(neighbor_types))
    state = make_state(batch, proj)
    return BondedTopology.from_state(state, batch.atomic_numbers)


# --- topology ----------------------------------------------------------------------------------

def test_nh3_improper_topology():
    batch = nh3_dimer()
    topo = _topology(batch, [1, 7])
    assert topo.bond_index.shape[1] == 6
    assert topo.angle_index.shape[1] == 6
    assert topo.n_impropers == 6                                # 3 legs x 2 nitrogens
    z = batch.atomic_numbers
    assert bool((z[topo.improper_index[0]] == 7).all())        # centers are N
    assert bool((z[topo.improper_index[1:]] == 1).all())       # legs / partners are H
    # each center's three rows use each H once as the leg
    for frag in (0, 1):
        legs = topo.improper_index[1][topo.improper_frag == frag]
        assert sorted(legs.tolist()) == [4 * frag + 1, 4 * frag + 2, 4 * frag + 3]
    assert torch.allclose(topo.improper_weight, torch.ones(6), atol=1e-15)


def test_no_impropers_for_water_hf_or_ions():
    assert _topology(water_cluster_batch(2), [1, 8]).n_impropers == 0
    topo = _topology(hf_dimer(), [1, 9])
    assert topo.n_impropers == 0 and topo.bond_index.shape[1] == 2
    assert topo.angle_index.shape[1] == 0
    topo = _topology(nacl_dimer(), [11, 17])
    assert topo.bond_index.shape[1] == 0 and topo.n_impropers == 0


# --- the functional form ----------------------------------------------------------------------

def _wilson_sin2_reference(x: torch.Tensor, c, leg, j, k) -> float:
    """sin^2 of the angle between bond c->leg and the plane spanned by c->j, c->k."""
    bl, bj, bk = x[leg] - x[c], x[j] - x[c], x[k] - x[c]
    n = torch.cross(bj, bk, dim=0)
    angle_to_normal = torch.arccos((bl @ n) / (bl.norm() * n.norm()))
    return float(torch.cos(angle_to_normal) ** 2)             # sin(pi/2 - a) = cos a


def test_improper_sin2_is_the_wilson_angle():
    g = torch.Generator().manual_seed(0)
    batch = nh3_dimer(jitter=0.1)
    topo = _topology(batch, [1, 7])
    s2 = topo.improper_sin2(batch.positions)
    for n in range(topo.n_impropers):
        c, leg, j, k = topo.improper_index[:, n].tolist()
        assert abs(float(s2[n]) - _wilson_sin2_reference(batch.positions, c, leg, j, k)) < 1e-10
    # equilibrium NH3: chi = 61.2 deg from the plane of the other two bonds
    single = _batch([nh3()], [7, 1, 1, 1], [0.0], [4])
    s2_eq = _topology(single, [1, 7]).improper_sin2(single.positions)
    assert torch.allclose(s2_eq, torch.full((3,), 0.76825), atol=1e-4)


def test_improper_even_under_inversion_and_partner_swap():
    pos = nh3(hnh_deg=110.0)
    single = _batch([pos], [7, 1, 1, 1], [0.0], [4])
    topo = _topology(single, [1, 7])
    inverted = pos * torch.tensor([1.0, 1.0, -1.0])            # umbrella flipped through N
    assert torch.allclose(topo.improper_sin2(pos), topo.improper_sin2(inverted), atol=1e-14)
    swapped = BondedTopology(**{**topo.__dict__})
    swapped.improper_index = topo.improper_index[[0, 1, 3, 2]]
    assert torch.allclose(topo.improper_sin2(pos), swapped.improper_sin2(pos), atol=1e-14)
    planar = nh3(hnh_deg=120.0)
    assert float(topo.improper_sin2(planar).abs().max()) < 1e-12


def test_wilson_form_limits():
    s = torch.linspace(0.0, 1.0, 101)
    k = torch.tensor(0.3)
    # zero at planarity whatever c is
    for c in (-0.5, 0.0, 0.4):
        assert float(wilson_improper_energy(torch.tensor(0.0), torch.tensor(c), k)) == 0.0
    # c > 0: double well with minimum -k c^2 / 2 at s = c
    e = wilson_improper_energy(s, torch.tensor(0.4), k)
    assert abs(float(s[e.argmin()]) - 0.4) < 1e-9
    assert abs(float(e.min()) + 0.5 * 0.3 * 0.16) < 1e-12
    # c < 0: monotone restoring toward planarity
    e = wilson_improper_energy(s, torch.tensor(-0.5), k)
    assert bool((e[1:] > e[:-1]).all())


# --- the head ----------------------------------------------------------------------------------

def test_head_without_impropers_has_the_old_parameter_set():
    head = BondedParameterHead(16, [1, 8])
    assert not any("improper" in name for name, _ in head.named_parameters())
    assert not any("improper" in name for name in head.state_dict())


def test_head_with_impropers_starts_at_the_prior():
    batch = nh3_dimer()
    proj = make_projector(neighbor_types=[1, 7])
    state = make_state(batch, proj)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    species_idx = proj.species_index(batch.atomic_numbers)
    head = BondedParameterHead(16, [1, 7], impropers=True)
    z = torch.randn(batch.positions.shape[0], 16)
    p = head(z, None, None, species_idx, topo)
    c0, k0 = DEFAULT_IMPROPER_PRIOR[7]
    assert torch.allclose(p.c_chi, torch.full((6,), c0), atol=1e-12)
    assert torch.allclose(p.k_chi, torch.full((6,), k0), atol=1e-12)
    theta0, kth0 = IMPROPER_ANGLE_PRIOR[7]
    assert torch.allclose(p.cos_theta_eq, torch.full((6,), math.cos(theta0)), atol=1e-12)
    assert torch.allclose(p.k_theta, torch.full((6,), kth0), atol=1e-12)
    # the regularizer sees the improper deviations too: 6 bonds x3 + 6 angles x2 + 6 legs x2
    assert p.delta_iso.numel() == 18 + 12 + 12
    # a water batch through the same head: no impropers, empty improper parameters
    wb = water_cluster_batch(2)
    wproj = make_projector(neighbor_types=[1, 7, 8])
    wtopo = BondedTopology.from_state(make_state(wb, wproj), wb.atomic_numbers)
    whead = BondedParameterHead(16, [1, 7, 8], impropers=True)
    wp = whead(torch.randn(6, 16), None, None, wproj.species_index(wb.atomic_numbers), wtopo)
    assert wp.c_chi.numel() == 0
    assert improper_energy(wb.positions, wtopo, wp) is None


def test_nh3_prior_reproduces_its_fit():
    """Bonded NH3 at the priors: HNH ~106.7 and the umbrella well below the E bend."""
    from rsfff.ff.film.bonded import (
        DEFAULT_BOND_PRIOR, cosine_angle_energy, morse_energy,
    )
    from rsfff.ff.units import BOHR_ANG

    r_eq, d, k = DEFAULT_BOND_PRIOR[(1, 7)]
    theta, k_theta = IMPROPER_ANGLE_PRIOR[7]
    c_chi, k_chi = DEFAULT_IMPROPER_PRIOR[7]
    single = _batch([nh3()], [7, 1, 1, 1], [0.0], [4])
    topo = _topology(single, [1, 7])

    def energy(x):
        r, cos_t = topo.geometry(x)
        t = torch.ones_like
        e = morse_energy(r, t(r) * r_eq, t(r) * d, t(r) * k).sum()
        e = e + cosine_angle_energy(cos_t, t(cos_t) * math.cos(theta), t(cos_t) * k_theta).sum()
        s2 = topo.improper_sin2(x)
        return e + wilson_improper_energy(s2, t(s2) * c_chi, t(s2) * k_chi).sum()

    best = min(
        ((float(energy(nh3(r=r_eq * BOHR_ANG, hnh_deg=a))), a)
         for a in torch.linspace(100.0, 115.0, 301).tolist()),
    )
    assert abs(best[1] - 106.7) < 0.15
    barrier = float(energy(nh3(r=r_eq * BOHR_ANG, hnh_deg=120.0))) - best[0]
    assert 5.0 < barrier * 627.5095 < 7.0          # ~5.9 kcal/mol at fixed bond length


# --- the full model ----------------------------------------------------------------------------

def _model(neighbor_types, ref, seed: int = 0, **film_over):
    torch.manual_seed(seed)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    film = types.SimpleNamespace(
        hidden=32, block_dim=24, head_hidden=24, head_depth=1,
        equiv_channels=6, bonded_hidden=24, nonbonded="exclusions", **film_over,
    )
    model = build_film_model(features, film, list(neighbor_types), torch.tensor(ref))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return model


CASES = {
    "nh3": (nh3_dimer, [1, 7], [-0.5, -54.5], dict(impropers=True)),
    "hf": (hf_dimer, [1, 9], [-0.5, -99.7], {}),
    "nacl": (nacl_dimer, [11, 17], [-161.9, -460.3], {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_forward_and_forces(name):
    make, types_, ref, over = CASES[name]
    model = _model(types_, ref, **over)
    batch = make()
    pos = batch.positions.clone().requires_grad_(True)
    batch.positions = pos
    out = model(batch)
    assert torch.isfinite(out.energy).all()
    (forces,) = torch.autograd.grad(out.energy.sum(), pos, create_graph=True)
    assert torch.isfinite(forces).all()
    # double backward reaches the parameters (force training)
    forces.pow(2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # finite-difference check of one force component
    with torch.no_grad():
        h = 1e-5
        x0 = batch.positions.detach().clone()
        for sign in (1.0, -1.0):
            x = x0.clone()
            x[1, 2] += sign * h
            batch.positions = x
            e = float(model(batch).energy.sum())
            if sign > 0:
                ep = e
            else:
                em = e
    assert abs(float(forces[1, 2]) - (ep - em) / (2 * h)) < 1e-5


@pytest.mark.parametrize("name", list(CASES))
def test_isolated_fragment_has_no_induction(name):
    make, types_, ref, over = CASES[name]
    model = _model(types_, ref, **over)
    batch = make()
    # pull the fragments 60 A apart: past every cutoff
    shift = 60.0 * batch.fragment_idx.to(batch.positions.dtype)
    batch.positions = batch.positions + shift.unsqueeze(1) * torch.tensor([1.0, 0.0, 0.0])
    out = model(batch)
    for key in ("elst", "pauli", "disp", "induction"):
        assert abs(float(out.interaction[key].sum())) < 1e-12, key


def test_monatomic_ion_fragment_energy_is_its_reference():
    """No bonded term can move a lone ion's energy: E_f == E0[Z] exactly. The reference JSON
    for an ionic dataset therefore has to carry the *ion* energies (scripts/ion_references.py)."""
    model = _model([11, 17], [-161.9, -460.3])
    out = model(nacl_pair())
    assert torch.allclose(out.fragment_energy, torch.tensor([-161.9, -460.3]), atol=1e-12)


def test_nacl_charges_are_formal():
    model = _model([11, 17], [-161.9, -460.3])
    out = model(nacl_pair())
    assert torch.allclose(out.charges, torch.tensor([1.0, -1.0]), atol=1e-12)
