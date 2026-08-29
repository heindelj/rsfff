"""Phase-6/7 tests: the assembled FilmModel.

The headline invariants:

* **vertex** -- a lone fragment has every env shift exactly zero and ``E_ind == 0.0`` (not
  small: the chi-trick makes the coupled solve's right-hand side an empty sum);
* **isolation** -- a spectator beyond every cutoff changes ``fragment_energy`` not at all;
* **invariance** -- rotation/translation of the energy, permutation of identical atoms and
  of identical waters;
* **forces** -- autograd against central finite differences, through the generated
  parameters, the charge projection and the coupled solve's adjoint;
* **accounting** -- the total is the sum of its parts; induction is variational at init.
"""

from __future__ import annotations

import types

import torch
from e3nn import o3

from rsfff.train.build_film import build_film_model
from rsfff.train.data import Batch

from film_helpers import water_cluster_batch


def small_model(seed: int = 0, *, randomize: bool = False, **film_over):
    torch.manual_seed(seed)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    film = types.SimpleNamespace(
        hidden=32, block_dim=24, head_hidden=24, head_depth=1,
        equiv_channels=6, bonded_hidden=24, **film_over,
    )
    model = build_film_model(features, film, [1, 8], torch.tensor([-0.5013, -75.0093]))
    if randomize:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def test_vertex_monomer():
    """Lone monomer: env exactly zero everywhere, induction exactly 0.0."""
    model = small_model(randomize=True)
    out = model(water_cluster_batch(1))
    assert out.interaction["induction"].item() == 0.0
    assert torch.count_nonzero(out.env_norm) == 0
    assert torch.count_nonzero(out.a_env) == 0
    for name, shift in out.env_shift.items():
        assert torch.count_nonzero(shift) == 0, name
    assert torch.allclose(out.energy, out.fragment_energy, atol=0, rtol=0)


def test_spectator_isolation():
    """A second water at 50 Angstrom changes fragment 0's energy not at all."""
    model = small_model(randomize=True)
    alone = water_cluster_batch(1)
    far = water_cluster_batch(2)
    far.positions = torch.cat(
        (alone.positions, alone.positions + torch.tensor([50.0, 0.0, 0.0]))
    )
    out_alone = model(alone)
    out_far = model(far)
    assert torch.allclose(
        out_far.fragment_energy[0], out_alone.fragment_energy[0], atol=1e-14, rtol=0
    )
    assert abs(out_far.interaction["induction"].item()) < 1e-14


def test_env_shift_zero_at_init_live_after():
    """Fresh model: theta == theta_0 even on a dimer (zero-init env sector). Then it moves."""
    batch = water_cluster_batch(2)
    fresh = small_model()
    out = fresh(batch)
    for name, shift in out.env_shift.items():
        assert torch.count_nonzero(shift) == 0, name

    live = small_model(randomize=True)
    out2 = live(batch)
    assert any(bool(s.abs().max() > 0) for s in out2.env_shift.values())


def test_rotation_translation_invariance():
    model = small_model(randomize=True)
    batch = water_cluster_batch(3)
    e0 = model(batch).energy

    R = o3.rand_matrix().to(batch.positions.dtype)
    shift = torch.tensor([1.7, -0.3, 2.4])
    moved = Batch(**{**batch.__dict__, "positions": batch.positions @ R.T + shift})
    e1 = model(moved).energy
    assert torch.allclose(e0, e1, atol=1e-10)


def test_force_equivariance():
    model = small_model(randomize=True)
    batch = water_cluster_batch(2)

    def forces(b):
        pos = b.positions.clone().requires_grad_(True)
        out = model(Batch(**{**b.__dict__, "positions": pos}))
        return -torch.autograd.grad(out.energy.sum(), pos)[0]

    f0 = forces(batch)
    R = o3.rand_matrix().to(batch.positions.dtype)
    rotated = Batch(**{**batch.__dict__, "positions": batch.positions @ R.T})
    f1 = forces(rotated)
    assert torch.allclose(f1, f0 @ R.T, atol=1e-9)


def test_identical_atom_and_water_permutation():
    model = small_model(randomize=True)
    batch = water_cluster_batch(2)
    e0 = model(batch).energy

    # swap the two H of the first water
    perm_h = torch.tensor([0, 2, 1, 3, 4, 5])
    swapped = Batch(**{**batch.__dict__, "positions": batch.positions[perm_h]})
    assert torch.allclose(model(swapped).energy, e0, atol=1e-12)

    # swap the two whole waters (atom blocks and fragment labels move together, so
    # fragment_idx stays sorted and every label follows its atoms)
    perm_w = torch.tensor([3, 4, 5, 0, 1, 2])
    swapped_w = Batch(**{**batch.__dict__, "positions": batch.positions[perm_w]})
    assert torch.allclose(model(swapped_w).energy, e0, atol=1e-12)


def _fd_check(model, batch, *, coords, eps=1e-5, tol=1e-6):
    pos = batch.positions.clone().requires_grad_(True)
    out = model(Batch(**{**batch.__dict__, "positions": pos}))
    grad = torch.autograd.grad(out.energy.sum(), pos)[0]

    for (a, x) in coords:
        plus = batch.positions.clone()
        plus[a, x] += eps
        minus = batch.positions.clone()
        minus[a, x] -= eps
        e_plus = model(Batch(**{**batch.__dict__, "positions": plus})).energy.sum()
        e_minus = model(Batch(**{**batch.__dict__, "positions": minus})).energy.sum()
        fd = (e_plus - e_minus) / (2 * eps)
        assert torch.allclose(grad[a, x], fd, atol=tol), (
            f"dE/dR[{a},{x}]: autograd {grad[a, x].item():.10f} vs FD {fd.item():.10f}"
        )


def test_finite_difference_forces_no_induction():
    model = small_model(randomize=True)
    model.induction = False
    batch = water_cluster_batch(2)
    _fd_check(model, batch, coords=[(0, 0), (1, 2), (3, 1), (5, 0)])


def test_finite_difference_forces_with_induction():
    """Through the coupled solve: the adjoint must carry every parameter-response term."""
    model = small_model(randomize=True, cg_rtol=1e-13, cg_atol=1e-15, cg_maxiter=400)
    batch = water_cluster_batch(2)
    _fd_check(model, batch, coords=[(0, 0), (2, 1), (4, 2)], tol=1e-6)


def test_accounting():
    model = small_model(randomize=True)
    batch = water_cluster_batch(3)
    out = model(batch)

    total = out.fragment_energy.new_zeros(1).index_add_(
        0, batch.fragment_to_batch.new_zeros(3), out.fragment_energy
    )
    for v in out.interaction.values():
        total = total + v
    assert torch.allclose(total, out.energy, atol=1e-12)

    assert torch.allclose(
        out.fragment_energy,
        out.energy_ref + out.energy_bonded + out.energy_intra,
        atol=1e-14,
    )
    # p_intra equals the boolean routing at a one-hot C
    assert torch.equal(out.p_intra > 0.5, out.is_intra)


def test_induction_variational_at_init():
    """theta == theta_0 at a fresh init, so E_ind is a pure relaxation: <= 0."""
    model = small_model()
    out = model(water_cluster_batch(2))
    assert out.interaction["induction"].item() <= 1e-12
    assert bool(out.solver["ind"][1].all())      # converged


def test_lambda_relabel_sweep():
    """C(lambda) between the two relabelings of a dimer: exact endpoints, smooth middle."""
    from rsfff.ff.film import StateDescriptor

    model = small_model(randomize=True)
    batch = water_cluster_batch(2)
    species_idx = model.projector.species_index(batch.atomic_numbers)
    a = StateDescriptor.from_batch(batch, species_idx, 2)
    b = a.permute_fragments(torch.tensor([1, 0]))

    e0 = model(batch).energy
    e_a = model(batch, StateDescriptor.blend(a, b, 0.0)).energy
    e_b = model(batch, StateDescriptor.blend(a, b, 1.0)).energy
    assert torch.equal(e_a, e_b)          # pure relabeling: identical physics
    assert torch.allclose(e_a, e0, atol=1e-12)

    # smoothness: nearby lambdas give nearby energies
    e1 = model(batch, StateDescriptor.blend(a, b, 0.50)).energy
    e2 = model(batch, StateDescriptor.blend(a, b, 0.51)).energy
    assert (e2 - e1).abs().item() < 0.05 * (e1 - e_a).abs().item() + 1e-9


def test_dissociation_limit():
    """Stretch both O-H bonds far: fragment_energy -> sum of atomic references."""
    model = small_model()   # fresh: parameters at the pyCMM priors
    batch = water_cluster_batch(1)
    batch.positions = torch.tensor(
        [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]]
    )
    out = model(batch, with_induction=False)
    e_ref = out.energy_ref
    # Morse -> 0, angle -> finite but tiny k*(cos - cos_eq)^2/2 bounded by k_theta
    assert (out.fragment_energy - e_ref).abs().item() < 0.2
    # and the bonded well is fully released relative to equilibrium
    eq = model(water_cluster_batch(1), with_induction=False)
    assert eq.fragment_energy.item() < out.fragment_energy.item() - 0.3
