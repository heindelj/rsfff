"""Molecular multipoles from atomic ones: conventions, invariance, and the shift.

The two errors these guard against are both silent. A wrong Buckingham factor is
absorbed by the fit as a scale on the quadrupole head; a missing origin shift
biases every cluster fragment by a term that grows with its distance from the
cluster center. Neither shows up as a crash or an obviously bad loss curve.
"""

import numpy as np
import pytest
import torch

from rsfff.ff.molecular_multipoles import (
    buckingham_from_second_moment,
    center_of_nuclear_charge,
    fragment_multipoles,
    predicted_multipoles,
    reference_multipoles,
    shift_multipoles,
)
from rsfff.ff.multipole import (
    build_polytensor,
    damped_interaction_tensor,
    multipole_pair_energy,
    spherical_to_cartesian_quadrupole,
)
from rsfff.ff.units import BOHR_ANG

torch.set_default_dtype(torch.float64)


def _water(offset=(0.0, 0.0, 0.0)):
    """One water in Angstrom, plus its atomic numbers."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
    ) + torch.tensor(offset)
    return pos, torch.tensor([8, 1, 1])


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_spherical_quadrupole_is_the_buckingham_quadrupole():
    """``q20 = 1`` against a unit charge at ``r`` gives exactly ``1/r**3``.

    A traceless *second moment* would give ``(3/2) M_zz / r**3`` instead, so this
    single number pins which of the two ``build_polytensor`` speaks -- and hence
    the ``2/3`` in :func:`predicted_multipoles`.
    """
    r = 3.0
    q_s = torch.zeros(1, 5)
    q_s[0, 0] = 1.0
    quad = spherical_to_cartesian_quadrupole(q_s)
    assert quad[0].diagonal().tolist() == pytest.approx([-0.5, -0.5, 1.0])

    dr = torch.tensor([[0.0, 0.0, r]])
    tensor = damped_interaction_tensor(dr, None, 1.0 / torch.tensor([r]), max_rank=2)
    energy = multipole_pair_energy(
        build_polytensor(torch.zeros(1), None, quad, max_rank=2),
        build_polytensor(torch.ones(1), None, None, max_rank=2),
        tensor,
    )
    assert energy.item() == pytest.approx(1.0 / r**3, rel=1e-12)


def test_buckingham_projection_is_traceless_and_scales_by_three_halves():
    m = torch.randn(4, 3, 3)
    m = m + m.transpose(-1, -2)
    theta = buckingham_from_second_moment(m)
    assert theta.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-12
    assert torch.allclose(theta, theta.transpose(-1, -2))
    # On already-traceless input it is a pure 3/2 scale.
    traceless = m - torch.eye(3) * m.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3
    assert torch.allclose(buckingham_from_second_moment(traceless), 1.5 * traceless)


def test_point_charge_pair_reproduces_the_analytic_quadrupole():
    """Charges at ``+-d z-hat``, worked out by hand, no model involved.

    Like-signed: ``M_zz = 2 q d^2``, ``tr M = 2 q d^2``, so ``Theta_zz = 2 q d^2``
    and ``Theta_xx = Theta_yy = -q d^2``. Opposite-signed: the second moment
    cancels exactly and only a dipole survives.
    """
    d_ang = 0.5
    d = d_ang / BOHR_ANG
    positions = torch.tensor([[0.0, 0.0, d_ang], [0.0, 0.0, -d_ang]])
    group = torch.zeros(2, dtype=torch.long)

    _, moment = predicted_multipoles(torch.tensor([1.0, 1.0]), positions, group, 1)
    theta = buckingham_from_second_moment(moment)
    assert theta[0, 2, 2].item() == pytest.approx(2 * d**2, rel=1e-12)
    assert theta[0, 0, 0].item() == pytest.approx(-(d**2), rel=1e-12)

    dipole, moment = predicted_multipoles(
        torch.tensor([1.0, -1.0]), positions, group, 1
    )
    assert torch.allclose(buckingham_from_second_moment(moment), torch.zeros(3, 3), atol=1e-12)
    assert dipole[0, 2].item() == pytest.approx(2 * d, rel=1e-12)


def test_atomic_quadrupoles_pass_through_unchanged():
    """A traceless atomic Theta at the origin is the molecular Theta."""
    positions = torch.zeros(1, 3)
    charges = torch.zeros(1)
    q_s = torch.tensor([[0.3, -0.1, 0.2, 0.4, -0.25]])
    quad = spherical_to_cartesian_quadrupole(q_s)
    _, moment = predicted_multipoles(
        charges, positions, torch.zeros(1, dtype=torch.long), 1, None, quad
    )
    assert torch.allclose(buckingham_from_second_moment(moment), quad, atol=1e-12)


def test_fragment_dipole_matches_a_manual_sum():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = torch.tensor([-0.5, 0.5])
    mu = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    dipole, _ = predicted_multipoles(
        charges, positions, torch.zeros(2, dtype=torch.long), 1, mu
    )
    expected = (charges[:, None] * positions / BOHR_ANG).sum(0) + mu.sum(0)
    assert torch.allclose(dipole[0], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The origin shift
# ---------------------------------------------------------------------------


def test_center_of_nuclear_charge_is_qchems_origin():
    pos, z = _water()
    center = center_of_nuclear_charge(pos, z, torch.zeros(3, dtype=torch.long), 1)
    manual = (z[:, None].double() * pos).sum(0) / z.sum()
    assert torch.allclose(center[0], manual, atol=1e-14)


def test_shift_is_exact_against_an_explicit_recomputation():
    """The shift formula must equal rebuilding the moments about the new origin."""
    torch.manual_seed(0)
    positions = torch.randn(5, 3)
    charges = torch.randn(5)
    group = torch.zeros(5, dtype=torch.long)
    total = charges.sum().reshape(1)

    dipole, moment = predicted_multipoles(charges, positions, group, 1)
    center = torch.randn(1, 3)
    shifted_d, shifted_m = shift_multipoles(dipole, moment, total, center)

    # Rebuild directly about `center` (in bohr).
    moved = positions - center * BOHR_ANG
    direct_d, direct_m = predicted_multipoles(charges, moved, group, 1)
    assert torch.allclose(shifted_d, direct_d, atol=1e-12)
    assert torch.allclose(shifted_m, direct_m, atol=1e-12)


def test_quadrupole_target_is_translation_invariant_only_after_the_shift():
    """The test that fails loudly if the shift is dropped."""
    pos, z = _water()
    group = torch.zeros(3, dtype=torch.long)
    charges = torch.tensor([-0.7, 0.35, 0.35])
    offset = torch.tensor([2.5, -1.0, 4.0])

    def moments(p, shifted: bool):
        dipole, moment = predicted_multipoles(charges, p, group, 1)
        if shifted:
            center = center_of_nuclear_charge(p / BOHR_ANG, z, group, 1)
            dipole, moment = shift_multipoles(dipole, moment, torch.zeros(1), center)
        return dipole, buckingham_from_second_moment(moment)

    d0, t0 = moments(pos, True)
    d1, t1 = moments(pos + offset, True)
    assert torch.allclose(d0, d1, atol=1e-11)
    assert torch.allclose(t0, t1, atol=1e-11)

    _, raw0 = moments(pos, False)
    _, raw1 = moments(pos + offset, False)
    assert (raw0 - raw1).abs().max() > 1.0


def test_rotation_equivariance():
    pos, z = _water()
    group = torch.zeros(3, dtype=torch.long)
    charges = torch.tensor([-0.7, 0.35, 0.35])
    mu = torch.tensor([[0.0, 0.0, 0.2], [0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    quad = spherical_to_cartesian_quadrupole(torch.randn(3, 5) * 0.1)

    angle = torch.tensor(0.7)
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    d0, t0 = fragment_multipoles(charges, pos, z, group, 1, torch.zeros(1), mu, quad)
    d1, t1 = fragment_multipoles(
        charges,
        pos @ rot.T,
        z,
        group,
        1,
        torch.zeros(1),
        mu @ rot.T,
        rot @ quad @ rot.T,
    )
    assert torch.allclose(d1, d0 @ rot.T, atol=1e-12)
    assert torch.allclose(t1, rot @ t0 @ rot.T, atol=1e-12)


def test_groups_are_independent():
    """Two fragments give the moments they would give alone."""
    pos_a, z_a = _water()
    pos_b, z_b = _water(offset=(6.0, 0.0, 0.0))
    charges = torch.tensor([-0.7, 0.35, 0.35])

    both = fragment_multipoles(
        charges.repeat(2),
        torch.cat((pos_a, pos_b)),
        torch.cat((z_a, z_b)),
        torch.tensor([0, 0, 0, 1, 1, 1]),
        2,
        torch.zeros(2),
    )
    alone = fragment_multipoles(
        charges, pos_b, z_b, torch.zeros(3, dtype=torch.long), 1, torch.zeros(1)
    )
    assert torch.allclose(both[0][1], alone[0][0], atol=1e-12)
    assert torch.allclose(both[1][1], alone[1][0], atol=1e-12)


# ---------------------------------------------------------------------------
# Against the real labels
# ---------------------------------------------------------------------------


def test_reference_labels_land_on_a_physical_water_quadrupole(tmp_path):
    """The parsed w2 labels, put through the same chain, give real monomer values."""
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "wb97mv_tzvpd",
        "w2_wb97mv_tzvpd.xyz",
    )
    if not os.path.exists(path):
        pytest.skip("run scripts/parse_roundtrip.py first")
    from rsfff.train.data import load_extxyz

    batch = load_extxyz(path, dtype=torch.float64).flat_batch(range(20))
    dipole, theta = reference_multipoles(
        batch.fragment_dipole,
        batch.fragment_second_moment,
        batch.positions,
        batch.atomic_numbers,
        batch.fragment_idx,
        batch.n_fragments,
        batch.fragment_charge,
    )
    # A frozen water monomer: |mu| ~ 0.72-0.80 e*a0 (1.85-2.05 D) and a
    # Buckingham quadrupole with components of order 1 e*a0^2.
    norms = dipole.norm(dim=-1)
    assert float(norms.min()) > 0.6 and float(norms.max()) < 0.95
    assert theta.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-10
    assert 0.5 < float(theta.abs().max()) < 4.0

    # Neutral fragments: the dipole label is origin-independent, so the shift
    # must have left it alone.
    assert torch.allclose(dipole, batch.fragment_dipole, atol=1e-12)
