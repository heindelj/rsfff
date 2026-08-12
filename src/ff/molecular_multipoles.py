"""Molecular multipoles of a distributed point-multipole model, per fragment.

The electrostatics term solves for per-atom charges, dipoles and quadrupoles, but
the reference data constrains their *molecular* sums: a Q-Chem job reports one
dipole and one second moment per isolated fragment. This module builds the same
quantities from the model's atomic multipoles so the two can be compared.

Conventions, all measured rather than assumed
---------------------------------------------
**The atomic quadrupole is the Buckingham quadrupole.** Driving
:func:`rsfff.ff.multipole.build_polytensor` with ``q20 = 1`` against a unit point
charge at distance ``r`` gives exactly ``1/r**3``, which fixes

    Theta_ab = 1/2 sum_i q_i (3 x_a x_b - r^2 delta_ab)

with the interaction written ``E = (1/3) Theta : T`` -- that ``1/3`` is
``_QUAD_POLY_WEIGHTS``. A *primitive* (traced) second moment ``M_ab = sum q x_a x_b``
relates to it as ``Theta = (3 M - I tr M) / 2``, hence the ``2/3`` when an atomic
``Theta`` is folded back into a molecular second moment below.

**Q-Chem reports moments about the origin of its own standard nuclear
orientation**, which is the *whole system's* center of nuclear charge. For a
one-fragment job that is the molecule's own center, but for a cluster every
fragment's moments share the cluster's origin, and the ``2 mu . d`` cross term
that introduces reaches ~30 D*Ang for an outer monomer -- an order of magnitude
above the intrinsic signal.

So the prediction is built about the coordinate origin too, and prediction and
target then travel through the *same* :func:`shift_multipoles` and
:func:`buckingham_from_second_moment`. A convention error in the shift cancels
between the two sides instead of biasing the fit; building the prediction about
the fragment center directly would not have that property.

**Only the traceless part is comparable.** The primitive second moment carries
the electron cloud's spatial spread -- about ``-14.05 e*a0^2`` of trace for a
water monomer -- which no arrangement of point multipoles can reproduce. Every
comparison here therefore goes through the Buckingham projection, which removes
the trace structurally rather than down-weighting it.

Units
-----
================  ==============  ==========================================
quantity          unit            note
================  ==============  ==========================================
positions         Angstrom        model convention (rsfff.train.data)
charges           e
mu                e * bohr        as the electrostatics term produces them
quadrupoles       e * bohr^2      Cartesian, traceless (Buckingham)
returned moments  e * bohr^n      matching the parsed Q-Chem labels
================  ==============  ==========================================

The single Angstrom -> bohr conversion happens in :func:`predicted_multipoles`.
"""

from __future__ import annotations

import torch

from .units import BOHR_ANG


def _pool(values: torch.Tensor, group_idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    out = values.new_zeros((n_groups,) + values.shape[1:])
    return out.index_add_(0, group_idx, values)


def center_of_nuclear_charge(
    positions: torch.Tensor,        # (N, 3), any length unit
    atomic_numbers: torch.Tensor,   # (N,)
    group_idx: torch.Tensor,        # (N,)
    n_groups: int,
) -> torch.Tensor:
    """``sum_i Z_i r_i / sum_i Z_i`` per group: ``(G, 3)``, in the input's length unit.

    This is Q-Chem's own origin convention -- its standard nuclear orientation
    places this point at ``(0, 0, 0)``, verified to ``4e-11`` Angstrom on the
    monomer jobs. Using it here is what makes a shifted quadrupole target
    translation-invariant.
    """
    z = atomic_numbers.to(positions.dtype)
    weight = _pool(z, group_idx, n_groups).clamp(min=1e-12)
    return _pool(z.unsqueeze(-1) * positions, group_idx, n_groups) / weight.unsqueeze(-1)


def predicted_multipoles(
    charges: torch.Tensor,                    # (N,)   e
    positions: torch.Tensor,                  # (N, 3) Angstrom
    group_idx: torch.Tensor,                  # (N,)
    n_groups: int,
    dipoles: torch.Tensor | None = None,      # (N, 3) e*bohr
    quadrupoles: torch.Tensor | None = None,  # (N, 3, 3) e*bohr^2, traceless Buckingham
) -> tuple[torch.Tensor, torch.Tensor]:
    """Molecular ``(dipole (G,3), primitive second moment (G,3,3))`` **about the origin**.

    ::

        d_a   = sum_i [ q_i r_a + mu_a ]
        M_ab  = sum_i [ q_i r_a r_b + (mu_a r_b + mu_b r_a) + (2/3) Theta_ab ]

    The dipole term of ``M`` is the second moment of a point dipole,
    ``int rho x_a x_b = mu_a r_b + mu_b r_a``; the ``2/3`` converts the atomic
    Buckingham quadrupole back to the traceless part of a second moment, and
    contributes nothing to the trace.
    """
    r = positions / BOHR_ANG
    d_atom = charges.unsqueeze(-1) * r
    m_atom = charges[:, None, None] * r.unsqueeze(-1) * r.unsqueeze(-2)
    if dipoles is not None:
        d_atom = d_atom + dipoles
        m_atom = m_atom + dipoles.unsqueeze(-1) * r.unsqueeze(-2)
        m_atom = m_atom + r.unsqueeze(-1) * dipoles.unsqueeze(-2)
    if quadrupoles is not None:
        m_atom = m_atom + (2.0 / 3.0) * quadrupoles
    return _pool(d_atom, group_idx, n_groups), _pool(m_atom, group_idx, n_groups)


def shift_multipoles(
    dipole: torch.Tensor,          # (G, 3)     about the old origin
    second_moment: torch.Tensor,   # (G, 3, 3)  about the old origin, primitive
    charge: torch.Tensor,          # (G,)       formal charge of each group
    center: torch.Tensor,          # (G, 3)     the new origin, in bohr
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-reference moments from the coordinate origin to ``center``.

    ::

        d(C)     = d(0) - Q C
        M_ab(C)  = M_ab(0) - C_a d_b(0) - C_b d_a(0) + Q C_a C_b

    Exact, not an approximation: it is just ``int rho (x - C)_a (x - C)_b``
    expanded. For a neutral fragment the dipole is unchanged and only the cross
    term survives -- but that cross term is the dominant one, so it is never
    optional.
    """
    q = charge.unsqueeze(-1)
    shifted_dipole = dipole - q * center
    cross = center.unsqueeze(-1) * dipole.unsqueeze(-2)
    outer = center.unsqueeze(-1) * center.unsqueeze(-2)
    shifted_moment = (
        second_moment - cross - cross.transpose(-1, -2) + charge[:, None, None] * outer
    )
    return shifted_dipole, shifted_moment


def buckingham_from_second_moment(second_moment: torch.Tensor) -> torch.Tensor:
    """``Theta = (3 M - I tr M) / 2``: ``(..., 3, 3)``, symmetric and traceless.

    The projection that discards the electron-cloud spread a point-multipole
    model cannot carry. Idempotent on already-traceless input up to the ``3/2``
    scale, which is why prediction and target must both pass through it exactly
    once.
    """
    trace = second_moment.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(3, dtype=second_moment.dtype, device=second_moment.device)
    return 1.5 * second_moment - 0.5 * trace[..., None, None] * eye


def fragment_multipoles(
    charges: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    group_idx: torch.Tensor,
    n_groups: int,
    group_charge: torch.Tensor,
    dipoles: torch.Tensor | None = None,
    quadrupoles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted ``(dipole, Buckingham quadrupole)`` per group, about its own center.

    The full chain: build about the origin, shift to the group's center of
    nuclear charge, project out the trace. Apply
    :func:`reference_multipoles` to the *labels* so both sides take the same
    path.
    """
    dipole, moment = predicted_multipoles(
        charges, positions, group_idx, n_groups, dipoles, quadrupoles
    )
    center = center_of_nuclear_charge(
        positions / BOHR_ANG, atomic_numbers, group_idx, n_groups
    )
    dipole, moment = shift_multipoles(dipole, moment, group_charge, center)
    return dipole, buckingham_from_second_moment(moment)


def reference_multipoles(
    dipole: torch.Tensor,          # (G, 3)    label, about the coordinate origin
    second_moment: torch.Tensor,   # (G, 3, 3) label, primitive, about the origin
    positions: torch.Tensor,       # (N, 3)    Angstrom
    atomic_numbers: torch.Tensor,
    group_idx: torch.Tensor,
    n_groups: int,
    group_charge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same chain applied to parsed Q-Chem labels: ``(dipole, Buckingham Theta)``."""
    center = center_of_nuclear_charge(
        positions / BOHR_ANG, atomic_numbers, group_idx, n_groups
    )
    dipole, moment = shift_multipoles(dipole, second_moment, group_charge, center)
    return dipole, buckingham_from_second_moment(moment)


__all__ = [
    "buckingham_from_second_moment",
    "center_of_nuclear_charge",
    "fragment_multipoles",
    "predicted_multipoles",
    "reference_multipoles",
    "shift_multipoles",
]
