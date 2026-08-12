"""Closed-form EEM with dipole response (no interactions, no iterative solver).

Per molecule m (atoms i, total charge Q_m, optional uniform field F_m):

    E_EEM(q, mu) = sum_i [ chi_i q_i + 1/2 eta_i q_i^2 - q_i (r_i . F) ]
                 + sum_i [ chivec_i . mu_i + 1/2 mu_i^T alpha_i^{-1} mu_i - mu_i . F ]
    s.t.           sum_{i in m} q_i = Q_m

With no interaction blocks the minimization is separable and analytic:

    charges:  chi_i + eta_i q_i - r_i.F + lambda_m = 0   (Lagrange multiplier)
              q_i      = (r_i.F - chi_i - lambda_m) / eta_i
              lambda_m = ( sum_i (r_i.F - chi_i)/eta_i - Q_m ) / sum_i (1/eta_i)
    dipoles:  mu_i = alpha_i (F - chivec_i)
              minimized dipole energy = -1/2 (F - chivec_i)^T alpha_i (F - chivec_i)
              (alpha is never inverted)

chi/eta/chivec/alpha are *effective short-range parameters* -- the interaction
blocks of a full charge-equilibration model are folded into their environment
dependence (they come from heads on the SOAP features). Everything here is a
smooth closed-form function of the parameters and positions, so forces, dipole
derivatives, and training gradients are single ordinary autograd passes: with
eta > 0 and alpha PSD the energy is an exact strict minimum -- stable by
construction.

The molecular polarizability is analytic as well:

    alpha_mol = sum_i alpha_i + sum_i (r_i - Rbar)(r_i - Rbar)^T / eta_i,
    Rbar      = (sum_i r_i/eta_i) / (sum_i 1/eta_i)

(the second term is the charge-flow polarizability from dq/dF; a weighted
positional covariance -- symmetric, PSD, and translation-invariant).

Shapes: flat ragged batch, N atoms in M molecules, ``batch_idx`` (N,) labeling.
"""

from __future__ import annotations

import torch


def _pool(x: torch.Tensor, batch_idx: torch.Tensor, n_systems: int) -> torch.Tensor:
    """Per-molecule sum of a per-atom quantity: (N, ...) -> (M, ...)."""
    out = x.new_zeros((n_systems,) + x.shape[1:])
    return out.index_add_(0, batch_idx, x)


def eem_charges(
    chi: torch.Tensor,            # (N,) scalar electronegativity
    eta: torch.Tensor,            # (N,) hardness, strictly positive
    positions: torch.Tensor,      # (N, 3)
    batch_idx: torch.Tensor,      # (N,)
    n_systems: int,
    total_charge: torch.Tensor,   # (M,)
    field: torch.Tensor | None = None,  # (M, 3) uniform field per molecule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form constrained charge minimization. Returns ``(q (N,), lambda (M,))``.

    ``sum_{i in m} q_i = Q_m`` holds to machine precision by construction: the
    Lagrange multiplier is eliminated analytically, not iterated.
    """
    if field is None:
        b = -chi                                              # (N,) r.F - chi at F=0
    else:
        b = (positions * field[batch_idx]).sum(dim=-1) - chi  # (N,) r_i.F - chi_i
    inv_eta = 1.0 / eta
    s = _pool(inv_eta, batch_idx, n_systems)                  # (M,) sum 1/eta
    b_over_eta = _pool(b * inv_eta, batch_idx, n_systems)     # (M,) sum b/eta
    lam = (b_over_eta - total_charge) / s                     # (M,)
    q = (b - lam[batch_idx]) * inv_eta                        # (N,)
    return q, lam


def atomic_dipoles(
    chivec: torch.Tensor,         # (N, 3) vector electronegativity
    alpha: torch.Tensor,          # (N, 3, 3) atomic polarizability (PSD)
    batch_idx: torch.Tensor,
    field: torch.Tensor | None = None,  # (M, 3)
) -> torch.Tensor:
    """Minimizing atomic dipoles ``mu_i = alpha_i (F - chivec_i)``: (N, 3).

    At zero field ``mu_i = -alpha_i chivec_i`` -- the vector electronegativity is
    what makes atomic dipoles (and hence molecular multipoles) nonzero without an
    external field.
    """
    drive = -chivec if field is None else field[batch_idx] - chivec
    return torch.einsum("nab,nb->na", alpha, drive)


def atomic_quadrupoles(
    chiquad: torch.Tensor,      # (N, 5) spherical l=2 "quadrupole electronegativity"
    cquad: torch.Tensor,        # (N,) isotropic or (N, 5, 5) quadrupole polarizability
    batch_idx: torch.Tensor,
    field_gradient: torch.Tensor | None = None,   # (M, 5) spherical, per system
) -> torch.Tensor:
    """``Theta_i = C_i (gradF - chiquad_i)``; at zero field gradient, ``-C_i chiquad_i``.

    The rank-2 mirror of :func:`atomic_dipoles`. ``C`` is accepted either as a scalar per
    atom -- an isotropic ``c I_5``, which is manifestly PSD and manifestly equivariant
    because a multiple of the identity commutes with every Wigner-D -- or as a full
    ``(N, 5, 5)``, so an anisotropic ``0 + 2`` construction can be dropped in later without
    touching any caller.

    ``field_gradient`` is the conjugate of a quadrupole the way a uniform field is the
    conjugate of a dipole. It is the slot polarization will fill; nothing supplies it yet.

    **What this does and does not buy at zero field.** ``Theta = -C chiquad`` with a free
    equivariant ``chiquad`` and invertible ``C`` is a bijection onto all 5-vectors, so the
    reachable permanent quadrupoles are exactly those of an unconstrained head -- for *any*
    ``C``. What is gained is that the sector now has an energy
    (:func:`rsfff.mlip.sqe.atomic_quadrupole_energy`), which makes ``C`` a physical quantity
    a 1-body energy target can identify, bounds ``||Theta|| <= ||C|| ||chiquad||``, and gives
    the field gradient somewhere to enter. Until that energy is in a loss, ``(C, chiquad)``
    carries an exact gauge -- ``(lambda C, chiquad / lambda)`` leaves ``Theta`` unchanged --
    so those parameters are not identified and want weight decay.
    """
    drive = -chiquad if field_gradient is None else field_gradient[batch_idx] - chiquad
    if cquad.dim() == 1:
        return cquad.unsqueeze(-1) * drive
    return torch.einsum("nab,nb->na", cquad, drive)


def eem_energy(
    q: torch.Tensor,              # (N,) from eem_charges
    chi: torch.Tensor,
    eta: torch.Tensor,
    chivec: torch.Tensor,
    alpha: torch.Tensor,          # (N, 3, 3)
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    n_systems: int,
    field: torch.Tensor | None = None,
) -> torch.Tensor:
    """E_EEM per molecule at the closed-form minimum: (M,).

    The charge part is evaluated explicitly at q*; the dipole part uses the
    minimized quadratic form ``-1/2 drive^T alpha drive`` (drive = F - chivec),
    avoiding any inversion of alpha.
    """
    e_atom = chi * q + 0.5 * eta * q * q                      # (N,)
    if field is not None:
        e_atom = e_atom - q * (positions * field[batch_idx]).sum(dim=-1)
    drive = -chivec if field is None else field[batch_idx] - chivec
    e_atom = e_atom - 0.5 * torch.einsum("na,nab,nb->n", drive, alpha, drive)
    return _pool(e_atom, batch_idx, n_systems)


def charge_flow_polarizability(
    eta: torch.Tensor,            # (N,)
    positions: torch.Tensor,      # (N, 3)
    batch_idx: torch.Tensor,
    n_systems: int,
) -> torch.Tensor:
    """Charge-flow contribution to the molecular polarizability: (M, 3, 3).

    ``sum_i (r_i - Rbar)(r_i - Rbar)^T / eta_i`` with the hardness-weighted
    centroid ``Rbar = (sum r/eta)/(sum 1/eta)``: symmetric, PSD, and
    translation-invariant. Single atoms contribute exactly zero (no charge flow).
    """
    inv_eta = 1.0 / eta
    s = _pool(inv_eta, batch_idx, n_systems)                       # (M,)
    rbar = _pool(positions * inv_eta.unsqueeze(-1), batch_idx, n_systems) / s.unsqueeze(-1)
    d = positions - rbar[batch_idx]                                # (N, 3)
    outer = inv_eta.view(-1, 1, 1) * d.unsqueeze(-1) * d.unsqueeze(-2)  # (N, 3, 3)
    return _pool(outer, batch_idx, n_systems)
