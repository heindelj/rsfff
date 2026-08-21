"""Rotation invariants of the electronic state, contracted against the geometric features.

``M = (q, mu, Theta)`` and ``Phi = (V, E, grad E)`` are not scalars, so a per-atom energy that
depends on them cannot simply concatenate them onto ``inv_feats``. This module is the reduction
that makes them admissible, split out of :class:`rsfff.ff.atomic_energy.AtomicStateEnergy` so
that head and :class:`rsfff.ff.bond_energy.FragmentBondEnergy` share one implementation rather
than two copies that drift.

There are two kinds of invariant, and the second is the reason this is more than a function of
scalars:

* **self-contractions** of the state -- ``mu.mu``, ``mu.E``, ``Theta:Theta``, ``Theta:gradE``
  and so on. These see the *magnitude* of the state but nothing about how it is oriented in the
  molecule.
* **contractions against the geometric features**, ``mu . v_k`` and ``Theta : G_k``, where
  ``v_k`` and ``G_k`` are learned channel reductions of the lambda=1 and lambda=2 features.
  These see which way the dipole points *relative to the local geometry*, which for water is
  most of the physics.

The lambda=2 contractions are done in **Cartesian** form via
:func:`~rsfff.ff.multipole.spherical_to_cartesian_quadrupole`, not by dotting the five spherical
components together. ``A_ab B_ab`` is manifestly invariant whatever normalization the
five-component convention uses, whereas a naive 5-vector dot product is invariant only if that
basis happens to be orthonormal -- exactly the class of silent equivariance break that
:func:`~rsfff.ff.multipole.irrep2_to_spherical` documents.

Zero-fill rather than a variable width
--------------------------------------
Slots for absent quantities are filled with zeros instead of being dropped, so the width does
not depend on ``max_rank`` or on which response channels are switched on, and a checkpoint
written by one configuration still loads into another. Same convention, and same reason, as
:func:`rsfff.ff.environment.environment_pair_invariants`.
"""

from __future__ import annotations

import torch

from .multipole import spherical_to_cartesian_quadrupole

__all__ = ["n_state_invariants", "state_invariants"]


def n_state_invariants(equiv_channels: int) -> int:
    """Width of :func:`state_invariants`: ``8 + 4K``, whatever is actually live.

    2 rank-0 (``q``, ``V``) + 3 rank-1 self + 3 rank-2 self + 2K rank-1 cross + 2K rank-2 cross.
    """
    return 8 + 4 * int(equiv_channels)


def state_invariants(
    *,
    equiv_channels: int,
    vec_feats: torch.Tensor | None,       # (N, 3, p1)
    equiv_feats: torch.Tensor | None,     # (N, 5, p2)
    vec_reduce: torch.Tensor | None,      # (p1, K)
    equiv_reduce: torch.Tensor | None,    # (p2, K)
    to_spherical: torch.Tensor | None,    # (5, 5)
    q: torch.Tensor,                      # (N,)
    mu: torch.Tensor | None,              # (N, 3)
    quad_s: torch.Tensor | None,          # (N, 5) spherical
    potential: torch.Tensor | None,       # (N,)
    field: torch.Tensor | None,           # (N, 3)
    field_gradient: torch.Tensor | None,  # (N, 5) spherical
) -> torch.Tensor:
    """``(N, n_state_invariants(equiv_channels))``.

    Split out of the head so tests can check rotation invariance without the MLP in the way.
    ``vec_reduce`` / ``equiv_reduce`` are passed already assembled -- under the two-slot
    parameterization they are :func:`rsfff.mlip.heads.slot_reduce` of a fragment half and an
    environment half, and this function neither knows nor needs to know which it was handed.
    """
    n = q.shape[0]
    zero = q.new_zeros(n)
    k = int(equiv_channels)

    cols = [q, zero if potential is None else potential]

    # -- rank 1 --------------------------------------------------------------------------
    d = q.new_zeros(n, 3) if mu is None else mu
    f = q.new_zeros(n, 3) if field is None else field
    cols += [(d * d).sum(-1), (f * f).sum(-1), (d * f).sum(-1)]

    # -- rank 2, contracted in Cartesian form ---------------------------------------------
    t = q.new_zeros(n, 3, 3) if quad_s is None else spherical_to_cartesian_quadrupole(quad_s)
    g = (
        q.new_zeros(n, 3, 3) if field_gradient is None
        else spherical_to_cartesian_quadrupole(field_gradient)
    )
    cols += [
        (t * t).sum((-2, -1)),
        (g * g).sum((-2, -1)),
        (t * g).sum((-2, -1)),
    ]

    # -- rank 1 against the geometric features ---------------------------------------------
    if vec_reduce is None or vec_feats is None:
        cols += [q.new_zeros(n, 2 * k)]
    else:
        v_k = torch.einsum("nmp,pk->nmk", vec_feats, vec_reduce)          # (N, 3, K)
        cols += [
            torch.einsum("nm,nmk->nk", d, v_k),
            torch.einsum("nm,nmk->nk", f, v_k),
        ]

    # -- rank 2 against the geometric features ---------------------------------------------
    if equiv_reduce is None or equiv_feats is None:
        cols += [q.new_zeros(n, 2 * k)]
    else:
        e_k = torch.einsum("nmp,pk->nmk", equiv_feats, equiv_reduce)      # (N, 5, K)
        # (N, K, 5) in the backend basis -> spherical -> Cartesian, so the contraction below
        # is a genuine tensor double-dot rather than a basis-dependent dot product.
        g_k = spherical_to_cartesian_quadrupole(
            e_k.transpose(1, 2) @ to_spherical
        )                                                                 # (N, K, 3, 3)
        cols += [
            torch.einsum("nab,nkab->nk", t, g_k),
            torch.einsum("nab,nkab->nk", g, g_k),
        ]

    return torch.cat([c.reshape(n, -1) for c in cols], dim=-1)
