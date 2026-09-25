"""Formal charges in closed form: a readout plus a projection onto the frame's total charge.

The pairing model solves the formal electron count of every atom jointly with the bond orders
(:mod:`rsfff.ff.pairing.electronic_state`); that is where hydronium's oxygen gets capacity 3
and hydroxide's gets 1. The explicit model needs the same information without a solve::

    q_i = q~_i + w_i (Q_frame - sum_{k in frame} q~_k) / sum_{k in frame} w_k

with ``q~`` a zero-initialized readout of the topology latent (``TersoffHeads.q_mlp``) and
``w`` the projection weights, ``projection``:

``heavy_atoms``
    one on heavy atoms, zero on hydrogen (uniform in a frame with no heavy atom). H3O+ gives
    ``v_O = 3``, OH- ``v_O = 1``, and a Zundel frame gives *each* oxygen ``2.5`` wherever the
    proton is.

``uniform``
    one everywhere.

``overload`` (default)
    the charge follows the bonding, once: a first pass with the ``heavy_atoms`` weights gives
    bond orders ``p0``, and the weights of the second pass are the atoms' over- (``Q > 0``)
    or under-coordination (``Q < 0``) in that state past half an electron,
    ``w_i = softplus((+-(sum_j p0_ij - v0_i) - 1/2) / width)``, so a Zundel proton hands the
    charge -- and with it the third capacity -- to the oxygen it sits on, and the midpoint
    (both oxygens half an electron over) stays ``0.5 / 0.5``. A neutral frame keeps the
    ``heavy_atoms`` weights (they only carry the readout's own sum). This is ReaxFF's
    over-coordination ``Delta_i`` used as a charge prior; the readout learns what it misses.

At initialization ``q~ = 0`` and the weights alone decide; the readout lets the network move
charge onto or off a hydrogen (a leaving proton) as the data demand.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["PROJECTIONS", "overload_weights", "project_formal_charge"]

PROJECTIONS = ("overload", "heavy_atoms", "uniform")


def _heavy_weights(atomic_numbers, batch_idx, n_systems, dtype):
    w = (atomic_numbers > 1).to(dtype)
    w_sum = w.new_zeros(n_systems).index_add_(0, batch_idx, w)
    no_heavy = (w_sum == 0.0)[batch_idx]
    return torch.where(no_heavy, torch.ones_like(w), w)


def overload_weights(
    p0: torch.Tensor,            # (Pb,) first-pass bond orders
    sub_pairs: torch.Tensor,     # (2, Pb)
    v0: torch.Tensor,            # (N,) neutral capacities
    atomic_numbers: torch.Tensor,
    batch_idx: torch.Tensor,
    n_systems: int,
    total_charge: torch.Tensor,  # (B,)
    width: float = 0.1,
) -> torch.Tensor:
    """Over- / under-coordination weights per atom, by the sign of the frame charge."""
    n_atoms = int(v0.shape[0])
    coord = p0.new_zeros(n_atoms).index_add_(0, sub_pairs[0], p0).index_add_(0, sub_pairs[1], p0)
    sign = torch.sign(total_charge.to(p0.dtype))[batch_idx]
    w = F.softplus((sign * (coord - v0) - 0.5) / width)
    heavy = _heavy_weights(atomic_numbers, batch_idx, n_systems, p0.dtype)
    neutral = (sign == 0.0)
    return torch.where(neutral, heavy, w)


def project_formal_charge(
    q_raw: torch.Tensor,           # (N,)
    atomic_numbers: torch.Tensor,  # (N,)
    batch_idx: torch.Tensor,       # (N,)
    n_systems: int,
    total_charge: torch.Tensor,    # (B,)
    projection: str = "heavy_atoms",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """``q`` (N,) with ``sum_{i in frame} q_i = Q_frame`` exactly. ``weights`` (from
    :func:`overload_weights`) override the ``projection`` rule."""
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")
    if weights is not None:
        w = weights
    elif projection == "uniform":
        w = torch.ones_like(q_raw)
    else:
        w = _heavy_weights(atomic_numbers, batch_idx, n_systems, q_raw.dtype)
    w_sum = w.new_zeros(n_systems).index_add_(0, batch_idx, w).clamp(min=1.0e-12)
    q_sum = q_raw.new_zeros(n_systems).index_add_(0, batch_idx, q_raw)
    shift = (total_charge.to(q_raw.dtype) - q_sum) / w_sum
    return q_raw + w * shift[batch_idx]
