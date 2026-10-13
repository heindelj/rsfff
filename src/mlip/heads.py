"""Per-atom prediction heads for the MLIP, and rsfff's names for bb's head utilities.

The building blocks every head here is made of -- ``mlp``, the two-slot first layer
(``TwoSlotLinear``, ``two_slot_mlp``, ``slot_reduce``, ``env_reduce_parameter``), the
environment-slot tags (``mark_env_slot``, ``env_parameters``, ``core_parameters``) and the
weight-decay exemptions (``exempt_parameter``, ``exempt_from_weight_decay``,
``zero_init_readout``) -- live in :mod:`bb.nn` and are re-exported under their old names, so
``from rsfff.mlip.heads import env_parameters`` keeps working.

**Why a zero-initialized readout is also exempt from weight decay** (the rsfff measurements
behind :func:`bb.nn.zero_init_readout`). With the readout at zero every layer behind it has an
exactly zero gradient on the first step; weight decay is then the only force on those layers
and wins uncontested. Measured across a 24-epoch staged fit (first-layer weight norm, init ->
frozen -> pol -> ct), the blocks that lost the race and what each silently stopped doing::

    cquad_axis_head.axis.gate_mlp  4.64 -> 2e-05 -> 3e-10 -> 4e-22   quadrupole anisotropy
    disp_params.c6_mlp             4.60 -> 3e-04 -> 3e-09 -> 7e-18   environment-quenched C6
    compliance_head.net            4.63 -> 0.16  -> 8e-05 -> 7e-05   per-channel compliance
    pauli_params.q_mlp             4.58 -> 0.59  -> 0.031 -> 1e-04   Pauli monopole
    environment.inv_mlp            4.62 -> 6e-07 -> 1e-13 -> 4e-20   all many-body content

A zero-init gate also deadlocks the channel reduction it multiplies (``equiv_reduce``, now
``reduce.<irrep>`` in the bb readouts): each is the other's only gradient path, so the whole
equivariant head is exempt, not just its gate. Measured (``equiv_reduce`` norm, init ~5.66 ->
frozen -> pol -> ct)::

    cquad_axis_head.axis.equiv_reduce  0.0    0.0    0.0     deadlocked before epoch 1
    response.alpha_head.equiv_reduce   0.712  0.970  0.176   polarizability anisotropy
    pauli dipole_head.equiv_reduce     0.719  0.840  0.286   Pauli dipole anisotropy
    pauli quadrupole_head.equiv_reduce 1.074  0.997  0.475
    response.chivec_head.equiv_reduce  1.373  1.250  0.830
    response.chiquad_head.equiv_reduce 1.646  1.737  1.061

What replaces weight decay is the penalty that names the quantity it wants small
(``env_weight`` on ``||h_env - h_frag||``, ``r0_spread_weight``), not the raw weights of a
block meant to start at zero.

``two_slot_mlp(p, 0, ...)`` is exactly ``mlp(p, ...)`` (same modules and state-dict keys):
``rsfff.ff.v1`` builds its heads from these, so ``checkpoints/water_staged/best.pt`` keeps
loading, and ``tests/test_v1_checkpoint.py`` fails the moment that stops being true.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bb.nn import (  # noqa: F401  (re-exported under the rsfff names)
    TwoSlotLinear,
    core_parameters,
    env_parameters,
    env_reduce_parameter,
    exempt_from_weight_decay,
    exempt_parameter,
    mark_env_slot,
    mlp,
    slot_reduce,
    two_slot_mlp,
    zero_init_readout,
)

__all__ = [
    "AtomicEnergyHead",
    "TwoSlotLinear",
    "core_parameters",
    "env_parameters",
    "env_reduce_parameter",
    "exempt_from_weight_decay",
    "exempt_parameter",
    "mark_env_slot",
    "mlp",
    "slot_reduce",
    "two_slot_mlp",
    "zero_init_readout",
]


class AtomicEnergyHead(nn.Module):
    """Per-atom scalar energy from invariant features + a learned species embedding.

    Args
    ----
    p0        : width of the invariant feature vector (``featurizer.feature_dims[0]``).
    n_species : number of distinct atomic species (embedding table size).
    emb_dim   : per-species embedding dimension.
    hidden    : MLP hidden width.
    depth     : number of hidden layers.

    Forward
    -------
    inv_feats   : (N, p0) lambda=0 invariants.
    species_idx : (N,) long, values in [0, n_species).
    returns     : (N,) per-atom energy (unreferenced; the model adds E0 offsets).
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.species_emb = nn.Embedding(n_species, emb_dim)
        # Zero-init readout: the model starts at the per-species E0 reference because the
        # residual it predicts begins at exactly 0. See `zero_init_readout` for why that also
        # means this block must not see weight decay.
        self.mlp = zero_init_readout(mlp(p0 + emb_dim, hidden, depth, 1))

    def forward(self, inv_feats: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        emb = self.species_emb(species_idx)                      # (N, emb_dim)
        x = torch.cat((inv_feats, emb), dim=-1)                  # (N, p0 + emb_dim)
        return self.mlp(x).squeeze(-1)                           # (N,)
