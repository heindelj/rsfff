"""Pairwise energy heads: neural corrections that live on pairs, not atoms.

Predicting a **pair** energy rather than an atomic energy is what makes an interaction
rigorously decomposable. A sum of atomic energies cannot be split into intra- and
inter-molecular parts -- each atom's contribution already mixes both. A sum over pairs
can: restrict the pair list to inter-fragment pairs and the sum *is* the intermolecular
energy, directly comparable to an EDA component.

The head is symmetric in ``(i, j)`` by construction (it reads only the sum and the
absolute difference of the two atoms' features), so an undirected ``i < j`` pair list is
unambiguous, and it is multiplied by a compact-support envelope so it is **exactly** zero
beyond its cutoff -- the property a logistic switch cannot provide.

Closest existing relative is :class:`rsfff.mlip.sqe.PairComplianceHead`, which shares the
symmetrization trick but predicts a positive compliance rather than a signed energy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..features.features import BesselBasis
from .heads import mlp
from .switch import pairwise_switch


class PairEnergyHead(nn.Module):
    """Signed pair energy ``dE_ij`` over a supplied pair list: (P,).

    Args
    ----
    p0           : width of the invariant feature vector (``featurizer.feature_dims[0]``).
    n_species    : embedding table size.
    emb_dim      : per-species embedding width. The species embedding is **not** optional:
                   ``inv_feats`` is the power spectrum of the *neighbor* density and does
                   not encode the center atom's element, so without it the head cannot
                   tell an O-H pair from an H-H pair at the same distance in an otherwise
                   degenerate environment. ``AtomicEnergyHead`` adds one for the same
                   reason.
    n_radial     : Bessel radial basis size for the pair distance.
    r_on, r_off  : envelope window (Angstrom). ``dE`` is full strength below ``r_on`` and
                   exactly zero at and beyond ``r_off``, with C2 forces at both ends.
                   ``r_off`` should not exceed the feature cutoff -- past it the atomic
                   features no longer see each other and the head would be extrapolating.
    energy_scale : output scale in Hartree. Targets here are ~3e-3 Ha while a freshly
                   initialized MLP emits O(1) activations, so without this the head's
                   effective learning rate is off by three orders of magnitude relative to
                   the parameter heads it competes with.

    The readout is zero-initialized in both weight and bias, so ``dE == 0`` exactly at
    initialization and training starts from the pure force field.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        n_radial: int = 8,
        r_on: float = 4.0,
        r_off: float = 5.0,
        energy_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if not r_off > r_on:
            raise ValueError(f"PairEnergyHead needs r_off > r_on, got {r_on}, {r_off}")
        self.r_on = float(r_on)
        self.r_off = float(r_off)
        self.energy_scale = float(energy_scale)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.radial = BesselBasis(n_radial, r_off)
        h = p0 + emb_dim
        self.net = mlp(2 * h + n_radial, hidden, depth, 1)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(
        self,
        inv_feats: torch.Tensor,      # (N, p0)
        species_idx: torch.Tensor,    # (N,)
        positions: torch.Tensor,      # (N, 3) Angstrom
        pair_index: torch.Tensor,     # (2, P)
        r: torch.Tensor | None = None,  # (P,) reuse the pair list's distances
    ) -> torch.Tensor:
        i, j = pair_index[0], pair_index[1]
        if r is None:
            r = (positions[i] - positions[j]).norm(dim=-1)
        h = torch.cat((inv_feats, self.species_emb(species_idx)), dim=-1)   # (N, p0+emb)
        h_i, h_j = h[i], h[j]
        # sum and |difference| are both invariant under swapping i and j
        x = torch.cat((h_i + h_j, (h_i - h_j).abs(), self.radial(r)), dim=-1)
        raw = self.energy_scale * self.net(x).squeeze(-1)
        return raw * pairwise_switch(r, self.r_on, self.r_off)
