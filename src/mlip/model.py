"""Total-energy MLIP: SOAP featurizer -> per-atom energy head -> per-molecule sum.

``EnergyModel`` predicts a molecular total energy as

    E(mol) = sum_{i in mol} [ E0[species_i] + head(inv_feats_i, species_i) ]

The per-species reference energies ``E0`` (fit once from the training set) absorb the
large chemical baseline (~-76 Ha for water) so the head only learns a small residual.
``E0`` is position-independent, so forces ``-dE/dx`` are unaffected by it and come
entirely from the head + featurizer, which are fully differentiable in ``positions``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..features import FlatLambdaSOAPFeaturizer
from .heads import AtomicEnergyHead


class EnergyModel(nn.Module):
    """Featurizer + per-atom energy head + per-species reference + molecule pooling."""

    def __init__(
        self,
        featurizer: FlatLambdaSOAPFeaturizer,
        head: AtomicEnergyHead,
        reference_energies: torch.Tensor,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.head = head
        # Per-species baseline energy, indexed by species_idx (not atomic number).
        self.register_buffer("reference_energies", reference_energies.clone())

    def forward(self, batch) -> torch.Tensor:
        """Return per-molecule total energy, shape ``(n_systems,)``."""
        feats = self.featurizer(batch)
        e_atom = self.head(feats.inv_feats, feats.species_idx)          # (Ntot,)
        e_atom = e_atom + self.reference_energies[feats.species_idx]    # add E0 baseline
        n_systems = int(batch.n_systems)
        e_mol = e_atom.new_zeros(n_systems).index_add_(0, feats.batch_idx, e_atom)
        return e_mol


def build_model(
    features_cfg,
    mlip_cfg,
    neighbor_types: Sequence[int],
    reference_energies: torch.Tensor,
) -> EnergyModel:
    """Construct an :class:`EnergyModel` from config objects.

    ``features_cfg`` supplies the featurizer hyperparameters (cutoff/n_max/l_max/...),
    ``mlip_cfg`` the head hyperparameters (emb_dim/hidden/depth). ``neighbor_types`` is
    the sorted set of atomic numbers in the dataset; ``reference_energies`` is the
    per-species E0 baseline aligned to that sorted order.
    """
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        selected_lambdas=features_cfg.selected_lambdas,
        backend=features_cfg.backend,
        density_channels=features_cfg.density_channels,
    )
    p0 = featurizer.feature_dims[0]
    head = AtomicEnergyHead(
        p0,
        n_species=len(neighbor_types),
        emb_dim=mlip_cfg.emb_dim,
        hidden=mlip_cfg.hidden,
        depth=mlip_cfg.depth,
    )
    return EnergyModel(featurizer, head, reference_energies)
