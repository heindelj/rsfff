"""Per-atom prediction heads for the MLIP.

The energy head is a plain invariant MLP: it consumes the lambda=0 SOAP invariants
(``inv_feats``) concatenated with a learned per-species embedding and maps them to a
single scalar energy per atom. Rotation/translation invariance comes for free because
``inv_feats`` are already O3 invariants; the per-molecule energy is the sum of the
per-atom energies (done by the model wrapper, not here).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    """Simple SiLU MLP: ``depth`` hidden layers of width ``hidden``."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


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
        self.mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        # Zero-init the readout so the model starts at the per-species E0 reference
        # (the residual it predicts begins at ~0) -- a numerically gentle start.
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, inv_feats: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        emb = self.species_emb(species_idx)                      # (N, emb_dim)
        x = torch.cat((inv_feats, emb), dim=-1)                  # (N, p0 + emb_dim)
        return self.mlp(x).squeeze(-1)                           # (N,)
