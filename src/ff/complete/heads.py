"""Environment heads (complete model plan, C2): every emitted parameter is the isolated-atom value times a factor
that is exactly 1 when the environment vanishes,

    theta_i = theta0[species_i] * exp( a . I_i + NN(I_i) - NN(0) )

with I_i the scaled environment invariants of atom i. The linear term and the network are shared by all species
(the species enters only through theta0, the centre atom's own parameters); an optional species embedding can be
concatenated to the network input, in which case NN(0) is evaluated with the same embedding so the isolated limit
still holds. Both last layers start at zero, so training starts from the isolated-atom force field."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["EnvironmentHead"]


def _mlp(i, h, o, depth):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EnvironmentHead(nn.Module):
    """``forward(I, species=None) -> (N, n_out)``: the log-factor ``a . I + NN(I) - NN(0)``, exactly zero at I = 0."""

    def __init__(self, n_feat: int, n_out: int, hidden: int = 64, depth: int = 2, n_species: int = 0, emb: int = 8):
        super().__init__()
        self.lin = nn.Linear(n_feat, n_out, bias=False)
        nn.init.zeros_(self.lin.weight)
        self.emb = nn.Embedding(n_species, emb) if n_species else None
        self.net = _mlp(n_feat + (emb if n_species else 0), hidden, n_out, depth)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, I: torch.Tensor, species: torch.Tensor | None = None) -> torch.Tensor:
        z0 = torch.zeros_like(I)
        if self.emb is not None:
            e = self.emb(species)
            I, z0 = torch.cat([I, e], -1), torch.cat([z0, e], -1)
        return self.lin(I[:, : self.lin.in_features]) + self.net(I) - self.net(z0)
