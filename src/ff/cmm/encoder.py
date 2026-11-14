"""The cmm encoder: one latent per atom, decoded linearly into every environment factor.

    x_i = [three-body channels T, two-body channels S, gated probe invariants G]     (scaled, saturated)
    z_i = phi(x_i, c_s) - phi(0, c_s)                                                 (c_s: species embedding, FiLM)
    u_i = W z_i                                                                       (one bias-free row per readout)

z is exactly zero where every input is zero (an isolated atom), so every parameter is its isolated-species value
there. The readouts are shared by all species; the species enters through the FiLM conditioning of the trunk.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["READOUTS", "Encoder"]

#: the environment log-factors, in readout-row order
READOUTS = ("pauli", "disp", "w1_dip", "w1_quad", "w1_breath", "w2", "qD", "qA")


def _mlp(i, h, o, depth):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class _FiLMTrunk(nn.Module):
    """``depth`` FiLM-conditioned hidden layers (bb.nn.ConditionedTrunk) and a linear output."""

    def __init__(self, i, h, o, depth, cond_dim):
        super().__init__()
        from bb.nn.conditioning import ConditionedTrunk
        self.trunk = ConditionedTrunk(i, h, depth, d_c=cond_dim, mode="film")
        self.out = nn.Linear(h, o)

    def diff(self, x, c):
        """net(x, c) - net(0, c) in one pass: the FiLM generators run once on c, both inputs go through stacked."""
        tr = self.trunk
        h = torch.stack([x, torch.zeros_like(x)], 0)
        for k, layer in enumerate(tr.layers):
            h = layer(h, tr.generators[k](c) if tr.generators is not None else None)
        y = self.out(h)
        return y[0] - y[1]


class _PlainTrunk(nn.Module):
    def __init__(self, i, h, o, depth):
        super().__init__()
        self.net = _mlp(i, h, o, depth)

    def diff(self, x, c=None):
        y = self.net(torch.cat([x, torch.zeros_like(x[:1])], 0))
        return y[:-1] - y[-1:]


class Encoder(nn.Module):
    """``forward(x, c=None) -> (N, len(READOUTS))``: the raw log-factors u (caps are applied by the model)."""

    def __init__(self, n_in: int, latent: int = 6, hidden: int = 64, depth: int = 2, cond_dim: int = 0):
        super().__init__()
        self.n_in, self.latent, self.cond_dim = int(n_in), int(latent), int(cond_dim)
        self.trunk = (_FiLMTrunk(n_in, hidden, latent, depth, cond_dim) if cond_dim
                      else _PlainTrunk(n_in, hidden, latent, depth))
        self.readout = nn.Linear(latent, len(READOUTS), bias=False)
        nn.init.zeros_(self.readout.weight)            # u = 0: training starts from the isolated-species force field

    def latents(self, x, c=None):
        """(N, latent) z = phi(x, c) - phi(0, c), exactly 0 on rows where x is all zero."""
        if self.cond_dim and c is None:
            raise ValueError("the FiLM encoder needs the species conditioning c")
        live = (x != 0).any(1, keepdim=True)
        z = self.trunk.diff(x, c)
        return torch.where(live, z, torch.zeros_like(z))

    def forward(self, x, c=None):
        return self.readout(self.latents(x, c))
