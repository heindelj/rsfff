"""Charge-aware atomic embedding and the per-element charge-energy parabola.

The embedding ``z_i = f(element_i, q_i)`` is the layer-0 identity of an atom: a
learned per-element vector, FiLM-modulated by a smooth encoding of the continuous
per-atom charge ``q_i``. Everything downstream (density weights, energy head,
response heads) consumes ``z``, so charge-awareness propagates through the whole
model. At initialization the FiLM readouts are zero, so ``z(q) = element embedding``
exactly -- the model starts charge-agnostic and learns its q-dependence.

``ElementChargeEnergy`` is the EEM-style per-element term
``E_elem(s, q) = chi_s q + 1/2 eta_s q^2`` with a softplus-positive hardness
``eta``. It is *required* for the charge SCF to be well-posed at initialization:
every NN readout is zero-initialized, so without this term the energy would be
independent of q and the constrained Newton solve would face a singular KKT
system. It is also the natural carrier of the isolated-species integer-charge
anchors, and the future interface point for the long-range EEM/Ewald stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import mlp


class ChargeEmbedding(nn.Module):
    """FiLM-style charge-conditioned species embedding.

    ``z = e_s * (1 + gamma(q)) + beta(q)`` where ``e_s`` is a learned per-species
    vector and ``gamma, beta`` are small MLP readouts of the charge encoding.
    Smooth in q (SiLU everywhere); zero-initialized readouts give ``z(q) = e_s``
    at init.
    """

    def __init__(self, n_species: int, emb_dim: int, *, q_hidden: int = 16) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.q_encoder = mlp(1, q_hidden, 1, 2 * emb_dim)  # -> [gamma, beta]
        with torch.no_grad():
            self.q_encoder[-1].weight.zero_()
            self.q_encoder[-1].bias.zero_()

    def forward(self, species_idx: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """``species_idx``: (N,) long; ``q``: (N,) float -> ``z``: (N, emb_dim)."""
        e = self.species_emb(species_idx)                      # (N, D)
        gb = self.q_encoder(q.unsqueeze(-1))                   # (N, 2D)
        gamma, beta = gb[:, : self.emb_dim], gb[:, self.emb_dim :]
        return e * (1.0 + gamma) + beta


class AtomWeightNet(nn.Module):
    """Map an atomic embedding ``z`` to density-channel weights ``w(z)``: (N, Kw).

    ``w(z_j)`` weights neighbor j's contribution to the charge-aware density
    expansion ``A[i, k, n, lm] = sum_j w_k(z_j) R_n(r_ij) Y_lm f_cut`` -- the
    descriptor-based analog of message passing over charge-aware neighbor states.
    The weights are invariant scalars, so the density (and everything derived from
    it) stays exactly equivariant.
    """

    def __init__(self, emb_dim: int, k_weights: int, *, hidden: int = 32) -> None:
        super().__init__()
        self.k_weights = int(k_weights)
        self.net = mlp(emb_dim, hidden, 1, k_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)                                     # (N, Kw)


class ElementChargeEnergy(nn.Module):
    """Per-element charge parabola ``E_elem(s, q) = chi_s q + 1/2 eta_s q^2``.

    ``eta_s = softplus(raw) + floor`` is strictly positive, so at initialization
    (all NN readouts zero) the total energy is strictly convex in q and the
    constrained SCF has a unique solution. ``eta_init ~ 0.5 Ha/e^2`` is a typical
    chemical hardness; ``chi`` starts at 0 (electronegativity differences are
    learned, largely from the isolated-species anchors).
    """

    def __init__(
        self, n_species: int, *, eta_init: float = 0.5, eta_floor: float = 1e-2
    ) -> None:
        super().__init__()
        self.eta_floor = float(eta_floor)
        self.chi = nn.Parameter(torch.zeros(n_species))
        # softplus(raw) + floor == eta_init  =>  raw = log(expm1(eta_init - floor))
        raw = torch.log(torch.expm1(torch.tensor(float(eta_init) - self.eta_floor)))
        self.eta_raw = nn.Parameter(torch.full((n_species,), raw.item()))

    def hardness(self, species_idx: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.eta_raw[species_idx]) + self.eta_floor

    def forward(self, species_idx: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Per-atom charge energy (N,)."""
        chi = self.chi[species_idx]
        eta = self.hardness(species_idx)
        return chi * q + 0.5 * eta * q * q
