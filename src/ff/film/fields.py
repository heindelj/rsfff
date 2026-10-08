"""Electrostatic-environment features for the bonded parameters, second order by construction.

Model "C" of the monomer-bonded plan: the bonded (and torsion/coupling) parameters see the
potential, field and field gradient at each atom -- produced by **permanent** sources only
(external probe charges and uniform fields today; other fragments' permanent multipoles
later), never by induced moments -- on top of the geometric description of the neighbours.

**The rule this module enforces: the bonded energy has no first-order field dependence.**
At fixed geometry ``E_b(R; F) - E_b(R; 0) = O(F^2)``. The reasons, briefly (the plan note has
them in full):

* a first-order term ``-F . mu_b(R)`` would move the zero-field dipole away from the
  permanent heads, and no molecular label can tell the two apart; it would also book a
  first-order (electrostatic) energy outside the elst channel. It is not needed either: the
  permanent heads are geometry-dependent, so dipole derivatives already live there.
* with the rule, ``mu = -dE/dF`` at zero field is ``mu_perm`` (the definition is unchanged)
  and the molecular polarizability gains an exact extra term ``-d^2 E_b / dF^2``, computed by
  autograd (:meth:`rsfff.ff.film.FilmModel.response_properties`).

Construction. Per atom, **linear** rotation invariants of the sources in the atom's own frame:

    x0 = dphi_i / s0                      dphi = phi_i - <phi>_fragment   (gauge: a constant
                                                                           potential is invisible)
    x1 = E_i . V_ik / s1                  V = lambda=1 internal features, reduced to k channels
    x2 = gradE_i : Q_ik / s2              Q = lambda=2 internal features -> Cartesian, k channels

Only their **pairwise products** go further -- ``p = (A x) * (B x)``, learned bilinear forms --
and then ``shift = W2 tanh(W1 p)`` with ``W1``/``W2`` bias-free and ``W2`` zero-initialized.
``shift`` is therefore ``O(x^2) = O(F^2)`` exactly, zero in vacuum to the bit, and zero for a
fresh module. It is added to the bonded family's isolated latent in the *field-dressed*
evaluation only (``theta``, not ``theta_0``), so its energy lands where the film books every
environment response of the bonded potential: ``E_b(theta) - E_b(theta_0)`` in induction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...mlip.heads import exempt_from_weight_decay
from ..multipole import spherical_to_cartesian_quadrupole

__all__ = ["FieldFeatureShift"]


class FieldFeatureShift(nn.Module):
    """``(phi, E, gradE)`` per atom -> an O(F^2) shift of a latent of width ``latent_dim``."""

    def __init__(
        self,
        latent_dim: int,
        p1: int | None,
        p2: int | None,
        irrep2_to_spherical: torch.Tensor | None,
        *,
        ranks=(0, 1, 2),
        channels: int = 8,
        n_products: int = 16,
        hidden: int = 32,
        scales=(0.05, 0.02, 0.01),
    ) -> None:
        super().__init__()
        self.ranks = tuple(sorted(int(r) for r in ranks))
        if not set(self.ranks) <= {0, 1, 2} or not self.ranks:
            raise ValueError(f"field ranks must be a non-empty subset of {{0, 1, 2}}, got {ranks}")
        self.scales = tuple(float(s) for s in scales)
        d_x = 0
        if 0 in self.ranks:
            d_x += 1
        if 1 in self.ranks:
            if p1 is None:
                raise ValueError("rank-1 field features need lambda=1 internal features")
            self.reduce1 = nn.Parameter(torch.randn(p1, channels) / math.sqrt(p1))
            d_x += channels
        if 2 in self.ranks:
            if p2 is None or irrep2_to_spherical is None:
                raise ValueError("rank-2 field features need lambda=2 features + irrep2_to_spherical")
            self.reduce2 = nn.Parameter(torch.randn(p2, channels) / math.sqrt(p2))
            self.register_buffer("to_spherical", irrep2_to_spherical.clone(), persistent=False)
            d_x += channels
        self.d_x = d_x
        self.A = nn.Linear(d_x, n_products, bias=False)
        self.B = nn.Linear(d_x, n_products, bias=False)
        self.W1 = nn.Linear(n_products, hidden, bias=False)
        self.W2 = nn.Linear(hidden, latent_dim, bias=False)
        with torch.no_grad():
            self.W2.weight.zero_()
        exempt_from_weight_decay(self)

    def linear_invariants(self, phi, e, g, fragment_idx, n_fragments, x_in) -> torch.Tensor:
        """``(N, d_x)`` rotation invariants, each linear in the sources."""
        cols = []
        if 0 in self.ranks:
            n_f = phi.new_zeros(n_fragments).index_add_(0, fragment_idx, torch.ones_like(phi))
            mean = phi.new_zeros(n_fragments).index_add_(0, fragment_idx, phi) / n_f.clamp(min=1)
            cols.append(((phi - mean[fragment_idx]) / self.scales[0]).unsqueeze(-1))
        if 1 in self.ranks:
            v = torch.einsum("nmp,pk->nmk", x_in.vec_feats, self.reduce1)      # (N, 3, k)
            cols.append(torch.einsum("nm,nmk->nk", e, v) / self.scales[1])
        if 2 in self.ranks:
            q_irr = torch.einsum("nip,pk->nki", x_in.equiv_feats, self.reduce2)  # (N, k, 5)
            q_c = spherical_to_cartesian_quadrupole(q_irr @ self.to_spherical)   # (N, k, 3, 3)
            cols.append(torch.einsum("nab,nkab->nk", g, q_c) / self.scales[2])
        return torch.cat(cols, dim=-1)

    def forward(self, fields, fragment_idx, n_fragments, x_in) -> torch.Tensor:
        phi, e, g = fields
        x = self.linear_invariants(phi, e, g, fragment_idx, int(n_fragments), x_in)
        p = self.A(x) * self.B(x)
        return self.W2(torch.tanh(self.W1(p)))
