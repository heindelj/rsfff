"""The learned adiabaticization map: a nonlinear correction to the weighted diabatic mean.

This is the core of Revision 3 of docs/mixture_of_diabatic_embeddings.md (§7). Given the per-atom
feature bundles of the active diabats ``{ĥ_i^(K)}`` and their coefficients ``c_K``, the adiabatic
atomic feature is

    z_i^ad = Σ_K c_K ĥ_i^(K)                         (weighted mean μ_i, §7.1)
           + Σ_{K<J} 4 c_K c_J 𝒪_KJ Ψ_i^KJ          (nonlinear pair correction, §7.3)

:class:`AdiabaticCorrection` computes the second term. Two properties are structural, not trained:

* **Pure-state vertex identity (§7.4).** Every pair prefactor ``4 c_K c_J`` is zero whenever one
  coefficient is 1 and the rest are 0, so the correction vanishes *exactly* at every simplex
  vertex for arbitrary network weights. The shared decoder then sees exactly the pretrained
  diabatic feature whenever the active space collapses to a single state.
* **Overlap shutdown (§7.5, §8.3).** The correction carries a geometric overlap factor ``𝒪_KJ``
  that goes to zero (C²-smoothly) when the fragments involved in the transition separate, so no
  nonlinear resonance survives at dissociation.

The correction is equivariant by construction (§7.9): invariant scalar gates multiply equivariant
feature channels, and the ``m`` (component) axes are never mixed. It reads only symmetric
combinations of the two states (``ĥ^K + ĥ^J`` and ``(ĥ^K − ĥ^J)²``), so it is invariant under
swapping ``K ↔ J`` as a pair correction must be.

The final readouts are **small-initialized** (not zero): the vertex identity already forces the
correction off at every pure state, so a small nonzero weight simply lets the untrained demo show
a genuine correction rising through a crossover and returning to zero at both ends, without
disturbing the frozen fit (which lives at the pure-state ends).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import mlp


@dataclass
class AdiabaticDelta:
    """The additive correction to each channel of the weighted diabatic mean.

    d_inv   : (N, P0)      scalar (l=0) channels
    d_vec   : (N, 3, P1)   vector (l=1) channels
    d_equiv : (N, 5, P2)   rank-2 (l=2) channels
    d_emb   : (N, D)       reference-embedding channels
    """

    d_inv: torch.Tensor
    d_vec: torch.Tensor
    d_equiv: torch.Tensor
    d_emb: torch.Tensor

    def norm(self) -> torch.Tensor:
        """A single scalar magnitude of the whole correction (diagnostic)."""
        return (
            self.d_inv.pow(2).sum()
            + self.d_vec.pow(2).sum()
            + self.d_equiv.pow(2).sum()
            + self.d_emb.pow(2).sum()
        ).sqrt()


class AdiabaticCorrection(nn.Module):
    """``Ψ_i^KJ`` and the pairwise assembly of ``Δz_i^mix`` (docs §7.3, §7.8, §7.9).

    One shared network is used for every state pair. The invariant input to ``Ψ`` (§7.3) is built
    from the symmetric sum/squared-difference of the two states' scalar channels and reference
    embeddings, the weighted-mean channels ``μ`` (§7.1), the invariant state **spread** ``v_i``
    (§7.2), and a center-level latent ``z_R`` (§7.8) that is shared across all atoms of the
    reactive center so oxygen and a departing hydrogen make a consistent collective transition.
    From those invariants it produces the scalar/embedding corrections directly and per-channel
    invariant gates that reduce the equivariant sums ``vec^K+vec^J`` and ``equiv^K+equiv^J`` (the
    :class:`~rsfff.mlip.response_heads.AtomicVectorHead` construction), keeping every irrep pure.
    """

    def __init__(
        self,
        p0: int,
        p1: int,
        p2: int,
        emb_dim: int,
        *,
        hidden: int = 64,
        depth: int = 2,
        z_r_dim: int = 16,
        init_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.p0, self.p1, self.p2, self.emb_dim = int(p0), int(p1), int(p2), int(emb_dim)

        # Center latent z_R: a small MLP on the c-weighted pooled scalar mean of the center.
        self.zr_mlp = mlp(p0, hidden, 1, z_r_dim)

        # Invariant input to Psi, per (atom, pair):
        #   inv_sum(P0) + inv_sqdiff(P0) + emb_sum(D) + emb_sqdiff(D)
        #   + mu_inv(P0) + mu_emb(D) + spread_inv(P0) + spread_vec(P1) + spread_equiv(P2) + z_R
        in_dim = 4 * p0 + 3 * emb_dim + p1 + p2 + z_r_dim
        # Outputs: d_inv(P0) + d_emb(D) + gate_vec(P1) + gate_equiv(P2).
        self.psi = mlp(in_dim, hidden, depth, p0 + emb_dim + p1 + p2)
        with torch.no_grad():
            self.psi[-1].weight.mul_(init_scale)
            self.psi[-1].bias.mul_(init_scale)

    def forward(
        self,
        inv: torch.Tensor,       # (M, N, P0)   per-diabat scalar channels, original atom order
        vec: torch.Tensor,       # (M, N, 3, P1)
        equiv: torch.Tensor,     # (M, N, 5, P2)
        emb: torch.Tensor,       # (M, N, D)
        c: torch.Tensor,         # (M,)   mixing coefficients (sum to 1)
        overlap: torch.Tensor,   # ()     geometric overlap factor 𝒪 on the transition contact
        center_mask: torch.Tensor,  # (N,) bool -- atoms of the reactive center (for z_R pooling)
    ) -> AdiabaticDelta:
        M, N, p0 = inv.shape
        d_inv = inv.new_zeros(N, p0)
        d_emb = emb.new_zeros(N, self.emb_dim)
        d_vec = vec.new_zeros(N, vec.shape[2], self.p1)
        d_equiv = equiv.new_zeros(N, equiv.shape[2], self.p2)
        if M < 2:  # a single active diabat -> no pairs, no correction (vertex by construction)
            return AdiabaticDelta(d_inv, d_vec, d_equiv, d_emb)

        cc = c.reshape(M, 1, 1)
        mu_inv = (cc * inv).sum(0)                                   # (N, P0)
        mu_emb = (c.reshape(M, 1, 1) * emb).sum(0)                   # (N, D)
        mu_vec = (c.reshape(M, 1, 1, 1) * vec).sum(0)                # (N, 3, P1)
        mu_equiv = (c.reshape(M, 1, 1, 1) * equiv).sum(0)            # (N, 5, P2)

        # Invariant spreads v_i = Σ_K c_K (·−μ)^2 (§7.2); the equivariant spreads are summed over
        # the component axis, which makes them rotation invariant.
        spread_inv = (cc * (inv - mu_inv).pow(2)).sum(0)                                # (N, P0)
        spread_vec = (c.reshape(M, 1, 1, 1) * (vec - mu_vec).pow(2)).sum(0).sum(1)      # (N, P1)
        spread_equiv = (c.reshape(M, 1, 1, 1) * (equiv - mu_equiv).pow(2)).sum(0).sum(1)  # (N, P2)

        # Center latent z_R: pool the scalar mean over the reactive-center atoms (§7.8).
        denom = center_mask.sum().clamp(min=1).to(mu_inv.dtype)
        zr_in = (mu_inv * center_mask.unsqueeze(-1)).sum(0) / denom  # (P0,)
        z_R = self.zr_mlp(zr_in).unsqueeze(0).expand(N, -1)          # (N, z_r_dim)

        for K in range(M):
            for J in range(K + 1, M):
                w = 4.0 * c[K] * c[J] * overlap                     # scalar prefactor, zero at vertices
                inv_sum, inv_sqd = inv[K] + inv[J], (inv[K] - inv[J]).pow(2)
                emb_sum, emb_sqd = emb[K] + emb[J], (emb[K] - emb[J]).pow(2)
                x = torch.cat(
                    (inv_sum, inv_sqd, emb_sum, emb_sqd, mu_inv, mu_emb,
                     spread_inv, spread_vec, spread_equiv, z_R),
                    dim=-1,
                )
                out = self.psi(x)
                di, de, g_vec, g_equiv = torch.split(
                    out, (self.p0, self.emb_dim, self.p1, self.p2), dim=-1
                )
                d_inv = d_inv + w * di
                d_emb = d_emb + w * de
                # Per-channel invariant gate x equivariant sum -> stays l=1 / l=2 covariant.
                d_vec = d_vec + w * g_vec.unsqueeze(1) * (vec[K] + vec[J])
                d_equiv = d_equiv + w * g_equiv.unsqueeze(1) * (equiv[K] + equiv[J])
        return AdiabaticDelta(d_inv, d_vec, d_equiv, d_emb)
