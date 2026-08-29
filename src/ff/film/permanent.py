"""Permanent multipoles: direct outputs of the fragment-internal features, and nothing else.

The strict permanent/polarization separation of this model generation: the multipoles around
which polarization occurs are functions of the **fragment's own geometry** -- the isolated
latent and the internal feature blocks -- with *no environment columns in these heads at
all*, which is stronger than a zero-initialized environment sector. The full environmental
dependence of the multipoles comes from the polarization response and only from it.

Consequences relative to the v4 frozen level:

* there is no frozen SQE solve. Charges are a per-element prior plus a learned deviation,
  then an **exact projection** onto each fragment's formal charge;
* "permanent" is enforced by architecture, not by which slot a caller remembered to pass;
* the dipole/second-moment labels (``fragment_dipole``, ``fragment_second_moment``) pin these
  heads directly -- each label constrains one head, with no product of two heads in between.

The charge projection generalizes the v4 ``q0`` shift (``rsfff.ff.response``,
``response_parameters``) from ``fragment_idx`` to the assignment matrix ``C``: the shift is
distributed by membership, so it is the same exact per-fragment projection at a one-hot ``C``
and remains total-charge conserving and smooth for a fractional one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...features.features import LambdaFeatures
from ...mlip.heads import mlp, zero_init_readout
from ...mlip.response_heads import AtomicQuadrupoleHead, AtomicVectorHead
from .state import StateDescriptor

__all__ = ["PermanentMultipoleHeads"]


class PermanentMultipoleHeads(nn.Module):
    """``(q_perm, mu_perm, quad_perm)`` from the isolated latent and internal features.

    ``q``    : ``q0_prior[species] + MLP(z_iso, emb)`` (zero-init readout), then the exact
               fragment-charge projection. Starts at the pyCMM baseline charges, which is
               load-bearing -- see ``DEFAULT_Q0_PRIOR`` in :mod:`rsfff.ff.response` for the
               measured odd/even branch argument.
    ``mu``   : :class:`AtomicVectorHead` on the lambda=1 internal features (e*bohr).
    ``quad`` : :class:`AtomicQuadrupoleHead` on the lambda=2 internal features, spherical
               components (e*bohr^2). ``None`` when ``max_rank < 2``.
    """

    def __init__(
        self,
        latent_dim: int,
        p1: int | None,
        p2: int | None,
        n_species: int,
        *,
        q0_prior: torch.Tensor,                     # (n_species,)
        irrep2_to_spherical: torch.Tensor | None,
        max_rank: int = 2,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
    ) -> None:
        super().__init__()
        self.max_rank = int(max_rank)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.register_buffer("q0_prior", q0_prior.clone())
        self.q_mlp = zero_init_readout(mlp(latent_dim + emb_dim, hidden, depth, 1))

        self.mu_head = None
        self.quad_head = None
        if self.max_rank >= 1:
            if p1 is None:
                raise ValueError(
                    "permanent dipoles need lambda=1 features; set "
                    "features.selected_lambdas: [0, 1, 2] (or max_rank: 0)"
                )
            self.mu_head = AtomicVectorHead(
                latent_dim, p1, emb_dim,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            )
        if self.max_rank >= 2:
            if p2 is None or irrep2_to_spherical is None:
                raise ValueError(
                    "max_rank=2 needs lambda=2 features and the irrep2_to_spherical map"
                )
            self.quad_head = AtomicQuadrupoleHead(
                latent_dim, p2, emb_dim, irrep2_to_spherical,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            )

    @staticmethod
    def project_charges(
        q_raw: torch.Tensor, state: StateDescriptor
    ) -> torch.Tensor:
        """Exact charge projection through the assignment matrix.

        ``q = q_raw + C (Q - C^T q_raw) / n``, with ``n_f = sum_i C_if`` the (possibly
        fractional) fragment size. At a one-hot ``C`` each fragment's charges sum to ``Q_f``
        exactly, for any head output; for a fractional ``C`` the total charge is still
        conserved exactly and the correction varies smoothly with the assignment.
        """
        n_f = state.C.sum(dim=0)                                   # (F,)
        s_f = state.C.t() @ q_raw                                  # (F,)
        shift = (state.fragment_charge.to(q_raw.dtype) - s_f) / n_f.clamp(min=1.0)
        return q_raw + state.C @ shift

    def forward(
        self,
        z_iso: torch.Tensor,          # (N, latent) the permanent-family isolated latent
        x_in: LambdaFeatures,         # the internal feature blocks
        state: StateDescriptor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(q (N,), mu (N, 3) | None, quad_s (N, 5) | None)``, all a.u."""
        emb = self.species_emb(x_in.species_idx)
        q_raw = self.q0_prior[x_in.species_idx] + self.q_mlp(
            torch.cat((z_iso, emb), dim=-1)
        ).squeeze(-1)
        q = self.project_charges(q_raw, state)

        mu = None
        if self.mu_head is not None:
            mu = self.mu_head(z_iso, emb, x_in.vec_feats)
        quad_s = None
        if self.quad_head is not None:
            quad_s = self.quad_head(z_iso, emb, x_in.equiv_feats)
        return q, mu, quad_s
