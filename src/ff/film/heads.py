"""Per-family parameter heads reading the conditioned family latents.

The film model's heads deliberately **reuse** the v4 head classes wherever one exists --
:class:`AtomicAlphaHead` (PSD polarizability), :class:`PairComplianceHead` (symmetric
per-channel compliance), :class:`PauliMultipoleHeads`, :class:`DispersionParameterHeads` --
with one change of input: where v4 fed them the raw ``[h | eta]`` invariants, these read the
**family latent** ``z_fam`` from the conditioned trunk (:mod:`rsfff.ff.film.network`).

The two-evaluation convention survives, one level up: every head is called once with the
isolated latent (``theta_0``) and once with the env-dressed one (``theta``). The vertex
guarantee is structural at the *feature* level now -- an isolated fragment has ``x_env`` and
``x_cross`` exactly zero, the env/cross embedders are linear with no bias, so the two latents
are bit-identical and ``theta == theta_0`` without any convention for a caller to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...features.features import LambdaFeatures
from ...mlip.heads import mlp, zero_init_readout
from ...mlip.response_heads import AtomicAlphaHead, voigt_vector_to_symmetric_matrix
from ...mlip.sqe import PairComplianceHead

__all__ = ["FilmResponseHeads", "ResponseFamily"]


@dataclass
class ResponseFamily:
    """The polarization-response parameters at one evaluation.

    ``eta``        : (N,) on-site hardness, strictly positive.
    ``alpha``      : (N, 3, 3) atomic dipole polarizability, PSD.
    ``compliance`` : (Nb,) per-channel SQE compliance.
    ``z``, ``b``   : (N,) charge-penetration effective charge and Slater exponent.
    """

    eta: torch.Tensor
    alpha: torch.Tensor | None
    compliance: torch.Tensor
    z: torch.Tensor
    b: torch.Tensor


class FilmResponseHeads(nn.Module):
    """``(eta, alpha, compliance, Z, b)`` from the response-family latent.

    The construction mirrors :class:`rsfff.ff.response.ElectrostaticParameterHeads` minus
    everything permanent (chi, chivec/mu0, chiquad/quad0 -- the permanent multipoles have
    their own module and there is no frozen solve to drive). ``eta`` and the penetration
    parameters keep their per-species priors and positive transforms; ``alpha`` keeps the
    PSD construction; the compliance head keeps its ``s_init`` bias.
    """

    def __init__(
        self,
        latent_dim: int,
        p2: int | None,
        n_species: int,
        *,
        log_z_prior: torch.Tensor,        # (n_species,)
        log_b_prior: torch.Tensor,        # (n_species,)
        irrep6_to_voigt: torch.Tensor | None,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        eta_init: float = 0.5,
        eta_floor: float = 0.05,
        psd_floor: float = 1e-4,
        learn_alpha: bool = True,
        learn_z: bool = True,
        learn_b: bool = True,
        compliance_hidden: int = 64,
        compliance_depth: int = 2,
        compliance_radial: int = 8,
        compliance_cutoff: float = 5.0,
        s_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.eta_floor = float(eta_floor)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.register_buffer("log_z_prior", log_z_prior.clone())
        self.register_buffer("log_b_prior", log_b_prior.clone())

        eta0 = torch.full((n_species,), float(eta_init)).clamp(min=self.eta_floor + 1e-6)
        self.eta0_raw = nn.Parameter(torch.log(torch.expm1(eta0 - self.eta_floor)))
        self.eta_mlp = zero_init_readout(mlp(latent_dim + emb_dim, hidden, depth, 1))

        self.d_log_z = nn.Parameter(torch.zeros(n_species), requires_grad=learn_z)
        self.d_log_b = nn.Parameter(torch.zeros(n_species), requires_grad=learn_b)

        self.alpha_head = None
        if learn_alpha:
            if p2 is None or irrep6_to_voigt is None:
                raise ValueError(
                    "the dipole polarizability needs lambda=2 features and the "
                    "irrep6_to_voigt map; set features.selected_lambdas: [0, 1, 2]"
                )
            self.alpha_head = AtomicAlphaHead(
                latent_dim, p2, emb_dim, irrep6_to_voigt,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
                positive_isotropic=True, psd_floor=psd_floor,
            )

        self.compliance_head = PairComplianceHead(
            latent_dim,
            hidden=compliance_hidden, depth=compliance_depth,
            n_radial=compliance_radial, cutoff=compliance_cutoff, s_init=s_init,
        )

    def forward(
        self,
        z: torch.Tensor,              # (N, latent) the response-family latent
        x_in: LambdaFeatures,         # internal blocks: source of the equivariant channels
        species_idx: torch.Tensor,
        positions: torch.Tensor,
        bond_index: torch.Tensor,     # (2, Nb) the SQE channel graph
    ) -> ResponseFamily:
        emb = self.species_emb(species_idx)
        x = torch.cat((z, emb), dim=-1)
        eta = (
            torch.nn.functional.softplus(
                self.eta0_raw[species_idx] + self.eta_mlp(x).squeeze(-1)
            )
            + self.eta_floor
        )
        alpha = None
        if self.alpha_head is not None:
            alpha = voigt_vector_to_symmetric_matrix(
                self.alpha_head(z, emb, x_in.equiv_feats)
            )
        compliance = self.compliance_head(z, positions, bond_index)
        log_z = self.log_z_prior[species_idx] + self.d_log_z[species_idx]
        log_b = self.log_b_prior[species_idx] + self.d_log_b[species_idx]
        return ResponseFamily(
            eta=eta, alpha=alpha, compliance=compliance,
            z=log_z.exp(), b=log_b.exp(),
        )
