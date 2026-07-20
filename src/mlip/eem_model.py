"""Combined MLIP + closed-form-EEM model.

    E = sum_i [ E0_s + E_MLIP(inv_i, s_i) ] + E_EEM(q*, mu*; chi, eta, chivec, alpha)

The *parameter function* -- per-atom heads on the SOAP features conditioned on a
shared species embedding -- outputs the effective short-range EEM parameters:

    chi_i    scalar electronegativity  (drives nonzero charges at F = 0)
    chivec_i vector electronegativity  (drives nonzero atomic dipoles at F = 0)
    eta_i    hardness > 0              (softplus + floor; convex charge problem)
    alpha_i  dipole polarizability     (PSD via AtomicAlphaHead)

The EEM layer (``eem.py``) minimizes its energy in closed form with a Lagrange
multiplier enforcing the per-molecule total charge; the MLIP head corrects the
crude EEM short-range energy. Molecular observables come out analytically:

    mu_mol    = sum_i q_i r_i + sum_i mu_i
    alpha_mol = sum_i alpha_i + charge-flow term      (symmetric PSD)

Free-atom limit: a 1-atom system has zero features, so chi = chi0_s,
eta = eta0_s, chivec = 0 (zero vec features), alpha isotropic, q = Q (pinned by
the constraint) and E(Q) = E0 + E_MLIP(0, s) + chi0 Q + 1/2 eta0 Q^2 -- the
isolated-species anchors pin chi0/eta0 per element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..features import FlatLambdaSOAPFeaturizer, LambdaFeatures
from .eem import atomic_dipoles, charge_flow_polarizability, eem_charges, eem_energy
from .heads import AtomicEnergyHead, mlp
from .response_heads import (
    AtomicAlphaHead,
    AtomicVectorHead,
    voigt_vector_to_symmetric_matrix,
)


class EEMParameterHeads(nn.Module):
    """Per-atom EEM parameters (chi, eta, chivec, alpha) from features + species.

    All environment readouts are zero-initialized, so at init the parameters are
    the per-species baselines: chi = chi0 (0), eta = eta_init, chivec = 0, alpha
    isotropic -- a well-posed, strictly convex EEM from the first step.
    """

    def __init__(
        self,
        p0: int,
        p1: int,
        p2: int,
        n_species: int,
        irrep6_to_voigt: torch.Tensor,
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        eta_init: float = 0.5,
        eta_floor: float = 0.05,
        psd_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eta_floor = float(eta_floor)
        self.species_emb = nn.Embedding(n_species, emb_dim)

        # chi: per-species bias + zero-init environment MLP.
        self.chi0 = nn.Parameter(torch.zeros(n_species))
        self.chi_mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        # eta: softplus(bias + zero-init MLP) + floor, bias set so eta ~= eta_init.
        raw = torch.log(torch.expm1(torch.tensor(float(eta_init) - self.eta_floor)))
        self.eta0_raw = nn.Parameter(torch.full((n_species,), raw.item()))
        self.eta_mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        for m in (self.chi_mlp, self.eta_mlp):
            with torch.no_grad():
                m[-1].weight.zero_()
                m[-1].bias.zero_()

        self.chivec_head = AtomicVectorHead(
            p0, p1, emb_dim, hidden=hidden, depth=depth, equiv_channels=equiv_channels
        )
        self.alpha_head = AtomicAlphaHead(
            p0, p2, emb_dim, irrep6_to_voigt,
            hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            positive_isotropic=True, psd_floor=psd_floor,
        )

    def forward(self, feats: LambdaFeatures):
        """Returns ``(chi (N,), eta (N,), chivec (N,3), alpha (N,3,3))``."""
        s = feats.species_idx
        emb = self.species_emb(s)                                   # (N, D)
        x = torch.cat((feats.inv_feats, emb), dim=-1)               # (N, p0+D)
        chi = self.chi0[s] + self.chi_mlp(x).squeeze(-1)            # (N,)
        eta = (
            torch.nn.functional.softplus(
                self.eta0_raw[s] + self.eta_mlp(x).squeeze(-1)
            )
            + self.eta_floor
        )                                                           # (N,) > floor
        chivec = self.chivec_head(feats.inv_feats, emb, feats.vec_feats)      # (N, 3)
        alpha_voigt = self.alpha_head(feats.inv_feats, emb, feats.equiv_feats)  # (N, 6)
        alpha = voigt_vector_to_symmetric_matrix(alpha_voigt)       # (N, 3, 3)
        return chi, eta, chivec, alpha


@dataclass
class EEMOutput:
    """Model outputs at the closed-form EEM minimum.

    energy         : (M,)     total energy (E0 + MLIP + EEM, field term included)
    charges        : (N,)     EEM charges (sum per molecule == Q exactly)
    lam            : (M,)     Lagrange multipliers (negative chemical potential)
    atomic_dipoles : (N, 3)   mu_i = alpha_i (F - chivec_i)
    dipole         : (M, 3)   mu_mol = sum q r + sum mu_i (e*Angstrom, stored frame)
    alpha          : (M, 3, 3) analytic molecular polarizability (symmetric PSD)
    eta            : (N,)     hardnesses (diagnostic)
    feats          : per-atom features
    """

    energy: torch.Tensor
    charges: torch.Tensor
    lam: torch.Tensor
    atomic_dipoles: torch.Tensor
    dipole: torch.Tensor
    alpha: torch.Tensor
    eta: torch.Tensor
    feats: LambdaFeatures


class MLIPEEMModel(nn.Module):
    """Featurizer -> (MLIP energy head, EEM parameter heads) -> closed-form EEM."""

    def __init__(
        self,
        featurizer: FlatLambdaSOAPFeaturizer,
        energy_head: AtomicEnergyHead,
        params: EEMParameterHeads,
        reference_energies: torch.Tensor,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.energy_head = energy_head
        self.params = params
        self.register_buffer("reference_energies", reference_energies.clone())

    def forward(self, batch, *, field: torch.Tensor | None = None) -> EEMOutput:
        positions = batch.positions
        n_systems = int(batch.n_systems)
        feats = self.featurizer(batch)
        s = feats.species_idx

        if batch.total_charge is not None:
            total_q = batch.total_charge.to(positions.dtype)
        else:
            total_q = positions.new_zeros(n_systems)

        chi, eta, chivec, alpha = self.params(feats)

        q, lam = eem_charges(
            chi, eta, positions, feats.batch_idx, n_systems, total_q, field
        )
        mu_i = atomic_dipoles(chivec, alpha, feats.batch_idx, field)
        e_eem = eem_energy(
            q, chi, eta, chivec, alpha, positions, feats.batch_idx, n_systems, field
        )

        e_atom = self.energy_head(feats.inv_feats, s) + self.reference_energies[s]
        e_mol = e_eem + e_atom.new_zeros(n_systems).index_add_(0, feats.batch_idx, e_atom)

        mu_atom = q.unsqueeze(-1) * positions + mu_i                # (N, 3)
        mu_mol = mu_atom.new_zeros(n_systems, 3).index_add_(0, feats.batch_idx, mu_atom)

        alpha_mol = alpha.new_zeros(n_systems, 3, 3).index_add_(
            0, feats.batch_idx, alpha
        ) + charge_flow_polarizability(eta, positions, feats.batch_idx, n_systems)

        return EEMOutput(
            energy=e_mol,
            charges=q,
            lam=lam,
            atomic_dipoles=mu_i,
            dipole=mu_mol,
            alpha=alpha_mol,
            eta=eta,
            feats=feats,
        )


def build_eem_model(
    features_cfg,
    mlip_cfg,
    eem_cfg,
    neighbor_types: Sequence[int],
    reference_energies: torch.Tensor,
) -> MLIPEEMModel:
    """Construct a :class:`MLIPEEMModel` from config objects.

    ``features_cfg.selected_lambdas`` must contain 0, 1, and 2: the vector
    electronegativity head needs the lambda=1 (odd) features and the alpha head
    the lambda=2 features.
    """
    lambdas = tuple(int(v) for v in features_cfg.selected_lambdas)
    missing = {0, 1, 2} - set(lambdas)
    if missing:
        raise ValueError(
            f"MLIP+EEM needs lambdas 0, 1, 2 in features.selected_lambdas "
            f"(missing {sorted(missing)}): chivec uses lambda=1, alpha lambda=2"
        )
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        selected_lambdas=lambdas,
        backend=features_cfg.backend,
        density_channels=features_cfg.density_channels,
    )
    p0 = featurizer.feature_dims[0]
    p1 = featurizer.feature_dims[1]
    p2 = featurizer.feature_dims[2]
    n_species = len(neighbor_types)
    energy_head = AtomicEnergyHead(
        p0, n_species=n_species,
        emb_dim=mlip_cfg.emb_dim, hidden=mlip_cfg.hidden, depth=mlip_cfg.depth,
    )
    params = EEMParameterHeads(
        p0, p1, p2, n_species,
        featurizer.backend.irrep6_to_voigt(),
        emb_dim=eem_cfg.emb_dim,
        hidden=eem_cfg.hidden,
        depth=eem_cfg.depth,
        equiv_channels=eem_cfg.equiv_channels,
        eta_init=eem_cfg.eta_init,
        eta_floor=eem_cfg.eta_floor,
        psd_floor=eem_cfg.psd_floor,
    )
    return MLIPEEMModel(featurizer, energy_head, params, reference_energies)
