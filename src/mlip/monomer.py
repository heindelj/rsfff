"""Phase-1 frozen monomer stack: reference states -> monomer features -> SQE observables.

This assembles the components of docs/mixture_of_diabatic_embeddings.md §2.1-§2.4 that Phase 1
trains and then freezes permanently::

    diabatic state (Q_a, S_a, channel graph)
        |
    ReferenceEmbedding  ->  e_i, q_i^(0)                        (§2.1 step 1, §2.4.1)
        |
    AtomWeightNet -> w(e_i) -> FlatStateSOAPFeaturizer -> h_{i,0} (§2.1 step 2, "F_mono")
        |
        +-- E_1(h, e)          1-body short-range energy         (§2.3.1)
        +-- chi_{i,0}(h, e)    baseline electronegativity
        +-- eta_i(h, e)        on-site hardness
        +-- chivec_{i,0}(h, e) vector electronegativity (permanent atomic dipoles)
        +-- alpha_{i,0}(h, e)  baseline atomic polarizability (PSD)
        +-- s_ij(h, r_ij)      intra-fragment channel compliance (§2.3.3)
        |
    SQE solve  ->  charges, dipole, polarizability, energy       (§2.4)

**The free-atom limit is exact, not fitted.** A lone atom has an all-zero density, so every
head reduces to a function of its reference embedding alone, and with no channels the SQE solve
returns ``q = q^(0) = Q``. The energy collapses to

    E = E0[Z] + E_1(e) + chi Q + 1/2 eta Q^2

Initializing the per-element ``chi`` and ``eta`` biases at the Mulliken electronegativity
``(IP + EA)/2`` and the chemical hardness ``IP - EA`` therefore reproduces the isolated-atom
energies at ``Q = 0, +1, -1`` *exactly at initialization* (the zero-initialized ``E_1`` readout
contributes nothing): ``chi + eta/2 = IP`` and ``-chi + eta/2 = -EA`` identically. The atomic
reference states are not merely a loss term -- they fix where the model starts.

Phase 2 will freeze everything here and add the state-decorated ACE correction ``Delta h`` on
top; the subtraction-trick heads of §2.3.2 then keep these predictions untouched at isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..features import FlatStateSOAPFeaturizer, LambdaFeatures
from .eem import atomic_dipoles
from .heads import mlp
from .reference_states import AtomicStateReference, AtomWeightNet, ReferenceEmbedding
from .response_heads import (
    AtomicAlphaHead,
    AtomicVectorHead,
    voigt_vector_to_symmetric_matrix,
)
from .sqe import PairComplianceHead, atomic_dipole_energy, sqe_solve


class MonomerParameterHeads(nn.Module):
    """The 1-body heads on ``(h_{i,0}, e_i)``: energy, chi, eta, chivec, alpha.

    Every head takes the reference embedding ``e_i`` as its conditioning vector, which is what
    gives the free-atom limit described in the module docstring: with zero features the heads
    are functions of ``(element, Q, S)`` alone.

    All environment readouts are zero-initialized, so at step 0 the model *is* its per-element
    reference: ``E_1 = 0``, ``chi = chi_0``, ``eta = softplus^-1(eta_0)``, ``chivec = 0``, alpha
    isotropic. ``chi_0`` and ``eta_0`` are seeded from the atomic IP/EA data when available.
    """

    def __init__(
        self,
        p0: int,
        p1: int,
        p2: int,
        n_species: int,
        emb_dim: int,
        irrep6_to_voigt: torch.Tensor,
        *,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        chi_init: torch.Tensor | None = None,   # (n_species,) Hartree
        eta_init: torch.Tensor | None = None,   # (n_species,) Hartree
        eta_floor: float = 0.05,
        psd_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eta_floor = float(eta_floor)
        n_species = int(n_species)
        for name, init in (("chi_init", chi_init), ("eta_init", eta_init)):
            if init is not None and init.shape != (n_species,):
                raise ValueError(
                    f"{name} has shape {tuple(init.shape)}, expected ({n_species},) to match "
                    f"the featurizer's species count"
                )

        self.energy_mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        self.chi_mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        self.eta_mlp = mlp(p0 + emb_dim, hidden, depth, 1)
        for m in (self.energy_mlp, self.chi_mlp, self.eta_mlp):
            with torch.no_grad():
                m[-1].weight.zero_()
                m[-1].bias.zero_()

        # Per-element biases. eta is passed through softplus + floor to stay strictly positive
        # (a convex charge problem), so the stored parameter is the pre-softplus value.
        chi0 = torch.zeros(n_species) if chi_init is None else chi_init.clone()
        eta0 = (
            torch.full((n_species,), 0.5) if eta_init is None else eta_init.clone()
        ).clamp(min=self.eta_floor + 1e-6)
        self.chi0 = nn.Parameter(chi0)
        self.eta0_raw = nn.Parameter(torch.log(torch.expm1(eta0 - self.eta_floor)))

        self.chivec_head = AtomicVectorHead(
            p0, p1, emb_dim, hidden=hidden, depth=depth, equiv_channels=equiv_channels
        )
        self.alpha_head = AtomicAlphaHead(
            p0, p2, emb_dim, irrep6_to_voigt,
            hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            positive_isotropic=True, psd_floor=psd_floor,
        )

    def forward(self, feats: LambdaFeatures, emb: torch.Tensor):
        """Returns ``(e1 (N,), chi (N,), eta (N,), chivec (N,3), alpha (N,3,3))``."""
        s = feats.species_idx
        x = torch.cat((feats.inv_feats, emb), dim=-1)
        e1 = self.energy_mlp(x).squeeze(-1)
        chi = self.chi0[s] + self.chi_mlp(x).squeeze(-1)
        eta = (
            torch.nn.functional.softplus(self.eta0_raw[s] + self.eta_mlp(x).squeeze(-1))
            + self.eta_floor
        )
        chivec = self.chivec_head(feats.inv_feats, emb, feats.vec_feats)
        alpha = voigt_vector_to_symmetric_matrix(
            self.alpha_head(feats.inv_feats, emb, feats.equiv_feats)
        )
        return e1, chi, eta, chivec, alpha


@dataclass
class MonomerOutput:
    """Phase-1 model outputs at the split-charge minimum.

    energy          : (M,)      total energy (E0 + E_1 + SQE charge + dipole sectors)
    charges         : (N,)      SQE charges; sum per fragment == formal charge exactly
    transfers       : (Nb,)     split charges p_ij along each channel
    compliance      : (Nb,)     learned channel compliances s_ij
    baseline_charge : (N,)      q_i^(0)
    atomic_dipoles  : (N, 3)    mu_i = alpha_i (F - chivec_i)
    dipole          : (M, 3)    mu_mol = sum q r + sum mu_i (e*Angstrom, stored frame)
    alpha           : (M, 3, 3) sum_i alpha_i + charge-flow term (symmetric PSD)
    eta             : (N,)      on-site hardnesses (diagnostic)
    embedding       : (N, D)    reference embeddings e_i
    feats           : per-atom monomer features h_{i,0}
    """

    energy: torch.Tensor
    charges: torch.Tensor
    transfers: torch.Tensor
    compliance: torch.Tensor
    baseline_charge: torch.Tensor
    atomic_dipoles: torch.Tensor
    dipole: torch.Tensor
    alpha: torch.Tensor
    eta: torch.Tensor
    embedding: torch.Tensor
    feats: LambdaFeatures


class MonomerModel(nn.Module):
    """The Phase-1 frozen monomer stack (see module docstring)."""

    def __init__(
        self,
        embedding: ReferenceEmbedding,
        weight_net: AtomWeightNet,
        featurizer: FlatStateSOAPFeaturizer,
        params: MonomerParameterHeads,
        compliance_head: PairComplianceHead,
        reference_energies: torch.Tensor,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.weight_net = weight_net
        self.featurizer = featurizer
        self.params = params
        self.compliance_head = compliance_head
        self.register_buffer("reference_energies", reference_energies.clone())

    def forward(self, batch, *, field: torch.Tensor | None = None) -> MonomerOutput:
        if batch.fragment_idx is None or batch.bond_index is None:
            raise ValueError(
                "MonomerModel needs the diabatic assignment on the batch (fragment_idx, "
                "fragment_charge, fragment_two_s, bond_index, bond_batch). Load the dataset "
                "with a DiabaticStateLibrary: load_datasets(..., library=...)."
            )
        positions = batch.positions
        n_systems = int(batch.n_systems)
        n_fragments = int(batch.n_fragments)
        species_idx = self.featurizer.species_index(batch.atomic_numbers)

        ref = self.embedding(
            species_idx,
            batch.fragment_idx,
            batch.fragment_charge.to(positions.dtype),
            batch.fragment_two_s.to(positions.dtype),
            n_fragments,
        )
        feats = self.featurizer(
            batch, self.weight_net(ref.embedding), fragment_idx=batch.fragment_idx
        )
        e1, chi, eta, chivec, alpha_i = self.params(feats, ref.embedding)

        compliance = self.compliance_head(feats.inv_feats, positions, batch.bond_index)
        sol = sqe_solve(
            chi, eta, compliance, ref.baseline_charge, positions,
            batch.bond_index, feats.batch_idx, batch.bond_batch, n_systems, field,
        )

        mu_i = atomic_dipoles(chivec, alpha_i, feats.batch_idx, field)
        e_dip = atomic_dipole_energy(chivec, alpha_i, feats.batch_idx, n_systems, field)

        e_atom = e1 + self.reference_energies[species_idx]
        e_mol = (
            sol.energy
            + e_dip
            + e_atom.new_zeros(n_systems).index_add_(0, feats.batch_idx, e_atom)
        )

        mu_atom = sol.charges.unsqueeze(-1) * positions + mu_i
        mu_mol = mu_atom.new_zeros(n_systems, 3).index_add_(0, feats.batch_idx, mu_atom)
        alpha_mol = alpha_i.new_zeros(n_systems, 3, 3).index_add_(
            0, feats.batch_idx, alpha_i
        ) + sol.alpha_flow

        return MonomerOutput(
            energy=e_mol,
            charges=sol.charges,
            transfers=sol.transfers,
            compliance=compliance,
            baseline_charge=ref.baseline_charge,
            atomic_dipoles=mu_i,
            dipole=mu_mol,
            alpha=alpha_mol,
            eta=eta,
            embedding=ref.embedding,
            feats=feats,
        )


def build_monomer_model(
    features_cfg,
    monomer_cfg,
    sqe_cfg,
    neighbor_types: Sequence[int],
    reference_energies: torch.Tensor,
    atomic_states: AtomicStateReference | None = None,
) -> MonomerModel:
    """Construct a :class:`MonomerModel` from config objects.

    ``features_cfg.selected_lambdas`` must contain 0, 1 and 2: chivec needs the lambda=1
    (odd-parity) channels and alpha the lambda=2 ones. ``atomic_states``, when supplied, seeds
    the per-element chi/eta biases from the isolated-atom IP/EA -- see the module docstring for
    why that makes the free-atom energies exact at initialization.
    """
    lambdas = tuple(int(v) for v in features_cfg.selected_lambdas)
    missing = {0, 1, 2} - set(lambdas)
    if missing:
        raise ValueError(
            f"the monomer stack needs lambdas 0, 1, 2 in features.selected_lambdas "
            f"(missing {sorted(missing)}): chivec uses lambda=1, alpha lambda=2"
        )
    n_species = len(neighbor_types)

    embedding = ReferenceEmbedding(
        n_species,
        emb_dim=monomer_cfg.emb_dim,
        hidden=monomer_cfg.hidden,
        depth=1,
    )
    weight_net = AtomWeightNet(
        monomer_cfg.emb_dim, monomer_cfg.weight_channels, hidden=monomer_cfg.hidden, depth=1
    )
    featurizer = FlatStateSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        weight_channels=monomer_cfg.weight_channels,
        selected_lambdas=lambdas,
        backend=features_cfg.backend,
    )
    chi_init = eta_init = None
    if atomic_states is not None:
        chi_init, eta_init = atomic_states.head_bias_init(eta_default=sqe_cfg.eta_init)
    params = MonomerParameterHeads(
        featurizer.feature_dims[0],
        featurizer.feature_dims[1],
        featurizer.feature_dims[2],
        n_species,
        monomer_cfg.emb_dim,
        featurizer.backend.irrep6_to_voigt(),
        hidden=monomer_cfg.hidden,
        depth=monomer_cfg.depth,
        equiv_channels=monomer_cfg.equiv_channels,
        chi_init=chi_init,
        eta_init=eta_init,
        eta_floor=sqe_cfg.eta_floor,
        psd_floor=sqe_cfg.psd_floor,
    )
    compliance_head = PairComplianceHead(
        featurizer.feature_dims[0],
        hidden=monomer_cfg.hidden,
        depth=monomer_cfg.depth,
        n_radial=sqe_cfg.n_radial,
        cutoff=features_cfg.cutoff,
        s_init=sqe_cfg.s_init,
        s_floor=sqe_cfg.s_floor,
    )
    return MonomerModel(
        embedding, weight_net, featurizer, params, compliance_head, reference_energies
    )
