"""Charge-aware MLIP: embedding z(element, q) -> weighted density -> heads, with a
charge SCF inner loop.

Energy of a batch (per molecule m, atoms i):

    E_m(q) = sum_{i in m} [ E0_s + E_elem(s_i, q_i) + head(inv_i(q), z_i) ]
             - mu_m(q) . F_m,        mu_m = sum_{i in m} q_i r_i (+ mu_i atomic)

where every q-dependence flows through the embedding ``z_i = f(s_i, q_i)``: the
heads consume z directly, and the density expansion weights each *neighbor's*
contribution by ``w(z_j)``, so the environment features are charge-aware
throughout. The charges are determined by the constrained SCF
``min_q E(q) s.t. sum_{i in m} q_i = Q_m`` (see ``charge_scf.py``) and re-attached
to the graph with implicit differentiation, so forces, dipole derivatives
``d mu/dR`` and the field polarizability ``alpha = -d2 E*/dF2`` are all exact
derivatives *through* the SCF.

The field term lives inside the SCF energy even at F = 0: that is what makes the
charge response ``dq*/dF`` (and hence alpha) available by differentiating the
attached solution at zero field.

Free-atom limit: a 1-atom system has no edges, all SOAP features are exactly
zero, and the constraint pins q = Q -- the model output reduces to
``E0 + E_elem(s, Q) + head(0, z(s, Q))``, anchored by the isolated-species data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..features import FlatLambdaSOAPFeaturizer, GeometryCache, LambdaFeatures
from .charge_embedding import AtomWeightNet, ChargeEmbedding, ElementChargeEnergy
from .charge_scf import RaggedIndex, ScfInfo, attach_charges, solve_charges
from .heads import ChargeEnergyHead
from .response_heads import (
    AtomicAlphaHead,
    AtomicVectorHead,
    voigt_vector_to_symmetric_matrix,
)


@dataclass
class ChargeAwareOutput:
    """Everything the losses and diagnostics consume, from one final evaluation
    at the attached charges.

    energy     : (M,)    per-molecule total energy (field term included)
    charges    : (N,)    attached SCF charges (graph carries the implicit diff)
    dipole     : (M, 3)  mu = sum q_i r_i (+ atomic dipoles), model units e*Angstrom
    alpha_head : (M, 3, 3) head-based molecular polarizability (sum of atomic alphas)
    field      : (M, 3)  the applied-field tensor (differentiate mu w.r.t. this for
                         the field-route polarizability)
    feats      : per-atom features at the attached charges
    scf_info   : forward-solver diagnostics (None when charges were frozen)
    """

    energy: torch.Tensor
    charges: torch.Tensor
    dipole: torch.Tensor
    alpha_head: torch.Tensor
    field: torch.Tensor
    feats: LambdaFeatures
    scf_info: ScfInfo | None


class ChargeAwareModel(nn.Module):
    """Featurizer + charge embedding + SCF + energy/response heads."""

    def __init__(
        self,
        featurizer: FlatLambdaSOAPFeaturizer,
        charge_embedding: ChargeEmbedding,
        weight_net: AtomWeightNet,
        element_energy: ElementChargeEnergy,
        energy_head: ChargeEnergyHead,
        alpha_head: AtomicAlphaHead,
        dipole_head: AtomicVectorHead | None,
        reference_energies: torch.Tensor,
        *,
        scf_max_iter: int = 30,
        scf_tol: float = 1e-9,
        scf_damping: float = 0.0,
        attach_steps: int = 1,
        hessian_in_graph: bool = True,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.charge_embedding = charge_embedding
        self.weight_net = weight_net
        self.element_energy = element_energy
        self.energy_head = energy_head
        self.alpha_head = alpha_head
        self.dipole_head = dipole_head
        self.register_buffer("reference_energies", reference_energies.clone())
        self.scf_max_iter = int(scf_max_iter)
        self.scf_tol = float(scf_tol)
        self.scf_damping = float(scf_damping)
        self.attach_steps = int(attach_steps)
        self.hessian_in_graph = bool(hessian_in_graph)

    # ------------------------------------------------------------------
    # Core evaluation at a given charge vector
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        cache: GeometryCache,
        positions: torch.Tensor,
        field: torch.Tensor,          # (M, 3)
        n_systems: int,
        q: torch.Tensor,              # (N,)
        *,
        compute_response: bool = False,
    ):
        """Full model evaluation at charges ``q``.

        Returns ``(e_mol, mu, feats, z, alpha_head)`` -- the response head only
        when requested (the SCF inner loop skips it).
        """
        s = cache.species_idx
        z = self.charge_embedding(s, q)                            # (N, D)
        w = self.weight_net(z)                                     # (N, Kw)
        feats = self.featurizer.features_from_weights(cache, w)

        e_atom = (
            self.energy_head(feats.inv_feats, z)
            + self.reference_energies[s]
            + self.element_energy(s, q)
        )                                                          # (N,)
        e_mol = e_atom.new_zeros(n_systems).index_add_(0, cache.batch_idx, e_atom)

        # Molecular dipole mu = sum_i q_i r_i (+ atomic dipoles). Charged systems:
        # origin-dependent, compared in the stored coordinate frame (see data.py).
        mu_atom = q.unsqueeze(-1) * positions                      # (N, 3)
        if self.dipole_head is not None:
            mu_atom = mu_atom + self.dipole_head(feats.inv_feats, z, feats.vec_feats)
        mu = mu_atom.new_zeros(n_systems, 3).index_add_(0, cache.batch_idx, mu_atom)

        # Applied uniform field: E -= mu . F (inside the SCF energy so dq*/dF and
        # hence alpha = -d2E*/dF2 are captured by the attach step).
        e_mol = e_mol - (mu * field).sum(dim=-1)

        alpha = None
        if compute_response:
            alpha_voigt = self.alpha_head(feats.inv_feats, z, feats.equiv_feats)  # (N, 6)
            alpha_atom = voigt_vector_to_symmetric_matrix(alpha_voigt)            # (N, 3, 3)
            alpha = alpha_atom.new_zeros(n_systems, 3, 3).index_add_(
                0, cache.batch_idx, alpha_atom
            )
        return e_mol, mu, feats, z, alpha

    # ------------------------------------------------------------------
    # Forward: SCF + attached final evaluation
    # ------------------------------------------------------------------

    def forward(
        self,
        batch,
        *,
        field: torch.Tensor | None = None,
        freeze_charges: bool = False,
    ) -> ChargeAwareOutput:
        """Solve the charge SCF and evaluate all outputs at the attached charges.

        ``field``: optional (M, 3) applied uniform field; pass a zeros tensor with
        ``requires_grad_(True)`` to obtain the polarizability by differentiating
        ``output.dipole`` w.r.t. it. ``freeze_charges=True`` pins q at the uniform
        feasible point Q/n and skips the SCF (the doc's ablation).
        """
        positions = batch.positions
        n_systems = int(batch.n_systems)
        cache = self.featurizer.precompute_geometry(batch)
        ragged = RaggedIndex.build(batch.batch_idx, n_systems)

        if batch.total_charge is not None:
            total_q = batch.total_charge.to(positions.dtype)
        else:
            total_q = positions.new_zeros(n_systems)
        if field is None:
            field = positions.new_zeros(n_systems, 3)

        # Uniform feasible initialization q_i = Q_m / n_m.
        q0 = (total_q / ragged.counts.clamp(min=1).to(positions.dtype))[batch.batch_idx]

        def energy_fn(q: torch.Tensor) -> torch.Tensor:
            e_mol, _, _, _, _ = self._evaluate(
                cache, positions, field, n_systems, q, compute_response=False
            )
            return e_mol

        scf_info: ScfInfo | None = None
        if freeze_charges:
            q_att = q0
        else:
            q_star, scf_info = solve_charges(
                energy_fn, q0, ragged,
                max_iter=self.scf_max_iter, tol=self.scf_tol, damping=self.scf_damping,
            )
            q_att = attach_charges(
                energy_fn, q_star, ragged,
                steps=self.attach_steps, hessian_in_graph=self.hessian_in_graph,
            )

        e_mol, mu, feats, _, alpha = self._evaluate(
            cache, positions, field, n_systems, q_att, compute_response=True
        )
        return ChargeAwareOutput(
            energy=e_mol,
            charges=q_att,
            dipole=mu,
            alpha_head=alpha,
            field=field,
            feats=feats,
            scf_info=scf_info,
        )


def build_charge_model(
    features_cfg,
    charge_cfg,
    neighbor_types: Sequence[int],
    reference_energies: torch.Tensor,
) -> ChargeAwareModel:
    """Construct a :class:`ChargeAwareModel` from config objects.

    ``features_cfg`` supplies the featurizer hyperparameters, ``charge_cfg`` the
    embedding / head / SCF hyperparameters (see ``train.config.ChargeConfig``).
    """
    lambdas = tuple(int(v) for v in features_cfg.selected_lambdas)
    if charge_cfg.dipole_head and 1 not in lambdas:
        raise ValueError(
            "the atomic dipole head needs lambda=1 features: add 1 to "
            "features.selected_lambdas"
        )
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        selected_lambdas=lambdas,
        backend=features_cfg.backend,
        density_channels=features_cfg.density_channels,
        charge_channels=charge_cfg.charge_channels,
    )
    n_species = len(neighbor_types)
    embedding = ChargeEmbedding(
        n_species, charge_cfg.emb_dim, q_hidden=charge_cfg.q_hidden
    )
    weight_net = AtomWeightNet(
        charge_cfg.emb_dim, charge_cfg.charge_channels, hidden=charge_cfg.weight_hidden
    )
    element_energy = ElementChargeEnergy(n_species, eta_init=charge_cfg.eta_init)
    p0 = featurizer.feature_dims[0]
    p2 = featurizer.feature_dims[2]
    energy_head = ChargeEnergyHead(
        p0, charge_cfg.emb_dim, hidden=charge_cfg.hidden, depth=charge_cfg.depth
    )
    alpha_head = AtomicAlphaHead(
        p0, p2, charge_cfg.emb_dim,
        featurizer.backend.irrep6_to_voigt(),
        hidden=charge_cfg.alpha_hidden,
        depth=charge_cfg.alpha_depth,
        equiv_channels=charge_cfg.alpha_equiv_channels,
        positive_isotropic=charge_cfg.alpha_positive_isotropic,
    )
    dipole_head = None
    if charge_cfg.dipole_head:
        p1 = featurizer.feature_dims[1]
        dipole_head = AtomicVectorHead(
            p0, p1, charge_cfg.emb_dim,
            hidden=charge_cfg.alpha_hidden,
            depth=charge_cfg.alpha_depth,
            equiv_channels=charge_cfg.alpha_equiv_channels,
        )
    return ChargeAwareModel(
        featurizer, embedding, weight_net, element_energy, energy_head,
        alpha_head, dipole_head, reference_energies,
        scf_max_iter=charge_cfg.scf_max_iter,
        scf_tol=charge_cfg.scf_tol,
        scf_damping=charge_cfg.scf_damping,
        attach_steps=charge_cfg.attach_steps,
        hessian_in_graph=charge_cfg.hessian_in_graph,
    )
