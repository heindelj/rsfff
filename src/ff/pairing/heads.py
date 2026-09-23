"""The pairing family's heads: valence polytensor, pairing exponent, hardness, valence capacity.

The bare pairing energy ``J_ij`` is the Pauli operator applied to *valence* multipoles, so the
head that emits them is literally :class:`rsfff.ff.pauli.PauliMultipoleHeads` with different
priors: the monopole prior is the valence count (O: 2, H: 1), the exponent prior is what puts
the water O-H minimum at the pyCMM ``r_eq`` with the pyCMM well depth against the pyCMM Pauli
repulsion (``docs/fff_pairing.md`` §4), and the dipole/quadrupole heads are zero-initialized
so the rank-0 model is the initial model exactly.

``kappa`` (the pairing hardness) is a positive per-atom scalar in log space, per-species prior
plus a zero-initialized readout of the family latent -- the same construction as ``q`` and
``b``.

The valence capacity ``v`` is charge dependent. Its prior puts the fragment's formal charge on
its heavy atoms::

    v_i = v_0(Z_i) + Q_f w_i           w_i = [i heavy] / n_heavy(f)

so the oxygen of H3O+ can form three bonds (3), that of OH- one (1), and the two oxygens of a
Zundel cation taken as one fragment get 2.5 each -- which is exactly the pairing a shared
proton needs. A fragment without heavy atoms puts ``-|Q|`` on its hydrogens (a bare proton or
hydride pairs nothing). On top of the prior a zero-initialized readout of the family latent
(FiLM-conditioned on the fragment state, so it can tell H3O+ from H2O) scales it as
``v = v_prior exp(delta)``. That readout is the reactive-phase extension point: when the
fragment labels go, ``v`` has to come from the local electron count instead (``docs/fff_pairing.md`` §6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.heads import mlp, zero_init_readout
from ..pauli import PauliMultipoleHeads
from .bond_order import valence_table
from .reference import formal_charge_share

__all__ = ["PairingFamily", "PairingHeads", "DEFAULT_PAIRING_PRIOR", "build_pairing_priors"]

#: Per-element ``(b_pair [1/bohr], kappa [Ha])`` priors. With the monopole prior equal to the
#: valence count these reproduce, against the pyCMM Pauli priors and with the Pauli multipoles
#: at rank 0, an O-H minimum at r_eq = 1.812 bohr with D = 0.1997 Ha (the pyCMM water Morse;
#: the curvature comes out at 0.37 Ha/bohr^2 against Morse's 0.54, which the network absorbs).
#: One exponent for both elements: the calibration fixes only the pair value.
DEFAULT_PAIRING_PRIOR: dict[int, tuple[float, float]] = {
    8: (0.834, 0.2),
    1: (0.834, 0.2),
}
GENERIC_PAIRING_PRIOR: tuple[float, float] = (0.834, 0.2)


def build_pairing_priors(
    neighbor_types,
    *,
    valence: dict[int, float] | None = None,
    b_prior: dict[int, float] | None = None,
    kappa_prior: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(valence, log q_v, log b_pair, log kappa)`` per species, ordered like ``neighbor_types``."""
    v = valence_table(neighbor_types, valence)
    table = dict(DEFAULT_PAIRING_PRIOR)
    for z in neighbor_types:
        table.setdefault(int(z), GENERIC_PAIRING_PRIOR)
    if b_prior:
        table = {z: (float(b_prior.get(z, b)), k) for z, (b, k) in table.items()}
    if kappa_prior:
        table = {z: (b, float(kappa_prior.get(z, k))) for z, (b, k) in table.items()}
    log_b = torch.tensor([table[int(z)][0] for z in neighbor_types]).log()
    log_kappa = torch.tensor([table[int(z)][1] for z in neighbor_types]).log()
    # a closed-shell atom (valence 0) keeps a tiny monopole so its log is finite; its pairs
    # then carry no energy to speak of and the valence marginal forbids pairing anyway
    log_q = v.clamp(min=1.0e-3).log()
    return v, log_q, log_b, log_kappa


@dataclass
class PairingFamily:
    """The pairing parameters at one evaluation.

    ``q``, ``mu``, ``quad_s`` : the valence polytensor (monopole ``(N,)``, dipole ``(N, 3)``,
                                spherical quadrupole ``(N, 5)``), in the Pauli conventions.
    ``b``                     : (N,) pairing exponent, 1/bohr.
    ``kappa``                 : (N,) pairing hardness, Hartree.
    ``valence``               : (N,) valence capacity.
    """

    q: torch.Tensor
    b: torch.Tensor
    mu: torch.Tensor | None
    quad_s: torch.Tensor | None
    kappa: torch.Tensor
    valence: torch.Tensor


class PairingHeads(nn.Module):
    """``(q_v, b, mu_v, quad_v, kappa, v)`` from the pairing-family latent."""

    def __init__(
        self,
        latent_dim: int,
        p1: int | None,
        n_species: int,
        *,
        valence: torch.Tensor,            # (n_species,)
        log_q_prior: torch.Tensor,        # (n_species,)
        log_b_prior: torch.Tensor,        # (n_species,)
        log_kappa_prior: torch.Tensor,    # (n_species,)
        dipole_scale: torch.Tensor,       # (n_species,)
        p2: int | None = None,
        quad_scale: torch.Tensor | None = None,
        irrep2_to_spherical: torch.Tensor | None = None,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        max_rank: int = 1,
        environment_q: bool = True,
        environment_b: bool = True,
        environment_kappa: bool = True,
        learn_kappa: bool = True,
        environment_valence: bool = True,
        heavy: torch.Tensor | None = None,     # (n_species,) bool: Z > 1
    ) -> None:
        super().__init__()
        self.multipoles = PauliMultipoleHeads(
            latent_dim, p1, n_species,
            log_q_prior=log_q_prior, log_b_prior=log_b_prior, dipole_scale=dipole_scale,
            p2=p2, quad_scale=quad_scale, irrep2_to_spherical=irrep2_to_spherical,
            emb_dim=emb_dim, hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            max_rank=max_rank, environment_q=environment_q, environment_b=environment_b,
        )
        self.register_buffer("valence_prior_table", valence.clone())
        self.register_buffer("log_kappa_prior", log_kappa_prior.clone())
        self.d_log_kappa = nn.Parameter(torch.zeros(n_species), requires_grad=learn_kappa)
        self.kappa_mlp = (
            zero_init_readout(mlp(latent_dim + emb_dim, hidden, depth, 1))
            if environment_kappa else None
        )
        self.valence_mlp = (
            zero_init_readout(mlp(latent_dim + emb_dim, hidden, depth, 1))
            if environment_valence else None
        )
        if heavy is None:
            heavy = torch.ones(n_species, dtype=torch.bool)
        self.register_buffer("heavy", heavy.clone())

    def valence_prior(
        self,
        species_idx: torch.Tensor,        # (N,)
        fragment_idx: torch.Tensor,       # (N,)
        fragment_charge: torch.Tensor,    # (F,)
    ) -> torch.Tensor:
        """``(N,)`` the charge-dependent valence prior of the class docstring."""
        v0 = self.valence_prior_table[species_idx]
        q_share, has_heavy = formal_charge_share(
            self.heavy[species_idx], fragment_idx, fragment_charge, dtype=v0.dtype
        )
        v = torch.where(has_heavy, v0 + q_share, v0 - q_share.abs())
        return v.clamp(min=0.0)

    @property
    def max_rank(self) -> int:
        return self.multipoles.max_rank

    def forward(
        self,
        z: torch.Tensor,                          # (N, latent)
        species_idx: torch.Tensor,                # (N,)
        vec_feats: torch.Tensor | None = None,    # (N, 3, p1)
        equiv_feats: torch.Tensor | None = None,  # (N, 5, p2)
        *,
        fragment_idx: torch.Tensor | None = None,     # (N,)
        fragment_charge: torch.Tensor | None = None,  # (F,)
    ) -> PairingFamily:
        q, b, mu, quad_s = self.multipoles(z, species_idx, vec_feats, equiv_feats)
        emb = self.multipoles.species_emb(species_idx)
        x = torch.cat((z, emb), dim=-1)
        log_kappa = self.log_kappa_prior[species_idx] + self.d_log_kappa[species_idx]
        if self.kappa_mlp is not None:
            log_kappa = log_kappa + self.kappa_mlp(x).squeeze(-1)
        if fragment_idx is None or fragment_charge is None:
            valence = self.valence_prior_table[species_idx]
        else:
            valence = self.valence_prior(species_idx, fragment_idx, fragment_charge)
        if self.valence_mlp is not None:
            valence = valence * self.valence_mlp(x).squeeze(-1).exp()
        return PairingFamily(
            q=q, b=b, mu=mu, quad_s=quad_s,
            kappa=log_kappa.exp(),
            valence=valence,
        )
