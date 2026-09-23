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

The valence capacity is not a head at all any more: it is ``v(n)`` of the atom's formal
electron count ``n``, which the joint solve of :mod:`rsfff.ff.pairing.electronic_state`
determines from the geometry (capacity polynomial per element, the charged atomic reference
``chi`` / ``eta`` per element). What the head still owns is a zero-initialized scale on the
neutral capacity ``v0`` -- the one place the network can bend the octet rule -- and the
per-element tables it hands to the solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.heads import mlp, zero_init_readout
from ..pauli import PauliMultipoleHeads
from .bond_order import valence_table
from .electronic_state import capacity_table

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
    ``tables``                : (N, 5) capacity polynomial ``(n0, v0, a, b, shell)`` per atom,
                                ``v0`` carrying the learned scale.
    ``chi``, ``eta``          : (N,) the charged atomic reference's (IP + EA)/2 and IP - EA.
    """

    q: torch.Tensor
    b: torch.Tensor
    mu: torch.Tensor | None
    quad_s: torch.Tensor | None
    kappa: torch.Tensor
    tables: torch.Tensor
    chi: torch.Tensor
    eta: torch.Tensor

    @property
    def valence(self) -> torch.Tensor:
        """The neutral capacity ``v0`` per atom (what the film-style diagnostics read)."""
        return self.tables[:, 1]


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
        capacity: torch.Tensor | None = None,  # (n_species, 5) from capacity_table
        chi: torch.Tensor | None = None,       # (n_species,)
        eta: torch.Tensor | None = None,       # (n_species,)
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
        if capacity is None:
            capacity = torch.zeros(n_species, 5)
            capacity[:, 1] = valence
            capacity[:, 4] = 8.0
        self.register_buffer("capacity", capacity.clone())
        zeros = torch.zeros(n_species)
        self.register_buffer("chi", zeros.clone() if chi is None else chi.clone())
        self.register_buffer("eta", zeros.clone() if eta is None else eta.clone())

    @property
    def max_rank(self) -> int:
        return self.multipoles.max_rank

    def forward(
        self,
        z: torch.Tensor,                          # (N, latent)
        species_idx: torch.Tensor,                # (N,)
        vec_feats: torch.Tensor | None = None,    # (N, 3, p1)
        equiv_feats: torch.Tensor | None = None,  # (N, 5, p2)
    ) -> PairingFamily:
        q, b, mu, quad_s = self.multipoles(z, species_idx, vec_feats, equiv_feats)
        emb = self.multipoles.species_emb(species_idx)
        x = torch.cat((z, emb), dim=-1)
        log_kappa = self.log_kappa_prior[species_idx] + self.d_log_kappa[species_idx]
        if self.kappa_mlp is not None:
            log_kappa = log_kappa + self.kappa_mlp(x).squeeze(-1)
        tables = self.capacity[species_idx]
        if self.valence_mlp is not None:
            scale = self.valence_mlp(x).squeeze(-1).exp()
            tables = torch.cat((tables[:, :1], tables[:, 1:2] * scale.unsqueeze(-1), tables[:, 2:]), dim=-1)
        return PairingFamily(
            q=q, b=b, mu=mu, quad_s=quad_s,
            kappa=log_kappa.exp(),
            tables=tables, chi=self.chi[species_idx], eta=self.eta[species_idx],
        )
