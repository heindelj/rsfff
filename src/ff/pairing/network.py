"""The conditioned parameter network with the bonded family swapped for the pairing family.

Everything the film network does -- block embedders, shared conditioned trunk, per-family
FiLM adapters, the two-evaluation ``theta`` / ``theta_0`` convention -- is inherited
unchanged. The only differences are the family list (``pairing`` in place of ``bonded``) and
the head that reads that family's latent. ``env_shift`` reports the pairing parameters'
``|theta - theta_0|`` so the training loop's environment penalty sees them by name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..film.conditioning import ConditionedTrunk  # noqa: F401  (re-export for builders)
from ..film.network import ConditionedParameterNetwork, FilmParameters, _log_shift
from ..film.projector import ProjectedFeatures
from ..film.state import StateDescriptor
from .heads import PairingFamily, PairingHeads

__all__ = ["PAIRING_FAMILIES", "PairingParameterNetwork", "PairingParameters"]

PAIRING_FAMILIES = ("pairing", "permanent", "response", "pauli", "disp")


@dataclass
class PairingParameters(FilmParameters):
    """:class:`FilmParameters` with the pairing family in place of the bonded one.

    ``bonded``/``bonded0`` are ``None``: there is no assigned bonded potential. The training
    loop's geometry-independence regularizer reads ``bonded0.delta_iso`` when present and
    skips it otherwise.
    """

    pairing: PairingFamily | None = None
    pairing0: PairingFamily | None = None

    def env_shift(self) -> dict[str, torch.Tensor]:
        out = {
            "eta": _log_shift(self.response.eta, self.response0.eta),
            "compliance": _log_shift(
                self.response.compliance.clamp(min=1e-12),
                self.response0.compliance.clamp(min=1e-12),
            ),
            "pauli_q": _log_shift(self.pauli[0], self.pauli0[0]),
            "pauli_b": _log_shift(self.pauli[1], self.pauli0[1]),
            "c6": _log_shift(self.disp[0], self.disp0[0]),
            "b_disp": _log_shift(self.disp[1], self.disp0[1]),
            "pair_q": _log_shift(self.pairing.q, self.pairing0.q),
            "pair_b": _log_shift(self.pairing.b, self.pairing0.b),
            "kappa": _log_shift(self.pairing.kappa, self.pairing0.kappa),
            "valence": (self.pairing.valence - self.pairing0.valence).abs(),
        }
        if self.response.alpha is not None:
            out["alpha"] = (self.response.alpha - self.response0.alpha).flatten(1).norm(dim=-1)
        if self.pauli[2] is not None:
            out["pauli_mu"] = (self.pauli[2] - self.pauli0[2]).norm(dim=-1)
        if self.pauli[3] is not None:
            out["pauli_quad"] = (self.pauli[3] - self.pauli0[3]).norm(dim=-1)
        if self.pairing.mu is not None:
            out["pair_mu"] = (self.pairing.mu - self.pairing0.mu).norm(dim=-1)
        if self.pairing.quad_s is not None:
            out["pair_quad"] = (self.pairing.quad_s - self.pairing0.quad_s).norm(dim=-1)
        return out


class PairingParameterNetwork(ConditionedParameterNetwork):
    """Block embedders + shared trunk + per-family adapters + the family heads, pairing edition."""

    def __init__(self, *, pairing_heads: PairingHeads, **kwargs) -> None:
        kwargs.setdefault("families", PAIRING_FAMILIES)
        super().__init__(bonded_head=None, **kwargs)
        self.pairing_heads = pairing_heads

    def forward(  # type: ignore[override]
        self,
        pf: ProjectedFeatures,
        c: torch.Tensor | None,
        state: StateDescriptor,
        positions: torch.Tensor,
        bond_index: torch.Tensor,
    ) -> PairingParameters:
        species_idx = pf.x_in.species_idx
        x_iso = self.embed_in(pf.x_in.inv_feats)
        if getattr(self, "sync_free", False) or bool(pf.a_env.any()):
            x_joined = (
                x_iso + self.embed_env(pf.x_env.inv_feats) + self.embed_cross(pf.cross_inv)
            )
            z_iso = self._family_latents(x_iso, c)
            z_joined = self._family_latents(x_joined, c)
        else:
            z_iso = self._family_latents(x_iso, c)
            z_joined = z_iso
        gate = self.gate(pf.a_env)

        frag = dict(fragment_idx=state.fragment_idx, fragment_charge=state.fragment_charge)
        pairing = self.pairing_heads(
            z_joined["pairing"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats, **frag
        )
        pairing0 = self.pairing_heads(
            z_iso["pairing"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats, **frag
        )

        q_perm, mu_perm, quad_perm = self.permanent_heads(
            z_iso["permanent"], pf.x_in, state
        )
        response = self.response_heads(
            z_joined["response"], pf.x_in, species_idx, positions, bond_index
        )
        response0 = self.response_heads(
            z_iso["response"], pf.x_in, species_idx, positions, bond_index
        )
        pauli = self.pauli_heads(
            z_joined["pauli"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats
        )
        pauli0 = self.pauli_heads(
            z_iso["pauli"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats
        )
        disp = self.disp_heads(z_joined["disp"], species_idx)
        disp0 = self.disp_heads(z_iso["disp"], species_idx)

        return PairingParameters(
            bonded=None, bonded0=None,
            q_perm=q_perm, mu_perm=mu_perm, quad_perm=quad_perm,
            response=response, response0=response0,
            pauli=pauli, pauli0=pauli0,
            disp=disp, disp0=disp0,
            gate=gate,
            pairing=pairing, pairing0=pairing0,
        )
