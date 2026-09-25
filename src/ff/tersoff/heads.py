"""The Tersoff family's heads: the pairing heads plus a formal-charge readout.

Everything the pairing model emits -- valence polytensor, pairing exponent, hardness, the
capacity tables with their learned scale -- is inherited unchanged. What is added is ``q~``,
a zero-initialized per-atom readout of the family latent that :mod:`formal_charge` projects
onto the frame's total charge; at initialization it is zero and the heavy-atom prior of the
projection decides the capacities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...mlip.heads import mlp, zero_init_readout
from ..pairing.heads import PairingFamily, PairingHeads

__all__ = ["TersoffFamily", "TersoffHeads"]


@dataclass
class TersoffFamily(PairingFamily):
    """:class:`PairingFamily` plus the raw formal-charge readout ``q_raw`` (N,), zero when the
    head is off."""

    q_raw: torch.Tensor | None = None


class TersoffHeads(PairingHeads):
    """``PairingHeads`` + ``q_raw``. ``formal_charge_readout=False`` pins ``q_raw = 0`` (the
    projection prior alone decides the formal charges)."""

    def __init__(self, *args, formal_charge_readout: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        emb_dim = self.multipoles.species_emb.embedding_dim
        latent_dim = args[0]
        hidden = kwargs.get("hidden", 64)
        depth = kwargs.get("depth", 2)
        self.q_mlp = (
            zero_init_readout(mlp(latent_dim + emb_dim, hidden, depth, 1))
            if formal_charge_readout else None
        )

    def forward(self, z, species_idx, vec_feats=None, equiv_feats=None) -> TersoffFamily:
        fam = super().forward(z, species_idx, vec_feats, equiv_feats)
        if self.q_mlp is not None:
            emb = self.multipoles.species_emb(species_idx)
            q_raw = self.q_mlp(torch.cat((z, emb), dim=-1)).squeeze(-1)
        else:
            q_raw = fam.q.new_zeros(fam.q.shape[0])
        return TersoffFamily(**fam.__dict__, q_raw=q_raw)
