"""The per-atom ``(Q_f, 2S_f)`` block: which member of its family a fragment is.

Split out of :mod:`rsfff.ff.unified` for the fragment-expert model, where it has a sharper job
than it had before. Experts are keyed on **composition** alone (:mod:`rsfff.ff.expert`), so one
``"OH"`` expert covers hydroxide and the OH radical -- and this is what tells it which one it is
looking at. Without it those two are indistinguishable, and an H2O and an H3O+ differ precisely
in which fragment carries the charge.

It joins the **fragment** slot, never the environment one: charge and multiplicity are
properties of the fragment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import mlp

__all__ = ["FragmentStateEmbedding"]


class FragmentStateEmbedding(nn.Module):
    """Per-atom block encoding its fragment's ``(Q_f, 2S_f)``, zero at the neutral singlet.

    The featurizer every force-field term uses (``FlatLambdaSOAPFeaturizer``) carries species
    and geometry only -- no fragment charge, no multiplicity. On water that is invisible,
    because every fragment is a neutral singlet. On H5O2+ it is fatal: the two fragmentations
    differ precisely in which fragment carries the charge, an H2O and an H3O+ have very
    different internal energies, and with no fragment-level information the model can only
    tell them apart by geometry -- while an OH radical and a hydroxide are indistinguishable
    outright.

    The output is ``net(Q, 2S) - net(0, 0)``, so it is **identically zero for a neutral
    singlet no matter what the weights do**. That matters more than a zero-initialized
    readout would: on water-only data the input is the constant ``(0, 0)``, and a plain
    zero-init readout would still drift to some arbitrary constant that downstream biases
    absorb. Anchoring at the neutral reference means water training genuinely cannot move
    this block, and for a charged fragment the block reads as the *deviation from neutral*,
    which is the interpretable thing to condition on.

    Be clear about what this buys today: nothing. A constant-zero input receives no gradient,
    so the block is not trained until charged-fragment data arrives. It is here so that step
    is an addition rather than a retrofit of fragment-state awareness into the whole
    force-field stack at the same time as the fragmentation mixture.

    ``dim=0`` disables it entirely and is bit-identical to a model built without it.

    **Exempt from weight decay**, for the same reason :class:`EnvironmentResidual` is and with
    the same evidence. Anchoring makes the block's gradient a *difference*, water-only data
    holds its input at the constant ``(0, 0)`` so that difference is exactly zero, and decay is
    then unopposed: measured over a staged fit its first-layer weight norm went 3.17 -> 1.96 ->
    0.24 -> 9e-13. Nothing was wrong with that while the input stays constant -- but the whole
    point of the block is to be ready when charged-fragment data arrives, and a flattened block
    is not ready. Left alone it sits at its initialization instead, which is.
    """

    #: See the note above. Weight decay would flatten a block that water-only data cannot
    #: train, leaving nothing for H5O2+ data to start from.
    no_weight_decay = True

    def __init__(self, dim: int = 4, *, hidden: int = 32, depth: int = 1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = mlp(2, hidden, depth, self.dim) if self.dim else None

    def forward(self, batch, fragment_idx: torch.Tensor, dtype, device) -> torch.Tensor | None:
        if self.net is None:
            return None
        n_frag = int(batch.n_fragments)
        zeros = torch.zeros(n_frag, dtype=dtype, device=device)
        q = zeros if batch.fragment_charge is None else batch.fragment_charge.to(dtype)
        s = zeros if batch.fragment_two_s is None else batch.fragment_two_s.to(dtype)
        x = torch.stack((q, s), dim=-1)
        ref = self.net(torch.zeros_like(x[:1]))
        return (self.net(x) - ref)[fragment_idx]
