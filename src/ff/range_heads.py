"""Per-atom ``r0`` and per-channel ``alpha`` for the Fermi range separation.

Split out of :mod:`rsfff.ff.unified` when the two-slot parameterization arrived: it is a
parameter head like the ones in :mod:`rsfff.ff.response`, :mod:`rsfff.ff.dispersion` and
:mod:`rsfff.ff.pauli`, and it belongs beside them rather than inside the model that assembles
them. ``rsfff.ff.v1`` keeps its own frozen copy, so this move cannot disturb
``checkpoints/water_staged/best.pt``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import two_slot_mlp, zero_init_readout
from .range_priors import RANGE_CHANNELS

__all__ = ["RangeSeparationHeads"]


class RangeSeparationHeads(nn.Module):
    """Per-atom ``r0`` and per-channel ``alpha`` for the Fermi range separation.

    ``r0`` is per **atom** and combined across a pair as the geometric mean, the same
    log-space rule every other pair parameter here uses. A per-atom ``r0`` cannot distinguish
    an intramolecular O-H from an intermolecular one -- it is the same hydrogen in both -- and
    does not have to: that discrimination comes from ``r``, which is what a range separation
    is for. The parameter only sets *where* the handoff sits for an element pair in a channel.

    One ``r0`` and one ``alpha`` per channel, because the channels are not descriptions of
    equal fidelity. The classical electrostatics stays valid to shorter range than the
    Tang-Toennies dispersion does, so their handoff points genuinely differ, and a single
    shared parameter would impose the worst channel's handoff on all of them. They start from
    the same prior only because the measurement in :mod:`rsfff.ff.range_priors` constrains
    where the *bonded* region ends, which is common to all three; the fit is free to separate
    them.

    ``environment_r0`` defaults off, matching the treatment of the other damping exponents
    (``b``, ``z``): a range separation that varies with the environment competes directly with
    the pair correction for the same mid-range energy, so the first fit should move only the
    per-element values. The MLP is zero-initialized, so turning it on starts from exactly the
    per-element result.

    ``alpha`` is kept positive by ``softplus`` rather than by a runtime check, which is the
    guarantee :func:`rsfff.ff.damping.fermi_switch` relies on when handed a tensor.

    Under the two slots this head is called **twice**: once on the joined descriptor for the
    inter-fragment gate and once on the isolated one for the intra gate. Only ``r0_mlp`` can
    differ between the two -- the per-element table and ``alpha`` are shared -- so with
    ``environment_r0`` off the two calls return identical tensors and the caller skips the
    second. That is the same short circuit the previous model had; what changed is that the
    isolated call is now ``theta_0`` by construction rather than by passing a different vector.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        log_r0_prior: torch.Tensor,          # (n_channels, n_species), rows like `channels`
        alpha_init: float,
        p_env: int = 0,
        channels: tuple[str, ...] = RANGE_CHANNELS,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        learn_r0: bool = True,
        environment_r0: bool = False,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__()
        if not alpha_init > 0.0:
            raise ValueError(f"RangeSeparationHeads needs alpha_init > 0, got {alpha_init}")
        self.channel_names = tuple(channels)
        if log_r0_prior.dim() == 1:      # one row broadcast to every channel
            log_r0_prior = log_r0_prior.expand(len(self.channel_names), -1)
        if log_r0_prior.shape[0] != len(self.channel_names):
            raise ValueError(
                f"log_r0_prior has {log_r0_prior.shape[0]} rows for "
                f"{len(self.channel_names)} channels {self.channel_names}; the dispersion "
                f"prior differs from the others (see rsfff.ff.range_priors.CHANNEL_R0_PRIOR) "
                f"so the rows are not interchangeable"
            )
        self.register_buffer("log_r0_prior", log_r0_prior.clone().contiguous())
        self.species_emb = nn.Embedding(n_species, emb_dim)
        alpha_raw = float(torch.log(torch.expm1(torch.tensor(float(alpha_init)))))

        self.d_log_r0 = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(n_species), requires_grad=learn_r0)
                for name in self.channel_names
            }
        )
        self.alpha_raw = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(alpha_raw), requires_grad=learn_alpha)
                for name in self.channel_names
            }
        )
        self.r0_mlp = None
        if environment_r0:
            self.r0_mlp = nn.ModuleDict(
                {
                    name: two_slot_mlp(p0, p_env, hidden, depth, 1, p_tail=emb_dim)
                    for name in self.channel_names
                }
            )
            for m in self.r0_mlp.values():   # start at exactly the per-element value
                zero_init_readout(m)

    def alphas(self) -> dict[str, torch.Tensor]:
        """``{channel: alpha () Angstrom^-1}`` on its own, without evaluating ``r0``.

        ``alpha`` has no atom axis, so unlike ``r0`` it cannot be gathered per expert and
        stitched back. A model with several experts reads it from one of them and ties the
        parameters so that "one of them" is not a choice --
        :func:`rsfff.train.build_expert.build_expert_model` does the tying and says why.
        """
        return {
            name: torch.nn.functional.softplus(self.alpha_raw[name])
            for name in self.channel_names
        }

    def forward(
        self,
        inv_feats: torch.Tensor,     # (N, p0)
        species_idx: torch.Tensor,   # (N,)
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """``({channel: r0 (N,) Angstrom}, {channel: alpha () Angstrom^-1})``."""
        s = species_idx
        x = None
        if self.r0_mlp is not None:
            x = torch.cat((inv_feats, self.species_emb(s)), dim=-1)
        r0, alpha = {}, {}
        for c, name in enumerate(self.channel_names):
            log_r0 = self.log_r0_prior[c][s] + self.d_log_r0[name][s]
            if self.r0_mlp is not None:
                log_r0 = log_r0 + self.r0_mlp[name](x).squeeze(-1)
            r0[name] = log_r0.exp()
            alpha[name] = torch.nn.functional.softplus(self.alpha_raw[name])
        return r0, alpha
