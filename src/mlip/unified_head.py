"""One pair trunk, several energy channels: the shared short-range correction.

:class:`rsfff.mlip.pair_heads.PairEnergyHead` is one network per energy term, each with its
own copy of the same symmetrization and the same radial basis. That is fine while the terms
are fit separately, and wrong once they are fit together: a change in an atom's environment
should move the electrostatic correction, the Pauli correction, the dispersion correction
*and* the bond energy at once, because physically it is one change. Four independent networks
can only learn that coupling four times, from four separate gradients.

So the featurization and the body are shared and only the readout is per channel::

    u_ij   = trunk([h_i + h_j, |h_i - h_j|, bessel(r)])
    dE_c   = scale_c * W_c(u_ij) * envelope(r; r_on_c, r_off_c)

This is the mechanism ``docs/range_separated_mlip.md`` §4.6 is describing when it says the
correction heads absorb what the force field hands over: they see one representation of the
pair, and the split between them is a linear readout rather than four disjoint models.

**Per-channel scale and envelope are not optional.** A covalent bond energy is ~0.2 Hartree
and an intermolecular correction ~1e-3, two hundred times smaller. Sharing one output scale
would let the bond channel's gradient set the effective learning rate for all of them; the
per-channel ``energy_scale`` is what keeps them comparable, exactly as the single-channel
head's docstring argues. The envelopes differ for the same reason -- the bond channel has to
reach the 0.96 and 1.5 Angstrom intramolecular distances, the corrections operate at 4-5.

Every readout is zero-initialized, so at initialization the model is the pure force field and
training starts from it. Compact support (:func:`rsfff.mlip.switch.pairwise_switch`, exactly
zero at ``r_off`` with C2 forces) is what stops a correction channel from quietly taking over
the long-range tail that the classical form exists to provide -- the gauge leakage of §7.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import BesselBasis
from .switch import pairwise_switch


@dataclass(frozen=True)
class ChannelSpec:
    """Output scale and envelope for one correction channel.

    r_on, r_off  : envelope window in Angstrom. ``dE`` is full strength below ``r_on`` and
                   exactly zero at and beyond ``r_off``. ``r_off`` should not exceed the
                   feature cutoff -- past it the atomic features no longer see each other and
                   the channel would be extrapolating.
    energy_scale : output scale in Hartree, set to the size of the thing being corrected.
    """

    r_on: float
    r_off: float
    energy_scale: float

    def __post_init__(self) -> None:
        if not self.r_off > self.r_on:
            raise ValueError(
                f"ChannelSpec needs r_off > r_on, got {self.r_on}, {self.r_off}"
            )
        if not self.energy_scale > 0.0:
            raise ValueError(f"ChannelSpec needs energy_scale > 0, got {self.energy_scale}")


class UnifiedPairHead(nn.Module):
    """Shared pair trunk with one zero-initialized energy readout per channel.

    Args
    ----
    p0        : width of the invariant feature vector handed in. This is the *augmented*
                width -- :class:`rsfff.ff.unified.UnifiedPairModel` appends its fragment-state
                block before calling -- not ``featurizer.feature_dims[0]``.
    n_species : embedding table size. The species embedding is **not** optional: ``inv_feats``
                is the power spectrum of the *neighbor* density and does not encode the center
                atom's element, so without it the head cannot tell an O-H pair from an H-H
                pair at the same distance in an otherwise degenerate environment.
    channels  : ``{name: ChannelSpec}``. Insertion order fixes nothing; look results up by
                name.
    n_radial  : Bessel radial basis size for the pair distance. One basis is shared, so its
                cutoff is the largest ``r_off`` over the channels.

    Forward returns ``{name: (P,)}`` in Hartree.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        channels: dict[str, ChannelSpec],
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        n_radial: int = 8,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("UnifiedPairHead needs at least one channel")
        self.channels = dict(channels)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        # One radial basis for the shared trunk, so it has to span the widest channel. The
        # per-channel envelopes below still cut each channel off at its own r_off.
        self.radial = BesselBasis(n_radial, max(c.r_off for c in channels.values()))

        h = p0 + emb_dim
        layers: list[nn.Module] = []
        d = 2 * h + n_radial
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.readout = nn.ModuleDict(
            {name: nn.Linear(hidden, 1) for name in self.channels}
        )
        with torch.no_grad():
            for lin in self.readout.values():
                lin.weight.zero_()
                lin.bias.zero_()

    def forward(
        self,
        inv_feats: torch.Tensor,      # (N, p0)
        species_idx: torch.Tensor,    # (N,)
        pair_index: torch.Tensor,     # (2, P)
        r: torch.Tensor,              # (P,) Angstrom
    ) -> dict[str, torch.Tensor]:
        i, j = pair_index[0], pair_index[1]
        h = torch.cat((inv_feats, self.species_emb(species_idx)), dim=-1)
        h_i, h_j = h[i], h[j]
        # sum and |difference| are both invariant under swapping i and j, so an undirected
        # i<j pair list is unambiguous and no channel can depend on the storage order.
        u = self.trunk(torch.cat((h_i + h_j, (h_i - h_j).abs(), self.radial(r)), dim=-1))
        return {
            name: spec.energy_scale
            * self.readout[name](u).squeeze(-1)
            * pairwise_switch(r, spec.r_on, spec.r_off)
            for name, spec in self.channels.items()
        }


__all__ = ["ChannelSpec", "UnifiedPairHead"]
