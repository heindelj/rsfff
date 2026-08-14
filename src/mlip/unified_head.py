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
        range_channels: tuple[str, ...] = (),
        extra_dim: int = 0,
        max_log_dev: float = 0.7,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        n_radial: int = 8,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("UnifiedPairHead needs at least one channel")
        self.channels = dict(channels)
        self.range_channels = tuple(range_channels)
        self.extra_dim = int(extra_dim)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        # One radial basis for the shared trunk, so it has to span the widest channel. The
        # per-channel envelopes below still cut each channel off at its own r_off.
        self.radial = BesselBasis(n_radial, max(c.r_off for c in channels.values()))

        h = p0 + emb_dim
        layers: list[nn.Module] = []
        d = 2 * h + n_radial + self.extra_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.readout = nn.ModuleDict(
            {name: nn.Linear(hidden, 1) for name in self.channels}
        )
        # Per-pair range-separation deviation, in log space so it composes with the per-atom
        # geometric mean by addition. This is what a per-atom r0 cannot express: an ethane
        # geminal H-H (1.78 A) and an H-H across the C-C bond (2.27 A) are the same element
        # pair, so one threshold cannot put the first in the bonding term and leave the second
        # to the classical form. The trunk sees both atoms' environments and can tell that the
        # first pair shares a bonded carbon.
        #
        # Bounded by `max_log_dev * tanh(.)`, so the element prior stays the anchor and the
        # network supplies a correction within a fixed factor of it. Unbounded, an exponential
        # of an MLP output is a runaway: a perturbed head produced r0 of 1533 Angstrom in
        # testing, which switches the classical form off everywhere and cannot recover,
        # because past saturation the gate has no gradient to come back on.
        self.max_log_dev = float(max_log_dev)
        self.range_readout = nn.ModuleDict(
            {name: nn.Linear(hidden, 1) for name in self.range_channels}
        )
        with torch.no_grad():
            for lin in list(self.readout.values()) + list(self.range_readout.values()):
                lin.weight.zero_()
                lin.bias.zero_()

    def forward(
        self,
        inv_i: torch.Tensor,          # (P, p0) invariants of each pair's first atom
        inv_j: torch.Tensor,          # (P, p0) ... and its second
        species_i: torch.Tensor,      # (P,)
        species_j: torch.Tensor,      # (P,)
        r: torch.Tensor,              # (P,) Angstrom
        extra: torch.Tensor | None = None,   # (P, extra_dim) pair-invariant side channel
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """``({channel: dE (P,)}, {channel: log-r0 deviation (P,)})``.

        Features arrive **already gathered onto pairs**, not as per-atom arrays.

        ``extra`` carries the electrostatic environment
        (:func:`rsfff.ff.environment.environment_pair_invariants`) when the caller wants the
        bond energy to respond to an external field. It must already be reduced to scalars that
        are both rotation-invariant and symmetric under swapping ``i`` and ``j`` -- the trunk
        below symmetrizes the *atomic* features but cannot fix an odd side channel, so a raw
        ``F_i . rhat`` would silently make the output depend on pair storage order. Passing
        zeros here is what defines the reference level each anchored correction is measured
        against, so the same head serves both.

        The caller does the gathering because it partitions the pair list and calls this once
        per part: :class:`rsfff.ff.unified.UnifiedPairModel` scores intra-fragment pairs from
        the fragment-confined descriptor (for the bond channel) and inter-fragment pairs from
        the environment-aware one (for the three interaction corrections). That keeps the
        1-body term matchable against an isolated-fragment label while the interactions carry
        many-body content, and because the parts are disjoint it costs one evaluation's worth
        of work in total. Gathering here would force one stream on every pair.
        """
        h_i = torch.cat((inv_i, self.species_emb(species_i)), dim=-1)
        h_j = torch.cat((inv_j, self.species_emb(species_j)), dim=-1)
        # sum and |difference| are both invariant under swapping i and j, so an undirected
        # i<j pair list is unambiguous and no channel can depend on the storage order.
        parts = [h_i + h_j, (h_i - h_j).abs(), self.radial(r)]
        if self.extra_dim:
            if extra is None:
                extra = r.new_zeros(r.shape[0], self.extra_dim)
            elif extra.shape[-1] != self.extra_dim:
                raise ValueError(
                    f"extra has width {extra.shape[-1]}, head was built for {self.extra_dim}"
                )
            parts.append(extra)
        u = self.trunk(torch.cat(parts, dim=-1))
        energies = {
            name: spec.energy_scale
            * self.readout[name](u).squeeze(-1)
            * pairwise_switch(r, spec.r_on, spec.r_off)
            for name, spec in self.channels.items()
        }
        log_r0_dev = {
            name: self.max_log_dev * torch.tanh(self.range_readout[name](u).squeeze(-1))
            for name in self.range_channels
        }
        return energies, log_r0_dev


__all__ = ["ChannelSpec", "UnifiedPairHead"]
