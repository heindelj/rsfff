"""One pair list, one trunk, four channels: the range-separated model without the mask.

Every pair in the system carries the classical backbone and the neural correction. How much
of the classical form is switched on is decided per pair and per channel by a **learned range
separation**; which training label a pair's energy is compared against is decided by the
fragmentation. Those are two different jobs and this module keeps them apart::

    gate_ij^c = fermi_switch(r, r0_ij^c, alpha^c) * taper(r; cut_c)
    E_ij^c    = gate_ij^c * e_classical^c  +  dE_ij^c

    E_elst  = sum over INTER pairs of E_ij^elst      ->  eda_cls_elec
    E_pauli = sum over INTER pairs of E_ij^pauli     ->  eda_mod_pauli
    E_disp  = sum over INTER pairs of E_ij^disp      ->  eda_disp
    E_frag  = E0 + E_internal + E_atom^0 + sum over INTRA pairs of ( E^elst + E^pauli + E^disp )
    E_total = sum_f E_frag(f) + E_elst + E_pauli + E_disp + E_pol + E_ct

Where the bonding energy lives
------------------------------
``E_atom^0`` is :class:`rsfff.ff.atomic_energy.AtomicStateEnergy`, a **per-atom** energy of the
self-consistent electronic state -- ``E_theta(h_i, M_i, Phi_i)`` in the notation of
``docs/atomic_response_functional.md``. It replaced a per-pair ``bond`` correction channel, and
the reason is worth keeping:

That readout was carrying three jobs at once. It was the covalent bond energy (~-566 kJ/mol per
water). It was the only free knob left inside ``fragment_energy`` once ``freeze_frozen_level``
pins ``E_internal``. And through a ``bond_pol``/``bond_ct`` telescoping it carried the whole of
``eda_pol`` and ``eda_ct``. Two of those three have no label, so the split between them drifted:
measured, a **constant -1.686 kJ/mol per fragment** appeared in ``fragment_energy``, identical
to three decimals across w2--w5, which the frame-pooled diagnostics drew as a size-dependent
offset. Alongside it, ``eda_ct`` came out **96.7% neural** -- -5.24 kJ/mol per fragment of
correction against -0.18 of classical.

The levels now telescope through the *state* rather than through three readouts::

    E_atom^0   = E_theta( h_frag , M^frozen , Phi^intra )    -> fragment_energy
    E_atom^pol = E_theta( h_frag , M^pol    , Phi^pol   )    -> eda_pol gets (pol - 0)
    E_atom^ct  = E_theta( h_env  , M^ct     , Phi^ct    )    -> eda_ct  gets (ct  - pol)

One set of weights, three electronic states. ``Phi^intra`` is gated down to intra-fragment
pairs, which is what keeps ``E_atom^0`` an isolated-fragment quantity while still letting an
atom feel its own molecule's field. The feature stream switches at **ct**, not pol: relaxing a
fragment's multipoles against its neighbours leaves it the same fragment, whereas letting charge
cross the boundary does not.

**Charge transfer gets no correction readout of its own**, permanently. It is the residue of
lifting the last constraint, so what it is entitled to is what it shares with electrostatics and
polarization -- the coupled solve's energy difference and the atomic energy's response to it.
If the pair corrections are switched back on, ``ct_bond`` does not come back with them.

Two things this buys over the per-term modules
----------------------------------------------
**Same-fragment pairs get real electrostatics.** :class:`rsfff.ff.onebody.OneBodyEnergy` is
``E0 + E_internal + bond_head``, and the bond head's envelope dies at 4 Angstrom, so two atoms
8 Angstrom apart in the same fragment interact not at all. That is invisible on water and
wrong for any fragment larger than a monomer. Here they interact classically, and the energy
is routed to ``fragment_energy`` because that is the label it belongs to.

**The intra/inter decision is differentiable in the geometry.** A proton mid-transfer at
1.4 Angstrom is genuinely half-bond and half-hydrogen-bond; the boolean mask in
:func:`rsfff.ff.pairs.inter_fragment_pairs` has to call it one or the other, and a smooth
switch does not.

Routing is not a modeling choice
--------------------------------
``eda_cls_elec`` *is defined* as the interaction between the given fragments and
``fragment_energy`` *is defined* as the energy of one fragment. So which sum a pair lands in
follows from what the labels mean, not from anything learned -- if a free network decided it,
it could move energy between the buckets at zero loss and all four targets would stop being
well-posed.

What routing is *not* is a no-op on the total. It is tempting to say that summing every
channel makes the partition invisible, and that is wrong on two counts, both measurable:

1. **The atomic energy is per fragment.** ``E_atom`` is summed over the atoms of a fragment
   and booked to that fragment's label, and the frozen EDA components carry no bonding term at
   all. So relabelling changes which bucket the bonding energy lands in.
2. **The descriptors are fragment-confined**, hence partition-dependent themselves. Relabelling
   a dimer as one fragment changes every atom's ``inv_feats`` (measured: max change 0.19), so
   every classical and correction energy moves too.

Both are correct rather than regrettable: relabelling asserts that two molecules are one, and
the model is supposed to answer differently. Point 1 in particular is where the eventual
charge-transfer energy should surface -- a pair transitioning from inter to intra picks up
exactly the bond-channel term whose growth is what CT looks like in this decomposition.

What *is* exact is the accounting, and ``tests/test_ff_unified.py`` pins it::

    energy == sum_i E0[Z_i] + sum_f ( E_internal(f) + E_atom(f) )
              + sum_{all pairs} sum_c ( e_ff^c + e_corr^c )

Every pair appears exactly once, in exactly one bucket: no double counting and no gap. That
is the invariant that has to survive when ``is_intra`` softens into a mixture weight.

Under multiple fragmentations the routing becomes ambiguous and softens into a mixture
weight. Nothing here hardcodes the single-fragmentation case: ``is_intra`` and ``pair_frag``
come out of :func:`rsfff.ff.pairs.union_pairs` as functions of the supplied assignment. (The
plan called for a size-1 fragmentation axis on every output tensor; that is a degenerate axis
that would obscure every shape for no present benefit, so the forward-compatibility lives in
where the routing is *computed* instead.)

Two feature streams, one split
-----------------------------
With :class:`EnvironmentResidual` attached, an atom carries two descriptors: ``h_frag``,
confined to its own fragment, and ``h_env = h_frag + g(h_full)``. Only **one** consumer needs
the fragment-confined one:

* ``h_frag`` -> the frozen response solve and ``E_atom`` at the frozen and polarized levels.
  Together those are ``fragment_energy`` up to a few thousandths of a kJ/mol, and that label is
  Q-Chem's *isolated*-fragment energy, which has no environment dependence to fit.
* ``h_env`` -> everything else: the interaction parameters (``C6``, the Slater exponents, the
  Pauli multipoles, ``r0``), the coupled solve's response parameters, and ``E_atom`` at the CT
  level. This is where effective many-body physics such as environment-quenched ``C6`` belongs.

**Pauli and dispersion read ``h_env`` on every pair, intra ones included.** An earlier version
evaluated those heads on *both* streams and selected per pair, so that an intra-fragment pair's
classical energy used isolated-fragment parameters. That bought exactness in
``fragment_energy`` at the price of making a bonded pair answer to a different network than a
hydrogen-bonded one, and it is no longer where the label's integrity comes from: the two terms
that carry ``fragment_energy`` -- ``E_internal`` and ``E_atom^0`` -- are fragment-confined *by
construction*, and the range separation switches the intra classical remainder down to well
under a kJ/mol per fragment.

So ``fragment_energy`` is exactly one-body in its two dominant terms and approximately so in
the remainder. ``tests/test_ff_unified.py`` asserts the first bitwise and bounds the second
rather than asserting it away.

What is deliberately not here
-----------------------------
The **variational** coupling of ``docs/atomic_response_functional.md`` §2, where ``E_atom``
sits inside the functional and ``dE_SCF/dM = 0`` includes it. Here ``E_atom`` is evaluated at
the *converged* multipoles, so the coupling runs one way (``M* -> E_atom``) and not back. That
is a deliberate first step, and it is safe because
:class:`rsfff.ff.coupled_solve._CoupledSolve`'s adjoint exists precisely to make a
non-variational consumer of ``M*`` differentiate correctly. Closing the loop would mean
replacing ``pcg``'s linear ``a_mul`` with a nonlinear root find and generalising ``A`` in the
adjoint from a constant operator to the Hessian at ``M*``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.heads import mlp, zero_init_readout
from ..mlip.switch import pairwise_switch
from ..mlip.unified_head import ChannelSpec, UnifiedPairHead
from .atomic_energy import AtomicStateEnergy
from .damping import fermi_switch
from .dispersion import DispersionParameterHeads, tt_damped_c6_energy
from .electrostatics import slater_elec_pair_energy
from .environment import OneBodyEnvironment, electrostatic_environment
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    spherical_to_cartesian_quadrupole,
)
from .pairs import union_channels, union_pairs
from .pauli import PauliMultipoleHeads, slater_pauli_pair_energy
from .polarization import LevelOutput, coupled_response
from .range_priors import RANGE_CHANNELS, build_range_priors
from .response import FragmentResponse, FragmentResponseOutput
from .units import BOHR_ANG


@dataclass(frozen=True)
class ClassicalSpec:
    """Neighbor-list reach of one classical channel, in Angstrom.

    ``cutoff`` is where the channel is tapered to exactly zero and ``taper_width`` how long
    that takes. These are per channel because the terms decay at wildly different rates: the
    electrostatics has a genuine ``1/r`` tail and truncating it early is a real error, while
    the Slater Pauli form is ~1e-8 Hartree by 7 Angstrom.
    """

    cutoff: float
    taper_width: float = 1.0

    def __post_init__(self) -> None:
        if not self.cutoff > self.taper_width:
            raise ValueError(
                f"ClassicalSpec needs cutoff > taper_width, got {self.cutoff}, "
                f"{self.taper_width}"
            )


#: Reach of each classical channel. Copied from the per-term modules so the unified model
#: starts from the same physics they were validated at.
DEFAULT_CLASSICAL: dict[str, ClassicalSpec] = {
    "elst": ClassicalSpec(12.0),
    "pauli": ClassicalSpec(7.0),
    "disp": ClassicalSpec(10.0),
}


def _resolve_correction_channels(spec, available) -> frozenset[str]:
    """``True``/``False``/a list of names -> the set of consumed correction channels.

    ``True`` means every channel the head was actually built with, so enabling a channel is
    always a matter of building its readout rather than of listing it twice.

    A name that the head has no readout for is an **error**, not a no-op. The two ways to get
    one are a typo and asking for ``ct_bond`` on a model built without it, and both are
    failures that would otherwise present as a channel that silently contributes nothing --
    which is indistinguishable from a channel that was fitted to zero.
    """
    available = set(available)
    if isinstance(spec, bool):
        return frozenset(available) if spec else frozenset()
    if isinstance(spec, str):
        raise TypeError(
            f"pair_corrections must be a bool or a list of channel names, got the bare "
            f"string {spec!r}; write [{spec!r}] to name one channel"
        )
    wanted = frozenset(str(name) for name in spec)
    unknown = wanted - available
    if unknown:
        raise ValueError(
            f"pair_corrections names channel(s) {sorted(unknown)} that the pair head has no "
            f"readout for; it was built with {sorted(available)}"
        )
    return wanted


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


class EnvironmentResidual(nn.Module):
    """``h_env = h_frag + g(h_full)``: what an atom's parameters learn from its surroundings.

    A fragment-confined descriptor cannot express environment screening -- an isolated water's
    C6 and a water's C6 inside a cluster are the same number by construction, and the
    dispersion term is then rigorously two-body with *exactly* zero many-body content
    (``tests/test_many_body.py``). That is a useful ablation and the wrong physics: effective
    atomic C6 really is quenched by the environment.

    Written as a **residual on the fragment-confined features** rather than as a straight swap
    to the environment-aware ones, for three reasons:

    * ``g`` is zero-initialized, so ``h_env == h_frag`` exactly at initialization and turning
      environment awareness on starts from the validated fragment-confined model rather than
      from a different one -- and it is written as a **difference**, ``g(h_full) - g(h_frag)``,
      so ``h_env == h_frag`` also holds exactly for an isolated fragment at *any* stage of
      training (see :meth:`forward`);
    * ``||g(h_full)||`` is then a direct measurable of how much many-body content the model
      actually wants, rather than something that has to be inferred from a fit quality
      difference;
    * both descriptors come from one neighbor search and one set of spherical harmonics
      (:meth:`rsfff.features.features.FlatLambdaSOAPFeaturizer.forward` with
      ``also_ungrouped=True``), so the difference is attributable to the environment and not
      to two independently-built geometries.

    Equivariance: the invariant channel gets a plain MLP residual, and each equivariant
    channel is gated by an **invariant** function of the environment times the environment's
    own equivariant features. A scalar gate times an ``l``-equivariant tensor is
    ``l``-equivariant, which is the same construction :class:`rsfff.mlip.adiabatic.
    AdiabaticCorrection` uses -- never a scalar MLP on concatenated components, which would
    break rotation equivariance outright (``docs/range_separated_mlip.md`` §7).

    **What must not consume this.** The response solve and the bond channel must stay on
    ``h_frag``: ``fragment_energy`` is Q-Chem's isolated-fragment energy, which has no
    environment dependence *by construction*, so a 1-body term built on ``h_env`` is fitting a
    function that cannot match its target. Those two are the whole of the restriction --
    everything else reads ``h_env``, because the range separation already switches the intra
    classical contribution below a kilojoule per fragment.

    **This block must never see weight decay.** Its three MLPs go through
    :func:`rsfff.mlip.heads.zero_init_readout`, which exempts them; the class attribute below
    extends that to ``species_emb``, which exists only to feed them and so shares their fate.
    Left decaying, the invariant stream collapsed to *identically* zero over a staged fit --
    ``h_env == h_frag`` exactly, ``dC6/dR`` for an atom in another fragment at 7e-64, the
    dispersion silently back to rigorously two-body, and ``env_norm`` reading a true zero that
    looked like a model choosing not to use its environment. ``unified.env_weight`` is the
    regularizer that belongs here: it penalizes ``||h_env - h_frag||``, which is the quantity
    that should be small.
    """

    #: Extends the readouts' exemption to ``species_emb``. See the note above.
    no_weight_decay = True

    def __init__(
        self,
        p0: int,
        p1: int | None,
        p2: int | None,
        n_species: int,
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.inv_mlp = mlp(p0 + emb_dim, hidden, depth, p0)
        self.vec_gate = mlp(p0 + emb_dim, hidden, depth, p1) if p1 else None
        self.equiv_gate = mlp(p0 + emb_dim, hidden, depth, p2) if p2 else None
        for m in (self.inv_mlp, self.vec_gate, self.equiv_gate):
            if m is not None:
                zero_init_readout(m)

    def forward(self, frag: LambdaFeatures, full: LambdaFeatures) -> LambdaFeatures:
        """``h_env = h_frag + g(h_full) - g(h_frag)``, anchored at the isolated fragment.

        The subtraction is not cosmetic. For a fragment alone in space every neighbor within
        the feature cutoff is one of its own atoms, so ``h_full == h_frag`` and the residual
        vanishes -- not merely for small ``g``, and not merely at initialization. Everything
        downstream that measures environment dependence as a difference between the two streams
        (the polarization and charge-transfer corrections, which are defined that way) is
        therefore zero on a monomer, which is what their labels require.

        Exact in exact arithmetic; ~1e-16 in float64, because the two descriptors come from
        two scatters that sum the same edges in different orders
        (:meth:`rsfff.features.features.FlatLambdaSOAPFeaturizer.forward`).

        Without it an isolated fragment still picks up ``g(h_frag)``, some arbitrary constant
        the rest of the model would have to learn to cancel, and the charge-transfer energy of
        a lone water would be spuriously nonzero.
        """
        emb = self.species_emb(full.species_idx)
        x_full = torch.cat((full.inv_feats, emb), dim=-1)
        x_frag = torch.cat((frag.inv_feats, emb), dim=-1)
        inv = frag.inv_feats + self.inv_mlp(x_full) - self.inv_mlp(x_frag)
        vec, equiv = frag.vec_feats, frag.equiv_feats
        if self.vec_gate is not None and vec is not None and full.vec_feats is not None:
            # scalar gate times an l-equivariant tensor stays l-equivariant, on both terms
            vec = (
                vec
                + self.vec_gate(x_full).unsqueeze(1) * full.vec_feats
                - self.vec_gate(x_frag).unsqueeze(1) * frag.vec_feats
            )
        if self.equiv_gate is not None and equiv is not None and full.equiv_feats is not None:
            equiv = (
                equiv
                + self.equiv_gate(x_full).unsqueeze(1) * full.equiv_feats
                - self.equiv_gate(x_frag).unsqueeze(1) * frag.equiv_feats
            )
        return replace(frag, inv_feats=inv, vec_feats=vec, equiv_feats=equiv)

    def magnitude(self, frag: LambdaFeatures, env: LambdaFeatures) -> torch.Tensor:
        """``||h_env - h_frag||`` per atom: how much environment the model is using."""
        return (env.inv_feats - frag.inv_feats).norm(dim=-1)


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
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        log_r0_prior: torch.Tensor,          # (n_channels, n_species), rows like `channels`
        alpha_init: float,
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
                {name: mlp(p0 + emb_dim, hidden, depth, 1) for name in self.channel_names}
            )
            for m in self.r0_mlp.values():   # start at exactly the per-element value
                zero_init_readout(m)

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


@dataclass
class UnifiedOutput:
    """Per-frame totals, per-fragment energies, and the full per-pair breakdown."""

    energy: torch.Tensor                      # (B,) the total, all channels
    fragment_energy: torch.Tensor             # (F,) -> batch.fragment_energy
    #: Inter-fragment channel sums, the EDA targets. Keys: elst, pauli, disp.
    interaction: dict[str, torch.Tensor]      # each (B,)
    interaction_ff: dict[str, torch.Tensor]   # each (B,) classical part only
    interaction_corr: dict[str, torch.Tensor]  # each (B,) correction part only
    energy_ref: torch.Tensor                  # (F,) sum_i E0[Z_i], frozen
    energy_internal: torch.Tensor             # (F,) the response solve's internal energy
    energy_bond: torch.Tensor                 # (F,) intra pairs: classical + all corrections
    #: (F,) the **frozen** per-atom energy of the electronic state, pooled per fragment:
    #: ``E_theta(h_frag, M^frozen, Phi^intra)``. This is what carries the covalent bond energy
    #: inside ``fragment_energy`` now that the bond channel does not, and it is the pair of
    #: numbers to watch alongside ``energy_internal`` -- their split is unlabeled, so a drift
    #: between stages is the failure mode this design exists to remove. Zero when no
    #: :class:`rsfff.ff.atomic_energy.AtomicStateEnergy` head is attached.
    energy_atom: torch.Tensor                 # (F,)
    pair_index: torch.Tensor                  # (2, P)
    r: torch.Tensor                           # (P,) Angstrom
    is_intra: torch.Tensor                    # (P,) bool, routing only -- never a gate
    pair_frag: torch.Tensor                   # (P,) fragment id, -1 for inter pairs
    #: Per-pair breakdown before routing, for every pair. Keys: elst, pauli, disp (both) and
    #: additionally bond in ``e_pair_corr``. Summing these back up is the accounting identity
    #: ``energy == E0 + E_internal + sum_pairs sum_c (ff + corr) + sum_intra bond``.
    e_pair_ff: dict[str, torch.Tensor]        # each (P,)
    e_pair_corr: dict[str, torch.Tensor]      # each (P,)
    gate: dict[str, torch.Tensor]             # (P,) fermi * taper, per classical channel
    r0: dict[str, torch.Tensor]               # (N,) per-atom base midpoint, per channel
    #: (P,) the midpoint each pair actually used: the per-atom base combined by geometric
    #: mean, times the learned per-pair correction. This is the one to read when asking what
    #: the range separation decided -- ``r0`` above is only its per-element starting point.
    r0_pair: dict[str, torch.Tensor]
    alpha: dict[str, torch.Tensor]            # () per-channel width
    species_idx: torch.Tensor                 # (N,) index into neighbor_types
    #: Per channel, (N,) ``log r0`` each atom would have from its element prior alone -- and
    #: per channel because dispersion's prior is not the others' (see
    #: :data:`rsfff.ff.range_priors.CHANNEL_R0_PRIOR`). The penalty that keeps a learned ``r0``
    #: near its element value is the residual against this.
    log_r0_prior: dict[str, torch.Tensor]
    #: Per channel, (P,) the element prior combined across each pair by geometric mean. This is
    #: the floor the ``r0`` barrier is measured against, and it is the pair-level quantity --
    #: what escaped the old penalty was the *per-pair* deviation, not the per-element value.
    log_r0_prior_pair: dict[str, torch.Tensor]
    #: The **frozen** response: parameters from fragment-confined features, no external
    #: field. Named so the polarized solve lands beside it rather than on top of it.
    response: FragmentResponseOutput
    #: (N,) ``||h_env - h_frag||`` per atom, or None when environment awareness is off. How
    #: much of its surroundings the model has decided each atom's parameters need; zero at
    #: initialization by construction, and zero for an isolated fragment at any stage.
    environment_norm: torch.Tensor | None = None
    #: (B,) the environment-dependent part of the electrostatic channel -- correction and gate
    #: together -- that was booked as polarization rather than left in ``cls_elec``. It is
    #: already included in ``interaction["pol"]``; this exposes it on its own because the size
    #: of what moved is the thing worth watching. ``None`` below the polarized level, where
    #: there is no channel to receive it and it is dropped instead.
    #:
    #: Expect it to start large at the polarized stage and settle. Nothing constrains the
    #: electrostatic readout's ``h_env`` value during the frozen stage -- it is evaluated only
    #: to be discarded -- so it enters stage 2 at whatever the shared trunk drifted to, and the
    #: ``pol`` label is what pulls it into place.
    elst_env: torch.Tensor | None = None

    #: The **polarized** and **CT** levels: the same functional as the frozen one, minimized
    #: with one more constraint lifted each time. ``None`` when that level is switched off.
    #: Their multipoles are *not* forwarded by the properties above -- those stay frozen,
    #: because the fragment-multipole labels are frozen-monomer values.
    level_pol: "LevelOutput | None" = None
    level_ct: "LevelOutput | None" = None
    #: (F,) the same atomic-energy network read at the polarized and CT electronic states.
    #: ``None`` when that level is off. Their differences are what ``pol`` and ``ct`` receive,
    #: so the three telescope: ``energy_atom + (pol - 0) + (ct - pol) == energy_atom_ct``.
    #: Exposed because the *levels* of these, not only their differences, are what says
    #: whether the atomic energy is describing a relaxation or re-deciding what a bond is.
    energy_atom_pol: torch.Tensor | None = None
    energy_atom_ct: torch.Tensor | None = None
    #: (B,) the frozen level's own total, ``E_internal + all-pair electrostatics``. This is the
    #: reference ``E_pol`` is measured against, and it is the *same functional* evaluated at
    #: the frozen multipoles -- which is what makes ``E_pol`` a pure relaxation.
    energy_frozen_total: torch.Tensor | None = None
    #: Per-atom electrostatic environment at the highest active level, built from the
    #: **converged** multipoles. What the bond channel reads.
    environment: OneBodyEnvironment | None = None
    #: Solver diagnostics per level: ``{level: (n_iter, converged (B,), pd_fail (B,))}``.
    #: A rising iteration count is the early warning that the polarization is running away,
    #: well before it becomes a NaN.
    solver: dict[str, tuple] | None = None

    # The solved multipoles, forwarded from the frozen response so this output satisfies the
    # same duck type as `ElectrostaticsOutput` and can be handed straight to
    # `rsfff.train.loss.fragment_multipole_loss` without a shim. When a polarized solve is
    # added these must keep pointing at the *frozen* one -- the fragment multipole labels are
    # frozen-monomer values.
    @property
    def charges(self) -> torch.Tensor:
        return self.response.charges

    @property
    def mu(self) -> torch.Tensor | None:
        return self.response.mu

    @property
    def quad_s(self) -> torch.Tensor | None:
        return self.response.quad_s

    @property
    def polarizability(self) -> torch.Tensor | None:
        """(F, 3, 3) the **frozen** molecular polarizability, or None if it was not asked for.

        Frozen for the same reason the multipoles above are: the label is an isolated
        monomer's, and an isolated monomer's response is the frozen one at every level.
        """
        return self.response.polarizability


class UnifiedPairModel(nn.Module):
    """Featurizer + response + range separation + shared correction trunk, as one term.

    Args
    ----
    featurizer     : produces the fragment-confined descriptor ``h_frag``.
    response       : the frozen :class:`rsfff.ff.response.FragmentResponse`.
    disp_params    : per-atom ``(C6, b)``.
    pauli_params   : per-atom Pauli multipoles.
    range_heads    : per-atom ``r0`` and per-channel ``alpha``.
    pair_head      : the shared trunk with one readout per channel, including ``bond``.
    fragment_state : the ``(Q_f, 2S_f)`` block; ``dim=0`` disables it.
    reference_energies : (n_species,) isolated-atom energies in Hartree, frozen.
    max_rank       : multipole rank, must match the response heads'.
    classical      : per-channel neighbor-list reach.
    atomic_energy  : :class:`rsfff.ff.atomic_energy.AtomicStateEnergy`, the per-atom energy of
                     the electronic state. ``None`` falls back to the bond channel.
    pair_corrections : which of the pair head's energy readouts are consumed. ``True`` is all
                     of them, ``False`` none, and an iterable of channel names exactly those.
                     A channel that is off leaves its readout built -- and hence loadable, and
                     re-enablable -- while its ``e_corr`` stays at zero. See the note in
                     :meth:`forward`.
    """

    def __init__(
        self,
        featurizer,
        response: FragmentResponse,
        disp_params: DispersionParameterHeads,
        pauli_params: PauliMultipoleHeads,
        range_heads: RangeSeparationHeads,
        pair_head: UnifiedPairHead,
        fragment_state: FragmentStateEmbedding,
        reference_energies: torch.Tensor,
        *,
        atomic_energy: "AtomicStateEnergy | None" = None,
        pair_corrections: "bool | Sequence[str]" = True,
        environment: "EnvironmentResidual | None" = None,
        max_rank: int = 1,
        classical: dict[str, ClassicalSpec] | None = None,
        max_num_neighbors: int = 512,
        polarization: bool = False,
        charge_transfer: bool = False,
        ct_channel_cutoff: float = 5.0,
        ct_channel_taper: float = 1.0,
        ct_compliance_scale: float = 0.1,
        cg_rtol: float = 1.0e-9,
        cg_atol: float = 1.0e-12,
        cg_maxiter: int = 100,
    ) -> None:
        super().__init__()
        if int(max_rank) != int(response.max_rank):
            raise ValueError(
                f"max_rank {max_rank} does not match the response heads' "
                f"{response.max_rank}; the heads decide which multipoles exist and this "
                f"decides which ones the interaction tensor carries"
            )
        self.classical = dict(classical or DEFAULT_CLASSICAL)
        missing = set(range_heads.channel_names) - set(self.classical)
        if missing:
            raise ValueError(f"range separation for channels with no classical form: {missing}")
        correction_channels = _resolve_correction_channels(
            pair_corrections, pair_head.channels
        )
        # Exactly one of the two has to be carrying the intramolecular deformation energy that
        # `fragment_energy` measures. Historically that was the bond channel; it is now
        # `atomic_energy`, and the check is that *something* is there rather than that a
        # particular one is.
        if atomic_energy is None and "bond" not in correction_channels:
            raise ValueError(
                "nothing is carrying the intramolecular deformation energy that "
                "fragment_energy measures: pass an `atomic_energy` head, or include 'bond' "
                "in `pair_corrections` with a 'bond' channel in the pair head"
            )
        self.featurizer = featurizer
        self.response = response
        self.disp_params = disp_params
        self.pauli_params = pauli_params
        self.range_heads = range_heads
        self.pair_head = pair_head
        self.atomic_energy = atomic_energy
        #: Which of the pair head's *energy* readouts are read, as a frozenset of channel
        #: names. The module stays built whatever this holds: every readout keeps its
        #: parameters, its ``range_readout`` may still be live, its state_dict stays loadable
        #: across the switch, and turning a channel back on is a config change rather than a
        #: refit from scratch. Note that with this **empty** and ``pair_range_separation`` off,
        #: nothing calls the head at all -- see ``parameter_groups`` in
        #: :mod:`rsfff.train.term_loop` for why it then has to be kept out of the optimizer
        #: rather than left to weight decay.
        self.correction_channels = correction_channels
        self.fragment_state = fragment_state
        self.environment = environment
        self.max_rank = int(max_rank)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(c.cutoff for c in self.classical.values())
        self.register_buffer("reference_energies", reference_energies.clone())

        if charge_transfer and not polarization:
            raise ValueError(
                "charge_transfer needs polarization: `ct` is defined as E_ct - E_pol, so "
                "without the polarized level there is nothing to difference against and the "
                "label would silently absorb the polarization energy too"
            )
        self.polarization = bool(polarization)
        self.charge_transfer = bool(charge_transfer)
        #: Whether the pair head carries the cross-fragment bond channel. Read from the head
        #: rather than configured here, so the two cannot disagree: the channel exists exactly
        #: when there is a readout for it.
        self.ct_bond = "ct_bond" in self.correction_channels
        self.ct_channel_cutoff = float(ct_channel_cutoff)
        self.ct_channel_taper = float(ct_channel_taper)
        self.ct_compliance_scale = float(ct_compliance_scale)
        self.cg = dict(rtol=float(cg_rtol), atol=float(cg_atol), maxiter=int(cg_maxiter))
        if self.polarization and self.pair_corrections and pair_head.extra_dim == 0:
            raise ValueError(
                "polarization needs a pair head built with extra_dim = "
                "rsfff.ff.environment.N_PAIR_INVARIANTS; the bond channel's response to the "
                "external field is what relays the environment into the bonded region, and "
                "with no side channel there is nowhere for the field to enter"
            )
        if self.polarization and self.atomic_energy is None and not self.pair_corrections:
            raise ValueError(
                "polarization has no route into the energy: with every pair correction off "
                "and no `atomic_energy` head, nothing consumes the relaxed multipoles"
            )

    def _augment(self, feats: LambdaFeatures, batch, frag) -> LambdaFeatures:
        """Concatenate the fragment-state block onto a descriptor's invariants."""
        state = self.fragment_state(
            batch, frag, feats.inv_feats.dtype, feats.inv_feats.device
        )
        if state is None:
            return feats
        return replace(feats, inv_feats=torch.cat((feats.inv_feats, state), dim=-1))

    @property
    def pair_corrections(self) -> bool:
        """Whether *any* correction channel is consumed.

        Derived from :attr:`correction_channels` rather than stored, and deliberately
        read-only. It was a stored bool when the switch was all-or-nothing, and the two then
        had to be kept in step by hand -- assigning ``model.pair_corrections = False`` looked
        like it turned the corrections off while ``forward`` went on reading the set and
        happily scored every channel. Making it a property turns that into an ``AttributeError``
        at the assignment instead of a silent wrong answer several layers down.
        """
        return bool(self.correction_channels)

    def _descriptors(self, batch, frag, *, want_env: bool = True):
        """``(h_frag, h_env)`` -- the fragment-confined and environment-aware descriptors.

        ``h_frag`` is what the response solve and the atomic energy read: ``fragment_energy``
        is an isolated-fragment label, so a 1-body term built on anything else is fitting a
        function that cannot match its target. ``h_env`` adds the zero-initialized environment
        residual and is what the *interaction* parameters read. With ``environment`` off the
        two are the same object and everything downstream runs once.

        ``want_env=False`` returns ``(h_frag, None)`` and skips the residual, for callers that
        only need the frozen response. It still asks the featurizer for ``also_ungrouped``:
        the grouped-only and paired code paths sum edges in different orders, so taking
        ``grouped`` from the pair is what keeps ``h_frag`` bit-identical to the one
        :meth:`forward` builds. The second power spectrum is the price of that identity, and
        it is not what makes the cheap path cheap -- skipping the pair model is.
        """
        if self.environment is None:
            h = self._augment(self.featurizer(batch, frag), batch, frag)
            return h, h
        grouped, full = self.featurizer(batch, frag, also_ungrouped=True)
        h_frag = self._augment(grouped, batch, frag)
        if not want_env:
            return h_frag, None
        return h_frag, self.environment(h_frag, self._augment(full, batch, frag))

    def frozen_polarizability(self, batch) -> torch.Tensor | None:
        """``(F, 3, 3)`` each fragment's frozen molecular polarizability, without the pair model.

        The same tensor :attr:`UnifiedOutput.polarizability` carries, reached by the short
        route: featurizer, then the frozen response solve. Nothing else contributes to it --
        the pair list, the classical channels, the correction trunk and the atomic energy all
        feed *energies*, not the response -- so this is not an approximation to the full
        forward, it is the same computation with the unused three quarters left out.

        It exists for :func:`rsfff.train.loss.free_atom_polarizability_loss`, whose batch is a
        handful of **single atoms**. Through the full forward that measured 6.7 ms a step for
        two atoms, more than a 350-atom cluster forward, because essentially all of it was the
        fixed Python cost of machinery that had nothing to do on an empty pair list.
        """
        frag = batch.fragment_idx
        if frag is None:
            raise ValueError(
                "frozen_polarizability is a per-fragment quantity but batch.fragment_idx is "
                "None; the extxyz needs a `fragment_idx` column"
            )
        h_frag, _ = self._descriptors(batch, frag, want_env=False)
        return self.response(batch, h_frag, with_polarizability=True).polarizability

    def _pauli_multipoles(self, feats: LambdaFeatures):
        """``(polytensor (N, K), b (N,))`` for the Slater Pauli term."""
        q, b, mu, quad_s = self.pauli_params(
            feats.inv_feats, feats.species_idx, feats.vec_feats, feats.equiv_feats
        )
        poly = build_polytensor(
            q, mu,
            None if quad_s is None else spherical_to_cartesian_quadrupole(quad_s),
            max_rank=self.max_rank,
        )
        return poly, b

    def forward(self, batch, *, with_polarizability: bool = False) -> UnifiedOutput:
        """``with_polarizability`` asks the frozen solve for each fragment's molecular
        polarizability as well, which only the isolated-monomer anchor has a label for."""
        if batch.fragment_idx is None:
            raise ValueError(
                "the unified model routes pair energies to per-fragment and per-frame labels "
                "but batch.fragment_idx is None; the extxyz needs a `fragment_idx` column"
            )
        positions = batch.positions
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)
        n_sys = int(batch.n_systems)

        # --- features ------------------------------------------------------------------
        h_frag, h_env = self._descriptors(batch, frag)

        # --- the frozen response solve, shared by the 1-body and elst channels -----------
        res = self.response(batch, h_frag, with_polarizability=with_polarizability)

        # --- one pair list, nothing dropped ---------------------------------------------
        pair_index, r, is_intra, pair_frag = union_pairs(
            positions, batch.batch_idx, frag, self.cutoff_max,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG

        # --- the correction trunk, partitioned by routing --------------------------------
        # Each pair is scored on the stream its label demands. Intra pairs read `h_frag` and
        # take the bond channel; inter pairs read `h_env` and take the three interaction
        # corrections. Both partitions also emit a per-pair range-separation deviation, so the
        # *within-fragment* range separation is a function of fragment-confined features alone.
        #
        # Intra pairs deliberately do *not* get the interaction corrections. Their classical
        # counterpart is what the range separation is deciding about, and a correction there
        # would be an unconstrained second bond term, degenerate with the real one.
        #
        # This first pass is the **frozen** level, and `d_log_r0` is taken from it alone. That
        # is what makes the gate -- and hence the classical backbone at every level -- a
        # frozen-level object, fixed before any coupled solve runs. Without it the environment
        # features would depend on a gate that depended on them.
        sp = h_frag.species_idx
        e_corr = {name: torch.zeros_like(r) for name in self.pair_head.channels}
        d_log_r0 = {name: torch.zeros_like(r) for name in self.classical}
        intra_idx = is_intra.nonzero().squeeze(-1)
        inter_idx = (~is_intra).nonzero().squeeze(-1)

        def score(idx, feats, extra=None):
            a, b_ = i[idx], j[idx]
            return self.pair_head(
                feats.inv_feats[a], feats.inv_feats[b_], sp[a], sp[b_], r[idx],
                extra=None if extra is None else extra[idx],
            )

        # `d_log_r0_env` shadows `d_log_r0` on the electrostatic channel alone: it is the pair
        # of gates whose *difference* is polarization, built below. On the other two channels,
        # and everywhere `h_env is h_frag`, the two dicts hold the same tensors.
        d_log_r0_env = dict(d_log_r0)
        # Two independent consumers of the trunk, and either can be switched off on its own:
        # the energy readouts (`correction_channels`) and the per-pair range separation
        # (`range_channels`). With both off the head is never called and `e_corr`/`d_log_r0`
        # keep the zeros they were initialized with, which is exactly the "atomic energy plus
        # range-separated FF" model -- no branch anywhere downstream has to know.
        #
        # The energy switch is **per channel**: a name absent from `correction_channels` keeps
        # its zeros here and is silent everywhere downstream, while the head still evaluates
        # once per partition either way (one trunk, all readouts) so the choice costs nothing
        # to make. `elst` alone is the setting that corrects electrostatics -- and, through the
        # split below, polarization -- while leaving `bond` to `atomic_energy`.
        want = self.correction_channels
        want_corr = bool(want)
        want_r0 = bool(self.pair_head.range_channels)
        if want_corr or want_r0:
            for idx, feats, routed in (
                (intra_idx, h_frag, ("bond",)),
                (inter_idx, h_env, tuple(self.classical)),
            ):
                if not idx.numel():
                    continue
                energies, devs = score(idx, feats)
                for name in routed:
                    if name in want and name in energies:
                        e_corr[name] = e_corr[name].index_copy(0, idx, energies[name])
                for name in devs:
                    d_log_r0[name] = d_log_r0[name].index_copy(0, idx, devs[name])
                    d_log_r0_env[name] = d_log_r0[name]

        # --- the electrostatic channel is fragment-confined, at every level ----------------
        # `eda_frz_elec` is the Coulomb interaction between *superimposed frozen monomer
        # densities*. That is rigorously pairwise, so `cls_elec` has to be a function of the
        # two fragments alone and of nothing else. Two things reach it from `h_env` and both
        # are re-scored on `h_frag` here:
        #
        #     the correction   W(u(h_frag))                                -> cls_elec
        #     the gate         fermi_switch(r, r0(d_log_r0(u(h_frag))))     -> cls_elec
        #
        # and each one's `h_env - h_frag` difference is **polarization**, not electrostatics.
        # It is a real, wanted, environment-dependent energy -- the surroundings do move where
        # the classical form takes over -- it simply belongs in the other column:
        #
        #     dE_pol = [W(u(h_env)) - W(u(h_frag))] + [gate_env - gate_frag] * E_classical
        #
        # Both are identities, so the total energy is untouched and the split telescopes away
        # outside training. Measured on the checkpoint that had only the correction half of
        # this, the gate half was leaving 1.1 / 2.3 / 2.7 / 3.2 kJ/mol per frame of
        # polarization inside `cls_elec` on w2 / w3 / w4 / w5 -- four to six times that
        # channel's own MAE, and nonzero even on dimers, where there is no many-body content
        # at all to explain it.
        #
        # Below the polarized level there is no `pol` channel to receive the difference, so it
        # is dropped rather than rebooked: `cls_elec` is the same fragment-confined object at
        # every stage, which is what makes the staged fit's frozen numbers comparable across
        # stages at all.
        elst_env_pair = torch.zeros_like(r)
        # The correction half of the split needs the `elst` *readout* specifically, not merely
        # some channel being on: with `pair_corrections: [bond]` there is no electrostatic
        # correction to re-score and only the gate half has anything to say.
        want_elst_corr = "elst" in want
        split_elst = (
            (want_elst_corr or want_r0) and bool(inter_idx.numel()) and h_env is not h_frag
        )
        if split_elst:
            frozen_elst, frozen_devs = score(inter_idx, h_frag)
            # Each half is guarded by its own switch. The correction half writes *into*
            # `e_corr`, so running it with the `elst` correction off would resurrect exactly
            # the readout that was supposed to be silent -- measured at 0.004 Ha on `cls_elec`
            # before this guard existed, which is the whole channel.
            if want_elst_corr:
                elst_env_pair = elst_env_pair.index_copy(
                    0, inter_idx, e_corr["elst"][inter_idx] - frozen_elst["elst"]
                )
                e_corr["elst"] = e_corr["elst"].index_copy(
                    0, inter_idx, frozen_elst["elst"]
                )
            if want_r0:
                d_log_r0["elst"] = d_log_r0["elst"].index_copy(
                    0, inter_idx, frozen_devs["elst"]
                )

        # --- classical backbones, every pair ---------------------------------------------
        # ``r0`` is a per-element base (the covalent-distance knowledge of
        # :mod:`rsfff.ff.range_priors`) times a learned per-pair correction. The base cannot
        # tell topologically distinct pairs of the same elements apart; the correction can,
        # and that is the whole reason it exists -- see :class:`UnifiedPairHead`.
        # The per-atom base is read on `h_env` for Pauli and dispersion, whose environment
        # dependence is wanted, and on `h_frag` for electrostatics, whose is not -- the same
        # split as the per-pair deviation above. With `environment_r0` off the head ignores
        # `inv_feats` entirely and the two calls return identical tensors, so the second one is
        # skipped rather than merely wasted.
        r0_atom, alpha = self.range_heads(h_env.inv_feats, h_env.species_idx)
        r0_atom_frag = r0_atom
        if self.range_heads.r0_mlp is not None and h_env is not h_frag:
            r0_atom_frag, _ = self.range_heads(h_frag.inv_feats, h_frag.species_idx)
        log_r0_prior = {
            name: self.range_heads.log_r0_prior[c][h_frag.species_idx]
            for c, name in enumerate(self.range_heads.channel_names)
        }

        def build_gate(name, spec, base, dev):
            log_r0_ij = 0.5 * (base[name][i].log() + base[name][j].log()) + dev[name]
            r0 = log_r0_ij.exp()
            switch = fermi_switch(r, r0, alpha[name]) * pairwise_switch(
                r, spec.cutoff - spec.taper_width, spec.cutoff
            )
            return switch, r0

        gate, r0_pair, log_r0_prior_pair = {}, {}, {}
        for name, spec in self.classical.items():
            base = r0_atom_frag if name == "elst" else r0_atom
            gate[name], r0_pair[name] = build_gate(name, spec, base, d_log_r0)
            if name in log_r0_prior:
                log_r0_prior_pair[name] = 0.5 * (
                    log_r0_prior[name][i] + log_r0_prior[name][j]
                )

        # The environment-aware electrostatic gate, kept only for the difference that becomes
        # the polarization correction. `gate["elst"]` -- the one `cls_elec` uses, the one the
        # `r0` penalties see, and the one on `UnifiedOutput` -- stays fragment-confined.
        gate_elst_env = None
        if split_elst:
            gate_elst_env, _ = build_gate(
                "elst", self.classical["elst"], r0_atom, d_log_r0_env
            )

        # The multipoles come from the frozen response, which is fragment-confined -- see the
        # module docstring on the ceiling that binds there.
        quad_c = None if res.quad_s is None else spherical_to_cartesian_quadrupole(res.quad_s)
        m_real = build_polytensor(res.charges, res.mu, quad_c, max_rank=self.max_rank)
        m_shell = build_polytensor(res.charges - res.z, res.mu, quad_c, max_rank=self.max_rank)
        m_nuc = build_polytensor(res.z, None, None, max_rank=self.max_rank)
        e_point, e_pen = slater_elec_pair_energy(
            dr_au, r_au, m_real, m_shell, m_nuc, res.b, pair_index, max_rank=self.max_rank
        )

        # The gate half of the electrostatic split, now that the classical energy the two gates
        # multiply exists. `inter *` is applied when this is pooled, so restricting it to inter
        # pairs is already handled; intra pairs share one gate anyway, because their
        # `d_log_r0` was read from `h_frag` in the first place.
        if gate_elst_env is not None:
            elst_env_pair = elst_env_pair + (gate_elst_env - gate["elst"]) * (
                e_point + e_pen
            )
        # Below the polarized level there is nothing to receive it, so it is dropped -- see the
        # routing comment above. `cls_elec` is fragment-confined either way.
        corr_pol_pair = elst_env_pair if self.polarization else torch.zeros_like(r)

        # **Pauli and dispersion read `h_env` on every pair, at every level.** Their parameters
        # are supposed to be environment-dependent: a quenched `C6` and an environment-softened
        # Slater exponent *are* the effective many-body content of those channels, and that is
        # the reason the environment descriptor exists.
        #
        # This used to select per pair -- `h_frag` for intra, `h_env` for inter -- to keep
        # `fragment_energy` free of environment dependence, at a measured cost of ~1.1 kJ/mol.
        # That protection is no longer where the one-body label's integrity comes from. The
        # intra classical channels are switched down to well under a kJ/mol per fragment by the
        # range separation, and what actually carries `fragment_energy` is `E_internal` (from
        # the fragment-confined response) plus `E_atom^0` (from `h_frag`, by construction).
        # Paying two per-atom MLP evaluations to defend a term that is already switched off is
        # the wrong trade, and it made the Pauli and dispersion parameters of a bonded pair
        # answer to a different network than those of a hydrogen-bonded one.
        poly_e, b_p_e = self._pauli_multipoles(h_env)
        b_p_ij = (0.5 * (b_p_e[i].log() + b_p_e[j].log())).exp()
        e_pauli = slater_pauli_pair_energy(
            dr_au, r_au, poly_e[i], poly_e[j], b_p_ij, max_rank=self.max_rank,
        )

        c6_e, b_d_e = self.disp_params(h_env.inv_feats, h_env.species_idx)
        c6_ij = (0.5 * (c6_e[i].log() + c6_e[j].log())).exp()
        b_d_ij = (0.5 * (b_d_e[i].log() + b_d_e[j].log())).exp()
        e_disp = tt_damped_c6_energy(r, c6_ij, b_d_ij)

        e_ff = {
            "elst": gate["elst"] * (e_point + e_pen),
            "pauli": gate["pauli"] * e_pauli,
            "disp": gate["disp"] * e_disp,
        }

        # --- routing: which label does each pair answer to? ------------------------------
        inter = (~is_intra).to(r.dtype)
        intra = is_intra.to(r.dtype)
        pair_batch = batch.batch_idx[i]

        def pool_batch(x):
            return x.new_zeros(n_sys).index_add_(0, pair_batch, x)

        interaction_ff, interaction_corr, interaction = {}, {}, {}
        for name in e_ff:
            interaction_ff[name] = pool_batch(inter * e_ff[name])
            interaction_corr[name] = pool_batch(inter * e_corr[name])
            interaction[name] = interaction_ff[name] + interaction_corr[name]

        # --- the per-atom energy of the electronic state ----------------------------------
        # `t_point` is hoisted out of the polarized branch because the *frozen* atomic energy
        # needs it too: an isolated fragment's atoms genuinely feel their own molecule's field,
        # and that field is part of what its bonds are worth.
        t_point = damped_interaction_tensor(dr_au, None, 1.0 / r_au, max_rank=self.max_rank)

        def atom_energy(feats, q_, mu_, quad_s_, env_):
            """``(n_frag,)`` -- the per-atom energies of one level, pooled per fragment."""
            if self.atomic_energy is None:
                return positions.new_zeros(n_frag)
            e = self.atomic_energy(
                feats.inv_feats, feats.species_idx, feats.vec_feats, feats.equiv_feats,
                q_, mu_, quad_s_, env_,
            )
            return e.new_zeros(n_frag).index_add_(0, frag, e)

        # `Phi^intra`: gating the environment down to intra-fragment pairs is the whole of what
        # makes `E_atom^0` an isolated-fragment quantity. Do not simplify it to `phi = 0` --
        # that was the bond channel's workaround for having no way to tell the two apart, and
        # it left the intramolecular field invisible to the term that describes bonding.
        env_frozen = electrostatic_environment(
            positions, pair_index, t_point, gate["elst"] * intra, m_real,
            max_rank=self.max_rank,
        )
        energy_atom = atom_energy(h_frag, res.charges, res.mu, res.quad_s, env_frozen)
        energy_atom_pol = energy_atom_ct = None

        # --- the polarized and CT levels -------------------------------------------------
        f2b_ = batch.fragment_to_batch
        if f2b_ is None:
            f2b_ = batch.batch_idx.new_zeros(n_frag).scatter_(0, frag, batch.batch_idx)

        level_pol = level_ct = None
        env = None
        energy_frozen_total = None
        solver: dict[str, tuple] = {}
        if self.polarization:
            # The frozen level's own total: the *same functional* the coupled levels minimize,
            # evaluated at the frozen multipoles. `E_pol` is the difference, so it is exactly
            # zero when nothing moves and (sharing parameters) negative by the variational
            # principle -- a pure relaxation rather than two models differenced.
            energy_frozen_total = res.internal_energy.new_zeros(n_sys).index_add(
                0, f2b_, res.internal_energy
            ) + pool_batch(e_ff["elst"])

            def run_level(bond_index, bond_batch, envelope, ct_channels=None):
                rp = self.response.response_parameters(
                    batch, h_env, bond_index=bond_index, envelope=envelope,
                    ct_channels=ct_channels,
                )
                return coupled_response(
                    rp, positions=positions, batch_idx=batch.batch_idx, n_systems=n_sys,
                    bond_index=bond_index, bond_batch=bond_batch, pair_index=pair_index,
                    gate=gate["elst"], max_rank=self.max_rank, **self.cg,
                )

            def env_extra(level):
                """The per-atom ``(V, E, grad E)`` at one level's converged multipoles.

                It used to also return the per-*pair* reduction
                (:func:`~rsfff.ff.environment.environment_pair_invariants`) as a side channel
                for the bond readout's field dependence. The bond telescoping is gone, and
                nothing else consumed it, so computing it every step was pure cost. If the
                pair corrections are switched back on and wanted field-aware, rebuild it here
                from ``e`` and ``r_hat`` -- both are still to hand, and the head's
                ``extra_dim`` columns are still sized for it.
                """
                quad_c = (
                    None if level.quad_s is None
                    else spherical_to_cartesian_quadrupole(level.quad_s)
                )
                m = build_polytensor(
                    level.charges, level.mu, quad_c, max_rank=self.max_rank
                )
                return electrostatic_environment(
                    positions, pair_index, t_point, gate["elst"], m, max_rank=self.max_rank
                )

            # Polarized: response parameters see the environment and the multipoles relax
            # against each other, but charge still cannot cross a fragment boundary.
            ch_pol, chb_pol, _ = union_channels(positions, batch.batch_idx, frag, 0.0)
            level_pol = run_level(ch_pol, chb_pol, None)
            solver["pol"] = (level_pol.n_iter, level_pol.converged, level_pol.pd_fail)
            # Kept under its own name: the CT branch below rebinds `env` to the fully relaxed
            # environment, and the polarized atomic energy must not silently read that one.
            env_pol = env_extra(level_pol)
            env = env_pol

            if self.charge_transfer:
                # CT: every constraint lifted. The channel graph opens to all pairs within a
                # cutoff, with the radius-derived ones enveloped so they appear and disappear
                # smoothly, and the solve groups by frame -- charge conserves per frame while
                # each fragment's charge is free to move, which is what CT is.
                ch_ct, chb_ct, from_radius = union_channels(
                    positions, batch.batch_idx, frag, self.ct_channel_cutoff,
                    max_num_neighbors=self.max_num_neighbors,
                )
                # `ct_compliance_scale` is the charge-transfer analogue of the correction
                # channels' `energy_scale`, and it is there for the same reason those are.
                # SQE has no structural reason to prefer an intramolecular channel over an
                # intermolecular one -- the electronegativity difference driving an O-H
                # transfer is the same whether the two atoms share a molecule -- so with both
                # starting at `s_init` the model opens covalent-strength channels across every
                # hydrogen bond. Measured on a freshly initialized w2, unscaled `ct` starts at
                # -52 kJ/mol against a true `eda_ct` of -8.2, with only 0.0005 e of net charge
                # actually crossing -- it is many soft channels each relaxing a little, not a
                # large transfer. The default 0.1 was calibrated against that label:
                #
                #     scale   1.0     0.3     0.1     0.03
                #     w2 ct  -52.0   -19.8   -7.1    -2.2     (kJ/mol, target -8.2)
                #
                # The scale sets the range the head operates over, not a ceiling: softplus is
                # unbounded, so any compliance remains reachable. This only says where to
                # start looking.
                r_ch = (positions[ch_ct[0]] - positions[ch_ct[1]]).norm(dim=-1)
                envelope = torch.where(
                    from_radius,
                    self.ct_compliance_scale * pairwise_switch(
                        r_ch,
                        self.ct_channel_cutoff - self.ct_channel_taper,
                        self.ct_channel_cutoff,
                    ),
                    torch.ones_like(r_ch),
                )
                # `from_radius` also selects which channels the dedicated CT compliance head
                # answers for, so the intra-fragment ones keep reading the frozen head even
                # here -- they are the same covalent channels the frozen level solved.
                level_ct = run_level(ch_ct, chb_ct, envelope, ct_channels=from_radius)
                solver["ct"] = (level_ct.n_iter, level_ct.converged, level_ct.pd_fail)
                env_ct = env_extra(level_ct)
                env = env_ct

            # The atomic energy's lifted constraints, telescoping so that the sum is one
            # evaluation at the fully relaxed level:
            #
            #     E_atom^0   = E_theta(h_frag, M^frozen, Phi^intra)                -> 1-body
            #     E_atom^pol = E_theta(h_frag, M^pol,    Phi^pol )  - E_atom^0      -> pol
            #     E_atom^ct  = E_theta(h_env,  M^ct,     Phi^ct )   - E_atom^pol    -> ct
            #
            # What moves at each step is the *electronic state*, not a parameter: one set of
            # weights, three inputs. That is the difference from the bond channel this
            # replaces, where each level was a separate evaluation of a readout that also
            # defined the one-body zero, so two of the three were unlabeled and the split
            # between them drifted.
            #
            # The feature stream changes at **ct**, not pol. Relaxing a fragment's own
            # multipoles against its neighbours' leaves it the same fragment; letting charge
            # cross the boundary does not, and that is also exactly when its descriptor should
            # stop being fragment-confined.
            #
            # On an isolated fragment `ct` is **exactly** zero: `h_env == h_frag`, no channel
            # reaches outside, so `M^ct == M^pol` and `Phi^ct == Phi^pol` bitwise. `pol` is
            # zero only up to the fragment's own variational relaxation -- `Phi^pol` and
            # `Phi^intra` coincide there (every pair is intra), but `M^pol != M^frozen`,
            # because the polarized level minimizes with the intramolecular electrostatics
            # *inside* the functional while the frozen level adds it afterwards. Measured, that
            # relaxation is 2.6e-5 e and leaves a few hundredths of a kJ/mol in `pol`. It is a
            # property of what the levels *are*, not of this term, and it predates it.
            energy_atom_pol = atom_energy(
                h_frag, level_pol.charges, level_pol.mu, level_pol.quad_s, env_pol
            )

            interaction_ff["pol"] = level_pol.energy - energy_frozen_total
            interaction_corr["pol"] = (
                pool_batch(inter * corr_pol_pair)
                + (energy_atom_pol - energy_atom).new_zeros(n_sys).index_add_(
                    0, f2b_, energy_atom_pol - energy_atom
                )
            )
            interaction["pol"] = interaction_ff["pol"] + interaction_corr["pol"]

            if self.charge_transfer:
                # **CT gets no correction readout of its own.** It is the residue of lifting
                # the last constraint, so what it is entitled to is the coupled solve's own
                # energy difference plus the atomic energy's response to the state that solve
                # produced -- both of which it shares with electrostatics and polarization.
                # A dedicated `ct_bond` readout gave it a private lever instead, and measured
                # on the checkpoint that had one it supplied 96.7% of the channel: -5.24 kJ/mol
                # per fragment of correction against -0.18 of classical. That is not a charge
                # transfer model, it is a neural network wearing the label.
                #
                # This is the step where the descriptor stops being fragment-confined, because
                # it is the step where the fragment stops being a closed system.
                energy_atom_ct = atom_energy(
                    h_env, level_ct.charges, level_ct.mu, level_ct.quad_s, env_ct
                )
                interaction_ff["ct"] = level_ct.energy - level_pol.energy
                d_ct = energy_atom_ct - energy_atom_pol
                interaction_corr["ct"] = d_ct.new_zeros(n_sys).index_add_(0, f2b_, d_ct)
                interaction["ct"] = interaction_ff["ct"] + interaction_corr["ct"]

        # The intra bucket: the bond channel (zero unless 'bond' is a correction channel) plus
        # whatever classical energy survives the range separation. The latter is the term that
        # does not exist in the per-term stack, and it is why a same-fragment pair at
        # 8 Angstrom is no longer inert -- while at bonded range it is switched off to under a
        # kJ/mol per fragment.
        e_intra = e_corr.get("bond", torch.zeros_like(r)) + sum(e_ff.values())
        energy_bond = r.new_zeros(n_frag).index_add_(
            0, pair_frag[intra_idx], (intra * e_intra)[intra_idx]
        )

        e0 = self.reference_energies[h_frag.species_idx]
        energy_ref = e0.new_zeros(n_frag).index_add_(0, frag, e0)
        fragment_energy = energy_ref + res.internal_energy + energy_bond + energy_atom

        energy = fragment_energy.new_zeros(n_sys).index_add_(0, f2b_, fragment_energy)
        for name in interaction:
            energy = energy + interaction[name]

        return UnifiedOutput(
            energy=energy,
            fragment_energy=fragment_energy,
            interaction=interaction,
            interaction_ff=interaction_ff,
            interaction_corr=interaction_corr,
            energy_ref=energy_ref,
            energy_internal=res.internal_energy,
            energy_bond=energy_bond,
            energy_atom=energy_atom,
            energy_atom_pol=energy_atom_pol,
            energy_atom_ct=energy_atom_ct,
            pair_index=pair_index,
            r=r,
            is_intra=is_intra,
            pair_frag=pair_frag,
            e_pair_ff=e_ff,
            e_pair_corr=e_corr,
            gate=gate,
            r0=r0_atom,
            r0_pair=r0_pair,
            alpha=alpha,
            species_idx=h_frag.species_idx,
            log_r0_prior=log_r0_prior,
            log_r0_prior_pair=log_r0_prior_pair,
            environment_norm=(
                None if self.environment is None
                else self.environment.magnitude(h_frag, h_env)
            ),
            elst_env=(
                pool_batch(inter * elst_env_pair)
                if split_elst and self.polarization else None
            ),
            response=res,
            level_pol=level_pol,
            level_ct=level_ct,
            energy_frozen_total=energy_frozen_total,
            environment=env,
            solver=solver or None,
        )


__all__ = [
    "DEFAULT_CLASSICAL",
    "ClassicalSpec",
    "EnvironmentResidual",
    "FragmentStateEmbedding",
    "RangeSeparationHeads",
    "UnifiedOutput",
    "UnifiedPairModel",
]
