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
    E_frag  = E0 + E_internal + sum over INTRA pairs of ( E^elst + E^pauli + E^disp + dE^bond )
    E_total = sum_f E_frag(f) + E_elst + E_pauli + E_disp

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

1. **The bond channel is intra-only.** There is no inter-fragment counterpart, because the
   frozen EDA components are a complete decomposition of the frozen interaction and carry no
   bonding term. So a pair that moves from inter to intra *gains* an energy channel.
2. **The descriptors are fragment-confined**, hence partition-dependent themselves. Relabelling
   a dimer as one fragment changes every atom's ``inv_feats`` (measured: max change 0.19), so
   every classical and correction energy moves too.

Both are correct rather than regrettable: relabelling asserts that two molecules are one, and
the model is supposed to answer differently. Point 1 in particular is where the eventual
charge-transfer energy should surface -- a pair transitioning from inter to intra picks up
exactly the bond-channel term whose growth is what CT looks like in this decomposition.

What *is* exact is the accounting, and ``tests/test_ff_unified.py`` pins it::

    energy == sum_i E0[Z_i] + sum_f E_internal(f)
              + sum_{all pairs} sum_c ( e_ff^c + e_corr^c )
              + sum_{intra pairs} e_corr^bond

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

* ``h_frag`` -> the response solve and the **bond channel**. Together those are
  ``fragment_energy`` up to a kilojoule, and that label is Q-Chem's *isolated*-fragment
  energy, which has no environment dependence to fit.
* ``h_env`` -> everything else: the interaction parameters (``C6``, the Slater exponents, the
  Pauli multipoles, ``r0``) and the three correction channels, which is where effective
  many-body physics such as environment-quenched ``C6`` belongs.

An earlier version evaluated every interaction parameter head on *both* streams and selected
per pair, so that an intra-fragment pair's classical energy used isolated-fragment parameters.
That was over-built. The range separation already switches the intra classical contribution
down to **under 1 kJ/mol per fragment** (:mod:`rsfff.ff.range_priors`), so there is no
meaningful path from the environment into an isolated-fragment label through it -- while the
term that *does* carry that label, the bond channel at ~-640 kJ/mol per fragment, is
unaffected by any gate and is exactly what the split now protects.

The residual cost is that ``fragment_energy`` is no longer environment-independent *by
construction*, only to within that switched-off remainder; ``tests/test_ff_unified.py``
measures it rather than asserting it away.

What is deliberately not here
-----------------------------
The polarized and CT response levels, and the electrostatic-environment features of
``docs/range_separated_mlip.md`` §4.2. The response parameters stay fragment-confined and no
external field enters the solve; see §5.1 for why that ceiling binds on the frozen level only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.heads import mlp
from ..mlip.switch import pairwise_switch
from ..mlip.unified_head import ChannelSpec, UnifiedPairHead
from .damping import fermi_switch
from .dispersion import DispersionParameterHeads, tt_damped_c6_energy
from .electrostatics import slater_elec_pair_energy
from .multipole import build_polytensor, spherical_to_cartesian_quadrupole
from .pairs import union_pairs
from .pauli import PauliMultipoleHeads, slater_pauli_pair_energy
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
    """

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
      from a different one;
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
    """

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
        with torch.no_grad():
            for m in (self.inv_mlp, self.vec_gate, self.equiv_gate):
                if m is not None:
                    m[-1].weight.zero_()
                    m[-1].bias.zero_()

    def forward(self, frag: LambdaFeatures, full: LambdaFeatures) -> LambdaFeatures:
        x = torch.cat((full.inv_feats, self.species_emb(full.species_idx)), dim=-1)
        inv = frag.inv_feats + self.inv_mlp(x)
        vec, equiv = frag.vec_feats, frag.equiv_feats
        if self.vec_gate is not None and vec is not None and full.vec_feats is not None:
            vec = vec + self.vec_gate(x).unsqueeze(1) * full.vec_feats
        if self.equiv_gate is not None and equiv is not None and full.equiv_feats is not None:
            equiv = equiv + self.equiv_gate(x).unsqueeze(1) * full.equiv_feats
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
        log_r0_prior: torch.Tensor,          # (n_species,)
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
        self.register_buffer("log_r0_prior", log_r0_prior.clone())
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
            with torch.no_grad():   # start at exactly the per-element value
                for m in self.r0_mlp.values():
                    m[-1].weight.zero_()
                    m[-1].bias.zero_()

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
        for name in self.channel_names:
            log_r0 = self.log_r0_prior[s] + self.d_log_r0[name][s]
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
    #: (N,) ``log r0`` each atom would have from its element prior alone. The penalty that
    #: keeps a learned ``r0`` near its element value is the residual against this.
    log_r0_prior: torch.Tensor
    #: The **frozen** response: parameters from fragment-confined features, no external
    #: field. Named so the polarized solve lands beside it rather than on top of it.
    response: FragmentResponseOutput
    #: (N,) ``||h_env - h_frag||`` per atom, or None when environment awareness is off. How
    #: much of its surroundings the model has decided each atom's parameters need; zero at
    #: initialization by construction.
    environment_norm: torch.Tensor | None = None

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
        environment: "EnvironmentResidual | None" = None,
        max_rank: int = 1,
        classical: dict[str, ClassicalSpec] | None = None,
        max_num_neighbors: int = 512,
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
        if "bond" not in pair_head.channels:
            raise ValueError(
                "the pair head needs a 'bond' channel; it is the only thing carrying the "
                "intramolecular deformation energy that fragment_energy measures"
            )
        self.featurizer = featurizer
        self.response = response
        self.disp_params = disp_params
        self.pauli_params = pauli_params
        self.range_heads = range_heads
        self.pair_head = pair_head
        self.fragment_state = fragment_state
        self.environment = environment
        self.max_rank = int(max_rank)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(c.cutoff for c in self.classical.values())
        self.register_buffer("reference_energies", reference_energies.clone())

    def _augment(self, feats: LambdaFeatures, batch, frag) -> LambdaFeatures:
        """Concatenate the fragment-state block onto a descriptor's invariants."""
        state = self.fragment_state(
            batch, frag, feats.inv_feats.dtype, feats.inv_feats.device
        )
        if state is None:
            return feats
        return replace(feats, inv_feats=torch.cat((feats.inv_feats, state), dim=-1))

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

    def forward(self, batch) -> UnifiedOutput:
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
        # `h_frag` is fragment-confined and is what the response solve and the bond channel
        # read: `fragment_energy` is an isolated-fragment label, so a 1-body term built on
        # anything else is fitting a function that cannot match its target. `h_env` adds the
        # zero-initialized environment residual and is what the *interaction* parameters
        # read. With `environment` off the two are the same object and everything below runs
        # once.
        if self.environment is None:
            h_frag = h_env = self._augment(self.featurizer(batch, frag), batch, frag)
        else:
            grouped, full = self.featurizer(batch, frag, also_ungrouped=True)
            h_frag = self._augment(grouped, batch, frag)
            h_env = self.environment(h_frag, self._augment(full, batch, frag))

        # --- the frozen response solve, shared by the 1-body and elst channels -----------
        res = self.response(batch, h_frag)

        # --- one pair list, nothing dropped ---------------------------------------------
        pair_index, r, is_intra, pair_frag = union_pairs(
            positions, batch.batch_idx, frag, self.cutoff_max,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG

        # --- the correction trunk, partitioned by routing --------------------------------
        # Each pair is scored once, on the stream its label demands. Intra pairs read
        # `h_frag` and take the bond channel; inter pairs read `h_env` and take the three
        # interaction corrections. Both partitions also emit a per-pair range-separation
        # deviation, so the *within-fragment* range separation is a function of
        # fragment-confined features alone.
        #
        # Intra pairs deliberately do *not* get the interaction corrections. Their classical
        # counterpart is what the range separation is deciding about, and a correction there
        # would be an unconstrained second bond term, degenerate with the real one.
        sp = h_frag.species_idx
        e_corr = {name: torch.zeros_like(r) for name in self.pair_head.channels}
        d_log_r0 = {name: torch.zeros_like(r) for name in self.classical}
        intra_idx = is_intra.nonzero().squeeze(-1)
        inter_idx = (~is_intra).nonzero().squeeze(-1)
        for idx, feats, want in (
            (intra_idx, h_frag, ("bond",)),
            (inter_idx, h_env, tuple(self.classical)),
        ):
            if not idx.numel():
                continue
            a, b_ = i[idx], j[idx]
            energies, devs = self.pair_head(
                feats.inv_feats[a], feats.inv_feats[b_], sp[a], sp[b_], r[idx]
            )
            for name in want:
                e_corr[name] = e_corr[name].index_copy(0, idx, energies[name])
            for name in devs:
                d_log_r0[name] = d_log_r0[name].index_copy(0, idx, devs[name])

        # --- classical backbones, every pair ---------------------------------------------
        # ``r0`` is a per-element base (the covalent-distance knowledge of
        # :mod:`rsfff.ff.range_priors`) times a learned per-pair correction. The base cannot
        # tell topologically distinct pairs of the same elements apart; the correction can,
        # and that is the whole reason it exists -- see :class:`UnifiedPairHead`.
        r0_atom, alpha = self.range_heads(h_env.inv_feats, h_env.species_idx)
        gate, r0_pair = {}, {}
        for name, spec in self.classical.items():
            log_r0_ij = (
                0.5 * (r0_atom[name][i].log() + r0_atom[name][j].log()) + d_log_r0[name]
            )
            r0_pair[name] = log_r0_ij.exp()
            gate[name] = fermi_switch(r, r0_pair[name], alpha[name]) * pairwise_switch(
                r, spec.cutoff - spec.taper_width, spec.cutoff
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

        # The Pauli and dispersion parameters are the *only* remaining path by which the
        # environment could reach an isolated-fragment label, so they are the only ones
        # evaluated on both streams. Everything else is already safe: the electrostatic
        # multipoles come from the fragment-confined response, the per-atom `r0` base has no
        # environment dependence unless `environment_r0` is on, and the per-pair `r0`
        # correction for intra pairs is read from the fragment-confined trunk.
        #
        # This costs two per-atom MLP evaluations and it is worth it. Left on `h_env` alone,
        # a fragment's energy inside a cluster differs from that fragment alone by ~1.1 kJ/mol
        # at the environment strength this model actually trains to -- roughly half the target
        # `ob_mae`, i.e. a systematic error large enough to be mistaken for fit error.
        def _select(frag_side, env_side, index):
            if frag_side is env_side:
                return frag_side[index]
            wide = is_intra.reshape((-1,) + (1,) * (frag_side.dim() - 1))
            return torch.where(wide, frag_side[index], env_side[index])

        poly_f, b_p_f = self._pauli_multipoles(h_frag)
        poly_e, b_p_e = (
            (poly_f, b_p_f) if h_env is h_frag else self._pauli_multipoles(h_env)
        )
        b_p_ij = (0.5 * (
            _select(b_p_f, b_p_e, i).log() + _select(b_p_f, b_p_e, j).log()
        )).exp()
        e_pauli = slater_pauli_pair_energy(
            dr_au, r_au,
            _select(poly_f, poly_e, i), _select(poly_f, poly_e, j),
            b_p_ij, max_rank=self.max_rank,
        )

        c6_f, b_d_f = self.disp_params(h_frag.inv_feats, h_frag.species_idx)
        c6_e, b_d_e = (
            (c6_f, b_d_f) if h_env is h_frag
            else self.disp_params(h_env.inv_feats, h_env.species_idx)
        )
        c6_ij = (0.5 * (
            _select(c6_f, c6_e, i).log() + _select(c6_f, c6_e, j).log()
        )).exp()
        b_d_ij = (0.5 * (
            _select(b_d_f, b_d_e, i).log() + _select(b_d_f, b_d_e, j).log()
        )).exp()
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

        # The intra bucket: the bond channel plus whatever classical energy survives the
        # range separation. The latter is the term that does not exist in the per-term stack,
        # and it is why a same-fragment pair at 8 Angstrom is no longer inert -- while at
        # bonded range it is switched off to under a kJ/mol per fragment.
        e_intra = e_corr["bond"] + sum(e_ff.values())
        energy_bond = r.new_zeros(n_frag).index_add_(
            0, pair_frag[intra_idx], (intra * e_intra)[intra_idx]
        )

        e_atom = self.reference_energies[h_frag.species_idx]
        energy_ref = e_atom.new_zeros(n_frag).index_add_(0, frag, e_atom)
        fragment_energy = energy_ref + res.internal_energy + energy_bond

        f2b = batch.fragment_to_batch
        if f2b is None:
            f2b = batch.batch_idx.new_zeros(n_frag).scatter_(0, frag, batch.batch_idx)
        energy = fragment_energy.new_zeros(n_sys).index_add_(0, f2b, fragment_energy)
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
            log_r0_prior=self.range_heads.log_r0_prior[h_frag.species_idx],
            environment_norm=(
                None if self.environment is None
                else self.environment.magnitude(h_frag, h_env)
            ),
            response=res,
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
