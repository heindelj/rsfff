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

What is deliberately not here
-----------------------------
Environment-aware descriptors (``h_env``), the polarized and CT response levels, and the
electrostatic-environment features of ``docs/range_separated_mlip.md`` §4.2. ``h_env`` is
plumbed as a separate name that currently aliases ``h_frag``, so wiring it later is an
addition; see :class:`UnifiedPairModel.forward`.
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
    r0: dict[str, torch.Tensor]               # (N,) per-atom midpoint, per channel
    alpha: dict[str, torch.Tensor]            # () per-channel width
    species_idx: torch.Tensor                 # (N,) index into neighbor_types
    #: (N,) ``log r0`` each atom would have from its element prior alone. The penalty that
    #: keeps a learned ``r0`` near its element value is the residual against this.
    log_r0_prior: torch.Tensor
    #: The **frozen** response: parameters from fragment-confined features, no external
    #: field. Named so the polarized solve lands beside it rather than on top of it.
    response: FragmentResponseOutput

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
        self.max_rank = int(max_rank)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(c.cutoff for c in self.classical.values())
        self.register_buffer("reference_energies", reference_energies.clone())

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
        # Fragment-confined, which is what keeps the 1-body term exactly one-body and the
        # elst channel exactly two-body. `h_env` is the hook for the environment-aware
        # descriptor (one extra unmasked scatter over the same geometric basis, plus a
        # zero-init residual); it aliases `h_frag` until that is wired, so the split costs
        # nothing today but the consumers are already separated.
        feats = self.featurizer(batch, frag)
        state = self.fragment_state(batch, frag, positions.dtype, positions.device)
        if state is not None:
            feats = replace(feats, inv_feats=torch.cat((feats.inv_feats, state), dim=-1))
        h_frag = feats
        h_env = feats

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

        # --- classical backbones, every pair --------------------------------------------
        r0_atom, alpha = self.range_heads(h_env.inv_feats, h_env.species_idx)
        gate = {}
        for name, spec in self.classical.items():
            r0_ij = (0.5 * (r0_atom[name][i].log() + r0_atom[name][j].log())).exp()
            gate[name] = fermi_switch(r, r0_ij, alpha[name]) * pairwise_switch(
                r, spec.cutoff - spec.taper_width, spec.cutoff
            )

        quad_c = None if res.quad_s is None else spherical_to_cartesian_quadrupole(res.quad_s)
        m_real = build_polytensor(res.charges, res.mu, quad_c, max_rank=self.max_rank)
        m_shell = build_polytensor(res.charges - res.z, res.mu, quad_c, max_rank=self.max_rank)
        m_nuc = build_polytensor(res.z, None, None, max_rank=self.max_rank)
        e_point, e_pen = slater_elec_pair_energy(
            dr_au, r_au, m_real, m_shell, m_nuc, res.b, pair_index, max_rank=self.max_rank
        )

        q_p, b_p, mu_p, quad_p = self.pauli_params(
            h_env.inv_feats, h_env.species_idx, h_env.vec_feats, h_env.equiv_feats
        )
        poly_p = build_polytensor(
            q_p, mu_p,
            None if quad_p is None else spherical_to_cartesian_quadrupole(quad_p),
            max_rank=self.max_rank,
        )
        b_p_ij = (0.5 * (b_p[i].log() + b_p[j].log())).exp()
        e_pauli = slater_pauli_pair_energy(
            dr_au, r_au, poly_p[i], poly_p[j], b_p_ij, max_rank=self.max_rank
        )

        c6, b_d = self.disp_params(h_env.inv_feats, h_env.species_idx)
        c6_ij = (0.5 * (c6[i].log() + c6[j].log())).exp()
        b_d_ij = (0.5 * (b_d[i].log() + b_d[j].log())).exp()
        e_disp = tt_damped_c6_energy(r, c6_ij, b_d_ij)

        e_ff = {
            "elst": gate["elst"] * (e_point + e_pen),
            "pauli": gate["pauli"] * e_pauli,
            "disp": gate["disp"] * e_disp,
        }

        # --- the shared correction trunk, every pair, every channel ----------------------
        e_corr = self.pair_head(h_env.inv_feats, h_env.species_idx, pair_index, r)

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

        # The intra bucket takes the same classical channels plus the bond correction. This
        # is the term that does not exist in the per-term stack, and it is why a same-fragment
        # pair at 8 Angstrom is no longer inert.
        e_intra = e_corr["bond"] + sum(e_ff[n] + e_corr[n] for n in e_ff)
        sel = is_intra.nonzero().squeeze(-1)
        energy_bond = r.new_zeros(n_frag).index_add_(
            0, pair_frag[sel], (intra * e_intra)[sel]
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
            alpha=alpha,
            species_idx=h_frag.species_idx,
            log_r0_prior=self.range_heads.log_r0_prior[h_frag.species_idx],
            response=res,
        )


__all__ = [
    "DEFAULT_CLASSICAL",
    "ClassicalSpec",
    "FragmentStateEmbedding",
    "RangeSeparationHeads",
    "UnifiedOutput",
    "UnifiedPairModel",
]
