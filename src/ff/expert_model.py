"""The fragment model: two slots, one pair list, and a one-body sector nothing can reach.

``docs/fff_v2.md`` assembled, v4. Every parameter comes from the one shared decoder
(:mod:`rsfff.ff.decoder`) evaluated on two slots plus the fragment's state block, and every
quantity therefore exists at two values::

    theta   = D( h , eta , k )      the in-medium parameter
    theta_0 = D( h ,  0  , k )      the isolated-fragment parameter

``k = E(Q_f, 2S_f, n_f)`` is the fragment's state (:mod:`rsfff.ff.fragment_state`): charge,
multiplicity and composition, carrying what a lambda-SOAP descriptor cannot know. It is *not* a
learned latent, and that is deliberate -- a latent has no canonical frame, so two of them have
no defined midpoint, while ``Q = 0.5`` is a point one network can be asked about. That is what
makes a fragmentation crossover a state being lifted off an integer rather than two unrelated
encodings being averaged (:mod:`rsfff.ff.partition`).

``eta`` is the cross-fragment lambda-SOAP, identically zero when a fragment is alone
(:meth:`rsfff.features.features.FlatLambdaSOAPFeaturizer.forward`, ``also_cross=True``), so
``theta_0`` is exactly what the model says about a fragment on its own -- not by an anchoring
subtraction and not only at initialization.

The accounting
--------------
::

    E_f      = sum_i E0[Z_i] + E_internal(theta_0) + sum_intra gate(theta_0) E_class(theta_0)
                             + sum_i E_bond(h_i, M_i, Phi_i^intra ; theta_0)

    E_inter  = sum_{inter, c} gate_c E_class^c        -> eda_cls_elec / mod_pauli / disp

    E_ind    = [ coupled solve with theta ] - [ same functional at frozen multipoles ]
             + sum_i [ E_bond(...; theta) - E_bond(...; theta_0) ]
             + the electrostatic channel's own theta - theta_0 difference

    E_total  = sum_f E_f + sum_c E_inter^c + E_ind

**No ``eta`` appears anywhere in ``E_f``.** That is the property the whole design exists for:
``fragment_energy`` is exactly the isolated-fragment quantity its label is, at every geometry,
at any separation from any neighbour, at every point in training. It needs no freeze, because
there is no path for the environment to take. ``tests/test_one_body_isolation.py`` is where that
is checked rather than asserted.

Which slot each channel reads, and why it is not uniform
-------------------------------------------------------
Intra-fragment pairs always read ``theta_0``. Inter-fragment pairs read whichever slot their
*label* is a function of, which differs by channel and is a fact about ALMO-EDA rather than a
modelling choice:

* ``elst`` -> ``theta_0``. ``eda_cls_elec`` is the Coulomb interaction between superimposed
  frozen monomer densities, which is **rigorously pairwise**: it is a function of the two
  fragments alone and of nothing else. Its ``theta - theta_0`` difference is therefore not
  electrostatics at all -- it is polarization, and it is booked to induction. Measured on a
  checkpoint that left it in place, that difference was 1.1 / 2.3 / 2.7 / 3.2 kJ/mol per frame
  on w2 / w3 / w4 / w5, four to six times the channel's own MAE, and nonzero even on dimers
  where there is no many-body content to explain it.
* ``pauli``, ``disp`` -> ``theta``. Neither label is pairwise. The modified Pauli term
  antisymmetrizes the product of *all* the monomer densities and the dispersion is a
  supersystem difference, so both carry genuine many-body content. An environment-quenched
  ``C6`` and an environment-softened Slater exponent are what that content looks like here, and
  denying them the environment slot would make the dispersion rigorously two-body -- a useful
  ablation (:attr:`ClassicalSpec.environment` off) and the wrong physics.

So the per-channel slot is data, not a special case, and the hand-written electrostatic
re-scoring the previous model needed is gone: every channel gets both evaluations by
construction.

What is deliberately not here
-----------------------------
**Neural pair corrections.** No per-pair energy readout and no per-pair ``r0`` deviation. The
neural content lives in the parameters an expert emits and the classical forms are evaluated as
written. A readout on top is a second, unlabeled model competing for the same energy, which is
how -39 kJ/mol per fragment of "intramolecular dispersion" once ended up between bonded atoms.

**The variational coupling** of ``docs/atomic_response_functional.md`` §2, where ``E_bond`` sits
inside the functional and ``dE/dM = 0`` includes it. Here ``E_bond`` is evaluated at the
*converged* multipoles, so the coupling runs one way. That is safe because
:class:`rsfff.ff.coupled_solve._CoupledSolve`'s adjoint exists precisely to let a
non-variational consumer of ``M*`` differentiate correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ..mlip.switch import pairwise_switch
from .bond_energy import FragmentBondEnergy  # noqa: F401  (re-exported for builders)
from .damping import fermi_switch
from .dispersion import tt_damped_c6_energy
from .electrostatics import slater_elec_pair_energy
from .environment import OneBodyEnvironment, electrostatic_environment
from .decoder import ParameterDecoder  # noqa: F401  (re-exported for builders)
from .fragment_state import FragmentStateEmbedding
from .partition import element_counts, mixed_state
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    spherical_to_cartesian_quadrupole,
)
from .pairs import intra_fragment_channels, union_channels, union_pairs
from .pauli import slater_pauli_pair_energy
from .polarization import LevelOutput, coupled_response
from .response import FragmentResponseOutput, ResponseParameters, solve_frozen  # noqa: F401
from .slots import SlotFeatures
from .units import BOHR_ANG

__all__ = [
    "ClassicalSpec",
    "DEFAULT_CLASSICAL",
    "Emission",
    "ExpertOutput",
    "FragmentExpertModel",
]


@dataclass(frozen=True)
class ClassicalSpec:
    """Reach of one classical channel, and which slot its inter-fragment form reads.

    ``cutoff`` is where the channel is tapered to exactly zero and ``taper_width`` how long that
    takes. These are per channel because the terms decay at wildly different rates: the
    electrostatics has a genuine ``1/r`` tail and truncating it early is a real error, while the
    Slater Pauli form is ~1e-8 Hartree by 7 Angstrom.

    ``environment`` says whether *inter*-fragment pairs read ``theta`` or ``theta_0``. See the
    module docstring: it follows from what each ALMO-EDA label is a function of. Setting it
    ``False`` on every channel is the two-body ablation. Intra pairs read ``theta_0`` whatever
    this says.

    ``to_induction`` books the channel's ``theta - theta_0`` difference to the induction label
    rather than dropping it. It is meaningful only when ``environment`` is off -- with the
    environment live the difference is already inside the channel's own energy.
    """

    cutoff: float
    taper_width: float = 1.0
    environment: bool = True
    to_induction: bool = False

    def __post_init__(self) -> None:
        if not self.cutoff > self.taper_width:
            raise ValueError(
                f"ClassicalSpec needs cutoff > taper_width, got {self.cutoff}, "
                f"{self.taper_width}"
            )
        if self.environment and self.to_induction:
            raise ValueError(
                "to_induction books a channel's theta - theta_0 difference to induction, "
                "which only makes sense when the channel itself reads theta_0 "
                "(environment=False); with environment=True that difference is already in the "
                "channel's own energy and booking it again would double count it"
            )


#: ``elst`` is fragment-confined because its label is rigorously pairwise, and its environment
#: difference is polarization; the other two carry real many-body content in their labels.
DEFAULT_CLASSICAL: dict[str, ClassicalSpec] = {
    "elst": ClassicalSpec(12.0, environment=False, to_induction=True),
    "pauli": ClassicalSpec(7.0),
    "disp": ClassicalSpec(10.0),
}


@dataclass
class Emission:
    """One fragmentation's parameters, before any solve or pair list. See ``emit``.

    Every field is either per atom or per channel of *this* decomposition. The mixture
    (``docs/fff_v2.md`` §8) combines several of these -- the per-atom fields by a weighted sum,
    because a ``C6`` is a ``C6`` whichever expert emitted it, and the compliances onto the
    union channel graph, because a channel only one decomposition opens still carries its
    weight.
    """

    slots: SlotFeatures
    #: ``h`` alone (with the fragment-state block), and ``[h | eta]``. The decoder reads the
    #: first for ``theta_0`` and the second for ``theta``; the narrow-input convention is what
    #: makes that a property of the *input* rather than a flag.
    iso: object
    joined: object
    has_env: bool
    channels: torch.Tensor                    # (2, Nc) this decomposition's channel graph


@dataclass
class ExpertOutput:
    """Per-frame totals, per-fragment energies, and the full per-pair breakdown."""

    energy: torch.Tensor                      # (B,) the total, all channels
    fragment_energy: torch.Tensor             # (F,) -> batch.fragment_energy
    #: Inter-fragment channel sums, the EDA targets. Keys: elst, pauli, disp, induction.
    interaction: dict[str, torch.Tensor]      # each (B,)
    energy_ref: torch.Tensor                  # (F,) sum_i E0[Z_i], frozen
    energy_internal: torch.Tensor             # (F,) the response solve's internal energy
    energy_intra: torch.Tensor                # (F,) intra pairs, classical, at theta_0
    #: (F,) ``sum_i E_bond(h_i, M^frozen, Phi^intra ; theta_0)`` -- the covalent bonding energy
    #: inside ``fragment_energy``. Its split against ``energy_internal`` is unlabeled, so watch
    #: their spread rather than their values.
    energy_bond: torch.Tensor                 # (F,)
    pair_index: torch.Tensor                  # (2, P)
    r: torch.Tensor                           # (P,) Angstrom
    is_intra: torch.Tensor                    # (P,) bool, routing only -- never a gate
    pair_frag: torch.Tensor                   # (P,) fragment id, -1 for inter pairs
    e_pair: dict[str, torch.Tensor]           # (P,) per channel, gate * classical
    gate: dict[str, torch.Tensor]             # (P,) fermi * taper, per classical channel
    r0: dict[str, torch.Tensor]               # (N,) per-atom midpoint at theta_0
    r0_pair: dict[str, torch.Tensor]          # (P,) the midpoint each pair actually used
    alpha: dict[str, torch.Tensor]            # () per-channel width
    species_idx: torch.Tensor                 # (N,)
    log_r0_prior: dict[str, torch.Tensor]     # (N,) per channel, the element prior
    log_r0_prior_pair: dict[str, torch.Tensor]  # (P,) combined by geometric mean
    #: The **frozen** response: parameters from the isolated slot, no external field.
    response: FragmentResponseOutput
    #: (N,) ``||eta||`` per atom. Zero for an isolated fragment by construction, so a rising
    #: value is the model asking for many-body content and an exact zero on a cluster is a bug.
    env_norm: torch.Tensor
    #: ``{quantity: ()}`` -- how far each parameter moved between ``theta_0`` and ``theta``,
    #: in log space for the positive ones. **The number this design exists to produce**: it is
    #: what ``L_env`` penalizes and what says whether a channel's explanation lives in the
    #: fragment or in its surroundings.
    env_shift: dict[str, torch.Tensor]
    #: (N,) ``energy_bond`` before it is pooled to fragments. The pooled form hides *where* the
    #: covalent energy sits, which is exactly the question a proton transfer asks -- what became
    #: of the two oxygens and the shared hydrogen individually.
    energy_bond_atom: torch.Tensor | None = None
    #: (F,) each fragment's applicability score, or None when the head is not built. Untrained.
    applicability: torch.Tensor | None = None
    #: (B,) the electrostatic channel's ``theta - theta_0`` difference, booked to induction.
    #: Exposed on its own because the size of what moved is the thing worth watching.
    elst_env: torch.Tensor | None = None
    level_ind: "LevelOutput | None" = None
    #: (F,) the bond energy at the induced state and the in-medium parameters. Its difference
    #: from ``energy_bond`` is what induction receives, so the two telescope.
    energy_bond_ind: torch.Tensor | None = None
    #: (F,) the same evaluation at ``theta_0``. Nothing consumes it; it exists so the *slot*
    #: swap is measurable separately from the state relaxation -- the quantity that was 100% of
    #: the old ``ct`` channel and that nothing in the printed line reported.
    energy_bond_ind_iso: torch.Tensor | None = None
    energy_frozen_total: torch.Tensor | None = None
    environment: OneBodyEnvironment | None = None
    solver: dict[str, tuple] | None = None

    # Forwarded from the frozen response so this satisfies the same duck type as
    # `ElectrostaticsOutput` and can be handed straight to `rsfff.train.loss`. These must keep
    # pointing at the *frozen* solve: the fragment multipole labels are frozen-monomer values.
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
        return self.response.polarizability


class FragmentExpertModel(nn.Module):
    """Featurizer + expert bank + range separation + coupled solve, as one term.

    Args
    ----
    featurizer  : produces both slots in one neighbor search (``also_cross=True``).
    fragment_state : the ``(Q_f, 2S_f, n_f)`` block, one embedding for the whole model.
    decoder     : the one :class:`rsfff.ff.decoder.ParameterDecoder`.
    reference_energies : (n_species,) isolated-atom energies in Hartree, frozen. The one piece
                  of external information the one-body sector gets, and what makes
                  ``fragment_energy`` an atomization energy.
    max_rank    : multipole rank, must match the response heads'.
    classical   : per-channel reach and slot choice.
    environment_features : build the environment slot at all. ``False`` makes every parameter
                  fragment-confined and the model rigorously two-body -- the ablation.
    """

    def __init__(
        self,
        featurizer,
        fragment_state: FragmentStateEmbedding,
        decoder: "ParameterDecoder",
        reference_energies: torch.Tensor,
        *,
        max_rank: int = 2,
        applicability=None,
        classical: dict[str, ClassicalSpec] | None = None,
        environment_features: bool = True,
        max_num_neighbors: int = 512,
        induction: bool = False,
        cg_rtol: float = 1.0e-9,
        cg_atol: float = 1.0e-12,
        cg_maxiter: int = 100,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        #: The fragment's state -- ``(Q, 2S, n)`` -- as a per-atom block appended to the
        #: **fragment** slot. The one thing a lambda-SOAP descriptor cannot know about a
        #: fragment, and the object a mixture actually moves (:mod:`rsfff.ff.partition`).
        self.fragment_state = fragment_state
        #: The one shared decoder. Every parameter in the model comes out of it, so an H3O+
        #: oxygen and an H2O oxygen are separated by their *input* and not by which network
        #: ran -- see :mod:`rsfff.ff.decoder`.
        self.decoder = decoder
        #: Scores whether *this* fragmentation is a good description of the system. Reads the
        #: joined slot and contributes nothing to the energy; ``None`` when not built.
        self.applicability = applicability
        self.classical = dict(classical or DEFAULT_CLASSICAL)
        self.max_rank = int(max_rank)
        self.environment_features = bool(environment_features)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(c.cutoff for c in self.classical.values())
        self.induction = bool(induction)
        self.cg = dict(rtol=float(cg_rtol), atol=float(cg_atol), maxiter=int(cg_maxiter))
        self.register_buffer("reference_energies", reference_energies.clone())

        missing = set(self.decoder.range_heads.channel_names) - set(self.classical)
        if missing:
            raise ValueError(
                f"the decoder range-separates channels with no classical form: "
                f"{sorted(missing)}"
            )
        if int(max_rank) != int(self.decoder.response.max_rank):
            raise ValueError(
                f"max_rank {max_rank} does not match the decoder's response heads' "
                f"{self.decoder.response.max_rank}; the heads decide which multipoles exist "
                f"and this decides which ones the interaction tensor carries"
            )

    # -- features ------------------------------------------------------------------------

    def slots(self, batch, frag, *, want_env: bool = True, edge_weight=None, state=None):
        """``SlotFeatures`` -- both descriptors, with the fragment-state block on the first.

        ``want_env=False`` skips the cross descriptor, for callers that only need the frozen
        response (the free-atom and monomer paths). The fragment slot is bit-identical either
        way only up to scatter ordering; see ``FlatLambdaSOAPFeaturizer.forward``.

        ``edge_weight`` is the **soft partition** ``s_e`` a mixture supplies
        (:func:`rsfff.ff.partition.soft_partition`): the boolean membership test becomes a
        weight, and the two descriptors are the power spectra of ``A(s)`` and ``A(1-s)``. It
        requires the environment slot, since softening a boundary is meaningless without one.

        ``state`` is the pre-mixed per-atom fragment-state block, for the same caller -- a
        mixture's ``(Q*, 2S*, n*)`` are fractional and come from
        :func:`rsfff.ff.partition.mixed_state` rather than from ``batch``. Left ``None`` the
        block is built from the batch's definite labels.

        The state block joins the **fragment** slot and never the environment one: charge,
        multiplicity and composition are properties of the fragment.
        """
        want = self.environment_features and want_env
        if edge_weight is not None and not want:
            raise ValueError(
                "slots got an edge_weight but no environment slot; a soft partition moves "
                "the boundary between the fragment and its environment and there is no "
                "boundary to move when environment_features is off"
            )
        if not want:
            grouped, cross = self.featurizer(batch, frag), None
        elif edge_weight is None:
            grouped, cross = self.featurizer(batch, frag, also_cross=True)
        else:
            grouped, cross = self.featurizer(
                batch, frag, also_cross=True, edge_weight=edge_weight
            )
        return SlotFeatures(self._augment(grouped, batch, frag, state), cross)

    def _augment(self, feats, batch, frag, state=None):
        """Concatenate the ``(Q_f, 2S_f, n_f)`` block onto the fragment slot's invariants.

        One embedding for the whole model, not one per composition. That is the v4 change and
        the reason it matters is that a *shared* net over continuous labels is the only form a
        mixture can interpolate: two per-composition encodings of "charge" have no defined
        midpoint, while ``Q = 0.5`` is a point one net can be asked about.
        """
        if self.fragment_state.net is None:
            return feats
        if state is None:
            state = self.fragment_state(
                batch, frag, feats.inv_feats.dtype, feats.inv_feats.device,
                element_counts=element_counts(
                    frag, feats.species_idx, self.fragment_state.n_species
                ),
            )
        return replace(feats, inv_feats=torch.cat((feats.inv_feats, state), dim=-1))

    def frozen_polarizability(self, batch) -> torch.Tensor | None:
        """``(F, 3, 3)`` each fragment's frozen molecular polarizability, without the pair model.

        The same tensor :attr:`ExpertOutput.polarizability` carries, by the short route: the
        featurizer, then the frozen response solve on the isolated slot. Nothing else
        contributes to it, so this is the same computation with the unused three quarters left
        out rather than an approximation.
        """
        frag = batch.fragment_idx
        if frag is None:
            raise ValueError(
                "frozen_polarizability is a per-fragment quantity but batch.fragment_idx is "
                "None; the extxyz needs a `fragment_idx` column"
            )
        slots = self.slots(batch, frag, want_env=False)
        iso = slots.isolated()
        rp = self.decode_response(
            batch, iso, iso.species_idx, batch.batch_idx,
            bond_index=intra_fragment_channels(frag)[0],
        )
        return solve_frozen(
            rp, batch,
            direct_multipoles=self._direct_multipoles,
            with_polarizability=True,
        ).polarizability

    @property
    def _direct_multipoles(self) -> bool:
        """Whether the response heads emit permanent multipoles directly.

        There is one decoder, so there is one answer. The v2 model had to *check* that every
        expert agreed, because disagreement would have made
        :func:`rsfff.ff.response.solve_frozen` read one expert's permanent dipoles as another's
        dipole drives -- silently, and with plausible magnitudes. That failure mode no longer
        exists.
        """
        return self.decoder.response.direct_multipoles

    # -- parameters ----------------------------------------------------------------------

    def _pauli_from(self, feats, species_idx):
        """``(polytensor (N, K), b (N,))`` for the Slater Pauli term, from one slot."""
        q, b, mu, quad_s = self.decoder.pauli(feats, species_idx)
        poly = build_polytensor(
            q, mu,
            None if quad_s is None else spherical_to_cartesian_quadrupole(quad_s),
            max_rank=self.max_rank,
        )
        return poly, b

    def emit(self, batch, frag, *, bond_index=None, edge_weight=None, state=None) -> "Emission":
        """Every parameter this model's experts emit for **one** fragmentation, unassembled.

        ``forward`` assembles an energy from these; :func:`rsfff.ff.mixture_model.
        mixture_forward` mixes several of them first and assembles once. Factoring the
        emission out is what lets the mixture reuse the parameterization exactly rather than
        reimplementing it -- the thing that would otherwise drift.

        No solve happens here and no pair list is built. Both are properties of the
        *assembly*: a mixture solves on the union of its decompositions' channel graphs, so a
        solve done per decomposition would be the wrong one and would have to be discarded.

        ``bond_index`` overrides the intra-fragment channel enumeration, which the mixture
        needs because :func:`rsfff.ff.pairs.intra_fragment_channels` requires atoms grouped by
        fragment and no atom order satisfies that for more than one decomposition at once.
        """
        slots = self.slots(batch, frag, edge_weight=edge_weight, state=state)
        if bond_index is None:
            bond_index, _ = intra_fragment_channels(frag)
        return Emission(
            slots=slots, iso=slots.isolated(), joined=slots.joined(),
            has_env=slots.dims.has_env, channels=bond_index,
        )

    def decode_response(self, batch, feats, species_idx, batch_idx, *, bond_index):
        """Response parameters from a descriptor. The decoder is shared, so this runs once.

        ``species_idx``/``batch_idx`` are taken separately rather than off ``feats`` so a
        mixture can pass a frame-blocked indexing that its own ``feats`` does not carry.
        """
        return self.decoder.response.response_parameters(
            batch,
            replace(feats, species_idx=species_idx, batch_idx=batch_idx),
            bond_index=bond_index,
        )

    def forward(
        self,
        batch,
        *,
        with_polarizability: bool = False,
        with_induction: bool | None = None,
    ) -> ExpertOutput:
        """``with_polarizability`` also asks the frozen solve for the molecular polarizability,
        which only the isolated-monomer anchor has a label for.

        ``with_induction=False`` skips the coupled solve for this call. **The isolated training
        streams use it, and not only to save the solve.** For a batch of lone fragments the
        three classical channels are empty sums, but induction is not quite zero: the coupled
        level minimizes with the intramolecular electrostatics *inside* the functional while the
        frozen level adds it afterwards, so ``M^ind != M^frozen`` even with nothing nearby, and
        the fragment relaxes against its own field. ALMO-EDA reports ``eda_pol = eda_ct = 0``
        for an isolated monomer by definition, so that residue is an artifact of how the levels
        are defined rather than an energy the label knows about -- v1 measured it at 2.6e-5 e
        and a few hundredths of a kJ/mol on a trained model.

        Leaving it in would also quietly break the monomer force term, which takes its gradient
        from ``out.energy`` on the premise that a one-fragment system has
        ``energy == fragment_energy`` (:func:`rsfff.train.loss.onebody_anchor_loss`). With this
        off that premise is exact again.
        """
        induction = self.induction if with_induction is None else bool(with_induction)
        if batch.fragment_idx is None:
            raise ValueError(
                "the expert model routes pair energies to per-fragment and per-frame labels "
                "but batch.fragment_idx is None; the extxyz needs a `fragment_idx` column"
            )
        positions = batch.positions
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)
        n_sys = int(batch.n_systems)

        # One featurizer call, one fragment-state block, one decoder. Nothing is gathered per
        # composition and stitched back: an H3O+ oxygen and an H2O oxygen differ by their
        # *input* to the same networks, which is what lets a mixture move between them.
        slots = self.slots(batch, frag)
        iso, joined = slots.isolated(), slots.joined()
        has_env = slots.dims.has_env

        # --- the frozen response solve, on the ISOLATED slot ----------------------------
        # `fragment_energy` is an isolated-fragment label, so a one-body term built on anything
        # else is fitting a function that cannot match its target.
        res = solve_frozen(
            self.decode_response(
                batch, iso, iso.species_idx, batch.batch_idx,
                bond_index=intra_fragment_channels(frag)[0],
            ),
            batch,
            direct_multipoles=self._direct_multipoles,
            with_polarizability=with_polarizability,
        )

        # --- one pair list, nothing dropped ---------------------------------------------
        pair_index, r, is_intra, pair_frag = union_pairs(
            positions, batch.batch_idx, frag, self.cutoff_max,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG
        inter = (~is_intra).to(r.dtype)
        intra = is_intra.to(r.dtype)
        pair_batch = batch.batch_idx[i]

        def pool_batch(x):
            return x.new_zeros(n_sys).index_add_(0, pair_batch, x)

        # --- every parameter, at both evaluations ---------------------------------------
        # `theta_0 = D(k_0)` and `theta = D(k)`. Both come from the **one** shared decoder, so
        # what separates them is which key it was handed and nothing else -- the per-expert
        # fan-out that used to wrap every head is gone with the per-expert heads.
        env_shift: dict[str, torch.Tensor] = {}
        species = iso.species_idx

        # `r0` reads the element and nothing else, so it is identical at both evaluations and
        # identical across compositions. That is the point: in v2 it was per expert, and an
        # oxygen's `r0_elst` jumped 0.905 -> 1.13 Angstrom as the description swapped, moving
        # the Fermi gate a long way for no physical reason. It cannot now.
        r0_iso, alpha = self.decoder.r0(species)
        r0_env = r0_iso

        pauli_iso = self._pauli_from(iso, species)
        pauli_env = pauli_iso if not has_env else self._pauli_from(joined, species)
        disp_iso = self.decoder.dispersion(iso, species)
        disp_env = disp_iso if not has_env else self.decoder.dispersion(joined, species)
        if has_env:
            env_shift["c6"] = (disp_env[0].log() - disp_iso[0].log()).abs().mean()
            env_shift["b_disp"] = (disp_env[1].log() - disp_iso[1].log()).abs().mean()
            env_shift["pauli_multipole"] = (
                pauli_env[0] - pauli_iso[0]
            ).norm(dim=-1).mean()
            env_shift["b_pauli"] = (pauli_env[1].log() - pauli_iso[1].log()).abs().mean()

        def per_pair(iso_atom, env_atom, index, *, use_env: bool):
            """Select each pair's per-atom parameter: intra always isolated, inter by channel."""
            a, b_ = iso_atom[index], env_atom[index]
            if not use_env or a is b_:
                return a
            mask = is_intra.reshape(-1, *([1] * (a.dim() - 1)))
            return torch.where(mask, a, b_)

        # Per element and per channel, a lookup rather than a head, and now a single shared
        # table rather than one per expert.
        channel_names = tuple(self.decoder.range_heads.channel_names)
        log_r0_prior = {
            name: self.decoder.range_heads.log_r0_prior[c][species]
            for c, name in enumerate(channel_names)
        }

        def build_gate(name, spec, *, use_env: bool):
            base_i = per_pair(r0_iso[name], r0_env[name], i, use_env=use_env)
            base_j = per_pair(r0_iso[name], r0_env[name], j, use_env=use_env)
            r0_ij = (0.5 * (base_i.log() + base_j.log())).exp()
            switch = fermi_switch(r, r0_ij, alpha[name]) * pairwise_switch(
                r, spec.cutoff - spec.taper_width, spec.cutoff
            )
            return switch, r0_ij

        gate, r0_pair, log_r0_prior_pair = {}, {}, {}
        for name, spec in self.classical.items():
            gate[name], r0_pair[name] = build_gate(name, spec, use_env=spec.environment)
            if name in log_r0_prior:
                log_r0_prior_pair[name] = 0.5 * (
                    log_r0_prior[name][i] + log_r0_prior[name][j]
                )

        # --- classical backbones --------------------------------------------------------
        # The multipoles come from the frozen response, which is isolated by construction, so
        # `cls_elec` is a function of the two fragments alone -- which is what its label is.
        quad_c = None if res.quad_s is None else spherical_to_cartesian_quadrupole(res.quad_s)
        m_real = build_polytensor(res.charges, res.mu, quad_c, max_rank=self.max_rank)
        m_shell = build_polytensor(res.charges - res.z, res.mu, quad_c, max_rank=self.max_rank)
        m_nuc = build_polytensor(res.z, None, None, max_rank=self.max_rank)
        e_point, e_pen = slater_elec_pair_energy(
            dr_au, r_au, m_real, m_shell, m_nuc, res.b, pair_index, max_rank=self.max_rank
        )

        spec_pauli = self.classical["pauli"]
        poly_i = per_pair(pauli_iso[0], pauli_env[0], i, use_env=spec_pauli.environment)
        poly_j = per_pair(pauli_iso[0], pauli_env[0], j, use_env=spec_pauli.environment)
        b_p_i = per_pair(pauli_iso[1], pauli_env[1], i, use_env=spec_pauli.environment)
        b_p_j = per_pair(pauli_iso[1], pauli_env[1], j, use_env=spec_pauli.environment)
        e_pauli = slater_pauli_pair_energy(
            dr_au, r_au, poly_i, poly_j,
            (0.5 * (b_p_i.log() + b_p_j.log())).exp(), max_rank=self.max_rank,
        )

        spec_disp = self.classical["disp"]
        c6_i = per_pair(disp_iso[0], disp_env[0], i, use_env=spec_disp.environment)
        c6_j = per_pair(disp_iso[0], disp_env[0], j, use_env=spec_disp.environment)
        bd_i = per_pair(disp_iso[1], disp_env[1], i, use_env=spec_disp.environment)
        bd_j = per_pair(disp_iso[1], disp_env[1], j, use_env=spec_disp.environment)
        e_disp = tt_damped_c6_energy(
            r,
            (0.5 * (c6_i.log() + c6_j.log())).exp(),
            (0.5 * (bd_i.log() + bd_j.log())).exp(),
        )

        e_pair = {
            "elst": gate["elst"] * (e_point + e_pen),
            "pauli": gate["pauli"] * e_pauli,
            "disp": gate["disp"] * e_disp,
        }

        # --- the channels whose environment difference is booked elsewhere ---------------
        # `elst` reads `theta_0` above; its `theta` gate would have moved the classical energy,
        # and that movement is polarization rather than electrostatics.
        to_induction = torch.zeros_like(r)
        elst_env = None
        for name, spec in self.classical.items():
            if not spec.to_induction or not has_env:
                continue
            gate_env, _ = build_gate(name, spec, use_env=True)
            shift = (gate_env - gate[name]) * (
                e_point + e_pen if name == "elst" else e_pair[name] / gate[name].clamp(min=1e-30)
            )
            env_shift[f"gate_{name}"] = (gate_env - gate[name]).abs().mean()
            if induction:
                to_induction = to_induction + inter * shift
            if name == "elst":
                elst_env = pool_batch(inter * shift)

        interaction = {name: pool_batch(inter * e_pair[name]) for name in e_pair}

        # --- the per-atom energy of the electronic state ---------------------------------
        t_point = damped_interaction_tensor(dr_au, None, 1.0 / r_au, max_rank=self.max_rank)

        def bond_energy_atoms(feats, q_, mu_, quad_s_, env_):
            """The **per-atom** bond energy, unpooled.

            ``feats`` is the isolated slot for the frozen evaluation and the joined one for
            the induced level. One decoder, so which slot it is handed *is* the difference.

            Exposed because the pooled quantity hides where the covalent energy actually sits.
            On a proton transfer the interesting question is what happens to the two oxygens
            and the shared hydrogen individually, and a per-fragment sum cannot answer it --
            `notebooks/mediator_plotting.ipynb` reads this.
            """
            return self.decoder.bond_energy(
                feats, species, q_, mu_, quad_s_, env_
            )

        def bond_energy(feats, q_, mu_, quad_s_, env_):
            """Per-atom bond energy, pooled to its fragment."""
            e = bond_energy_atoms(feats, q_, mu_, quad_s_, env_)
            return e.new_zeros(n_frag).index_add_(0, frag, e)

        # `Phi^intra`: gating the environment down to intra-fragment pairs is what makes the
        # frozen bond energy an isolated-fragment quantity. Do not simplify it to `phi = 0` --
        # the field an atom feels from its own molecule is part of what its bonds are worth.
        env_frozen = electrostatic_environment(
            positions, pair_index, t_point, gate["elst"] * intra, m_real,
            max_rank=self.max_rank,
        )
        energy_bond_atom = bond_energy_atoms(
            iso, res.charges, res.mu, res.quad_s, env_frozen
        )
        energy_bond = energy_bond_atom.new_zeros(n_frag).index_add_(
            0, frag, energy_bond_atom
        )

        f2b = batch.fragment_to_batch
        if f2b is None:
            f2b = batch.batch_idx.new_zeros(n_frag).scatter_(0, frag, batch.batch_idx)

        # --- induction -------------------------------------------------------------------
        level_ind = energy_bond_ind = energy_bond_ind_iso = None
        environment = energy_frozen_total = None
        solver: dict[str, tuple] = {}
        if induction:
            # The frozen level's own total: the *same functional* the coupled level minimizes,
            # evaluated at the frozen multipoles. The difference is a pure relaxation, so it is
            # exactly zero when nothing moves and negative by the variational principle.
            energy_frozen_total = res.internal_energy.new_zeros(n_sys).index_add(
                0, f2b, res.internal_energy
            ) + pool_batch(e_pair["elst"])

            ch_ind, chb_ind, _ = union_channels(positions, batch.batch_idx, frag, 0.0)
            # The joined slot, unlike the frozen level. `coupled_response` runs once for the
            # whole batch, which it must: unlike the frozen solve it couples *across*
            # fragments through the electrostatic gate.
            rp = self.decode_response(
                batch, joined, iso.species_idx, batch.batch_idx, bond_index=ch_ind
            )
            level_ind = coupled_response(
                rp, positions=positions, batch_idx=batch.batch_idx, n_systems=n_sys,
                bond_index=ch_ind, bond_batch=chb_ind, pair_index=pair_index,
                gate=gate["elst"], max_rank=self.max_rank, **self.cg,
            )
            solver["ind"] = (level_ind.n_iter, level_ind.converged, level_ind.pd_fail)

            quad_ind = (
                None if level_ind.quad_s is None
                else spherical_to_cartesian_quadrupole(level_ind.quad_s)
            )
            environment = electrostatic_environment(
                positions, pair_index, t_point,
                gate["elst"],
                build_polytensor(
                    level_ind.charges, level_ind.mu, quad_ind, max_rank=self.max_rank
                ),
                max_rank=self.max_rank,
            )

            # `E_bond` at the induced state **and the in-medium parameters**. Its difference
            # from the frozen evaluation is the charge transfer: same weights, read at theta
            # instead of theta_0 and at M^ind instead of M^frozen.
            energy_bond_ind = bond_energy(
                joined, level_ind.charges, level_ind.mu, level_ind.quad_s, environment
            )
            # The same state at `theta_0`. Nothing consumes it; it makes the *slot* swap
            # measurable separately from the state relaxation. That separation matters: the
            # swap alone was 100% of the old `ct` channel, with 4.6e-5 e crossing a boundary,
            # and nothing in the printed line said so.
            energy_bond_ind_iso = bond_energy(
                iso, level_ind.charges, level_ind.mu, level_ind.quad_s, environment
            )

            d_bond = energy_bond_ind - energy_bond
            interaction["induction"] = (
                (level_ind.energy - energy_frozen_total)
                + pool_batch(to_induction)
                + d_bond.new_zeros(n_sys).index_add_(0, f2b, d_bond)
            )
            if has_env:
                env_shift["e_bond"] = (
                    energy_bond_ind - energy_bond_ind_iso
                ).abs().mean()

        # --- assembly ---------------------------------------------------------------------
        energy_intra = r.new_zeros(n_frag).index_add_(
            0, pair_frag[is_intra], (intra * sum(e_pair.values()))[is_intra]
        )
        e0 = self.reference_energies[iso.species_idx]
        energy_ref = e0.new_zeros(n_frag).index_add_(0, frag, e0)
        fragment_energy = energy_ref + res.internal_energy + energy_intra + energy_bond

        energy = fragment_energy.new_zeros(n_sys).index_add_(0, f2b, fragment_energy)
        for value in interaction.values():
            energy = energy + value

        applicability = None
        if self.applicability is not None:
            # The joined slot, deliberately: the score says whether *this decomposition* is
            # the best description of the system, which no fragment-confined descriptor can
            # answer. See `ApplicabilityHead`.
            applicability = self.applicability(
                joined.inv_feats, frag, n_frag,
                batch.fragment_charge, batch.fragment_two_s,
            )

        return ExpertOutput(
            energy=energy,
            fragment_energy=fragment_energy,
            interaction=interaction,
            energy_ref=energy_ref,
            energy_internal=res.internal_energy,
            energy_intra=energy_intra,
            energy_bond=energy_bond,
            energy_bond_atom=energy_bond_atom,
            pair_index=pair_index,
            r=r,
            is_intra=is_intra,
            pair_frag=pair_frag,
            e_pair=e_pair,
            gate=gate,
            r0=r0_iso,
            r0_pair=r0_pair,
            alpha=alpha,
            species_idx=iso.species_idx,
            log_r0_prior=log_r0_prior,
            log_r0_prior_pair=log_r0_prior_pair,
            response=res,
            env_norm=slots.env_norm(),
            env_shift=env_shift,
            applicability=applicability,
            elst_env=elst_env,
            level_ind=level_ind,
            energy_bond_ind=energy_bond_ind,
            energy_bond_ind_iso=energy_bond_ind_iso,
            energy_frozen_total=energy_frozen_total,
            environment=environment,
            solver=solver or None,
        )
