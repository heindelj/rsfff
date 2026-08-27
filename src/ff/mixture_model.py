"""Mediated mixtures of fragmentations: one geometry, several decompositions, one energy.

``docs/fff_v2.md`` §8 assembled. :class:`rsfff.ff.expert_model.FragmentExpertModel` answers for
*a* fragmentation. When a proton sits between two oxygens there are two, both defensible, and
choosing discontinuously gives a discontinuous energy. This module evaluates every candidate,
asks the mediator (:mod:`rsfff.ff.mediator`) for a partition of unity over them, and combines.

One operator, not four levels
-----------------------------
v2 mixed different quantities at different levels -- the pair routing at the accounting, ``C6``
and the Pauli multipoles at the parameter, ``E_bond`` at the output, the response at the
equations -- on the argument that each becomes commensurable somewhere different. The argument
was true and the result was still wrong: the classical forms are nonlinear enough in those
parameters that a halfway parameter set is not a halfway energy, and the mediated energy left
the interval spanned by its two vertices by 162 kJ/mol at a proton transfer.

v4 mixes in one place, and it is upstream of all of them. A fragmentation is a partition of the
frame's **edges**, every candidate partitions the *same* edge set, and what a mixture chooses is
therefore where the boundary sits::

    s_e = sum_m w_m [ frag_m(i) == frag_m(j) ]

That one scalar (:mod:`rsfff.ff.partition`) builds the descriptors, scales the SQE channel
compliances, and routes the pair energies -- three consumers, one definition. Because the
density is linear in the edge sum, ``h`` and ``eta`` stay genuine power spectra all along the
path; a convex combination of two *finished* descriptors would not be, since the power spectrum
is quadratic and their average is the spectrum of no density at all.

The fragment's state -- charge, multiplicity, composition -- is mixed the same way and for the
same reason: the fractional labels go into one embedding, rather than two embeddings being
averaged. ``E(Q=0.5)`` is a real output of a network that is continuous in ``Q``. Half a proton
transferred genuinely leaves a water half-charged with two and a half hydrogens, and that is a
point the model can be asked about rather than a midpoint between two unrelated encodings.

Why this is a second forward and not a flag on the first
--------------------------------------------------------
``FragmentExpertModel.forward`` is the single-fragmentation model and every existing test, the
whole water corpus, and the EDA channel labels depend on it being exactly what it is. Threading
a soft ``b_ij`` through it would put a lerp on the hot path of a model that mostly does not
need one. So the mixture is assembled here, reusing ``forward``'s *helpers* -- the featurizer,
``emit``, the response assembly, the classical backbones -- and duplicating only the assembly,
where the soft routing genuinely changes every line.

The gap between the two has narrowed a lot in v4: with ``s_e`` as the parameter, a definite
fragmentation is the case ``s_e in {0, 1}`` and a mixture the case ``s_e in [0, 1]``, so the
two differ in the union channel graph and the frame-level charge blocking rather than in how
anything is parameterized. Collapsing them entirely is a later change, not this one.

The duplication is the cost of Invariant 1 being *checkable*: ``tests/test_mediator.py`` runs
this forward with a one-hot membership and requires it to reproduce ``forward`` to machine
precision. Two implementations that must agree is a stronger statement than one implementation
with a branch.

Charge conservation changes, and that is the point
--------------------------------------------------
A single fragmentation conserves charge per *fragment*: ``sqe_solve`` is block diagonal over
``intra_fragment_channels``. A mixture solves on the **union** of its decompositions' channel
graphs, whose connected component is the whole frame, so charge conserves per frame and is free
to cross what either assignment alone calls a fragment boundary. That is not a relaxation of a
constraint, it is the physics: while a proton is being transferred, both compliances are
nonzero and charge genuinely flows. A partially-open channel is rescaled by its weight ``S``
and **not** ``sqrt(S)`` -- the closed limit is a training NaN the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mlip.switch import pairwise_switch
from .damping import fermi_switch
from .expert_model import Emission  # noqa: F401  (typing only, for MixtureOutput)
from .dispersion import tt_damped_c6_energy
from .electrostatics import slater_elec_pair_energy
from .environment import electrostatic_environment
from .partition import mixed_state, soft_partition
from .mediator import MediatorOutput
from .pauli import slater_pauli_pair_energy
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    spherical_to_cartesian_quadrupole,
)
from .polarization import coupled_response
from .response import ResponseParameters, solve_frozen
from .slots import select_atoms
from .units import BOHR_ANG

__all__ = [
    "MixtureGroup",
    "MixtureOutput",
    "intra_pairs_unsorted",
    "mixture_forward",
    "routing_weight",
    "union_pair_list",
]


def intra_pairs_unsorted(fragments: torch.Tensor) -> torch.Tensor:
    """``(2, Nc)`` every ``i<j`` pair inside a fragment, with **no ordering requirement**.

    :func:`rsfff.ff.pairs.intra_fragment_channels` is the fast path and requires atoms grouped
    by fragment, because it offsets a triangular block per fragment rather than gathering. A
    mixture cannot satisfy that for more than one of its decompositions at once -- the whole
    point is that they disagree about which atoms are grouped -- so this enumerates from the
    co-membership mask instead. ``O(N^2)`` on a cluster of tens of atoms, which is the same
    regime the fast path's own docstring already accepts.
    """
    n = int(fragments.shape[0])
    same = (fragments.unsqueeze(-1) == fragments.unsqueeze(-2))
    iu = torch.triu_indices(n, n, offset=1, device=fragments.device)
    keep = same[iu[0], iu[1]]
    return iu[:, keep]


def union_pair_list(
    positions: torch.Tensor,
    fragments: torch.Tensor,       # (M, N)
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pair_index (2,P), r (P,))`` -- every pair within ``cutoff``, plus every intra pair of
    every decomposition.

    The mixture analogue of :func:`rsfff.ff.pairs.union_pairs`. One list serves every
    decomposition, so a pair's *energy* is computed once and only its booking depends on the
    membership -- which is what makes the accounting continuous. Intra pairs of **every**
    decomposition are unioned in with no distance test, for the reason ``union_pairs`` gives:
    a stretched bond must never be dropped by a pair-list radius.
    """
    n = int(positions.shape[0])
    iu = torch.triu_indices(n, n, offset=1, device=positions.device)
    r_all = (positions[iu[0]] - positions[iu[1]]).norm(dim=-1)
    keep = r_all <= cutoff
    for m in range(fragments.shape[0]):
        same = fragments[m][iu[0]] == fragments[m][iu[1]]
        keep = keep | same
    pair_index = iu[:, keep]
    return pair_index, (positions[pair_index[0]] - positions[pair_index[1]]).norm(dim=-1)


def mixed_q0(model, group, weights, species_idx):
    """``(N,)`` the baseline charges for a mixture: one prior, the *shifts* mixed.

    ``q0`` is the one response quantity the key cannot carry, and the reason is worth stating.
    It is a per-**element** prior plus a uniform per-**fragment** shift chosen so that ``q0``
    sums exactly to each fragment's formal charge -- the shift is bookkeeping about a
    partition, not a learned parameter, and a mixture has no single partition to read it from.

    So the prior is decoded once and the shifts are combined convexly. Every decomposition's
    ``q0`` sums to the same *frame* total (fragmentations of one geometry share their nuclei
    and their charge), so any convex combination of them does too -- SQE then conserves that
    identically, for any compliances.

    At a one-hot membership this is exactly the vertex's own ``q0``, which is what Invariant 1
    needs and what a frame-level shift would quietly break: it would spread one fragment's
    charge over the whole system.
    """
    prior = model.decoder.response.params.q0_prior[species_idx]
    shift = torch.zeros_like(prior)
    for m in range(group.fragments.shape[0]):
        frag = group.fragments[m]
        n_frag = int(frag.max()) + 1
        counts = torch.bincount(frag, minlength=n_frag).to(prior.dtype)
        prior_sum = prior.new_zeros(n_frag).index_add_(0, frag, prior)
        # Each fragment's formal charge, recovered from the per-atom form.
        order = torch.zeros(n_frag, dtype=torch.long, device=frag.device)
        order.scatter_(0, frag, torch.arange(frag.shape[0], device=frag.device))
        frag_q = group.atom_charge[m][order].to(prior.dtype)
        shift = shift + weights[m] * ((frag_q - prior_sum) / counts.clamp(min=1))[frag]
    return prior + shift


def frame_batch(group, positions, species_idx):
    """A one-frame :class:`~rsfff.train.data.Batch` for the *mixture*, blocked by frame.

    The decoder's response head needs a batch for ``fragment_idx``/``n_fragments``, but a
    mixture has no single fragmentation. It is handed one block covering the whole frame, which
    is the correct conservation law here: with the union channel graph open, charge conserves
    per frame and is free to cross what either assignment alone calls a fragment boundary.
    """
    from ..train.data import Batch

    n = positions.shape[0]
    zeros = torch.zeros(n, dtype=torch.long, device=positions.device)
    return Batch(
        positions=positions,
        atomic_numbers=group.atomic_numbers,
        batch_idx=zeros,
        n_systems=1,
        energy=positions.new_zeros(1),
        fragment_idx=zeros,
        fragment_charge=positions.new_zeros(1),
        fragment_two_s=positions.new_zeros(1),
        fragment_to_batch=torch.zeros(1, dtype=torch.long, device=positions.device),
        n_fragments=1,
    )


def routing_weight(
    fragments: torch.Tensor,       # (M, N)
    weights: torch.Tensor,         # (M,)
    pair_index: torch.Tensor,      # (2, P)
) -> torch.Tensor:
    """``b_ij = sum_m w_m [frag_m(i) == frag_m(j)]`` -- how intra each pair is.

    **The row of §8's table most easily left out, and the one that would break it.** This is
    where energy crosses the boundary between ``fragment_energy`` and the EDA channels, so a
    hard ``is_intra`` under a soft ``w`` is not merely inconsistent -- it is a discontinuity in
    the accounting itself. Since ``sum_m w_m = 1``, every pair is still counted exactly once
    across the intra and inter buckets: ``b_ij + (1 - b_ij) = 1`` identically.
    """
    return soft_partition(fragments, weights, pair_index)


@dataclass
class MixtureGroup:
    """One geometry and every decomposition of it, in **one shared atom order**.

    The dataset stores each decomposition as its own frame, re-sorted so its own
    ``fragment_idx`` is non-decreasing (:func:`rsfff.train.data._sort_by_fragment`), which
    means two decompositions of one geometry generally arrive in *different* atom orders. A
    mixture cannot combine per-atom quantities across them until they agree about which row is
    which atom, so :func:`rsfff.train.data.mixture_groups` undoes those sorts and hands the
    result here in the file's canonical (trajectory) order.

    Nothing downstream re-sorts. That is why :func:`intra_pairs_unsorted` and
    :func:`union_pair_list` exist: the sorted-input fast paths in :mod:`rsfff.ff.pairs` cannot
    serve more than one decomposition at a time, and picking one to privilege would silently
    make the others wrong.
    """

    positions: torch.Tensor            # (N, 3)
    atomic_numbers: torch.Tensor       # (N,)
    fragments: torch.Tensor            # (M, N) long, fragment id per decomposition
    atom_charge: torch.Tensor          # (M, N) the host fragment's formal charge, per atom
    atom_two_s: torch.Tensor           # (M, N) the host fragment's 2S, per atom
    contested: torch.Tensor            # (D,) the atoms whose membership the mediator decides
    energy: torch.Tensor | None = None       # () the fragmentation-invariant total
    forces: torch.Tensor | None = None       # (N, 3) likewise
    #: (M,) each decomposition's ``|eda_pol + eda_ct|``, when the frames carry it. The label
    #: form of ``L_ct``'s magnitude -- see :class:`rsfff.train.mixture_stream.MixtureStream`.
    vertex_induction_label: torch.Tensor | None = None
    #: Which *geometry* this is, in the same numbering
    #: :func:`rsfff.train.data.load_cluster_datasets` gives its frames. This is what lets the
    #: mediator be held out on the same split as everything else.
    group_id: int = -1

    def batch(self, m: int):
        """A single-frame :class:`rsfff.train.data.Batch` for decomposition ``m``.

        Built rather than stored because the per-decomposition fragment bookkeeping
        (``fragment_charge``, ``fragment_to_batch``, ``n_fragments``) is cheap to derive and
        expensive to keep consistent by hand across ``M`` copies of the same geometry.
        """
        from ..train.data import Batch

        frag = self.fragments[m]
        n_frag = int(frag.max()) + 1
        device = frag.device
        # Per-fragment charge/spin, recovered from the per-atom form by taking each
        # fragment's first atom -- they agree by construction on every atom of a fragment.
        order = torch.zeros(n_frag, dtype=torch.long, device=device)
        order.scatter_(0, frag, torch.arange(frag.shape[0], device=device))
        return Batch(
            positions=self.positions,
            atomic_numbers=self.atomic_numbers,
            batch_idx=torch.zeros(frag.shape[0], dtype=torch.long, device=device),
            n_systems=1,
            energy=self.positions.new_zeros(1),
            forces=None,
            fragment_idx=frag,
            fragment_charge=self.atom_charge[m][order],
            fragment_two_s=self.atom_two_s[m][order],
            fragment_to_batch=torch.zeros(n_frag, dtype=torch.long, device=device),
            n_fragments=n_frag,
        )


@dataclass
class MixtureOutput:
    """One mediated geometry: the total, the weights, and the vertex outputs behind them.

    ``energy`` is the only *energy* label a mixture has (§8: every EDA channel is defined
    relative to a choice of fragments, so a 60/40 mixture has no ``eda_cls_elec``). The vertex
    outputs are carried alongside because they are what the EDA channels supervise and what
    ``L_ct`` ranks -- and because Invariant 1 is the statement that this reduces to one of them.
    """

    energy: torch.Tensor                   # () total, Hartree
    charges: torch.Tensor                  # (N,) mixed SQE charges; sum = frame formal charge
    b_ij: torch.Tensor                     # (P,) routing weight per pair
    pair_index: torch.Tensor               # (2, P)
    e_pair: dict[str, torch.Tensor]        # (P,) per channel
    energy_intra: torch.Tensor             # () pairs booked to fragments, by b_ij
    energy_inter: dict[str, torch.Tensor]  # () per channel, booked by (1 - b_ij)
    energy_bond: torch.Tensor              # () bond energy, decoded from the mixed key
    energy_bond_atom: torch.Tensor         # (N,) the same, before summing over atoms
    energy_internal: torch.Tensor          # () union-graph response internal energy
    energy_ref: torch.Tensor               # () sum_i E0[Z_i], fragmentation invariant
    energy_induction: torch.Tensor         # () the coupled relaxation plus the E_bond shift
    mediator: MediatorOutput
    #: (M,) each decomposition's induction energy, for the mediator's shaping prior.
    #:
    #: **Always ``None`` today, deliberately.** Filling it needs one coupled solve *per
    #: decomposition* on top of the mixture's own, tripling the cost of the term that is
    #: explicitly meant to be a weak prior. Every contested geometry in this corpus carries
    #: ``eda_pol + eda_ct`` per vertex, so ``L_ct`` reads the label instead and this field is
    #: the hook for a corpus that does not. See
    #: :meth:`rsfff.train.mixture_stream.MixtureStream._induction_magnitude`.
    vertex_induction: torch.Tensor | None = None
    #: (M,) each pure vertex's total energy. The disagreement between these through the
    #: crossover is the residual the mixture exists to remove (§8).
    vertex_energy: torch.Tensor | None = None
    #: The **mixed description** every parameter above was decoded from: ``emission.iso`` is
    #: ``h`` at the soft partition (with the fractional state block) and ``emission.joined`` is
    #: ``[h | eta]``. Exposed so a consumer that wants a parameter this output does not already
    #: carry -- a plot of ``C6`` through the crossover, say -- can decode it from the same
    #: description rather than rebuilding the partition itself. Rebuilding it is the drift this
    #: module exists to avoid; there is no second definition of the mixed input.
    emission: "Emission | None" = None


def mixture_forward(
    model,
    group,
    mediator,
    *,
    with_induction: bool = False,
) -> MixtureOutput:
    """Evaluate one geometry under every decomposition and mix. See the module docstring.

    ``model`` is a :class:`rsfff.ff.expert_model.FragmentExpertModel` -- this is written as a
    function taking it rather than as a method so the mixture stays a *reader* of that model:
    it calls its helpers and owns none of its weights, which is what keeps "switch the mediator
    off and you have the model of §5-§7" a true statement about the code and not only about the
    math.
    """
    positions = group.positions
    fragments = group.fragments                     # (M, N) long
    n_dec, n_atoms = fragments.shape
    dtype = positions.dtype

    # --- per decomposition: features, experts, and every parameter at both evaluations -----
    emitted = [
        model.emit(
            group.batch(m), fragments[m], bond_index=intra_pairs_unsorted(fragments[m])
        )
        for m in range(n_dec)
    ]

    # --- the membership ---------------------------------------------------------------------
    med = mediator(
        torch.stack([e.joined.inv_feats for e in emitted]),
        fragments,
        positions,
        group.atom_charge,
        group.atom_two_s,
        group.contested,
    )
    w = med.weights

    # --- one soft partition, and one description built from it ------------------------------
    # This is the whole mixing operator. Every candidate assignment partitions the *same* edge
    # set -- `A(s) + A(1-s) = A_full` for any `s` -- so what a mixture chooses is where the
    # boundary sits, not what the descriptor says. Interpolating the boundary keeps `h` and
    # `eta` genuine power spectra; interpolating two finished descriptors would not, because
    # the power spectrum is quadratic and their average is the spectrum of no density at all.
    species = emitted[0].iso.species_idx
    batch_idx = torch.zeros(n_atoms, dtype=torch.long, device=positions.device)

    # The state is mixed at the **input**: fractional charge, multiplicity and element census
    # into one embedding, rather than an average of two embeddings. `E(Q=0.5)` is a real
    # output of that net; the midpoint of two latents is a point nothing produced.
    q_star, two_s_star, counts_star = mixed_state(
        fragments, w, group.atom_charge, group.atom_two_s, species,
        model.fragment_state.n_species,
    )
    state = model.fragment_state.mixed(q_star, two_s_star, counts_star)

    channels = torch.unique(
        torch.cat([e.channels for e in emitted], dim=1).T, dim=0
    ).T                                                                     # (2, Nc)

    frame = frame_batch(group, positions, species)
    mixed = model.emit(
        frame, batch_idx, bond_index=channels,
        # A callable, so `s_e` is computed on the edge list the featurizer itself builds.
        edge_weight=lambda edges: soft_partition(fragments, w, edges),
        state=state,
    )
    iso, joined = mixed.iso, mixed.joined

    # --- the solve: one union graph, compliance scaled by the same partition ------------------
    rp_full = model.decode_response(
        frame, iso, species, batch_idx, bond_index=channels
    )

    #: How open each channel of the union graph is -- the *same* scalar the featurizer and the
    #: pair accounting use. A channel joins two atoms of one fragment, so "open under `m`" is
    #: exactly "same fragment under `m`", and this used to be written out separately as an
    #: `isin` over each decomposition's channel list. One definition, three consumers.
    channel_open = soft_partition(fragments, w, channels)

    def scaled_compliance(base):
        """The union graph's compliances, each channel carrying the weight that opens it.

        Scaled by ``S`` and **not** ``sqrt(S)``: the closed limit is a training NaN the other
        way round. A channel no decomposition opens keeps the zero, which is the ``s -> 0``
        limit and contributes nothing to the solve.
        """
        return channel_open * base

    q0 = mixed_q0(model, group, w, species)
    rp = ResponseParameters(
        **{
            name: getattr(rp_full, name)
            for name in (
                "chi", "eta", "chivec", "alpha", "chiquad", "cquad", "z", "b",
                "mu0", "quad0",
            )
        },
        q0=q0,
        compliance=scaled_compliance(rp_full.compliance),
    )

    block = batch_idx
    res = solve_frozen(
        rp, frame,
        direct_multipoles=model._direct_multipoles,
        bond_index=channels,
        bond_batch=torch.zeros(channels.shape[1], dtype=torch.long, device=positions.device),
        block_idx=block,
        n_blocks=1,
    )

    # --- one pair list, soft routing ----------------------------------------------------------
    pair_index, r = union_pair_list(positions, fragments, model.cutoff_max)
    i, j = pair_index[0], pair_index[1]
    b_ij = routing_weight(fragments, w, pair_index)
    dr_au = (positions[j] - positions[i]) / BOHR_ANG
    r_au = r / BOHR_ANG

    # Every per-atom parameter, mixed, then lerped between the isolated and in-medium
    # evaluation by `b_ij`. `torch.where(is_intra, ...)` is the discontinuity a hard routing
    # would introduce; this is the same selection made continuous.
    def pair_param(iso_atom, env_atom, index, *, use_env: bool):
        a = iso_atom[index]
        if not use_env or env_atom is iso_atom:
            return a
        b_ = env_atom[index]
        shape = b_ij.reshape(-1, *([1] * (a.dim() - 1)))
        return shape * a + (1.0 - shape) * b_

    # Decoded from the mixed keys, not averaged across decompositions. `r0` does not appear on
    # either side of that distinction: it reads the element alone, so it is the same number
    # whichever description is live -- which is exactly the discontinuity v2 had and this does
    # not.
    r0_iso, alpha = model.decoder.r0(species)
    r0_env = r0_iso
    pauli_iso = model._pauli_from(iso, species)
    pauli_env = model._pauli_from(joined, species)
    disp_iso = model.decoder.dispersion(iso, species)
    disp_env = model.decoder.dispersion(joined, species)

    def build_gate(name, spec, *, use_env):
        base_i = pair_param(r0_iso[name], r0_env[name], i, use_env=use_env)
        base_j = pair_param(r0_iso[name], r0_env[name], j, use_env=use_env)
        r0_ij = (0.5 * (base_i.log() + base_j.log())).exp()
        return fermi_switch(r, r0_ij, alpha[name]) * pairwise_switch(
            r, spec.cutoff - spec.taper_width, spec.cutoff
        )

    gate = {
        name: build_gate(name, spec, use_env=spec.environment)
        for name, spec in model.classical.items()
    }

    # --- classical backbones (identical forms; only the parameters and the booking moved) -----
    quad_c = None if res.quad_s is None else spherical_to_cartesian_quadrupole(res.quad_s)
    m_real = build_polytensor(res.charges, res.mu, quad_c, max_rank=model.max_rank)
    m_shell = build_polytensor(res.charges - res.z, res.mu, quad_c, max_rank=model.max_rank)
    m_nuc = build_polytensor(res.z, None, None, max_rank=model.max_rank)
    e_point, e_pen = slater_elec_pair_energy(
        dr_au, r_au, m_real, m_shell, m_nuc, res.b, pair_index, max_rank=model.max_rank
    )

    sp = model.classical["pauli"]
    e_pauli = slater_pauli_pair_energy(
        dr_au, r_au,
        pair_param(pauli_iso[0], pauli_env[0], i, use_env=sp.environment),
        pair_param(pauli_iso[0], pauli_env[0], j, use_env=sp.environment),
        (
            0.5 * (
                pair_param(pauli_iso[1], pauli_env[1], i, use_env=sp.environment).log()
                + pair_param(pauli_iso[1], pauli_env[1], j, use_env=sp.environment).log()
            )
        ).exp(),
        max_rank=model.max_rank,
    )

    sd = model.classical["disp"]
    e_disp = tt_damped_c6_energy(
        r,
        (0.5 * (
            pair_param(disp_iso[0], disp_env[0], i, use_env=sd.environment).log()
            + pair_param(disp_iso[0], disp_env[0], j, use_env=sd.environment).log()
        )).exp(),
        (0.5 * (
            pair_param(disp_iso[1], disp_env[1], i, use_env=sd.environment).log()
            + pair_param(disp_iso[1], disp_env[1], j, use_env=sd.environment).log()
        )).exp(),
    )

    e_pair = {
        "elst": gate["elst"] * (e_point + e_pen),
        "pauli": gate["pauli"] * e_pauli,
        "disp": gate["disp"] * e_disp,
    }

    # --- the bond energy -----------------------------------------------------------------------
    t_point = damped_interaction_tensor(dr_au, None, 1.0 / r_au, max_rank=model.max_rank)
    env_frozen = electrostatic_environment(
        positions, pair_index, t_point, gate["elst"] * b_ij, m_real,
        max_rank=model.max_rank,
    )
    # v2 mixed this at the *output* -- each expert's bond energy, weighted -- because the two
    # experts shared no input space. There is one decoder now and one description, so it is
    # read exactly as `forward` reads it, from the isolated slot.
    energy_bond_atom = model.decoder.bond_energy(
        iso, species, res.charges, res.mu, res.quad_s, env_frozen
    )
    energy_bond = energy_bond_atom.sum()

    # --- induction: the coupled solve on the union graph, at the in-medium parameters --------
    # `E_ind` is a *relaxation*: the same functional the frozen level was evaluated at, now
    # minimized with the response parameters read in-medium and the multipoles free to move.
    # Omitting it would leave the mixture's total short by the whole induction energy -- tens
    # of kJ/mol on these ion clusters -- and the mediator would be asked to remove a residual
    # it has no way to touch.
    # A channel whose *own* energy reads `theta_0` still has a `theta` evaluation, and the
    # difference is polarization rather than that channel -- `elst` is the case that matters,
    # since `eda_cls_elec` is rigorously pairwise. `forward` books it to induction and so must
    # this, or the two disagree by exactly that term at a one-hot membership.
    to_induction = positions.new_zeros(())
    has_env = emitted[0].has_env
    for name, spec in model.classical.items():
        if not spec.to_induction or not has_env:
            continue
        gate_env = build_gate(name, spec, use_env=True)
        base = (
            (e_point + e_pen) if name == "elst"
            else e_pair[name] / gate[name].clamp(min=1e-30)
        )
        to_induction = to_induction + ((1.0 - b_ij) * (gate_env - gate[name]) * base).sum()

    energy_ind = positions.new_zeros(())
    vertex_induction = None
    if with_induction:
        rp_ind_full = model.decode_response(
            frame, joined, species, batch_idx, bond_index=channels
        )
        rp_ind = ResponseParameters(
            **{
                name: getattr(rp_ind_full, name)
                for name in (
                    "chi", "eta", "chivec", "alpha", "chiquad", "cquad", "z", "b",
                    "mu0", "quad0",
                )
            },
            q0=q0,
            compliance=scaled_compliance(rp_ind_full.compliance),
        )
        frozen_total = res.internal_energy.sum() + e_pair["elst"].sum()
        level = coupled_response(
            rp_ind,
            positions=positions,
            batch_idx=block,
            n_systems=1,
            bond_index=channels,
            bond_batch=torch.zeros(
                channels.shape[1], dtype=torch.long, device=positions.device
            ),
            pair_index=pair_index,
            gate=gate["elst"],
            max_rank=model.max_rank,
            **model.cg,
        )
        quad_ind = (
            None if level.quad_s is None
            else spherical_to_cartesian_quadrupole(level.quad_s)
        )
        env_ind = electrostatic_environment(
            positions, pair_index, t_point, gate["elst"],
            build_polytensor(level.charges, level.mu, quad_ind, max_rank=model.max_rank),
            max_rank=model.max_rank,
        )
        # The bond energy at the induced state **and** the in-medium parameters. Its difference
        # from the frozen evaluation is the charge transfer: the same weights, read at `theta`
        # instead of `theta_0` and at `M^ind` instead of `M^frozen`.
        energy_bond_ind = model.decoder.bond_energy(
            joined, species, level.charges, level.mu, level.quad_s, env_ind
        ).sum()
        energy_ind = (
            (level.energy[0] - frozen_total)
            + to_induction
            + (energy_bond_ind - energy_bond)
        )

    # --- assembly -----------------------------------------------------------------------------
    total_pair = sum(e_pair.values())
    energy_intra = (b_ij * total_pair).sum()
    energy_inter = {k: ((1.0 - b_ij) * v).sum() for k, v in e_pair.items()}
    energy_ref = model.reference_energies[emitted[0].iso.species_idx].sum()

    energy = (
        energy_ref
        + res.internal_energy.sum()
        + energy_intra
        + energy_bond
        + sum(energy_inter.values())
        + energy_ind
    )

    return MixtureOutput(
        energy=energy,
        charges=res.charges,
        b_ij=b_ij,
        pair_index=pair_index,
        e_pair=e_pair,
        energy_intra=energy_intra,
        energy_inter=energy_inter,
        energy_bond=energy_bond,
        energy_bond_atom=energy_bond_atom,
        energy_internal=res.internal_energy.sum(),
        energy_ref=energy_ref,
        energy_induction=energy_ind,
        mediator=med,
        vertex_induction=vertex_induction,
        emission=mixed,
    )
