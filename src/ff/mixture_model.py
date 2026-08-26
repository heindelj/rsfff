"""Mediated mixtures of fragmentations: one geometry, several decompositions, one energy.

``docs/fff_v2.md`` §8 assembled. :class:`rsfff.ff.expert_model.FragmentExpertModel` answers for
*a* fragmentation. When a proton sits between two oxygens there are two, both defensible, and
choosing discontinuously gives a discontinuous energy. This module evaluates every candidate,
asks the mediator (:mod:`rsfff.ff.mediator`) for a partition of unity over them, and combines.

The rule, and where each quantity obeys it
------------------------------------------
> Mix at the lowest level at which the quantity means the same thing to both experts.

============================  ==================  ==================================
Quantity                      Mixed at            Implemented in
============================  ==================  ==================================
pair routing ``b_ij``         the accounting      :func:`routing_weight`
``C6``, ``b``, ``r0``,        the parameter       :func:`_mix`, applied per field
Pauli multipoles, response
``E_bond``                    the output          the ``bond`` loop in :meth:`forward`
the response solve            the equations       union graph + compliance scale
============================  ==================  ==================================

Four levels, because the quantities differ in where they become commensurable. Feature space
is a fifth candidate and fails the test -- see §8's *Why not mix in feature space*: the two
experts' descriptors do not mean the same thing, and there is no network defined at a
fractional composition to read a blend of them anyway.

Why this is a second forward and not a flag on the first
--------------------------------------------------------
``FragmentExpertModel.forward`` is the single-fragmentation model and every existing test, the
whole water corpus, and the EDA channel labels depend on it being exactly what it is. Threading
a soft ``b_ij`` through it would put a lerp on the hot path of a model that mostly does not
need one. So the mixture is assembled here, reusing ``forward``'s *helpers* -- the featurizer,
the expert fan-out, the response assembly, the classical backbones -- and duplicating only the
assembly, where the soft routing genuinely changes every line.

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
from .dispersion import tt_damped_c6_energy
from .electrostatics import slater_elec_pair_energy
from .environment import electrostatic_environment
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
    i, j = pair_index[0], pair_index[1]
    same = (fragments[:, i] == fragments[:, j]).to(weights.dtype)     # (M, P)
    return (weights.reshape(-1, 1) * same).sum(0)


def _mix(values, weights: torch.Tensor):
    """``sum_m w_m v_m`` over a list of tensors, broadcasting ``w`` against any trailing shape.

    ``None`` in, ``None`` out -- several response fields are optional and every decomposition
    agrees about which, because they are all the same heads on the same configuration.
    """
    if values[0] is None:
        return None
    stacked = torch.stack(list(values))
    w = weights.reshape(-1, *([1] * (stacked.dim() - 1)))
    return (w * stacked).sum(0)


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
    energy_bond: torch.Tensor              # () output-mixed bond energy
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

    # --- mix the parameters (the "same physical number" tier) --------------------------------
    _RP_FIELDS = (
        "chi", "eta", "q0", "chivec", "alpha", "chiquad", "cquad", "z", "b", "mu0", "quad0",
    )
    fields = {
        name: _mix([getattr(e.rp, name) for e in emitted], w) for name in _RP_FIELDS
    }

    # --- the solve: one union graph, compliance scaled by weight ------------------------------
    channels = torch.unique(
        torch.cat([e.channels for e in emitted], dim=1).T, dim=0
    ).T                                                                     # (2, Nc)
    key = channels[0] * n_atoms + channels[1]

    def mixed_compliance(pick):
        """The union graph's compliances, each channel carrying the weight that opens it.

        Scaled by ``S`` and **not** ``sqrt(S)``: the closed limit is a training NaN the other
        way round. A channel no decomposition opens keeps the zero, which is the ``s -> 0``
        limit and contributes nothing to the solve.
        """
        out = torch.zeros(channels.shape[1], dtype=dtype, device=positions.device)
        for m, e in enumerate(emitted):
            e_key = e.channels[0] * n_atoms + e.channels[1]
            out = out + w[m] * _scatter_by_key(key, e_key, pick(e))
        return out

    compliance = mixed_compliance(lambda e: e.rp.compliance)
    rp = ResponseParameters(compliance=compliance, **fields)

    frame = group.batch(0)
    block = torch.zeros(n_atoms, dtype=torch.long, device=positions.device)
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

    r0_iso = {k: _mix([e.r0_iso[k] for e in emitted], w) for k in emitted[0].r0_iso}
    r0_env = {k: _mix([e.r0_env[k] for e in emitted], w) for k in emitted[0].r0_env}
    pauli_iso = tuple(_mix([e.pauli_iso[t] for e in emitted], w) for t in (0, 1))
    pauli_env = tuple(_mix([e.pauli_env[t] for e in emitted], w) for t in (0, 1))
    disp_iso = tuple(_mix([e.disp_iso[t] for e in emitted], w) for t in (0, 1))
    disp_env = tuple(_mix([e.disp_env[t] for e in emitted], w) for t in (0, 1))
    alpha = emitted[0].alpha

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

    # --- the bond energy: mixed at the OUTPUT ------------------------------------------------
    # `FragmentBondEnergy` is per expert and two experts share no input space, so there is
    # nothing to mix upstream of it. Each expert reads its *own* descriptor and the *shared*
    # mixed electronic state, and the answers are combined.
    t_point = damped_interaction_tensor(dr_au, None, 1.0 / r_au, max_rank=model.max_rank)
    env_frozen = electrostatic_environment(
        positions, pair_index, t_point, gate["elst"] * b_ij, m_real,
        max_rank=model.max_rank,
    )
    bond = [
        _bond_sum(model, e, e.iso, res.charges, res.mu, res.quad_s, env_frozen)
        for e in emitted
    ]
    energy_bond = (w * torch.stack(bond)).sum()

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
    if with_induction and emitted[0].rp_env is not None:
        env_fields = {
            name: _mix([getattr(e.rp_env, name) for e in emitted], w) for name in _RP_FIELDS
        }
        # The **joined**-slot compliance, not the isolated one. The compliance head reads
        # features like every other parameter, so `theta` and `theta_0` give different channel
        # stiffnesses -- and the induction level is by definition the in-medium evaluation.
        # Reusing the frozen level's compliance here is a silent 0.1 kJ/mol disagreement with
        # `forward` at a one-hot membership, which is how this was caught.
        rp_ind = ResponseParameters(
            compliance=mixed_compliance(lambda e: e.rp_env.compliance), **env_fields
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
        bond_ind = []
        for m, e in enumerate(emitted):
            bond_ind.append(
                _bond_sum(model, e, e.joined, level.charges, level.mu, level.quad_s, env_ind)
            )
        energy_bond_ind = (w * torch.stack(bond_ind)).sum()
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
        energy_internal=res.internal_energy.sum(),
        energy_ref=energy_ref,
        energy_induction=energy_ind,
        mediator=med,
        vertex_induction=vertex_induction,
    )


def _bond_sum(model, emission, feats, q, mu, quad_s, env):
    """``sum_i E_bond`` for one decomposition's experts, at a **shared** electronic state.

    Each expert reads its *own* descriptor -- the two share no input space, which is why §8
    mixes this at the output -- and the state it is evaluated at is the mixture's, because
    there is only one electronic state once the solve has run.
    """
    def one(g):
        f = select_atoms(feats, g.atom_index)
        return g.expert.bond(
            f.inv_feats, f.species_idx, f.vec_feats, f.equiv_feats,
            _rows(q, g.atom_index),
            _rows(mu, g.atom_index),
            _rows(quad_s, g.atom_index),
            _env_rows(env, g.atom_index),
        )

    return model._fan_out(emission.groups, one, feats.inv_feats.shape[0]).sum()


def _rows(value, index):
    if value is None or index is None:
        return value
    return value[index]


def _env_rows(env, index):
    from dataclasses import replace

    if env is None or index is None:
        return env
    return replace(
        env,
        potential=_rows(env.potential, index),
        field=_rows(env.field, index),
        field_gradient=_rows(env.field_gradient, index),
    )


def _scatter_by_key(key: torch.Tensor, src_key: torch.Tensor, src: torch.Tensor):
    """``src`` re-indexed from ``src_key``'s order onto ``key``'s, zero where absent."""
    out = torch.zeros(key.shape[0], dtype=src.dtype, device=src.device)
    # `key` and `src_key` are both sorted-unique integer keys, so a searchsorted lands each
    # source row on its slot; rows of `key` with no source keep the zero.
    pos = torch.searchsorted(key, src_key)
    out[pos] = src
    return out
