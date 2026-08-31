"""Geometry -> candidate decompositions, at runtime.

Everything the mediator does starts from a *list* of competing fragmentations of one
geometry, and until now that list only ever came out of a file: the AIMD harvester
enumerated it offline, ran an ALMO-EDA job per decomposition, and wrote the result in the
multi-fragmentation extxyz schema of :mod:`rsfff.qcgen.multifrag`. Dynamics cannot work that
way -- the candidates change as the proton moves -- so this module does the enumeration from
coordinates.

The rule is not a new one. It is the *same* rule, ported
--------------------------------------------------------
``qchem_roundtrip/scripts/qchem_roundtrip.py`` holds the enumeration that produced every
labeled frame in ``data/wb97mv_tzvpd``: give each oxygen two hydrogens, give the
charge-carrying one ``+-|Q|``, and assign hydrogens to oxygens by exactly minimizing the total
O-H distance -- once per choice of which fragment hosts the charge.

That script is deliberately standalone: it runs on a cluster login node with no ``rsfff``
install, so it cannot import from here and this cannot import from it. The functions below are
therefore a **port**, and ``tests/test_reactive_md.py`` pins them against the corpus itself --
the decompositions this module produces must equal the ones stored in the training files, on
every frame of all four ion datasets. That test is what makes MD sample the partition family
the mediator was actually fitted on, rather than a similar-looking one.

Why the candidates are generated locally, and not re-ranked every step
---------------------------------------------------------------------
The offline enumerator answers "which fragment carries the charge" by solving the O-H
assignment problem once per candidate host and ranking the results. That is exactly right for
labeling a fixed frame and exactly wrong for dynamics, because **an argmin jumps**. Walk a
proton across a hydrogen bond in an H3O+(H2O)3 cluster and the rank-0 assignment flips
partway, the hop distances are then measured from a different reference, and a candidate that
was one hop becomes two and is discarded -- ``M`` falls from 3 to 2 between one step and the
next, with the dropped candidate still carrying real weight. Measured directly: the energy
steps, the Langevin thermostat cannot absorb it, and a 4-water run heats from 300 K to 577 K
and aborts on a 6.8 eV/Angstrom force at step 940.

Dropping the hop filter does not help; it moves the problem rather than removing it. The
per-host assignment is *itself* an argmin, so it jumps on its own.

So candidates are generated **relative to a base assignment that the caller holds fixed**:

* the base names which fragment is the ion and which hydrogens belong to which oxygen;
* a candidate is the base with **one** hydrogen moved off the ion (cation) or onto the ion
  (anion), for every acceptor within the validity envelope.

Nothing is minimized, so nothing jumps. The candidate set changes only when some
``r(H, O)`` crosses ``hi0``, which is exactly where that candidate's weight reaches zero -- so
the set is a continuous function of geometry in the only sense that matters.

The base is refreshed by :class:`~rsfff.md.calculator.MediatedCalculator` under hysteresis,
once the mediator has decisively committed to a different assignment. That is a genuine
discrete event; it is counted and reported rather than smoothed over.

Only the ion's own bonds are candidates
---------------------------------------
Moving a hydrogen between two *neutral* waters would make an OH- and an H3O+ where the frame
had one ion, so the frame would carry three ions in place of one. Nothing in this corpus looks
like that and the model has never seen it. The restriction falls out of the rule above rather
than being imposed on top of it: a one-hop move is defined off the ion, so autoionization is
never enumerated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..ff.mediator import align_fragments, contact_distance, contested_atoms
from ..ff.mixture_model import MixtureGroup
from ..mlip.switch import validity_bump

__all__ = [
    "DEFAULT_BUMP",
    "FragmentAssignment",
    "base_assignment",
    "enumerate_group",
    "one_hop_candidates",
    "h_counts_for_charge_state",
    "minimized_oh_assignment",
    "rank_oh_fragment_assignments",
]

#: The mediator's validity envelope on a contested atom's contact distance, in Angstrom.
#: Matches ``ExpertConfig.mediator_bump_hi1 / hi0`` (``configs/ion_mediator_v4.yaml``) and the
#: default in :class:`rsfff.ff.mediator.MediatorHead`. Enumeration and the head must use the
#: *same* envelope or the pre-filter drops candidates the head would have weighted.
DEFAULT_BUMP = dict(lo0=0.0, lo1=0.0, hi1=1.50, hi0=2.20)


class AssignmentError(ValueError):
    """Raised when a geometry cannot be decomposed into O/H fragments."""


@dataclass(frozen=True)
class FragmentAssignment:
    """One decomposition, ranked against the others of the same geometry.

    Mirrors the dataclass of the same name in ``qchem_roundtrip/scripts/qchem_roundtrip.py``
    so the two can be compared field by field.
    """

    rank: int
    charge_fragment: int
    fragment_idx: list[int]
    fragment_charges: list[int]
    fragment_multiplicities: list[int]
    total_distance: float
    excess_distance: float


def h_counts_for_charge_state(
    total_charge: int, n_oxygen: int, charge_fragment: int, n_hydrogen: int
) -> list[int]:
    """How many hydrogens each oxygen gets, if ``charge_fragment`` carries the charge.

    This is where H3O+ and OH- are *defined*: two hydrogens per oxygen, then the charged one
    gains or loses ``|Q|``. There is no bond-order model and no distance threshold behind it.
    """
    counts = [2] * n_oxygen
    if total_charge > 0:
        counts[charge_fragment] += abs(total_charge)
    elif total_charge < 0:
        counts[charge_fragment] -= abs(total_charge)
    if any(c < 0 for c in counts) or sum(counts) != n_hydrogen:
        raise AssignmentError(
            "cannot infer O-H fragment counts from charge, oxygen count, and hydrogen count "
            f"(charge={total_charge}, n_oxygen={n_oxygen}, n_hydrogen={n_hydrogen})"
        )
    return counts


def minimized_oh_assignment(
    symbols: list[str], coords: np.ndarray, h_counts: list[int]
) -> tuple[list[int], float]:
    """Assign hydrogens to oxygens minimizing the total O-H distance, exactly.

    Branch and bound rather than a greedy nearest-oxygen pass, because greedy is wrong exactly
    where this matters: at a transfer geometry the shared proton is near-equidistant, and
    taking it greedily can leave a later hydrogen with no capacity and force a much worse
    global assignment. These clusters are small enough to solve exactly.
    """
    oxygen = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    hydrogen = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    if len(oxygen) != len(h_counts):
        raise AssignmentError("number of oxygen atoms does not match fragment count")
    if sum(h_counts) != len(hydrogen):
        raise AssignmentError("hydrogen assignment capacities do not sum to hydrogen count")

    # (n_H, n_O) distances, computed once. The search below reads it ~n_H * n_O times.
    d = np.linalg.norm(
        coords[np.asarray(hydrogen)][:, None, :] - coords[np.asarray(oxygen)][None, :, :],
        axis=-1,
    )
    n_h, _n_o = d.shape

    # A capacitated assignment problem, solved exactly by expanding each oxygen into one
    # column per hydrogen it can hold and running the Hungarian algorithm on the square cost
    # matrix. The branch-and-bound this replaces returned the same optimum -- it is pinned
    # frame by frame against the corpus and against this solver -- but its cost is exponential
    # in the worst case, and a 7-water cluster is where that bites: ~1.2 s per frame against
    # ~1 ms here, which is the difference between a diversity pass over 15000 structures
    # taking three hours and taking one minute.
    slot_owner = np.repeat(np.arange(len(h_counts)), h_counts)
    if slot_owner.size != n_h:
        raise AssignmentError("hydrogen assignment capacities do not sum to hydrogen count")
    rows, cols = linear_sum_assignment(d[:, slot_owner])
    best = {int(h): int(slot_owner[c]) for h, c in zip(rows, cols)}
    best_cost = float(d[rows, slot_owner[cols]].sum())

    fragment_idx = [-1] * len(symbols)
    for o_pos, atom in enumerate(oxygen):
        fragment_idx[atom] = o_pos
    for h_pos, atom in enumerate(hydrogen):
        fragment_idx[atom] = best[h_pos]
    if any(i < 0 for i in fragment_idx):
        raise AssignmentError("O/H clusters only: some atom is neither oxygen nor hydrogen")
    return fragment_idx, best_cost


def rank_oh_fragment_assignments(
    symbols: list[str],
    coords: np.ndarray,
    total_charge: int,
    fragment_multiplicities: list[int] | None = None,
) -> list[FragmentAssignment]:
    """Every decomposition of an O/H cluster, ranked by total O-H distance.

    One candidate per oxygen -- "what if *this* fragment carried the charge" -- so a neutral
    cluster returns exactly one and a singly-charged one returns ``n_oxygen``. Rank 0 is the
    chemically obvious reading; the rest are the competitors the mediator exists to weigh.
    """
    n_o = sum(1 for s in symbols if s.upper() == "O")
    n_h = sum(1 for s in symbols if s.upper() == "H")
    if n_o == 0:
        raise AssignmentError("fragment enumeration requires at least one oxygen atom")
    if n_o + n_h != len(symbols):
        raise AssignmentError("fragment enumeration supports O/H clusters only")
    mults = list(fragment_multiplicities or [1] * n_o)
    if len(mults) != n_o:
        raise AssignmentError("fragment multiplicity count does not match oxygen count")

    coords = np.asarray(coords, dtype=float)
    total_charge = int(total_charge)
    hosts = range(n_o) if total_charge != 0 else range(1)

    found = []
    for host in hosts:
        counts = h_counts_for_charge_state(total_charge, n_o, host, n_h)
        idx, dist = minimized_oh_assignment(symbols, coords, counts)
        charges = [0] * n_o
        charges[host] = total_charge
        found.append((host, idx, charges, dist))
    found.sort(key=lambda item: (item[3], item[0]))
    best = found[0][3]
    return [
        FragmentAssignment(
            rank=rank,
            charge_fragment=host,
            fragment_idx=idx,
            fragment_charges=charges,
            fragment_multiplicities=mults,
            total_distance=dist,
            excess_distance=dist - best,
        )
        for rank, (host, idx, charges, dist) in enumerate(found)
    ]


def base_assignment(
    symbols: list[str], coords: np.ndarray, total_charge: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(fragment_idx, fragment_charges)`` for the chemically obvious reading of a geometry.

    Rank 0 of :func:`rank_oh_fragment_assignments`. Used to *seed* a trajectory and to reseed
    it when the mediator commits to a different assignment -- never per step, because that is
    the argmin whose jumps this module exists to avoid.
    """
    best = rank_oh_fragment_assignments(symbols, coords, total_charge)[0]
    return np.asarray(best.fragment_idx), np.asarray(best.fragment_charges)


def one_hop_candidates(
    symbols: list[str],
    base_idx: np.ndarray,
    base_charges: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The base, followed by every single-proton relabeling of it that is within range.

    For a cation the ion *donates*: each of its hydrogens may move to any oxygen inside the
    envelope, and the accepting fragment becomes the new ion. For an anion the ion *accepts*:
    any hydrogen of a neutral fragment may move onto it, and the donating fragment becomes the
    new hydroxide. Both leave exactly one fragment charged, which is the only composition the
    model has been fitted on.

    A neutral frame returns just the base -- no ion, no competition, ``M = 1``.

    **Generated generously and filtered by ``Omega``, never the other way round.** It is
    tempting to skip an acceptor whose ``r(H, O)`` already exceeds ``hi0``, and it is wrong:
    the head closes a candidate on ``contact_distance``, the distance to the *nearest atom* of
    the host, which is at most ``r(H, O)`` and is often a hydrogen instead. Screening on the
    O-H distance therefore discards candidates the head would still have weighted -- measured
    as a genuine step in the total energy, second-difference ratio 1.13 where a C2 energy gives
    4. Generation is cheap (at most ``3n`` candidates, distances only); only survivors of the
    ``Omega`` filter in :func:`enumerate_group` are ever featurized.
    """
    out = [(base_idx.copy(), base_charges.copy())]
    q = int(base_charges.sum())
    if q == 0:
        return out

    ion = int(np.argmax(np.abs(base_charges)))
    oxygen = np.array([i for i, s in enumerate(symbols) if s.upper() == "O"])
    hydrogen = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    # Which oxygen represents each fragment. `base_idx` numbers fragments by oxygen position.
    o_of_frag = {int(base_idx[o]): int(o) for o in oxygen}

    if q > 0:
        movers = [(h, f) for h in hydrogen if base_idx[h] == ion
                  for f in o_of_frag if f != ion]
    else:
        movers = [(h, ion) for h in hydrogen if base_idx[h] != ion]

    for h, dest in movers:
        idx = base_idx.copy()
        idx[h] = dest
        charges = np.zeros_like(base_charges)
        charges[dest if q > 0 else int(base_idx[h])] = q
        out.append((idx, charges))
    return out


def enumerate_group(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    total_charge: int,
    *,
    bump: dict | None = None,
    base: np.ndarray | None = None,
) -> MixtureGroup:
    """A :class:`~rsfff.ff.mixture_model.MixtureGroup` for one geometry.

    ``base`` is the held assignment described in the module docstring. Pass ``None`` and it is
    seeded from rank 0, which is right for a one-off evaluation and wrong inside a trajectory:
    :class:`~rsfff.md.calculator.MediatedCalculator` holds one and refreshes it under
    hysteresis.

    ``positions`` may carry ``requires_grad``. Everything discrete here reads it only through
    ``.detach()`` and the returned group holds the original tensor, so the graph from the
    caller's coordinates into :func:`~rsfff.ff.mixture_model.mixture_forward` is intact. The
    enumeration is a *constant* of the step: which candidates exist is a discrete fact, and
    only the weights attached to them are differentiable.
    """
    bump = dict(bump or DEFAULT_BUMP)
    pos_np = positions.detach().cpu().numpy()
    z = torch.as_tensor(atomic_numbers).reshape(-1)
    symbols = ["O" if int(v) == 8 else "H" if int(v) == 1 else "?" for v in z]

    if base is None:
        base_idx, base_charges = base_assignment(symbols, pos_np, total_charge)
    else:
        base_idx = np.asarray(base)
        base_charges = np.zeros(int(base_idx.max()) + 1, dtype=int)
        if total_charge:
            base_charges[_ion_fragment(symbols, base_idx, total_charge)] = int(total_charge)

    keep = one_hop_candidates(symbols, base_idx, base_charges)
    frag = align_fragments(torch.as_tensor(np.array([idx for idx, _q in keep])))
    contested = contested_atoms(frag)

    if contested.numel() > 0:
        # Close a candidate whose moved atom has left bonding range of the host it was given.
        # This is the *only* place a candidate is dropped, and it reads the mediator's own
        # `rho` -- the nearest atom of the host, not its oxygen -- so the drop lands exactly
        # where the head's weight reaches zero. Any cheaper screen fires early and steps the
        # energy; see :func:`one_hop_candidates`.
        rho = contact_distance(positions.detach(), frag, contested)
        omega = torch.stack(
            [torch.stack([validity_bump(r, **bump) for r in row]).prod() for row in rho]
        )
        live = [0] + [i for i in range(1, len(keep)) if float(omega[i]) > 0.0]
        if len(live) < len(keep):
            keep = [keep[i] for i in live]
            frag = align_fragments(frag[live])
            contested = contested_atoms(frag)

    if contested.numel() == 0:                  # nothing competing: one decomposition, w = 1
        keep, frag = keep[:1], frag[:1]

    n_dec, n_atoms = frag.shape
    dtype = positions.dtype
    atom_charge = torch.zeros(n_dec, n_atoms, dtype=dtype)
    for m, (idx, charges) in enumerate(keep):
        # Gather through the *pre-alignment* numbering: `align_fragments` renumbered the ids.
        atom_charge[m] = torch.as_tensor([float(charges[j]) for j in idx], dtype=dtype)
    return MixtureGroup(
        positions=positions,
        atomic_numbers=z,
        fragments=frag,
        atom_charge=atom_charge,
        atom_two_s=torch.zeros_like(atom_charge),   # closed-shell throughout this corpus
        contested=contested,
    )


def _ion_fragment(symbols: list[str], fragment_idx: np.ndarray, total_charge: int) -> int:
    """Which fragment is the ion, read off its hydrogen count rather than a stored label."""
    counts = np.bincount(
        fragment_idx[[i for i, s in enumerate(symbols) if s.upper() == "H"]],
        minlength=int(fragment_idx.max()) + 1,
    )
    want = 2 + total_charge
    hits = np.nonzero(counts == want)[0]
    if hits.size != 1:
        raise AssignmentError(
            f"expected exactly one fragment with {want} hydrogens for total charge "
            f"{total_charge:+d}, found {hits.size} (counts {counts.tolist()})"
        )
    return int(hits[0])
