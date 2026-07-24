"""Diabatic fragment-state library and per-frame state assignment.

A *diabatic state* fixes everything about a fragment's one-body identity: which atoms belong to
it, its formal charge ``Q_a`` and spin multiplicity ``2S_a + 1``, and its **channel graph** --
the bonds along which split charge may flow in the SQE solve
(docs/mixture_of_diabatic_embeddings.md §2.4.3).

The channel graph comes from the state assignment, never from a distance rule. This matters as
soon as geometries are distorted: a stretched O-H in a sampled H3O+ frame must keep its channel
(the diabat says the bond is there), and a hydrogen drifting to bonding distance must *not*
acquire one unless a diabat places it in the fragment. A covalent-radius cutoff gets both cases
wrong, and worse, would make the channel topology a discontinuous function of the coordinates --
exactly the force-noise pathology §2.4.4 is built to avoid.

Phase 1 reads the assignment straight off the data: one fragment per frame, keyed by the extxyz
``config_type`` header. Phase 2/3 will infer it from coordination-defect analysis (§3, Steps
1-2) and enumerate several competing states per reactive center, but the consumer interface --
:class:`FragmentAssignment` and the arrays it produces -- is meant to be the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

DEFAULT_LIBRARY_PATH = "data/diabatic_states.yaml"


@dataclass(frozen=True)
class DissociationChannel:
    """A way one parent state can fall apart into smaller library states.

    A channel is a *partition* of the parent's atoms into sub-states (each a key in the same
    library), plus the parent bond whose length drives the channel's validity envelope. It is
    the object the mixture machinery enumerates as a competing diabat: the bound parent versus
    this dissociated partition (docs/mixture_of_diabatic_embeddings.md §2.2, §2.4.1).

    name                  : label for diagnostics
    fragments             : tuple of ``(sub_state_key, parent_atom_indices)`` where the indices
                            are into the parent's ``elements`` and ordered to match the sub-state's
                            own canonical ``elements``
    order_parameter_bond  : the parent bond ``(a, b)`` whose length is the reaction coordinate for
                            the validity envelopes (typically the bond being broken)
    """

    name: str
    fragments: tuple[tuple[str, tuple[int, ...]], ...]
    order_parameter_bond: tuple[int, int]


@dataclass(frozen=True)
class FragmentState:
    """One formal diabatic state of a chemical fragment.

    key           : registry name (matches the extxyz ``config_type``)
    charge        : formal fragment charge Q_a (e)
    multiplicity  : spin multiplicity 2S_a + 1
    elements      : canonical atom ordering, as element symbols
    bonds         : channel graph, as index pairs into ``elements``
    dissociation  : competing dissociated diabats (empty for a state that is only ever a product)
    """

    key: str
    charge: int
    multiplicity: int
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    dissociation: tuple[DissociationChannel, ...] = ()

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def two_s(self) -> int:
        """``2S_a`` -- the number of unpaired electrons (0 for a closed-shell singlet)."""
        return self.multiplicity - 1

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        from ase.data import atomic_numbers as _z

        return tuple(int(_z[s]) for s in self.elements)


class DiabaticStateLibrary:
    """The registry of :class:`FragmentState` entries, loaded from YAML.

    Lookup is case-insensitive so the lowercase ``config_type`` headers in ``data/labels``
    resolve against however the states are spelled in the file.
    """

    def __init__(self, states: Sequence[FragmentState]) -> None:
        self._states = {s.key: s for s in states}
        self._by_fold = {s.key.casefold(): s for s in states}
        if len(self._by_fold) != len(self._states):
            raise ValueError("diabatic state keys collide under case folding")
        # Stable integer ids, for embedding tables and for storing the assignment in tensors.
        self._index = {key: i for i, key in enumerate(sorted(self._states))}

    @classmethod
    def from_yaml(cls, path=DEFAULT_LIBRARY_PATH) -> "DiabaticStateLibrary":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        entries = raw.get("states")
        if not entries:
            raise ValueError(f"{path}: no `states:` block found")
        states = []
        for key, spec in entries.items():
            elements = tuple(str(e) for e in spec["elements"])
            bonds = tuple((int(a), int(b)) for a, b in spec.get("bonds", ()))
            for a, b in bonds:
                if not (0 <= a < len(elements) and 0 <= b < len(elements)):
                    raise ValueError(
                        f"{path}: state {key!r} has bond ({a}, {b}) out of range for "
                        f"{len(elements)} atoms"
                    )
                if a == b:
                    raise ValueError(f"{path}: state {key!r} has a self-bond at atom {a}")
            if len({frozenset(b) for b in bonds}) != len(bonds):
                raise ValueError(f"{path}: state {key!r} lists a duplicate bond")
            channels = []
            for ch in spec.get("dissociation", ()) or ():
                frags = tuple(
                    (str(f["state"]), tuple(int(i) for i in f["atoms"]))
                    for f in ch["fragments"]
                )
                covered = [i for _, atoms in frags for i in atoms]
                if sorted(covered) != list(range(len(elements))):
                    raise ValueError(
                        f"{path}: dissociation {ch.get('name', '?')!r} of {key!r} must partition "
                        f"all {len(elements)} atoms exactly once, got {sorted(covered)}"
                    )
                ob = tuple(int(i) for i in ch["order_parameter_bond"])
                channels.append(
                    DissociationChannel(
                        name=str(ch.get("name", "dissoc")),
                        fragments=frags,
                        order_parameter_bond=(ob[0], ob[1]),
                    )
                )
            states.append(
                FragmentState(
                    key=str(key),
                    charge=int(spec["charge"]),
                    multiplicity=int(spec["multiplicity"]),
                    elements=elements,
                    bonds=bonds,
                    dissociation=tuple(channels),
                )
            )
        return cls(states)

    def __getitem__(self, key: str) -> FragmentState:
        try:
            return self._by_fold[str(key).casefold()]
        except KeyError:
            raise KeyError(
                f"no diabatic state {key!r} in the library "
                f"(have: {sorted(self._states)}); add it to {DEFAULT_LIBRARY_PATH}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return str(key).casefold() in self._by_fold

    def __len__(self) -> int:
        return len(self._states)

    @property
    def keys(self) -> list[str]:
        return sorted(self._states)

    def state_index(self, key: str) -> int:
        """Stable integer id of a state (embedding-table index)."""
        return self._index[self[key].key]


@dataclass
class FragmentAssignment:
    """The diabatic assignment of one frame: which atoms form which fragment states.

    ``fragments`` is a list of ``(FragmentState, atom_indices)`` where ``atom_indices`` are
    frame-local and ordered to match the state's canonical ``elements``.

    ``inter_bonds`` are channels that cross fragment boundaries (the bonds broken by a
    dissociation, in frame-local atom indices); they belong to no single fragment and carry the
    switching function (§2.4.3). A plain one-fragment assignment has none. ``order_bond`` is the
    reaction-coordinate bond (frame-local) whose length drives the validity envelopes, or None.

    The array properties are the form the model consumes; all indices stay frame-local and are
    re-offset when frames are concatenated into a ragged batch.
    """

    n_atoms: int
    fragments: list[tuple[FragmentState, np.ndarray]]
    inter_bonds: tuple[tuple[int, int], ...] = ()
    order_bond: tuple[int, int] | None = None

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    @property
    def fragment_idx(self) -> np.ndarray:
        """``(n_atoms,)`` -- which fragment each atom belongs to."""
        out = np.full(self.n_atoms, -1, dtype=np.int64)
        for a, (_, atoms) in enumerate(self.fragments):
            out[atoms] = a
        if (out < 0).any():
            missing = np.flatnonzero(out < 0).tolist()
            raise ValueError(f"atoms {missing} belong to no fragment in the assignment")
        return out

    @property
    def bond_index(self) -> np.ndarray:
        """``(2, n_bonds)`` -- the channel graph in frame-local atom indices.

        Intra-fragment channels first (in fragment order), then the inter-fragment channels;
        ``bond_is_inter`` marks the split.
        """
        cols = [
            (atoms[a], atoms[b])
            for state, atoms in self.fragments
            for a, b in state.bonds
        ]
        cols += [(a, b) for a, b in self.inter_bonds]
        if not cols:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(cols, dtype=np.int64).T

    @property
    def bond_is_inter(self) -> np.ndarray:
        """``(n_bonds,)`` bool -- True for inter-fragment channels (which carry the switch)."""
        n_intra = sum(state.n_bonds for state, _ in self.fragments)
        out = np.zeros(n_intra + len(self.inter_bonds), dtype=bool)
        out[n_intra:] = True
        return out

    @property
    def bond_fragment(self) -> np.ndarray:
        """``(n_bonds,)`` -- fragment id each intra channel belongs to; -1 for inter channels."""
        intra = [a for a, (state, _) in enumerate(self.fragments) for _ in state.bonds]
        return np.asarray(intra + [-1] * len(self.inter_bonds), dtype=np.int64)

    def order_parameter(self, positions: np.ndarray) -> float:
        """Length of the reaction-coordinate bond (Angstrom); NaN if this diabat has none."""
        if self.order_bond is None:
            return float("nan")
        a, b = self.order_bond
        return float(np.linalg.norm(np.asarray(positions)[a] - np.asarray(positions)[b]))

    @property
    def fragment_charge(self) -> np.ndarray:
        return np.asarray([s.charge for s, _ in self.fragments], dtype=np.float64)

    @property
    def fragment_two_s(self) -> np.ndarray:
        return np.asarray([s.two_s for s, _ in self.fragments], dtype=np.float64)

    @property
    def fragment_n_atoms(self) -> np.ndarray:
        return np.asarray([s.n_atoms for s, _ in self.fragments], dtype=np.int64)

    def state_indices(self, library: DiabaticStateLibrary) -> np.ndarray:
        return np.asarray(
            [library.state_index(s.key) for s, _ in self.fragments], dtype=np.int64
        )


def assign_from_headers(
    library: DiabaticStateLibrary,
    symbols: Sequence[str],
    *,
    config_type: str,
    charge: float | None = None,
    multiplicity: int | None = None,
) -> FragmentAssignment:
    """Phase-1 assignment: the whole frame is one fragment, named by ``config_type``.

    The frame is validated against the registry entry rather than reshaped to fit it: the
    element sequence must match the state's canonical ordering exactly, and the ``charge`` /
    ``multiplicity`` headers (when present) must agree with the state. A mismatch is a hard
    error -- silently reordering atoms or trusting a stale header would corrupt the channel
    graph in a way that is very hard to notice downstream.
    """
    state = library[config_type]
    got = tuple(str(s) for s in symbols)
    if got != state.elements:
        raise ValueError(
            f"frame labeled config_type={config_type!r} has element order {got}, but the "
            f"diabatic state {state.key!r} is defined for {state.elements}. Atom ordering "
            f"must match the registry (bonds index into it)."
        )
    if charge is not None and int(round(float(charge))) != state.charge:
        raise ValueError(
            f"frame config_type={config_type!r} has charge={charge}, but state "
            f"{state.key!r} declares charge={state.charge}"
        )
    if multiplicity is not None and int(multiplicity) != state.multiplicity:
        raise ValueError(
            f"frame config_type={config_type!r} has multiplicity={multiplicity}, but state "
            f"{state.key!r} declares multiplicity={state.multiplicity}"
        )
    return FragmentAssignment(
        n_atoms=len(got), fragments=[(state, np.arange(len(got), dtype=np.int64))]
    )


def _dissociated_assignment(
    library: DiabaticStateLibrary,
    parent: FragmentState,
    symbols: tuple[str, ...],
    channel: DissociationChannel,
) -> FragmentAssignment:
    """Build the :class:`FragmentAssignment` for one dissociation channel of ``parent``.

    Each sub-state's element sequence is validated against the library (as in
    :func:`assign_from_headers`), its own ``bonds`` become intra-fragment channels, and the
    parent bonds crossing the partition become inter-fragment channels carrying the switch.
    """
    n = len(symbols)
    atom_to_frag = np.full(n, -1, dtype=np.int64)
    fragments: list[tuple[FragmentState, np.ndarray]] = []
    for f_idx, (sub_key, atoms) in enumerate(channel.fragments):
        sub = library[sub_key]
        got = tuple(symbols[i] for i in atoms)
        if got != sub.elements:
            raise ValueError(
                f"dissociation {channel.name!r} of {parent.key!r}: fragment atoms {atoms} are "
                f"{got}, but sub-state {sub.key!r} is defined for {sub.elements}"
            )
        fragments.append((sub, np.asarray(atoms, dtype=np.int64)))
        atom_to_frag[list(atoms)] = f_idx

    # Parent bonds whose two atoms land in different fragments are the broken (inter) channels.
    inter = tuple(
        (a, b) for a, b in parent.bonds if atom_to_frag[a] != atom_to_frag[b]
    )
    return FragmentAssignment(
        n_atoms=n, fragments=fragments, inter_bonds=inter,
        order_bond=channel.order_parameter_bond,
    )


def enumerate_diabats(
    library: DiabaticStateLibrary,
    symbols: Sequence[str],
    *,
    config_type: str,
    charge: float | None = None,
    multiplicity: int | None = None,
) -> list[FragmentAssignment]:
    """The active space of a frame: the bound parent state plus each dissociated diabat.

    Element ``[0]`` is always the bound assignment (identical to :func:`assign_from_headers`);
    the rest are the parent's declared dissociation channels. Every assignment is tagged with the
    same reaction-coordinate bond so their validity envelopes read a common order parameter. A
    parent with no ``dissociation`` block yields a single (bound) diabat -- the Phase-1 behavior.
    """
    bound = assign_from_headers(
        library, symbols, config_type=config_type, charge=charge, multiplicity=multiplicity,
    )
    parent = library[config_type]
    symbols = tuple(str(s) for s in symbols)
    if parent.dissociation:
        # All diabats of one reactive center share the reaction coordinate; this demo declares a
        # single channel per state, so the bound state inherits that channel's order bond.
        bound.order_bond = parent.dissociation[0].order_parameter_bond
    return [bound] + [
        _dissociated_assignment(library, parent, symbols, channel)
        for channel in parent.dissociation
    ]


def finest_common_refinement(assignments: Sequence[FragmentAssignment]) -> np.ndarray:
    """Meet of the active partitions: ``fragment_idx`` on the finest common refinement.

    Two atoms share a block iff they share a fragment in *every* active assignment
    (docs/mixture_of_diabatic_embeddings.md §2.4.1). The SQE solve of the mixture runs on this
    partition, so that inter-fragment channels of any active diabat are resolved consistently.
    """
    if not assignments:
        raise ValueError("no assignments to refine")
    n = assignments[0].n_atoms
    if any(a.n_atoms != n for a in assignments):
        raise ValueError("assignments cover different atom counts")
    labels = np.stack([a.fragment_idx for a in assignments], axis=1)  # (n, n_assign)
    _, inv = np.unique(labels, axis=0, return_inverse=True)
    return inv.astype(np.int64).reshape(n)
