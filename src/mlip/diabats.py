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
class FragmentState:
    """One formal diabatic state of a chemical fragment.

    key           : registry name (matches the extxyz ``config_type``)
    charge        : formal fragment charge Q_a (e)
    multiplicity  : spin multiplicity 2S_a + 1
    elements      : canonical atom ordering, as element symbols
    bonds         : channel graph, as index pairs into ``elements``
    """

    key: str
    charge: int
    multiplicity: int
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]

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
            states.append(
                FragmentState(
                    key=str(key),
                    charge=int(spec["charge"]),
                    multiplicity=int(spec["multiplicity"]),
                    elements=elements,
                    bonds=bonds,
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

    The array properties are the form the model consumes; all indices stay frame-local and are
    re-offset when frames are concatenated into a ragged batch.
    """

    n_atoms: int
    fragments: list[tuple[FragmentState, np.ndarray]]

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
        """``(2, n_bonds)`` -- the channel graph in frame-local atom indices."""
        cols = [
            (atoms[a], atoms[b])
            for state, atoms in self.fragments
            for a, b in state.bonds
        ]
        if not cols:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(cols, dtype=np.int64).T

    @property
    def bond_fragment(self) -> np.ndarray:
        """``(n_bonds,)`` -- which fragment each channel belongs to."""
        return np.asarray(
            [a for a, (state, _) in enumerate(self.fragments) for _ in state.bonds],
            dtype=np.int64,
        )

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
