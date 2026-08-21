"""The multi-fragmentation extxyz schema: one geometry, several decompositions.

Why this exists
---------------
``scripts/parse_roundtrip.py`` writes one fragmentation per frame, because for a
neutral water cluster there is only one: every fragment is a water. Put an
excess proton or an excess hole into the cluster and that stops being true. An
``H3O+(H2O)2`` frame can be read as ``H3O+ | H2O | H2O``, ``H2O | H3O+ | H2O`` or
``H2O | H2O | H3O+`` -- three different diabatic decompositions of the *same*
wavefunction, and ``qchem_roundtrip``'s AIMD harvester runs an ALMO-EDA job for
each. Which one is "right" is exactly the question a reactive model has to
answer, so the data format must be able to state all of them rather than pick.

This module owns that generalized schema: the frame assembly, the writer, the
reader, and the geometry algebra that makes the labels commensurable in the
first place.

Getting the frames commensurable
--------------------------------
The labels come from separate Q-Chem jobs and none of them is expressed in the
frame the trajectory lives in:

* **Atom order differs per fragmentation.** The EDA ``$molecule`` block is
  written grouped by fragment, so ``H2O | H3O+ | H2O`` orders the atoms
  differently from ``H3O+ | H2O | H2O``. The permutation back to trajectory order
  is recorded exactly, in the harvester's ``assignment_signature``.
* **Orientation differs per fragmentation.** Q-Chem reorients each job into its
  own standard nuclear orientation, and that orientation depends on the atom
  order -- so two fragmentations of one geometry come back rotated relative to
  each other.
* **The origin, however, is shared.** Q-Chem's standard orientation puts the
  center of *nuclear charge* at the origin (measured: 2e-11 Angstrom on these
  jobs). Recentering the trajectory geometry the same way reduces the map from
  every EDA job to a **pure rotation**, which is the whole reason this module
  can get away with rotating multipoles instead of translating them. Translating
  a primitive multipole is not a rotation's inverse-free cousin: it mixes ranks
  and needs the lower moments, and getting it subtly wrong is invisible.

:func:`rotation_between` therefore solves the *uncentered* orthogonal Procrustes
problem. That is the check as much as the transform: if the two frames did not
in fact share an origin, the residual RMSD blows up from ~1e-8 Angstrom to
something of order the molecular size, and the caller refuses the frame instead
of writing a quietly-translated label.

The canonical frame
-------------------
The trajectory's own coordinates, translated so the center of nuclear charge is
at the origin. Forces are translation invariant and come through untouched;
every multipole from every EDA job is rotated into it.

Units, matching every other file in ``data/``: positions Angstrom, energies
Hartree, forces Hartree/bohr, multipoles ``e*a0^n``, polarizabilities ``a0^3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase.data import atomic_numbers

from .qchem_out import QChemParseError, expand_multipole, unique_components

#: Order of EDA components on the header line: the five requested ones first,
#: then the bookkeeping terms. Matches ``scripts/parse_roundtrip.py``.
EDA_KEY_ORDER = (
    "cls_elec", "mod_pauli", "disp", "pol", "ct",
    "prp", "frz", "int", "elec", "pauli", "cls_pauli", "cls_disp",
)

MULTIPOLE_NAMES = ("dipole", "quadrupole", "octopole", "hexadecapole")

#: Two frames of the same structure must agree to this after the rotation
#: (Angstrom). Q-Chem prints 10 decimals, so a genuine rigid-body match lands at
#: ~5e-9; anything above this is not the same geometry, or not the same origin.
RMSD_TOL = 1e-6

#: Conventional names for the fragments this data contains. Hill ordering would
#: spell hydroxide ``HO-``; the rest of the repo, the harvest labels and the
#: literature all say ``OH-``, so the display name is looked up rather than
#: derived. Anything not listed falls back to Hill order plus a charge suffix.
_FORMULA_ALIASES = {
    ("H2O", 0): "H2O",
    ("H3O", 1): "H3O+",
    ("HO", -1): "OH-",
    ("HO", 0): "OH",
}


# ---------------------------------------------------------------------------
# Geometry algebra
# ---------------------------------------------------------------------------


def nuclear_charges(symbols) -> np.ndarray:
    """Atomic numbers for a list of element symbols."""
    return np.array([float(atomic_numbers[s.capitalize()]) for s in symbols])


def center_of_nuclear_charge(symbols, positions) -> np.ndarray:
    """The point Q-Chem's ``Standard Nuclear Orientation`` puts at the origin."""
    z = nuclear_charges(symbols)
    return (z[:, None] * np.asarray(positions)).sum(0) / z.sum()


def recenter(symbols, positions) -> np.ndarray:
    """Translate so the center of nuclear charge is at the origin."""
    return np.asarray(positions, dtype=np.float64) - center_of_nuclear_charge(
        symbols, positions
    )


def rotation_between(source, target, rmsd_tol: float = RMSD_TOL) -> tuple[np.ndarray, float]:
    """Proper rotation ``R`` with ``source @ R.T ~= target``, plus the residual RMSD.

    Orthogonal Procrustes on **uncentered** coordinates: both arguments are
    expected to already share an origin (see the module docstring), and the
    residual is the evidence that they do.

    ``R`` is constrained to ``det = +1``. An improper transform would be an
    equally good coordinate fit for a planar or near-planar fragment -- water is
    planar, so this is not hypothetical -- and would reflect every tensor carried
    through it. Raises when the residual exceeds ``rmsd_tol``, since past that
    point ``R`` is a fit rather than a frame change.
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected matching (n, 3) arrays, got {a.shape} and {b.shape}")

    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    rmsd = float(np.sqrt((((a @ rot.T) - b) ** 2).sum(axis=1).mean()))
    if rmsd > rmsd_tol:
        raise QChemParseError(
            f"frames do not align: residual RMSD {rmsd:.3g} Angstrom after a proper "
            f"rotation (tolerance {rmsd_tol:g}). The two jobs are not the same geometry, "
            f"or their coordinate origins differ"
        )
    return rot, rmsd


def rotate_multipole(tensor, rot) -> np.ndarray:
    """Rotate a fully symmetric Cartesian tensor of any rank: ``T'_ij.. = R_ia R_jb.. T_ab..``.

    Each pass contracts the leading axis and sends it to the back, so after
    ``ndim`` passes every axis has been rotated exactly once and the axis order
    is the one it started in.
    """
    out = np.asarray(tensor, dtype=np.float64)
    for _ in range(out.ndim):
        out = np.moveaxis(np.tensordot(rot, out, axes=([1], [0])), 0, -1)
    return out


def rotate_second_moments(unique, rot) -> np.ndarray:
    """Rotate ``(n, 6)`` unique quadrupole components through the full tensor and back."""
    labels = ["XX", "XY", "YY", "XZ", "YZ", "ZZ"]
    out = []
    for row in np.atleast_2d(np.asarray(unique, dtype=np.float64)):
        full = expand_multipole(dict(zip(labels, row)), 2)
        out.append(unique_components(rotate_multipole(full, rot), "quadrupole"))
    return np.array(out)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def fragment_formula(symbols, charge: int = 0) -> str:
    """``H2O``, ``H3O+``, ``OH-`` -- a fragment's conventional name."""
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    hill = "".join(
        f"{s}{counts[s] if counts[s] > 1 else ''}" for s in sorted(counts)
    )
    alias = _FORMULA_ALIASES.get((hill, int(charge)))
    if alias is not None:
        return alias
    sign = "+" * charge if charge > 0 else "-" * (-charge)
    return hill + sign


def fragmentation_config_type(symbols, fragment_idx, fragment_charges) -> str:
    """``H2O_H3O+_H2O``: the per-fragment formulas of one fragmentation, in order."""
    fragment_idx = np.asarray(fragment_idx)
    return "_".join(
        fragment_formula(
            [s for s, i in zip(symbols, fragment_idx) if i == f], fragment_charges[f]
        )
        for f in range(len(fragment_charges))
    )


def canonical_basis(basis: str) -> str:
    """``def2-tzvpd`` -> ``def2-TZVPD``.

    The AIMD and EDA templates in ``qchem_roundtrip/templates/`` spell the same
    basis with different case, and Q-Chem echoes back whichever it was given. The
    file name slug is case-insensitive so it does not care, but a ``basis=``
    header that varies between files does, so it is normalized here rather than
    left to whoever typed the template.
    """
    if basis.lower().startswith("def2-"):
        return "def2-" + basis[5:].upper()
    return basis


def same_level_of_theory(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Whether two ``(method, basis)`` pairs name the same level, ignoring case."""
    return tuple(x.lower() for x in a) == tuple(x.lower() for x in b)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


def fmt(arr) -> str:
    """Flatten to a space-separated ``%.12e`` string, as ``parse_roundtrip.py`` does."""
    return " ".join(f"{v:.12e}" for v in np.asarray(arr).ravel())


@dataclass
class Fragmentation:
    """One decomposition of a frame, with everything the EDA job said about it.

    Every array is in the frame's canonical atom order and canonical orientation,
    not the EDA job's.
    """

    #: 0-based fragment index per atom, ``(n_atoms,)``.
    fragment_idx: np.ndarray
    fragment_charges: list[int]
    fragment_mults: list[int]
    #: Isolated-fragment SCF energies, Hartree.
    fragment_energies: np.ndarray
    #: Isolated-fragment dipoles ``(n_frag, 3)``, e*a0, about the frame origin.
    fragment_dipoles: np.ndarray
    #: Isolated-fragment second moments ``(n_frag, 6)``, e*a0^2, same origin.
    fragment_second_moments: np.ndarray
    #: Isolated-fragment Mulliken charges, ``(n_atoms,)``.
    fragment_mulliken: np.ndarray
    #: EDA components, Hartree, keyed without the ``eda_`` prefix.
    eda: dict[str, float]
    #: The harvester's rank: 0 is the O-H assignment with the smallest total
    #: bond-length sum, and higher ranks are progressively more strained.
    rank: int = 0
    #: Which fragment carries the excess charge.
    charge_fragment: int = 0
    #: Total O-H distance sum in excess of the rank-0 assignment's, Angstrom.
    excess_distance: float = 0.0
    source: str = ""

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_charges)


@dataclass
class MultiFragFrame:
    """One geometry with its forces and every fragmentation's EDA labels."""

    symbols: list[str]
    #: Angstrom, canonical frame (center of nuclear charge at the origin).
    positions: np.ndarray
    #: Hartree/bohr, ``-dE/dR``, same frame.
    forces: np.ndarray
    #: Supersystem SCF energy, Hartree. One wavefunction, so one value.
    energy: float
    #: Supersystem Mulliken charges, ``(n_atoms,)``. Also fragmentation independent.
    mulliken: np.ndarray
    #: Supersystem Cartesian multipoles, full symmetric tensors, rotated into the
    #: canonical frame. Fragmentation independent for the same reason.
    multipoles: dict[str, np.ndarray]
    fragmentations: list[Fragmentation]
    total_charge: int
    multiplicity: int
    method: str
    basis: str
    config_type: str
    sample_id: int = 0
    #: Q-Chem's 1-based AIMD ``TIME STEP #``.
    aimd_step: int = 0
    source: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_fragmentations(self) -> int:
        return len(self.fragmentations)

    def properties(self) -> str:
        cols = ["species:S:1", "pos:R:3", "mulliken_charges:R:1"]
        for k in range(self.n_fragmentations):
            cols.append("fragment_idx:I:1" if k == 0 else f"fragment_idx_{k}:I:1")
        cols.append("forces:R:3")
        return "Properties=" + ":".join(cols)

    def header(self) -> str:
        frags = self.fragmentations
        # Built up front: nested quotes inside an f-string expression need
        # Python 3.12, and this package supports 3.10.
        config_types = " ".join(
            fragmentation_config_type(self.symbols, f.fragment_idx, f.fragment_charges)
            for f in frags
        )
        charges = " ".join(str(c) for f in frags for c in f.fragment_charges)
        mults = " ".join(str(m) for f in frags for m in f.fragment_mults)
        counts = " ".join(str(f.n_fragments) for f in frags)
        ranks = " ".join(str(f.rank) for f in frags)
        charge_fragments = " ".join(str(f.charge_fragment) for f in frags)
        keys = [self.properties(), f"energy={self.energy:.12e}"]

        # One value per fragmentation, in rank order.
        for name in EDA_KEY_ORDER:
            if all(name in f.eda for f in frags):
                keys.append(f'eda_{name}="{fmt([f.eda[name] for f in frags])}"')

        keys += [
            f"n_fragmentations={self.n_fragmentations}",
            f'n_fragments="{counts}"',
            f'fragmentation_ranks="{ranks}"',
            f'fragmentation_charge_fragment="{charge_fragments}"',
            f'fragmentation_excess_distance="{fmt([f.excess_distance for f in frags])}"',
            f'fragmentation_config_types="{config_types}"',
            f'fragment_charges="{charges}"',
            f'fragment_multiplicities="{mults}"',
            f'fragment_energies="{fmt(np.concatenate([f.fragment_energies for f in frags]))}"',
            f'fragment_dipoles="{fmt(np.concatenate([f.fragment_dipoles for f in frags]))}"',
            f'fragment_second_moments='
            f'"{fmt(np.concatenate([f.fragment_second_moments for f in frags]))}"',  # noqa: E501
            f'fragment_mulliken="{fmt(np.concatenate([f.fragment_mulliken for f in frags]))}"',
        ]
        for name in MULTIPOLE_NAMES:
            keys.append(f'{name}="{fmt(self.multipoles[name])}"')
        keys += [
            "multipole_format=tensor",
            f"charge={self.total_charge}",
            f"multiplicity={self.multiplicity}",
            f"method={self.method}",
            f"basis={self.basis}",
            f"config_type={self.config_type}",
            f"sample_id={self.sample_id}",
            f"aimd_step={self.aimd_step}",
            f"source={self.source}",
        ]
        if any(f.source for f in frags):
            sources = " ".join(f.source for f in frags)
            # Not `eda_sources`: the `eda_` prefix is reserved for energy
            # components, which load_extxyz reads as floats.
            keys.append(f'label_sources="{sources}"')
        keys += [f"{k}={v}" for k, v in self.extra.items()]
        keys.append("units=atomic")
        return " ".join(keys)

    def write(self, fh) -> None:
        fh.write(f"{self.n_atoms}\n{self.header()}\n")
        idx = [np.asarray(f.fragment_idx, dtype=int) for f in self.fragmentations]
        for a in range(self.n_atoms):
            x, y, z = self.positions[a]
            fx, fy, fz = self.forces[a]
            row = (
                f"{self.symbols[a]:<3} {x:18.10f} {y:18.10f} {z:18.10f} "
                f"{self.mulliken[a]:16.10f}"
            )
            row += "".join(f" {int(col[a]):4d}" for col in idx)
            row += f" {fx:22.14e} {fy:22.14e} {fz:22.14e}\n"
            fh.write(row)


@dataclass
class TrajectoryFrame:
    """One AIMD frame of a system that has no EDA decomposition to carry.

    The bare ``H3O+`` and ``OH-`` trajectories are single fragments, so there is
    nothing for an EDA job to decompose and no ``eda_*`` labels exist for them.
    What they do have is the thing the cluster frames lack: a **true one-body**
    energy and gradient at thousands of geometries, which is what a monomer
    anchor is for.

    Written with the ordinary single-fragmentation schema so
    ``rsfff.train.data.load_extxyz`` reads them with no special case. There are no
    Mulliken charges or multipoles: an AIMD run prints neither per step.
    """

    symbols: list[str]
    #: Angstrom, canonical frame (center of nuclear charge at the origin).
    positions: np.ndarray
    #: Hartree/bohr, ``-dE/dR``, same frame.
    forces: np.ndarray
    energy: float
    total_charge: int
    multiplicity: int
    method: str
    basis: str
    config_type: str
    sample_id: int = 0
    aimd_step: int = 0
    source: str = ""
    #: Optional 3x3 polarizability, a0^3, in the same frame.
    polarizability: np.ndarray | None = None
    #: Optional supersystem Mulliken charges and Cartesian multipoles, for frames
    #: that came from a job that printed them (an opt+freq reference, say).
    mulliken: np.ndarray | None = None
    multipoles: dict[str, np.ndarray] | None = None
    frequencies: np.ndarray | None = None
    ir_intensities: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def properties(self) -> str:
        cols = ["species:S:1", "pos:R:3"]
        if self.mulliken is not None:
            cols.append("mulliken_charges:R:1")
        cols += ["fragment_idx:I:1", "forces:R:3"]
        return "Properties=" + ":".join(cols)

    def header(self) -> str:
        keys = [self.properties(), f"energy={self.energy:.12e}"]
        keys.append(f'fragment_energies="{self.energy:.12e}"')
        if self.multipoles is not None:
            for name in MULTIPOLE_NAMES:
                keys.append(f'{name}="{fmt(self.multipoles[name])}"')
            # A one-fragment system's fragment moments *are* its molecular ones.
            keys.append(f'fragment_dipoles="{fmt(self.multipoles["dipole"])}"')
            keys.append(
                f'fragment_second_moments='
                f'"{fmt(unique_components(self.multipoles["quadrupole"], "quadrupole"))}"'
            )
            keys.append("multipole_format=tensor")
        if self.polarizability is not None:
            keys.append(f'polarizability="{fmt(self.polarizability)}"')
        if self.frequencies is not None and len(self.frequencies):
            keys.append(f'frequencies="{fmt(self.frequencies)}"')
        if self.ir_intensities is not None and len(self.ir_intensities):
            keys.append(f'ir_intensities="{fmt(self.ir_intensities)}"')
        keys += [
            "n_fragments=1",
            f"charge={self.total_charge}",
            f"multiplicity={self.multiplicity}",
            f'fragment_charges="{self.total_charge}"',
            f'fragment_multiplicities="{self.multiplicity}"',
            f"method={self.method}",
            f"basis={self.basis}",
            f"config_type={self.config_type}",
            f"sample_id={self.sample_id}",
            f"aimd_step={self.aimd_step}",
            f"source={self.source}",
            "units=atomic",
        ]
        return " ".join(keys)

    def write(self, fh) -> None:
        fh.write(f"{self.n_atoms}\n{self.header()}\n")
        for a in range(self.n_atoms):
            x, y, z = self.positions[a]
            fx, fy, fz = self.forces[a]
            row = f"{self.symbols[a]:<3} {x:18.10f} {y:18.10f} {z:18.10f}"
            if self.mulliken is not None:
                row += f" {self.mulliken[a]:16.10f}"
            row += f" {0:4d} {fx:22.14e} {fy:22.14e} {fz:22.14e}\n"
            fh.write(row)


def write_frames(path, frames) -> None:
    """Write a list of :class:`MultiFragFrame` or :class:`TrajectoryFrame` to one file."""
    with open(path, "w") as fh:
        for frame in frames:
            frame.write(fh)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_multifrag_extxyz(path) -> list[dict]:
    """Read a multi-fragmentation file back into per-frame dicts.

    Every per-fragmentation quantity comes back with a leading axis of length
    ``n_fragmentations``, so ``frame["eda"]["ct"][k]`` and
    ``frame["fragment_idx"][k]`` refer to the same decomposition.

    This is the access path for a model that mixes over the fragmentations.
    Training code that wants a single one should use
    ``rsfff.train.data.load_extxyz(path, fragmentation=k)``, which returns the
    ordinary single-fragmentation dataset.
    """
    from ase.io import iread

    frames = []
    for atoms in iread(str(path), index=":"):
        info = atoms.info
        n_frag_sets = int(info["n_fragmentations"])
        counts = np.asarray(info["n_fragments"], dtype=int).reshape(n_frag_sets)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        total = int(counts.sum())
        n = len(atoms)

        def per_fragment(key, width):
            flat = np.asarray(info[key], dtype=np.float64).reshape(total, width)
            if width == 1:
                flat = flat[:, 0]
            return [flat[offsets[k] : offsets[k + 1]] for k in range(n_frag_sets)]

        fragment_idx = np.stack(
            [
                np.asarray(
                    atoms.arrays["fragment_idx" if k == 0 else f"fragment_idx_{k}"],
                    dtype=np.int64,
                )
                for k in range(n_frag_sets)
            ]
        )
        # ASE recognizes "dipole" as a calculator property and moves it out of
        # info, exactly as rsfff.train.data.load_extxyz has to work around.
        def multipole(name):
            value = info.get(name)
            if value is None and atoms.calc is not None:
                value = atoms.calc.results.get(name)
            if value is None:
                raise KeyError(f"{path}: frame has no {name!r} header")
            return np.asarray(value, dtype=np.float64)

        eda = {
            key[4:]: np.asarray(value, dtype=np.float64).reshape(n_frag_sets)
            for key, value in info.items()
            if key.startswith("eda_")
        }
        frames.append(
            {
                "symbols": atoms.get_chemical_symbols(),
                "positions": np.asarray(atoms.get_positions()),
                "forces": np.asarray(atoms.get_forces()),
                "energy": float(atoms.get_potential_energy()),
                "mulliken": np.asarray(atoms.arrays["mulliken_charges"]),
                "n_fragmentations": n_frag_sets,
                "n_fragments": counts,
                "fragment_idx": fragment_idx,
                "fragment_charges": per_fragment("fragment_charges", 1),
                "fragment_multiplicities": per_fragment("fragment_multiplicities", 1),
                "fragment_energies": per_fragment("fragment_energies", 1),
                "fragment_dipoles": per_fragment("fragment_dipoles", 3),
                "fragment_second_moments": per_fragment("fragment_second_moments", 6),
                "fragment_mulliken": np.asarray(
                    info["fragment_mulliken"], dtype=np.float64
                ).reshape(n_frag_sets, n),
                "eda": eda,
                "ranks": np.asarray(info["fragmentation_ranks"], dtype=int),
                "excess_distance": np.asarray(
                    info["fragmentation_excess_distance"], dtype=np.float64
                ).reshape(n_frag_sets),
                # One space-separated token per fragmentation. ASE keeps a quoted
                # non-numeric value as a single string, so split it back out here.
                "config_types": " ".join(
                    str(s) for s in np.atleast_1d(info["fragmentation_config_types"])
                ).split(),
                "multipoles": {
                    name: multipole(name).reshape((3,) * rank)
                    for rank, name in enumerate(MULTIPOLE_NAMES, start=1)
                },
                "info": info,
            }
        )
    return frames


__all__ = [
    "EDA_KEY_ORDER",
    "MULTIPOLE_NAMES",
    "RMSD_TOL",
    "Fragmentation",
    "MultiFragFrame",
    "TrajectoryFrame",
    "canonical_basis",
    "center_of_nuclear_charge",
    "fmt",
    "fragment_formula",
    "fragmentation_config_type",
    "nuclear_charges",
    "read_multifrag_extxyz",
    "recenter",
    "rotate_multipole",
    "rotate_second_moments",
    "rotation_between",
    "same_level_of_theory",
    "write_frames",
]
