"""Parser for Q-Chem ``JOBTYPE = opt`` output files that end in a frequency job.

``qchem_roundtrip/templates/opt_and_freq.in`` asks for ``hessian_verify =
recomputed`` and ``final_vibrational_analysis = true``, so these outputs carry,
after the geometry optimization, an analytic Hessian at the converged structure
and -- because Q-Chem solves the CPSCF equations on the way there -- a
**polarizability**. For H3O+ and OH- that is the only polarizability reference
that exists anywhere in this repo.

Two details carry the whole file's correctness, and neither is visible in the
numbers themselves:

**The CPSCF block prints the negative of the polarizability.** The table under
``Polarizability Matrix (a.u.)`` is ``d^2E/dF^2``, so every diagonal entry comes
out negative::

    Polarizability Matrix (a.u.)
                1           2           3
        1  -7.0347950  -0.0000000   0.0000055
        2  -0.0000000  -7.0347948  -0.0000000
        3   0.0000055  -0.0000000  -5.7382329

This is the *opposite* sign convention to the dedicated ``JOBTYPE =
polarizability`` block that ``scripts/parse_polarizability.py`` reads, which
prints ``+alpha``. :func:`parse_opt_output` negates, then symmetrizes -- the
finite CPSCF solve leaves the raw matrix very slightly asymmetric and the
physical object is symmetric, so symmetrizing beats picking a triangle.

**Pair the tensor with the geometry that precedes it, not the one that follows
it.** These jobs print a *further* ``Standard Nuclear Orientation`` after the
CPSCF solve, during the vibrational analysis, where Q-Chem reports a point group
and may reorient onto the symmetry axes even with ``SYMMETRY = false``. So the
geometry, energy, Mulliken charges, multipoles and gradient are all taken from
the last blocks *before* the ``Polarizability Matrix`` line.

In the three monomer jobs this repo has, that later block turns out to be the
same geometry to 0 (H3O+, OH-) or 1.7e-5 Angstrom (H2O) -- their optimized
structures are already symmetry aligned, so taking the wrong one would not
actually have hurt. The rule is here because it is right by construction rather
than by luck: for an input that is not already aligned the two blocks are a
rotation apart, and a rotated tensor is not a detectable error. Its eigenvalues,
its isotropic average and its anisotropy all survive a rotation intact, so no
unit or magnitude check downstream would ever notice.

The gradient at that geometry is the converged one (~1e-7 Hartree/bohr) and is
returned as forces, so the frame can be written with the same schema as every
other labeled frame rather than with a hole in it.

Normal-mode displacement vectors are deliberately **not** returned: they are
printed in that later, reoriented frame, and only the frequencies and IR
intensities -- which are rotation invariant -- are safe to carry without doing
the alignment. Energies are Hartree, forces Hartree/bohr, positions Angstrom,
multipoles ``e*a0^n`` after :func:`to_atomic_units`, and the polarizability
``a0^3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .qchem_force import parse_gradient
from .qchem_out import (
    SCF_ITER,
    SNO_HEADER,
    QChemParseError,
    find_all,
    job_completed,
    method_and_basis,
    multipoles_to_atomic_units,
    parse_geometry,
    parse_molecule_block,
    parse_mulliken,
    parse_multipoles,
    parse_rem,
)

_POLARIZABILITY_HEADER = "Polarizability Matrix (a.u.)"
_CONVERGED_MARKER = "OPTIMIZATION CONVERGED"
_FINAL_ENERGY = "Final energy is"

#: The SCF row and the ``Final energy is`` line are two printings of the same
#: number, to 10 and 12 decimals. Disagreement beyond this means they are *not*
#: the same wavefunction, i.e. the block pairing above went wrong.
ENERGY_ATOL = 1e-8


@dataclass
class OptRecord:
    """Everything pulled out of one Q-Chem opt+freq output, at the converged geometry."""

    path: str = ""
    symbols: list[str] = field(default_factory=list)
    #: Angstrom, the last standard nuclear orientation before the CPSCF solve.
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Hartree/bohr, ``-dE/dR``, in the same frame. ~1e-7 at a converged minimum.
    forces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Total electronic energy at the converged geometry, Hartree.
    energy: float = float("nan")
    total_charge: int = 0
    multiplicity: int = 1
    mulliken_charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Cartesian multipoles as full symmetric tensors, about the frame origin.
    multipoles: dict[str, np.ndarray] = field(default_factory=dict)
    #: Symmetric 3x3 polarizability, sign-corrected, in the same frame, a0^3.
    polarizability: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    #: Harmonic frequencies, cm^-1. Negative entries are imaginary modes.
    frequencies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: IR intensities, km/mol, aligned with ``frequencies``.
    ir_intensities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = ""
    basis: str = ""
    #: The optimizer printed "OPTIMIZATION CONVERGED".
    converged: bool = False
    #: Q-Chem printed its sign-off, i.e. the file is not truncated.
    completed: bool = False
    units: str = "qchem"

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def _parse_polarizability(lines: list[str], start: int) -> np.ndarray:
    """The 3x3 table under ``start``, negated and symmetrized. See the module docstring."""
    rows: list[list[float]] = []
    for ln in lines[start + 1 :]:
        toks = ln.split()
        # The column-index header is three bare integers; a data row is a row
        # index followed by three decimals.
        if len(toks) == 4 and toks[0] in ("1", "2", "3"):
            try:
                rows.append([float(t) for t in toks[1:]])
            except ValueError:
                break
            if len(rows) == 3:
                break
            continue
        if rows:
            break
    if len(rows) != 3:
        raise QChemParseError(f"malformed '{_POLARIZABILITY_HEADER}' table")
    alpha = -np.asarray(rows, dtype=np.float64)
    return 0.5 * (alpha + alpha.T)


def _parse_frequencies(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies cm^-1, IR intensities km/mol)`` from the vibrational analysis."""
    freqs: list[float] = []
    intens: list[float] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("Frequency:"):
            freqs += [float(t) for t in stripped.split()[1:]]
        elif stripped.startswith("IR Intens:"):
            intens += [float(t) for t in stripped.split()[2:]]
    if freqs and intens and len(freqs) != len(intens):
        raise QChemParseError(
            f"{len(freqs)} frequencies against {len(intens)} IR intensities"
        )
    return np.array(freqs), np.array(intens)


def _scf_energy_before(lines: list[str], stop: int) -> float:
    """The last SCF iteration energy before line ``stop``."""
    energy = None
    for ln in lines[:stop]:
        m = SCF_ITER.match(ln)
        if m:
            energy = float(m.group(1))
    if energy is None:
        raise QChemParseError("no SCF iterations before the polarizability block")
    return energy


def _last_before(indices: list[int], stop: int) -> int | None:
    hits = [i for i in indices if i < stop]
    return hits[-1] if hits else None


def parse_opt_output(path: str) -> OptRecord:
    """Parse one Q-Chem opt+freq output into an :class:`OptRecord` (native units)."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()

    pol_idx = find_all(lines, _POLARIZABILITY_HEADER)
    if not pol_idx:
        raise QChemParseError(
            f"no '{_POLARIZABILITY_HEADER}' block: the job did not reach the CPSCF solve"
        )
    pol_line = pol_idx[-1]

    sno = _last_before(find_all(lines, SNO_HEADER), pol_line)
    if sno is None:
        raise QChemParseError("no Standard Nuclear Orientation before the polarizability")
    symbols, positions = parse_geometry(lines, sno)

    _, _, frag_natoms, total_charge, multiplicity = parse_molecule_block(lines)
    if sum(frag_natoms) != len(symbols):
        raise QChemParseError(
            f"geometry has {len(symbols)} atoms but $molecule declares {sum(frag_natoms)}"
        )

    energy = _scf_energy_before(lines, pol_line)
    final = [ln for ln in lines if _FINAL_ENERGY in ln]
    if final and abs(float(final[-1].split()[-1]) - energy) > ENERGY_ATOL:
        raise QChemParseError(
            f"SCF energy before the polarizability ({energy:.10f}) does not match "
            f"'{_FINAL_ENERGY}' ({final[-1].split()[-1]}); the blocks were mispaired"
        )

    mulliken_at = _last_before(find_all(lines, "Mulliken Net Atomic Charges"), pol_line)
    multipole_at = _last_before(find_all(lines, "Cartesian Multipole Moments"), pol_line)
    if multipole_at is None:
        raise QChemParseError("no Cartesian multipoles before the polarizability")

    frequencies, ir_intensities = _parse_frequencies(lines)
    method, basis = method_and_basis(parse_rem(lines))

    return OptRecord(
        path=path,
        symbols=symbols,
        positions=positions,
        forces=-parse_gradient(lines[:pol_line], len(symbols)),
        energy=energy,
        total_charge=total_charge,
        multiplicity=multiplicity,
        mulliken_charges=parse_mulliken(lines, len(symbols), start=mulliken_at),
        multipoles=parse_multipoles(lines, start=multipole_at),
        polarizability=_parse_polarizability(lines, pol_line),
        frequencies=frequencies,
        ir_intensities=ir_intensities,
        method=method,
        basis=basis,
        converged=any(_CONVERGED_MARKER in ln for ln in lines),
        completed=job_completed(text),
        units="qchem",
    )


def to_atomic_units(rec: OptRecord) -> OptRecord:
    """Convert the multipoles to ``e*a0^n``; everything else is already atomic.

    The CPSCF polarizability is printed in a.u. (``a0^3``) and so is untouched --
    unlike the multipoles, which Q-Chem prints in Debye-Angstrom^(n-1).
    Mutates and returns ``rec``; raises if already converted.
    """
    if rec.units == "atomic":
        raise ValueError("record is already in atomic units")
    rec.multipoles = multipoles_to_atomic_units(rec.multipoles)
    rec.units = "atomic"
    return rec


def check_consistency(rec: OptRecord, max_force: float = 1e-4) -> list[str]:
    """Human-readable warnings about a parsed opt record.

    ``max_force`` is in Hartree/bohr and is tight on purpose: this is a converged
    minimum, so anything above ~1e-5 means the geometry taken is not the one the
    optimizer settled on.
    """
    msgs = []
    if not rec.completed:
        msgs.append("no Q-Chem sign-off: the output is truncated")
    if not rec.converged:
        msgs.append(f"the optimizer never printed '{_CONVERGED_MARKER}'")
    peak = float(np.abs(rec.forces).max()) if rec.forces.size else 0.0
    if peak > max_force:
        msgs.append(f"residual force {peak:.3g} Ha/bohr at a supposedly converged geometry")
    eigvals = np.linalg.eigvalsh(rec.polarizability)
    if (eigvals <= 0).any():
        msgs.append(
            f"polarizability is not positive definite (eigenvalues {eigvals}); the sign "
            f"convention of the CPSCF block may have changed"
        )
    imaginary = int((rec.frequencies < 0).sum())
    if imaginary:
        msgs.append(f"{imaginary} imaginary frequency/frequencies: not a minimum")
    return msgs


__all__ = [
    "OptRecord",
    "QChemParseError",
    "check_consistency",
    "parse_opt_output",
    "to_atomic_units",
]
