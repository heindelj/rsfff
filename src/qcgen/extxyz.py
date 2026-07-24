"""Extended-XYZ I/O for the reference-data pipeline.

The labeled-frame writer reproduces, key-for-key, the header schema already used
in ``data/labels/*.extxyz`` (see the psi4 pipeline in
``scripts/generate_dataset.py``), so datasets from the pyscf/gpu4pyscf pipeline
are drop-in compatible with existing consumers. Only the ``method``/``basis``
*values* change (e.g. ``wB97M-V`` instead of ``b3lyp``).

All array-valued quantities are stored in atomic units:
energy (Hartree), forces (Hartree/Bohr), dipole (e*a0), quadrupole (e*a0^2),
polarizability (a0^3), dipole derivatives (e). Positions are in Angstrom.
"""

from __future__ import annotations

import numpy as np


def read_xyz(path):
    """Return ``(symbols, coords)`` from a plain ``.xyz`` file (coords in Angstrom)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    natoms = int(lines[0].split()[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + natoms]:
        tok = line.split()
        symbols.append(tok[0])
        coords.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return symbols, np.array(coords)


def write_xyz(path, symbols, coords, comment=""):
    """Write a single-frame plain ``.xyz`` file."""
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for sym, (x, y, z) in zip(symbols, np.asarray(coords)):
            fh.write(f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f}\n")


def fmt(arr):
    """Flatten a numeric array to a space-separated ``%.10e`` string."""
    return " ".join(f"{v:.10e}" for v in np.asarray(arr).ravel())


def write_geoms(path, symbols, geometries, comment_fn=None):
    """Write an unlabeled multi-frame ``.xyz`` (one frame per sampled geometry).

    Used by the Wigner stage: geometries only, no properties yet.
    """
    with open(path, "w") as fh:
        for idx, geom in enumerate(np.asarray(geometries)):
            comment = comment_fn(idx) if comment_fn else f"sample_index={idx}"
            fh.write(f"{len(symbols)}\n{comment}\n")
            for sym, (x, y, z) in zip(symbols, geom):
                fh.write(f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f}\n")


def read_geoms(path):
    """Read an unlabeled multi-frame ``.xyz``; return ``(symbols, [coords, ...])``.

    Assumes every frame shares the same atom ordering (true for Wigner samples of
    a single molecule). The comment line of each frame is ignored.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    symbols, frames = None, []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].split()[0])
        block = lines[i + 2 : i + 2 + natoms]
        syms, coords = [], []
        for line in block:
            tok = line.split()
            syms.append(tok[0])
            coords.append([float(tok[1]), float(tok[2]), float(tok[3])])
        if symbols is None:
            symbols = syms
        frames.append(np.array(coords))
        i += 2 + natoms
    return symbols, frames


def frame_header(data, meta):
    """Build the extended-XYZ comment line for one labeled frame.

    ``data`` holds the computed properties (energy, forces, dipole, quadrupole,
    polarizability, and optionally dipole_derivatives); ``meta`` holds
    bookkeeping keys (charge, mult, method, basis, name, index, temperature).
    Key order matches the existing ``data/labels/*.extxyz`` schema.

    ``dipole_derivatives`` is emitted only when present in ``data`` -- it is
    expensive and off by default (see ``compute.compute_reference_data``).
    """
    head = (
        "Properties=species:S:1:pos:R:3:forces:R:3 "
        f'energy={data["energy"]:.12e} '
        f'dipole="{fmt(data["dipole"])}" '
        f'quadrupole="{fmt(data["quadrupole"])}" '
        f'polarizability="{fmt(data["polarizability"])}" '
    )
    if "dipole_derivatives" in data:
        head += f'dipole_derivatives="{fmt(data["dipole_derivatives"])}" '
    return head + (
        f'charge={meta["charge"]} multiplicity={meta["mult"]} '
        f'method={meta["method"]} basis={meta["basis"]} '
        f'config_type={meta["name"]} sample_index={meta["index"]} '
        f'temperature={meta["temperature"]} units=atomic'
    )


def write_frame(fh, symbols, coords, data, meta):
    """Append one labeled extended-XYZ frame to an open file handle.

    Per-atom columns are ``species pos(3) forces(3)``; all other labels live on
    the header line built by :func:`frame_header`.
    """
    natoms = len(symbols)
    forces = np.asarray(data["forces"]).reshape(natoms, 3)
    fh.write(f"{natoms}\n{frame_header(data, meta)}\n")
    for sym, (x, y, z), (fx, fy, fz) in zip(symbols, coords, forces):
        fh.write(
            f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f} "
            f"{fx:20.12e} {fy:20.12e} {fz:20.12e}\n"
        )
