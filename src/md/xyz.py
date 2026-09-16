"""Plain-XYZ read/write that keeps the comment line intact.

``ase.io.read`` runs every comment through the extended-XYZ key=value parser, which turns a
free-form label like ``# w6 cage`` into ``{'#w6': True, 'cage': True}`` -- the cluster names in
a hand-curated reference file are exactly what gets destroyed. These two functions do nothing
clever: the comment is a string in and a string out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

__all__ = ["Frame", "read_xyz", "write_xyz"]


@dataclass
class Frame:
    symbols: list[str]
    positions: np.ndarray          # (N, 3) Angstrom
    comment: str = ""

    @property
    def numbers(self) -> np.ndarray:
        return np.array([atomic_numbers[s] for s in self.symbols], dtype=np.int64)

    @property
    def masses(self) -> np.ndarray:
        return atomic_masses[self.numbers]

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def read_xyz(path) -> list[Frame]:
    lines = Path(path).read_text().splitlines()
    frames: list[Frame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        body = lines[i + 2: i + 2 + n]
        if len(body) != n:
            raise ValueError(f"{path}: frame starting at line {i + 1} is truncated")
        symbols, positions = [], []
        for row in body:
            fields = row.split()
            symbol = fields[0]
            if symbol.isdigit():
                symbol = chemical_symbols[int(symbol)]
            symbols.append(symbol)
            positions.append([float(v) for v in fields[1:4]])
        frames.append(Frame(symbols, np.asarray(positions, dtype=float), comment))
        i += n + 2
    return frames


def write_xyz(path, frames) -> None:
    with open(path, "w") as handle:
        for frame in frames:
            handle.write(f"{frame.n_atoms}\n{frame.comment}\n")
            for symbol, (x, y, z) in zip(frame.symbols, frame.positions):
                handle.write(f"{symbol:<2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")
