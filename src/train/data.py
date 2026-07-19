"""Dataset, ragged batching, and reference-energy loading for MLIP training.

Frames are read from ASE extended-XYZ (``energy`` in the header, per-atom ``forces``
columns) and stored as flat concatenated tensors. A minibatch is a single ragged graph:
all atoms of the selected frames are concatenated, and ``batch_idx`` labels each atom
with its frame -- exactly the layout the featurizer and ``index_add_`` pooling expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import json
from pathlib import Path

import numpy as np
import torch


@dataclass
class Batch:
    """A ragged batch of molecules (one concatenated graph).

    positions      : (Ntot, 3) float
    atomic_numbers : (Ntot,)   long
    batch_idx      : (Ntot,)   long, frame id per atom in [0, n_systems)
    n_systems      : number of frames B in this batch
    energy         : (B,)      per-molecule total energy target
    forces         : (Ntot, 3) per-atom force target
    """

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    batch_idx: torch.Tensor
    n_systems: int
    energy: torch.Tensor
    forces: torch.Tensor

    def to(self, device) -> "Batch":
        return Batch(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            batch_idx=self.batch_idx.to(device),
            n_systems=self.n_systems,
            energy=self.energy.to(device),
            forces=self.forces.to(device),
        )


class MoleculeDataset:
    """In-memory dataset of labeled frames with flat storage and ragged batching."""

    def __init__(
        self,
        positions: torch.Tensor,   # (Ntot_all, 3)
        atomic_numbers: torch.Tensor,  # (Ntot_all,)
        forces: torch.Tensor,      # (Ntot_all, 3)
        energy: torch.Tensor,      # (n_frames,)
        counts: torch.Tensor,      # (n_frames,) atoms per frame
    ) -> None:
        self._pos = positions
        self._num = atomic_numbers.long()
        self._forces = forces
        self._energy = energy
        self._counts = counts.long()
        self._offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.cumsum(self._counts, 0))
        )

    def __len__(self) -> int:
        return int(self._counts.shape[0])

    @property
    def unique_atomic_numbers(self) -> list[int]:
        return sorted(int(z) for z in torch.unique(self._num).tolist())

    def flat_batch(self, indices) -> Batch:
        """Assemble the frames in ``indices`` into one ragged :class:`Batch`."""
        idx = torch.as_tensor(indices, dtype=torch.long)
        counts = self._counts[idx]
        # gather atom rows for each selected frame
        atom_slices = [
            torch.arange(self._offsets[i], self._offsets[i + 1]) for i in idx.tolist()
        ]
        rows = torch.cat(atom_slices) if atom_slices else torch.empty(0, dtype=torch.long)
        batch_idx = torch.repeat_interleave(torch.arange(idx.shape[0]), counts)
        return Batch(
            positions=self._pos[rows].clone(),
            atomic_numbers=self._num[rows],
            batch_idx=batch_idx,
            n_systems=int(idx.shape[0]),
            energy=self._energy[idx],
            forces=self._forces[rows],
        )


def load_extxyz(path, dtype: torch.dtype = torch.float32) -> MoleculeDataset:
    """Read every frame of an extended-XYZ file into a :class:`MoleculeDataset`.

    ASE attaches ``energy=`` and the per-atom ``forces`` to a ``SinglePointCalculator``,
    so they are read via ``get_potential_energy()`` / ``get_forces()`` (not ``info`` /
    ``arrays``). Positions are Angstrom and energies Hartree, as stored. The dataset's
    forces are stored in **Hartree/Bohr** (psi4 gradient convention, see
    ``scripts/generate_dataset.py``); we divide by ``ase.units.Bohr`` to convert to
    **Hartree/Angstrom** so they match autograd forces ``-dE/d(positions in Angstrom)``.
    (Verified by finite difference: stored/Bohr == -dE/dx to ~1e-4.)
    """
    import ase.units
    from ase.io import iread

    pos_list, num_list, force_list, energy_list, counts = [], [], [], [], []
    for atoms in iread(str(path), index=":"):
        n = len(atoms)
        pos_list.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        num_list.append(np.asarray(atoms.numbers, dtype=np.int64))
        force_list.append(np.asarray(atoms.get_forces(), dtype=np.float64) / ase.units.Bohr)
        energy_list.append(float(atoms.get_potential_energy()))
        counts.append(n)

    positions = torch.tensor(np.concatenate(pos_list), dtype=dtype)
    atomic_numbers = torch.tensor(np.concatenate(num_list), dtype=torch.long)
    forces = torch.tensor(np.concatenate(force_list), dtype=dtype)
    energy = torch.tensor(energy_list, dtype=dtype)
    counts_t = torch.tensor(counts, dtype=torch.long)
    return MoleculeDataset(positions, atomic_numbers, forces, energy, counts_t)


def split_indices(n: int, holdout_fraction: float, seed: int = 0):
    """Deterministic train/val split; returns (train_idx, val_idx) long tensors."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_val = int(round(holdout_fraction * n))
    return perm[n_val:], perm[:n_val]


def load_reference_energies(path, neighbor_types) -> torch.Tensor:
    """Load per-species reference energies aligned to ``neighbor_types``.

    ``path`` is the JSON written by ``scripts/atomic_references.py`` (element symbol ->
    Hartree). ``neighbor_types`` is the sorted list of atomic numbers used by the
    featurizer; the returned tensor is indexed by species index (its position in that
    sorted list), matching ``LambdaFeatures.species_idx``.
    """
    from ase.data import chemical_symbols

    ref = json.loads(Path(path).read_text())["energies"]
    e0 = torch.empty(len(neighbor_types), dtype=torch.get_default_dtype())
    for i, z in enumerate(neighbor_types):
        sym = chemical_symbols[int(z)]
        if sym not in ref:
            raise KeyError(
                f"no atomic reference energy for {sym} (Z={z}) in {path}; "
                f"run: python scripts/atomic_references.py {sym}"
            )
        e0[i] = float(ref[sym])
    return e0
