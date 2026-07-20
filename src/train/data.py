"""Dataset, ragged batching, and reference-energy loading for MLIP training.

Frames are read from ASE extended-XYZ (``energy`` in the header, per-atom ``forces``
columns) and stored as flat concatenated tensors. A minibatch is a single ragged graph:
all atoms of the selected frames are concatenated, and ``batch_idx`` labels each atom
with its frame -- exactly the layout the featurizer and ``index_add_`` pooling expect.

Units convention (single source of truth for the training pipeline)
-------------------------------------------------------------------
The extxyz labels are in atomic units (see ``scripts/generate_dataset.py``); the model
works in mixed Hartree/Angstrom units. Conversions applied at load time:

===================  ==============  =================  ==========================
quantity             stored (a.u.)   model units        conversion on load
===================  ==============  =================  ==========================
positions            Angstrom        Angstrom           none (stored in Angstrom)
energy               Hartree         Hartree            none
forces               Ha/Bohr         Ha/Angstrom        / Bohr
total charge         e               e                  none
dipole               e*a0            e*Angstrom         * Bohr
dipole derivatives   e (a0/a0)       e (Angstrom/Ang.)  none (dimensionless)
polarizability       a0^3            e^2*Ang^2/Ha       * Bohr^2  (a0^3 = e^2 a0^2/Ha)
===================  ==============  =================  ==========================

Dipoles/response labels are in the *stored* coordinate frame (psi4 ran with
``fix_frame``: no COM shift, no reorientation), so ``sum_i q_i r_i`` over the stored
positions is directly comparable -- including charged systems like H3O+, whose dipole
is origin-dependent. Dipole-derivative layout is ``(atom, d/dR_a, mu_b)``: entry
``[i, a, b] = d mu_b / d R_{i,a}``.
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

    positions           : (Ntot, 3) float
    atomic_numbers      : (Ntot,)   long
    batch_idx           : (Ntot,)   long, frame id per atom in [0, n_systems)
    n_systems           : number of frames B in this batch
    energy              : (B,)      per-molecule total energy target
    forces              : (Ntot, 3) per-atom force target
    total_charge        : (B,)      per-molecule total charge (e), or None
    dipole              : (B, 3)    molecular dipole target (e*Angstrom), or None
    polarizability      : (B, 3, 3) molecular polarizability (e^2*Ang^2/Ha), or None
    dipole_derivatives  : (Ntot, 3, 3) d mu_b / d R_{i,a} (e), or None
    """

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    batch_idx: torch.Tensor
    n_systems: int
    energy: torch.Tensor
    forces: torch.Tensor
    total_charge: torch.Tensor | None = None
    dipole: torch.Tensor | None = None
    polarizability: torch.Tensor | None = None
    dipole_derivatives: torch.Tensor | None = None

    def to(self, device) -> "Batch":
        opt = lambda t: t.to(device) if t is not None else None  # noqa: E731
        return Batch(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            batch_idx=self.batch_idx.to(device),
            n_systems=self.n_systems,
            energy=self.energy.to(device),
            forces=self.forces.to(device),
            total_charge=opt(self.total_charge),
            dipole=opt(self.dipole),
            polarizability=opt(self.polarizability),
            dipole_derivatives=opt(self.dipole_derivatives),
        )


class MoleculeDataset:
    """In-memory dataset of labeled frames with flat storage and ragged batching.

    The response-property fields are all-or-nothing per dataset: either every frame
    carries the label (tensor stored) or the field is ``None`` and the corresponding
    Batch field is ``None``.
    """

    def __init__(
        self,
        positions: torch.Tensor,   # (Ntot_all, 3)
        atomic_numbers: torch.Tensor,  # (Ntot_all,)
        forces: torch.Tensor,      # (Ntot_all, 3)
        energy: torch.Tensor,      # (n_frames,)
        counts: torch.Tensor,      # (n_frames,) atoms per frame
        *,
        total_charge: torch.Tensor | None = None,        # (n_frames,)
        dipole: torch.Tensor | None = None,              # (n_frames, 3)
        polarizability: torch.Tensor | None = None,      # (n_frames, 3, 3)
        dipole_derivatives: torch.Tensor | None = None,  # (Ntot_all, 3, 3)
    ) -> None:
        self._pos = positions
        self._num = atomic_numbers.long()
        self._forces = forces
        self._energy = energy
        self._counts = counts.long()
        self._offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.cumsum(self._counts, 0))
        )
        self._total_charge = total_charge
        self._dipole = dipole
        self._polarizability = polarizability
        self._dipole_derivatives = dipole_derivatives

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
            total_charge=self._total_charge[idx] if self._total_charge is not None else None,
            dipole=self._dipole[idx] if self._dipole is not None else None,
            polarizability=(
                self._polarizability[idx] if self._polarizability is not None else None
            ),
            dipole_derivatives=(
                self._dipole_derivatives[rows]
                if self._dipole_derivatives is not None
                else None
            ),
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

    bohr = float(ase.units.Bohr)  # Angstrom per bohr
    pos_list, num_list, force_list, energy_list, counts = [], [], [], [], []
    charge_list, dip_list, pol_list, dmu_list = [], [], [], []
    for atoms in iread(str(path), index=":"):
        n = len(atoms)
        pos_list.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        num_list.append(np.asarray(atoms.numbers, dtype=np.int64))
        force_list.append(np.asarray(atoms.get_forces(), dtype=np.float64) / bohr)
        energy_list.append(float(atoms.get_potential_energy()))
        counts.append(n)

        info = atoms.info
        charge_list.append(float(info.get("charge", 0.0)))
        # ASE recognizes "dipole" as a calculator property and moves it out of info.
        dip = info.get("dipole")
        if dip is None and atoms.calc is not None:
            dip = atoms.calc.results.get("dipole")
        if dip is not None:  # e*a0 -> e*Angstrom
            dip_list.append(np.asarray(dip, dtype=np.float64).reshape(3) * bohr)
        if "polarizability" in info:  # a0^3 -> e^2*Ang^2/Ha
            pol_list.append(
                np.asarray(info["polarizability"], dtype=np.float64).reshape(3, 3)
                * bohr**2
            )
        if "dipole_derivatives" in info:  # (atom, d/dR, mu) layout; dimensionless (e)
            dmu_list.append(
                np.asarray(info["dipole_derivatives"], dtype=np.float64).reshape(n, 3, 3)
            )

    n_frames = len(counts)
    if len(dip_list) not in (0, n_frames) or len(pol_list) not in (0, n_frames) or len(
        dmu_list
    ) not in (0, n_frames):
        raise ValueError(
            f"{path}: response labels present on some frames but not all "
            f"(dipole {len(dip_list)}, polarizability {len(pol_list)}, "
            f"dipole_derivatives {len(dmu_list)} of {n_frames})"
        )

    positions = torch.tensor(np.concatenate(pos_list), dtype=dtype)
    atomic_numbers = torch.tensor(np.concatenate(num_list), dtype=torch.long)
    forces = torch.tensor(np.concatenate(force_list), dtype=dtype)
    energy = torch.tensor(energy_list, dtype=dtype)
    counts_t = torch.tensor(counts, dtype=torch.long)
    return MoleculeDataset(
        positions, atomic_numbers, forces, energy, counts_t,
        total_charge=torch.tensor(charge_list, dtype=dtype),
        dipole=torch.tensor(np.stack(dip_list), dtype=dtype) if dip_list else None,
        polarizability=torch.tensor(np.stack(pol_list), dtype=dtype) if pol_list else None,
        dipole_derivatives=(
            torch.tensor(np.concatenate(dmu_list), dtype=dtype) if dmu_list else None
        ),
    )


def concatenate_datasets(datasets: "list[MoleculeDataset]") -> MoleculeDataset:
    """Concatenate datasets into one (frames kept in order).

    Optional label fields survive only if present on every input dataset (a mixed
    concatenation would silently drop supervision otherwise -- refuse instead).
    """
    if not datasets:
        raise ValueError("no datasets to concatenate")
    if len(datasets) == 1:
        return datasets[0]

    def _cat_optional(name: str):
        fields = [getattr(d, name) for d in datasets]
        present = [f is not None for f in fields]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"cannot concatenate: {name} present on some datasets only")
        return torch.cat(fields)

    return MoleculeDataset(
        torch.cat([d._pos for d in datasets]),
        torch.cat([d._num for d in datasets]),
        torch.cat([d._forces for d in datasets]),
        torch.cat([d._energy for d in datasets]),
        torch.cat([d._counts for d in datasets]),
        total_charge=_cat_optional("_total_charge"),
        dipole=_cat_optional("_dipole"),
        polarizability=_cat_optional("_polarizability"),
        dipole_derivatives=_cat_optional("_dipole_derivatives"),
    )


def load_datasets(paths, dtype: torch.dtype = torch.float32) -> MoleculeDataset:
    """Load one or more extxyz files into a single concatenated dataset."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return concatenate_datasets([load_extxyz(p, dtype=dtype) for p in paths])


def load_isolated_species(path, dtype: torch.dtype = torch.float32) -> Batch:
    """Load the isolated-species anchor systems (one ragged :class:`Batch`).

    ``path`` is the extxyz written by ``scripts/isolated_species.py``: single atoms,
    ions, and small fragments with ``energy=`` and ``charge=`` headers (energies in
    Hartree, *absolute* -- the model adds its own E0 baseline). No forces are stored
    (zeros are filled in; the anchor loss is energy-only).
    """
    from ase.io import iread

    pos_list, num_list, counts, energy_list, charge_list = [], [], [], [], []
    for atoms in iread(str(path), index=":"):
        pos_list.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        num_list.append(np.asarray(atoms.numbers, dtype=np.int64))
        counts.append(len(atoms))
        # ASE moves the special "energy" header key onto a SinglePointCalculator.
        if "energy" in atoms.info:
            energy_list.append(float(atoms.info["energy"]))
        else:
            energy_list.append(float(atoms.get_potential_energy()))
        charge_list.append(float(atoms.info.get("charge", 0.0)))

    positions = torch.tensor(np.concatenate(pos_list), dtype=dtype)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long)
    )
    return Batch(
        positions=positions,
        atomic_numbers=torch.tensor(np.concatenate(num_list), dtype=torch.long),
        batch_idx=batch_idx,
        n_systems=len(counts),
        energy=torch.tensor(energy_list, dtype=dtype),
        forces=torch.zeros_like(positions),
        total_charge=torch.tensor(charge_list, dtype=dtype),
    )


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
