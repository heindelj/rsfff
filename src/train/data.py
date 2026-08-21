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
    forces              : (Ntot, 3) per-atom force target, or None when the reference
                                    calculation produced no nuclear gradient (e.g. the
                                    Q-Chem ALMO-EDA single points)
    total_charge        : (B,)      per-molecule total charge (e), or None
    dipole              : (B, 3)    molecular dipole target (e*Angstrom), or None
    polarizability      : (B, 3, 3) molecular polarizability (e^2*Ang^2/Ha), or None
    dipole_derivatives  : (Ntot, 3, 3) d mu_b / d R_{i,a} (e), or None
    eda                 : {name: (B,)} energy-decomposition components in Hartree, keyed
                                    without the ``eda_`` prefix (``cls_elec``, ``disp``,
                                    ``pol``, ``ct``, ``mod_pauli``, ``int``, ...), or None

    Fragment fields, from either a diabatic state library (``rsfff.mlip.diabats``) or the
    per-atom ``fragment_idx`` column of the extxyz. Indices are batch-global, re-offset by
    :meth:`MoleculeDataset.flat_batch`:

    fragment_idx        : (Ntot,)   long, fragment id per atom in [0, n_fragments)
    fragment_charge     : (F,)      formal charge Q_a of each fragment
    fragment_two_s      : (F,)      2S_a of each fragment
    n_fragments         : number of fragments F in this batch
    fragment_to_batch   : (F,)      long, frame id per fragment in [0, n_systems)

    Isolated-fragment (frozen-monomer) labels, from the per-fragment SCF blocks a Q-Chem
    EDA job prints with ``SCF_PRINT_FRGM = true``. All three are ``None`` unless the file
    carries them:

    fragment_energy        : (F,)      isolated-fragment SCF energy, Hartree
    fragment_dipole        : (F, 3)    e*a0, about the coordinate origin
    fragment_second_moment : (F, 3, 3) e*a0^2, **primitive** (traced), about the
                                       coordinate origin

    The last two are deliberately left unshifted and untraced -- Q-Chem reports them about
    the *supersystem's* center of nuclear charge, and the consumer shifts prediction and
    target through the same algebra so a convention error cancels rather than biasing a
    fit. See :mod:`rsfff.ff.molecular_multipoles`.

    Channel-graph fields, available only from a state library (a distance-independent
    partition cannot supply them):

    bond_index          : (2, Nb)   long, the channel graph in global atom indices
    bond_batch          : (Nb,)     long, frame id per channel in [0, n_systems)
    """

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    batch_idx: torch.Tensor
    n_systems: int
    energy: torch.Tensor
    forces: torch.Tensor | None = None
    total_charge: torch.Tensor | None = None
    dipole: torch.Tensor | None = None
    polarizability: torch.Tensor | None = None
    dipole_derivatives: torch.Tensor | None = None
    eda: dict[str, torch.Tensor] | None = None
    fragment_idx: torch.Tensor | None = None
    fragment_charge: torch.Tensor | None = None
    fragment_two_s: torch.Tensor | None = None
    fragment_to_batch: torch.Tensor | None = None
    fragment_energy: torch.Tensor | None = None
    fragment_dipole: torch.Tensor | None = None
    fragment_second_moment: torch.Tensor | None = None
    n_fragments: int = 0
    bond_index: torch.Tensor | None = None
    bond_batch: torch.Tensor | None = None

    def to(self, device) -> "Batch":
        opt = lambda t: t.to(device) if t is not None else None  # noqa: E731
        return Batch(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            batch_idx=self.batch_idx.to(device),
            n_systems=self.n_systems,
            energy=self.energy.to(device),
            forces=opt(self.forces),
            total_charge=opt(self.total_charge),
            dipole=opt(self.dipole),
            polarizability=opt(self.polarizability),
            dipole_derivatives=opt(self.dipole_derivatives),
            eda=(
                {k: v.to(device) for k, v in self.eda.items()}
                if self.eda is not None
                else None
            ),
            fragment_idx=opt(self.fragment_idx),
            fragment_charge=opt(self.fragment_charge),
            fragment_two_s=opt(self.fragment_two_s),
            fragment_to_batch=opt(self.fragment_to_batch),
            fragment_energy=opt(self.fragment_energy),
            fragment_dipole=opt(self.fragment_dipole),
            fragment_second_moment=opt(self.fragment_second_moment),
            n_fragments=self.n_fragments,
            bond_index=opt(self.bond_index),
            bond_batch=opt(self.bond_batch),
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
        forces: torch.Tensor | None,   # (Ntot_all, 3), or None if unlabeled
        energy: torch.Tensor,      # (n_frames,)
        counts: torch.Tensor,      # (n_frames,) atoms per frame
        *,
        total_charge: torch.Tensor | None = None,        # (n_frames,)
        dipole: torch.Tensor | None = None,              # (n_frames, 3)
        polarizability: torch.Tensor | None = None,      # (n_frames, 3, 3)
        dipole_derivatives: torch.Tensor | None = None,  # (Ntot_all, 3, 3)
        eda: dict[str, torch.Tensor] | None = None,      # {name: (n_frames,)}
        fragment_idx: torch.Tensor | None = None,        # (Ntot_all,) frame-local
        fragment_charge: torch.Tensor | None = None,     # (Nfrag_all,)
        fragment_two_s: torch.Tensor | None = None,      # (Nfrag_all,)
        fragment_energy: torch.Tensor | None = None,     # (Nfrag_all,)
        fragment_dipole: torch.Tensor | None = None,     # (Nfrag_all, 3)
        fragment_second_moment: torch.Tensor | None = None,  # (Nfrag_all, 3, 3)
        fragment_counts: torch.Tensor | None = None,     # (n_frames,) fragments per frame
        bond_index: torch.Tensor | None = None,          # (2, Nbond_all) frame-local
        bond_counts: torch.Tensor | None = None,         # (n_frames,) channels per frame
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
        self._eda = eda

        # Fragment partition and channel graph: stored frame-local and re-offset in
        # flat_batch, exactly like the atom rows. They are *separate* groups -- an extxyz
        # with a `fragment_idx` column gives a partition but no channel graph, which only
        # a diabatic state library can supply.
        self._fragment_idx = fragment_idx
        self._fragment_charge = fragment_charge
        self._fragment_two_s = fragment_two_s
        self._fragment_energy = fragment_energy
        self._fragment_dipole = fragment_dipole
        self._fragment_second_moment = fragment_second_moment
        self._fragment_counts = fragment_counts.long() if fragment_counts is not None else None
        self._bond_index = bond_index
        self._bond_counts = bond_counts.long() if bond_counts is not None else None
        if self._fragment_counts is not None:
            self._frag_offsets = torch.cat(
                (torch.zeros(1, dtype=torch.long), torch.cumsum(self._fragment_counts, 0))
            )
            n_frag = int(self._fragment_counts.sum())
            for name in ("energy", "dipole", "second_moment"):
                value = getattr(self, f"_fragment_{name}")
                if value is not None and value.shape[0] != n_frag:
                    raise ValueError(
                        f"fragment_{name} has {value.shape[0]} rows but the dataset holds "
                        f"{n_frag} fragments"
                    )
        if self._bond_counts is not None:
            self._bond_offsets = torch.cat(
                (torch.zeros(1, dtype=torch.long), torch.cumsum(self._bond_counts, 0))
            )

    @property
    def has_fragments(self) -> bool:
        """A fragment partition is available (from a state library or a column)."""
        return self._fragment_counts is not None

    @property
    def has_channels(self) -> bool:
        """An SQE charge-transfer channel graph is available (state library only)."""
        return self._bond_counts is not None

    @property
    def has_forces(self) -> bool:
        return self._forces is not None

    @property
    def has_diabats(self) -> bool:
        """Both a partition and its channel graph -- what the SQE models require."""
        return self.has_fragments and self.has_channels

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

        # New-batch atom offsets, so frame-local indices become batch-global ones.
        atom_offsets = torch.cumsum(counts, 0) - counts

        fragments: dict = {}
        if self.has_fragments:
            frag_counts = self._fragment_counts[idx]
            frag_offsets = torch.cumsum(frag_counts, 0) - frag_counts
            frag_rows = torch.cat(
                [
                    torch.arange(self._frag_offsets[i], self._frag_offsets[i + 1])
                    for i in idx.tolist()
                ]
            ) if idx.numel() else torch.empty(0, dtype=torch.long)

            fragments = dict(
                fragment_idx=(
                    self._fragment_idx[rows]
                    + torch.repeat_interleave(frag_offsets, counts)
                ),
                fragment_charge=self._fragment_charge[frag_rows],
                fragment_two_s=self._fragment_two_s[frag_rows],
                fragment_to_batch=torch.repeat_interleave(
                    torch.arange(idx.shape[0]), frag_counts
                ),
                n_fragments=int(frag_counts.sum()),
            )
            for name in ("energy", "dipole", "second_moment"):
                value = getattr(self, f"_fragment_{name}")
                if value is not None:
                    fragments[f"fragment_{name}"] = value[frag_rows]

        channels: dict = {}
        if self.has_channels:
            bond_counts = self._bond_counts[idx]
            bond_rows = torch.cat(
                [
                    torch.arange(self._bond_offsets[i], self._bond_offsets[i + 1])
                    for i in idx.tolist()
                ]
            ) if idx.numel() else torch.empty(0, dtype=torch.long)

            channels = dict(
                bond_index=(
                    self._bond_index[:, bond_rows]
                    + torch.repeat_interleave(atom_offsets, bond_counts)
                ),
                bond_batch=torch.repeat_interleave(torch.arange(idx.shape[0]), bond_counts),
            )

        return Batch(
            positions=self._pos[rows].clone(),
            atomic_numbers=self._num[rows],
            batch_idx=batch_idx,
            n_systems=int(idx.shape[0]),
            energy=self._energy[idx],
            forces=self._forces[rows] if self._forces is not None else None,
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
            eda=(
                {k: v[idx] for k, v in self._eda.items()} if self._eda is not None else None
            ),
            **fragments,
            **channels,
        )


def fragment_view(dataset: MoleculeDataset, indices=None) -> MoleculeDataset:
    """Every fragment of every frame, as a frame of its own.

    The w2-w5 files carry ``fragment_energies`` -- **isolated**-fragment SCF energies, from the
    ALMO-EDA that produced the interaction labels -- along with ``fragment_dipoles`` and
    ``fragment_second_moments``. Those are labels for a monomer at the geometry a cluster
    actually visits, and there are roughly 34k of them against the 499 in the dedicated monomer
    set. Exploding the corpus this way turns them into a training stream.

    It is the stream on which ``eta`` is **identically zero**: a lone fragment has no
    cross-fragment edges, so every parameter is evaluated at ``theta_0`` and this data is a
    direct measurement of the isolated-fragment sector. It also runs without a coupled solve and
    with a pair list that has no inter-fragment pairs, so it is cheap.

    **Forces are dropped, deliberately.** The per-atom forces in a cluster file are
    ``-dE/dR`` of the *whole cluster*; the gradient of an isolated fragment's energy is a
    different quantity, and training the one-body term against the cluster gradient would be
    fitting the wrong function while looking like supervision. The dedicated monomer set has
    true one-body forces and keeps them.

    ``eda_*`` goes too -- those are frame-level interaction labels and a single fragment has no
    interactions. ``energy`` is set to the fragment energy: for a one-fragment frame every
    classical channel is an empty sum, so the model's total *is* its fragment energy -- provided
    the coupled solve is off, which is why the isolated streams pass ``with_induction=False``
    (see :meth:`rsfff.ff.expert_model.FragmentExpertModel.forward`). With it on a lone fragment
    still relaxes against its own field, which the label does not know about.

    ``indices`` restricts the source frames. Pass the **training** split: the fragment stream
    and the cluster stream share every weight, so exploding a validation cluster into the
    training stream leaks exactly the quantity the fragment-stream validation number is there
    to measure.
    """
    if not dataset.has_fragments:
        raise ValueError(
            "fragment_view needs a fragment partition; the extxyz needs a `fragment_idx` column"
        )
    if dataset._fragment_energy is None:
        raise ValueError(
            "fragment_view needs `fragment_energies` in the file -- the isolated-fragment SCF "
            "energies. Without them the exploded frames would carry no label at all"
        )

    keep = (
        torch.arange(len(dataset)) if indices is None
        else torch.as_tensor(indices, dtype=torch.long)
    )
    frame_of_atom = torch.repeat_interleave(
        torch.arange(len(dataset)), dataset._counts
    )
    # Frame-local fragment ids -> globally unique ones, the same offsetting `flat_batch` does.
    global_frag = dataset._fragment_idx + dataset._frag_offsets[frame_of_atom]
    # Stable, so atoms keep their within-fragment order; nothing downstream depends on it, but
    # a reproducible layout makes a diff of two datasets meaningful.
    # Restrict to the selected frames before sorting, then renumber the surviving fragments
    # so the ids stay dense.
    atom_keep = torch.isin(frame_of_atom, keep)
    global_frag = global_frag[atom_keep]
    frag_keep = torch.cat(
        [torch.arange(dataset._frag_offsets[i], dataset._frag_offsets[i + 1])
         for i in keep.tolist()]
    ) if keep.numel() else torch.empty(0, dtype=torch.long)
    renumber = torch.full((int(dataset._frag_offsets[-1]),), -1, dtype=torch.long)
    renumber[frag_keep] = torch.arange(frag_keep.shape[0])
    global_frag = renumber[global_frag]

    atom_rows = atom_keep.nonzero().squeeze(-1)
    order = atom_rows[torch.argsort(global_frag, stable=True)]
    counts = torch.bincount(global_frag, minlength=frag_keep.shape[0])
    if frag_keep.numel() and int(counts.min()) == 0:
        raise ValueError(
            "some fragment has no atoms; the `fragment_idx` column and the per-fragment "
            "headers disagree about how many fragments the frame has"
        )

    n_frag = int(frag_keep.shape[0])
    pick = lambda x: None if x is None else x[frag_keep].clone()  # noqa: E731
    energy = dataset._fragment_energy[frag_keep].clone()
    return MoleculeDataset(
        positions=dataset._pos[order].clone(),
        atomic_numbers=dataset._num[order],
        forces=None,
        energy=energy,
        counts=counts,
        fragment_idx=torch.zeros(order.shape[0], dtype=torch.long),
        fragment_charge=(
            pick(dataset._fragment_charge) if dataset._fragment_charge is not None
            else torch.zeros(n_frag)
        ),
        fragment_two_s=(
            pick(dataset._fragment_two_s) if dataset._fragment_two_s is not None
            else torch.zeros(n_frag)
        ),
        fragment_energy=energy.clone(),
        fragment_dipole=pick(dataset._fragment_dipole),
        fragment_second_moment=pick(dataset._fragment_second_moment),
        fragment_counts=torch.ones(n_frag, dtype=torch.long),
    )


def _expand_second_moments(flat: np.ndarray, n_frag: int) -> np.ndarray:
    """``(F*6,)`` unique components in Q-Chem's order -> ``(F, 3, 3)`` symmetric tensors.

    The print order is ``XX XY YY XZ YZ ZZ`` -- note it is *not* the diagonal-first
    Voigt order used elsewhere in this repo, which is why this is spelled out rather
    than routed through :func:`rsfff.mlip.response_heads.voigt_vector_to_symmetric_matrix`.
    """
    u = flat.reshape(n_frag, 6)
    xx, xy, yy, xz, yz, zz = (u[:, i] for i in range(6))
    return np.stack(
        (
            np.stack((xx, xy, xz), axis=-1),
            np.stack((xy, yy, yz), axis=-1),
            np.stack((xz, yz, zz), axis=-1),
        ),
        axis=-2,
    )


def _select_fragmentation(atoms, fragmentation: int, path) -> None:
    """Collapse a multi-fragmentation frame in place to one of its fragmentations.

    Files written by ``scripts/parse_aimd_eda.py`` carry several decompositions of
    the same geometry (see :mod:`rsfff.qcgen.multifrag`): an ``n_fragmentations``
    header, one ``fragment_idx``/``fragment_idx_k`` column per decomposition, and
    per-decomposition vectors for every ``eda_*`` and ``fragment_*`` quantity.
    Rewriting the frame into the ordinary single-fragmentation form here means the
    rest of the loader -- and every model and loss downstream of it -- needs no
    special case.

    A model that *mixes* over the decompositions wants all of them at once and
    should read the file with :func:`rsfff.qcgen.multifrag.read_multifrag_extxyz`
    instead; this function deliberately throws the alternatives away.
    """
    info = atoms.info
    n_sets = info.get("n_fragmentations")
    if n_sets is None:
        if fragmentation:
            raise ValueError(
                f"{path}: fragmentation={fragmentation} was requested but this file has "
                f"one fragmentation per frame (no `n_fragmentations` header)"
            )
        return
    n_sets = int(n_sets)
    k = int(fragmentation)
    if not 0 <= k < n_sets:
        raise ValueError(
            f"{path}: fragmentation={k} out of range; the frame has {n_sets}"
        )

    counts = np.asarray(info["n_fragments"], dtype=int).reshape(n_sets)
    lo = int(counts[:k].sum())
    hi = lo + int(counts[k])

    column = "fragment_idx" if k == 0 else f"fragment_idx_{k}"
    atoms.arrays["fragment_idx"] = np.asarray(atoms.arrays[column], dtype=np.int64)
    for j in range(1, n_sets):
        atoms.arrays.pop(f"fragment_idx_{j}", None)

    for key in ("fragment_charges", "fragment_multiplicities", "fragment_energies"):
        info[key] = np.asarray(info[key], dtype=np.float64)[lo:hi]
    for key, width in (("fragment_dipoles", 3), ("fragment_second_moments", 6)):
        flat = np.asarray(info[key], dtype=np.float64).reshape(-1, width)
        info[key] = flat[lo:hi].ravel()
    if "fragment_mulliken" in info:
        info["fragment_mulliken"] = np.asarray(
            info["fragment_mulliken"], dtype=np.float64
        ).reshape(n_sets, len(atoms))[k]

    for key in list(info):
        if key.startswith("eda_"):
            info[key] = float(np.asarray(info[key], dtype=np.float64).reshape(n_sets)[k])
    for key in (
        "fragmentation_ranks",
        "fragmentation_charge_fragment",
        "fragmentation_excess_distance",
    ):
        if key in info:
            info[key.replace("fragmentation_", "fragmentation_selected_")] = np.asarray(
                info[key], dtype=np.float64
            ).reshape(n_sets)[k]
    types = " ".join(
        str(t) for t in np.atleast_1d(info.get("fragmentation_config_types", ""))
    ).split()
    if types:
        info["fragmentation_config_type"] = types[k]
    info["n_fragments"] = int(counts[k])
    info["selected_fragmentation"] = k


def load_extxyz(
    path,
    dtype: torch.dtype = torch.float32,
    library=None,
    fragmentation: int = 0,
) -> MoleculeDataset:
    """Read every frame of an extended-XYZ file into a :class:`MoleculeDataset`.

    ASE attaches ``energy=`` and the per-atom ``forces`` to a ``SinglePointCalculator``,
    so they are read via ``get_potential_energy()`` / ``get_forces()`` (not ``info`` /
    ``arrays``). Positions are Angstrom and energies Hartree, as stored. The dataset's
    forces are stored in **Hartree/Bohr** (psi4 gradient convention, see
    ``scripts/generate_dataset.py``); we divide by ``ase.units.Bohr`` to convert to
    **Hartree/Angstrom** so they match autograd forces ``-dE/d(positions in Angstrom)``.
    (Verified by finite difference: stored/Bohr == -dE/dx to ~1e-4.)

    Forces are optional: reference calculations that produce no nuclear gradient (the
    Q-Chem ALMO-EDA single points in ``data/eda_data``) yield ``forces=None`` rather than
    a block of zeros, so a mis-wired loss raises instead of quietly training against a
    flat surface. All-or-nothing across frames.

    ``eda_*`` headers are collected into an ``eda`` dict keyed without the prefix
    (``eda_disp`` -> ``"disp"``), in Hartree as stored.

    The fragment partition comes from one of two places, in order of precedence:

    1. ``library``, an optional :class:`rsfff.mlip.diabats.DiabaticStateLibrary`. Each
       frame's ``config_type`` header is resolved against it to obtain the partition,
       formal charges/spins, and the **channel graph** for the SQE solve. The frame is
       validated against the registry (element order, charge, multiplicity) rather than
       coerced to fit -- see ``rsfff.mlip.diabats.assign_from_headers``.
    2. a per-atom ``fragment_idx`` column, with ``fragment_charges`` /
       ``fragment_multiplicities`` headers. This gives a partition but **no channel
       graph** (a distance-independent set of charge-transfer channels cannot be inferred
       from a partition alone), so SQE models still need a library.

    ``fragmentation`` selects which decomposition to read from a *multi*-fragmentation
    file -- the H3O+/OH- microsolvation data, where one geometry carries an ALMO-EDA
    decomposition for every placement of the excess charge. ``0`` is the harvester's
    rank-0 assignment, the one with the smallest total O-H bond-length sum. Everything
    downstream then sees an ordinary single-fragmentation dataset; see
    :func:`_select_fragmentation` for what is discarded, and
    :func:`rsfff.qcgen.multifrag.read_multifrag_extxyz` for reading all of them at once.
    """
    import ase.units
    from ase.io import iread

    bohr = float(ase.units.Bohr)  # Angstrom per bohr
    pos_list, num_list, force_list, energy_list, counts = [], [], [], [], []
    charge_list, dip_list, pol_list, dmu_list = [], [], [], []
    eda_lists: dict[str, list[float]] = {}
    frag_idx_list, frag_q_list, frag_s_list, frag_counts = [], [], [], []
    frag_e_list, frag_dip_list, frag_m2_list = [], [], []
    bond_list, bond_counts = [], []
    for atoms in iread(str(path), index=":"):
        _select_fragmentation(atoms, fragmentation, path)
        n = len(atoms)
        pos_list.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        num_list.append(np.asarray(atoms.numbers, dtype=np.int64))
        try:
            force_list.append(np.asarray(atoms.get_forces(), dtype=np.float64) / bohr)
        except (NotImplementedError, RuntimeError):
            pass  # no gradient in the reference calculation; checked for consistency below
        energy_list.append(float(atoms.get_potential_energy()))
        counts.append(n)

        info = atoms.info
        charge_list.append(float(info.get("charge", 0.0)))
        for key, value in info.items():
            if key.startswith("eda_"):
                eda_lists.setdefault(key[4:], []).append(float(value))
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

        if library is not None:
            from ..mlip.diabats import assign_from_headers

            config_type = info.get("config_type")
            if config_type is None:
                raise ValueError(
                    f"{path}: frame has no `config_type` header, so its diabatic state cannot "
                    f"be resolved; regenerate the labels or load without a state library"
                )
            assignment = assign_from_headers(
                library,
                atoms.get_chemical_symbols(),
                config_type=str(config_type),
                charge=info.get("charge"),
                multiplicity=info.get("multiplicity"),
            )
            frag_idx_list.append(assignment.fragment_idx)
            frag_q_list.append(assignment.fragment_charge)
            frag_s_list.append(assignment.fragment_two_s)
            frag_counts.append(assignment.n_fragments)
            bond_list.append(assignment.bond_index)
            bond_counts.append(assignment.bond_index.shape[1])
        elif "fragment_idx" in atoms.arrays:
            fi = np.asarray(atoms.arrays["fragment_idx"], dtype=np.int64)
            nf = int(fi.max()) + 1 if fi.size else 0
            frag_idx_list.append(fi)
            frag_counts.append(nf)
            q = info.get("fragment_charges")
            m = info.get("fragment_multiplicities")
            frag_q_list.append(
                np.asarray(q, dtype=np.float64).reshape(nf) if q is not None
                else np.zeros(nf)
            )
            frag_s_list.append(
                np.asarray(m, dtype=np.float64).reshape(nf) - 1.0 if m is not None
                else np.zeros(nf)
            )

        # Isolated-fragment labels. Only reachable via the fragment_idx column path --
        # a diabatic library supplies a partition, not reference values for it.
        nf = frag_counts[-1] if frag_counts else 0
        if "fragment_energies" in info:
            frag_e_list.append(
                np.asarray(info["fragment_energies"], dtype=np.float64).reshape(nf)
            )
        if "fragment_dipoles" in info:
            frag_dip_list.append(
                np.asarray(info["fragment_dipoles"], dtype=np.float64).reshape(nf, 3)
            )
        if "fragment_second_moments" in info:
            frag_m2_list.append(
                _expand_second_moments(
                    np.asarray(info["fragment_second_moments"], dtype=np.float64), nf
                )
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
    if len(force_list) not in (0, n_frames):
        raise ValueError(
            f"{path}: forces present on {len(force_list)} of {n_frames} frames; "
            f"labels must be all-or-nothing"
        )
    bad_eda = {k: len(v) for k, v in eda_lists.items() if len(v) != n_frames}
    if bad_eda:
        raise ValueError(
            f"{path}: EDA components present on some frames but not all "
            f"(of {n_frames} frames: {bad_eda})"
        )

    positions = torch.tensor(np.concatenate(pos_list), dtype=dtype)
    atomic_numbers = torch.tensor(np.concatenate(num_list), dtype=torch.long)
    forces = torch.tensor(np.concatenate(force_list), dtype=dtype) if force_list else None
    energy = torch.tensor(energy_list, dtype=dtype)
    counts_t = torch.tensor(counts, dtype=torch.long)

    frag_labels = {
        "fragment_energy": frag_e_list,
        "fragment_dipole": frag_dip_list,
        "fragment_second_moment": frag_m2_list,
    }
    bad_frag = {k: len(v) for k, v in frag_labels.items() if len(v) not in (0, n_frames)}
    if bad_frag:
        raise ValueError(
            f"{path}: isolated-fragment labels present on some frames but not all "
            f"(of {n_frames} frames: {bad_frag})"
        )

    partition: dict = {}
    if frag_counts:
        partition = dict(
            fragment_idx=torch.tensor(np.concatenate(frag_idx_list), dtype=torch.long),
            fragment_charge=torch.tensor(np.concatenate(frag_q_list), dtype=dtype),
            fragment_two_s=torch.tensor(np.concatenate(frag_s_list), dtype=dtype),
            fragment_counts=torch.tensor(frag_counts, dtype=torch.long),
        )
        for name, values in frag_labels.items():
            if values:
                partition[name] = torch.tensor(np.concatenate(values), dtype=dtype)
    if bond_counts:  # channel graph, from a state library only
        partition.update(
            bond_index=torch.tensor(np.concatenate(bond_list, axis=1), dtype=torch.long),
            bond_counts=torch.tensor(bond_counts, dtype=torch.long),
        )

    return MoleculeDataset(
        positions, atomic_numbers, forces, energy, counts_t,
        total_charge=torch.tensor(charge_list, dtype=dtype),
        dipole=torch.tensor(np.stack(dip_list), dtype=dtype) if dip_list else None,
        polarizability=torch.tensor(np.stack(pol_list), dtype=dtype) if pol_list else None,
        dipole_derivatives=(
            torch.tensor(np.concatenate(dmu_list), dtype=dtype) if dmu_list else None
        ),
        eda=(
            {k: torch.tensor(v, dtype=dtype) for k, v in eda_lists.items()}
            if eda_lists else None
        ),
        **partition,
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

    def _cat_optional(name: str, dim: int = 0):
        fields = [getattr(d, name) for d in datasets]
        present = [f is not None for f in fields]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"cannot concatenate: {name} present on some datasets only")
        return torch.cat(fields, dim=dim)

    def _cat_eda():
        dicts = [d._eda for d in datasets]
        present = [e is not None for e in dicts]
        if not any(present):
            return None
        if not all(present):
            raise ValueError("cannot concatenate: eda labels present on some datasets only")
        keys = set(dicts[0])
        for e in dicts[1:]:
            if set(e) != keys:
                raise ValueError(
                    f"cannot concatenate: EDA component sets differ "
                    f"({sorted(keys)} vs {sorted(e)})"
                )
        return {k: torch.cat([e[k] for e in dicts]) for k in dicts[0]}

    return MoleculeDataset(
        torch.cat([d._pos for d in datasets]),
        torch.cat([d._num for d in datasets]),
        _cat_optional("_forces"),
        torch.cat([d._energy for d in datasets]),
        torch.cat([d._counts for d in datasets]),
        total_charge=_cat_optional("_total_charge"),
        dipole=_cat_optional("_dipole"),
        polarizability=_cat_optional("_polarizability"),
        dipole_derivatives=_cat_optional("_dipole_derivatives"),
        eda=_cat_eda(),
        # Frame-local indices, so plain concatenation is correct -- flat_batch re-offsets.
        fragment_idx=_cat_optional("_fragment_idx"),
        fragment_charge=_cat_optional("_fragment_charge"),
        fragment_two_s=_cat_optional("_fragment_two_s"),
        fragment_energy=_cat_optional("_fragment_energy"),
        fragment_dipole=_cat_optional("_fragment_dipole"),
        fragment_second_moment=_cat_optional("_fragment_second_moment"),
        fragment_counts=_cat_optional("_fragment_counts"),
        bond_index=_cat_optional("_bond_index", dim=1),
        bond_counts=_cat_optional("_bond_counts"),
    )


def load_datasets(paths, dtype: torch.dtype = torch.float32, library=None) -> MoleculeDataset:
    """Load one or more extxyz files into a single concatenated dataset."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return concatenate_datasets([load_extxyz(p, dtype=dtype, library=library) for p in paths])


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


def load_atomic_reference_batch(states, neighbor_types) -> Batch:
    """Assemble the isolated-atom reference states into one ragged anchor :class:`Batch`.

    ``states`` is an :class:`rsfff.mlip.reference_states.AtomicStateReference`. Each state
    becomes a one-atom system at the origin, its own fragment, carrying the state's formal
    charge and ``2S`` and **no channels** (a lone atom has nothing to transfer along). The
    model's prediction for these systems is its exact free-atom limit: zero SOAP features, so
    every head reduces to a function of the reference embedding, and ``q = q^(0) = Q``.

    The energy targets ride on ``Batch.energy``; polarizability targets on
    ``Batch.polarizability``. Forces are zero by symmetry for a single atom.
    """
    n = len(states)
    types = [int(t) for t in neighbor_types]
    atomic_numbers = torch.tensor(
        [types[int(i)] for i in states.species_idx.tolist()], dtype=torch.long
    )
    dtype = states.energy.dtype
    idx = torch.arange(n)
    return Batch(
        positions=torch.zeros(n, 3, dtype=dtype),
        atomic_numbers=atomic_numbers,
        batch_idx=idx,
        n_systems=n,
        energy=states.energy.clone(),
        forces=torch.zeros(n, 3, dtype=dtype),
        total_charge=states.charge.clone(),
        polarizability=states.alpha.clone(),
        fragment_idx=idx,
        fragment_charge=states.charge.clone(),
        fragment_two_s=states.two_s.clone(),
        n_fragments=n,
        bond_index=torch.zeros(2, 0, dtype=torch.long),
        bond_batch=torch.zeros(0, dtype=torch.long),
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


def load_monomer_batch(path, dtype: torch.dtype = torch.float32, limit: int | None = None):
    """One ragged :class:`Batch` of every frame in an isolated-monomer extxyz.

    The multipole anchor for the electrostatics fit. These frames carry the
    frozen-monomer dipole and second moment but no ``eda_*`` component, and they carry
    forces where the cluster frames' EDA labels do not -- so they cannot be concatenated
    with the training set (``concatenate_datasets`` refuses partial label presence, and
    correctly). Loading them whole and evaluating them as an anchor is the same pattern
    ``load_atomic_reference_batch`` uses; 500 water monomers is 1500 atoms, small enough
    that batching them adds nothing.

    ``limit`` truncates for smoke tests.
    """
    dataset = load_extxyz(path, dtype=dtype)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    return dataset.flat_batch(range(n))
