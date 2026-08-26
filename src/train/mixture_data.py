"""Loading competing fragmentations as mixtures rather than as separate frames.

:func:`rsfff.train.data.load_cluster_datasets` explodes a multi-fragmentation frame into one
training frame per decomposition, because for the model of §5-§7 the partition is an *input*
and two decompositions of one geometry are two different questions. The mediator (§8) asks a
third question -- how much of each is right -- and that needs all of them at once, in **one
shared atom order**.

Why not read the exploded dataset back
--------------------------------------
Each exploded frame is re-sorted so its own ``fragment_idx`` is non-decreasing
(:func:`rsfff.train.data._sort_by_fragment`), which the sorted-input fast paths in
:mod:`rsfff.ff.pairs` require. Two decompositions that disagree about which atoms are grouped
therefore arrive in *different* atom orders, and nothing in the dataset records the permutation
back. Rather than thread a ``source_index`` column through ``MoleculeDataset`` -- which every
single-fragmentation consumer would then carry for nothing -- this reads the file through
:func:`rsfff.qcgen.multifrag.read_multifrag_extxyz`, whose whole purpose is handing back every
decomposition in the file's own canonical order. ``_select_fragmentation``'s docstring already
points here.

What is refused, and why loudly
-------------------------------
A frame is refused when some decomposition is more than **one atom's move** away from the
reference: that is a genuinely *concerted* rearrangement, which §8 mediates one atom at a time
and §10 defers, so treating it as a single hop would answer a different question plausibly and
wrongly.

That test is deliberately **pairwise-against-the-reference**, not on the size of the union
``D``. §8 asserts ``|D| = 1`` for every frame in this corpus. That is true of any *two*
decompositions and false across three: an ``OH-(H2O)2`` frame carries three, each one hop from
the reference but each moving a *different* hydrogen, so ``|D| = 2`` while nothing concerted
has happened. Rejecting on ``|D| > 1`` throws away 199 of the 399 contested frames -- every
``w2_oh-`` geometry -- which is how this was found.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ff.mediator import align_fragments, contested_atoms
from ..ff.mixture_model import MixtureGroup

__all__ = ["MixtureDataset", "load_mixture_groups"]

#: The EDA channels whose sum is the induction magnitude the mediator's prior ranks by.
#: ``argmin |E_pol + E_ct|`` picks the chemically obvious assignment in 398 of 399 frames.
INDUCTION_COMPONENTS = ("pol", "ct")


@dataclass
class MixtureDataset:
    """Every mediable geometry of a corpus, plus what the frames it came from could not use.

    ``skipped`` is not diagnostics decoration: a corpus that silently lost most of its
    contested frames to the concerted-rearrangement check would train a mediator on almost
    nothing and report a perfectly healthy loss while doing it.
    """

    groups: list[MixtureGroup]
    n_frames: int
    n_uncontested: int
    n_concerted: int

    def __len__(self) -> int:
        return len(self.groups)

    def summary(self) -> str:
        return (
            f"{len(self.groups)} mediable geometries from {self.n_frames} frames "
            f"({self.n_uncontested} with a single fragmentation, "
            f"{self.n_concerted} concerted and skipped)"
        )


def load_mixture_groups(
    paths, dtype: torch.dtype = torch.float64
) -> MixtureDataset:
    """Read multi-fragmentation files into :class:`rsfff.ff.mixture_model.MixtureGroup`.

    Single-fragmentation files contribute nothing and are skipped rather than raising, so a
    mixed corpus of neutral water clusters and ion clusters needs no special casing at the call
    site -- the water half simply has no competition to mediate.
    """
    from pathlib import Path

    from ..qcgen.multifrag import read_multifrag_extxyz

    if isinstance(paths, (str, Path)):
        paths = [paths]

    groups: list[MixtureGroup] = []
    n_frames = n_uncontested = n_concerted = 0
    # Geometry ids must agree with `rsfff.train.data.load_cluster_datasets`, because the
    # train/val split is computed there and the mediator must not be trained on a geometry
    # whose vertices are held out. That function walks these same paths in this same order and
    # numbers each file's frames from a running base, so reproducing the base is what makes
    # the two numberings the same rather than merely similar.
    base = 0
    for path in paths:
        n_in_file = _frame_count(path)
        file_base, base = base, base + n_in_file
        try:
            frames = read_multifrag_extxyz(path)
        except (KeyError, ValueError):
            # No `n_fragmentations` header: an ordinary single-fragmentation file, which has
            # no competition to mediate. Not an error.
            continue
        for frame_index, frame in enumerate(frames):
            n_frames += 1
            fragments = torch.as_tensor(frame["fragment_idx"], dtype=torch.long)
            if fragments.shape[0] < 2:
                n_uncontested += 1
                continue
            fragments = align_fragments(fragments)
            contested = contested_atoms(fragments)
            if contested.numel() == 0:
                n_uncontested += 1
                continue
            # Concerted means one decomposition moved several atoms at once, not that the
            # union over decompositions is larger than one. See the module docstring.
            aligned_ref = fragments[0]
            hops = (fragments != aligned_ref).sum(dim=1)
            if int(hops.max()) > 1:
                n_concerted += 1
                continue

            n_dec, n_atoms = fragments.shape
            charges = frame["fragment_charges"]
            spins = frame["fragment_multiplicities"]
            atom_charge = torch.zeros(n_dec, n_atoms, dtype=dtype)
            atom_two_s = torch.zeros(n_dec, n_atoms, dtype=dtype)
            for m in range(n_dec):
                # `align_fragments` renumbered the ids, so the per-fragment vectors have to be
                # gathered through the *original* numbering, not the aligned one.
                raw = torch.as_tensor(frame["fragment_idx"][m], dtype=torch.long)
                q = torch.as_tensor(charges[m], dtype=dtype)
                # Q-Chem writes multiplicity 2S+1; the model's fragment state block wants 2S.
                s = torch.as_tensor(spins[m], dtype=dtype) - 1.0
                atom_charge[m] = q[raw]
                atom_two_s[m] = s[raw]

            eda = frame.get("eda") or {}
            label = None
            if all(k in eda for k in INDUCTION_COMPONENTS):
                label = torch.as_tensor(
                    sum(eda[k] for k in INDUCTION_COMPONENTS), dtype=dtype
                ).abs()

            groups.append(
                MixtureGroup(
                    positions=torch.as_tensor(frame["positions"], dtype=dtype),
                    atomic_numbers=torch.as_tensor(
                        [_Z[s] for s in frame["symbols"]], dtype=torch.long
                    ),
                    fragments=fragments,
                    atom_charge=atom_charge,
                    atom_two_s=atom_two_s,
                    contested=contested,
                    energy=torch.as_tensor(frame["energy"], dtype=dtype),
                    # Stored in Ha/bohr, like every other force label in the corpus; the
                    # model works in Ha/Angstrom.
                    forces=torch.as_tensor(frame["forces"], dtype=dtype) / _BOHR,
                    vertex_induction_label=label,
                    group_id=file_base + frame_index,
                )
            )
    return MixtureDataset(
        groups=groups,
        n_frames=n_frames,
        n_uncontested=n_uncontested,
        n_concerted=n_concerted,
    )


def _frame_count(path) -> int:
    """How many frames a file holds, without materializing their labels."""
    from ase.io import iread

    return sum(1 for _ in iread(str(path), index=":"))


_BOHR = 0.52917721067

_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}
