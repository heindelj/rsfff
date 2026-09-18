"""A committee of film models, and the disagreement the loop selects and stops on.

A single model cannot say where it is wrong. Several fits of the same architecture on the same
data, differing in their initialization, can: where the data pins the surface they agree, and
where it does not they diverge. That disagreement is the whole basis of the selection in
:class:`sampling.SelectByCommittee` and of the convergence test in ``assess_stage``.

    from committee import Committee
    c = Committee.load("runs/water/iter_000/train/committee")
    scores = c.spread(frames)         # one row per frame
    scores[0].sigma_forces            # Hartree/Angstrom, the largest per-atom disagreement

Two numbers per frame, because they fail differently. ``sigma_energy`` is the standard
deviation across members of the total energy **per atom** -- per atom so that a 25-water
cluster is not automatically more uncertain than a dimer. ``sigma_forces`` is DeePMD's
``model_devi_f``: per atom, the root-mean-square distance of the members' force vectors from
their mean, then the largest over atoms. An energy can agree by cancellation while the forces
that shape a trajectory do not, so the force number is the one that usually moves first.

Iteration 0 has no committee -- there is one starting checkpoint -- and a one-member committee
has no spread at all. :meth:`Committee.load` accepts that case (a plain ``.pt`` path) and
reports ``n_members == 1``; the selection stage checks for it and falls back to spreading its
choice over the pool rather than pretending to a preference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model import LoadedFilmModel, fragment_or_none

__all__ = ["MANIFEST", "Committee", "Spread", "sampling_checkpoint"]

#: Written by the train stage into the committee directory.
MANIFEST = "committee.json"


def sampling_checkpoint(path) -> Path:
    """One checkpoint to drive dynamics with, given a committee directory or a checkpoint.

    Minimizing and running MD on the committee *mean* would cost one evaluation per member for
    a surface no member actually has, so a trajectory is driven by a single member -- the
    first. The committee's job is to judge the frames afterwards, not to produce them, and any
    member is a fit of the same data.
    """
    path = Path(path)
    if not path.is_dir():
        return path
    manifest = path / MANIFEST
    if manifest.exists():
        return Path(json.loads(manifest.read_text())["members"][0]["checkpoint"])
    members = sorted(path.glob("member_*/*/best.pt")) or sorted(path.glob("member_*/best.pt"))
    if not members:
        raise FileNotFoundError(f"no committee members under {path}")
    return members[0]


@dataclass(frozen=True)
class Spread:
    """One frame's disagreement, and the mean prediction it is a spread about."""

    sigma_energy: float      # Hartree per atom
    sigma_forces: float      # Hartree/Angstrom, largest per-atom deviation
    mean_energy: float       # Hartree, total
    n_members: int

    def to_info(self) -> dict:
        return {
            "committee_sigma_energy": round(self.sigma_energy, 12),
            "committee_sigma_forces": round(self.sigma_forces, 12),
            "committee_mean_energy": round(self.mean_energy, 12),
            "committee_members": self.n_members,
        }


class Committee:
    """Several film checkpoints, evaluated together."""

    def __init__(self, members: list[LoadedFilmModel], directory: Path | None = None):
        if not members:
            raise ValueError("a committee needs at least one member")
        self.members = members
        self.directory = directory

    # --- loading --------------------------------------------------------------------------

    @classmethod
    def load(cls, path, *, device: str = "cpu") -> "Committee":
        """A committee directory, or a single checkpoint (iteration 0's starting model)."""
        path = Path(path)
        if path.is_dir():
            manifest = path / MANIFEST
            if manifest.exists():
                entries = json.loads(manifest.read_text())["members"]
                paths = [Path(e["checkpoint"]) for e in entries]
            else:  # a directory of member_NN/ without a manifest: take what is there
                paths = sorted(path.glob("member_*/best.pt"))
            if not paths:
                raise FileNotFoundError(f"no committee members under {path}")
        else:
            paths = [path]
        return cls([LoadedFilmModel(p, device=device) for p in paths], directory=path)

    @property
    def n_members(self) -> int:
        return len(self.members)

    def describe(self) -> dict:
        return {"n_members": self.n_members,
                "directory": str(self.directory) if self.directory else None,
                "checkpoints": [str(m.path) for m in self.members]}

    # --- prediction -----------------------------------------------------------------------

    def predict(self, species, positions) -> tuple[np.ndarray, np.ndarray]:
        """``(energies (K,), forces (K, N, 3))`` for one structure, in Hartree and Hartree/A."""
        positions = np.asarray(positions, dtype=float)
        energies, forces = [], []
        for member in self.members:
            potential, _ = member.potential(species, positions)
            energy, gradient = potential.energy_and_gradient(positions)
            energies.append(float(energy[0]))
            forces.append(-gradient[0])
        return np.array(energies), np.array(forces)

    def spread(self, frame: dict) -> Spread:
        """The disagreement on one easyal frame. Raises if it is not intact water."""
        species = list(frame["arrays"]["species"])
        positions = np.asarray(frame["arrays"]["pos"], dtype=float)
        if fragment_or_none(species, positions) is None:
            raise ValueError("frame is not a set of intact waters")
        energies, forces = self.predict(species, positions)
        n_atoms = positions.shape[0]
        if self.n_members == 1:
            return Spread(0.0, 0.0, float(energies[0]), 1)
        # ddof=1: these are a sample of the fits the data admits, not the population
        sigma_energy = float(np.std(energies, ddof=1)) / n_atoms
        deviation = forces - forces.mean(axis=0, keepdims=True)          # (K, N, 3)
        per_atom = np.sqrt((deviation ** 2).sum(axis=-1).mean(axis=0))   # (N,)
        return Spread(sigma_energy, float(per_atom.max()), float(energies.mean()),
                      self.n_members)

    def spreads(self, frames, *, on_error: str = "raise") -> list[Spread | None]:
        """:meth:`spread` over many frames; ``on_error="skip"`` returns ``None`` for a bad one."""
        out: list[Spread | None] = []
        for frame in frames:
            try:
                out.append(self.spread(frame))
            except Exception:
                if on_error != "skip":
                    raise
                out.append(None)
        return out
