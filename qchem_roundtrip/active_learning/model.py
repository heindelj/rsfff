"""The trained film model, wrapped for the two sampling stages.

One loaded model, many structures
---------------------------------
``FilmPotential`` is built per *topology* -- it holds the species and the fragmentation, so the
caller passes coordinates alone -- while the network itself is the expensive thing to load.
:class:`LoadedFilmModel` keeps the network and hands out a potential per frame.

Fragmentation
-------------
``water_fragment_index`` groups each H with its nearest O and raises when that does not give
every O exactly two H. That is the right behaviour for the film model, which takes one fixed
fragmentation, but a sampling stage must not die because one frame out of two hundred came
apart: :func:`fragment_or_none` turns the failure into ``None`` and the stage counts it.

Where this runs
---------------
Nothing here needs the repository: ``load_film_model`` takes the isolated-atom reference
energies out of the checkpoint's own state dict rather than re-reading the JSON the config
names, so a run directory on scratch with an absolute ``--checkpoint`` is enough.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers as ATOMIC_NUMBERS

from rsfff.md.confine import flat_bottom_sphere
from rsfff.md.film_driver import FilmPotential, load_film_model, water_fragment_index

__all__ = ["HARTREE_TO_EV", "LoadedFilmModel", "FilmCalculator", "fragment_or_none",
           "numbers_of"]

HARTREE_TO_EV = 27.211386245988


def numbers_of(species) -> np.ndarray:
    return np.array([ATOMIC_NUMBERS[str(s)] for s in species], dtype=int)


def fragment_or_none(species, positions):
    """``water_fragment_index`` or ``None`` when the frame is not a set of intact waters."""
    try:
        return water_fragment_index(np.asarray(positions, float), numbers_of(species))
    except ValueError:
        return None


class LoadedFilmModel:
    """A film checkpoint loaded once; ``potential(species, positions)`` per structure.

    ``ctx.model`` in the loop is the checkpoint of the iteration (``initial_model`` in
    iteration 0, the ``train`` output afterwards), so a stage builds one of these per run and
    reuses it across every frame.
    """

    def __init__(self, checkpoint, *, device: str = "cpu"):
        self.path = Path(checkpoint).resolve()
        model, config = load_film_model(self.path, device=device)
        self.model = model
        self.config = config
        self.device = device

    def describe(self) -> dict:
        return {"checkpoint": str(self.path), "dtype": str(self.config.dtype),
                "device": self.device,
                "n_parameters": sum(p.numel() for p in self.model.parameters())}

    def potential(self, species, positions, *, with_induction: bool = True):
        """``(FilmPotential, fragment_idx)``; raises when the frame is not intact water."""
        z = numbers_of(species)
        frag = water_fragment_index(np.asarray(positions, float), z)
        return FilmPotential(self.model, z, frag, with_induction=with_induction), frag


class FilmCalculator(Calculator):
    """ASE calculator over one :class:`FilmPotential`, with the flat-bottom wall folded in.

    The wall (``rsfff.md.confine.flat_bottom_sphere``) is added here rather than inside the
    model: it is a restraint on the sampling, not part of the potential being learned, and
    keeping it separate means the energy the stage records for a frame can exclude it. Both
    energies are reported -- ``energy`` is what the integrator sees (model + wall) and
    ``results["model_energy_hartree"]`` is the model alone, which is the number worth storing.

    The fragmentation is fixed at construction, as the film model requires; a trajectory that
    breaks a molecule is detected afterwards, not by re-fragmenting mid-flight.
    """

    implemented_properties = ("energy", "forces", "free_energy")

    def __init__(self, potential: FilmPotential, *, wall_radius: float | None = None,
                 wall_k: float = 0.0, h_slack: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.potential = potential
        self.wall_radius = wall_radius
        self.wall_k = float(wall_k)
        self.h_slack = float(h_slack)
        self._z = torch.as_tensor(potential.atomic_numbers, dtype=torch.long)

    @property
    def has_wall(self) -> bool:
        return self.wall_radius is not None and self.wall_k != 0.0

    def _wall(self, positions: np.ndarray):
        """``(energy, gradient)`` of the wall in Hartree and Hartree/Angstrom."""
        if not self.has_wall:
            return 0.0, np.zeros_like(positions)
        pos = torch.as_tensor(positions, dtype=torch.get_default_dtype()).requires_grad_(True)
        energy = flat_bottom_sphere(pos, self._z, radius=self.wall_radius, k=self.wall_k,
                                    h_slack=self.h_slack)
        if energy.requires_grad:
            (grad,) = torch.autograd.grad(energy, pos)
        else:  # every atom is inside its shell, so the term is a constant zero
            grad = torch.zeros_like(pos)
        return float(energy.detach()), grad.detach().numpy()

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        energy, gradient = self.potential.energy_and_gradient(positions)
        model_energy = float(energy[0])
        model_gradient = gradient[0]
        wall_energy, wall_gradient = self._wall(positions)
        total = model_energy + wall_energy
        self.results = {
            "energy": total * HARTREE_TO_EV,
            "free_energy": total * HARTREE_TO_EV,
            "forces": -(model_gradient + wall_gradient) * HARTREE_TO_EV,
            "model_energy_hartree": model_energy,
            "wall_energy_hartree": wall_energy,
            "model_max_force_hartree": float(np.abs(model_gradient).max()),
        }
