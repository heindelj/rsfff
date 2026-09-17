"""Active learning for the range-separated force-field functional, on top of ``easyal``.

    build      packmol packs (H2O)n into a spherical cavity        -> structures.extxyz
    optimize   relax every packing on the model's own surface      -> samples.extxyz
    dynamics   Langevin NVT from each minimum, inside a soft wall  -> samples.extxyz
    label      Q-Chem EDA + force on the selected frames           -> labeled.extxyz
    train      refit the film model on everything labeled so far   -> model
    assess     measure the model on held-out structures            -> metrics.json

The first three are implemented here; ``label``, ``train`` and ``assess`` are still
placeholders (see ``run_water.py``), which is why a run stops ``pending`` after ``dynamics``.

``easyal`` gives every stage its own directory, contract check, input/output hashes, params
and metrics under ``<root>/iter_NNN/<stage>/``, so the provenance of a structure is a path:
which packing it came from, which model minimized it, which trajectory and which step.
"""

from .build import PackmolWaterClusters, cavity_radius
from .model import FilmCalculator, LoadedFilmModel
from .sampling import CARRIED, SAMPLED, STRUCTURE, DynamicsSample, MinimizeSample

__all__ = [
    "PackmolWaterClusters", "cavity_radius",
    "MinimizeSample", "DynamicsSample",
    "LoadedFilmModel", "FilmCalculator",
    "STRUCTURE", "SAMPLED", "CARRIED",
]
