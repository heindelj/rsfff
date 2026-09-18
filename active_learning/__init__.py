"""Active learning for the range-separated force-field functional, on top of ``easyal``.

    build      packmol packs (H2O)n into a spherical cavity        -> structures.extxyz
    optimize   relax every packing on the model's own surface      -> samples.extxyz
    dynamics   Langevin NVT from each minimum, inside a soft wall  -> samples.extxyz
    label      Q-Chem EDA + force from the round-trip job pool     -> labeled.extxyz
    train      refit the film model on everything labeled so far   -> model          (TODO)
    assess     measure the new model on its holdout                -> metrics.json   (TODO)

``workflows.py`` wires them together and is the entry point; ``common.py`` is how the stages
reach the Q-Chem job pool in ``qchem_roundtrip/``.

``easyal`` gives every stage its own directory, contract check, input/output hashes, params
and metrics under ``<root>/iter_NNN/<stage>/``, so the provenance of a training frame is a
path: which packing it came from, which model minimized it, which trajectory step it is, and
which pair of Q-Chem jobs labeled it.
"""

import sys as _sys
from pathlib import Path as _Path

# The modules here import each other by plain name (``from common import ...``), because this
# directory is also run as a script directory -- ``python workflows.py water --root ...`` --
# the way the HBQ project's job_runner/active_learning is. Putting it on sys.path lets the
# same files serve both, at the cost of the modules being importable under two names.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from build import PackmolWaterClusters, cavity_radius  # noqa: E402
from label_stage import QCHEM_TRAINING, QChemLabel  # noqa: E402
from model import FilmCalculator, LoadedFilmModel  # noqa: E402
from sampling import CARRIED, SAMPLED, STRUCTURE, DynamicsSample, MinimizeSample  # noqa: E402

__all__ = [
    "PackmolWaterClusters", "cavity_radius",
    "MinimizeSample", "DynamicsSample",
    "QChemLabel", "QCHEM_TRAINING",
    "LoadedFilmModel", "FilmCalculator",
    "STRUCTURE", "SAMPLED", "CARRIED",
]
