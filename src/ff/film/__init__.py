"""The state-conditioned, fragment-projected model of ``docs/fff_film.md``.

A parallel model generation built alongside :class:`rsfff.ff.expert_model.FragmentExpertModel`.
It shares the featurizer primitives and every physical energy term with v4; what is new is:

- a first-class atom-to-fragment assignment matrix ``C`` (:class:`StateDescriptor`),
  continuous from day one even though Phase I supplies one-hot columns;
- projection of the shared neighbor density into internal/environment channels *before* the
  nonlinear contractions, plus the bilinear cross block between them
  (:class:`FragmentProjector`);
- a FiLM-conditioned shared parameter trunk (:mod:`rsfff.ff.film.conditioning`);
- physically-parameterized bonded terms -- Morse bonds and cosine angles -- in place of the
  NN bond-energy head (:mod:`rsfff.ff.film.bonded`);
- direct permanent multipoles on the fragment-internal features only, with the polarization
  response solved *around* them (no frozen SQE solve).
"""

from .state import StateDescriptor
from .projector import FragmentProjector, ProjectedFeatures
from .conditioning import ConditionedTrunk, FiLMGenerator, FiLMLayer
from .bonded import (
    BondedParameterHead,
    BondedParameters,
    BondedTopology,
    cosine_angle_energy,
    morse_energy,
)
from .heads import FilmResponseHeads, ResponseFamily
from .permanent import PermanentMultipoleHeads
from .network import ConditionedParameterNetwork, FilmParameters
from .model import DEFAULT_FILM_CLASSICAL, FilmModel, FilmOutput

__all__ = [
    "StateDescriptor",
    "FragmentProjector",
    "ProjectedFeatures",
    "ConditionedTrunk",
    "FiLMGenerator",
    "FiLMLayer",
    "BondedParameterHead",
    "BondedParameters",
    "BondedTopology",
    "cosine_angle_energy",
    "morse_energy",
    "FilmResponseHeads",
    "ResponseFamily",
    "PermanentMultipoleHeads",
    "ConditionedParameterNetwork",
    "FilmParameters",
    "DEFAULT_FILM_CLASSICAL",
    "FilmModel",
    "FilmOutput",
]
