"""Moved to bb: features live in :mod:`bb.features`, backends in :mod:`bb.o3`.

This alias keeps ``import rsfff.features`` working for scripts and notebooks for one
release; new code imports from bb directly.
"""

from bb.features import (  # noqa: F401
    BesselBasis,
    DensityExpansion,
    E3nnWeightedBispectrum,
    FlatLambdaSOAPFeaturizer,
    FlatStateSOAPFeaturizer,
    LambdaFeatures,
    SOAPFeaturizer,
    StateSOAPFeaturizer,
    bispectrum,
    cosine_cutoff,
    equivariant_power_spectrum,
    power_spectrum,
)
from bb.o3 import CueBackend, E3nnBackend, EquivariantBackend, get_backend  # noqa: F401
