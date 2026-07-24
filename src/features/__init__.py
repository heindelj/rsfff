"""GPU-native SOAP / ACE-style equivariant atomic features.

Two interchangeable equivariant backends are provided:

  * ``"e3nn"`` -- the default, pure-``e3nn`` path (CPU/CUDA/MPS).
  * ``"cue"``  -- a ``cuequivariance`` path for the fused nu=3 bispectrum on CUDA
                  (install the optional ``cue`` extra: ``pip install rsfff[cue]``).

The two are a faithful numerical swap for the density and nu=2 power spectrum;
only the learnable bispectrum weights differ. See ``equivariant_backend`` for
details.
"""

from .equivariant_backend import (
    CueBackend,
    E3nnBackend,
    E3nnWeightedBispectrum,
    EquivariantBackend,
    get_backend,
)
from .features import (
    BesselBasis,
    DensityExpansion,
    FlatLambdaSOAPFeaturizer,
    FlatStateSOAPFeaturizer,
    LambdaFeatures,
    SoapAceFeatures,
    bispectrum,
    cosine_cutoff,
    equivariant_power_spectrum,
    power_spectrum,
)

__all__ = [
    # backends
    "EquivariantBackend",
    "E3nnBackend",
    "CueBackend",
    "E3nnWeightedBispectrum",
    "get_backend",
    # featurizers
    "BesselBasis",
    "DensityExpansion",
    "SoapAceFeatures",
    "FlatLambdaSOAPFeaturizer",
    "FlatStateSOAPFeaturizer",
    "LambdaFeatures",
    # functional
    "power_spectrum",
    "bispectrum",
    "equivariant_power_spectrum",
    "cosine_cutoff",
]
