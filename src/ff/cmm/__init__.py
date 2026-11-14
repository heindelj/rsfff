"""cmm: the polarizable charge-transfer force field with one learned environment latent (the converged complete model).

See :mod:`rsfff.ff.cmm.model` for the functional form and :class:`rsfff.ff.cmm.CmmConfig` for the architecture choices.
"""
from .config import CmmConfig  # noqa: F401
from .encoder import READOUTS, Encoder  # noqa: F401
from .model import CmmModel  # noqa: F401
from .legacy import convert_state, load_legacy  # noqa: F401
