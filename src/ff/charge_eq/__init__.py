"""Charge equilibration solvers: EEM (atomic electronegativity equalization) and SQE (split-charge equilibration
over bond / channel graphs). Moved from the retired ``rsfff.mlip`` package; the solvers are the part that stays."""
from .eem import *  # noqa: F401,F403
from .sqe import *  # noqa: F401,F403
