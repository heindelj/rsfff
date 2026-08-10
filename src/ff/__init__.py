"""Physical force-field terms: the rigid backbone the networks correct.

Each term here is an explicit, analytic function of positions and a small set of
per-atom parameters. The parameters come from a learned parameterizer, but the
*functional form* does not -- that is what keeps the mid-range physics from being
absorbed into a neural network (``docs/range_separated_mlip.md`` §7, "gauge leakage").
"""

from .damping import fermi_switch, tang_toennies
from .dispersion import (
    DEFAULT_B_PRIOR,
    DEFAULT_C6_PRIOR,
    DispersionOutput,
    DispersionParameterHeads,
    TTDispersion,
    build_log_priors,
    tt_damped_c6_energy,
)
from .pairs import inter_fragment_pairs
from .units import BOHR_ANG, KJMOL_PER_HARTREE

__all__ = [
    "fermi_switch",
    "tang_toennies",
    "inter_fragment_pairs",
    "tt_damped_c6_energy",
    "DispersionParameterHeads",
    "DispersionOutput",
    "TTDispersion",
    "build_log_priors",
    "DEFAULT_C6_PRIOR",
    "DEFAULT_B_PRIOR",
    "BOHR_ANG",
    "KJMOL_PER_HARTREE",
]
