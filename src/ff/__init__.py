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
from .molecular_multipoles import (
    buckingham_from_second_moment,
    center_of_nuclear_charge,
    fragment_multipoles,
    reference_multipoles,
)
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    multipole_pair_energy,
    slater_one_center_damp,
    slater_two_center_damp,
)
from .onebody import OneBodyEnergy, OneBodyModel, OneBodyOutput
from .onebody_elec import OneBodyElecModel, OneBodyElecOutput
from .pairs import inter_fragment_pairs, intra_fragment_pairs, union_pairs
from .response import FragmentResponse, FragmentResponseOutput
from .pauli import (
    DEFAULT_PAULI_DIPOLE_SCALE,
    DEFAULT_PAULI_PRIOR,
    PauliModel,
    PauliMultipoleHeads,
    PauliOutput,
    SlaterPauli,
    build_pauli_priors,
)
from .units import BOHR_ANG, KJMOL_PER_HARTREE

__all__ = [
    "fermi_switch",
    "tang_toennies",
    "inter_fragment_pairs",
    "intra_fragment_pairs",
    "union_pairs",
    "FragmentResponse",
    "FragmentResponseOutput",
    "OneBodyEnergy",
    "OneBodyModel",
    "OneBodyOutput",
    "OneBodyElecModel",
    "OneBodyElecOutput",
    "slater_two_center_damp",
    "slater_one_center_damp",
    "damped_interaction_tensor",
    "multipole_pair_energy",
    "build_polytensor",
    "buckingham_from_second_moment",
    "center_of_nuclear_charge",
    "fragment_multipoles",
    "reference_multipoles",
    "PauliMultipoleHeads",
    "PauliOutput",
    "SlaterPauli",
    "PauliModel",
    "build_pauli_priors",
    "DEFAULT_PAULI_PRIOR",
    "DEFAULT_PAULI_DIPOLE_SCALE",
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
