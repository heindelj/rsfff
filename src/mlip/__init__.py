"""Machine-learned interatomic potential (MLIP) models and heads."""

from .eem import (
    atomic_dipoles,
    charge_flow_polarizability,
    eem_charges,
    eem_energy,
)
from .diabats import (
    DiabaticStateLibrary,
    FragmentAssignment,
    FragmentState,
    assign_from_headers,
)
from .eem_model import EEMOutput, EEMParameterHeads, MLIPEEMModel, build_eem_model
from .heads import AtomicEnergyHead, mlp
from .model import EnergyModel, build_model
from .monomer import (
    MonomerModel,
    MonomerOutput,
    MonomerParameterHeads,
    build_monomer_model,
)
from .reference_states import (
    AtomicStateReference,
    AtomWeightNet,
    ReferenceEmbedding,
    ReferenceStateOutput,
)
from .sqe import (
    PairComplianceHead,
    SQESolution,
    atomic_dipole_energy,
    sqe_solve,
)
from .response_heads import (
    AtomicAlphaHead,
    AtomicVectorHead,
    min_eig_sym3x3,
    symmetric_matrix_to_voigt_vector,
    voigt_vector_to_symmetric_matrix,
)

__all__ = [
    "AtomicEnergyHead",
    "mlp",
    "EnergyModel",
    "build_model",
    # EEM
    "eem_charges",
    "atomic_dipoles",
    "eem_energy",
    "charge_flow_polarizability",
    "EEMParameterHeads",
    "MLIPEEMModel",
    "EEMOutput",
    "build_eem_model",
    # diabatic state library
    "DiabaticStateLibrary",
    "FragmentState",
    "FragmentAssignment",
    "assign_from_headers",
    # reference states / monomer stack
    "ReferenceEmbedding",
    "ReferenceStateOutput",
    "AtomWeightNet",
    "AtomicStateReference",
    "MonomerModel",
    "MonomerOutput",
    "MonomerParameterHeads",
    "build_monomer_model",
    # SQE
    "sqe_solve",
    "SQESolution",
    "PairComplianceHead",
    "atomic_dipole_energy",
    # response heads
    "AtomicAlphaHead",
    "AtomicVectorHead",
    "min_eig_sym3x3",
    "voigt_vector_to_symmetric_matrix",
    "symmetric_matrix_to_voigt_vector",
]
