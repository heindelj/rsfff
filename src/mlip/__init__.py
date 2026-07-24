"""Machine-learned interatomic potential (MLIP) models and heads."""

from .eem import (
    atomic_dipoles,
    charge_flow_polarizability,
    eem_charges,
    eem_energy,
)
from .diabats import (
    DiabaticStateLibrary,
    DissociationChannel,
    FragmentAssignment,
    FragmentState,
    assign_from_headers,
    enumerate_diabats,
    finest_common_refinement,
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
from .mixture import EnvelopeConfig, MixtureModel, MixtureOutput, mixture_channel_graph
from .switch import pairwise_switch, smoothstep, validity_bump
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
    "DissociationChannel",
    "FragmentState",
    "FragmentAssignment",
    "assign_from_headers",
    "enumerate_diabats",
    "finest_common_refinement",
    # diabatic mixture (Phase 2)
    "MixtureModel",
    "MixtureOutput",
    "EnvelopeConfig",
    "mixture_channel_graph",
    "smoothstep",
    "pairwise_switch",
    "validity_bump",
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
