"""Machine-learned interatomic potential (MLIP) models and heads."""

from .charge_embedding import AtomWeightNet, ChargeEmbedding, ElementChargeEnergy
from .charge_model import ChargeAwareModel, ChargeAwareOutput, build_charge_model
from .charge_scf import RaggedIndex, ScfInfo, attach_charges, solve_charges
from .heads import AtomicEnergyHead, ChargeEnergyHead, mlp
from .model import EnergyModel, build_model
from .response_heads import (
    AtomicAlphaHead,
    AtomicVectorHead,
    min_eig_sym3x3,
    symmetric_matrix_to_voigt_vector,
    voigt_vector_to_symmetric_matrix,
)

__all__ = [
    "AtomicEnergyHead",
    "ChargeEnergyHead",
    "mlp",
    "EnergyModel",
    "build_model",
    # charge-aware model
    "ChargeAwareModel",
    "ChargeAwareOutput",
    "build_charge_model",
    "ChargeEmbedding",
    "AtomWeightNet",
    "ElementChargeEnergy",
    "RaggedIndex",
    "ScfInfo",
    "solve_charges",
    "attach_charges",
    # response heads
    "AtomicAlphaHead",
    "AtomicVectorHead",
    "min_eig_sym3x3",
    "voigt_vector_to_symmetric_matrix",
    "symmetric_matrix_to_voigt_vector",
]
