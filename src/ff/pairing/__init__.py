"""The pairing model: bonding as a variational spin-pairing energy with a learned bond order.

``docs/fff_pairing.md`` in code. The film model's assigned bonded topology (Morse + cosine
angle on the fragment's covalent graph) is replaced by

    E_pair = sum_ij [ -J_ij p_ij + 1/2 kappa_ij p_ij^2 ] + entropic barrier,
             sum_j p_ij + u_i = v_i,   0 <= p_ij <= 1,   u_i >= 0

where ``J_ij`` is a Slater-damped multipolar contraction of per-atom *valence* polytensors
(the same operator as the Pauli repulsion), ``kappa`` is a per-atom pairing hardness, ``v_i``
the valence capacity and ``p_ij`` the bond order, solved for variationally. The bond order is
then the fragment co-membership: it replaces the assigned ``p_intra`` everywhere the film
model used one -- the intra/inter split of the classical channels, the induction gate and the
SQE conductances -- so no range-separation function touches a bonded pair.
"""

from .bond_order import (
    BondOrderSolution,
    DEFAULT_VALENCE,
    comembership_from_bond_order,
    pairing_energy,
    solve_bond_order,
    valence_table,
)
from .heads import PairingFamily, PairingHeads
from .network import PairingParameterNetwork, PairingParameters
from .model import PairingModel, PairingOutput

__all__ = [
    "BondOrderSolution",
    "DEFAULT_VALENCE",
    "PairingFamily",
    "PairingHeads",
    "PairingModel",
    "PairingOutput",
    "PairingParameterNetwork",
    "PairingParameters",
    "comembership_from_bond_order",
    "pairing_energy",
    "solve_bond_order",
    "valence_table",
]
