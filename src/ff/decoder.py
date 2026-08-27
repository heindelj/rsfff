"""The shared parameter decoder: descriptors and fragment state in, physics out.

``docs/fff_v2.md`` v4. Every parameter the force field evaluates is emitted here, from an
atom's two-slot lambda-SOAP description, its fragment's state block and its element. There is
exactly one of these for the whole model, and that is the load-bearing property this design
kept from v3.

**One decoder, one frame.** The v2 model gave every composition its own copy of every head, so
an ``H3O+`` oxygen and an ``H2O`` oxygen were parameterized by networks that had never been
required to agree about anything. Nothing then made their outputs commensurable, and a
crossover between them was an average of two unrelated answers -- measured, 162 kJ/mol of
excursion beyond the interval spanned by the two vertices. With one decoder there is one
answer, and what distinguishes the two oxygens is their *input*: a different descriptor and a
different fragment state, both of which are continuous and both of which a mixture can move.

Why the heads are unchanged classes
-----------------------------------
:class:`~rsfff.ff.response.ElectrostaticParameterHeads`,
:class:`~rsfff.ff.dispersion.DispersionParameterHeads`,
:class:`~rsfff.ff.pauli.PauliMultipoleHeads` and
:class:`~rsfff.ff.bond_energy.FragmentBondEnergy` already consume
``(inv_feats, vec_feats, equiv_feats, species_idx)`` with the two-slot split in their first
layer. So they are reused verbatim: every physical form, prior, positivity constraint and
initialization survives, which is most of what makes this a re-plumbing rather than a new
model.

The two evaluations, and where the isolated one comes from
----------------------------------------------------------
``theta = D(h, eta, k)`` and ``theta_0 = D(h, 0, k)``. Nothing here has to know which is
which: :class:`rsfff.mlip.heads.TwoSlotLinear` reads a *narrow* input as "drop the environment
term", so handing it :meth:`rsfff.ff.slots.SlotFeatures.isolated` gives the isolated
evaluation with no flag, no zero-padding and no convention for a caller to remember.

``L_env`` acts on ``theta - theta_0`` per quantity, never on a feature-space norm -- §4
rejected the latter as having no physical interpretation and hence no defensible weight.
Decoding twice is one extra pass and keeps the per-quantity readout (``env_c6``,
``env_b_disp``, ``env_pauli_multipole``, ``env_e_bond``) that is the number this design exists
to produce.

What does *not* read the description
------------------------------------
``r0``, ``b`` and ``Z`` are per-element tables.

``r0`` is the one that changed, and for a measured reason: in v2 it was per *expert*, and the
proton-transfer scan showed ``r0_elst`` on an oxygen jumping 0.905 -> 1.13 Angstrom as the
expert swapped, which moves the Fermi gate a long way and fed the crossover excursion
directly. Global and element-keyed, it is fragmentation-invariant by construction and the gate
does not move when the description does. **Do not let the fragment state back into it.**
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bond_energy import FragmentBondEnergy
from .dispersion import DispersionParameterHeads
from .pauli import PauliMultipoleHeads
from .range_heads import RangeSeparationHeads
from .response import FragmentResponse

__all__ = ["ParameterDecoder"]


class ParameterDecoder(nn.Module):
    """Every parameter head, at feature widths, shared by every fragment in the model.

    A container: the orchestration -- which evaluation is isolated, which pairs route where,
    when the coupled solve runs -- belongs to the model that assembles the energy. What this
    owns is the weights, and owning them *together and only once* is the whole point.
    """

    def __init__(
        self,
        *,
        response: FragmentResponse,
        disp_params: DispersionParameterHeads,
        pauli_params: PauliMultipoleHeads,
        range_heads: RangeSeparationHeads,
        bond: FragmentBondEnergy,
    ) -> None:
        super().__init__()
        self.response = response
        self.disp_params = disp_params
        self.pauli_params = pauli_params
        self.range_heads = range_heads
        self.bond = bond

    # -- the pieces the model assembles ------------------------------------------------------

    def dispersion(self, feats, species_idx: torch.Tensor):
        """``(C6 (N,), b_disp (N,))``."""
        return self.disp_params(feats.inv_feats, species_idx)

    def pauli(self, feats, species_idx: torch.Tensor):
        """``(q (N,), b (N,), mu (N,3)|None, quad_s (N,5)|None)``."""
        return self.pauli_params(
            feats.inv_feats, species_idx, feats.vec_feats, feats.equiv_feats
        )

    def r0(self, species_idx: torch.Tensor):
        """``({channel: r0 (N,)}, {channel: alpha ()})`` -- **element only**, no description.

        No descriptor is passed and there is nowhere for one to enter: ``r0`` is
        fragmentation-invariant by construction, which is the property the v2 scan showed was
        missing. ``inv_feats`` is still a required argument of the head, so a zero-width tensor
        of the right length is handed in; the head ignores it when ``r0_mlp is None``.
        """
        zeros = species_idx.new_zeros((species_idx.shape[0], 0), dtype=torch.get_default_dtype())
        return self.range_heads(zeros, species_idx)

    def bond_energy(self, feats, species_idx, q, mu, quad_s, env):
        """``(N,)`` the per-atom energy of the electronic state."""
        return self.bond(
            feats.inv_feats, species_idx, feats.vec_feats, feats.equiv_feats,
            q, mu, quad_s, env,
        )

    def extra_repr(self) -> str:
        return "one instance, shared by every fragment"
