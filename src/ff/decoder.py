"""The shared parameter decoder: keys in, physics out. One instance for the whole model.

``docs/fff_v2.md`` v3. Every parameter the force field evaluates is emitted here, from an
atom's key (:mod:`rsfff.ff.keys`) and its element. There is exactly one of these, shared by
every composition, and that is the load-bearing property:

**The shared decoder is what makes two experts' keys commensurable.** Nothing pins the latent
of the ``H3O`` encoder to the latent of the ``H2O`` encoder a priori -- a latent has no units
and no canonical frame. What pins them is that both must decode through *this* module to the
same physical quantities against the same labels, so training forces them into one frame. The
gauge is enforced rather than assumed, and it is precisely why the diabatic mixture stack could
blend features at all while the v2 fragment-expert model could not.

Why the heads are unchanged classes
-----------------------------------
:class:`~rsfff.ff.response.ElectrostaticParameterHeads`,
:class:`~rsfff.ff.dispersion.DispersionParameterHeads`,
:class:`~rsfff.ff.pauli.PauliMultipoleHeads` and
:class:`~rsfff.ff.bond_energy.FragmentBondEnergy` already consume
``(inv_feats, vec_feats, equiv_feats, species_idx)`` -- exactly a :class:`KeyBundle`'s shape.
So they are reused verbatim at *key widths* rather than rewritten. Two consequences worth
stating:

* every physical form, prior, positivity constraint and initialization survives the move
  untouched, which is most of what makes this a re-plumbing rather than a new model;
* ``p_env = 0`` everywhere in here. **The two-slot split now lives entirely in the encoder.**
  That is a simplification, not a loss: the environment still enters through one named,
  zero-initialized set of tensors, ``env_parameters`` still finds it, and the ablation still
  works -- there is now one such set instead of one per head.

``L_env`` stays in parameter space
----------------------------------
The penalty is on ``D(k) - D(k_0)`` per quantity, **never** on ``||k - k_0||``. §4 rejected a
feature-space norm as having no physical interpretation and hence no defensible weight, and a
key-space norm has exactly that problem. Decoding twice is one extra pass and keeps §9's
per-quantity readout (``env_c6``, ``env_b_disp``, ``env_pauli_multipole``, ``env_e_bond``),
which is the number this design exists to produce.

What does *not* read the key
----------------------------
``r0``, ``b`` and ``Z`` are per-element tables with no key dependence at all.

``r0`` is the one that changed, and for a measured reason: in v2 it was per *expert*, and the
proton-transfer scan showed ``r0_elst`` on an oxygen jumping 0.905 -> 1.13 Angstrom as the
expert swapped, which moves the Fermi gate a long way and fed the crossover excursion directly.
Global and element-keyed, it is fragmentation-invariant by construction and the gate does not
move when the description does.

``b`` and ``Z`` were already ``log_prior[s] + d_log[s]`` per expert; here they are simply
global as well.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bond_energy import FragmentBondEnergy
from .dispersion import DispersionParameterHeads
from .keys import KeyBundle
from .pauli import PauliMultipoleHeads
from .range_heads import RangeSeparationHeads
from .response import FragmentResponse

__all__ = ["ParameterDecoder"]


class ParameterDecoder(nn.Module):
    """Every parameter head, at key widths, shared across compositions.

    A container, like :class:`rsfff.ff.expert.FragmentExpert` was: the orchestration -- which
    evaluation is isolated, which pairs route where, when the coupled solve runs -- belongs to
    the model that assembles the energy. What this owns is the weights, and owning them
    *together and only once* is the whole point.
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

    def dispersion(self, key: KeyBundle, species_idx: torch.Tensor):
        """``(C6 (N,), b_disp (N,))``."""
        return self.disp_params(key.k0, species_idx)

    def pauli(self, key: KeyBundle, species_idx: torch.Tensor):
        """``(q (N,), b (N,), mu (N,3)|None, quad_s (N,5)|None)``."""
        return self.pauli_params(key.k0, species_idx, key.k1, key.k2)

    def r0(self, species_idx: torch.Tensor):
        """``({channel: r0 (N,)}, {channel: alpha ()})`` -- **element only**, no key.

        The key is not passed and there is nowhere for it to enter: ``r0`` is
        fragmentation-invariant by construction, which is the property the v2 scan showed was
        missing. ``inv_feats`` is still a required argument of the head, so a zero-width tensor
        of the right length is handed in; the head ignores it when ``r0_mlp is None``.
        """
        zeros = species_idx.new_zeros((species_idx.shape[0], 0), dtype=torch.get_default_dtype())
        return self.range_heads(zeros, species_idx)

    def bond_energy(self, key: KeyBundle, species_idx, q, mu, quad_s, env):
        """``(N,)`` the per-atom energy of the electronic state."""
        return self.bond(key.k0, species_idx, key.k1, key.k2, q, mu, quad_s, env)

    def extra_repr(self) -> str:
        return "shared across all compositions"
