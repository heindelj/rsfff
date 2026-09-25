"""The Tersoff model: the pairing model with its solve replaced by an explicit bond order.

``docs/fff_tersoff.md``. Relative to :class:`rsfff.ff.pairing.PairingModel` exactly one thing
changes: :meth:`_pairing`. The coupling ``J`` (valence polytensor through the Pauli overlap),
the hardness, the capacity tables, the co-membership ``c`` from ``p`` and every role it plays
(intra/inter split, induction gate, SQE conductances), the two-stage topology -> parameter
network conditioned on ``(q, u)``, the ``theta_0`` / ``theta`` evaluations, the four exact
energy buckets: all inherited. The electronic state is

* formal charges ``q`` from the family's readout projected onto the frame charge
  (:mod:`formal_charge`), capacities ``v(n0 - q)`` from the pairing capacity polynomial;
* bond orders ``p`` from :func:`rsfff.ff.tersoff.bond_order.explicit_state`
  (``saturation: rebo | waterfill``);
* ``u = v - sum_j p``.

and the pairing energy is the functional at that state, ``-J p + kappa p^2 / 2 + E0(q)``.
There is no solver state to cache and no host offload: the model runs wherever its tensors
are, and every derivative is plain autograd.

``overbinding_penalty > 0`` adds the ReaxFF-style ``sum_i kappa_i softplus(N_i - v_i)^2`` on
the raw coordination to the per-atom energy (an ablation; the capacity bound is already built
into ``p``).
"""

from __future__ import annotations

import torch

from ..pairing.electronic_state import ElectronicState
from ..pairing.model import PairingModel, PairingOutput
from .bond_order import (
    SATURATIONS,
    explicit_state,
    overbinding_penalty,
    raw_bond_order,
    tersoff_state_energy,
)
from .formal_charge import PROJECTIONS, overload_weights, project_formal_charge
from .heads import TersoffFamily

__all__ = ["TersoffModel", "TersoffOutput"]

TersoffOutput = PairingOutput


class TersoffModel(PairingModel):
    """Args beyond the pairing model's (whose ``bo_*`` / ``state_cache`` are accepted and
    ignored):

    saturation           : ``"waterfill"`` (per-atom energetic sharing) or ``"rebo"``
                           (multiplicative, Tersoff / REBO).
    saturation_width     : rebo only, electrons over which the factor bends at capacity.
    fill_steps           : safeguarded Newton steps of the water filling.
    formal_charge        : ``"overload"`` (two passes: the charge follows the bonding),
                           ``"heavy_atoms"`` or ``"uniform"`` projection of the readout.
    formal_charge_width  : electrons over which the overload weights turn on.
    reconcile_width      : the smooth-minimum width reconciling a pair's two ends.
    dual_sweeps          : Jacobi sweeps of the coupled dual on top of the per-atom filling
                           (0: the pure per-atom rule).
    overbinding_penalty  : weight of the ReaxFF-style penalty on the raw coordination.
    """

    def __init__(
        self,
        *args,
        saturation: str = "waterfill",
        saturation_width: float = 0.02,
        fill_steps: int = 8,
        formal_charge: str = "overload",
        formal_charge_width: float = 0.1,
        reconcile_width: float = 2.0e-3,
        dual_sweeps: int = 0,
        overbinding_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        kwargs["state_cache"] = False
        kwargs.setdefault("bo_device", "same")
        super().__init__(*args, **kwargs)
        if saturation not in SATURATIONS:
            raise ValueError(f"saturation must be one of {SATURATIONS}, got {saturation!r}")
        if formal_charge not in PROJECTIONS:
            raise ValueError(f"formal_charge must be one of {PROJECTIONS}, got {formal_charge!r}")
        self.saturation = str(saturation)
        self.saturation_width = float(saturation_width)
        self.fill_steps = int(fill_steps)
        self.formal_charge = str(formal_charge)
        self.formal_charge_width = float(formal_charge_width)
        self.reconcile_width = float(reconcile_width)
        self.dual_sweeps = int(dual_sweeps)
        self.overbinding_penalty = float(overbinding_penalty)

    # the pairing model's warm-start plumbing is inert here
    def _warm_start(self, batch, batch_idx, n_sys, slot):
        return None, None

    def _store_state(self, batch, batch_idx, n_sys, st, slot):
        return None

    def _pairing(self, fam: TersoffFamily, positions, sub_pairs, dr_au, r_au, r_ang, batch_idx,
                 n_sys, total_charge, two_s, warm=None, warm_mask=None):
        """The explicit state and its energy: ``(state, J, kappa, e_pair (Pb,), e_atom (N,))``.

        ``two_s`` has no explicit analogue (the multiplicity enters the parameter network
        through the ``u`` conditioning only); ``warm`` is ignored.
        """
        J, kappa = self._coupling(fam, positions, sub_pairs, dr_au, r_au, r_ang)
        n_atoms = int(fam.q.shape[0])
        q_raw = fam.q_raw if getattr(fam, "q_raw", None) is not None else J.new_zeros(n_atoms)
        # the species table of the projector is what the heads index; atomic numbers come
        # from it so the projection does not need the batch
        atomic_numbers = self._atomic_numbers
        state_kwargs = dict(
            saturation=self.saturation, width=self.saturation_width, k_steps=self.fill_steps,
            reconcile_width=self.reconcile_width, dual_sweeps=self.dual_sweeps,
        )
        if self.formal_charge == "overload":
            q0 = project_formal_charge(
                q_raw, atomic_numbers, batch_idx, n_sys, total_charge, projection="heavy_atoms"
            )
            st0 = explicit_state(
                J, kappa, fam.tables, q0, sub_pairs, batch_idx, n_sys, self.temperature,
                **state_kwargs,
            )
            w = overload_weights(
                st0.p, sub_pairs, fam.tables[:, 1], atomic_numbers, batch_idx, n_sys,
                total_charge, width=self.formal_charge_width,
            )
            q = project_formal_charge(
                q_raw, atomic_numbers, batch_idx, n_sys, total_charge, weights=w
            )
        else:
            q = project_formal_charge(
                q_raw, atomic_numbers, batch_idx, n_sys, total_charge, projection=self.formal_charge
            )
        st = explicit_state(
            J, kappa, fam.tables, q, sub_pairs, batch_idx, n_sys, self.temperature, **state_kwargs
        )
        e_pair, e_atom = tersoff_state_energy(J, kappa, fam.chi, fam.eta, st)
        if self.overbinding_penalty > 0.0:
            b = raw_bond_order(J, kappa, self.temperature)
            e_atom = e_atom + self.overbinding_penalty * overbinding_penalty(
                b, st.valence, sub_pairs[0], sub_pairs[1], n_atoms, fam.kappa
            )
        return st, J, kappa, e_pair, e_atom

    def forward(self, batch, state=None, **kwargs) -> TersoffOutput:
        # stash what the projection needs; ``_pairing`` receives no batch
        self._atomic_numbers = batch.atomic_numbers
        try:
            return super().forward(batch, state, **kwargs)
        finally:
            self._atomic_numbers = None
