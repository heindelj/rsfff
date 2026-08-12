"""1-body energy and classical electrostatics, sharing one per-fragment response solve.

    E_total(frame) = sum_f [ sum_i E0[Z_i] + E_internal(f) + E_bond(f) ]      1-body
                   + sum_{i<j, different fragments} [ ... ]                   2-body

Both halves are functions of the *same* :class:`rsfff.ff.response.FragmentResponse`, solved
once per forward pass and handed to both. That is the point of this module: the multipoles
that interact and the internal energy that paid for them are the same object, so they cannot
drift apart the way two separately-trained copies would.

Why they have to be trained together
------------------------------------
Each target on its own leaves the response under-determined, and in a different direction:

* ``eda_cls_elec`` constrains the multipoles only through their pairwise interaction, which
  many different atomic partitions reproduce equally well -- measured: at ``max_rank 2`` the
  charges collapse toward zero and the electronegativity ordering inverts while the energy
  MAE keeps improving.
* the **fragment multipole** targets fix ``q``, ``mu`` and ``Theta`` themselves, but say
  nothing about the response energy: ``(C, chiquad) -> (lambda C, chiquad / lambda)`` leaves
  ``Theta`` exactly invariant while scaling ``E_quad`` by ``1 / lambda``.
* the **fragment energy** target sees ``E_internal``, which is what closes that gauge -- but
  only softly, because the bond head is also a smooth function of the same geometry and can
  absorb part of what ``E_internal`` would otherwise have to explain.

Together they pin the multipoles (multipole targets), the cost of producing them (fragment
energies), and their interaction (``cls_elec``) at once.

What stays exact
----------------
Everything the two terms guaranteed separately survives, because sharing the solve changes
nothing about its grouping:

* the descriptor and the solve are both grouped by ``fragment_idx``, so the 1-body part is
  exactly one-body and the electrostatics exactly two-body;
* fragment charge is conserved to machine precision;
* ``E_internal`` appears in the 1-body energy and **not** in the interaction, which is the
  same statement ``SlaterElectrostatics`` has always made about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .electrostatics import ElectrostaticsOutput, SlaterElectrostatics
from .onebody import OneBodyEnergy, OneBodyOutput
from .response import FragmentResponseOutput


@dataclass
class OneBodyElecOutput:
    """Both terms' outputs plus the shared solve they were built from."""

    #: (B,) the 1-body energy summed over the frame's fragments **plus** the interaction.
    #: This is a total energy in the atomization sense, not an interaction energy -- the
    #: two constituent fields below are what the two losses actually compare against.
    energy: torch.Tensor
    onebody: OneBodyOutput
    elec: ElectrostaticsOutput
    response: FragmentResponseOutput

    @property
    def fragment_energy(self) -> torch.Tensor:
        """(F,) per-fragment 1-body energy -- the ``fragment_energies`` target."""
        return self.onebody.fragment_energy

    @property
    def interaction_energy(self) -> torch.Tensor:
        """(B,) the electrostatic interaction -- the ``eda_cls_elec`` target."""
        return self.elec.energy

    # The multipole loss is duck-typed on these three, so the composite output can be
    # handed to it directly instead of reaching into `.response` at every call site.
    @property
    def charges(self) -> torch.Tensor:
        return self.response.charges

    @property
    def mu(self) -> torch.Tensor | None:
        return self.response.mu

    @property
    def quad_s(self) -> torch.Tensor | None:
        return self.response.quad_s


class OneBodyElecModel(nn.Module):
    """Featurizer (grouped by fragment) -> one shared solve -> 1-body + electrostatics.

    The featurizer runs once and the solve runs once; the two terms are then evaluated on
    that shared result. Grouping the descriptor by ``fragment_idx`` is not optional here for
    the same reason it is not optional in either term alone -- a monomer whose descriptor
    sees its neighbours has neither an exactly-one-body energy nor an exactly-two-body
    interaction, and the loss would not show it.
    """

    def __init__(
        self,
        featurizer,
        onebody: OneBodyEnergy,
        elec: SlaterElectrostatics,
    ) -> None:
        super().__init__()
        if onebody.response is None:
            raise ValueError(
                "the composite model needs the 1-body term to hold the shared response; "
                "build OneBodyEnergy with response=<the same FragmentResponse>"
            )
        if onebody.response is not elec.response:
            raise ValueError(
                "the 1-body and electrostatics terms hold *different* FragmentResponse "
                "instances. Sharing one is the entire point: otherwise the multipoles that "
                "interact and the internal energy that paid for them are computed from two "
                "parameter sets that are free to drift apart."
            )
        self.featurizer = featurizer
        self.onebody = onebody
        self.elec = elec

    @property
    def response(self):
        return self.onebody.response

    @property
    def params(self):
        return self.response.params

    @property
    def r0(self) -> torch.Tensor:
        return self.elec.r0

    def forward(self, batch) -> OneBodyElecOutput:
        feats = self.featurizer(batch, batch.fragment_idx)
        res = self.response(batch, feats)
        one = self.onebody(batch, feats, response=res)
        two = self.elec(batch, feats, response=res)
        return OneBodyElecOutput(
            energy=one.energy + two.energy,
            onebody=one,
            elec=two,
            response=res,
        )


__all__ = ["OneBodyElecModel", "OneBodyElecOutput"]
