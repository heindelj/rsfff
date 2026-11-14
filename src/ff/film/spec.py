"""Reach of the film model's classical channels (moved from the retired ``rsfff.ff.expert_model``)."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ClassicalSpec", "DEFAULT_CLASSICAL"]


@dataclass(frozen=True)
class ClassicalSpec:
    """Reach of one classical channel, and which slot its inter-fragment form reads.

    ``cutoff`` is where the channel is tapered to exactly zero and ``taper_width`` how long that
    takes. These are per channel because the terms decay at wildly different rates: the
    electrostatics has a genuine ``1/r`` tail and truncating it early is a real error, while the
    Slater Pauli form is ~1e-8 Hartree by 7 Angstrom.

    ``environment`` says whether *inter*-fragment pairs read ``theta`` or ``theta_0``. See the
    module docstring: it follows from what each ALMO-EDA label is a function of. Setting it
    ``False`` on every channel is the two-body ablation. Intra pairs read ``theta_0`` whatever
    this says.

    ``to_induction`` books the channel's ``theta - theta_0`` difference to the induction label
    rather than dropping it. It is meaningful only when ``environment`` is off -- with the
    environment live the difference is already inside the channel's own energy.
    """

    cutoff: float
    taper_width: float = 1.0
    environment: bool = True
    to_induction: bool = False

    def __post_init__(self) -> None:
        if not self.cutoff > self.taper_width:
            raise ValueError(
                f"ClassicalSpec needs cutoff > taper_width, got {self.cutoff}, "
                f"{self.taper_width}"
            )
        if self.environment and self.to_induction:
            raise ValueError(
                "to_induction books a channel's theta - theta_0 difference to induction, "
                "which only makes sense when the channel itself reads theta_0 "
                "(environment=False); with environment=True that difference is already in the "
                "channel's own energy and booking it again would double count it"
            )

