"""Per-element range-separation priors: where each classical form stops being evaluated.

The unified model (:mod:`rsfff.ff.unified`) puts **every** pair through the classical
backbone, including bonded ones, which :func:`rsfff.ff.pairs.inter_fragment_pairs` used to
mask out. A covalent O-H at 0.96 Angstrom is ~1400 kJ/mol of Slater Pauli repulsion and a
large point-multipole 1/r; without a switch that is on the loss at step one and training
diverges immediately. These priors are what places the switch.

Measured, not guessed
---------------------
Over the 9579 frames of ``data/wb97mv_tzvpd/w{2,3,4,5}``, the intra- and inter-fragment
distance ranges per element pair are::

    pair    intra min  intra max  inter min  inter max
    H-H         1.355      1.729      1.611      8.268     <- OVERLAP
    O-H         0.890      1.072      1.538      7.951     <- clean gap
    O-O            --         --      2.487      7.729

**O-H separates cleanly** and that is the pair that matters: the gap 1.072 -> 1.538 is where
the switch goes, and it is the only pair whose classical form is dangerous at bonded range.

**H-H does not separate, and must not be made to.** An intramolecular H-H reaches 1.729 while
an intermolecular one starts at 1.611, so no distance threshold divides them -- see
``docs/range_separated_mlip.md`` and :func:`rsfff.ff.pairs.union_pairs`. This is not a
problem to engineer around, because the *routing* keeps the EDA channels clean on its own: an
intra H-H's energy is pooled into ``fragment_energy``, never into ``eda_cls_elec``, whatever
the switch does. So H-H is simply left switched **on** everywhere it physically occurs, and an
H-H at 1.6 Angstrom is treated as an H-H at 1.6 Angstrom whether or not the two atoms share a
fragment. That is the honest treatment, and it is only available because the switch is not
being asked to do the mask's old job.

Combination rule
----------------
``r0`` is a **per-atom** parameter combined as the geometric mean, the same log-space rule
every other pair parameter in this package uses
(:class:`rsfff.ff.dispersion.DispersionParameterHeads`,
:class:`rsfff.ff.pauli.PauliMultipoleHeads`). A per-atom ``r0`` cannot tell an intra O-H from
an inter O-H -- it is the *same* hydrogen in both -- and does not need to: the discrimination
comes from ``r``, which is what a range separation is for. The parameter only sets *where* the
handoff sits for that element pair.

Solving the geometric mean for the two targets above::

    r0(H) = 1.30                  ->  r0(H,H) = 1.30      on for every real H-H
    r0(O) = 1.20                  ->  r0(O,H) = 1.249     inside the 1.072 -> 1.538 gap
                                  ->  r0(O,O) = 1.20      well under the 2.487 minimum

Width
-----
``alpha`` has to be far steeper than the 8.0 the per-term modules use, because it now has to
cross a 0.47 Angstrom gap rather than taper a mid-range handoff. At ``alpha = 40`` the residual
Pauli leaking through at the *longest* intramolecular O-H (1.072) is ~0.6 kJ/mol, and the
classical form is fully on (0.99999) at the *shortest* intermolecular one (1.538). At
``alpha = 25`` the same leak is ~8.5 kJ/mol, which is larger than the signal being fit.

One consequence to keep in mind for the reactive regime: a steep switch puts a sharp feature
in the force at ``r0``. For equilibrium water that is harmless -- no O-H pair in the dataset
sits between 1.072 and 1.538 at all -- but a proton-transfer trajectory passes straight
through it. ``alpha`` is therefore learnable per channel, and is expected to soften once there
is data in that window for it to answer to.
"""

from __future__ import annotations

import torch

#: Per-element range-separation midpoint in Angstrom, before the geometric-mean combination.
#: See the module docstring for how these were solved for from the measured distance gaps.
DEFAULT_R0_PRIOR: dict[int, float] = {8: 1.20, 1: 1.30}

#: Crossover width in Angstrom^-1, shared by every channel at initialization.
DEFAULT_ALPHA_PRIOR: float = 40.0

#: The channels that carry a classical backbone and therefore need a range separation.
#: ``bond`` is absent deliberately -- it is a pure neural term with no divergent form to
#: protect.
RANGE_CHANNELS: tuple[str, ...] = ("elst", "pauli", "disp")


def build_range_priors(
    neighbor_types,
    *,
    r0_prior: dict[int, float] | None = None,
) -> torch.Tensor:
    """Per-species ``log r0`` prior ordered like ``neighbor_types``: ``(n_species,)``.

    Returned in **log space** because that is where the geometric-mean combination is linear
    (``log r0_ij = (log r0_i + log r0_j) / 2``), so no ``sqrt`` -- whose derivative is
    unbounded at zero -- appears anywhere in the switch. Same argument as
    :func:`rsfff.ff.dispersion.build_log_priors`.

    An element absent from the table raises rather than falling back to an average. A guessed
    ``r0`` is worse than no model: too low and a covalent pair gets a divergent classical
    energy on the first forward pass, too high and the classical form is silently switched off
    across the entire interaction region it exists to describe. Pass ``r0_prior`` to extend the
    table, having first measured the intra/inter gap for the new element the way the module
    docstring does.
    """
    table = dict(DEFAULT_R0_PRIOR)
    if r0_prior:
        table.update(r0_prior)
    missing = [int(z) for z in neighbor_types if int(z) not in table]
    if missing:
        raise KeyError(
            f"no range-separation prior for atomic number(s) {missing}; measure the intra/"
            f"inter distance gap for those elements and extend DEFAULT_R0_PRIOR, or pass "
            f"r0_prior={{Z: value}} in Angstrom. Refusing to guess."
        )
    bad = [int(z) for z in neighbor_types if not table[int(z)] > 0.0]
    if bad:
        raise ValueError(f"range-separation prior must be positive; got {bad} <= 0")
    return torch.tensor([float(table[int(z)]) for z in neighbor_types]).log()


__all__ = [
    "DEFAULT_ALPHA_PRIOR",
    "DEFAULT_R0_PRIOR",
    "RANGE_CHANNELS",
    "build_range_priors",
]
