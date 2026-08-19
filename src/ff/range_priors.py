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

    pair    intra min  intra max  inter min  inter max   pairs (intra / inter)
    H-H         1.355      1.729      1.608      8.268    33,522 / 191,536
    O-H         0.890      1.072      1.538      7.951
    O-O            --         --      2.487      7.729

**Read the populations, not the extremes.** The H-H ranges appear to overlap on
``intra max 1.729 > inter min 1.608``, and an earlier version of this module concluded from
exactly that that H-H could not be separated and should be left switched on. That was wrong:
min and max over 225,000 pairs are the tails, not the distribution. The actual contamination
is **19 inter pairs out of 191,536 (0.01%)** below the intra maximum, and the intra
distribution's 99.9th percentile is 1.679. A threshold placed at ~1.75 separates the two
classes essentially perfectly.

So all three element pairs are separable and all three are separated. Measured on the same
data, per frame, with ``alpha = 40``::

    r0(H,H)   intra gate   leaks into fragment_energy   removed from the inter sum
    1.30         0.9996             122.02 kJ/mol                0.00 kJ/mol
    1.75         0.0006               0.07 kJ/mol                0.13 kJ/mol

The old 1.30 left every intramolecular H-H fully switched on, pushing 122 kJ/mol per frame of
intramolecular electrostatics into ``fragment_energy`` for the bond head to absorb. That is
not wrong in the sense of breaking an invariant -- routing keeps the EDA channels clean either
way -- but it is a needlessly poor decomposition, and it costs 0.13 kJ/mol per frame of real
inter-fragment energy to fix.

Combination rule
----------------
``r0`` is per **channel** and per **atom**, combined across a pair as the geometric mean -- the
same log-space rule
(:class:`rsfff.ff.dispersion.DispersionParameterHeads`,
:class:`rsfff.ff.pauli.PauliMultipoleHeads`). A per-atom ``r0`` cannot tell an intra O-H from
an inter O-H -- it is the *same* hydrogen in both -- and does not need to: the discrimination
comes from ``r``, which is what a range separation is for. The parameter only sets *where* the
handoff sits for that element pair.

Two constraints fix both numbers: ``r0(H,H)`` lands above the intramolecular H-H
distribution, and ``r0(O,H)`` lands in the 1.072 -> 1.538 gap::

    r0(H) = 1.75                  ->  r0(H,H) = 1.75      above intra max 1.729
    r0(O) = 0.893                 ->  r0(O,H) = 1.250     inside the 1.072 -> 1.538 gap
                                  ->  r0(O,O) = 0.893     well under the 2.487 minimum

``r0(O)`` is not a physical radius and should not be read as one -- only the pairwise
combinations are constrained by anything, and the geometric mean has one free scale per
element that the data does not fix.

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
DEFAULT_R0_PRIOR: dict[int, float] = {8: 0.893, 1: 1.75}

#: Channels whose prior differs from :data:`DEFAULT_R0_PRIOR`. Absent channels use it.
#:
#: **Dispersion needs a wider bonded exclusion than the other two**, and the reason is the
#: energy scale rather than anything about where the bond ends. A covalent O-H at 0.97 Angstrom
#: carries ``C6/r^6 ~ 480 kJ/mol`` of undamped dispersion against ``C6(O,H) ~ 8.4`` a.u., so a
#: gate of 0.1 there is 48 kJ/mol of "intramolecular dispersion" -- twenty times the ``ob_mae``
#: the fit is judged on, and exactly degenerate with the bond channel that is supposed to carry
#: it. Electrostatics and Pauli have no comparable amplification: the same gate on the Pauli
#: channel is a rounding error.
#:
#: The values are solved for the way the shared ones were, by measuring both sides. On 200 w3
#: frames with a trained model's ``C6``, sweeping the pair of per-element values at
#: ``alpha = 40`` (intra leak in kJ/mol per fragment, inter loss in kJ/mol per frame against
#: a total inter-fragment dispersion of -17.5)::
#:
#:     r0(H)  r0(O)   r0(H,H)  r0(O,H)   intra leak   inter lost
#:      1.75   0.893    1.750    1.250      -0.0020      -0.0002    <- the shared prior
#:      1.90   1.10     1.900    1.446      -0.0000      -0.0068
#:      2.00   1.30     2.000    1.612      -0.0000      -0.0814    <- chosen
#:      2.10   1.50     2.100    1.775      -0.0000      -1.5861
#:      2.20   1.70     2.200    1.934      -0.0000      -4.7152
#:
#: The window closes fast above ``r0(O,H) ~ 1.7`` because the shortest intermolecular O-H is
#: 1.538 Angstrom. (2.00, 1.30) buys 0.26 in ``log r0`` over the shared prior for 0.5% of the
#: intermolecular dispersion, and that margin is the point -- see the caution below.
#:
#: **A higher prior alone does not close the leak**, and it is worth being explicit about why,
#: because the obvious reading of this table is that the shared prior was already fine. It is,
#: *at the prior*. What escapes is the per-pair ``d_log_r0`` from
#: :class:`rsfff.mlip.unified_head.UnifiedPairHead`, bounded by ``max_log_dev = 0.7``. Measured
#: on a trained frozen checkpoint, that deviation had reached **-0.42 on intra and inter O-H
#: alike** -- a uniform downward shift, not a discrimination -- dropping ``r0_pair`` to 0.79,
#: opening the bonded gate to 0.999, and putting **-39 kJ/mol per fragment** of dispersion
#: between covalently bonded atoms. The driver was the ``r0`` penalty, which was an *unbounded
#: linear pull downward*; its restriction to inter-fragment pairs did not protect the bonded
#: region, because the shared trunk applies one deviation to both. That penalty is now a
#: one-sided barrier about the prior (:class:`rsfff.train.train_unified.AnchorTerms`), which is
#: what makes a prior a floor rather than a starting point.
CHANNEL_R0_PRIOR: dict[str, dict[int, float]] = {
    "disp": {8: 1.30, 1: 1.60},
}

#: Crossover width in Angstrom^-1, shared by every channel at initialization.
DEFAULT_ALPHA_PRIOR: float = 40.0

#: The channels that carry a classical backbone and therefore need a range separation.
#: ``bond`` is absent deliberately -- it is a pure neural term with no divergent form to
#: protect.
RANGE_CHANNELS: tuple[str, ...] = ("elst", "pauli", "disp")


def build_range_priors(
    neighbor_types,
    *,
    channels: tuple[str, ...] = RANGE_CHANNELS,
    r0_prior: dict[int, float] | None = None,
) -> torch.Tensor:
    """Per-channel, per-species ``log r0`` prior: ``(len(channels), n_species)``.

    Rows are ordered like ``channels`` and columns like ``neighbor_types``. Each row is
    :data:`DEFAULT_R0_PRIOR` unless :data:`CHANNEL_R0_PRIOR` overrides that channel; an
    explicit ``r0_prior`` overrides both, for every channel, and is the escape hatch for a
    new element rather than for retuning a channel.

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
    rows = []
    for name in channels:
        table = dict(DEFAULT_R0_PRIOR)
        table.update(CHANNEL_R0_PRIOR.get(name, {}))
        if r0_prior:
            table.update(r0_prior)
        missing = [int(z) for z in neighbor_types if int(z) not in table]
        if missing:
            raise KeyError(
                f"no range-separation prior for atomic number(s) {missing} on the {name!r} "
                f"channel; measure the intra/inter distance gap for those elements and extend "
                f"DEFAULT_R0_PRIOR, or pass r0_prior={{Z: value}} in Angstrom. Refusing to guess."
            )
        bad = [int(z) for z in neighbor_types if not table[int(z)] > 0.0]
        if bad:
            raise ValueError(f"range-separation prior must be positive; got {bad} <= 0")
        rows.append([float(table[int(z)]) for z in neighbor_types])
    return torch.tensor(rows).log()


__all__ = [
    "CHANNEL_R0_PRIOR",
    "DEFAULT_ALPHA_PRIOR",
    "DEFAULT_R0_PRIOR",
    "RANGE_CHANNELS",
    "build_range_priors",
]
