"""Collective variables on the mediator, and the harmonic bias that drives them.

The point of biasing here is that a plain trajectory of an ion in a small water cluster spends
almost all of its time with the membership decided -- one weight at 0.999, the mediator idle --
and the frames worth labeling are the rare ones where it is not. A restraint on how *split* the
membership is turns those frames from rare into the default.

Nothing in this file computes a gradient. The bias returns a scalar that is already in the
autograd graph of the caller's positions, and
:class:`~rsfff.md.calculator.MediatedCalculator` adds it to the model energy before a single
backward. That is the whole of "include the gradient of the routing weight in the dynamics":
``w`` is differentiable in ``R`` through the contact distance (into the C² ``validity_bump``)
and through the lambda-SOAP features (into the score net), so the bias force comes out of the
same backward pass that produces the physical force.

Two coordinates
---------------
``ambiguity`` is the one the exercise is about; ``delta`` is its control. Biasing on
``ambiguity`` samples where *this* mediator believes the ambiguity is, which is exactly the
model we are trying to improve -- self-referential, though not circular, since the enumeration
that decides which candidates exist at all is geometric and the eventual labels come from
Q-Chem. A ``delta``-biased run at matched statistics is the only clean way to tell a real
sampling gain from the model steering toward its own opinion, so both are here and the driver
can run either.
"""

from __future__ import annotations

import torch

__all__ = ["HarmonicBias", "ambiguity", "logit", "transfer_delta"]


def ambiguity(weights: torch.Tensor) -> torch.Tensor:
    """``A = 1 - sum_m w_m^2``: 0 when the mediator has decided, ``1 - 1/M`` when it is torn.

    Deliberately not :attr:`~rsfff.ff.mediator.MediatorOutput.occupancy`, which is
    ``1 - max_m w_m``. That is the right *diagnostic* -- it reads directly as "how much weight
    is off the winner" -- but it is the wrong thing to differentiate: ``max`` has a kink
    wherever two weights tie, which is precisely the geometry a bias on it would drive the
    system to sit at. This form is a polynomial in ``w`` and smooth everywhere, and at ``M=2``
    it is ``2 * w_0 * w_1``, half of §10's ambiguity gate ``A = 4 pi_1 pi_2``.
    """
    return 1.0 - (weights**2).sum()


def logit(weights: torch.Tensor, *, clamp: float = 8.0) -> torch.Tensor:
    """``log(w_0 / (1 - w_0))``: how decisively the base decomposition is winning.

    **The routing-weight coordinate to actually bias on**, and the reason is conditioning
    rather than taste. A softmax weight saturates -- ``dw/dR`` is proportional to
    ``w(1 - w)``, which collapses to nothing once the mediator has decided -- so any
    coordinate built directly from ``w`` has a gradient spanning orders of magnitude, and a
    force constant large enough to move the decided region is catastrophic at the crossover.
    Measured: a harmonic on :func:`ambiguity` at ``k = 10`` reached 51.6 eV/Angstrom and
    aborted the trajectory at step 40. The logit is the exact inverse of that saturation: the
    ``w(1 - w)`` in the numerator cancels the one in the denominator and what is left is
    essentially the softmax score gap, which is close to *linear* in the transfer coordinate
    (measured slope ~10.7 per Angstrom over the whole crossover).

    Zero at a 50/50 membership, positive while the base is winning. Generalizes past ``M = 2``
    without a second definition: everything that is not the base is the competitor.

    ``clamp`` bounds it with ``L * tanh(l / L)`` rather than ``torch.clamp``. A candidate's
    weight goes to exactly zero as its envelope closes, so the raw logit diverges precisely at
    the enumeration boundary -- and a hard clamp would put a kink in the bias force there, at
    the one geometry the C² argument of :mod:`rsfff.md.assign` is about. The soft form costs a
    little pull from far away and keeps the derivative continuous.
    """
    w0 = weights[0].clamp(1e-12, 1.0 - 1e-12)
    raw = torch.log(w0) - torch.log1p(-w0)
    return clamp * torch.tanh(raw / clamp)


def transfer_delta(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    fragments: torch.Tensor,
    contested: torch.Tensor,
    *,
    beta: float = 10.0,
) -> torch.Tensor:
    """``r(H, O_donor) - r(H, O_acceptor)`` for the contested atom nearest to transferring.

    The proton-transfer coordinate as the scans in ``notebooks/mediator_plotting.ipynb`` draw
    it: negative while the atom is still bonded where the base put it, zero at the transfer
    point. Model-free -- geometry and the enumeration, never a network output -- which is the
    entire reason it exists alongside :func:`ambiguity`.

    Three things it has to get right, each of which was wrong in an earlier form and each of
    which quietly produces a coordinate that looks plausible and restrains nothing:

    1. **Distances to each host's oxygen, not the mediator's ``rho``.** ``contact_distance``
       returns the nearest atom of the host, which on a compact cluster is frequently another
       hydrogen. Restraining that form to zero was satisfied by rotating a neighbouring water:
       ``cv`` reached +0.001 while the real O-H asymmetry stayed at -0.713 Angstrom.
    2. **The acceptor is taken over candidates that actually move this atom.** A candidate that
       moves some *other* hydrogen leaves this one's host untouched, so it reports the donor
       distance back, and a softmin including it returns the donor distance -- giving
       ``delta = 0`` at every geometry.
    3. **A smooth max over the contested atoms, not a sum.** Generation is generous, so ``D``
       picks up terminal hydrogens whose "transfer" is a 2.3 Angstrom reach worth about -1.3
       Angstrom of coordinate. Summing buries the bridge under them; the max is the atom that
       is actually reacting.

    Both reductions are soft (``beta = 10 / Angstrom``, within ~0.01 Angstrom of the hard
    min/max) so the coordinate stays differentiable where two candidates are equidistant.
    """
    if fragments.shape[0] < 2 or contested.numel() == 0:
        return positions.new_zeros(())
    is_o = torch.as_tensor(atomic_numbers).reshape(-1) == 8

    def host_oxygen_distance(m: int, atom: int) -> torch.Tensor:
        mates = (fragments[m] == fragments[m][atom]) & is_o
        if not bool(mates.any()):                      # a bare ion host has no oxygen
            return positions.new_tensor(float("inf"))
        return (positions[mates] - positions[atom]).norm(dim=-1).min()

    per_atom = []
    for atom in contested.reshape(-1).tolist():
        movers = [m for m in range(1, fragments.shape[0])
                  if int(fragments[m][atom]) != int(fragments[0][atom])]
        if not movers:
            continue
        donor = host_oxygen_distance(0, atom)
        others = torch.stack([host_oxygen_distance(m, atom) for m in movers])
        others = torch.nan_to_num(others, posinf=1.0e3)   # inf would poison the logsumexp
        acceptor = -torch.logsumexp(-beta * others, dim=0) / beta
        per_atom.append(donor - acceptor)
    if not per_atom:
        return positions.new_zeros(())
    return torch.logsumexp(beta * torch.stack(per_atom), dim=0) / beta


class HarmonicBias:
    """``E = 0.5 * k * (cv - target)^2`` on one mediator collective variable.

    Harmonic rather than a linear ``-k * cv`` reward so the pull is bounded and ``target`` is a
    knob: a ladder of targets sweeps the crossover, where a linear bias only ever leans one
    way with no way to say how far. ``k = 0`` returns an exact zero -- not a small number --
    so an unbiased run is bit-identical to plain mediated dynamics rather than merely close.

    **Scale, measured, not estimated.** The restraint force is ``k * (cv - target) * dcv/dR``,
    and for ``ambiguity`` both factors are small out in the decided region -- which is where a
    trajectory starts and where the bias has to do its work. ``dA/ddelta`` reaches ~3 per
    Angstrom *inside* the crossover but is 0.01-0.1 outside it, so sizing ``k`` from the
    crossover slope underestimates it by two orders of magnitude.

    Relaxing an H5O2+ from a localized start on ``checkpoints/ion_mediator_v4_full``:

    ===============  ==========================  ==============================
    ``cv``           useful ``k``                what it reaches
    ===============  ==========================  ==============================
    ``ambiguity``    **10 Ha** (5 is too weak)   ``A = 0.50``, proton centred
    ``delta``        **0.3-1 Ha/Angstrom^2**     ``delta`` -0.77 -> -0.14 Angstrom
    ===============  ==========================  ==============================

    ``delta`` is far better conditioned, being a distance with an order-unity gradient, and
    that is another reason to keep it: a control run does not have to be tuned separately for
    every checkpoint. The large ``ambiguity`` constant makes the *biased* energy physically
    meaningless as a number -- which is why ``results["energy_hartree"]`` carries the model's
    own energy and the driver logs that one.

    **Do not target the maximum.** ``A`` is *stationary* at equal weights -- ``dA/dw = -2w`` is
    uniform there and the weights sum to one, so ``dA/dR`` vanishes identically at a 50/50
    membership. With ``target`` at the maximum, both factors of ``k * (A - target) * dA/dR``
    go to zero together and the restoring force decays as the cube of the deviation: a very
    soft restraint exactly where it is supposed to bite. The default ``0.4`` sits on the flank,
    where ``dA/dR`` is order 1 per Angstrom and the restraint has real gradient to work with.
    """

    def __init__(
        self,
        cv: str = "logit",
        k: float = 0.0,
        target: float = 0.0,
        max_deviation: float | None = 1.0,
    ) -> None:
        if cv not in ("logit", "ambiguity", "delta"):
            raise ValueError(
                f"unknown bias coordinate {cv!r}, expected 'logit', 'ambiguity' or 'delta'"
            )
        self.cv, self.k, self.target = cv, float(k), float(target)
        #: Beyond this deviation the restraint goes **linear**, so the force saturates at
        #: ``k * max_deviation * |dcv/dR|`` instead of growing without bound. A pure harmonic
        #: dumps ``0.5 * k * (cv - target)^2`` into the system the instant it is switched on --
        #: measured at 16 eV for a trajectory starting at ``logit = 6.35`` with ``k = 0.03``,
        #: which took a 13-atom cluster to 7700 K in twenty steps. The linear tail is a Huber
        #: restraint and is C^1 at the join. ``None`` restores the pure harmonic.
        self.max_deviation = None if max_deviation is None else float(max_deviation)

    def __repr__(self) -> str:                                          # pragma: no cover
        return (f"HarmonicBias(cv={self.cv!r}, k={self.k}, target={self.target}, "
                f"max_deviation={self.max_deviation})")

    def value(self, out, group) -> torch.Tensor:
        """The collective variable itself, in the graph. Useful for logging without a rerun."""
        if self.cv == "logit":
            return logit(out.mediator.weights)
        if self.cv == "ambiguity":
            return ambiguity(out.mediator.weights)
        return transfer_delta(
            group.positions, group.atomic_numbers, group.fragments, group.contested
        )

    def __call__(self, out, group) -> tuple[torch.Tensor, torch.Tensor]:
        """``(bias energy, cv)``. Both are returned because the driver logs the second."""
        cv = self.value(out, group)
        if self.k == 0.0:
            return group.positions.new_zeros(()), cv
        d = cv - self.target
        w = self.max_deviation
        if w is None or float(d.detach().abs()) <= w:
            return 0.5 * self.k * d**2, cv
        # Linear continuation, matched in value and slope at |d| = w.
        return self.k * w * (d.abs() - 0.5 * w), cv
