"""A weak spherical wall, so a handful of waters stays a cluster.

Small ionic water clusters in vacuum evaporate: a hydrogen bond breaks, one water wanders off,
and the rest of the trajectory samples a monomer and a smaller cluster. A flat-bottomed sphere
costs nothing inside and turns that into a bounce.

Two decisions here are not cosmetic.

**The sphere is centered on the instantaneous center of mass, not on the origin.** A fixed
center is a potential that depends on absolute position, so it exerts a net force on the
cluster whenever the cluster is not concentric with it, and the trajectory acquires a drift
that has to be projected out afterwards. Measuring from the running COM makes the potential
translation-invariant, and a translation-invariant potential has zero net force by
construction -- ``forces.sum(0) == 0``, which ``tests/test_reactive_md.py`` checks directly.

**It acts on atoms, never on fragments.** "The center of mass of a water" is not defined
without a fragment assignment, and the assignment is exactly the thing that changes when a
proton hops -- so a COM-based wall would redefine itself mid-flight and put a step in the
force. Oxygens carry ~89% of a water's mass and sit within 0.07 Angstrom of its COM, so
restraining oxygens is the same wall without the discontinuity. Hydrogens get their own,
slacker shell whose only job is to stop a dissociated proton leaving; it is far enough out
that a bonded hydrogen never reaches it.
"""

from __future__ import annotations

import torch

__all__ = ["flat_bottom_sphere"]

#: Atomic masses in amu for the elements this corpus contains. Only the ratio matters -- the
#: COM this defines is a geometric center, not a dynamical quantity.
_MASS = {1: 1.008, 8: 15.999}


def flat_bottom_sphere(
    positions: torch.Tensor,       # (N, 3) Angstrom
    atomic_numbers: torch.Tensor,  # (N,)
    *,
    radius: float,
    k: float,
    h_slack: float = 1.2,
) -> torch.Tensor:
    """``sum_i 0.5 k max(0, |r_i - R_com| - R_i)^2``, with ``R_i`` wider for hydrogen.

    ``k`` is in Hartree/Angstrom^2 and ``radius`` in Angstrom. Exactly zero -- not merely
    small -- while every atom is inside its shell, so an unconfined region of a trajectory is
    unperturbed rather than gently squeezed.

    C^1 but not C^2: the quadratic meets zero with matching value and slope, and the curvature
    steps. That is the standard flat-bottom restraint and it is fine for a thermostatted
    integrator, which never sees the second derivative. It is *not* fine for the C^2 argument
    the mediator rests on, which is why the wall is kept out of the mixture entirely and added
    as a separate term.
    """
    if k == 0.0:
        return positions.new_zeros(())
    z = torch.as_tensor(atomic_numbers).reshape(-1)
    mass = torch.tensor(
        [_MASS.get(int(v), 1.0) for v in z], dtype=positions.dtype, device=positions.device
    )
    # Differentiating through the COM is what makes the net force vanish identically rather
    # than merely numerically: every atom's displacement enters both its own term and the
    # center, and the two contributions cancel in the sum.
    com = (mass[:, None] * positions).sum(0) / mass.sum()
    shell = torch.where(
        z == 1,
        positions.new_full((z.shape[0],), radius + h_slack),
        positions.new_full((z.shape[0],), radius),
    )
    excess = ((positions - com).norm(dim=-1) - shell).clamp(min=0.0)
    return 0.5 * k * (excess**2).sum()
