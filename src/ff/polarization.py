"""The polarized and charge-transfer levels: one functional, three sets of constraints.

``docs/range_separated_mlip.md`` §5.1 defines the two remaining EDA components as what happens
when constraints are lifted from the *same* energy expression at a *fixed* fragmentation. This
module is that expression and those constraints::

    level       response features   external field   charge flow          label
    frozen      h_frag              none             intra-fragment       cls_elec, fragment_energy
    polarized   h_env               on, coupled      intra-fragment       pol = E1 - E0
    CT          h_env               on, coupled      inter-fragment open  ct  = E2 - E1

There is no separate polarization model here. The frozen level is
:class:`rsfff.ff.response.FragmentResponse` minimizing the internal energy alone; the polarized
level minimizes that *plus* the inter-fragment electrostatic interaction; the CT level does the
same with a wider channel graph. So each label is a difference of one function evaluated under
one more freedom, which is exactly what ALMO-EDA reports.

Why the same operator as ``cls_elec``
-------------------------------------
The coupling uses :func:`rsfff.ff.electrostatics.slater_elec_tensors` -- the frozen channel's
own point-plus-penetration operator, under the frozen channel's own learned range separation --
rather than the separate polarization damper a conventional polarizable force field would
introduce. Two consequences, both wanted:

* ``E_pol`` is **exactly zero** when the multipoles do not move, because the frozen energy is
  the same functional evaluated at ``M0``. It is a pure relaxation rather than the residue of
  two nearly-agreeing operators. Since ``M0`` minimizes only the internal part, the coupled
  minimum can only be lower, so ``E_pol <= 0`` holds by the variational principle wherever the
  two levels share response parameters -- which they do exactly at initialization.
* The polarization catastrophe is already handled. An ungated dipole-dipole coupling at bonded
  range drives the smallest eigenvalue of the functional negative (measured: -0.80 against
  +0.13 gated, ``tests/test_ff_coupled_solve.py``), turning the "solution" into a saddle and
  the energy unbounded below. The range separation switches that region off before it can
  happen, so no Thole-style damper is needed and none is introduced.

What the levels do *not* share is the response parameters: the polarized and CT levels evaluate
the same heads on the environment-aware descriptor, which is where genuine many-body content
enters. That breaks the sign guarantee once ``g`` trains away from zero -- watch it rather than
assume it.

Grouping
--------
Everything here is per **frame**, not per fragment. With the coupling on, the energy is not
decomposable per fragment: that is what "many-body" means. Only the frozen level keeps a
per-fragment internal energy, and that is exactly why ``fragment_energy`` is its label alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .coupled_solve import (
    CoupledSystem,
    coupled_energy,
    coupled_solve,
    multipoles_from_state,
)
from .electrostatics import slater_elec_pair_energy, slater_elec_tensors
from .multipole import build_polytensor, spherical_to_cartesian_quadrupole
from .response import ResponseParameters
from .units import BOHR_ANG


@dataclass
class LevelOutput:
    """One constraint level: the relaxed multipoles and the energy they minimize."""

    charges: torch.Tensor              # (N,) e
    mu: torch.Tensor | None            # (N, 3) e*bohr
    quad_s: torch.Tensor | None        # (N, 5) spherical, e*bohr^2
    #: (M,) per **frame**. The internal (on-site + channel) part of the functional.
    energy_internal: torch.Tensor
    #: (P,) gated classical electrostatics at these multipoles, from the *frozen channel's*
    #: own kernel. Summing it and adding ``energy_internal`` gives ``energy``.
    e_pair: torch.Tensor
    energy: torch.Tensor               # (M,) the minimized total for this level
    n_iter: int
    converged: torch.Tensor            # (M,) bool
    pd_fail: torch.Tensor              # (M,) bool


def build_coupled_system(
    rp: ResponseParameters,
    *,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    n_systems: int,
    bond_index: torch.Tensor,
    bond_batch: torch.Tensor,
    pair_index: torch.Tensor,
    gate: torch.Tensor,                # (P,) the elst channel's range separation * taper
    max_rank: int = 1,
) -> tuple[CoupledSystem, tuple]:
    """Assemble the solver's view of one level. Returns ``(system, tensors)``.

    ``tensors`` is the un-gated :func:`~rsfff.ff.electrostatics.slater_elec_tensors` tuple,
    handed back so the pair energy at the converged multipoles can reuse it instead of
    rebuilding four ``(P, K, K)`` blocks.
    """
    i, j = pair_index[0], pair_index[1]
    dr_au = (positions[j] - positions[i]) / BOHR_ANG
    r_au = (positions[i] - positions[j]).norm(dim=-1) / BOHR_ANG
    tensors = slater_elec_tensors(dr_au, r_au, rp.b, pair_index, max_rank=max_rank)
    w = gate[:, None, None]
    sys = CoupledSystem(
        n_systems=int(n_systems),
        n_atoms=int(positions.shape[0]),
        batch_idx=batch_idx,
        bond_index=bond_index,
        bond_batch=bond_batch,
        pair_index=pair_index,
        max_rank=int(max_rank),
        chi=rp.chi,
        eta=rp.eta,
        q0=rp.q0,
        compliance=rp.compliance,
        chivec=rp.chivec,
        mu0=rp.mu0,
        quad0=rp.quad0,
        alpha=rp.alpha,
        chiquad=rp.chiquad,
        cquad=rp.cquad,
        t_point=w * tensors[0],
        t_ss=w * tensors[1],
        t_1c_i=w * tensors[2],
        t_1c_j=w * tensors[3],
        m_nuc=build_polytensor(rp.z, None, None, max_rank=max_rank),
    )
    return sys, tensors


def coupled_response(
    rp: ResponseParameters,
    *,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    n_systems: int,
    bond_index: torch.Tensor,
    bond_batch: torch.Tensor,
    pair_index: torch.Tensor,
    gate: torch.Tensor,
    max_rank: int = 1,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    maxiter: int = 100,
) -> LevelOutput:
    """Minimize the coupled functional and report what it converged to.

    Differentiable: the solve itself runs under ``no_grad`` and the gradient comes back through
    the adjoint of :class:`rsfff.ff.coupled_solve._CoupledSolve`. That matters because the
    electrostatic environment features are built from ``charges``/``mu``/``quad_s`` and feed a
    correction head that is *not* variational in them -- see ``docs/range_separated_mlip.md``
    §6.1 for why detaching would give analytic forces that are simply wrong.
    """
    sys, tensors = build_coupled_system(
        rp, positions=positions, batch_idx=batch_idx, n_systems=n_systems,
        bond_index=bond_index, bond_batch=bond_batch, pair_index=pair_index,
        gate=gate, max_rank=max_rank,
    )
    info_out: list = []
    state, n_iter = coupled_solve(
        sys, rtol=rtol, atol=atol, maxiter=maxiter, info_out=info_out
    )
    info = info_out[0]
    q, mu, theta = multipoles_from_state(sys, state)
    mu = mu if mu.numel() else None
    theta = theta if theta.numel() else None

    energy_internal = coupled_energy(sys, state)

    # The pair half, through the frozen channel's own function on the relaxed multipoles.
    quad_c = None if theta is None else spherical_to_cartesian_quadrupole(theta)
    m_real = build_polytensor(q, mu, quad_c, max_rank=max_rank)
    m_shell = build_polytensor(q - rp.z, mu, quad_c, max_rank=max_rank)
    m_nuc = build_polytensor(rp.z, None, None, max_rank=max_rank)
    i, j = pair_index[0], pair_index[1]
    dr_au = (positions[j] - positions[i]) / BOHR_ANG
    r_au = (positions[i] - positions[j]).norm(dim=-1) / BOHR_ANG
    e_point, e_pen = slater_elec_pair_energy(
        dr_au, r_au, m_real, m_shell, m_nuc, rp.b, pair_index,
        max_rank=max_rank, tensors=tensors,
    )
    e_pair = gate * (e_point + e_pen)

    pair_batch = batch_idx[i]
    energy = energy_internal + e_pair.new_zeros(n_systems).index_add(0, pair_batch, e_pair)

    return LevelOutput(
        charges=q,
        mu=mu,
        quad_s=theta,
        energy_internal=energy_internal,
        e_pair=e_pair,
        energy=energy,
        n_iter=n_iter,
        converged=info.converged,
        pd_fail=info.pd_fail,
    )


__all__ = ["LevelOutput", "build_coupled_system", "coupled_response"]
