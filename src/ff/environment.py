"""The electrostatic environment at each atom, and its rotation-invariant reduction to pairs.

This is the missing channel from the environment to the bonded region. Everything else the
bond term sees is geometry; what it cannot see is that its surroundings are pulling on it.

The conventional fix -- switching on intramolecular electrostatics for the induced moments, as
AMOEBA does -- is deliberately **not** what happens here. Intramolecular point multipoles are a
poor model of how a bond responds to a field, the model would then need two different range
separations for the permanent and induced parts, and the final model is supposed to make no
distinction between the two. Instead the bond energy takes the external potential, field and
field gradient as *features* and learns its own response.

What is computed
----------------
For each atom, the action of everybody else's multipoles on it, in atomic units::

    phi_i    = sum_j gate_ij (T_ij^T M_j)[0]          Hartree / e
    F_i      = -sum_j gate_ij (T_ij^T M_j)[1:4]       Hartree / (e a0)
    gradF_i  = -sum_j gate_ij (T_ij^T M_j)[4:] D^T    Hartree / (e a0^2), spherical l=2

The signs are fixed by how the conjugate quantities already enter the response functional:
:func:`rsfff.mlip.eem.atomic_dipoles` minimizes ``chivec.mu + 1/2 mu^T alpha^-1 mu - F.mu``, so
whatever multiplies ``mu`` in the coupling *is* ``-F``. Deriving them that way rather than from
a sign convention means a field feature and the polarization solve cannot disagree about which
direction the environment pulls.

Three choices worth stating:

* **Undamped.** ``t_point`` only, no penetration -- the environment felt by a point core
  charge. Simple, and the thing being fed to an MLP does not need the short-range physics that
  the penetration term exists to supply.
* **Gated, not fragment-masked.** Weighted by the electrostatic channel's own range
  separation, so bonded contributions are switched off without any same-fragment indicator.
  A same-fragment pair *at range* still contributes, which is the whole point of the union pair
  list: in a large molecule the far end of the chain really is part of the environment. On
  water the switched-off remainder is small but not exactly zero, and that is measured rather
  than assumed.
* **From the converged multipoles**, not the frozen ones. The field an atom feels includes
  everybody else's *induced* response, which is where mutual polarization becomes visible to
  the bond. This is also what makes the adjoint mandatory: these features feed a head that is
  not variational in the multipoles.

The invariants
--------------
:class:`rsfff.ff.onebody.OneBodyEnvironment` worked out the constraint already: the bond head
must stay rotation-invariant, so the field cannot be handed to it raw, and the pair list is
undirected, so every quantity must also survive swapping ``i`` and ``j``. Of the available
contractions, the ones involving ``rhat`` flip sign under the swap -- ``F_i . rhat`` alone
would make the head's output depend on the arbitrary storage order of a pair. The two that
survive are ``(F_i - F_j) . rhat`` (both factors flip, so the product does not) and
``|(F_i + F_j) . rhat|``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .coupled_solve import _spherical_to_poly_map
from .multipole import spherical_to_cartesian_quadrupole
from .onebody import OneBodyEnvironment

#: Width of :func:`environment_pair_invariants` output, for sizing a head's input.
N_PAIR_INVARIANTS = 9


def electrostatic_environment(
    positions: torch.Tensor,        # (N, 3) Angstrom
    pair_index: torch.Tensor,       # (2, P)
    t_point: torch.Tensor,          # (P, K, K) undamped interaction tensor, ungated
    gate: torch.Tensor,             # (P,) the electrostatic range separation * taper
    m_real: torch.Tensor,           # (N, K) converged multipoles, polytensor layout
    *,
    max_rank: int = 1,
) -> OneBodyEnvironment:
    """Per-atom ``(potential, field, field_gradient)`` in atomic units.

    Populates the dataclass :mod:`rsfff.ff.onebody` has carried unused since the 1-body term
    was written; the invariants it documents are what :func:`environment_pair_invariants`
    implements.
    """
    i, j = pair_index[0], pair_index[1]
    w = gate.unsqueeze(-1)
    # g_i is the action of j on i and vice versa. Same contraction as the coupled solve's
    # operator restricted to its undamped block -- an atom's environment is literally the
    # gradient of the interaction with respect to its own multipoles.
    g_i = w * torch.einsum("pab,pa->pb", t_point, m_real[j])
    g_j = w * torch.einsum("pab,pb->pa", t_point, m_real[i])
    acc = torch.zeros_like(m_real).index_add(0, i, g_i).index_add(0, j, g_j)

    potential = acc[:, 0]
    field = -acc[:, 1:4] if max_rank >= 1 else acc.new_zeros(acc.shape[0], 3)
    grad = None
    if max_rank >= 2:
        d_map = _spherical_to_poly_map(acc.dtype, acc.device)
        grad = -(acc[:, 4:] @ d_map.t())
    return OneBodyEnvironment(potential=potential, field=field, field_gradient=grad)


def environment_pair_invariants(
    env: OneBodyEnvironment,
    pair_index: torch.Tensor,       # (2, P)
    r_hat: torch.Tensor,            # (P, 3) unit vector along r_j - r_i
) -> torch.Tensor:
    """``(P, 9)`` scalars: rotation-invariant and symmetric under swapping ``i`` and ``j``.

    Both properties are structural rather than trained, and both are checked in
    ``tests/test_ff_polarization.py``. The ordering is fixed:

    ``[V_i+V_j, |V_i-V_j|, |F_i|+|F_j|, ||F_i|-|F_j||, F_i.F_j,
       (F_i-F_j).rhat, |(F_i+F_j).rhat|, G_i+G_j, |G_i-G_j|]``

    with ``G = rhat^T gradF rhat``. Zeros fill the gradient slots when the model carries no
    rank-2 response, so the width does not depend on ``max_rank`` and a head sized for one
    works for the other.
    """
    i, j = pair_index[0], pair_index[1]
    v_i, v_j = env.potential[i], env.potential[j]
    f_i, f_j = env.field[i], env.field[j]

    n_i, n_j = f_i.norm(dim=-1), f_j.norm(dim=-1)
    cols = [
        v_i + v_j,
        (v_i - v_j).abs(),
        n_i + n_j,
        (n_i - n_j).abs(),
        (f_i * f_j).sum(-1),
        # even under the swap because the difference and rhat both change sign
        ((f_i - f_j) * r_hat).sum(-1),
        # odd, so take the magnitude
        ((f_i + f_j) * r_hat).sum(-1).abs(),
    ]
    if env.field_gradient is not None:
        g = spherical_to_cartesian_quadrupole(env.field_gradient)      # (N, 3, 3)
        proj = torch.einsum("pa,pab,pb->p", r_hat, g[i], r_hat)
        proj_j = torch.einsum("pa,pab,pb->p", r_hat, g[j], r_hat)
        cols += [proj + proj_j, (proj - proj_j).abs()]
    else:
        zero = v_i.new_zeros(v_i.shape[0])
        cols += [zero, zero]
    return torch.stack(cols, dim=-1)


__all__ = [
    "N_PAIR_INVARIANTS",
    "OneBodyEnvironment",
    "electrostatic_environment",
    "environment_pair_invariants",
]
