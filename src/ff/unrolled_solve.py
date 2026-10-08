"""The non-variational polarization solve: a fixed number of iterations, unrolled.

``docs/fff_nonvariational.md`` in code. :mod:`rsfff.ff.coupled_solve` minimizes the coupled
functional ``E(x) = 1/2 x^T A x + b^T x`` to a tolerance and recovers gradients through an
adjoint. This module instead produces a state ``x_K`` by ``K`` fixed iterations of

    x^(k+1) = Phi( F0 + T~ . P x^(k) ),        x^(0) = Phi(F0),

and hands it back for the caller to evaluate the **physical** functional at -- the same
:func:`~rsfff.ff.coupled_solve.coupled_energy` and pair energy as the converged path, at a
point that is deliberately *not* its minimum. Everything is ordinary autograd; there is no
``autograd.Function`` here, no adjoint, no tolerance and no host sync.

The pieces, in the solver's rescaled variables ``x = (v, u, w)``:

``F0``
    the raw coupling gradient ``(g_q, g_mu, g_theta)`` at the *permanent* multipoles
    ``m(0)`` -- the field of every other fragment's permanent shell and nucleus, with the
    physical damping. Constant through the loop.
``Phi``
    the exact fragment-internal solve at given fields. The dipole and quadrupole sectors are
    inverse-free in the rescaled variables (``u = -(chivec + g_mu)``, ``w = -(chiquad +
    g_theta)``: the rescaling ``mu = alpha u`` is what makes ``alpha^-1`` unnecessary, the
    same reason the CG matvec never forms it). The charge sector is ``sqe_solve``'s system
    ``(I + L S) v = -B^T (chi + eta q0 + g_q)`` with ``L = B^T eta B``, factorised **per
    fragment** (or per frame) as a small dense block. ``I + L S`` is invertible for every
    ``s >= 0``, so closed channels need no floor. This is the CG preconditioner of
    :class:`~rsfff.ff.coupled_solve._Preconditioner`, now inside the graph.
``T~ . P x``
    the shell-shell interaction applied to the **induced** multipoles ``P x = m(x) - m(0)``
    only, with the *induced-density* widths ``b_ind`` in place of the physical ``b``
    (:class:`MutualOperator`). With zero nuclear moments the one-centre terms drop out, so
    this is the existing tensor code / kernel with swapped widths and nothing else. The
    direct induction ``F0`` and the energy keep ``b``; ``T~ - T`` is a difference of
    penetration terms and vanishes exponentially with distance.
``c_k``
    optional global weights on the corrections ``x^(k) - x^(k-1)`` (OPT-style); ``c_1`` must
    stay 1 for the first-order long-range induction to be exact. Global scalars keep the
    response matrix symmetric; per-site or per-iteration weights would not.

Units are atomic, as in :mod:`rsfff.ff.coupled_solve`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .coupled_solve import (
    CoupledSystem,
    State,
    _apply_cquad,
    _coupling_grad,
    _from_polytensor,
    _grad_state,
    _incidence_adjoint,
    _incidence_apply,
    _spherical_to_poly_map,
    _to_polytensor,
    multipoles_from_state,
    state_max_abs,
)
from .electrostatics import slater_elec_tensors
from ..linalg_guard import finite_or_identity, poison
from .units import BOHR_ANG

__all__ = [
    "MutualOperator",
    "UnrolledInfo",
    "build_mutual_operator",
    "fragment_block_solve",
    "internal_solve",
    "unrolled_solve",
]


@dataclass
class UnrolledInfo:
    """What the loop can report. ``residual`` is the physical ``max |A x_K + b|`` per frame
    (an extra matvec, only when asked for)."""

    n_iter: int
    residual: torch.Tensor | None


# ---------------------------------------------------------------------------
# the mutual operator: shell-shell interaction on induced moments, learned widths
# ---------------------------------------------------------------------------

@dataclass
class MutualOperator:
    """``delta_m -> T~ delta_m`` for the induced multipoles, in one of two forms.

    **Precomputed** (torch path): ``t_point`` (shared with the physical operator) and ``t_ss``
    built from ``b_ind``, both gate-scaled ``(P, K, K)``. **On the fly** (torchff path):
    ``positions``, ``b_ind``, ``gate`` and the kernel rebuilds per matvec. ``b_ind is None``
    means the physical widths, i.e. ``T~ = T`` and the loop is the plain truncated series.
    """

    pair_index: torch.Tensor
    t_point: torch.Tensor | None = None
    t_ss: torch.Tensor | None = None
    positions: torch.Tensor | None = None     # Angstrom
    b_ind: torch.Tensor | None = None
    gate: torch.Tensor | None = None

    def __call__(self, delta_m: torch.Tensor, *, differentiable: bool = False) -> torch.Tensor:
        """``(N, K)`` field of the induced multipoles, i.e. ``dE_coupling/dm`` restricted to
        the shell-shell part. Symmetric as an operator (``H[i,j] = T^T``, ``H[j,i] = T``)."""
        if self.t_point is None:
            from .backend import slater_elec_field

            return slater_elec_field(
                self.positions, self.pair_index, self.b_ind, self.gate,
                delta_m, torch.zeros_like(delta_m), reference=differentiable,
            )
        i, j = self.pair_index[0], self.pair_index[1]
        t = self.t_point + self.t_ss
        g_i = torch.einsum("pab,pa->pb", t, delta_m[j])
        g_j = torch.einsum("pab,pb->pa", t, delta_m[i])
        out = torch.zeros_like(delta_m)
        return out.index_add(0, i, g_i).index_add(0, j, g_j)


def build_mutual_operator(
    sys: CoupledSystem,
    *,
    b_ind: torch.Tensor | None,
    positions: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> MutualOperator:
    """The mutual operator matching ``sys``'s backend path.

    On the torch path ``sys.t_point``/``sys.t_ss`` are already gate-scaled; with ``b_ind``
    the shell-shell block is rebuilt from ``positions`` (needed: the torch-path system does
    not carry them) with the induced widths and the same gate.
    """
    if sys.on_the_fly:
        return MutualOperator(
            sys.pair_index,
            positions=sys.positions,
            b_ind=sys.b if b_ind is None else b_ind,
            gate=sys.gate,
        )
    if b_ind is None:
        return MutualOperator(sys.pair_index, t_point=sys.t_point, t_ss=sys.t_ss)
    if positions is None or gate is None:
        raise ValueError("the torch path needs positions and gate to rebuild t_ss with b_ind")
    i, j = sys.pair_index[0], sys.pair_index[1]
    dr_au = (positions[j] - positions[i]) / BOHR_ANG
    r_au = dr_au.norm(dim=-1)
    _, t_ss, _, _ = slater_elec_tensors(dr_au, r_au, b_ind, sys.pair_index, max_rank=sys.max_rank)
    return MutualOperator(sys.pair_index, t_point=sys.t_point, t_ss=gate[:, None, None] * t_ss)


# ---------------------------------------------------------------------------
# the exact fragment-internal solve
# ---------------------------------------------------------------------------

def _padded_index(group: torch.Tensor, n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(local index within group, per-group count)`` without assuming contiguity."""
    counts = torch.bincount(group, minlength=n_groups)
    offsets = torch.cumsum(counts, 0) - counts
    order = torch.argsort(group, stable=True)
    local = torch.empty_like(group)
    local[order] = (
        torch.arange(group.shape[0], device=group.device) - offsets[group[order]]
    )
    return local, counts


class fragment_block_solve:
    """``v = (I + L S)^-1 rhs`` on the channel graph, one dense block per group.

    Built once per solve (the block depends on ``eta``, ``compliance`` and the graph, not on
    the fields), applied once per iteration. The inverse is formed explicitly: the blocks are
    tiny (three channels for a water) and ``torch.linalg.inv`` differentiates to any order,
    which the force loss needs. Padding rows are identity rows, so a padded channel returns
    its (zero) right-hand side.

    ``group`` is the group id of each **atom**; channels take their head atom's group (both
    ends of an intra-fragment channel share it). Pass the fragment index for the per-fragment
    block, or ``batch_idx`` for the per-frame block of the preconditioner.
    """

    def __init__(self, sys: CoupledSystem, group: torch.Tensor, n_groups: int) -> None:
        self.sys = sys
        self.nb = int(sys.bond_index.shape[1])
        if self.nb == 0:
            return
        dtype, device = sys.eta.dtype, sys.eta.device
        head, tail = sys.bond_index[0], sys.bond_index[1]
        self.bond_group = group[head]
        atom_local, atom_counts = _padded_index(group, n_groups)
        bond_local, bond_counts = _padded_index(self.bond_group, n_groups)
        n_max, nb_max = int(atom_counts.max()), int(bond_counts.max())
        self.bond_local, self.n_groups, self.nb_max = bond_local, n_groups, nb_max

        # B: (G, n_max, nb_max) incidence, +1 head / -1 tail, matching `_incidence_apply`
        B = torch.zeros(n_groups, n_max, nb_max, dtype=dtype, device=device)
        B[self.bond_group, atom_local[head], bond_local] = 1.0
        B[self.bond_group, atom_local[tail], bond_local] = -1.0
        eta_p = torch.zeros(n_groups, n_max, dtype=dtype, device=device)
        eta_p[group, atom_local] = sys.eta
        s_p = torch.zeros(n_groups, nb_max, dtype=dtype, device=device)
        s_p[self.bond_group, bond_local] = sys.compliance
        L = torch.einsum("gie,gif->gef", B, eta_p.unsqueeze(-1) * B)
        eye = torch.eye(nb_max, dtype=dtype, device=device)
        mat, ok = finite_or_identity(eye + L * s_p.unsqueeze(-2))      # non-finite groups: NaN, not a GPU fault
        self.inv = poison(torch.linalg.inv(mat), ok)                     # (I + L S)^-1

    def __call__(self, rhs: torch.Tensor) -> torch.Tensor:
        if self.nb == 0:
            return rhs
        pad = rhs.new_zeros(self.n_groups, self.nb_max)
        pad = pad.index_put((self.bond_group, self.bond_local), rhs)
        sol = torch.einsum("gef,gf->ge", self.inv, pad)
        return sol[self.bond_group, self.bond_local]


def internal_solve(
    sys: CoupledSystem,
    fields: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block: fragment_block_solve,
) -> State:
    """``Phi``: the state that zeroes ``_grad_state`` at fixed coupling fields
    ``(g_q, g_mu, g_theta)``. Inverse-free in the moment sectors."""
    g_q, g_mu, g_theta = fields
    d = sys.chi + sys.eta * sys.q0 + g_q
    v = block(-_incidence_adjoint(sys.bond_index, d))
    if sys.has_dipole:
        u = -(g_mu if sys.mu0 is not None else sys.chivec + g_mu)
    else:
        u = g_mu.new_zeros(0, 3)
    if sys.has_quad:
        w = -(g_theta if sys.quad0 is not None else sys.chiquad + g_theta)
    else:
        w = g_theta.new_zeros(0, 5)
    return v, u, w


def induced_polytensor(sys: CoupledSystem, x: State, d_map) -> torch.Tensor:
    """``P x = m(x) - m(0)``: the induced multipoles as a polytensor, linear in ``x``."""
    v, u, w = x
    n = sys.n_atoms
    dq = _incidence_apply(sys.bond_index, sys.compliance * v, n)
    dmu = (
        torch.einsum("nab,nb->na", sys.alpha, u)
        if sys.has_dipole and u.numel() else dq.new_zeros(0, 3)
    )
    dth = _apply_cquad(sys.cquad, w) if sys.has_quad and w.numel() else dq.new_zeros(0, 5)
    return _to_polytensor(dq, dmu, dth, sys.max_rank, d_map)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def unrolled_solve(
    sys: CoupledSystem,
    mutual: MutualOperator,
    *,
    n_iter: int,
    iterate_weights: torch.Tensor | None = None,
    group: torch.Tensor | None = None,
    n_groups: int | None = None,
    differentiable: bool = False,
    with_residual: bool = False,
    d_map=None,
) -> tuple[State, UnrolledInfo]:
    """``x^(0) = Phi(F0)``, then ``n_iter`` mutual iterations; returns ``(x, info)``.

    ``iterate_weights`` (``n_iter``,) scale the corrections ``x^(k) - x^(k-1)``, ``k = 1..``;
    ``None`` is all ones (the plain iterate ``x^(K)``). ``group``/``n_groups`` choose the block
    of the charge solve (fragment index recommended; default: the frame). ``differentiable``
    asks the on-the-fly kernel for its reference field, which double backward needs.
    """
    dtype, device = sys.chi.dtype, sys.chi.device
    if d_map is None:
        d_map = _spherical_to_poly_map(dtype, device)
    if group is None:
        group, n_groups = sys.batch_idx, sys.n_systems
    block = fragment_block_solve(sys, group, int(n_groups))

    # F0: the permanent field, physical damping, nuclei included. `m(0)` through the same
    # map the energy uses, so the permanent multipoles cannot drift from it.
    zero = (
        torch.zeros(block.nb, dtype=dtype, device=device),
        torch.zeros(sys.n_atoms if sys.has_dipole else 0, 3, dtype=dtype, device=device),
        torch.zeros(sys.n_atoms if sys.has_quad else 0, 5, dtype=dtype, device=device),
    )
    q0, mu0, th0 = multipoles_from_state(sys, zero, d_map)
    m0 = _to_polytensor(q0, mu0, th0, sys.max_rank, d_map)
    f0 = _from_polytensor(_coupling_grad(sys, m0, differentiable=differentiable), sys.max_rank, d_map)

    x = internal_solve(sys, f0, block)
    acc = x
    for k in range(int(n_iter)):
        dm = induced_polytensor(sys, x, d_map)
        fk = _from_polytensor(mutual(dm, differentiable=differentiable), sys.max_rank, d_map)
        x_new = internal_solve(sys, tuple(a + b for a, b in zip(f0, fk)), block)
        if iterate_weights is None:
            acc = x_new
        else:
            c = iterate_weights[k]
            acc = tuple(a + c * (n - o) for a, n, o in zip(acc, x_new, x))
        x = x_new
    x = acc

    residual = None
    if with_residual:
        with torch.no_grad():
            r = _grad_state(sys, tuple(t.detach() for t in x), d_map)   # = A x + b
            residual = state_max_abs(sys, r)
    return x, UnrolledInfo(n_iter=int(n_iter), residual=residual)
