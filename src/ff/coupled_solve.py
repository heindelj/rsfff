"""The coupled response solve: mutual polarization as one preconditioned CG.

:class:`rsfff.ff.response.FragmentResponse` minimizes each fragment's *internal* energy in
isolation. Polarization and charge transfer are what happens when the inter-fragment
electrostatic interaction is moved **inside** that functional, so the multipoles relax against
each other rather than being solved once and then interacted::

    E(M) = sum_f E_internal(M; chi, eta, chivec, alpha, chiquad, C, s)
         + sum_pairs gate_ij * [ E_point + E_pen ](M_i, M_j)

The minimizer of this is mutually polarized to all orders; the minimum *value*, differenced
against the frozen level, is what ``eda_pol`` and ``eda_ct`` label.

Adapted from ``pyCMM/cmm/polarization.py`` -- preconditioned CG with the uncoupled analytic
solution as the preconditioner -- with four changes:

* **SQE instead of constrained EEM.** pyCMM enforces per-group charge conservation with
  Lagrange multipliers, so its system is an indefinite KKT saddle point that CG has no
  guarantee on. SQE conserves charge by *construction* (``q = q0 + B p``), so the system here
  is genuinely symmetric PSD. Do not reintroduce a multiplier block.
* **Batched.** pyCMM solves one system. Every CG scalar here -- ``rz``, ``pAp``, the step and
  the convergence test -- is **per frame**, with converged frames frozen rather than the whole
  batch iterating to the slowest. Getting this wrong makes the result silently depend on which
  other frames shared the minibatch, which ``tests/test_ff_coupled_solve.py`` checks directly.
* **No solver-state extrapolation.** pyCMM carries a guess across MD steps; training frames
  are independent, so there is nothing to extrapolate from.
* **An adjoint backward** -- see below.

The change of variables
-----------------------
The coupled functional contains ``1/2 mu^T alpha^-1 mu``, and this codebase deliberately never
inverts ``alpha``: :func:`rsfff.mlip.sqe.atomic_dipole_energy` evaluates the quadratic form at
its minimum precisely to avoid it. So rescale, exactly as SQE rescales ``p = S v``::

    q = q0 + B S v        mu = alpha u        Theta = c w        x = (v, u, w)

Every block of the **matvec** is then polynomial in ``(s, alpha, c)``, no inverse is formed,
and the unpolarizable limits ``alpha -> 0``, ``c -> 0``, ``s -> 0`` are well-conditioned rather
than singular -- the same argument :mod:`rsfff.mlip.sqe` records for choosing ``S`` over
``S^(1/2)``, and load-bearing for force training where a channel is closed. In these variables
the charge block is ``S L S + S``, which is **symmetric**; the asymmetric ``L S + I`` that
``sqe_solve`` solves is that same system left-divided by ``S``, and CG needs the symmetric one.

The **preconditioner may invert freely**, including ``alpha^-1`` behind its ``psd_floor``,
because it runs entirely under ``no_grad`` and never enters a gradient. A poor preconditioner
costs iterations, never correctness. That is the rule worth remembering: *inverse-free in the
differentiated matvec, inverses allowed in the undifferentiated preconditioner.*

One function, not two
---------------------
The matvec is **derived from** the energy rather than transcribed alongside it::

    A x = grad_E(x) - grad_E(0)          b = grad_E(0)

Both come from :func:`_grad_state`, which is the gradient of the very expression
:func:`coupled_energy` reports. A sign error in the interaction tensor therefore cannot make
the solver and the energy disagree -- it moves both together, and
``tests/test_ff_coupled_solve.py`` pins ``_grad_state`` against autograd of the energy. This is
why no dense Hessian is assembled anywhere: there is nothing to assemble it *from* that is not
already the gradient.

Gradients: why the adjoint is not optional
------------------------------------------
pyCMM solves under ``no_grad`` and recovers forces from stationarity: with ``A x = -b`` and
``E = 1/2 x^T A x + b^T x``, the derivative at *fixed* ``x`` is already the total derivative,
because the missing term carries the factor ``(A x + b) = 0``. That still holds here and it
covers the energy for free.

It does **not** cover everything downstream. The electrostatic environment features are built
from the converged multipoles and feed the bond correction, which is not variational in ``M*``
-- exactly the failure ``docs/range_separated_mlip.md`` §6.1 names ("detaching ``M*`` yields
analytic forces that do not equal ``-dE/dR``"). So :class:`_CoupledSolve` implements §6.2:
forward runs CG under ``no_grad``; backward receives ``dL/dx``, solves the adjoint
``A lambda = dL/dx`` with the same CG, and contributes ``-lambda^T dR/dtheta`` where
``R(x, theta) = grad_E(x; theta)`` is the residual whose root the forward found.

Two things fall out of that arrangement. Memory is flat in the iteration count, unlike an
unrolled loop. And when nothing non-variational consumes ``x``, ``dL/dx`` is zero to CG
tolerance, the adjoint solve exits immediately, and the cost is genuinely zero -- the
Hellmann-Feynman case is not special-cased, it is just the limit.

Units are atomic units throughout, matching :mod:`rsfff.ff.multipole`: bohr, e, Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mlip.sqe import _local_index
from .multipole import build_polytensor, spherical_to_cartesian_quadrupole

#: A CG step is refused when ``p^T A p`` is not positive: the functional has lost positive
#: definiteness, which for a polarization model means it is running away. Reported rather than
#: clamped -- see :class:`CoupledInfo`.
_PD_EPS = 0.0


@dataclass
class CoupledSystem:
    """Everything the matvec needs, with the differentiable tensors kept separable.

    The four interaction tensors arrive **already multiplied by the range-separation gate**,
    so this module never sees a switching function and the caller keeps sole ownership of what
    "switched on" means. They are the same objects
    :func:`rsfff.ff.electrostatics.slater_elec_pair_energy` contracts, in the same
    ``m_j^T T m_i`` convention:

    t_point : (P, K, K)  undamped, the 1/r tail
    t_ss    : (P, K, K)  shell-shell, two-center Slater damped
    t_1c_i  : (P, K, K)  nucleus j against shell i, one-center damped by ``b_i``
    t_1c_j  : (P, K, K)  shell j against nucleus i, one-center damped by ``b_j``

    ``alpha``/``cquad`` being ``None`` removes that sector entirely; its state component is a
    zero-size tensor rather than a special case in every loop.
    """

    n_systems: int
    n_atoms: int
    batch_idx: torch.Tensor        # (N,) frame of each atom
    bond_index: torch.Tensor       # (2, Nb) channel graph
    bond_batch: torch.Tensor       # (Nb,) frame of each channel
    pair_index: torch.Tensor       # (2, P)
    max_rank: int

    # --- differentiable ---------------------------------------------------------------
    chi: torch.Tensor              # (N,)
    eta: torch.Tensor              # (N,) strictly positive
    q0: torch.Tensor               # (N,) baseline charges, sum per fragment = formal charge
    compliance: torch.Tensor       # (Nb,) >= 0
    chivec: torch.Tensor | None    # (N, 3)
    alpha: torch.Tensor | None     # (N, 3, 3) PSD
    chiquad: torch.Tensor | None   # (N, 5)
    #: (N,) isotropic ``c I5``, or (N, 5, 5) PSD -- the axially anisotropic form of
    #: :class:`rsfff.mlip.response_heads.AxialQuadrupolePolarizabilityHead`. Both are
    #: applied, never inverted, so the rescaling ``Theta = C w`` works unchanged.
    cquad: torch.Tensor | None     # (N,) or (N, 5, 5), PSD
    t_point: torch.Tensor          # (P, K, K), gate-scaled
    t_ss: torch.Tensor             # (P, K, K), gate-scaled
    t_1c_i: torch.Tensor           # (P, K, K), gate-scaled
    t_1c_j: torch.Tensor           # (P, K, K), gate-scaled
    m_nuc: torch.Tensor            # (N, K) point nuclear multipoles, constant in x

    #: **The two parameterizations of the same functional.** Exactly one of ``chivec`` and
    #: ``mu0`` is ever non-``None``, and likewise for ``chiquad``/``quad0``:
    #:
    #:   chivec  ->  E = chivec.mu + 1/2 mu^T alpha^-1 mu          minimized at mu = -alpha chivec
    #:   mu0     ->  E = 1/2 (mu - mu0)^T alpha^-1 (mu - mu0)      minimized at mu = mu0
    #:
    #: They are related by ``mu0 = -alpha chivec``, so this is a change of variables and not a
    #: different model. What it buys is conditioning. Under ``chivec`` the permanent dipole is
    #: a *product* of two heads, so the multipole labels and the polarizability labels pull on
    #: both at once and an error in ``alpha`` corrupts ``mu``; under ``mu0`` each label pins one
    #: head. It also separates two statements that ``chivec`` conflates: ``alpha -> 0`` there
    #: forces ``mu -> 0``, i.e. an unpolarizable atom is forced to be non-polar, whereas here it
    #: correctly leaves a rigid ``mu0``.
    #:
    #: Both forms leave ``A`` identical -- the linear term moves into ``b``, not the operator --
    #: so the preconditioner, the CG and the adjoint are untouched, and the functional stays
    #: quadratic.
    mu0: torch.Tensor | None = None     # (N, 3)
    quad0: torch.Tensor | None = None   # (N, 5) spherical

    @property
    def has_dipole(self) -> bool:
        return self.alpha is not None

    @property
    def has_quad(self) -> bool:
        return self.cquad is not None


@dataclass
class CoupledInfo:
    """Solver diagnostics. ``converged`` and ``pd_fail`` are per frame."""

    n_iter: int
    converged: torch.Tensor        # (M,) bool
    pd_fail: torch.Tensor          # (M,) bool: p^T A p <= 0 at some iteration
    residual: torch.Tensor         # (M,) final max|r| per frame


# ---------------------------------------------------------------------------
# state algebra: x = (v, u, w) as three ragged tensors
# ---------------------------------------------------------------------------

State = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def zero_state(sys: CoupledSystem, dtype, device) -> State:
    nb = int(sys.bond_index.shape[1])
    n = sys.n_atoms
    return (
        torch.zeros(nb, dtype=dtype, device=device),
        torch.zeros(n if sys.has_dipole else 0, 3, dtype=dtype, device=device),
        torch.zeros(n if sys.has_quad else 0, 5, dtype=dtype, device=device),
    )


def state_dot(sys: CoupledSystem, a: State, b: State) -> torch.Tensor:
    """Per-frame inner product: (M,). The reason CG here is not pyCMM's CG."""
    out = a[0].new_zeros(sys.n_systems)
    if a[0].numel():
        out = out.index_add(0, sys.bond_batch, a[0] * b[0])
    for k in (1, 2):
        if a[k].numel():
            out = out.index_add(0, sys.batch_idx, (a[k] * b[k]).sum(-1))
    return out


def state_axpy(sys: CoupledSystem, a: torch.Tensor, x: State, y: State) -> State:
    """``y + a * x`` with a **per-frame** scalar ``a`` (M,)."""
    return (
        y[0] + a[sys.bond_batch] * x[0] if x[0].numel() else y[0],
        y[1] + a[sys.batch_idx].unsqueeze(-1) * x[1] if x[1].numel() else y[1],
        y[2] + a[sys.batch_idx].unsqueeze(-1) * x[2] if x[2].numel() else y[2],
    )


def state_max_abs(sys: CoupledSystem, x: State) -> torch.Tensor:
    """Per-frame max |component|: (M,), the CG convergence measure."""
    out = x[0].new_zeros(sys.n_systems)
    if x[0].numel():
        out = out.scatter_reduce(0, sys.bond_batch, x[0].abs(), "amax", include_self=True)
    for k in (1, 2):
        if x[k].numel():
            out = out.scatter_reduce(
                0, sys.batch_idx, x[k].abs().amax(-1), "amax", include_self=True
            )
    return out


# ---------------------------------------------------------------------------
# the multipole map: x -> M, and its adjoint
# ---------------------------------------------------------------------------

def _spherical_to_poly_map(dtype, device) -> torch.Tensor:
    """(5, 6): spherical quadrupole -> the polytensor's six weighted Cartesian slots.

    Derived by pushing the identity through the *existing*
    :func:`rsfff.ff.multipole.spherical_to_cartesian_quadrupole` and
    :func:`~rsfff.ff.multipole.build_polytensor` rather than transcribing their composition,
    so the ``[1/3, 2/3, ...]`` polytensor weights and the spherical convention can never drift
    from what the interaction tensor is contracted against.
    """
    eye = torch.eye(5, dtype=dtype, device=device)
    cart = spherical_to_cartesian_quadrupole(eye)                 # (5, 3, 3)
    poly = build_polytensor(
        torch.zeros(5, dtype=dtype, device=device), None, cart, max_rank=2
    )                                                             # (5, 10)
    return poly[:, 4:].contiguous()


def _to_polytensor(q, mu, theta_s, max_rank, d_map) -> torch.Tensor:
    if max_rank == 0:
        return q.unsqueeze(-1)
    blocks = [q.unsqueeze(-1), mu if mu.numel() else q.new_zeros(q.shape[0], 3)]
    if max_rank >= 2:
        blocks.append(
            theta_s @ d_map if theta_s.numel() else q.new_zeros(q.shape[0], 6)
        )
    return torch.cat(blocks, dim=-1)


def _from_polytensor(g, max_rank, d_map):
    """Adjoint of :func:`_to_polytensor`: ``(dq, dmu, dtheta_s)``."""
    q = g[:, 0]
    mu = g[:, 1:4] if max_rank >= 1 else g.new_zeros(g.shape[0], 3)
    theta = g[:, 4:] @ d_map.t() if max_rank >= 2 else g.new_zeros(g.shape[0], 5)
    return q, mu, theta


def _apply_cquad(cquad: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``C w`` for either shape of quadrupole polarizability: ``(N,)`` or ``(N, 5, 5)``.

    The isotropic and axially anisotropic forms differ only here. Both *apply* ``C``; neither
    inverts it, which is what keeps the ``c -> 0`` limit well conditioned -- the same reason
    the dipole sector is rescaled as ``mu = alpha u``.
    """
    if cquad.dim() == 1:
        return cquad.unsqueeze(-1) * w
    return torch.einsum("nab,nb->na", cquad, w)


def _incidence_apply(bond_index, p, n_atoms) -> torch.Tensor:
    """``B p``: +1 on the head atom of a channel, -1 on the tail (matching ``sqe_solve``)."""
    out = p.new_zeros(n_atoms)
    if p.numel():
        out = out.index_add(0, bond_index[0], p).index_add(0, bond_index[1], -p)
    return out


def _incidence_adjoint(bond_index, g) -> torch.Tensor:
    """``B^T g``."""
    if bond_index.shape[1] == 0:
        return g.new_zeros(0)
    return g[bond_index[0]] - g[bond_index[1]]


def multipoles_from_state(sys: CoupledSystem, x: State, d_map=None):
    """``(q, mu, theta_s)`` in physical units from the rescaled state.

    ``q = q0 + B S v``, ``mu = alpha u``, ``Theta = c w`` -- the change of variables the module
    docstring justifies. Every one applies a parameter, never its inverse.
    """
    v, u, w = x
    n = sys.n_atoms
    p = sys.compliance * v
    q = sys.q0 + _incidence_apply(sys.bond_index, p, sys.n_atoms)
    # **A sector's response and its permanent multipole are independent.** `has_dipole` /
    # `has_quad` say only whether the *state* carries a variable for that sector; whether the
    # sector contributes a multipole at all is `mu0`/`quad0`. Deciding the width from the state
    # was a trap: with `cquad = None` (quadrupole response off, permanent quadrupoles kept)
    # `theta` came out `(0, 5)`, the `theta.numel()` guard below then skipped `quad0`, and
    # `_to_polytensor` zero-filled the block -- so the frozen level carried permanent
    # quadrupoles and the coupled level silently did not. Sizing by the permanent multipole
    # instead makes "no polarizability" mean a rigid moment, which is what it should mean.
    if sys.has_dipole and u.numel():
        mu = torch.einsum("nab,nb->na", sys.alpha, u)
    else:
        mu = u.new_zeros(n if sys.mu0 is not None else 0, 3)
    if sys.has_quad and w.numel():
        theta = _apply_cquad(sys.cquad, w)
    else:
        theta = w.new_zeros(n if sys.quad0 is not None else 0, 5)
    # Under the direct parameterization the state carries only the *induced* part, so the
    # permanent multipole is added back here -- and nowhere else, which is what keeps the
    # quadratic block `1/2 u^T alpha u` (and hence `A`) exactly as it was.
    if sys.mu0 is not None:
        mu = mu + sys.mu0
    if sys.quad0 is not None:
        theta = theta + sys.quad0
    return q, mu, theta


def _coupling_grad(sys: CoupledSystem, m: torch.Tensor) -> torch.Tensor:
    """``dE_coupling/dM`` at multipoles ``m``: (N, K).

    The coupling energy is exactly what
    :func:`rsfff.ff.electrostatics.slater_elec_pair_energy` sums, with ``m_shell = m - m_nuc``
    substituted so the whole thing is an explicit quadratic in ``m``::

        E = sum_p [ m_j^T T_pt m_i
                  + (m_j - n_j)^T T_ss (m_i - n_i)
                  + n_j^T T_1c_i (m_i - n_i)
                  + (m_j - n_j)^T T_1c_j n_i ]

    The tensors are **not** symmetric -- ``T[0,1] = -tx`` against ``T[1,0] = +tx`` is the
    charge-dipole convention of ``m_j^T T m_i`` with ``dr = r_j - r_i`` -- so the two sides of
    a pair contract opposite index orders. The assembled operator *is* symmetric
    (``H[i,j] = T^T`` and ``H[j,i] = T``), which is what CG requires.
    """
    i, j = sys.pair_index[0], sys.pair_index[1]
    m_i, m_j = m[i], m[j]
    n_i, n_j = sys.m_nuc[i], sys.m_nuc[j]
    s_i, s_j = m_i - n_i, m_j - n_j

    # d/dm_i: contract the first tensor index (T^T applied to the j side).
    g_i = (
        torch.einsum("pab,pa->pb", sys.t_point, m_j)
        + torch.einsum("pab,pa->pb", sys.t_ss, s_j)
        + torch.einsum("pab,pa->pb", sys.t_1c_i, n_j)
    )
    # d/dm_j: contract the second index (T applied to the i side).
    g_j = (
        torch.einsum("pab,pb->pa", sys.t_point, m_i)
        + torch.einsum("pab,pb->pa", sys.t_ss, s_i)
        + torch.einsum("pab,pb->pa", sys.t_1c_j, n_i)
    )
    out = torch.zeros_like(m)
    return out.index_add(0, i, g_i).index_add(0, j, g_j)


def _grad_state(sys: CoupledSystem, x: State, d_map) -> State:
    """``dE/d(v, u, w)`` at state ``x``. The single source of both ``A`` and ``b``.

    ``A x = _grad_state(x) - _grad_state(0)`` and ``b = _grad_state(0)``, because ``E`` is
    quadratic so this is affine. Deriving the operator from the energy rather than writing it
    out separately is what makes a sign error in the interaction tensor impossible to hide:
    it moves the solve and the reported energy together.
    """
    q, mu, theta = multipoles_from_state(sys, x, d_map)

    m = _to_polytensor(q, mu, theta, sys.max_rank, d_map)
    g_q, g_mu, g_theta = _from_polytensor(
        _coupling_grad(sys, m), sys.max_rank, d_map
    )

    # charge sector: chi q + 1/2 eta q^2 + 1/2 s v^2, pulled back through q = q0 + B S v
    d_q = sys.chi + sys.eta * q + g_q
    g_v = sys.compliance * (
        _incidence_adjoint(sys.bond_index, d_q) + x[0]
    )

    # dipole sector, pulled back through mu = mu0 + alpha u (mu0 = 0 in the chivec form).
    #
    #   chivec form:  E = chivec.mu + 1/2 u^T alpha u   ->  dE/du = alpha (chivec + g_mu) + alpha u
    #   mu0 form:     E = 1/2 u^T alpha u               ->  dE/du = alpha (g_mu + u)
    #
    # The `alpha u` term is written out rather than reusing `mu`, because under the direct
    # parameterization `mu` now carries `mu0` too and adding it here would be a linear term
    # that does not exist -- the whole point of that form is that there is none.
    if sys.has_dipole and x[1].numel():
        drive = g_mu if sys.mu0 is not None else sys.chivec + g_mu
        g_u = torch.einsum("nab,nb->na", sys.alpha, drive + x[1])
    else:
        g_u = x[1]

    # quadrupole sector: the same, pulled back through Theta = quad0 + C w.
    if sys.has_quad and x[2].numel():
        drive_q = g_theta if sys.quad0 is not None else sys.chiquad + g_theta
        g_w = _apply_cquad(sys.cquad, drive_q + x[2])
    else:
        g_w = x[2]

    return g_v, g_u, g_w


def coupled_energy(sys: CoupledSystem, x: State, d_map=None) -> torch.Tensor:
    """``(E_internal (M,), E_pair (P,))`` at state ``x``, in Hartree.

    The internal part mirrors :class:`rsfff.ff.response.FragmentResponse` term for term and
    reduces to it exactly when the coupling is absent: at the uncoupled minimum ``u = -chivec``
    gives ``mu = -alpha chivec`` and ``chivec.mu + 1/2 u.mu = -1/2 chivec^T alpha chivec``,
    which is what :func:`rsfff.mlip.sqe.atomic_dipole_energy` returns. Pooled by **frame**, not
    by fragment: with the coupling on, the energy is not decomposable per fragment.

    The pair part is deliberately *not* recomputed here -- see
    :func:`rsfff.ff.polarization.coupled_response`, which calls the frozen channel's own
    :func:`~rsfff.ff.electrostatics.slater_elec_pair_energy` on the converged multipoles, so
    the polarized and frozen electrostatics are literally the same function of their inputs.
    """
    if d_map is None:
        d_map = _spherical_to_poly_map(x[0].dtype, x[0].device)
    q, mu, theta = multipoles_from_state(sys, x, d_map)

    e_atom = sys.chi * q + 0.5 * sys.eta * q * q
    if sys.has_dipole and x[1].numel():
        # `1/2 u . (alpha u)`, i.e. the *induced* part only. Under the direct parameterization
        # `mu` includes `mu0`, so `1/2 u . mu` would smuggle in a linear term.
        mu_ind = mu if sys.mu0 is None else mu - sys.mu0
        e_atom = e_atom + 0.5 * (x[1] * mu_ind).sum(-1)
        if sys.mu0 is None:
            e_atom = e_atom + (sys.chivec * mu).sum(-1)
    if sys.has_quad and x[2].numel():
        th_ind = theta if sys.quad0 is None else theta - sys.quad0
        e_atom = e_atom + 0.5 * (x[2] * th_ind).sum(-1)
        if sys.quad0 is None:
            e_atom = e_atom + (sys.chiquad * theta).sum(-1)

    energy = e_atom.new_zeros(sys.n_systems).index_add(0, sys.batch_idx, e_atom)
    if x[0].numel():
        # 1/2 kappa p^2 = 1/2 s v^2 in the rescaled variable; 1/s is never formed.
        e_chan = 0.5 * sys.compliance * x[0] * x[0]
        energy = energy + e_chan.new_zeros(sys.n_systems).index_add(
            0, sys.bond_batch, e_chan
        )
    return energy


# ---------------------------------------------------------------------------
# preconditioner: the uncoupled analytic solution
# ---------------------------------------------------------------------------

class _Preconditioner:
    """``M^-1`` from the **uncoupled** problem -- pyCMM's ``direct_polarization_guess``.

    Built once per solve, entirely under ``no_grad``, so it is free to invert anything: a poor
    preconditioner costs iterations and never correctness. That is what lets the charge block
    be factorized densely and ``alpha`` be inverted behind a floor, while the differentiated
    matvec stays strictly inverse-free.

    The charge block ``S L S + S`` is assembled per **frame** rather than per fragment. When
    the channel graph is intra-fragment only it is block diagonal by fragment anyway, so the
    frame-level factorization gives the same answer and costs nothing at cluster sizes; when
    inter-fragment channels are open (the CT level) the frame is the correct block.

    What is left for CG is the electrostatic pair coupling. Its intra-fragment part is gated
    off by the range separation, so in practice CG is resolving inter-fragment coupling only,
    which is weak -- expect single-digit to low-teens iterations.
    """

    def __init__(self, sys: CoupledSystem, floor: float = 1.0e-9) -> None:
        dtype, device = sys.chi.dtype, sys.chi.device
        n_sys, n_atoms = sys.n_systems, sys.n_atoms
        n_bonds = int(sys.bond_index.shape[1])

        self.sys = sys
        self.lu = None
        if n_bonds:
            atom_local, atom_counts = _local_index(sys.batch_idx, n_sys)
            bond_local, bond_counts = _local_index(sys.bond_batch, n_sys)
            n_max = int(atom_counts.max()) if n_atoms else 0
            nb_max = int(bond_counts.max())

            eta_p = torch.ones(n_sys, n_max, dtype=dtype, device=device)
            eta_p[sys.batch_idx, atom_local] = sys.eta
            s_p = torch.zeros(n_sys, nb_max, dtype=dtype, device=device)
            s_p[sys.bond_batch, bond_local] = sys.compliance

            B = torch.zeros(n_sys, n_max, nb_max, dtype=dtype, device=device)
            B[sys.bond_batch, atom_local[sys.bond_index[0]], bond_local] = 1.0
            B[sys.bond_batch, atom_local[sys.bond_index[1]], bond_local] = -1.0

            L = torch.einsum("mie,mif->mef", B, eta_p.unsqueeze(-1) * B)
            eye = torch.eye(nb_max, dtype=dtype, device=device)
            # S L S + S, plus a floor so a closed channel's zero row stays invertible. The
            # floor is invisible in the answer: a closed channel has a zero residual too.
            mat = (
                s_p.unsqueeze(-1) * L * s_p.unsqueeze(-2)
                + s_p.unsqueeze(-1) * eye
                + floor * eye
            )
            self.lu = torch.linalg.lu_factor(mat)
            self.bond_local, self.nb_max = bond_local, nb_max

        self.alpha_inv = None
        if sys.has_dipole:
            eye3 = torch.eye(3, dtype=dtype, device=device)
            self.alpha_inv = torch.linalg.inv(sys.alpha + floor * eye3)
        self.cquad_inv = None
        if sys.has_quad:
            if sys.cquad.dim() == 1:
                self.cquad_inv = 1.0 / (sys.cquad + floor)
            else:
                eye5 = torch.eye(5, dtype=dtype, device=device)
                self.cquad_inv = torch.linalg.inv(sys.cquad + floor * eye5)

    def __call__(self, r: State) -> State:
        sys = self.sys
        z_v = r[0]
        if self.lu is not None and r[0].numel():
            rhs = torch.zeros(
                sys.n_systems, self.nb_max, 1, dtype=r[0].dtype, device=r[0].device
            )
            rhs[sys.bond_batch, self.bond_local, 0] = r[0]
            sol = torch.linalg.lu_solve(*self.lu, rhs)
            z_v = sol[sys.bond_batch, self.bond_local, 0]
        z_u = (
            torch.einsum("nab,nb->na", self.alpha_inv, r[1])
            if self.alpha_inv is not None and r[1].numel() else r[1]
        )
        z_w = (
            _apply_cquad(self.cquad_inv, r[2])
            if self.cquad_inv is not None and r[2].numel() else r[2]
        )
        return z_v, z_u, z_w


# ---------------------------------------------------------------------------
# preconditioned CG
# ---------------------------------------------------------------------------

def pcg(
    sys: CoupledSystem,
    rhs: State | None = None,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    maxiter: int = 200,
    d_map=None,
) -> tuple[State, CoupledInfo]:
    """Solve ``A x = rhs`` (default ``rhs = -b``, i.e. minimize the functional).

    Every scalar is per frame and converged frames are frozen by zeroing their step, so a
    frame's answer does not depend on which others shared the minibatch.
    """
    dtype, device = sys.chi.dtype, sys.chi.device
    if d_map is None:
        d_map = _spherical_to_poly_map(dtype, device)

    zero = zero_state(sys, dtype, device)
    g0 = _grad_state(sys, zero, d_map)              # = b

    def a_mul(p: State) -> State:
        g = _grad_state(sys, p, d_map)
        return (g[0] - g0[0], g[1] - g0[1], g[2] - g0[2])

    if rhs is None:
        rhs = (-g0[0], -g0[1], -g0[2])

    pre = _Preconditioner(sys)
    x = zero
    r = rhs                                          # r = rhs - A*0
    z = pre(r)
    p = z
    rz = state_dot(sys, r, z)
    target = torch.clamp(rtol * state_max_abs(sys, rhs), min=atol)

    converged = state_max_abs(sys, r) <= target
    pd_fail = torch.zeros(sys.n_systems, dtype=torch.bool, device=device)
    n_iter = 0
    for n_iter in range(1, int(maxiter) + 1):
        if bool(converged.all()):
            break
        ap = a_mul(p)
        p_ap = state_dot(sys, p, ap)
        bad = (p_ap <= _PD_EPS) & ~converged
        pd_fail = pd_fail | bad
        active = ~converged & ~bad
        step = torch.where(active, rz / torch.where(p_ap == 0, torch.ones_like(p_ap), p_ap),
                           torch.zeros_like(rz))
        x = state_axpy(sys, step, p, x)
        r = state_axpy(sys, -step, ap, r)
        converged = converged | bad | (state_max_abs(sys, r) <= target)
        if bool(converged.all()):
            break
        z = pre(r)
        rz_new = state_dot(sys, r, z)
        beta = torch.where(rz == 0, torch.zeros_like(rz), rz_new / rz)
        p = state_axpy(sys, beta, p, z)
        rz = rz_new

    residual = state_max_abs(sys, r)
    return x, CoupledInfo(
        n_iter=n_iter,
        converged=converged & ~pd_fail,
        pd_fail=pd_fail,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# the differentiable entry point
# ---------------------------------------------------------------------------

#: The differentiable leaves handed to :class:`_CoupledSolve`. ``mu0``/``quad0`` belong here
#: for the same reason ``chivec``/``chiquad`` do: they enter the residual, so the adjoint has to
#: be able to differentiate it with respect to them. Leaving them out would silently zero the
#: permanent multipoles' gradient.
_PARAM_FIELDS = (
    "chi", "eta", "q0", "compliance", "chivec", "alpha", "chiquad", "cquad",
    "t_point", "t_ss", "t_1c_i", "t_1c_j", "m_nuc", "mu0", "quad0",
)


def _rebuild(sys: CoupledSystem, params) -> CoupledSystem:
    from dataclasses import replace
    return replace(sys, **dict(zip(_PARAM_FIELDS, params)))


class _CoupledSolve(torch.autograd.Function):
    """Forward: CG under ``no_grad``. Backward: the adjoint of §6.2.

    ``R(x, theta) = grad_E(x; theta)`` is the residual the forward drove to zero, so the
    implicit function theorem gives ``dx/dtheta = -A^-1 dR/dtheta`` and hence
    ``dL/dtheta = -lambda^T dR/dtheta`` with ``A lambda = dL/dx``. ``A`` is symmetric, so the
    adjoint reuses the same CG and the same preconditioner.

    When nothing non-variational consumes ``x``, ``dL/dx`` is zero to CG tolerance: the adjoint
    exits on the first convergence test and contributes nothing, leaving exactly the
    Hellmann-Feynman gradient that the caller gets from differentiating the energy at fixed
    ``x``. That case is not special-cased; it is the limit.
    """

    @staticmethod
    def forward(ctx, sys, cg_kwargs, info_out, *params):
        with torch.no_grad():
            live = _rebuild(sys, params)
            x, info = pcg(live, **cg_kwargs)
        if info_out is not None:
            info_out.append(info)
        ctx.sys = sys
        ctx.cg_kwargs = cg_kwargs
        ctx.n_params = len(params)
        ctx.save_for_backward(*x, *params)
        ctx.info = info
        return x[0], x[1], x[2], torch.tensor(
            [info.n_iter], dtype=torch.int64, device=x[0].device
        )

    @staticmethod
    def backward(ctx, g_v, g_u, g_w, _g_iter):
        saved = ctx.saved_tensors
        x = tuple(saved[:3])
        params = saved[3:]
        sys = ctx.sys

        grad_x = (
            torch.zeros_like(x[0]) if g_v is None else g_v,
            torch.zeros_like(x[1]) if g_u is None else g_u,
            torch.zeros_like(x[2]) if g_w is None else g_w,
        )
        with torch.no_grad():
            live = _rebuild(sys, params)
            if max(float(t.abs().max()) if t.numel() else 0.0 for t in grad_x) == 0.0:
                lam = zero_state(live, x[0].dtype, x[0].device)
            else:
                lam, _ = pcg(live, rhs=grad_x, **ctx.cg_kwargs)

        # -lambda^T dR/dtheta, with R evaluated at the *fixed* converged x.
        with torch.enable_grad():
            leaves = [
                None if p is None else p.detach().requires_grad_(True) for p in params
            ]
            live2 = _rebuild(sys, leaves)
            d_map = _spherical_to_poly_map(x[0].dtype, x[0].device)
            res = _grad_state(live2, tuple(t.detach() for t in x), d_map)
            needed = [p for p in leaves if p is not None]
            grads = torch.autograd.grad(
                [t for t in res if t.numel()],
                needed,
                grad_outputs=[-l for l, t in zip(lam, res) if t.numel()],
                allow_unused=True,
            )
        out, k = [], 0
        for p in leaves:
            if p is None:
                out.append(None)
            else:
                out.append(grads[k])
                k += 1
        return (None, None, None, *out)


def coupled_solve(
    sys: CoupledSystem,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    maxiter: int = 200,
    info_out: list | None = None,
) -> tuple[State, int]:
    """Minimize the coupled functional. Differentiable through the adjoint.

    Returns the rescaled state and the iteration count. Feed the state to
    :func:`multipoles_from_state` for ``(q, mu, Theta)`` and to :func:`coupled_energy` for the
    internal part of the minimum. Pass ``info_out=[]`` to collect the :class:`CoupledInfo` of
    the forward solve -- the per-frame convergence and positive-definiteness flags, which are
    what to log during training and cannot be recovered afterwards.
    """
    params = [getattr(sys, f) for f in _PARAM_FIELDS]
    kwargs = dict(rtol=rtol, atol=atol, maxiter=maxiter)
    v, u, w, n_iter = _CoupledSolve.apply(sys, kwargs, info_out, *params)
    return (v, u, w), int(n_iter.item())


def coupled_solve_dense(sys: CoupledSystem) -> State:
    """Direct solve by materializing ``A`` one basis vector at a time. **Tests only.**

    The oracle the matrix-free path is checked against. It is ``O(n^2)`` matvecs and exists
    purely so a CG bug -- a mis-broadcast per-frame scalar, a frame frozen too early -- cannot
    masquerade as a physics result. It shares :func:`_grad_state` with the real solver, so it
    does *not* independently check the operator; that is what the autograd test of
    ``_grad_state`` against :func:`coupled_energy` is for.
    """
    dtype, device = sys.chi.dtype, sys.chi.device
    d_map = _spherical_to_poly_map(dtype, device)
    zero = zero_state(sys, dtype, device)
    g0 = _grad_state(sys, zero, d_map)
    sizes = [t.numel() for t in zero]
    n = sum(sizes)

    def unflatten(flat):
        parts = torch.split(flat, sizes)
        return parts[0], parts[1].reshape(-1, 3), parts[2].reshape(-1, 5)

    def flatten(state):
        return torch.cat([t.reshape(-1) for t in state])

    cols = []
    eye = torch.eye(n, dtype=dtype, device=device)
    for k in range(n):
        g = _grad_state(sys, unflatten(eye[k]), d_map)
        cols.append(flatten(g) - flatten(g0))
    a = torch.stack(cols, dim=1)
    x = torch.linalg.solve(a, -flatten(g0))
    return unflatten(x)


__all__ = [
    "CoupledInfo",
    "CoupledSystem",
    "State",
    "coupled_energy",
    "coupled_solve",
    "coupled_solve_dense",
    "multipoles_from_state",
    "pcg",
    "zero_state",
]
