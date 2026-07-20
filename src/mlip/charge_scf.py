"""Constrained charge SCF: minimize E(q) with per-molecule total-charge constraints.

For a ragged batch of M molecules (``batch_idx`` labels atoms), solve

    min_q E(q)   s.t.   sum_{i in m} q_i = Q_m   for every molecule m,

with a damped projected-Newton iteration, then re-attach the solution to the
autograd graph with one or two *differentiable* Newton-KKT steps -- the implicit
function theorem in "solve detached, correct differentiably" form. No solver
unrolling: first derivatives of the attached charges (forces, d mu/dR, alpha) are
exact at stationarity after one step; two steps recover exact second-order
training gradients for losses built on those first derivatives.

Structure exploited throughout: ``radius_graph`` builds only intra-molecule edges,
so E(q) = sum_m E_m(q_m) and the Hessian d2E/dq2 is block-diagonal by molecule.
The blocks are extracted with at most ``max_atoms_per_molecule`` Hessian-vector
products regardless of batch size (the local-index indicator trick), and all
per-molecule KKT systems are solved in one batched ``torch.linalg.solve``.

``energy_fn`` is a closure ``q (N,) -> per-molecule energies (M,)`` built by the
model; it must be differentiable in q (and carry the graph to parameters,
positions, and the applied field for the attach step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class RaggedIndex:
    """Precomputed indexing for per-molecule padding of a flat atom axis.

    batch_idx : (N,) molecule id per atom
    local_idx : (N,) atom index *within* its molecule, in [0, counts[m])
    counts    : (M,) atoms per molecule
    n_pad     : max(counts) -- the padded block size
    """

    batch_idx: torch.Tensor
    local_idx: torch.Tensor
    counts: torch.Tensor
    n_pad: int

    @staticmethod
    def build(batch_idx: torch.Tensor, n_systems: int) -> "RaggedIndex":
        counts = torch.bincount(batch_idx, minlength=n_systems)
        # local index = position of each atom within its molecule. Atoms of a molecule
        # are contiguous in the flat layout (see MoleculeDataset.flat_batch), so the
        # running offset per molecule gives the local index.
        offsets = torch.cumsum(counts, 0) - counts             # (M,) start of each mol
        local_idx = torch.arange(batch_idx.shape[0], device=batch_idx.device) - offsets[batch_idx]
        return RaggedIndex(
            batch_idx=batch_idx,
            local_idx=local_idx,
            counts=counts,
            n_pad=int(counts.max().item()) if counts.numel() else 0,
        )

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """(N,) -> (M, n_pad), zeros in padded slots (differentiable)."""
        out = x.new_zeros(self.counts.shape[0], self.n_pad)
        return out.index_put((self.batch_idx, self.local_idx), x)

    def unpad(self, x_pad: torch.Tensor) -> torch.Tensor:
        """(M, n_pad) -> (N,)."""
        return x_pad[self.batch_idx, self.local_idx]

    @property
    def slot_mask(self) -> torch.Tensor:
        """(M, n_pad) bool, True on real (non-padded) slots."""
        ar = torch.arange(self.n_pad, device=self.counts.device)
        return ar.unsqueeze(0) < self.counts.unsqueeze(1)


@dataclass
class ScfInfo:
    """Diagnostics from :func:`solve_charges`."""

    converged: bool
    n_iter: int
    residual: float           # final max |projected gradient| (chemical-potential eq.)
    min_tangent_eig: float    # smallest eigenvalue of the tangent-space Hessian


def projected_gradient(g: torch.Tensor, ragged: RaggedIndex) -> torch.Tensor:
    """Remove the per-molecule mean of g: the residual of chemical-potential
    equalization (dE/dq_i equal within each molecule at the constrained optimum)."""
    m = ragged.counts.shape[0]
    sums = g.new_zeros(m).index_add_(0, ragged.batch_idx, g)
    mean = sums / ragged.counts.clamp(min=1).to(g.dtype)
    return g - mean[ragged.batch_idx]


def hessian_blocks(
    g: torch.Tensor,
    q: torch.Tensor,
    ragged: RaggedIndex,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Per-molecule Hessian blocks (M, n_pad, n_pad) from <= n_pad HVPs.

    ``g`` must be ``autograd.grad(E.sum(), q, create_graph=True)``. Because the
    Hessian is block-diagonal by molecule, the HVP against the indicator of local
    index k, ``v_k[i] = [local_idx[i] == k]``, returns column k of *every* block at
    once: ``(H v_k)[i] = H_m[local_idx[i], k]``.
    """
    cols = []
    for k in range(ragged.n_pad):
        v_k = (ragged.local_idx == k).to(g.dtype)
        (hv,) = torch.autograd.grad(
            g, q, grad_outputs=v_k, create_graph=create_graph, retain_graph=True
        )
        cols.append(ragged.pad(hv))                            # (M, n_pad) = block col k
    return torch.stack(cols, dim=2)                            # (M, n_pad, n_pad)


def kkt_solve(
    H: torch.Tensor,          # (M, n_pad, n_pad) Hessian blocks (zeros in padded slots)
    g_pad: torch.Tensor,      # (M, n_pad) gradient, zeros in padded slots
    ragged: RaggedIndex,
    *,
    damping: float = 0.0,
) -> torch.Tensor:
    """Solve every molecule's Newton-KKT system in one batched call.

        [ H_m + tau*I   1 ] [dq]   [-g_m]
        [ 1^T           0 ] [nu] = [ 0  ]

    The constraint row/column contains 1 only on real atom slots, so ``1^T dq = 0``
    keeps every iterate exactly on the charge-conservation hyperplane. Padded slots
    are decoupled (identity diagonal, zero rhs -> dq = 0). Returns dq: (M, n_pad).
    """
    m, n_pad = g_pad.shape
    mask = ragged.slot_mask.to(g_pad.dtype)                    # (M, n_pad)
    eye = torch.eye(n_pad, dtype=g_pad.dtype, device=g_pad.device)

    K = g_pad.new_zeros(m, n_pad + 1, n_pad + 1)
    # Damping on real slots; identity on padded slots to keep K nonsingular.
    K[:, :n_pad, :n_pad] = H + damping * eye + torch.diag_embed(1.0 - mask)
    K[:, :n_pad, n_pad] = mask
    K[:, n_pad, :n_pad] = mask

    rhs = g_pad.new_zeros(m, n_pad + 1)
    rhs[:, :n_pad] = -g_pad
    sol = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)  # (M, n_pad+1)
    return sol[:, :n_pad]


def _tangent_min_eig(H: torch.Tensor, ragged: RaggedIndex) -> float:
    """Smallest eigenvalue of the Hessian restricted to the constraint tangent space.

    Diagnostic only (never in the autograd graph). P = diag(mask) - m m^T / n
    projects onto {sum dq = 0}; directions outside the tangent space (constraint
    normal and padded slots) are shifted up by a large constant so they cannot
    produce the minimum. Molecules with a single atom have an empty tangent space
    and are skipped (+inf).
    """
    mask = ragged.slot_mask.to(H.dtype)                        # (M, n_pad)
    n = ragged.counts.clamp(min=1).to(H.dtype)                 # (M,)
    P = torch.diag_embed(mask) - mask.unsqueeze(-1) * mask.unsqueeze(-2) / n.view(-1, 1, 1)
    php = P @ H @ P
    shift = 1.0 + php.diagonal(dim1=-2, dim2=-1).abs().sum(-1)  # Gershgorin-safe bound
    eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
    complement = eye - P
    eigs = torch.linalg.eigvalsh(php + shift.view(-1, 1, 1) * complement)  # (M, n_pad)
    min_eig = eigs[:, 0]
    min_eig = torch.where(ragged.counts > 1, min_eig, torch.full_like(min_eig, float("inf")))
    return float(min_eig.min().item())


def solve_charges(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    q_init: torch.Tensor,      # (N,) feasible start (sum per molecule == Q)
    ragged: RaggedIndex,
    *,
    max_iter: int = 30,
    tol: float = 1e-10,
    damping: float = 0.0,
    max_damping_tries: int = 8,
    max_linesearch: int = 12,
) -> tuple[torch.Tensor, ScfInfo]:
    """Damped projected-Newton minimization of E(q) on the constraint hyperplane.

    Runs detached from the ambient graph (the caller re-attaches the solution via
    :func:`attach_charges`). ``q_init`` must be feasible; every Newton step is
    tangent to the constraints, so feasibility is preserved to machine precision.
    Levenberg damping is escalated whenever the step fails to decrease the energy
    (nonconvex E(q) mid-training); a backtracking line search guards each step.
    """
    q = q_init.detach().clone()
    n_iter = 0
    residual = float("inf")
    tau = float(damping)
    H = None

    for n_iter in range(1, max_iter + 1):
        with torch.enable_grad():
            q_leaf = q.detach().requires_grad_(True)
            e = energy_fn(q_leaf)
            (g,) = torch.autograd.grad(e.sum(), q_leaf, create_graph=True)
            H = hessian_blocks(g, q_leaf, ragged, create_graph=False)
        g = g.detach()
        e_total = float(e.sum().item())
        residual = float(projected_gradient(g, ragged).abs().max().item())
        if residual < tol:
            break

        g_pad = ragged.pad(g)
        accepted = False
        for _ in range(max_damping_tries):
            dq = ragged.unpad(kkt_solve(H, g_pad, ragged, damping=tau))
            step = 1.0
            for _ in range(max_linesearch):
                with torch.no_grad():
                    e_try = float(energy_fn(q + step * dq).sum().item())
                if e_try < e_total + 1e-14 * max(1.0, abs(e_total)):
                    q = q + step * dq
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                tau *= 0.5                                     # relax damping on success
                break
            tau = max(4.0 * tau, 1e-4)                         # escalate and re-solve
        if not accepted:
            break                                              # stuck; report residual

    min_eig = _tangent_min_eig(H, ragged) if H is not None else float("nan")
    return q.detach(), ScfInfo(
        converged=residual < tol,
        n_iter=n_iter,
        residual=residual,
        min_tangent_eig=min_eig,
    )


def attach_charges(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    q_star: torch.Tensor,      # (N,) converged solution from solve_charges
    ragged: RaggedIndex,
    *,
    steps: int = 1,
    hessian_in_graph: bool = True,
) -> torch.Tensor:
    """Re-attach q* to the autograd graph via differentiable Newton-KKT steps.

    Each step applies the (differentiable) Newton map ``q <- q - K(q)^{-1} g(q)``
    restricted to the constraint tangent space. Starting from the detached q*,
    the step leaves the *value* unchanged (g is already ~0) but its graph is the
    implicit-function-theorem solution ``dq/dp = -K^{-1} d(grad E)/dp`` for every
    upstream quantity p (parameters, positions, applied field). Later steps feed
    the previous attached q back through the map, which restores exactness of
    higher derivatives. ``hessian_in_graph=False`` detaches the Hessian blocks
    (cheaper graph; first derivatives unaffected, curvature terms of higher
    derivatives dropped).
    """
    q = q_star.detach().clone().requires_grad_(True)
    for _ in range(steps):
        e = energy_fn(q)
        (g,) = torch.autograd.grad(e.sum(), q, create_graph=True)
        H = hessian_blocks(g, q, ragged, create_graph=hessian_in_graph)
        if not hessian_in_graph:
            H = H.detach()
        dq = ragged.unpad(kkt_solve(H, ragged.pad(g), ragged))
        q = q + dq
    return q
