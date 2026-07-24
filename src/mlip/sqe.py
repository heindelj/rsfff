"""Split-Charge Equilibration: charge conservation as a coordinate system.

Conventional EEM (see :mod:`rsfff.mlip.eem`) minimizes a quadratic ``E({q_i})`` subject to a
per-fragment total-charge constraint enforced by a Lagrange multiplier. SQE
(docs/mixture_of_diabatic_embeddings.md §2.4) removes the constraint by changing variables:
charge exists only as antisymmetric transfers along the edges of an explicit **channel graph**,

    q_i = q_i^(0) + sum_{j in N(i)} p_ij,        p_ij = -p_ji

so every transfer adds to one atom exactly what it removes from another. The total charge of
each connected component is therefore conserved *identically, for arbitrary* ``{p_ij}`` -- no
constraints, no multipliers, and the minimization over the transfers is unconstrained.

Phase-1 functional (diagonal hardness; the damped Coulomb ``J_ij`` of §2.4.2 arrives with
inter-fragment interactions)::

    E(p) = sum_i [ chi_i q_i + 1/2 eta_i q_i^2 - q_i (r_i . F) ] + 1/2 sum_e kappa_e p_e^2

Writing ``q = q^(0) + B p`` with the incidence matrix ``B`` and differentiating gives one sparse
linear system per configuration,

    M p = -b,   M = B^T H B + K,   b = B^T (chi + eta*q^(0) - Phi),   Phi_i = r_i . F

with ``H = diag(eta)`` and ``K = diag(kappa)``.

**Compliance, not hardness.** The parameter head predicts the *compliance* ``s = 1/kappa``
(§2.3.3), so closing a channel is the bounded limit ``s -> 0`` rather than ``kappa -> inf``.
The solve is rescaled by the compliance itself, ``p = S v``. Using ``K S = I``::

    (L S + I) v = -b,     L = B^T H B,     p = S v

``kappa`` never appears and neither does any division by ``s``: the stiff limit is a
*well-conditioned* one (``L S + I -> I``, ``p -> 0``). The channel-energy term is
``1/2 sum_e s_e v_e^2``.

The scaling is deliberately ``S`` and not the symmetric ``S^(1/2)`` that would preserve
Cholesky. Everything above is **polynomial in s**, so all derivatives stay finite at ``s = 0``;
``S^(1/2)`` is not differentiable there, and since ``s -> 0`` is precisely the channel-closure
limit this model is built around -- and the limit force training must differentiate through --
an infinite derivative at the closed channel would defeat the whole parameterization. (It shows
up immediately in practice: padded channels in a mixed-size batch sit at exactly ``s = 0`` and
poison the second-order backward with ``0 * inf``.) The price is an LU solve instead of
Cholesky, which is nothing at these block sizes.

The charge-flow polarizability comes out of the same factorization for three extra right-hand
sides, using ``M^-1 = S (L S + I)^-1``::

    alpha_flow = (B^T R)^T M^-1 (B^T R) = G^T S (L S + I)^-1 G,     G = B^T R

symmetric, PSD, and translation-invariant because ``B^T 1 = 0``.

**The channel graph is supplied, never inferred.** ``bond_index`` comes from the diabatic state
assignment (:mod:`rsfff.mlip.diabats`); nothing here looks at distances to decide which channels
exist. See that module for why.

Shapes: flat ragged batch, N atoms and Nb channels across M systems, labeled by ``batch_idx``
(N,) and ``bond_batch`` (Nb,).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features import BesselBasis
from .heads import mlp


def _local_index(group_idx: torch.Tensor, n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of each element within its group, plus the per-group counts.

    Assumes elements are contiguous per group (the layout ``MoleculeDataset.flat_batch``
    produces). Validated rather than trusted -- a non-contiguous layout would silently scatter
    atoms into the wrong padded rows.
    """
    if group_idx.numel() and bool((group_idx[1:] < group_idx[:-1]).any()):
        raise ValueError(
            "SQE padding requires elements grouped contiguously and in non-decreasing "
            "order of system index"
        )
    counts = torch.bincount(group_idx, minlength=n_groups)
    offsets = torch.cumsum(counts, 0) - counts
    local = torch.arange(group_idx.shape[0], device=group_idx.device) - offsets[group_idx]
    return local, counts


@dataclass
class SQESolution:
    """Outputs at the split-charge minimum.

    charges    : (N,)      q_i = q_i^(0) + sum_j p_ij; sums to the fragment charge exactly
    transfers  : (Nb,)     p_e along each channel (diagnostic; not unique on cyclic graphs)
    energy     : (M,)      charge-sector energy at the minimum
    alpha_flow : (M, 3, 3) charge-flow polarizability, or None if not requested
    """

    charges: torch.Tensor
    transfers: torch.Tensor
    energy: torch.Tensor
    alpha_flow: torch.Tensor | None = None


def sqe_solve(
    chi: torch.Tensor,           # (N,) electronegativity
    eta: torch.Tensor,           # (N,) on-site hardness, strictly positive
    compliance: torch.Tensor,    # (Nb,) channel compliance s = 1/kappa, >= 0
    q0: torch.Tensor,            # (N,) baseline charges; sum per system = formal charge
    positions: torch.Tensor,     # (N, 3)
    bond_index: torch.Tensor,    # (2, Nb) channel graph, global atom indices
    batch_idx: torch.Tensor,     # (N,)
    bond_batch: torch.Tensor,    # (Nb,)
    n_systems: int,
    field: torch.Tensor | None = None,   # (M, 3) uniform field per system
    *,
    with_polarizability: bool = True,
) -> SQESolution:
    """Solve the split-charge problem in closed form. See the module docstring for the algebra.

    Total charge per system equals ``sum_i q_i^(0)`` to machine precision by construction, for
    any values of ``compliance`` -- it is a property of the coordinate system, not of the
    solution.
    """
    dtype, device = positions.dtype, positions.device
    n_atoms = int(positions.shape[0])
    n_bonds = int(bond_index.shape[1])

    atom_local, atom_counts = _local_index(batch_idx, n_systems)
    n_max = int(atom_counts.max()) if n_atoms else 0
    if n_bonds:
        bond_local, bond_counts = _local_index(bond_batch, n_systems)
        nb_max = int(bond_counts.max())
    else:  # a system of isolated atoms has no channels at all
        bond_local = torch.zeros(0, dtype=torch.long, device=device)
        nb_max = 0

    # --- dense padded per-system blocks -------------------------------------------------
    # Padding convention: padded *atoms* have an all-zero incidence column, so whatever fills
    # them never reaches L or b; padded *channels* get compliance 0, which makes their row of
    # A exactly the identity with a zero right-hand side, so p = 0 there with no special-casing.
    def _pad_atoms(x, fill=0.0):
        out = torch.full((n_systems, n_max) + x.shape[1:], fill, dtype=dtype, device=device)
        out[batch_idx, atom_local] = x
        return out

    eta_p = _pad_atoms(eta, fill=1.0)                       # (M, n_max)
    chi_p = _pad_atoms(chi)                                 # (M, n_max)
    q0_p = _pad_atoms(q0)                                   # (M, n_max)
    pos_p = _pad_atoms(positions)                           # (M, n_max, 3)

    if nb_max == 0:
        # No channels anywhere: charges are the baselines, nothing flows.
        q = q0
        e_atom = chi * q + 0.5 * eta * q * q
        if field is not None:
            e_atom = e_atom - q * (positions * field[batch_idx]).sum(dim=-1)
        energy = e_atom.new_zeros(n_systems).index_add_(0, batch_idx, e_atom)
        alpha = (
            positions.new_zeros(n_systems, 3, 3) if with_polarizability else None
        )
        return SQESolution(
            charges=q, transfers=positions.new_zeros(0), energy=energy, alpha_flow=alpha
        )

    s_p = torch.zeros(n_systems, nb_max, dtype=dtype, device=device)
    s_p[bond_batch, bond_local] = compliance

    # Incidence B: +1 on the head atom of a channel, -1 on the tail.
    B = torch.zeros(n_systems, n_max, nb_max, dtype=dtype, device=device)
    head_local = atom_local[bond_index[0]]
    tail_local = atom_local[bond_index[1]]
    B[bond_batch, head_local, bond_local] = 1.0
    B[bond_batch, tail_local, bond_local] = -1.0

    # --- assemble and solve --------------------------------------------------------------
    HB = eta_p.unsqueeze(-1) * B                                          # (M, n_max, Nb)
    L = torch.einsum("mie,mif->mef", B, HB)                               # B^T H B
    # A = L S + I. Invertible for any s >= 0: closed channels make their column a unit vector,
    # leaving a block-triangular system whose active block is (L_aa + K_a) S_a with L PSD and
    # K_a positive definite. TODO: swap in a sparse CG solve once systems outgrow a dense
    # per-molecule block.
    A = L * s_p.unsqueeze(-2) + torch.eye(nb_max, dtype=dtype, device=device)

    drive = chi_p + eta_p * q0_p                                          # (M, n_max)
    if field is not None:
        drive = drive - (pos_p * field.unsqueeze(1)).sum(dim=-1)          # - r_i . F
    b = torch.einsum("mie,mi->me", B, drive)                              # B^T (...)

    # One factorization for the transfers and (optionally) the three polarizability columns.
    G = torch.einsum("mie,mia->mea", B, pos_p) if with_polarizability else None  # (M, Nb, 3)
    rhs = -b.unsqueeze(-1) if G is None else torch.cat((-b.unsqueeze(-1), G), dim=-1)
    sol = torch.linalg.solve(A, rhs)                                      # (M, Nb, 1 [+3])

    v = sol[..., 0]                                                       # (M, Nb)
    p_p = s_p * v                                                         # (M, Nb)
    q_p = q0_p + torch.einsum("mie,me->mi", B, p_p)                       # (M, n_max)
    q = q_p[batch_idx, atom_local]                                        # (N,)
    p = p_p[bond_batch, bond_local]                                       # (Nb,)

    # --- energy at the minimum -----------------------------------------------------------
    # The channel term is 1/2 kappa p^2 = 1/2 s v^2 in the rescaled variable, so the stiff
    # limit costs nothing numerically and 1/s is never formed.
    e_atom = chi * q + 0.5 * eta * q * q
    if field is not None:
        e_atom = e_atom - q * (positions * field[batch_idx]).sum(dim=-1)
    energy = e_atom.new_zeros(n_systems).index_add_(0, batch_idx, e_atom)
    energy = energy + 0.5 * (s_p * v * v).sum(dim=-1)

    alpha_flow = None
    if with_polarizability:
        # alpha_flow = G^T S (L S + I)^-1 G with G = B^T R; symmetric in exact arithmetic
        # (it equals G^T M^-1 G) and translation-invariant since B^T 1 = 0.
        X = s_p.unsqueeze(-1) * sol[..., 1:]                               # S (LS+I)^-1 G
        alpha_flow = torch.einsum("mea,meb->mab", G, X)                    # (M, 3, 3)
        alpha_flow = 0.5 * (alpha_flow + alpha_flow.transpose(-1, -2))

    return SQESolution(charges=q, transfers=p, energy=energy, alpha_flow=alpha_flow)


class PairComplianceHead(nn.Module):
    """Per-channel compliance ``s_ij = softplus(MLP(h_i, h_j, e(r_ij)))``: (Nb,).

    Predicting compliance rather than hardness is deliberate (§2.3.3): channel closure is the
    bounded, well-conditioned limit ``s -> 0``, and the pairwise switching function of Phase 2
    acts multiplicatively on ``s``.

    The head is **symmetric in (i, j)** by construction -- it reads the sum and the absolute
    difference of the two atom features, both invariant under swapping them -- because a channel
    is an undirected object and ``p_ij = -p_ji`` already carries all the antisymmetry.

    The bias is initialized so an untrained channel starts at ``s_init``: soft enough that
    charge flows and gradients reach the head, stiff enough that the initial charges stay near
    their diabatic baselines.
    """

    def __init__(
        self,
        p0: int,
        *,
        hidden: int = 64,
        depth: int = 2,
        n_radial: int = 8,
        cutoff: float = 5.0,
        s_init: float = 0.5,
        s_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.s_floor = float(s_floor)
        self.radial = BesselBasis(n_radial, cutoff)
        self.net = mlp(2 * p0 + n_radial, hidden, depth, 1)
        raw = torch.log(torch.expm1(torch.tensor(max(float(s_init) - self.s_floor, 1e-6))))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(raw.item())

    def forward(
        self,
        h: torch.Tensor,             # (N, p0) per-atom invariant features
        positions: torch.Tensor,     # (N, 3)
        bond_index: torch.Tensor,    # (2, Nb)
    ) -> torch.Tensor:
        i, j = bond_index[0], bond_index[1]
        h_i, h_j = h[i], h[j]
        r = (positions[i] - positions[j]).norm(dim=-1)
        x = torch.cat((h_i + h_j, (h_i - h_j).abs(), self.radial(r)), dim=-1)
        return torch.nn.functional.softplus(self.net(x).squeeze(-1)) + self.s_floor


def atomic_dipole_energy(
    chivec: torch.Tensor,        # (N, 3) vector electronegativity
    alpha: torch.Tensor,         # (N, 3, 3) atomic polarizability (PSD)
    batch_idx: torch.Tensor,
    n_systems: int,
    field: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimized on-site dipole energy ``-1/2 drive^T alpha drive`` per system: (M,).

    ``drive = F - chivec``; the minimizing dipoles are ``mu_i = alpha_i drive_i``
    (:func:`rsfff.mlip.eem.atomic_dipoles`, reused unchanged). Evaluating the quadratic form at
    its minimum avoids ever inverting alpha.
    """
    drive = -chivec if field is None else field[batch_idx] - chivec
    e_atom = -0.5 * torch.einsum("na,nab,nb->n", drive, alpha, drive)
    return e_atom.new_zeros(n_systems).index_add_(0, batch_idx, e_atom)
