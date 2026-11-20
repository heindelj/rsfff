r"""A self-contained, differentiable split-charge (SQE) response model with no Coulomb coupling.

(Moved here from ``bb/scripts/ece_props/eceprops/response.py``, with one addition: ``dchi``, the
electronegativity difference along each bond, ``chi[second] - chi[first]``. The charge sector only
ever sees ``B^T chi``, so a model can emit that directly, with no gauge left over; ``chi`` and
``dchi`` add when both are given.)

Three independent sectors per molecule, all in atomic units and all about a per-system origin
``O`` (``r_i = R_i - O``):

**Charges (split-charge equilibration).** Charge moves only along the bonds,
``q = q0 + B p`` with ``B`` the bond incidence matrix (+1 on the second atom of a bond, -1 on the
first), so ``sum_i q_i = sum_i q0_i`` for any transfers ``p``::

    E_q(p; phi) = sum_i [ chi_i q_i + 1/2 eta_i q_i^2 + q_i phi_i ] + 1/2 sum_b p_b^2 / s_b

``eta_i > 0`` is the on-site hardness and ``s_b >= 0`` the bond compliance (``s = 1/kappa``;
``s -> 0`` closes a channel and is the well-conditioned limit). There is **no** ``J_ij`` between
atoms: this is the intramolecular response sector only. Minimizing, with ``S = diag(s)`` and
``L = B^T H B``::

    (L S + I) v = -B^T (chi + eta q0 + phi),     p = S v

which is one small dense LU solve per molecule (rsfff's ``sqe_solve`` construction: polynomial in
``s``, finite derivatives at ``s = 0``).

**Atomic dipoles.** Permanent ``mu0_i`` and a PSD atomic polarizability ``alpha_i``::

    E_mu = sum_i [ -mu0_i . E_i - 1/2 E_i . alpha_i E_i ]

**Atomic second moments.** Permanent primitive second moment ``M2_i`` (about atom i) and an
isotropic quadrupole polarizability ``c_i >= 0`` acting on the traceless field gradient::

    E_Q = sum_i [ -1/2 G : M2_i - 1/4 c_i G_T : G_T ]

The external potential is a uniform field ``F`` plus a uniform field gradient ``G`` (symmetric),
``phi(r) = -F . r - 1/2 r . G r``, ``E(r) = F + G r``.

**Zero-field subtraction.** The model's energy contribution is ``dE(F, G) = E(F, G) - E(0, 0)``
(each sector minimized), which is identically zero without a field, so this sector never touches
the potential energy surface (that is the ECE energy head's job) and only describes how the
molecule responds to an external field. Since there is no coupling, ``dE`` is exactly quadratic::

    dE(h) = - h . M0 - 1/2 h . P h,        h = [F, 1/2 vec(G)]  (12 numbers)

``M0 = [M1, vec(M2)]`` are the permanent (zero-field) dipole and primitive second moment about
``O``, and ``P`` (12 x 12, symmetric PSD) is the generalized polarizability:

    alpha_ab        = d M1_a / d F_b    = P[a, b]                      (dipole polarizability)
    A_a,bc          = d M1_a / d G_bc   = 1/2 P[a, 3+3b+c]              (dipole-quadrupole)
    C_ab,cd         = d M2_ab / d G_cd  = 1/2 P[3+3a+b, 3+3c+d]         (quadrupole polarizability)

so ``M = -d dE/dh`` and ``P = -d^2 dE/dh^2``: the moments and polarizabilities *are* field
derivatives of the subtracted energy. :func:`response` computes ``M0`` and ``P`` in closed form
from one factorization per molecule (the charge-flow part is ``P_q = Y^T K Y`` with
``K = B S (LS + I)^-1 B^T`` and ``Y_i = [r_i, vec(r_i r_i^T)]``, so it costs 13 right-hand sides);
:func:`field_energy` evaluates ``dE`` the long way, by solving at the field and at zero field and
subtracting, which the tests use to check the closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ResponseParameters", "ResponseSolution", "response", "field_energy", "traceless_projector"]


@dataclass
class ResponseParameters:
    """Per-atom / per-bond parameters of the three sectors. Any sector may be absent (None)."""

    q0: torch.Tensor                          # (N,) baseline charges; sum per molecule = total charge
    chi: torch.Tensor | None = None           # (N,) electronegativity
    dchi: torch.Tensor | None = None          # (Nb,) chi[second atom] - chi[first atom] of each bond
    eta: torch.Tensor | None = None           # (N,) hardness, > 0
    compliance: torch.Tensor | None = None    # (Nb,) s_b >= 0; None: no charge flow
    mu0: torch.Tensor | None = None           # (N, 3) permanent atomic dipoles
    alpha: torch.Tensor | None = None         # (N, 3, 3) atomic polarizabilities, PSD
    m2: torch.Tensor | None = None            # (N, 3, 3) permanent atomic primitive second moments
    c_quad: torch.Tensor | None = None        # (N,) isotropic quadrupole polarizabilities, >= 0


@dataclass
class ResponseSolution:
    charges: torch.Tensor             # (N,) permanent (zero-field) charges
    transfers: torch.Tensor           # (Nb,) bond transfers p_b at zero field
    dipole: torch.Tensor              # (M, 3) permanent dipole about the origin
    second_moment: torch.Tensor       # (M, 3, 3) permanent primitive second moment about the origin
    P: torch.Tensor                   # (M, 12, 12) generalized polarizability
    energy0: torch.Tensor             # (M,) zero-field energy of the sectors (NOT added to the PES)

    @property
    def polarizability(self) -> torch.Tensor:          # (M, 3, 3)
        return self.P[:, :3, :3]

    @property
    def dipole_quadrupole_polarizability(self) -> torch.Tensor:   # (M, 3, 3, 3): dM1_a/dG_bc
        return 0.5 * self.P[:, :3, 3:].reshape(-1, 3, 3, 3)

    @property
    def quadrupole_polarizability(self) -> torch.Tensor:          # (M, 3, 3, 3, 3): dM2_ab/dG_cd
        return 0.5 * self.P[:, 3:, 3:].reshape(-1, 3, 3, 3, 3)

    @property
    def moments(self) -> torch.Tensor:                 # (M, 12)
        return torch.cat((self.dipole, self.second_moment.reshape(-1, 9)), dim=-1)

    def energy(self, field=None, field_gradient=None) -> torch.Tensor:
        """``dE(F, G) = -h.M0 - 1/2 h.P h`` (M,); zero when both are None."""
        h = _h_vector(field, field_gradient, self.P)
        if h is None:
            return self.P.new_zeros(self.P.shape[0])
        return -(h * self.moments).sum(-1) - 0.5 * torch.einsum("ma,mab,mb->m", h, self.P, h)


def _h_vector(field, field_gradient, like):
    if field is None and field_gradient is None:
        return None
    M = like.shape[0]
    F = like.new_zeros(M, 3) if field is None else field.expand(M, 3)
    G = like.new_zeros(M, 3, 3) if field_gradient is None else field_gradient.expand(M, 3, 3)
    return torch.cat((F, 0.5 * G.reshape(M, 9)), dim=-1)


def traceless_projector(dtype=torch.float64, device=None) -> torch.Tensor:
    """(9, 9) projector onto symmetric traceless 3x3 matrices (row-major vec)."""
    eye = torch.eye(3, dtype=dtype, device=device)
    sym = 0.5 * (torch.einsum("ac,bd->abcd", eye, eye) + torch.einsum("ad,bc->abcd", eye, eye))
    tr = torch.einsum("ab,cd->abcd", eye, eye) / 3.0
    return (sym - tr).reshape(9, 9)


def _local_index(group_idx: torch.Tensor, n_groups: int):
    counts = torch.bincount(group_idx, minlength=n_groups)
    offsets = torch.cumsum(counts, 0) - counts
    local = torch.arange(group_idx.shape[0], device=group_idx.device) - offsets[group_idx]
    return local, counts


def _charge_blocks(p: ResponseParameters, r, bonds, batch_idx, bond_batch, M):
    """Dense padded per-molecule blocks of the charge sector."""
    dtype, device = r.dtype, r.device
    a_loc, a_cnt = _local_index(batch_idx, M)
    b_loc, b_cnt = _local_index(bond_batch, M)
    n, nb = int(a_cnt.max()), int(b_cnt.max())

    def pad(x, fill=0.0):
        out = torch.full((M, n) + x.shape[1:], fill, dtype=dtype, device=device)
        return out.index_put((batch_idx, a_loc), x)

    s = torch.zeros(M, nb, dtype=dtype, device=device).index_put((bond_batch, b_loc), p.compliance)
    B = torch.zeros(M, n, nb, dtype=dtype, device=device)
    B = B.index_put((bond_batch, a_loc[bonds[:, 1]], b_loc), torch.ones_like(p.compliance))
    B = B.index_put((bond_batch, a_loc[bonds[:, 0]], b_loc), -torch.ones_like(p.compliance))
    eta = pad(p.eta, 1.0)
    L = torch.einsum("mie,mi,mif->mef", B, eta, B)
    A = L * s.unsqueeze(-2) + torch.eye(nb, dtype=dtype, device=device)
    dchi = None if p.dchi is None else torch.zeros(M, nb, dtype=dtype, device=device).index_put((bond_batch, b_loc), p.dchi)
    return dict(pad=pad, a_loc=a_loc, b_loc=b_loc, s=s, B=B, A=A, eta=eta, dchi=dchi)


def _charge_solve(p: ResponseParameters, blk, phi_p, extra_rhs=None):
    """Transfers at drive ``chi + eta q0 + phi``. Returns padded (q, v, sol_extra)."""
    pad = blk["pad"]
    chi = pad(p.chi) if p.chi is not None else torch.zeros_like(blk["eta"])
    drive = chi + blk["eta"] * pad(p.q0)
    if phi_p is not None:
        drive = drive + phi_p
    b = torch.einsum("mie,mi->me", blk["B"], drive)
    if p.dchi is not None:
        b = b + blk["dchi"]
    rhs = -b.unsqueeze(-1)
    if extra_rhs is not None:
        rhs = torch.cat((rhs, extra_rhs), dim=-1)
    sol = torch.linalg.solve(blk["A"], rhs)
    v = sol[..., 0]
    q = pad(p.q0) + torch.einsum("mie,me->mi", blk["B"], blk["s"] * v)
    return q, v, (sol[..., 1:] if extra_rhs is not None else None)


def _quadratic_energy(p, q, v, s, r, batch_idx, M, phi=None, dchi=None):
    e = q.new_zeros(M)
    chi = p.chi if p.chi is not None else torch.zeros_like(q)
    e_at = chi * q + 0.5 * p.eta * q * q
    if phi is not None:
        e_at = e_at + q * phi
    e = e.index_add(0, batch_idx, e_at)
    if dchi is not None:
        e = e + (dchi * s * v).sum(-1)                  # sum_b dchi_b p_b,  p = S v
    return e + 0.5 * (s * v * v).sum(-1)


def response(p: ResponseParameters, positions, bonds, batch_idx, bond_batch, n_systems,
             origin=None) -> ResponseSolution:
    """Permanent moments, charges and the 12x12 generalized polarizability, in closed form.

    ``positions`` (N, 3) bohr; ``bonds`` (Nb, 2) global atom indices; ``origin`` (M, 3) or None
    (coordinate origin). Differentiable with respect to everything (positions included, to any
    order), so dipole / polarizability derivatives come from autograd.
    """
    M = int(n_systems)
    dtype, device = positions.dtype, positions.device
    r = positions if origin is None else positions - origin[batch_idx]
    N = r.shape[0]
    rr = r[:, :, None] * r[:, None, :]                                   # (N, 3, 3)
    Y = torch.cat((r, rr.reshape(N, 9)), dim=-1)                          # (N, 12)

    # ---- charge sector -------------------------------------------------------------------
    flow = p.compliance is not None and bonds.shape[0] > 0
    if flow:
        blk = _charge_blocks(p, r, bonds, batch_idx, bond_batch, M)
        Yp = blk["pad"](Y)                                                # (M, n, 12)
        Gq = torch.einsum("mie,mik->mek", blk["B"], Yp)                   # (M, nb, 12)
        q_p, v, X = _charge_solve(p, blk, None, Gq)
        q = q_p[batch_idx, blk["a_loc"]]
        transfers = (blk["s"] * v)[bond_batch, blk["b_loc"]]
        P = torch.einsum("mek,mel->mkl", Gq, blk["s"].unsqueeze(-1) * X)  # Gq^T S (LS+I)^-1 Gq
        P = 0.5 * (P + P.transpose(-1, -2))
        energy0 = _quadratic_energy(p, q, v, blk["s"], r, batch_idx, M, dchi=blk["dchi"])
    else:
        q = p.q0
        transfers = positions.new_zeros(bonds.shape[0])
        P = positions.new_zeros(M, 12, 12)
        if p.eta is not None:
            energy0 = positions.new_zeros(M).index_add(
                0, batch_idx, (p.chi if p.chi is not None else 0.0) * q + 0.5 * p.eta * q * q)
        else:
            energy0 = positions.new_zeros(M)
    mom = positions.new_zeros(M, 12).index_add(0, batch_idx, q[:, None] * Y)

    # ---- atomic dipoles ------------------------------------------------------------------
    if p.mu0 is not None or p.alpha is not None:
        eye = torch.eye(3, dtype=dtype, device=device)
        # E_i = J_i h:  J[a, b] = delta_ab;  J[a, 3+3c+d] = delta_ac r_d + delta_ad r_c
        Jg = torch.einsum("ac,nd->nacd", eye, r) + torch.einsum("ad,nc->nacd", eye, r)
        J = torch.cat((eye.expand(N, 3, 3), Jg.reshape(N, 3, 9)), dim=-1)  # (N, 3, 12)
        if p.mu0 is not None:
            mom = mom.index_add(0, batch_idx, torch.einsum("nak,na->nk", J, p.mu0))
            energy0 = energy0  # the permanent dipoles have no zero-field energy of their own
        if p.alpha is not None:
            P = P.index_add(0, batch_idx, torch.einsum("nak,nab,nbl->nkl", J, p.alpha, J))

    # ---- atomic second moments -----------------------------------------------------------
    if p.m2 is not None:
        mom = mom.index_add(0, batch_idx, torch.cat((r.new_zeros(N, 3), p.m2.reshape(N, 9)), -1))
    if p.c_quad is not None:
        T = traceless_projector(dtype, device)
        c_sys = positions.new_zeros(M).index_add(0, batch_idx, p.c_quad)
        P = P.clone()
        P[:, 3:, 3:] = P[:, 3:, 3:] + 2.0 * c_sys[:, None, None] * T

    return ResponseSolution(
        charges=q, transfers=transfers, dipole=mom[:, :3], second_moment=mom[:, 3:].reshape(M, 3, 3),
        P=P, energy0=energy0,
    )


def field_energy(p: ResponseParameters, positions, bonds, batch_idx, bond_batch, n_systems,
                 field=None, field_gradient=None, origin=None):
    """``dE(F, G) = E(F, G) - E(0, 0)`` by explicit minimization at both fields: (M,).

    The long way round, for checking :func:`response` and for anyone who wants the energy of
    the sectors in a field without trusting the closed form. ``field`` (M, 3) or (3,);
    ``field_gradient`` (M, 3, 3) or (3, 3), symmetric.
    """
    M = int(n_systems)
    r = positions if origin is None else positions - origin[batch_idx]
    N = r.shape[0]
    F = r.new_zeros(M, 3) if field is None else field.expand(M, 3)
    G = r.new_zeros(M, 3, 3) if field_gradient is None else field_gradient.expand(M, 3, 3)
    Fi, Gi = F[batch_idx], G[batch_idx]
    phi = -(Fi * r).sum(-1) - 0.5 * torch.einsum("na,nab,nb->n", r, Gi, r)
    Ei = Fi + torch.einsum("nab,nb->na", Gi, r)

    def sector_energy(phi, Ei, Gi):
        e = r.new_zeros(M)
        if p.compliance is not None and bonds.shape[0] > 0:
            blk = _charge_blocks(p, r, bonds, batch_idx, bond_batch, M)
            q_p, v, _ = _charge_solve(p, blk, blk["pad"](phi))
            q = q_p[batch_idx, blk["a_loc"]]
            e = e + _quadratic_energy(p, q, v, blk["s"], r, batch_idx, M, phi, dchi=blk["dchi"])
        else:
            e_at = p.q0 * phi
            if p.eta is not None:
                e_at = e_at + (p.chi if p.chi is not None else 0.0) * p.q0 + 0.5 * p.eta * p.q0 ** 2
            e = e.index_add(0, batch_idx, e_at)
        e_at = r.new_zeros(N)
        if p.mu0 is not None:
            e_at = e_at - (p.mu0 * Ei).sum(-1)
        if p.alpha is not None:
            e_at = e_at - 0.5 * torch.einsum("na,nab,nb->n", Ei, p.alpha, Ei)
        if p.m2 is not None:
            e_at = e_at - 0.5 * (Gi * p.m2).sum((-1, -2))
        if p.c_quad is not None:
            tr = torch.diagonal(Gi, dim1=-2, dim2=-1).sum(-1)
            GT = Gi - tr[:, None, None] / 3.0 * torch.eye(3, dtype=r.dtype, device=r.device)
            e_at = e_at - 0.25 * p.c_quad * (GT * GT).sum((-1, -2))
        return e.index_add(0, batch_idx, e_at)

    zero = torch.zeros_like(phi)
    return sector_energy(phi, Ei, Gi) - sector_energy(zero, torch.zeros_like(Ei), torch.zeros_like(Gi))
