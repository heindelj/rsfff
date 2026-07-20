"""Equivariant per-atom tensor heads for response properties.

Ported from the pauli_fierz_pinn repo (``src/models/heads.py``): the rank-2
:class:`AtomicAlphaHead` (per-atom symmetric polarizability) and the rank-1
:class:`AtomicVectorHead` (per-atom dipole). Both use the same construction: an
invariant MLP produces channel *gates*, and a learned channel matmul reduces the
equivariant feature channels without ever mixing the component (``m``) axis -- so
the output stays equivariant while its magnitude/sign is conditioned on the
invariant environment.

The ``emb`` input slot is the per-atom conditioning vector; here it receives the
charge-aware embedding ``z_i`` (see ``charge_embedding.py``), so for an isolated
atom (all SOAP features exactly zero) the heads reduce to functions of
``(element, q)`` alone -- the free-atom limit required by the design doc.

The ``(1x0e + 1x2e) -> Voigt(6)`` change of basis is injected from
``rsfff.features.EquivariantBackend.irrep6_to_voigt()`` (numerically identical to
the e3nn-derived constant used in the source repo).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .heads import mlp


def voigt_vector_to_symmetric_matrix(values: torch.Tensor) -> torch.Tensor:
    """Expand ``(..., 6)`` vectors in ``xx yy zz yz xz xy`` order to ``(..., 3, 3)``."""
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.get_default_dtype())
    if values.shape[-1] != 6:
        raise ValueError(
            f"Expected last dimension 6 for xx yy zz yz xz xy data, got {tuple(values.shape)}"
        )
    xx, yy, zz, yz, xz, xy = (values[..., i] for i in range(6))
    row_x = torch.stack((xx, xy, xz), dim=-1)
    row_y = torch.stack((xy, yy, yz), dim=-1)
    row_z = torch.stack((xz, yz, zz), dim=-1)
    return torch.stack((row_x, row_y, row_z), dim=-2)


def symmetric_matrix_to_voigt_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Collapse ``(..., 3, 3)`` symmetric matrices to Voigt ``(..., 6)`` vectors."""
    return torch.stack(
        (
            matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2],
            matrix[..., 1, 2], matrix[..., 0, 2], matrix[..., 0, 1],
        ),
        dim=-1,
    )


def min_eig_sym3x3(A: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of a batch of symmetric 3x3 matrices, in closed form.

    Trigonometric (Smith) solution of the characteristic cubic. Pure torch,
    differentiable, device-agnostic. Deliberately avoids ``torch.linalg.eigvalsh``,
    whose batched cuSOLVER kernel raises ``CUSOLVER_STATUS_INVALID_VALUE`` for large
    batch sizes and is unimplemented on MPS. ``A`` is assumed symmetric; only the
    upper entries are read.
    """
    a00 = A[..., 0, 0]; a11 = A[..., 1, 1]; a22 = A[..., 2, 2]
    a01 = A[..., 0, 1]; a02 = A[..., 0, 2]; a12 = A[..., 1, 2]
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    d0 = a00 - q; d1 = a11 - q; d2 = a22 - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    # p == 0 only for an isotropic (triple-root) matrix; the clamp keeps the division
    # and the sqrt gradient finite there, and the result correctly reduces to q.
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=1e-30))
    b00 = d0 / p; b11 = d1 / p; b22 = d2 / p
    b01 = a01 / p; b02 = a02 / p; b12 = a12 / p
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    # r in [-1, 1]; clamp strictly inside so that at a double root (r -> +/-1, where
    # acos' diverges) the 0*inf does not poison the backward pass with NaNs.
    r = torch.clamp(detB / 2.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    phi = torch.acos(r) / 3.0
    # Eigenvalues are q + 2p*cos(phi + 2*pi*k/3); the k=1 branch is the smallest.
    return q + 2.0 * p * torch.cos(phi + (2.0 * math.pi / 3.0))


class AtomicAlphaHead(nn.Module):
    """Per-atom symmetric rank-2 polarizability as a Voigt vector ``(N, 6)``.

    Predicts the irrep vector ``[a0 (0e), a2 (2e)]``: ``a0`` (isotropic) from an
    invariant MLP, ``a2`` (traceless) by invariant-gating a learned equivariant
    reduction of the lambda=2 channels.

    With ``positive_isotropic=True`` the polarizability is built as
    ``alpha = alpha_aniso + a0_iso * I`` directly in Cartesian space, where the
    isotropic shift ``a0_iso = softplus(a0_raw) + floor`` is strictly positive.
    When ``enforce_psd`` is on (the default) the shift is additionally raised above
    the most-negative anisotropic eigenvalue,
    ``a0_iso -> a0_iso - min_eig(alpha_aniso)``, so **every** eigenvalue of
    ``alpha`` is ``>= floor``. (Adding the isotropic part in Cartesian is what makes
    this exact: the ``0e`` irrep maps to ``(1/sqrt 3) I``, so shifting ``a0`` in
    irrep space would only move eigenvalues by ``1/sqrt 3`` and would *not*
    guarantee PSD.) PSD alpha is what the future long-range/Applequist stage needs;
    for the current short-range phase it is simply a physically sensible prior.
    """

    def __init__(
        self,
        p0: int,
        p2: int,
        emb_dim: int,
        irrep6_to_voigt: torch.Tensor,
        *,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        positive_isotropic: bool = True,
        psd_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.equiv_channels = equiv_channels
        self.positive_isotropic = positive_isotropic
        self.psd_floor = float(psd_floor)
        # Toggle for the strict-PSD eigenvalue shift; set False to skip the shift.
        self.enforce_psd = True
        self.equiv_reduce = nn.Parameter(torch.randn(p2, equiv_channels) / (p2 ** 0.5))
        self.base_mlp = mlp(p0 + emb_dim, hidden, depth, 1 + equiv_channels)
        self.register_buffer("_irrep6_to_voigt", irrep6_to_voigt, persistent=False)

    def forward(
        self,
        inv_feats: torch.Tensor,    # (N, P0)
        emb: torch.Tensor,          # (N, emb_dim)
        equiv_feats: torch.Tensor,  # (N, 5, P2)
    ) -> torch.Tensor:
        out = self.base_mlp(torch.cat((inv_feats, emb), dim=-1))   # (N, 1+K)
        a0_raw = out[:, :1]                                        # (N, 1)
        gate = out[:, 1:]                                          # (N, K)
        equiv_k = torch.einsum("nmp,pk->nmk", equiv_feats, self.equiv_reduce)  # (N,5,K)
        a2 = torch.einsum("nmk,nk->nm", equiv_k, gate)             # (N, 5)

        if not self.positive_isotropic:
            irrep6 = torch.cat((a0_raw, a2), dim=-1)               # (N, 6) [0e, 2e]
            return irrep6 @ self._irrep6_to_voigt.t()              # (N, 6) Voigt

        # Anisotropic (traceless) part in Voigt, then the strictly-positive
        # isotropic shift added on the diagonal -- all in Cartesian.
        a0_zero = torch.zeros_like(a0_raw)
        voigt_aniso = torch.cat((a0_zero, a2), dim=-1) @ self._irrep6_to_voigt.t()  # (N, 6)
        a0_iso = torch.nn.functional.softplus(a0_raw) + self.psd_floor              # (N, 1)

        if self.enforce_psd:
            # Raise the isotropic shift above the most-negative anisotropic
            # eigenvalue so min_eig(alpha) = min_eig(aniso) + a0_iso >= floor.
            alpha_aniso = voigt_vector_to_symmetric_matrix(voigt_aniso)             # (N, 3, 3)
            min_eig = min_eig_sym3x3(alpha_aniso).unsqueeze(-1).clamp(max=0.0)      # (N, 1)
            a0_iso = a0_iso - min_eig

        # alpha = alpha_aniso + a0_iso * I  ->  add a0_iso to xx, yy, zz Voigt entries.
        iso = torch.cat((a0_iso, a0_iso, a0_iso, a0_iso.new_zeros(a0_iso.shape[0], 3)), dim=-1)
        return voigt_aniso + iso


class AtomicVectorHead(nn.Module):
    """Per-atom equivariant vector ``(N, 3)`` (e.g. an atomic dipole, 1o).

    Reduces the lambda=1 (odd-parity) feature channels with an invariant gate,
    mirroring :class:`AtomicAlphaHead`'s 2e construction. Zero-initialized so the
    permanent atomic dipole starts at zero.
    """

    def __init__(
        self,
        p0: int,
        p1: int,
        emb_dim: int,
        *,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
    ) -> None:
        super().__init__()
        self.equiv_channels = equiv_channels
        self.equiv_reduce = nn.Parameter(torch.randn(p1, equiv_channels) / (p1 ** 0.5))
        self.gate_mlp = mlp(p0 + emb_dim, hidden, depth, equiv_channels)
        with torch.no_grad():
            self.gate_mlp[-1].weight.zero_()
            self.gate_mlp[-1].bias.zero_()

    def forward(
        self,
        inv_feats: torch.Tensor,  # (N, P0)
        emb: torch.Tensor,        # (N, emb_dim)
        vec_feats: torch.Tensor,  # (N, 3, P1)
    ) -> torch.Tensor:
        gate = self.gate_mlp(torch.cat((inv_feats, emb), dim=-1))         # (N, K)
        vec_k = torch.einsum("nmp,pk->nmk", vec_feats, self.equiv_reduce)  # (N,3,K)
        return torch.einsum("nmk,nk->nm", vec_k, gate)                     # (N, 3)
