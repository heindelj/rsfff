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


class AxialQuadrupolePolarizabilityHead(nn.Module):
    """Axially anisotropic quadrupole polarizability ``C``: ``(N, 5, 5)``, PSD.

    The isotropic ``C = c I5`` this replaces is one positive number per atom. The general
    symmetric map on the ``l=2`` space is fifteen (``Sym^2(l=2) = 0 + 2 + 4``, i.e.
    ``1 + 5 + 9``), which is far more freedom than any current target constrains. The middle
    ground taken here is the physically motivated one: an atom in a bond has an axis, and an
    operator with axial symmetry about ``n`` has eigenvalues depending only on ``|m|`` --
    so exactly **three** distinct values, for ``m = 0``, ``|m| = 1`` and ``|m| = 2``::

        C_i = c_0 P_0(n_i) + c_1 P_1(n_i) + c_2 P_2(n_i)

    with ``P_k`` the spectral projectors of ``M = -L(n)^2`` (eigenvalues ``m^2 = 0, 1, 4``),
    obtained by Lagrange interpolation on those three nodes. Three positive scalars plus a
    direction, against one scalar before.

    Three properties are structural rather than penalised:

    * **Equivariance.** ``n`` is an ``l=1`` head output and ``L`` are the ``l=2`` generators
      (:func:`rsfff.ff.multipole.l2_rotation_generators`), so ``C`` rotates correctly by
      construction. No fixed lab direction enters -- which is why the axis is normalised with
      ``v / sqrt(|v|^2 + eps^2)`` rather than by adding a bias vector.
    * **Positive semi-definiteness**, via the Gram form ``C = S S^T`` with
      ``S = sum_k sqrt(c_k) P_k``. When ``|n| = 1`` the ``P_k`` are exact orthogonal
      projectors and ``S S^T`` *is* ``sum_k c_k P_k``; when it is not -- the soft
      normalisation leaves ``|n| < 1`` near ``v = 0`` -- ``S`` is still symmetric, so the Gram
      form is still PSD. Writing it this way means PSD does not depend on the normalisation
      being exact, which a direct ``sum_k c_k P_k`` does: the Lagrange basis is built for
      eigenvalues ``{0, 1, 4}``, so off the unit sphere its "projectors" stop being positive.
      Measured at ``|n| = 0.9`` with ``c = (10, 0.01, 0.01)``, the naive sum has smallest
      eigenvalue ``-4.24`` where the Gram form gives ``+0.32``; both agree exactly at
      ``|n| = 1``. The failure needs ``c_0`` to dominate and ``|n|`` in roughly ``0.7-0.95``,
      which is precisely the regime a vector head passes through on its way off zero.
    * **The isotropic model is the ``c_0 = c_1 = c_2`` special case, exactly.** The Lagrange
      basis is a partition of unity for *any* ``M``, so equal ``c_k`` give ``C = c I5``
      independently of the axis. Initializing the three scalars equal therefore reproduces the
      previous head bit-for-bit, and leaves the axis inert until the fit separates them.

    ``axis_eps`` bounds the gradient of the normalisation at ``v = 0`` (where a bare
    ``v / |v|`` is NaN, and would poison the backward pass even though ``C`` does not depend
    on the axis there). Small enough not to distort a trained axis, large enough not to
    produce a ``1/eps`` spike while the vector head is still near its zero initialization.
    """

    def __init__(
        self,
        p0: int,
        p1: int,
        emb_dim: int,
        l2_generators: torch.Tensor,      # (3, 5, 5)
        *,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        axis_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if l2_generators.shape != (3, 5, 5):
            raise ValueError(
                f"l2_generators must be (3, 5, 5), got {tuple(l2_generators.shape)}"
            )
        self.axis = AtomicVectorHead(
            p0, p1, emb_dim, hidden=hidden, depth=depth, equiv_channels=equiv_channels
        )
        self.axis_eps = float(axis_eps)
        self.register_buffer("_gen", l2_generators, persistent=False)

    def forward(
        self,
        inv_feats: torch.Tensor,    # (N, P0)
        emb: torch.Tensor,          # (N, emb_dim)
        vec_feats: torch.Tensor,    # (N, 3, P1)
        c: torch.Tensor,            # (N, 3) strictly positive eigenvalues, m = 0, 1, 2
    ) -> torch.Tensor:
        v = self.axis(inv_feats, emb, vec_feats)                          # (N, 3)
        n = v / torch.sqrt((v * v).sum(-1, keepdim=True) + self.axis_eps ** 2)
        gen = torch.einsum("na,aij->nij", n, self._gen.to(v.dtype))       # (N, 5, 5)
        m = -torch.bmm(gen, gen)                                          # symmetric PSD
        eye = torch.eye(5, dtype=v.dtype, device=v.device).expand_as(m)

        # Lagrange basis on the nodes {0, 1, 4}; sums to the identity for any M.
        m1, m4 = m - eye, m - 4.0 * eye
        p0 = torch.bmm(m1, m4) / 4.0
        p1 = torch.bmm(m, m4) / -3.0
        p2 = torch.bmm(m, m1) / 12.0

        root = c.clamp_min(0.0).sqrt()[:, :, None, None]                  # (N, 3, 1, 1)
        s = root[:, 0] * p0 + root[:, 1] * p1 + root[:, 2] * p2
        return torch.bmm(s, s)


class AtomicQuadrupoleHead(nn.Module):
    """Per-atom traceless quadrupole as **spherical** components ``(N, 5)``.

    Same construction as :class:`AtomicVectorHead` one rank up: an invariant MLP gates a
    learned reduction of the lambda=2 (even-parity) channels, so the output is equivariant
    while its magnitude is conditioned on the invariant environment. Zero-initialized, so
    the quadrupole starts at exactly zero.

    Emits ``(q20, q21c, q21s, q22c, q22s)`` -- the convention the multipole force field
    quotes -- rather than the backend's raw lambda=2 slots. The two are *not* the same
    labeling: e3nn's real spherical harmonics carry a permuted axis convention that mixes
    m=0 with m=2c, so the ``irrep2_to_spherical`` change of basis is injected here (built
    by :func:`rsfff.ff.multipole.irrep2_to_spherical` from the backend's own tested
    constant) exactly as :class:`AtomicAlphaHead` injects ``irrep6_to_voigt``.

    Five components rather than six Cartesian ones is deliberate: the traceless constraint
    is then structural. A six-component prediction would carry an isotropic degree of
    freedom the interaction tensor cannot see -- an unconstrained gauge on the parameters.
    """

    def __init__(
        self,
        p0: int,
        p2: int,
        emb_dim: int,
        irrep2_to_spherical: torch.Tensor,   # (5, 5)
        *,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
    ) -> None:
        super().__init__()
        if irrep2_to_spherical.shape != (5, 5):
            raise ValueError(
                f"irrep2_to_spherical must be (5, 5), got {tuple(irrep2_to_spherical.shape)}"
            )
        self.equiv_channels = equiv_channels
        self.equiv_reduce = nn.Parameter(torch.randn(p2, equiv_channels) / (p2 ** 0.5))
        self.gate_mlp = mlp(p0 + emb_dim, hidden, depth, equiv_channels)
        self.register_buffer("_to_spherical", irrep2_to_spherical, persistent=False)
        with torch.no_grad():
            self.gate_mlp[-1].weight.zero_()
            self.gate_mlp[-1].bias.zero_()

    def forward(
        self,
        inv_feats: torch.Tensor,    # (N, P0)
        emb: torch.Tensor,          # (N, emb_dim)
        equiv_feats: torch.Tensor,  # (N, 5, P2)
    ) -> torch.Tensor:
        gate = self.gate_mlp(torch.cat((inv_feats, emb), dim=-1))              # (N, K)
        equiv_k = torch.einsum("nmp,pk->nmk", equiv_feats, self.equiv_reduce)  # (N,5,K)
        a2 = torch.einsum("nmk,nk->nm", equiv_k, gate)                         # (N, 5)
        return a2 @ self._to_spherical                                         # (N, 5)
