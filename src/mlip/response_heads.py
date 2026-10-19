"""Equivariant per-atom tensor heads for response properties, built on bb readouts.

Each head is a :class:`bb.heads.IrrepsReadout` configured for its output irreps: an invariant
MLP produces channel *gates*, and a learned channel matmul reduces the equivariant feature
channels without ever mixing the component (``m``) axis -- so the output stays equivariant
while its magnitude and sign are conditioned on the invariant environment. Physical meaning is
added here: the polarizability goes through :func:`bb.heads.psd_symmetric` / ``to_voigt``, the
quadrupole through the injected ``irrep2_to_spherical`` change of basis.

The ``emb`` input is the per-atom conditioning vector (the charge-aware embedding ``z_i``), so
for an isolated atom (all SOAP features exactly zero) the heads reduce to functions of
``(element, q)`` alone -- the free-atom limit the design requires.

**Two-slot inputs.** ``p_env`` / ``p1_env`` / ``p2_env`` are the trailing environment channels
of each input block (``bb.nn.TwoSlotLinear`` and ``env_reduce_parameter``); a block handed
without them is evaluated as the isolated fragment, bit-identically to zeroing them.

**Checkpoints written before the move to bb load unchanged.** Each head renames its old keys on
load (``base_mlp.* / gate_mlp.* -> mlp.*``, ``equiv_reduce(_env) -> reduce(_env).<irrep>``),
and the old attribute names are kept as read-only aliases (``head.gate_mlp``,
``head.equiv_reduce``, ...). Code that matches checkpoint keys by name itself goes through
:func:`bb.compat.upgrade_state_dict` first (``rsfff.train.term_loop.warm_start`` does).
"""

from __future__ import annotations


import torch
import torch.nn as nn

from bb.compat import register_legacy_keys
from bb.heads import IrrepsReadout, psd_symmetric, to_voigt
from bb.nn import exempt_from_weight_decay
from bb.o3 import matrix_to_voigt, min_eig_sym3x3, voigt_to_matrix

__all__ = [
    "voigt_vector_to_symmetric_matrix",
    "symmetric_matrix_to_voigt_vector",
    "min_eig_sym3x3",
    "AtomicAlphaHead",
    "AtomicVectorHead",
    "AxialQuadrupolePolarizabilityHead",
    "AtomicQuadrupoleHead",
]


def voigt_vector_to_symmetric_matrix(values: torch.Tensor) -> torch.Tensor:
    """Expand ``(..., 6)`` vectors in ``xx yy zz yz xz xy`` order to ``(..., 3, 3)``."""
    return voigt_to_matrix(values)


def symmetric_matrix_to_voigt_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Collapse ``(..., 3, 3)`` symmetric matrices to Voigt ``(..., 6)`` vectors."""
    return matrix_to_voigt(matrix)


def _env(widths: dict) -> dict | None:
    env = {k: int(v) for k, v in widths.items() if v}
    return env or None


def _legacy_rules(irrep: str, gate: str) -> list[tuple[str, str]]:
    return [(f"{gate}.", "mlp."), ("equiv_reduce_env", f"reduce_env.{irrep}"),
            ("equiv_reduce", f"reduce.{irrep}")]


class _GatedHead(IrrepsReadout):
    """The shared ``{0e, ir} -> 1x<ir>`` gated readout with its pre-bb key names."""

    _irrep: str = ""
    _gate: str = "gate_mlp"

    def _install_legacy_keys(self) -> None:
        register_legacy_keys(self, _legacy_rules(self._irrep, self._gate))

    @property
    def equiv_reduce(self) -> nn.Parameter:      # the pre-bb name
        return self.reduce[self._irrep]

    @property
    def equiv_reduce_env(self) -> nn.Parameter | None:
        return self.reduce_env[self._irrep] if self._irrep in self.reduce_env else None


class AtomicAlphaHead(_GatedHead):
    """Per-atom symmetric rank-2 polarizability as a Voigt vector ``(N, 6)``.

    A ``1x0e + 1x2e`` readout: ``a0`` (isotropic) from the invariant MLP, ``a2`` (traceless) by
    invariant-gating a learned reduction of the lambda=2 channels.

    With ``positive_isotropic=True`` the polarizability is ``alpha = alpha_aniso + a0_iso * I``
    built in Cartesian space with ``a0_iso = softplus(a0_raw) + floor``; with ``enforce_psd`` (the
    default) the shift is raised above the most negative anisotropic eigenvalue (the hard shift,
    ``smooth=None``), so **every** eigenvalue of ``alpha`` is ``>= floor``
    (:func:`bb.heads.psd_symmetric`). The ``0e`` irrep maps to ``(1/sqrt 3) I``, which is why the
    shift is applied in Cartesian rather than irrep space.

    Not zero-initialized -- a zero readout would pin every species at ``softplus(0) + floor`` --
    but still exempt from weight decay as a whole: decay pulls ``a0_raw`` toward the
    ``0.693 a0^3`` attractor and erodes the reduction until alpha is isotropic (measured
    ``equiv_reduce`` 0.712 -> 0.176 across a staged fit).
    """

    _irrep, _gate = "2e", "base_mlp"

    def __init__(
        self,
        p0: int,
        p2: int,
        emb_dim: int,
        irrep6_to_voigt: torch.Tensor,
        *,
        p_env: int = 0,
        p2_env: int = 0,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        positive_isotropic: bool = True,
        psd_floor: float = 1e-4,
    ) -> None:
        super().__init__(
            {"0e": p0 + p_env, "2e": p2 + p2_env}, "1x0e + 1x2e", scalar_dim=emb_dim,
            hidden=hidden, depth=depth, reduce_channels=equiv_channels, zero_init=False,
            exempt_decay=True, env_widths=_env({"0e": p_env, "2e": p2_env}),
        )
        self.equiv_channels = equiv_channels
        self.positive_isotropic = positive_isotropic
        self.psd_floor = float(psd_floor)
        # Toggle for the strict-PSD eigenvalue shift; set False to skip the shift.
        self.enforce_psd = True
        self.register_buffer("_irrep6_to_voigt", irrep6_to_voigt, persistent=False)
        self._install_legacy_keys()

    @property
    def base_mlp(self) -> nn.Sequential:        # the pre-bb name
        return self.mlp

    def forward(
        self,
        inv_feats: torch.Tensor,    # (N, P0)
        emb: torch.Tensor,          # (N, emb_dim)
        equiv_feats: torch.Tensor,  # (N, 5, P2)
    ) -> torch.Tensor:
        y = super().forward({"0e": inv_feats.unsqueeze(1), "2e": equiv_feats}, emb)
        a0, a2 = y["0e"][:, 0, :], y["2e"][..., 0]                  # (N, 1), (N, 5)
        if not self.positive_isotropic:
            return to_voigt(a0, a2, self._irrep6_to_voigt)           # (N, 6) Voigt
        return psd_symmetric(
            a0, a2, floor=self.psd_floor, enforce=self.enforce_psd, smooth=None,
            irrep6_to_voigt_matrix=self._irrep6_to_voigt,
        )


class AtomicVectorHead(_GatedHead):
    """Per-atom equivariant vector ``(N, 3)`` (e.g. an atomic dipole, 1o).

    A ``1x1o`` readout of the lambda=1 (odd-parity) channels. Zero-initialized, so the permanent
    atomic dipole starts at zero, and exempt from weight decay as a whole: the reduction has no
    gradient while the gate is zero (see :mod:`rsfff.mlip.heads`).
    """

    _irrep = "1o"

    def __init__(
        self,
        p0: int,
        p1: int,
        emb_dim: int,
        *,
        p_env: int = 0,
        p1_env: int = 0,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
    ) -> None:
        super().__init__(
            {"0e": p0 + p_env, "1o": p1 + p1_env}, "1x1o", scalar_dim=emb_dim, hidden=hidden,
            depth=depth, reduce_channels=equiv_channels, zero_init=True, exempt_decay=True,
            env_widths=_env({"0e": p_env, "1o": p1_env}),
        )
        self.equiv_channels = equiv_channels
        self._install_legacy_keys()

    @property
    def gate_mlp(self) -> nn.Sequential:        # the pre-bb name
        return self.mlp

    def forward(
        self,
        inv_feats: torch.Tensor,  # (N, P0)
        emb: torch.Tensor,        # (N, emb_dim)
        vec_feats: torch.Tensor,  # (N, 3, P1)
    ) -> torch.Tensor:
        y = super().forward({"0e": inv_feats.unsqueeze(1), "1o": vec_feats}, emb)
        return y["1o"][..., 0]                                       # (N, 3)


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
        p_env: int = 0,
        p1_env: int = 0,
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
            p0, p1, emb_dim, p_env=p_env, p1_env=p1_env,
            hidden=hidden, depth=depth, equiv_channels=equiv_channels
        )
        self.axis_eps = float(axis_eps)
        self.register_buffer("_gen", l2_generators, persistent=False)
        # Redundant today -- every parameter lives in `self.axis`, which exempts itself -- but
        # this head is the one that deadlocked completely, so it says so for itself.
        exempt_from_weight_decay(self)

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


class AtomicQuadrupoleHead(_GatedHead):
    """Per-atom traceless quadrupole as **spherical** components ``(N, 5)``.

    A ``1x2e`` readout of the lambda=2 (even-parity) channels, zero-initialized so the
    quadrupole starts at exactly zero.

    Emits ``(q20, q21c, q21s, q22c, q22s)`` -- the convention the multipole force field quotes --
    rather than the backend's raw lambda=2 slots. The two are *not* the same labeling: e3nn's
    real spherical harmonics carry a permuted axis convention that mixes m=0 with m=2c, so the
    ``irrep2_to_spherical`` change of basis is injected here (built by
    :func:`rsfff.ff.multipole.irrep2_to_spherical` from the backend's own tested constant).

    Five components rather than six Cartesian ones is deliberate: the traceless constraint is
    then structural. A six-component prediction would carry an isotropic degree of freedom the
    interaction tensor cannot see -- an unconstrained gauge on the parameters.
    """

    _irrep = "2e"

    def __init__(
        self,
        p0: int,
        p2: int,
        emb_dim: int,
        irrep2_to_spherical: torch.Tensor,   # (5, 5)
        *,
        p_env: int = 0,
        p2_env: int = 0,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
    ) -> None:
        if irrep2_to_spherical.shape != (5, 5):
            raise ValueError(
                f"irrep2_to_spherical must be (5, 5), got {tuple(irrep2_to_spherical.shape)}"
            )
        super().__init__(
            {"0e": p0 + p_env, "2e": p2 + p2_env}, "1x2e", scalar_dim=emb_dim, hidden=hidden,
            depth=depth, reduce_channels=equiv_channels, zero_init=True, exempt_decay=True,
            env_widths=_env({"0e": p_env, "2e": p2_env}),
        )
        self.equiv_channels = equiv_channels
        self.register_buffer("_to_spherical", irrep2_to_spherical, persistent=False)
        self._install_legacy_keys()

    @property
    def gate_mlp(self) -> nn.Sequential:        # the pre-bb name
        return self.mlp

    def forward(
        self,
        inv_feats: torch.Tensor,    # (N, P0)
        emb: torch.Tensor,          # (N, emb_dim)
        equiv_feats: torch.Tensor,  # (N, 5, P2)
    ) -> torch.Tensor:
        y = super().forward({"0e": inv_feats.unsqueeze(1), "2e": equiv_feats}, emb)
        return y["2e"][..., 0] @ self._to_spherical                  # (N, 5)
