"""Shared primitives, state-dependent projection, and the three feature blocks.

``docs/fff_film.md`` §4 in code. One neighbor search and one radial x angular expansion per
batch -- the primitives are fragmentation-independent -- then the assignment matrix projects
the *edge contributions* before any nonlinear contraction:

    rho_in  = sum_j P_ij psi_ij          rho_env = sum_j (1 - P_ij) psi_ij

    x_in    = P(rho_in)                  the internal power spectrum
    x_env   = P(rho_env)                 the environmental power spectrum
    x_cross = rho_in (x) rho_env         the bilinear cross block (new relative to v4)

The mechanics are bb's :class:`bb.features.PartitionedFeaturizer` (one featurizer, its
``edge_weight`` path, plus the cross spectrum and ``a_env``); this module only supplies the
membership -- the state's co-membership ``P_ij`` -- and rsfff's requirement that the invariant
cross block is always present. The featurizer is held as ``self.featurizer``, so checkpoint keys
(``projector.featurizer.*``) are unchanged.

Vertex guarantee, structural: at a one-hot ``C`` the co-membership is an exact 0/1 per edge,
so a lone fragment's environment density is an empty sum -- ``x_env``, ``x_cross`` and the
environment activity ``a_env`` are exact zeros, not small numbers.
"""

from __future__ import annotations

from bb.features import PartitionedFeatures, PartitionedFeaturizer, SOAPFeaturizer

from .state import StateDescriptor

__all__ = ["FragmentProjector", "ProjectedFeatures"]

#: ``x_in`` / ``x_env`` (:class:`bb.o3.IrrepFeatures`), ``x_cross`` ``{lam: (N, 2lam+1, P)}``,
#: ``a_env`` ``(N,)``, ``edge_index``, and the co-membership on those edges (``P_edge``, an alias
#: of ``edge_weight``); ``cross_inv`` is the lambda=0 cross slice ``(N, P)``.
ProjectedFeatures = PartitionedFeatures


class FragmentProjector(PartitionedFeaturizer):
    """A :class:`bb.features.PartitionedFeaturizer` whose membership is the state's ``P_ij``.

    The internal and environment blocks are bit-compatible with the v4 two-slot descriptor at a
    one-hot assignment; the only computation beyond the featurizer's own is the cross spectrum
    and the scalar ``a_env``.
    """

    def __init__(self, featurizer: SOAPFeaturizer, *, cross_lambdas: tuple[int, ...] = (0,)) -> None:
        if 0 not in {int(v) for v in cross_lambdas}:
            raise ValueError(
                "cross_lambdas must include 0; the invariant cross block is what conditions "
                "the parameter deltas and nothing reads the equivariant ones without it"
            )
        super().__init__(featurizer, cross_lambdas=cross_lambdas)

    def forward(self, batch, state: StateDescriptor) -> ProjectedFeatures:
        return super().forward(batch, edge_weight=state.edge_comembership)
