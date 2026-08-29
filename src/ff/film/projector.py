"""Shared primitives, state-dependent projection, and the three feature blocks.

``docs/fff_film.md`` §4 in code. One neighbor search and one radial x angular expansion per
batch -- the primitives are fragmentation-independent -- then the assignment matrix projects
the *edge contributions* before any nonlinear contraction:

    rho_in  = sum_j P_ij psi_ij          rho_env = sum_j (1 - P_ij) psi_ij

    x_in    = P(rho_in)                  the internal power spectrum
    x_env   = P(rho_env)                 the environmental power spectrum
    x_cross = rho_in (x) rho_env         the bilinear cross block (new relative to v4)

The first two reuse :class:`rsfff.features.features.FlatLambdaSOAPFeaturizer` exactly as the
v4 two-slot descriptor does (its ``edge_weight`` path *is* this projection); what forces this
module to orchestrate the featurizer's primitives directly rather than call its ``forward`` is
the cross block, which needs the raw densities ``A_in``/``A_env`` and not their finished
spectra.

Vertex guarantee, structural: at a one-hot ``C`` the co-membership is an exact 0/1 per edge,
so a lone fragment's environment density is an empty sum -- ``x_env``, ``x_cross`` and the
environment activity ``a_env`` are exact zeros, not small numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...features.features import (
    FlatLambdaSOAPFeaturizer,
    LambdaFeatures,
    _build_cross_power_spectrum_specs,
    cosine_cutoff,
    cross_equivariant_power_spectrum,
)
from .state import StateDescriptor

__all__ = ["FragmentProjector", "ProjectedFeatures"]


@dataclass
class ProjectedFeatures:
    """The three feature blocks plus the shared graph bookkeeping.

    ``x_in`` / ``x_env`` : :class:`LambdaFeatures` of the internal / environment densities.
    ``x_cross``          : ``{lam: (N, 2*lam+1, P)}`` cross blocks; ``cross_inv`` is the
                           lambda=0 slice ``(N, P)`` every invariant consumer reads.
    ``a_env``            : ``(N,)`` smooth nonnegative environment activity
                           ``sum_j (1 - P_ij) f_cut(r_ij)`` -- the gate argument of
                           ``docs/fff_film.md`` §5.1, exactly zero for an isolated fragment.
    ``edge_index``       : the one neighbor list everything above was built on.
    ``P_edge``           : ``(E,)`` co-membership on those edges, cached for reuse.
    """

    x_in: LambdaFeatures
    x_env: LambdaFeatures
    x_cross: dict[int, torch.Tensor]
    a_env: torch.Tensor
    edge_index: torch.Tensor
    P_edge: torch.Tensor

    @property
    def cross_inv(self) -> torch.Tensor:
        return self.x_cross[0][:, 0, :]


class FragmentProjector(nn.Module):
    """Wraps one :class:`FlatLambdaSOAPFeaturizer` and applies the ``C`` projection.

    The featurizer is used for everything it already owns -- the neighbor search, the Bessel x
    spherical-harmonic edge expansion, the optional channel compression, the power spectrum
    (and optional bispectrum) -- so the internal and environment blocks here are bit-compatible
    with the v4 two-slot descriptor at a one-hot assignment. The only new computation is the
    cross spectrum and the scalar ``a_env``.
    """

    def __init__(
        self,
        featurizer: FlatLambdaSOAPFeaturizer,
        *,
        cross_lambdas: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.cross_lambdas = tuple(sorted({int(v) for v in cross_lambdas}))
        if 0 not in self.cross_lambdas:
            raise ValueError(
                "cross_lambdas must include 0; the invariant cross block is what conditions "
                "the parameter deltas and nothing reads the equivariant ones without it"
            )
        n_channels = featurizer.density_channels or (
            featurizer.n_species * featurizer.n_max
        )
        self._cross_specs, buffers = _build_cross_power_spectrum_specs(
            l_max=featurizer.l_max,
            selected_lambdas=self.cross_lambdas,
            n_channels_a=n_channels,
            n_channels_b=n_channels,
            backend=featurizer.backend,
        )
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)
        #: per-lambda cross widths, for head sizing.
        self.cross_dims = {
            lam: sum(spec.width for spec in specs)
            for lam, specs in self._cross_specs.items()
        }

    @property
    def cutoff(self) -> float:
        return self.featurizer.cutoff

    @property
    def feature_dims(self) -> dict[int, int]:
        """Per-lambda widths of ``x_in`` (identical for ``x_env``)."""
        return self.featurizer.feature_dims

    def species_index(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        return self.featurizer._species_lut[atomic_numbers]

    @torch.compiler.disable
    def forward(self, batch, state: StateDescriptor) -> ProjectedFeatures:
        feat = self.featurizer
        positions = batch.positions
        species_idx = self.species_index(batch.atomic_numbers)
        n_atoms = int(positions.shape[0])

        edge_index = feat._build_edges(positions, batch.batch_idx)
        RY = feat.density.edge_expansion(positions, edge_index)
        P_e = state.edge_comembership(edge_index)

        A_in = feat._compress(
            feat.density.scatter_species(
                RY, edge_index, species_idx, n_atoms, edge_weight=P_e
            )
        )
        A_env = feat._compress(
            feat.density.scatter_species(
                RY, edge_index, species_idx, n_atoms, edge_weight=1.0 - P_e
            )
        )

        x_in = feat._features_from_density(A_in, species_idx, batch.batch_idx, edge_index)
        x_env = feat._features_from_density(
            A_env, species_idx, batch.batch_idx, edge_index
        )
        x_cross = cross_equivariant_power_spectrum(
            A_in,
            A_env,
            feat.l_max,
            self.cross_lambdas,
            specs_by_lam=self._cross_specs,
            cg_buffer_owner=self,
        )

        # Smooth environment activity: how much environment density each center receives.
        # Built from the same cutoff as the density so it vanishes exactly when and where the
        # environment slot does, and smoothly in between.
        i, j = edge_index[0], edge_index[1]
        r = (positions[j] - positions[i]).norm(dim=-1)
        w = (1.0 - P_e) * cosine_cutoff(r, feat.cutoff)
        a_env = torch.zeros(
            n_atoms, dtype=w.dtype, device=w.device
        ).index_add_(0, i, w)

        return ProjectedFeatures(
            x_in=x_in,
            x_env=x_env,
            x_cross=x_cross,
            a_env=a_env,
            edge_index=edge_index,
            P_edge=P_e,
        )
