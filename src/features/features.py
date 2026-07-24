"""
GPU-native SOAP / ACE-style atomic features using e3nn.

The pipeline:

    edge_index, positions, species
        |
        v
    DensityExpansion  ->  A[i, Z, n, l, m]    (per-atom equivariant density)
        |
        v
    Body-order construction (nu = 2, 3, ...)
        |
        v
    Invariant or equivariant per-atom features

The density expansion does one scatter-sum on the GPU and is shared across
all body orders.  Body-order construction is a small number of Clebsch-Gordan
contractions on top of A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn
from torch_cluster import radius_graph

if TYPE_CHECKING:
    from .equivariant_backend import EquivariantBackend

    # ``Batch`` is duck-typed: any object exposing ``positions`` (N, 3),
    # ``atomic_numbers`` (N,), and ``batch_idx`` (N,) tensors works. The concrete
    # dataclass lives in the consuming project, so we only need the name for hints.
    from typing import Any as Batch


def _resolve_backend(backend):
    """Accept an :class:`EquivariantBackend` instance or a name string ("e3nn"/"cue")."""
    if isinstance(backend, str):
        try:
            from .equivariant_backend import get_backend
        except ImportError:
            from equivariant_backend import get_backend
        return get_backend(backend)
    return backend


# --------------------------------------------------------------------------
# Radial basis and cutoff
# --------------------------------------------------------------------------

class BesselBasis(nn.Module):
    """Smooth Bessel radial basis (DimeNet-style), orthonormal on [0, r_cut]."""

    def __init__(self, n_max: int, cutoff: float):
        super().__init__()
        self.n_max = n_max
        self.cutoff = cutoff
        freqs = torch.arange(1, n_max + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)
        self.norm = math.sqrt(2.0 / cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (E,)  ->  (E, n_max)
        r_safe = r.clamp(min=1e-8).unsqueeze(-1)
        return self.norm * torch.sin(self.freqs * r_safe / self.cutoff) / r_safe


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cosine cutoff that goes to 0 at r = cutoff with zero derivative."""
    return 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0) * (r < cutoff).to(r.dtype)


# --------------------------------------------------------------------------
# Density expansion: the A coefficients
# --------------------------------------------------------------------------

class DensityExpansion(nn.Module):
    """
    A[i, Z, n, lm] = sum_{j ~ i, Z_j = Z} R_n(r_ij) * Y_lm(r_hat_ij) * f_cut(r_ij)

    Output shape:  (N, n_species, n_max, (l_max + 1)**2)
    """

    def __init__(self, n_max: int, l_max: int, cutoff: float, n_species: int,
                 backend: "EquivariantBackend | str" = "e3nn"):
        super().__init__()
        self.n_max = n_max
        self.l_max = l_max
        self.cutoff = cutoff
        self.n_species = n_species
        self.sph_dim = (l_max + 1) ** 2
        self.backend = _resolve_backend(backend)
        self.radial = BesselBasis(n_max, cutoff)

    def edge_expansion(
        self,
        positions: torch.Tensor,     # (N, 3)
        edge_index: torch.Tensor,    # (2, E)  rows are [center, neighbor]
    ) -> torch.Tensor:
        """Per-edge radial x angular expansion ``RY``: (E, n_max, sph_dim).

        The geometry-only part of the density, independent of how neighbors are weighted.
        Splitting it out lets several decorations of the same geometry share one basis
        construction -- the cross-diabat amortization of §2.1 ("the geometric density basis is
        state-independent and built once per configuration").
        """
        i, j = edge_index[0], edge_index[1]
        r_vec = positions[j] - positions[i]
        r = r_vec.norm(dim=-1)
        R = self.radial(r) * cosine_cutoff(r, self.cutoff).unsqueeze(-1)   # (E, n_max)
        Y = self.backend.spherical_harmonics(r_vec, self.l_max, normalize=True)  # (E, sph_dim)
        return R.unsqueeze(-1) * Y.unsqueeze(-2)                           # (E, n_max, sph_dim)

    def scatter_weighted(
        self,
        RY: torch.Tensor,            # (E, n_max, sph_dim) from edge_expansion
        edge_index: torch.Tensor,    # (2, E)
        atom_weights: torch.Tensor,  # (N, Kw) per-atom channel weights
        num_atoms: int,
    ) -> torch.Tensor:
        """Weighted density ``A[i,k,n,lm] = sum_{j~i} w_k(j) RY[ij,n,lm]``: (N, Kw, n_max, sph).

        Generalizes the species one-hot scatter of :meth:`forward`: with
        ``atom_weights = one_hot(species_idx)`` it reproduces that exactly, and with weights
        derived from reference embeddings it becomes the *state-decorated* density of §2.1 --
        neighbors enter through their diabatic identity rather than their bare element. The
        weights are invariant scalars, so equivariance is untouched.
        """
        i, j = edge_index[0], edge_index[1]
        k_w = atom_weights.shape[1]
        contrib = atom_weights[j][:, :, None, None] * RY[:, None]  # (E, Kw, n_max, sph)
        A = RY.new_zeros(num_atoms, k_w, self.n_max, self.sph_dim)
        return A.index_add_(0, i, contrib)

    def forward(
        self,
        positions: torch.Tensor,     # (N, 3)
        species_idx: torch.Tensor,   # (N,) ints in [0, n_species)
        edge_index: torch.Tensor,    # (2, E)  rows are [center, neighbor]
        num_atoms: int,
    ) -> torch.Tensor:
        RY = self.edge_expansion(positions, edge_index)                    # (E, n_max, sph_dim)
        i, j = edge_index[0], edge_index[1]
        z_j = species_idx[j]
        flat_idx = i * self.n_species + z_j
        A_flat = positions.new_zeros(num_atoms * self.n_species, self.n_max, self.sph_dim)
        A_flat.index_add_(0, flat_idx, RY)
        return A_flat.view(num_atoms, self.n_species, self.n_max, self.sph_dim)


# --------------------------------------------------------------------------
# Body-order invariants
# --------------------------------------------------------------------------

def _split_by_l(x: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """Split last dim of x (size (l_max+1)**2) into list of (..., 2l+1) per l."""
    out, start = [], 0
    for l in range(l_max + 1):
        size = 2 * l + 1
        out.append(x[..., start:start + size])
        start += size
    return out


def power_spectrum(A: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    nu = 2 invariants (SOAP power spectrum):
        p[i, Z, n, Z', n', l] = (1/sqrt(2l+1)) * sum_m A[i,Z,n,l,m] * A[i,Z',n',l,m]

    A: (N, Z, n, (l_max+1)**2)
    Returns: (N, n_features)
    """
    N = A.shape[0]
    feats = []
    for l, A_l in enumerate(_split_by_l(A, l_max)):
        # (N, Z, n, 2l+1) x (N, Z', n', 2l+1) -> (N, Z, n, Z', n')
        p = torch.einsum("nzim,nyjm->nzyij", A_l, A_l) / math.sqrt(2 * l + 1)
        feats.append(p.reshape(N, -1))
    return torch.cat(feats, dim=-1)


def bispectrum(A: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    nu = 3 invariants (SOAP bispectrum):
        B[i, ..., l1, l2, l3] = sum_{m1 m2 m3} CG^{l1 l2 l3}_{m1 m2 m3}
                                * A[i,Z1,n1,l1,m1] * A[i,Z2,n2,l2,m2] * A[i,Z3,n3,l3,m3]

    Only (l1, l2, l3) with valid triangle and l1+l2+l3 even contribute.

    A: (N, Z, n, (l_max+1)**2)
    Returns: (N, n_features)
    """
    from e3nn import o3  # lazy: legacy invariant helper, not on the fluctuation path
    N, Z, n, _ = A.shape
    A_by_l = _split_by_l(A, l_max)  # each (N, Z, n, 2l+1)
    feats = []
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            for l3 in range(l2, l_max + 1):
                if (l1 + l2 + l3) % 2 != 0:
                    continue                       # parity selection rule
                if l3 > l1 + l2 or l3 < abs(l1 - l2):
                    continue                       # triangle inequality
                cg = o3.wigner_3j(l1, l2, l3).to(A)   # (2l1+1, 2l2+1, 2l3+1)
                # b[i, Z1,n1, Z2,n2, Z3,n3] = sum_{m1,m2,m3} cg[m1,m2,m3]
                #                              * A_l1[i,Z1,n1,m1]
                #                              * A_l2[i,Z2,n2,m2]
                #                              * A_l3[i,Z3,n3,m3]
                b = torch.einsum(
                    "abc,nzia,nyjb,nxkc->nzyxijk",
                    cg, A_by_l[l1], A_by_l[l2], A_by_l[l3],
                )
                feats.append(b.reshape(N, -1))
    return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------
# Equivariant lambda-power-spectrum (lambda-SOAP, body order nu = 2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _EquivariantPairSpec:
    lam: int
    l1: int
    l2: int
    width: int
    cg_buffer_name: str


def _build_equivariant_power_spectrum_specs(
    *,
    l_max: int,
    selected_lambdas: Sequence[int],
    n_channels: int,
    backend: "EquivariantBackend | str" = "e3nn",
) -> tuple[dict[int, tuple[_EquivariantPairSpec, ...]], dict[str, torch.Tensor]]:
    """Build reusable metadata and constant tensors for lambda-SOAP."""
    backend = _resolve_backend(backend)
    specs: dict[int, list[_EquivariantPairSpec]] = {}
    buffers: dict[str, torch.Tensor] = {}
    cg_index = 0
    for lam in sorted({int(v) for v in selected_lambdas}):
        lam_specs: list[_EquivariantPairSpec] = []
        for l1 in range(l_max + 1):
            for l2 in range(l1, l_max + 1):
                if l2 < abs(l1 - lam) or l2 > l1 + lam:
                    continue
                if (l1 + l2 + lam) % 2 != 0:
                    continue
                cg_name = f"_equivariant_cg_{cg_index}"
                buffers[cg_name] = backend.wigner_3j(l1, l2, lam).to(torch.get_default_dtype())
                cg_index += 1
                width = (
                    n_channels * (n_channels + 1) // 2
                    if l1 == l2 else n_channels * n_channels
                )
                lam_specs.append(
                    _EquivariantPairSpec(
                        lam=lam,
                        l1=l1,
                        l2=l2,
                        width=width,
                        cg_buffer_name=cg_name,
                    )
                )
        if not lam_specs:
            raise ValueError(
                f"no (l1, l2) density pairs couple to lambda={lam} with l_max={l_max}; "
                "increase max_angular"
            )
        specs[lam] = lam_specs
    rows, cols = torch.triu_indices(n_channels, n_channels, offset=0)
    buffers["_equivariant_triu_rows"] = rows
    buffers["_equivariant_triu_cols"] = cols
    return {lam: tuple(v) for lam, v in specs.items()}, buffers


def equivariant_power_spectrum(
    A: torch.Tensor,
    l_max: int,
    selected_lambdas: Sequence[int],
    specs_by_lam: dict[int, tuple[_EquivariantPairSpec, ...]] | None = None,
    triu_rows: torch.Tensor | None = None,
    triu_cols: torch.Tensor | None = None,
    cg_lookup: dict[str, torch.Tensor] | None = None,
    cg_buffer_owner: nn.Module | None = None,
    backend: "EquivariantBackend | str" = "e3nn",
) -> dict[int, torch.Tensor]:
    """Equivariant SOAP power spectrum (symmetry-reduced).

        p^{lam}_mu[i, a, b] = sum_{m1,m2} W^{l1 l2 lam}_{m1 m2 mu}
                              * A[i, a, m1] * A[i, b, m2]

    where `a = (Z, n, l1)` and `b = (Z', n', l2)` are composite density channels
    ((Z, n) flattened into a single channel axis of size `C = n_species * n_max`).

    The two density factors are interchangeable: for the parity-selected even
    couplings used here (`l1 + l2` even, `lam` in {0, 2}) the 3j column-swap
    symmetry gives `p[a, b] == p[b, a]` exactly. We therefore enumerate only
    unordered pairs (`l1 <= l2`, and `c1 <= c2` when `l1 == l2`), which removes the
    redundant half-to-three-quarters of the channels with no loss of information.

    `A`: `(N, Z, n, (l_max+1)**2)` density coefficients from `DensityExpansion`.

    Returns a dict mapping each requested `lam` to a tensor of shape
    `(N, 2*lam + 1, P_lam)`, with the component axis in the middle to mirror the
    featomic block layout consumed by the network heads. `P_lam` is fixed by
    `n_species`/`n_max`/`l_max`, independent of which neighbor species actually
    appear in a given structure.
    """
    n_atoms, n_species, n_max, _ = A.shape
    n_channels = n_species * n_max
    # per-l composite channels: (N, C, 2l+1), C = n_species * n_max
    A_by_l = [
        a.reshape(n_atoms, n_channels, a.shape[-1]) for a in _split_by_l(A, l_max)
    ]
    if specs_by_lam is None:
        specs_by_lam, buffers = _build_equivariant_power_spectrum_specs(
            l_max=l_max, selected_lambdas=selected_lambdas, n_channels=n_channels,
            backend=backend,
        )
        triu_rows = buffers["_equivariant_triu_rows"].to(device=A.device)
        triu_cols = buffers["_equivariant_triu_cols"].to(device=A.device)
        cg_lookup = {
            name: tensor.to(device=A.device, dtype=A.dtype)
            for name, tensor in buffers.items()
            if name.startswith("_equivariant_cg_")
        }
    elif triu_rows is None or triu_cols is None or (cg_lookup is None and cg_buffer_owner is None):
        raise ValueError(
            "specs_by_lam requires matching triu_rows, triu_cols, and cached Wigner tensors"
        )

    out: dict[int, torch.Tensor] = {}
    for lam in selected_lambdas:
        lam = int(lam)
        lam_specs = specs_by_lam[lam]
        lam_out = A.new_empty(n_atoms, 2 * lam + 1, sum(spec.width for spec in lam_specs))
        offset = 0
        for spec in lam_specs:
            cg = (
                cg_lookup[spec.cg_buffer_name]
                if cg_lookup is not None else getattr(cg_buffer_owner, spec.cg_buffer_name)
            )
            if cg.device != A.device or cg.dtype != A.dtype:
                cg = cg.to(device=A.device, dtype=A.dtype)
            # -> (N, 2lam+1, C, C)
            p = torch.einsum("abc,nia,njb->ncij", cg, A_by_l[spec.l1], A_by_l[spec.l2])
            next_offset = offset + spec.width
            if spec.l1 == spec.l2:
                lam_out[:, :, offset:next_offset] = p[:, :, triu_rows, triu_cols]
            else:
                lam_out[:, :, offset:next_offset] = p.reshape(n_atoms, p.shape[1], spec.width)
            offset = next_offset
        out[lam] = lam_out
    return out


# --------------------------------------------------------------------------
# Wrapper module
# --------------------------------------------------------------------------

class SoapAceFeatures(nn.Module):
    """
    GPU-native SOAP / ACE-style atomic features.

    Args
    ----
    n_max : number of radial Bessel functions
    l_max : maximum spherical harmonic degree
    cutoff : radial cutoff (Angstrom or whatever units you use)
    n_species : number of distinct atomic species in the dataset
    body_order : 2 -> SOAP power spectrum
                 3 -> bispectrum
                 Use return_density=True to get the bare A coefficients
                 (equivalent to nu=1, equivariant) instead.
    return_density : if True, return the raw (equivariant) A tensor.

    Forward inputs
    --------------
    positions : (N, 3) float
    species_idx : (N,) long, values in [0, n_species)
    edge_index : (2, E) long; row 0 = center, row 1 = neighbor.
                 You provide this; any neighbor list builder works
                 (ase, matscipy, torch_cluster.radius_graph, ...).
    num_atoms : int, total atoms in batch.
    """

    def __init__(
        self,
        n_max: int,
        l_max: int,
        cutoff: float,
        n_species: int,
        body_order: int = 2,
        return_density: bool = False,
        backend: "EquivariantBackend | str" = "e3nn",
    ):
        super().__init__()
        if body_order not in (2, 3) and not return_density:
            raise ValueError("body_order must be 2 or 3 in this implementation. "
                             "For higher orders, iterate CG products on A.")
        self.density = DensityExpansion(n_max, l_max, cutoff, n_species, backend=backend)
        self.l_max = l_max
        self.body_order = body_order
        self.return_density = return_density

    def forward(self, positions, species_idx, edge_index, num_atoms):
        A = self.density(positions, species_idx, edge_index, num_atoms)
        if self.return_density:
            return A
        if self.body_order == 2:
            return power_spectrum(A, self.l_max)
        if self.body_order == 3:
            return bispectrum(A, self.l_max)
        raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Pipeline-facing flat featurizer
# --------------------------------------------------------------------------

@dataclass
class LambdaFeatures:
    """Per-atom equivariant features for a flat (ragged) batch.

    `inv_feats`:   (Ntot, P0)        lambda=0 invariants
    `equiv_feats`: (Ntot, 5, P2)     lambda=2 equivariant channels
    `species_idx`: (Ntot,)           species index per atom, in [0, n_species)
    `batch_idx`:   (Ntot,)           molecule id per atom
    `vec_feats`:   (Ntot, 3, P1)     lambda=1 (odd-parity, vector) channels, or
                                     ``None`` when lambda=1 was not requested.
    `edge_index`:  (2, E)            the `radius_graph` neighbor list used to build
                                     the features (row 0 = center i, row 1 = neighbor
                                     j); shared so downstream consumers (e.g. the
                                     interaction-tensor relay) use the same cutoff.
    """

    inv_feats: torch.Tensor
    equiv_feats: torch.Tensor
    species_idx: torch.Tensor
    batch_idx: torch.Tensor
    vec_feats: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None


class FlatLambdaSOAPFeaturizer(nn.Module):
    """Vectorized equivariant lambda-SOAP featurizer over a flat ragged batch.

    Edges are built in one fused call with `torch_cluster.radius_graph`; the
    density and the equivariant power spectrum are the existing scatter/CG kernels.
    There is no per-structure Python loop and no `nonzero`/`.item()`.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        n_max: int,
        l_max: int,
        neighbor_types: Sequence[int],
        selected_lambdas: Sequence[int] = (0, 2),
        backend: "EquivariantBackend | str" = "e3nn",
        bispectrum=None,
        density_channels: int | None = None,
    ) -> None:
        super().__init__()
        cleaned = tuple(sorted({int(z) for z in neighbor_types}))
        if not cleaned:
            raise ValueError("neighbor_types can not be empty")
        self.cutoff = float(cutoff)
        self.l_max = int(l_max)
        self.n_max = int(n_max)
        self.n_species = len(cleaned)
        self.selected_lambdas = tuple(sorted({int(v) for v in selected_lambdas}))
        self.backend = _resolve_backend(backend)

        # Vectorized atomic-number -> species-index lookup (no per-atom Python loop).
        lut = torch.full((max(cleaned) + 1,), -1, dtype=torch.long)
        for index, z in enumerate(cleaned):
            lut[z] = index
        self.register_buffer("_species_lut", lut, persistent=False)

        self.density = DensityExpansion(
            n_max=self.n_max, l_max=self.l_max, cutoff=self.cutoff,
            n_species=self.n_species, backend=self.backend,
        )

        # Optional learnable density-channel compression. The raw density has
        # C = n_species * n_max channels; the power spectrum is O(C^2), which explodes
        # with the element count. A single equivariant linear map on the channel axis
        # (mixing (Z, n) channels, leaving the m components untouched) projects C -> Kc
        # before the power spectrum / bispectrum, so the feature width scales as Kc^2.
        raw_channels = self.n_species * self.n_max
        self.density_channels = int(density_channels) if density_channels else None
        n_channels = self.density_channels or raw_channels
        if self.density_channels is not None:
            self.channel_proj = nn.Parameter(
                torch.randn(raw_channels, self.density_channels) / (raw_channels ** 0.5)
            )

        self._equivariant_specs, cached_tensors = _build_equivariant_power_spectrum_specs(
            l_max=self.l_max,
            selected_lambdas=self.selected_lambdas,
            n_channels=n_channels,
            backend=self.backend,
        )
        for name, tensor in cached_tensors.items():
            self.register_buffer(name, tensor, persistent=False)
        # Base (nu=2) per-lambda power-spectrum widths.
        self.feature_dims = {
            lam: sum(spec.width for spec in specs)
            for lam, specs in self._equivariant_specs.items()
        }

        # Optional trainable weighted bispectrum (nu=3), concatenated onto the nu=2
        # power spectrum per lambda. Presence of the `bispectrum` config enables it;
        # it is built through the same backend so it follows the e3nn/cue split. It
        # consumes the same (optionally compressed) density, so its channel count is Kc.
        self.bispectrum = None
        if bispectrum is not None:
            num_elements = (
                self.n_species if getattr(bispectrum, "per_species_weights", False) else 1
            )
            # Pass the effective channel count as (n_species=n_channels, n_max=1); only
            # the product matters. num_elements (center element) is set independently.
            bis_species, bis_nmax = (
                (n_channels, 1) if self.density_channels is not None
                else (self.n_species, self.n_max)
            )
            self.bispectrum = self.backend.make_weighted_bispectrum(
                n_species=bis_species,
                n_max=bis_nmax,
                l_max=self.l_max,
                selected_lambdas=self.selected_lambdas,
                out_mul=getattr(bispectrum, "out_mul", None),
                num_elements=num_elements,
                implementation=getattr(bispectrum, "implementation", "sym_contraction"),
                contraction_degree=getattr(bispectrum, "contraction_degree", 3),
                degrees=tuple(getattr(bispectrum, "degrees", (3,))),
                method=getattr(bispectrum, "method", "auto"),
            )
            self._bispectrum_on_cpu = self.backend.name == "cue" and not torch.cuda.is_available()
            for lam, width in self.bispectrum.feature_dims.items():
                self.feature_dims[lam] = self.feature_dims.get(lam, 0) + width

    def _build_edges(self, positions: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        # torch_cluster has CPU+CUDA kernels but no MPS kernel; fall back to CPU on MPS.
        if positions.device.type == "mps":
            return radius_graph(
                positions.cpu(), r=self.cutoff, batch=batch_idx.cpu(), loop=False
            ).to(positions.device)
        return radius_graph(positions, r=self.cutoff, batch=batch_idx, loop=False)

    def _run_bispectrum(self, A: torch.Tensor, species_idx: torch.Tensor) -> dict:
        # cuequivariance has no MPS kernel; off CUDA the cue bispectrum runs on CPU
        # (method="naive") and results are moved back. e3nn bispectrum runs natively.
        if getattr(self, "_bispectrum_on_cpu", False) and A.device.type != "cpu":
            dev = A.device
            bis = self.bispectrum(A.cpu(), species_idx.cpu())
            return {lam: t.to(dev) for lam, t in bis.items()}
        return self.bispectrum(A, species_idx)

    # The featurizer is a torch.compile boundary (runs eager). `torch_cluster.radius`
    # is an opaque custom op dynamo cannot trace (it also calls `.item()` internally),
    # and the e3nn spherical_harmonics / wigner_3j here previously triggered an inductor
    # recompile storm. Excluding it from the graph is cheap (radius build is fast, the
    # power-spectrum is already efficient cuBLAS bmm) and lets inductor cleanly compile
    # the dense readout in `pf_model`, which is where fusion actually helps.
    @torch.compiler.disable
    def forward(self, batch: "Batch") -> LambdaFeatures:
        positions = batch.positions
        species_idx = self._species_lut[batch.atomic_numbers]
        edge_index = self._build_edges(positions, batch.batch_idx)
        # radius_graph returns directed edges; DensityExpansion treats edge_index[0]
        # as the center i and [1] as the neighbor j. The graph is symmetric so either
        # orientation yields the same density.
        A = self.density(positions, species_idx, edge_index, int(positions.shape[0]))
        # Optional learnable channel compression: (N, n_species, n_max, L2) -> (N, Kc, 1, L2)
        # before the O(Kc^2) power spectrum / bispectrum.
        if self.density_channels is not None:
            n_atoms, _, _, l2 = A.shape
            A = torch.einsum("ncl,ck->nkl", A.reshape(n_atoms, -1, l2), self.channel_proj)
            A = A.unsqueeze(2)                                        # (N, Kc, 1, L2)
        per_lambda = equivariant_power_spectrum(
            A,
            self.l_max,
            self.selected_lambdas,
            specs_by_lam=self._equivariant_specs,
            triu_rows=self._equivariant_triu_rows,
            triu_cols=self._equivariant_triu_cols,
            cg_buffer_owner=self,
        )
        inv = per_lambda[0][:, 0, :]          # (Ntot, P0)  squeeze the single component
        equiv = per_lambda[2]                 # (Ntot, 5, P2)
        vec = per_lambda.get(1)               # (Ntot, 3, P1) odd-parity vector, or None

        # Trainable nu=3 bispectrum, concatenated per lambda onto the nu=2 features.
        if self.bispectrum is not None:
            bis = self._run_bispectrum(A, species_idx)
            inv = torch.cat((inv, bis[0][:, 0, :]), dim=-1)          # (Ntot, P0 + out_mul)
            equiv = torch.cat((equiv, bis[2]), dim=-1)               # (Ntot, 5, P2 + out_mul)
            if 1 in bis and vec is not None:
                vec = torch.cat((vec, bis[1]), dim=-1)               # (Ntot, 3, P1 + out_mul)

        return LambdaFeatures(
            inv_feats=inv,
            equiv_feats=equiv,
            species_idx=species_idx,
            batch_idx=batch.batch_idx,
            vec_feats=vec,
            edge_index=edge_index,
        )


class FlatStateSOAPFeaturizer(nn.Module):
    """State-decorated intra-fragment lambda-SOAP featurizer (``F_mono`` of §2.1).

    Same pipeline as :class:`FlatLambdaSOAPFeaturizer` -- one fused ``radius_graph``, one
    density scatter, the same CG power spectrum and optional trainable bispectrum -- with two
    changes that make it the monomer stack's featurizer rather than a generic descriptor:

    1. **The density channel axis is learned, not a species one-hot.** Neighbors are weighted by
       ``atom_weights`` (``(N, Kw)``, supplied by the caller from the reference embeddings; see
       ``rsfff.mlip.reference_states.AtomWeightNet``), so a neighbor enters through its diabatic
       identity -- element *and* fragment charge/spin -- instead of its bare element. This
       subsumes ``FlatLambdaSOAPFeaturizer``'s ``density_channels`` compression: the weight net
       *is* the compression, and it is state-aware.
    2. **Edges are confined to the fragment.** ``fragment_idx`` (not ``batch_idx``) is the
       ``radius_graph`` grouping, so message passing never crosses a fragment boundary. In
       Phase 1 there is one fragment per frame and this coincides with the per-frame graph; the
       argument is what keeps the monomer features frozen-and-isolated once several fragments
       share a configuration.

    Feature width scales as ``(Kw * n_max)^2``, so ``Kw`` is the knob controlling cost.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        n_max: int,
        l_max: int,
        neighbor_types: Sequence[int],
        weight_channels: int,
        selected_lambdas: Sequence[int] = (0, 1, 2),
        backend: "EquivariantBackend | str" = "e3nn",
        bispectrum=None,
    ) -> None:
        super().__init__()
        cleaned = tuple(sorted({int(z) for z in neighbor_types}))
        if not cleaned:
            raise ValueError("neighbor_types can not be empty")
        self.cutoff = float(cutoff)
        self.l_max = int(l_max)
        self.n_max = int(n_max)
        self.n_species = len(cleaned)
        self.weight_channels = int(weight_channels)
        self.selected_lambdas = tuple(sorted({int(v) for v in selected_lambdas}))
        self.backend = _resolve_backend(backend)

        lut = torch.full((max(cleaned) + 1,), -1, dtype=torch.long)
        for index, z in enumerate(cleaned):
            lut[z] = index
        self.register_buffer("_species_lut", lut, persistent=False)

        # n_species is the *weight* channel count here: the density scatter is over learned
        # channels, and DensityExpansion only uses it to size the one-hot path we bypass.
        self.density = DensityExpansion(
            n_max=self.n_max, l_max=self.l_max, cutoff=self.cutoff,
            n_species=self.weight_channels, backend=self.backend,
        )

        n_channels = self.weight_channels * self.n_max
        self._equivariant_specs, cached_tensors = _build_equivariant_power_spectrum_specs(
            l_max=self.l_max,
            selected_lambdas=self.selected_lambdas,
            n_channels=n_channels,
            backend=self.backend,
        )
        for name, tensor in cached_tensors.items():
            self.register_buffer(name, tensor, persistent=False)
        self.feature_dims = {
            lam: sum(spec.width for spec in specs)
            for lam, specs in self._equivariant_specs.items()
        }

        self.bispectrum = None
        if bispectrum is not None:
            num_elements = (
                self.n_species if getattr(bispectrum, "per_species_weights", False) else 1
            )
            self.bispectrum = self.backend.make_weighted_bispectrum(
                n_species=self.weight_channels,
                n_max=self.n_max,
                l_max=self.l_max,
                selected_lambdas=self.selected_lambdas,
                out_mul=getattr(bispectrum, "out_mul", None),
                num_elements=num_elements,
                implementation=getattr(bispectrum, "implementation", "sym_contraction"),
                contraction_degree=getattr(bispectrum, "contraction_degree", 3),
                degrees=tuple(getattr(bispectrum, "degrees", (3,))),
                method=getattr(bispectrum, "method", "auto"),
            )
            self._bispectrum_on_cpu = self.backend.name == "cue" and not torch.cuda.is_available()
            for lam, width in self.bispectrum.feature_dims.items():
                self.feature_dims[lam] = self.feature_dims.get(lam, 0) + width

    def species_index(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Map atomic numbers to this featurizer's species indices: (N,) -> (N,).

        Public because callers that build per-atom inputs *before* featurization (the reference
        embeddings of the monomer stack, for instance) need the same indexing the features will
        carry, and should not have to reach into the lookup buffer.
        """
        return self._species_lut[atomic_numbers]

    def _build_edges(self, positions: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        # torch_cluster has CPU+CUDA kernels but no MPS kernel; fall back to CPU on MPS.
        if positions.device.type == "mps":
            return radius_graph(
                positions.cpu(), r=self.cutoff, batch=group_idx.cpu(), loop=False
            ).to(positions.device)
        return radius_graph(positions, r=self.cutoff, batch=group_idx, loop=False)

    def _run_bispectrum(self, A: torch.Tensor, species_idx: torch.Tensor) -> dict:
        if getattr(self, "_bispectrum_on_cpu", False) and A.device.type != "cpu":
            dev = A.device
            bis = self.bispectrum(A.cpu(), species_idx.cpu())
            return {lam: t.to(dev) for lam, t in bis.items()}
        return self.bispectrum(A, species_idx)

    # See FlatLambdaSOAPFeaturizer.forward for why this is a torch.compile boundary.
    @torch.compiler.disable
    def forward(
        self,
        batch: "Batch",
        atom_weights: torch.Tensor,          # (N, Kw) from the reference embeddings
        fragment_idx: torch.Tensor | None = None,
    ) -> LambdaFeatures:
        positions = batch.positions
        species_idx = self._species_lut[batch.atomic_numbers]
        if atom_weights.shape[-1] != self.weight_channels:
            raise ValueError(
                f"atom_weights has {atom_weights.shape[-1]} channels, featurizer was built "
                f"for weight_channels={self.weight_channels}"
            )
        group = fragment_idx if fragment_idx is not None else batch.batch_idx
        edge_index = self._build_edges(positions, group)

        RY = self.density.edge_expansion(positions, edge_index)
        A = self.density.scatter_weighted(
            RY, edge_index, atom_weights, int(positions.shape[0])
        )                                                     # (N, Kw, n_max, sph_dim)

        per_lambda = equivariant_power_spectrum(
            A,
            self.l_max,
            self.selected_lambdas,
            specs_by_lam=self._equivariant_specs,
            triu_rows=self._equivariant_triu_rows,
            triu_cols=self._equivariant_triu_cols,
            cg_buffer_owner=self,
        )
        inv = per_lambda[0][:, 0, :]
        equiv = per_lambda[2]
        vec = per_lambda.get(1)

        if self.bispectrum is not None:
            bis = self._run_bispectrum(A, species_idx)
            inv = torch.cat((inv, bis[0][:, 0, :]), dim=-1)
            equiv = torch.cat((equiv, bis[2]), dim=-1)
            if 1 in bis and vec is not None:
                vec = torch.cat((vec, bis[1]), dim=-1)

        return LambdaFeatures(
            inv_feats=inv,
            equiv_feats=equiv,
            species_idx=species_idx,
            batch_idx=batch.batch_idx,
            vec_feats=vec,
            edge_index=edge_index,
        )


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from e3nn import o3  # lazy: smoke test only

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N = 12
    cutoff = 5.0
    positions = torch.randn(N, 3, device=device) * 3.0
    species_idx = torch.randint(0, 3, (N,), device=device)

    # naive O(N^2) neighbor list for the smoke test
    diff = positions[:, None, :] - positions[None, :, :]
    dist = diff.norm(dim=-1)
    mask = (dist < cutoff) & (dist > 0)
    edge_index = mask.nonzero().t().contiguous()  # (2, E)

    feat = SoapAceFeatures(
        n_max=6, l_max=4, cutoff=cutoff, n_species=3, body_order=2
    ).to(device)
    p = feat(positions, species_idx, edge_index, N)
    print(f"power spectrum:  {tuple(p.shape)}")

    feat3 = SoapAceFeatures(
        n_max=4, l_max=3, cutoff=cutoff, n_species=3, body_order=3
    ).to(device)
    b = feat3(positions, species_idx, edge_index, N)
    print(f"bispectrum:      {tuple(b.shape)}")

    # rotation invariance check
    R = o3.rand_matrix().to(device)
    p_rot = feat(positions @ R.T, species_idx, edge_index, N)
    print(f"power spectrum rotation invariant?  "
          f"max err = {(p - p_rot).abs().max().item():.2e}")
    b_rot = feat3(positions @ R.T, species_idx, edge_index, N)
    print(f"bispectrum rotation invariant?      "
          f"max err = {(b - b_rot).abs().max().item():.2e}")

    # equivariant lambda-power-spectrum: lambda=0 invariant, lambda=2 covariant
    density = DensityExpansion(n_max=4, l_max=4, cutoff=cutoff, n_species=3).to(device)

    def lambda_feats(pos):
        A = density(pos, species_idx, edge_index, N)
        return equivariant_power_spectrum(A, l_max=4, selected_lambdas=(0, 2))

    f = lambda_feats(positions)
    f_rot = lambda_feats(positions @ R.T)
    err0 = (f[0] - f_rot[0]).abs().max().item()
    wigner_d2 = o3.Irrep(2, 1).D_from_matrix(R).to(f[2].dtype)
    f2_rot_expected = torch.einsum("ij,njp->nip", wigner_d2, f[2])
    err2 = (f2_rot_expected - f_rot[2]).abs().max().item()
    print(f"lambda=0 block {tuple(f[0].shape)} invariant?   max err = {err0:.2e}")
    print(f"lambda=2 block {tuple(f[2].shape)} covariant?   max err = {err2:.2e}")
