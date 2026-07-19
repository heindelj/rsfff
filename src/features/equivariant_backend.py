"""Two interchangeable equivariant backends.

The featurizer/heads need only a small set of O3 primitives:

  * ``spherical_harmonics(vectors, l_max)`` -- component-normalized real SH,
  * ``wigner_3j(l1, l2, l3)``               -- unit-Frobenius CG coupling tensor,
  * ``irrep6_to_voigt()``                   -- the fixed ``(0e + 2e) -> Voigt(6)`` map,
  * ``make_weighted_bispectrum(...)``       -- a trainable nu=3 lambda-bispectrum.

``E3nnBackend`` implements these with ``e3nn``;
``CueBackend`` implements them with ``cuequivariance``.

The density ``A`` and the nu=2 power spectrum are byte-compatible across
backends and only the *learnable* bispectrum weights differ (random init, trained per model).

Each backend imports **only its own** library, lazily. (Import-order caveat: on
some OpenMP builds importing ``cuequivariance_torch`` *before* ``e3nn`` in the same
process segfaults; normal use only touches one library, so this bites only
cross-backend tests, which must import e3nn first.)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# small shared helper (duplicated from features_native to avoid a circular import)
# --------------------------------------------------------------------------
def _split_by_l(x: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """Split the last dim (size ``(l_max+1)**2``) into per-l blocks ``(..., 2l+1)``."""
    out, start = [], 0
    for l in range(l_max + 1):
        size = 2 * l + 1
        out.append(x[..., start:start + size])
        start += size
    return out


# --------------------------------------------------------------------------
# backend interface
# --------------------------------------------------------------------------
class EquivariantBackend(ABC):
    """Minimal set of O3 primitives shared by the fluctuation featurizer/heads."""

    name: str

    @abstractmethod
    def spherical_harmonics(
        self, vectors: torch.Tensor, l_max: int, *, normalize: bool = True
    ) -> torch.Tensor:
        """Real spherical harmonics, component-normalized. ``(E, (l_max+1)**2)``."""

    @abstractmethod
    def wigner_3j(self, l1: int, l2: int, l3: int) -> torch.Tensor:
        """CG coupling ``(2l1+1, 2l2+1, 2l3+1)``, unit Frobenius norm (e3nn convention)."""

    def irrep6_to_voigt(self) -> torch.Tensor:
        """Constant ``(1x0e + 1x2e) -> Voigt(6)`` map, ``(6_voigt, 6_irrep)``.

        Built purely from ``wigner_3j``: the Cartesian image of each irrep basis
        vector is ``sqrt(2L+1) * wigner_3j(1, 1, L)[:, :, m]`` (verified against
        ``e3nn.io.CartesianTensor("ij=ji")``). Identical across backends, so the
        heads' change of basis -- and every checkpoint trained with it -- is stable.
        """
        w0 = self.wigner_3j(1, 1, 0).to(torch.float64)          # (3, 3, 1)
        w2 = self.wigner_3j(1, 1, 2).to(torch.float64)          # (3, 3, 5)
        basis = [math.sqrt(1.0) * w0[:, :, 0]] + [
            math.sqrt(5.0) * w2[:, :, m] for m in range(5)
        ]                                                        # 6 x (3, 3)
        # Voigt order xx yy zz yz xz xy
        rows = [
            torch.stack((B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]))
            for B in basis
        ]
        M = torch.stack(rows, dim=0)                            # (6_irrep, 6_voigt)
        return M.t().contiguous().to(torch.get_default_dtype())  # (6_voigt, 6_irrep)

    @abstractmethod
    def make_weighted_bispectrum(self, **kwargs) -> nn.Module:
        """Trainable nu=3 lambda-bispectrum: ``forward(A, species_idx) -> {lam: tensor}``."""


# --------------------------------------------------------------------------
# e3nn backend
# --------------------------------------------------------------------------
class E3nnBackend(EquivariantBackend):
    name = "e3nn"

    def __init__(self) -> None:
        from e3nn import o3  # lazy: never import e3nn unless this backend is built

        self._o3 = o3
        self._sh_irreps: dict[int, object] = {}
        self._w3j: dict[tuple[int, int, int], torch.Tensor] = {}

    def spherical_harmonics(self, vectors, l_max, *, normalize=True):
        irreps = self._sh_irreps.get(l_max)
        if irreps is None:
            irreps = self._o3.Irreps.spherical_harmonics(l_max)
            self._sh_irreps[l_max] = irreps
        return self._o3.spherical_harmonics(
            irreps, vectors, normalize=normalize, normalization="component"
        )

    def wigner_3j(self, l1, l2, l3):
        key = (l1, l2, l3)
        w = self._w3j.get(key)
        if w is None:
            w = self._o3.wigner_3j(l1, l2, l3).to(torch.float64)
            self._w3j[key] = w
        return w.to(torch.get_default_dtype())

    def make_weighted_bispectrum(self, **kwargs):
        return E3nnWeightedBispectrum(backend=self, **kwargs)


# --------------------------------------------------------------------------
# cuequivariance backend
# --------------------------------------------------------------------------
class CueBackend(EquivariantBackend):
    name = "cue"

    def __init__(self, device: torch.device | str | None = None) -> None:
        import cuequivariance as cue          # lazy: only when this backend is built
        import cuequivariance_torch as cuet

        self._cue = cue
        self._cuet = cuet
        self._device = None if device is None else torch.device(device)
        self._sh: dict[tuple, nn.Module] = {}
        self._w3j: dict[tuple[int, int, int], torch.Tensor] = {}

    def spherical_harmonics(self, vectors, l_max, *, normalize=True):
        # cue has no MPS kernel; force the "naive" kernel anywhere but CUDA (on CUDA
        # method=None lets cue pick the fast kernel).
        method = "naive" if vectors.device.type != "cuda" else None
        key = (l_max, bool(normalize), vectors.device.type)
        sh = self._sh.get(key)
        if sh is None:
            sh = self._cuet.SphericalHarmonics(
                ls=list(range(l_max + 1)), normalize=normalize, method=method
            )
            self._sh[key] = sh
        return sh(vectors)

    def wigner_3j(self, l1, l2, l3):
        import numpy as np

        key = (l1, l2, l3)
        w = self._w3j.get(key)
        if w is None:
            ir = lambda l: self._cue.O3(l, (-1) ** l)  # parity (-1)^l, matches SH
            cg = np.asarray(self._cue.clebsch_gordan(ir(l1), ir(l2), ir(l3)))
            # cg has shape (n_paths, d1, d2, d3); n_paths == 0 when (l1, l2, l3) do not
            # couple (parity/triangle), matching e3nn's all-zeros wigner_3j.
            if cg.shape[0] == 0:
                w = torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)
            else:
                t = torch.tensor(cg[0], dtype=torch.float64)
                w = t / t.norm()                        # unit Frobenius -> e3nn convention
            self._w3j[key] = w
        return w.to(torch.get_default_dtype())

    def make_weighted_bispectrum(
        self,
        *,
        implementation: str = "sym_contraction",
        method: str = "auto",
        device: torch.device | str | None = None,
        **kwargs,
    ):
        try:
            from .weighted_bispectrum import LearnableBispectrum, LearnableSymContraction
        except ImportError:
            from weighted_bispectrum import LearnableBispectrum, LearnableSymContraction

        dev = torch.device(device) if device is not None else self._device
        resolved = self._resolve_method(method, dev)
        # The two cue kernels take different degree kwargs; route accordingly.
        degrees = kwargs.pop("degrees", (3,))
        contraction_degree = kwargs.pop("contraction_degree", 3)
        if implementation == "descriptor":
            return LearnableBispectrum(method=resolved, degrees=degrees, **kwargs)
        if implementation == "sym_contraction":
            # SymmetricContraction takes `method=None` to auto-pick; map "" -> None.
            return LearnableSymContraction(
                method=(resolved or None), contraction_degree=contraction_degree, **kwargs
            )
        raise ValueError(
            f"unknown cue bispectrum implementation {implementation!r}; "
            "use 'sym_contraction' or 'descriptor'"
        )

    @staticmethod
    def _resolve_method(method: str, device: torch.device | None) -> str:
        if method != "auto":
            return method
        on_cuda = (device is not None and device.type == "cuda") or (
            device is None and torch.cuda.is_available()
        )
        return "" if on_cuda else "naive"


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def get_backend(name: str, *, device: torch.device | str | None = None) -> EquivariantBackend:
    name = (name or "e3nn").lower()
    if name == "e3nn":
        return E3nnBackend()
    if name == "cue":
        return CueBackend(device=device)
    raise ValueError(f"unknown equivariant backend {name!r}; use 'e3nn' or 'cue'")


# --------------------------------------------------------------------------
# e3nn trainable weighted bispectrum (explicit CG analogue of the cue kernel)
# --------------------------------------------------------------------------
def _bispectrum_paths(l_max: int, lam: int) -> list[tuple[int, int, int, int]]:
    """Valid degree-3 coupling paths ``(l1, l2, l3, k)`` for output ``lam``.

    ``l1 <= l2 <= l3`` (the three density factors are the same tensor, so only the
    unordered multiset matters); ``k`` is the intermediate ``l1 (x) l2 -> k``; parity
    is enforced by the two CG selection rules, giving ``l1+l2+l3 == lam (mod 2)``.
    """
    paths: list[tuple[int, int, int, int]] = []
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            for l3 in range(l2, l_max + 1):
                for k in range(abs(l1 - l2), l1 + l2 + 1):
                    if (l1 + l2 + k) % 2 != 0:
                        continue
                    if lam < abs(k - l3) or lam > k + l3:
                        continue
                    if (k + l3 + lam) % 2 != 0:
                        continue
                    paths.append((l1, l2, l3, k))
    return paths


class E3nnWeightedBispectrum(nn.Module):
    """Trainable, channel-wise nu=3 lambda-bispectrum built from explicit CG einsums.

    The e3nn-backend analogue of :class:`weighted_bispectrum.LearnableBispectrum`,
    with the same I/O contract: ``forward(A, species_idx) -> {lam: (N, 2lam+1, out_mul)}``
    (lam=0 as ``(N, 1, out_mul)``). Each input channel ``c`` (of ``C = n_species*n_max``)
    is contracted with itself three times along every coupling path; a learnable weight
    then mixes the ``(path, channel)`` axis down to ``out_mul`` output channels (per
    element when ``num_elements > 1``, shared when ``num_elements == 1``). Weights act
    only on the feature axis, never on the ``m`` components, so the output is equivariant.

    This is the explicit (unfused, more expensive) counterpart to the cuequivariance
    kernel -- correctness/portability, not speed; cue is the fast path on CUDA.
    """

    def __init__(
        self,
        *,
        backend: EquivariantBackend,
        n_species: int,
        n_max: int,
        l_max: int,
        selected_lambdas: Sequence[int] = (0, 2),
        out_mul: int | None = None,
        num_elements: int | None = None,
        weight_init_std: float = 1.0,
        # accepted for signature parity with the cue path; unused here (pure nu=3):
        degrees: Sequence[int] = (3,),
        contraction_degree: int = 3,
        method: str | None = None,
        dtype: torch.dtype | None = None,
        **_ignored,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_species) * int(n_max)
        self.l_max = int(l_max)
        self.lambdas = tuple(sorted({int(v) for v in selected_lambdas}))
        self.out_mul = int(out_mul) if out_mul is not None else self.n_channels
        self.num_elements = int(num_elements) if num_elements is not None else int(n_species)
        dtype = dtype or torch.get_default_dtype()

        # Per lam: combined CG buffers for every path and a weight over (path, channel).
        self._paths: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
        self.weights = nn.ParameterDict()
        for lam in self.lambdas:
            paths = _bispectrum_paths(self.l_max, lam)
            if not paths:
                raise ValueError(
                    f"no nu=3 coupling paths reach lambda={lam} with l_max={self.l_max}; "
                    "increase max_angular"
                )
            self._paths[lam] = tuple(paths)
            for idx, (l1, l2, l3, k) in enumerate(paths):
                cg = self._combined_cg(backend, l1, l2, l3, k, lam).to(dtype)
                self.register_buffer(f"_cg_{lam}_{idx}", cg, persistent=False)
            n_feat = len(paths) * self.n_channels
            w = torch.randn(self.num_elements, n_feat, self.out_mul, dtype=dtype)
            self.weights[str(lam)] = nn.Parameter(w * (weight_init_std / math.sqrt(n_feat)))

    @staticmethod
    def _combined_cg(backend, l1, l2, l3, k, lam) -> torch.Tensor:
        # C[m1,m2,m3,mu] = sum_mk w3j(l1,l2,k)[m1,m2,mk] * w3j(k,l3,lam)[mk,m3,mu]
        a = backend.wigner_3j(l1, l2, k).to(torch.float64)      # (2l1+1, 2l2+1, 2k+1)
        b = backend.wigner_3j(k, l3, lam).to(torch.float64)     # (2k+1, 2l3+1, 2lam+1)
        return torch.einsum("abk,kcd->abcd", a, b)              # (.., .., .., 2lam+1)

    @property
    def feature_dims(self) -> dict[int, int]:
        return {lam: self.out_mul for lam in self.lambdas}

    def forward(self, A: torch.Tensor, species_idx: torch.Tensor) -> dict[int, torch.Tensor]:
        if A.dim() == 4:
            A = A.reshape(A.shape[0], self.n_channels, A.shape[-1])  # (N, C, (l_max+1)^2)
        n = A.shape[0]
        A_by_l = _split_by_l(A, self.l_max)                    # each (N, C, 2l+1)

        if self.num_elements == 1:
            elem = torch.zeros(n, dtype=torch.long, device=A.device)
        else:
            elem = species_idx.to(torch.long)

        out: dict[int, torch.Tensor] = {}
        for lam in self.lambdas:
            feats = []
            for idx, (l1, l2, l3, _k) in enumerate(self._paths[lam]):
                cg = getattr(self, f"_cg_{lam}_{idx}")
                if cg.device != A.device or cg.dtype != A.dtype:
                    cg = cg.to(device=A.device, dtype=A.dtype)
                # per-channel triple product coupled to mu (=s): (N, 2lam+1, C)
                t = torch.einsum(
                    "pqrs,ncp,ncq,ncr->nsc",
                    cg, A_by_l[l1], A_by_l[l2], A_by_l[l3],
                )
                feats.append(t)                                # (N, 2lam+1, C)
            f = torch.cat(feats, dim=-1)                       # (N, 2lam+1, n_paths*C)
            w = self.weights[str(lam)][elem]                   # (N, n_feat, out_mul)
            out[lam] = torch.einsum("nmf,nfu->nmu", f, w)      # (N, 2lam+1, out_mul)
        return out
