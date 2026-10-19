"""Finiteness guards for the batched dense factorizations of the charge/induction solves.

A batched LU on the GPU (``lu_factor`` / ``solve`` / ``inv``) handed a non-finite matrix
does not reliably return NaN: MAGMA's batched getrf can fault with an illegal memory access,
which poisons the CUDA context for the whole process (every later kernel fails too), so a
single blown-up MD replica takes down every member and every other replica with it.

:func:`finite_or_identity` swaps each non-finite matrix of the batch for the identity before
the factorization and returns the mask, with no host sync; :func:`poison` puts NaN back into
those entries of the result, so the failure stays visible downstream (a NaN energy for that
system, which the committee's per-system retry and the samplers already handle) instead of
turning into a finite but meaningless answer.
"""

from __future__ import annotations

import torch


def finite_or_identity(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mat with every non-finite (..., n, n) entry replaced by I, ok mask (...,))``."""
    ok = torch.isfinite(mat).flatten(-2).all(-1)
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)
    return torch.where(ok[..., None, None], mat, eye), ok


def poison(x: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """NaN in the batch entries of ``x`` (leading dims = ``ok``'s) where ``ok`` is False."""
    mask = ok.reshape(ok.shape + (1,) * (x.dim() - ok.dim()))
    return torch.where(mask, x, torch.full_like(x, float("nan")))


__all__ = ["finite_or_identity", "poison"]
