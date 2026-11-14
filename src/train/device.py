"""The training device: an explicit name, or "auto" (CUDA, else MPS unless float64 is asked for, else CPU)."""
from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(name: str, dtype: str | None = None) -> torch.device:
    """``name`` as a device; "auto" picks CUDA, then MPS (never for float64: MPS has none), then the CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and dtype != "float64":
        return torch.device("mps")
    return torch.device("cpu")
