"""The one place a neighbor list is built.

``torch_cluster.radius_graph`` takes ``max_num_neighbors=32`` by default and, on hitting
it, **silently drops** the excess edges: no error, no warning, just a descriptor (or a pair
energy) computed from a truncated environment. 32 is far below what this codebase needs --
a 5 A feature cutoff in bulk-like water already sits near 40 neighbors, and the force-field
cutoffs reach much further -- so every call goes through :func:`build_radius_graph`, which

1. always passes ``max_num_neighbors`` explicitly (default
   :data:`DEFAULT_MAX_NUM_NEIGHBORS`), so the library default can never leak back in, and
2. checks afterwards whether any atom came out *at* the cap, which is the only observable
   signature of truncation.

The check costs one ``bincount`` plus one host sync per build. That sync is already paid:
``radius_graph`` itself calls ``.item()`` internally, so this adds no new synchronization
point to the step.

Reaching the cap is reported three ways, in increasing order of insistence:

- :data:`CAP_EVENTS` counts hits per context, for a training loop to log or assert on
  (:func:`reset_cap_events` clears it);
- a ``UserWarning``, emitted once per ``(context, cap)`` so a hot loop cannot spam it;
- :class:`NeighborCapExceeded`, raised instead of warning when strict mode is on
  (``RSFFF_STRICT_NEIGHBORS=1``, or :func:`set_strict`).

The checking itself can be turned off with ``RSFFF_NEIGHBOR_CAP_CHECK=0`` or
:func:`set_cap_check`, which is only worth doing in a profiling run.
"""

from __future__ import annotations

import os
import warnings

import torch
from torch_cluster import radius_graph

__all__ = [
    "DEFAULT_MAX_NUM_NEIGHBORS",
    "config_max_num_neighbors",
    "NeighborCapExceeded",
    "CAP_EVENTS",
    "build_radius_graph",
    "reset_cap_events",
    "set_cap_check",
    "set_strict",
]

#: Cap handed to every ``radius_graph`` call unless a caller overrides it. Chosen well
#: above the ~40-60 neighbors a condensed-phase feature cutoff produces, so hitting it
#: means something is wrong (a too-large cutoff, a collapsed geometry) rather than merely
#: dense.
DEFAULT_MAX_NUM_NEIGHBORS = 256


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "false", "no", "off")


_STRICT = _env_flag("RSFFF_STRICT_NEIGHBORS", False)
_CHECK = _env_flag("RSFFF_NEIGHBOR_CAP_CHECK", True)

#: ``context -> number of builds that hit the cap`` since the last
#: :func:`reset_cap_events`. Counts builds, not atoms; the warning carries the atom count.
CAP_EVENTS: dict[str, int] = {}

_WARNED: set[tuple[str, int]] = set()


class NeighborCapExceeded(RuntimeError):
    """A neighbor list was truncated by ``max_num_neighbors``."""


def set_strict(strict: bool) -> None:
    """Raise :class:`NeighborCapExceeded` (instead of warning) when the cap is reached."""
    global _STRICT
    _STRICT = bool(strict)


def set_cap_check(enabled: bool) -> None:
    """Enable/disable the post-build cap check (it costs one bincount + one host sync)."""
    global _CHECK
    _CHECK = bool(enabled)


def reset_cap_events() -> None:
    """Clear :data:`CAP_EVENTS` and the once-per-context warning memory."""
    CAP_EVENTS.clear()
    _WARNED.clear()


def build_radius_graph(
    positions: torch.Tensor,               # (N, 3)
    r: float,
    batch: torch.Tensor,                   # (N,) long, grouping (frame or fragment)
    *,
    context: str,
    loop: bool = False,
    max_num_neighbors: int = DEFAULT_MAX_NUM_NEIGHBORS,
) -> torch.Tensor:
    """``torch_cluster.radius_graph`` with an explicit cap and truncation detection.

    ``context`` names the call site in the warning and in :data:`CAP_EVENTS`; it is
    required so a report always says *which* list was truncated.

    Handles the MPS fallback in one place: torch_cluster ships CPU and CUDA kernels but no
    MPS one, so on MPS the build runs on CPU and the edges come back to the input device.
    The returned edges are directed (both ``(i, j)`` and ``(j, i)``) exactly as
    ``radius_graph`` returns them; de-duplication to ``i < j`` is the caller's business.
    """
    cap = int(max_num_neighbors)
    if positions.device.type == "mps":
        edge = radius_graph(
            positions.cpu(), r=r, batch=batch.cpu(), loop=loop, max_num_neighbors=cap,
        ).to(positions.device)
    else:
        edge = radius_graph(
            positions, r=r, batch=batch, loop=loop, max_num_neighbors=cap,
        )

    if _CHECK:
        _check_cap(edge, positions.shape[0], r, cap, context)
    return edge


def _check_cap(edge: torch.Tensor, n_atoms: int, r: float, cap: int, context: str) -> None:
    """Flag atoms sitting *at* the cap -- the only visible trace of a truncated list.

    The cap applies per *query* atom, and with torch_cluster's default
    ``flow="source_to_target"`` the query is row 1 of the returned edges (row 0 is the
    neighbor, whose count is unbounded). Counting the wrong row would both miss truncation
    and invent it, so the row choice here is load-bearing -- it is asserted in
    ``tests/test_neighbors.py``.

    An atom with exactly ``cap`` neighbors may be a genuine boundary case rather than a
    truncated one, so this over-reports by at most the handful of atoms that happen to land
    on the boundary. That is the right side to err on: the alternative (rebuilding without
    the cap to compare) costs more than the list itself.
    """
    if edge.numel() == 0:
        return
    degree = torch.bincount(edge[1], minlength=n_atoms)
    n_at_cap = int((degree >= cap).sum())
    if n_at_cap == 0:
        return

    CAP_EVENTS[context] = CAP_EVENTS.get(context, 0) + 1
    message = (
        f"{context}: neighbor list reached max_num_neighbors={cap} for {n_at_cap} of "
        f"{n_atoms} atoms at cutoff {r} A. torch_cluster truncates silently, so those "
        f"atoms are missing neighbors and their features/pair energies are wrong. Raise "
        f"max_num_neighbors (or shrink the cutoff)."
    )
    if _STRICT:
        raise NeighborCapExceeded(message)
    key = (context, cap)
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=3)


def config_max_num_neighbors(features_cfg) -> int:
    """``max_num_neighbors`` from a features config, whether dataclass or plain dict.

    Defaulted rather than required so a config built by older code (or a hand-rolled
    namespace in a script) keeps working -- the point is only that the *library* default of
    32 never reaches a call.
    """
    if isinstance(features_cfg, dict):
        value = features_cfg.get("max_num_neighbors", DEFAULT_MAX_NUM_NEIGHBORS)
    else:
        value = getattr(features_cfg, "max_num_neighbors", DEFAULT_MAX_NUM_NEIGHBORS)
    return int(value if value is not None else DEFAULT_MAX_NUM_NEIGHBORS)
