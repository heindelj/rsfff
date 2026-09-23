"""Drive a trained film model over whole structures: energies, gradients, optimization, Hessians.

Everything here treats the model as a plain potential ``E(R)`` in **Hartree and Angstrom**.
That is the only thing the geometry optimizer and the Hessian need, and it is deliberately
kept separate from :mod:`rsfff.md.calculator`, which wraps the *mediated* expert model and
carries the decomposition-enumeration machinery a film evaluation does not use: the film
model takes one fixed fragmentation and evaluates once.

Two details are worth stating because they set the accuracy of everything downstream:

fragmentation is fixed at the input geometry
    Water clusters near a minimum do not transfer protons, so the O + nearest-two-H grouping
    computed once is valid along the whole optimization path. It is *not* recomputed per step
    -- a fragmentation that flips mid-optimization would put a discontinuity in the energy.
    :func:`water_fragment_index` returns a sorted index because the neighbor graph builder
    assumes fragment ids are non-decreasing in atom order.

the Hessian is a finite difference of *analytic* gradients
    The coupled induction solve is a CG fixed point; its first derivative is exact but the
    second is not something to trust through a double backward. Central differences of the
    analytic gradient cost 6N evaluations, and :func:`hessian` batches the displaced
    geometries into one ragged graph so those evaluations run a chunk at a time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from scipy.optimize import minimize

from ..ff.film.model import maybe_compile
from ..train.build_pairing import build_model
from ..train.data import Batch, load_reference_energies

__all__ = [
    "FilmPotential",
    "OptimizeResult",
    "hessian",
    "hydrogen_bonds",
    "load_film_model",
    "make_batch",
    "oh_bond_list",
    "optimize",
    "water_fragment_index",
]


def load_film_model(path, *, device: str = "cpu"):
    """Rebuild a trained film model from a checkpoint and load it strictly.

    Sets the global default dtype from the checkpoint's config, because every tensor built
    downstream (positions, displacements) has to match the model's parameters.

    **The checkpoint is self-contained.** The isolated-atom reference energies are a
    persistent buffer on the model (``FilmModel.register_buffer("reference_energies", ...)``),
    so they come back with the state dict and the JSON that ``config.data.reference_energies``
    names is never opened. This used to read that file, which made loading a model depend on
    the current directory being the repository root it was trained in -- a checkpoint copied
    to a cluster, or used from a run directory on scratch, would raise ``FileNotFoundError``
    on a path like ``data/atomic_references_wb97mv_tzvpd.json``. The buffer was always in the
    file; reading the JSON as well was redundant, and the redundancy was the bug.

    The JSON is still the fallback for a checkpoint written before the buffer existed, and
    :func:`rsfff.train.data.resolve_data_path` finds it without assuming the current
    directory.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    neighbor_types = tuple(int(z) for z in ckpt["neighbor_types"])
    state = ckpt["model_state"]
    if "reference_energies" in state:
        reference = state["reference_energies"].to(torch.get_default_dtype())
    elif "reference.energies" in state:
        reference = state["reference.energies"].to(torch.get_default_dtype())
    else:
        reference = load_reference_energies(
            config.data.reference_energies, neighbor_types
        ).to(torch.get_default_dtype())
    # the pairing model's charged reference tables are buffers: load_state_dict restores
    # them, so the builder only needs a placeholder of the right shape
    model = build_model(config.features, config.film, neighbor_types, reference)
    maybe_compile(model)                      # RSFFF_COMPILE: inference use, all parts allowed
    model.load_state_dict(state)
    model.eval().to(device)
    return model, config


def water_fragment_index(positions, atomic_numbers) -> np.ndarray:
    """Group each H with its nearest O; return a per-atom fragment id, sorted in atom order.

    "Sorted" means non-decreasing over atoms, which the neighbor graph requires: fragments are
    numbered by the atom index of their oxygen, so the usual ``O H H O H H ...`` file order
    comes out as ``0 0 0 1 1 1 ...`` unchanged. An O without exactly two H, or an H with no O
    in range, raises -- this routine is for water, and silently mis-fragmenting is worse than
    stopping.
    """
    positions = np.asarray(positions, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    o_atoms = np.flatnonzero(z == 8)
    h_atoms = np.flatnonzero(z == 1)
    if o_atoms.size * 3 != z.size or o_atoms.size + h_atoms.size != z.size:
        raise ValueError(f"expected pure water (O + 2H per fragment), got {np.bincount(z)}")

    d = np.linalg.norm(positions[h_atoms, None, :] - positions[None, o_atoms, :], axis=-1)
    owner = d.argmin(axis=1)
    counts = np.bincount(owner, minlength=o_atoms.size)
    if not np.all(counts == 2):
        raise ValueError(
            "nearest-oxygen assignment did not give every O exactly two H "
            f"(counts {counts.tolist()}); the geometry is not a set of intact waters"
        )

    frag = np.empty(z.size, dtype=np.int64)
    frag[o_atoms] = np.arange(o_atoms.size)
    frag[h_atoms] = owner
    order = np.argsort(frag, kind="stable")
    if not np.array_equal(order, np.arange(z.size)):
        raise ValueError(
            "atoms are not in fragment order; reorder the frame so each water's atoms are "
            "contiguous (the neighbor graph assumes non-decreasing fragment_idx)"
        )
    return frag


def oh_bond_list(fragment_idx, atomic_numbers) -> np.ndarray:
    """The intramolecular O-H pairs as an ``(n_bonds, 2)`` array of ``(o_index, h_index)``."""
    frag = np.asarray(fragment_idx)
    z = np.asarray(atomic_numbers, dtype=int)
    bonds = []
    for f in range(int(frag.max()) + 1):
        atoms = np.flatnonzero(frag == f)
        (o,) = atoms[z[atoms] == 8]
        bonds.extend((int(o), int(h)) for h in atoms[z[atoms] == 1])
    return np.asarray(bonds, dtype=np.int64)


def hydrogen_bonds(
    positions,
    bonds,
    fragment_idx,
    atomic_numbers,
    *,
    max_distance: float = 2.5,
    min_angle: float = 130.0,
) -> np.ndarray:
    """Classify every O-H bond as a hydrogen-bond donor or a free O-H.

    The descriptor is the standard geometric one: an O-H donates if some oxygen on *another*
    fragment sits within ``max_distance`` of the H and the O-H...O angle exceeds ``min_angle``.
    Both parts are needed -- distance alone counts an oxygen that happens to sit beside a free
    O-H, and the angle is what distinguishes donating from merely being nearby. Among the
    acceptors that pass, the closest wins.

    Returns a structured array with one row per bond in ``bonds``: ``is_donor`` (bool),
    ``acceptor`` (oxygen index, -1 if free), ``distance`` (H...O, Angstrom, to the nearest
    intermolecular oxygen whether or not it qualifies) and ``angle`` (O-H...O, degrees, to
    that same oxygen). Reporting the nearest-oxygen geometry even for a free O-H is what makes
    the column usable as a continuous descriptor rather than only as a label.
    """
    positions = np.asarray(positions, dtype=float)
    bonds = np.asarray(bonds, dtype=int)
    frag = np.asarray(fragment_idx)
    z = np.asarray(atomic_numbers, dtype=int)

    result = np.zeros(len(bonds), dtype=[
        ("is_donor", bool), ("acceptor", np.int64),
        ("distance", float), ("angle", float),
    ])
    for row, (o_index, h_index) in enumerate(bonds):
        candidates = np.flatnonzero((z == 8) & (frag != frag[h_index]))
        if candidates.size == 0:
            result[row] = (False, -1, np.nan, np.nan)
            continue
        vectors = positions[candidates] - positions[h_index]
        distances = np.linalg.norm(vectors, axis=1)
        to_donor = positions[o_index] - positions[h_index]
        cos = (vectors @ to_donor) / (distances * np.linalg.norm(to_donor))
        angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))    # O-H...O, 180 = linear

        qualifies = (distances <= max_distance) & (angles >= min_angle)
        nearest = int(distances.argmin())
        if qualifies.any():
            best = int(np.flatnonzero(qualifies)[distances[qualifies].argmin()])
            result[row] = (True, candidates[best], distances[best], angles[best])
        else:
            result[row] = (False, -1, distances[nearest], angles[nearest])
    return result


def make_batch(positions, atomic_numbers, fragment_idx, *, n_frames: int = 1) -> Batch:
    """One ragged graph holding ``n_frames`` copies of the same topology.

    ``positions`` is ``(n_frames * n_atoms, 3)``; the species and fragmentation are shared and
    tiled. Fragment ids are re-offset per frame so the frames stay disjoint, which is what
    makes a batched Hessian column sweep a single forward pass.
    """
    dtype = torch.get_default_dtype()
    z = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.long)
    frag = torch.as_tensor(np.asarray(fragment_idx), dtype=torch.long)
    n_atoms = z.numel()
    n_frag = int(frag.max()) + 1
    pos = torch.as_tensor(positions, dtype=dtype).reshape(n_frames * n_atoms, 3)

    frame = torch.arange(n_frames, dtype=torch.long)
    return Batch(
        positions=pos,
        atomic_numbers=z.repeat(n_frames),
        batch_idx=frame.repeat_interleave(n_atoms),
        n_systems=n_frames,
        energy=torch.zeros(n_frames, dtype=dtype),
        fragment_idx=frag.repeat(n_frames) + n_frag * frame.repeat_interleave(n_atoms),
        fragment_charge=torch.zeros(n_frames * n_frag, dtype=dtype),
        fragment_two_s=torch.zeros(n_frames * n_frag, dtype=dtype),
        fragment_to_batch=frame.repeat_interleave(n_frag),
        n_fragments=n_frames * n_frag,
    )


class FilmPotential:
    """``E(R)`` and ``dE/dR`` for one fixed topology, in Hartree and Angstrom.

    Holds the species and the fragmentation so the caller passes coordinates alone. Every
    method accepts and returns numpy arrays: the optimizer and the Hessian have no reason to
    see torch.
    """

    def __init__(self, model, atomic_numbers, fragment_idx, *, with_induction: bool = True):
        self.model = model
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self.fragment_idx = np.asarray(fragment_idx, dtype=np.int64)
        self.with_induction = bool(with_induction)
        self.n_atoms = self.atomic_numbers.size
        self.n_calls = 0

    def _forward(self, positions, n_frames: int):
        batch = make_batch(
            positions, self.atomic_numbers, self.fragment_idx, n_frames=n_frames
        )
        self.n_calls += n_frames
        return self.model(batch, with_induction=self.with_induction)

    def energy(self, positions) -> np.ndarray:
        """Total energies, ``(n_frames,)``, for ``(n_frames, n_atoms, 3)`` coordinates."""
        positions = np.asarray(positions, dtype=float).reshape(-1, self.n_atoms, 3)
        with torch.no_grad():
            out = self._forward(positions, positions.shape[0])
        return out.energy.detach().numpy().copy()

    def energy_and_gradient(self, positions):
        """``(energies, gradients)`` with shapes ``(n_frames,)`` and ``(n_frames, n_atoms, 3)``.

        The frames in a batch are independent, so one backward pass on the summed energy
        gives every frame's gradient in its own slice.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, self.n_atoms, 3)
        n_frames = positions.shape[0]
        pos = torch.as_tensor(
            positions.reshape(-1, 3), dtype=torch.get_default_dtype()
        ).requires_grad_(True)
        out = self._forward(pos, n_frames)
        (grad,) = torch.autograd.grad(out.energy.sum(), pos)
        return (
            out.energy.detach().numpy().copy(),
            grad.numpy().reshape(n_frames, self.n_atoms, 3).copy(),
        )

    def interaction(self, positions) -> dict[str, float]:
        """The per-channel interaction breakdown of a single frame, Hartree."""
        with torch.no_grad():
            out = self._forward(np.asarray(positions, dtype=float), 1)
        return {name: float(value[0]) for name, value in out.interaction.items()}


@dataclass
class OptimizeResult:
    positions: np.ndarray       # (n_atoms, 3) Angstrom
    energy: float               # Hartree
    max_force: float            # Hartree/Angstrom, max |component| of -dE/dR
    rms_force: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    message: str


def optimize(
    potential: FilmPotential,
    positions,
    *,
    gtol: float = 1e-7,
    max_iter: int = 2000,
    max_restarts: int = 4,
) -> OptimizeResult:
    """L-BFGS-B in Cartesian coordinates to a max-force criterion, with restarts.

    ``gtol`` is scipy's ``pgtol``: the convergence test is on the largest gradient component,
    which is the same quantity a force criterion uses, so the reported ``max_force`` and the
    stopping rule agree. ``ftol`` is set to machine level on purpose -- an energy-change test
    stops a shallow cluster mode long before its forces are small, and a loose minimum shows
    up as spurious imaginary frequencies in the Hessian that follows.

    On a large cluster the line search usually stalls a factor of a few above ``gtol`` while
    the limited-memory Hessian is stale from the early, large steps. Restarting from the
    current point discards that history and the next pass walks straight in, so a restart buys
    more than a larger ``maxcor`` does. Restarting stops as soon as a pass fails to improve
    the maximum force, which is the signal that round-off, not the optimizer, is the limit.
    """
    x = np.asarray(positions, dtype=float).reshape(-1)

    def fun(x):
        energy, gradient = potential.energy_and_gradient(x)
        return float(energy[0]), gradient.reshape(-1)

    n_iterations = n_evaluations = 0
    best = None
    message = ""
    for _ in range(max_restarts + 1):
        res = minimize(
            fun, x, jac=True, method="L-BFGS-B",
            options={"maxiter": max_iter, "maxfun": 10 * max_iter, "maxcor": 30,
                     "ftol": 1e-16, "gtol": gtol},
        )
        n_iterations += int(res.nit)
        n_evaluations += int(res.nfev)
        message = str(res.message)
        energy, gradient = potential.energy_and_gradient(res.x)
        max_force = float(np.abs(gradient).max())
        if best is not None and max_force >= best[1]:
            break
        x = res.x
        best = (res.x, max_force, float(energy[0]),
                float(np.sqrt((gradient**2).mean())))
        if max_force <= gtol:
            break

    x, max_force, energy, rms_force = best
    return OptimizeResult(
        positions=x.reshape(-1, 3),
        energy=energy,
        max_force=max_force,
        rms_force=rms_force,
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
        converged=bool(max_force <= gtol),
        message=message,
    )


def hessian(
    potential: FilmPotential,
    positions,
    *,
    delta: float = 1e-3,
    chunk: int = 16,
) -> np.ndarray:
    """Central-difference Hessian, ``(3N, 3N)`` in Hartree/Angstrom^2.

    ``delta`` (Angstrom) trades truncation error against the gradient's own noise; 1e-3 keeps
    both near 1e-6 Ha/A^2 in float64. Displaced geometries are evaluated ``chunk`` at a time
    in one ragged graph. The result is symmetrized, which halves the residual asymmetry the
    finite difference leaves behind.
    """
    x0 = np.asarray(positions, dtype=float).reshape(-1)
    n = x0.size
    columns = np.empty((n, n))

    displaced = np.repeat(x0[None, :], 2 * n, axis=0)
    rows = np.arange(n)
    displaced[2 * rows, rows] += delta
    displaced[2 * rows + 1, rows] -= delta

    gradients = np.empty((2 * n, n))
    for start in range(0, 2 * n, chunk):
        block = displaced[start:start + chunk]
        _, grad = potential.energy_and_gradient(block)
        gradients[start:start + block.shape[0]] = grad.reshape(block.shape[0], n)
    columns[:] = (gradients[0::2] - gradients[1::2]) / (2.0 * delta)
    return 0.5 * (columns + columns.T)
