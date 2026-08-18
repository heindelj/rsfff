"""Many-body expansion of an intermolecular energy over fragment subsets.

For a cluster of ``N`` fragments the interaction energy decomposes exactly as

    E(1..N) = sum_k  E^(k),     E^(k) = sum_{|S|=k} dE(S)

with the ``k``-body term of a subset obtained by Moebius inversion over its own subsets:

    dE(S) = sum_{T subset of S} (-1)^{|S|-|T|} E(T)

This is the diagnostic that says whether a *pairwise-summed* model is nevertheless
producing many-body physics. :class:`rsfff.ff.dispersion.TTDispersion` sums over pairs,
so with per-species C6 coefficients it is strictly additive and every ``E^(k>=3)`` is
identically zero. Once the C6 coefficients are emitted from environment-aware descriptors,
removing a neighboring molecule changes the coefficients of the ones that remain, and the
higher-order terms become nonzero -- that non-additivity *is* the learned many-body
dispersion, and this module measures it.

The energy of a subset is evaluated with only that subset present, so the descriptors see
exactly the sub-cluster: this is the standard counterpoise-free MBE of a supermolecular
calculation, not a partition of the full-cluster energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from ..train.data import Batch


@dataclass
class MBEResult:
    """Per-cluster many-body decomposition, all energies in Hartree.

    ``by_order[k]`` is ``(n_clusters,)``: the total ``k``-body contribution, i.e. the sum
    of ``dE(S)`` over all subsets of size ``k``. Index 0 and 1 are absent -- an isolated
    fragment has no inter-fragment pairs, so ``E^(1) = 0`` identically.

    ``total`` is the model on the full cluster and equals ``sum_k by_order[k]`` to solver
    precision; :func:`mbe_decompose` checks this rather than assuming it.
    """

    total: torch.Tensor                     # (n_clusters,)
    by_order: dict[int, torch.Tensor]       # k -> (n_clusters,)
    n_fragments: torch.Tensor               # (n_clusters,) long
    components: dict[str, "MBEResult"] | None = None   # e.g. {"ff": ..., "corr": ...}

    @property
    def two_body(self) -> torch.Tensor:
        return self.by_order[2]

    @property
    def many_body(self) -> torch.Tensor:
        """Everything beyond pairwise: ``total - E^(2)``."""
        return self.total - self.by_order[2]


def subset_batch(
    positions: torch.Tensor,     # (N, 3)
    atomic_numbers: torch.Tensor,   # (N,)
    fragment_idx: torch.Tensor,     # (N,) values in [0, n_frag)
    subsets: list[tuple[int, ...]],
    *,
    fragment_charge: torch.Tensor | None = None,   # (n_frag,)
    fragment_two_s: torch.Tensor | None = None,    # (n_frag,)
) -> Batch:
    """One :class:`Batch` holding each fragment subset as a separate system.

    Fragment ids are renumbered contiguously within each subset and offset to be
    batch-global, matching what ``MoleculeDataset.flat_batch`` produces -- so the atoms of
    each system stay grouped by fragment and the pair list's sortedness check passes.

    ``fragment_charge`` and ``fragment_two_s`` are gathered per subset rather than dropped.
    On neutral closed-shell fragments they change nothing -- ``FragmentStateEmbedding`` is
    identically zero there, which is why leaving them out went unnoticed on water -- but the
    moment a cluster carries a charged fragment, dropping them silently decomposes a
    *different system* from the one asked about: every fragment would read as neutral.
    """
    rows, batch_ids, frag_ids, frag_src, f2b = [], [], [], [], []
    n_frag_total = 0
    for system, subset in enumerate(subsets):
        for local, frag in enumerate(subset):
            sel = (fragment_idx == frag).nonzero(as_tuple=True)[0]
            rows.append(sel)
            batch_ids.append(torch.full((sel.numel(),), system, dtype=torch.long))
            frag_ids.append(torch.full((sel.numel(),), n_frag_total + local, dtype=torch.long))
            frag_src.append(frag)
            f2b.append(system)
        n_frag_total += len(subset)

    rows = torch.cat(rows)
    src = torch.tensor(frag_src, dtype=torch.long, device=positions.device)
    return Batch(
        positions=positions[rows],
        atomic_numbers=atomic_numbers[rows],
        batch_idx=torch.cat(batch_ids).to(positions.device),
        n_systems=len(subsets),
        energy=torch.zeros(len(subsets), dtype=positions.dtype, device=positions.device),
        fragment_idx=torch.cat(frag_ids).to(positions.device),
        n_fragments=n_frag_total,
        fragment_charge=None if fragment_charge is None else fragment_charge[src],
        fragment_two_s=None if fragment_two_s is None else fragment_two_s[src],
        fragment_to_batch=torch.tensor(f2b, dtype=torch.long, device=positions.device),
    )


def _moebius(subset_energy: dict[tuple[int, ...], float], subset: tuple[int, ...]) -> float:
    """``dE(S) = sum_{T subset S} (-1)^{|S|-|T|} E(T)``, with ``E(T) = 0`` for ``|T| < 2``."""
    k = len(subset)
    total = 0.0
    for size in range(2, k + 1):
        sign = -1.0 if (k - size) % 2 else 1.0
        for sub in combinations(subset, size):
            total += sign * subset_energy[sub]
    return total


@torch.no_grad()
def mbe_decompose(
    model,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    fragment_idx: torch.Tensor,
    *,
    max_order: int | None = None,
    split_components: bool = True,
    fragment_charge: torch.Tensor | None = None,
    fragment_two_s: torch.Tensor | None = None,
) -> MBEResult:
    """Many-body decomposition of ``model``'s energy for a single cluster.

    ``model`` is anything callable on a :class:`Batch` returning an object with an
    ``energy`` field of shape ``(n_systems,)`` -- i.e.
    :class:`rsfff.ff.dispersion.DispersionModel`.

    Every subset of size >= 2 is evaluated (2^N - N - 1 of them; 26 for a pentamer), all in
    a single batched forward. ``max_order`` truncates the expansion, which changes what is
    reported but not what is computed -- the subsets are needed either way.

    With ``split_components`` the force-field and correction parts are decomposed
    separately, which is what shows *where* the non-additivity comes from: environment
    dependence of the C6 coefficients, or the pair correction head.
    """
    n_frag = int(fragment_idx.max()) + 1
    order = n_frag if max_order is None else min(max_order, n_frag)
    subsets = [
        s for size in range(2, n_frag + 1) for s in combinations(range(n_frag), size)
    ]
    if not subsets:
        raise ValueError(f"need at least 2 fragments for an MBE, got {n_frag}")

    batch = subset_batch(
        positions, atomic_numbers, fragment_idx, subsets,
        fragment_charge=fragment_charge, fragment_two_s=fragment_two_s,
    )
    out = model(batch)

    fields = {"total": out.energy}
    if split_components:
        fields["ff"] = out.energy_ff
        fields["corr"] = out.energy_corr

    results: dict[str, MBEResult] = {}
    for name, energies in fields.items():
        lookup = {s: float(energies[i]) for i, s in enumerate(subsets)}
        by_order = {}
        for k in range(2, order + 1):
            terms = sum(
                _moebius(lookup, s) for s in combinations(range(n_frag), k)
            )
            by_order[k] = torch.tensor([terms], dtype=positions.dtype)
        results[name] = MBEResult(
            total=torch.tensor([lookup[tuple(range(n_frag))]], dtype=positions.dtype),
            by_order=by_order,
            n_fragments=torch.tensor([n_frag]),
        )

    root = results.pop("total")
    if order == n_frag:
        # This used to compare `sum_k E^(k)` against `E(full)`. That check could never fail:
        # both sides are built from the same `lookup` dict, and the Moebius inversion summing
        # back to its own input is an algebraic identity, not a property of the model. It read
        # as a guard against exactly the bug it could not see.
        #
        # What is worth checking is the assumption the decomposition actually rests on: that
        # putting every subset in one batch does not let them see each other. So re-evaluate
        # the full cluster *alone* and compare it against the copy that rode in the big batch.
        # A stale or cross-system pair list, a solver that pools over the whole batch, or a
        # dropped fragment field all show up here, and nothing else in this module would catch
        # them.
        full = tuple(range(n_frag))
        alone = model(subset_batch(
            positions, atomic_numbers, fragment_idx, [full],
            fragment_charge=fragment_charge, fragment_two_s=fragment_two_s,
        ))
        solo = float(alone.energy[0])
        if abs(solo - float(root.total)) > 1e-9 * max(1.0, abs(solo)):
            raise RuntimeError(
                f"the full cluster evaluates to {float(root.total):.9e} inside the subset "
                f"batch and {solo:.9e} on its own, so the subsets are not independent and "
                f"every term of this expansion is suspect"
            )
    root.components = results or None
    return root


def mbe_dataset(
    model,
    dataset,
    indices,
    *,
    max_order: int | None = None,
    split_components: bool = True,
    progress_every: int = 0,
) -> MBEResult:
    """Run :func:`mbe_decompose` over many frames and stack the results.

    Frames may have different fragment counts; ``by_order`` is padded with zeros for
    orders a given cluster cannot reach (a trimer contributes 0 to the 4-body column).
    """
    per_frame: list[MBEResult] = []
    for n, idx in enumerate(indices):
        frame = dataset.flat_batch([idx])
        per_frame.append(
            mbe_decompose(
                model, frame.positions, frame.atomic_numbers, frame.fragment_idx,
                max_order=max_order, split_components=split_components,
                fragment_charge=frame.fragment_charge,
                fragment_two_s=frame.fragment_two_s,
            )
        )
        if progress_every and (n + 1) % progress_every == 0:
            print(f"  MBE {n + 1}/{len(indices)}", flush=True)

    orders = sorted({k for r in per_frame for k in r.by_order})

    def stack(rs: list[MBEResult]) -> MBEResult:
        return MBEResult(
            total=torch.cat([r.total for r in rs]),
            by_order={
                k: torch.cat([
                    r.by_order.get(k, torch.zeros(1, dtype=r.total.dtype)) for r in rs
                ])
                for k in orders
            },
            n_fragments=torch.cat([r.n_fragments for r in rs]),
        )

    root = stack(per_frame)
    if per_frame[0].components is not None:
        root.components = {
            name: stack([r.components[name] for r in per_frame])
            for name in per_frame[0].components
        }
    return root
