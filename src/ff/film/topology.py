"""Covalent-graph enumeration for the bonded terms: torsions, coupling index lists, separations.

Everything here is a function of the bond list alone -- never of a distance -- and is
vectorized with the padded neighbor table that :func:`rsfff.ff.film.bonded._angles_from_bonds`
and :meth:`BondedTopology.exclusions` already use, so a batch of a few thousand atoms builds
its topology without a Python loop over atoms.

Index conventions (all global atom indices of one flat batch):

``bond_index``   (2, Nb)  ``i < j``, unique.
``angle_index``  (3, Na)  ``[i, apex, k]``.
``torsion_index``(4, Nt)  ``[a, b, c, d]`` with ``b < c`` -- each torsion once, its central bond
                          in storage order.
``angle_pairs``  (4, Naa) ``[apex, s, x, y]`` with ``x < y``: the two angles ``(s, apex, x)``
                          and ``(s, apex, y)`` share the apex *and* the leg ``apex-s`` -- the
                          pairs the angle-angle coupling acts on.
"""

from __future__ import annotations

import torch

__all__ = [
    "angle_pairs_from_angles",
    "atom_degrees",
    "canonical_bonds",
    "graph_separation",
    "lookup_angles",
    "lookup_bonds",
    "neighbor_table",
    "torsions_from_bonds",
]


def canonical_bonds(bonds: torch.Tensor) -> torch.Tensor:
    """``(2, Nb)`` with ``i < j``, duplicates and self-pairs removed, sorted by ``(i, j)``."""
    if bonds.numel() == 0:
        return bonds.reshape(2, 0).long()
    lo = torch.minimum(bonds[0], bonds[1]).long()
    hi = torch.maximum(bonds[0], bonds[1]).long()
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    n = int(hi.max()) + 1 if hi.numel() else 1
    keys = torch.unique(lo * n + hi)
    return torch.stack((keys // n, keys % n))


def atom_degrees(bond_index: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``(N,)`` number of covalent neighbors of each atom."""
    return torch.bincount(bond_index.reshape(-1), minlength=int(n_atoms))


def neighbor_table(bond_index: torch.Tensor, n_atoms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(nbr (N, deg_max) padded with -1, degree (N,))``."""
    device = bond_index.device
    atoms = torch.cat((bond_index[0], bond_index[1]))
    others = torch.cat((bond_index[1], bond_index[0]))
    counts = torch.bincount(atoms, minlength=int(n_atoms))
    if atoms.numel() == 0:
        return torch.full((int(n_atoms), 0), -1, dtype=torch.long, device=device), counts
    order = torch.argsort(atoms, stable=True)
    atoms_s, others_s = atoms[order], others[order]
    deg_max = int(counts.max())
    offsets = torch.cumsum(counts, 0) - counts
    slot = torch.arange(atoms_s.numel(), device=device) - offsets[atoms_s]
    nbr = torch.full((int(n_atoms), deg_max), -1, dtype=torch.long, device=device)
    nbr[atoms_s, slot] = others_s
    return nbr, counts


def _pair_keys(i: torch.Tensor, j: torch.Tensor, n: int) -> torch.Tensor:
    return torch.minimum(i, j) * n + torch.maximum(i, j)


def lookup_bonds(
    bond_index: torch.Tensor, n_atoms: int, i: torch.Tensor, j: torch.Tensor
) -> torch.Tensor:
    """Column of ``bond_index`` holding the (unordered) pair ``(i, j)``. The pair must exist."""
    n = int(n_atoms)
    keys = _pair_keys(bond_index[0], bond_index[1], n)
    order = torch.argsort(keys)
    pos = torch.searchsorted(keys[order], _pair_keys(i, j, n))
    return order[pos.clamp(max=max(keys.numel() - 1, 0))]


def lookup_angles(
    angle_index: torch.Tensor, n_atoms: int, i: torch.Tensor, apex: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Column of ``angle_index`` holding ``(i, apex, k)`` (end order immaterial)."""
    n = int(n_atoms)

    def key(a, c, b):
        return (torch.minimum(a, b) * n + c) * n + torch.maximum(a, b)

    keys = key(angle_index[0], angle_index[1], angle_index[2])
    order = torch.argsort(keys)
    pos = torch.searchsorted(keys[order], key(i, apex, k))
    return order[pos.clamp(max=max(keys.numel() - 1, 0))]


def torsions_from_bonds(bond_index: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``(4, Nt)`` every proper torsion ``a-b-c-d`` (``a != d``: no three-membered rings).

    One row per (outer, central, outer) choice, the central bond in its stored ``b < c``
    orientation so no torsion appears twice.
    """
    device = bond_index.device
    empty = torch.zeros(4, 0, dtype=torch.long, device=device)
    if bond_index.shape[1] == 0:
        return empty
    nbr, _ = neighbor_table(bond_index, n_atoms)
    if nbr.shape[1] == 0:
        return empty
    b, c = bond_index[0], bond_index[1]
    na = nbr[b]                                                  # (Nb, D) candidates for a
    nd = nbr[c]                                                  # (Nb, D) candidates for d
    a = na.unsqueeze(2).expand(-1, -1, nd.shape[1])
    d = nd.unsqueeze(1).expand(-1, na.shape[1], -1)
    bb = b.view(-1, 1, 1).expand_as(a)
    cc = c.view(-1, 1, 1).expand_as(a)
    keep = (a >= 0) & (d >= 0) & (a != cc) & (d != bb) & (a != d)
    return torch.stack((a[keep], bb[keep], cc[keep], d[keep]))


def angle_pairs_from_angles(bond_index: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``(4, Naa)`` rows ``[apex, s, x, y]``, ``x < y``: angle pairs sharing apex and leg ``s``."""
    device = bond_index.device
    empty = torch.zeros(4, 0, dtype=torch.long, device=device)
    if bond_index.shape[1] == 0:
        return empty
    nbr, counts = neighbor_table(bond_index, n_atoms)
    D = nbr.shape[1]
    if D < 3:
        return empty
    idx = torch.arange(D, device=device)
    s, x, y = torch.meshgrid(idx, idx, idx, indexing="ij")
    sel = (x < y) & (s != x) & (s != y)
    s, x, y = s[sel], x[sel], y[sel]                             # (K,) slot triples
    apex = torch.arange(int(n_atoms), device=device).unsqueeze(1).expand(-1, s.numel())
    ok = (
        (s.unsqueeze(0) < counts.unsqueeze(1))
        & (x.unsqueeze(0) < counts.unsqueeze(1))
        & (y.unsqueeze(0) < counts.unsqueeze(1))
    )
    a_s = nbr[:, s][ok]
    a_x = nbr[:, x][ok]
    a_y = nbr[:, y][ok]
    lo, hi = torch.minimum(a_x, a_y), torch.maximum(a_x, a_y)
    return torch.stack((apex[ok], a_s, lo, hi))


def graph_separation(
    bond_index: torch.Tensor, n_atoms: int, pair_index: torch.Tensor, max_sep: int = 3
) -> torch.Tensor:
    """``(P,)`` bond-count separation of each pair, capped at ``max_sep + 1`` ("further").

    ``1`` = 1-2, ``2`` = 1-3, ``3`` = 1-4. Pairs in different fragments are never connected
    and come out as ``max_sep + 1``.
    """
    n = int(n_atoms)
    out = torch.full((pair_index.shape[1],), max_sep + 1, dtype=torch.long,
                     device=pair_index.device)
    if bond_index.shape[1] == 0 or pair_index.shape[1] == 0:
        return out
    nbr, _ = neighbor_table(bond_index, n)
    src = torch.cat((bond_index[0], bond_index[1]))
    dst = torch.cat((bond_index[1], bond_index[0]))
    query = _pair_keys(pair_index[0], pair_index[1], n)
    for sep in range(1, max_sep + 1):
        found = src != dst
        keys = torch.unique(_pair_keys(src[found], dst[found], n))
        hit = torch.isin(query, keys) & (out > sep)
        out = torch.where(hit, torch.full_like(out, sep), out)
        if sep == max_sep:
            break
        nxt = nbr[dst]
        keep = nxt >= 0
        src = src.unsqueeze(1).expand_as(nxt)[keep]
        dst = nxt[keep]
    return out
