"""Fragment-assignment helpers for benchmark water-cluster Q-Chem inputs."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import dist, inf


Coord = tuple[float, float, float]
Atom = tuple[str, Coord]


@dataclass
class FragmentIssue:
    fragment_index: int
    atom_index: int
    symbol: str
    nearest_distance: float


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: float


def atom_line(symbol: str, coord: Coord) -> str:
    x, y, z = coord
    return f"  {symbol:<2} {x:16.8f} {y:16.8f} {z:16.8f}"


def fragment_h_counts(fragment_charges: list[int], n_hydrogen: int) -> list[int]:
    counts = [2 + charge for charge in fragment_charges]
    if any(count < 0 for count in counts) or sum(counts) != n_hydrogen:
        raise ValueError(
            "cannot infer O-H fragment counts from fragment charges "
            f"(charges={fragment_charges}, n_hydrogen={n_hydrogen})"
        )
    return counts


def _add_edge(graph: list[list[_Edge]], source: int, target: int, cap: int, cost: float) -> _Edge:
    forward = _Edge(to=target, rev=len(graph[target]), cap=cap, cost=cost)
    reverse = _Edge(to=source, rev=len(graph[source]), cap=0, cost=-cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _min_cost_flow(graph: list[list[_Edge]], source: int, sink: int, flow_needed: int) -> None:
    n_nodes = len(graph)
    potential = [0.0] * n_nodes
    flow = 0
    while flow < flow_needed:
        distances = [inf] * n_nodes
        previous: list[tuple[int, int] | None] = [None] * n_nodes
        distances[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            current_distance, node = heappop(queue)
            if current_distance != distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.cap <= 0:
                    continue
                reduced_cost = edge.cost + potential[node] - potential[edge.to]
                candidate = current_distance + reduced_cost
                if candidate < distances[edge.to]:
                    distances[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    heappush(queue, (candidate, edge.to))
        if previous[sink] is None:
            raise ValueError("could not assign all hydrogens to oxygen fragments")
        for node, value in enumerate(distances):
            if value < inf:
                potential[node] += value
        add = flow_needed - flow
        node = sink
        while node != source:
            prev_node, edge_index = previous[node]
            add = min(add, graph[prev_node][edge_index].cap)
            node = prev_node
        node = sink
        while node != source:
            prev_node, edge_index = previous[node]
            edge = graph[prev_node][edge_index]
            edge.cap -= add
            graph[node][edge.rev].cap += add
            node = prev_node
        flow += add


def minimized_oh_groups(
    symbols: list[str],
    coords: list[Coord],
    fragment_charges: list[int] | None = None,
) -> list[list[int]]:
    """Return atom indices grouped as O/H fragments by minimum total O-H distance."""

    oxygen_indices = [idx for idx, symbol in enumerate(symbols) if symbol.upper() == "O"]
    hydrogen_indices = [idx for idx, symbol in enumerate(symbols) if symbol.upper() == "H"]
    if len(oxygen_indices) + len(hydrogen_indices) != len(symbols):
        raise ValueError("benchmark fragment assignment currently supports O/H clusters only")
    if fragment_charges is None:
        fragment_charges = [0] * len(oxygen_indices)
    if len(fragment_charges) != len(oxygen_indices):
        raise ValueError("fragment charge count must match oxygen count")
    h_counts = fragment_h_counts(fragment_charges, len(hydrogen_indices))

    source = 0
    h_offset = 1
    o_offset = h_offset + len(hydrogen_indices)
    sink = o_offset + len(oxygen_indices)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    used_edges: list[tuple[int, int, _Edge]] = []

    for h_pos in range(len(hydrogen_indices)):
        _add_edge(graph, source, h_offset + h_pos, 1, 0.0)
    for h_pos, h_idx in enumerate(hydrogen_indices):
        for o_pos, o_idx in enumerate(oxygen_indices):
            edge = _add_edge(
                graph,
                h_offset + h_pos,
                o_offset + o_pos,
                1,
                dist(coords[h_idx], coords[o_idx]),
            )
            used_edges.append((h_pos, o_pos, edge))
    for o_pos, h_count in enumerate(h_counts):
        _add_edge(graph, o_offset + o_pos, sink, h_count, 0.0)

    _min_cost_flow(graph, source, sink, len(hydrogen_indices))
    assigned_h: list[list[int]] = [[] for _ in oxygen_indices]
    for h_pos, o_pos, edge in used_edges:
        if edge.cap == 0:
            assigned_h[o_pos].append(hydrogen_indices[h_pos])

    groups: list[list[int]] = []
    for o_pos, oxygen_idx in enumerate(oxygen_indices):
        hydrogens = sorted(assigned_h[o_pos], key=lambda h_idx: (dist(coords[h_idx], coords[oxygen_idx]), h_idx))
        groups.append([oxygen_idx, *hydrogens])
    return groups


def ordered_atoms_from_groups(symbols: list[str], coords: list[Coord], groups: list[list[int]]) -> list[Atom]:
    return [(symbols[idx], coords[idx]) for group in groups for idx in group]


def fragment_sanity_issues(fragments: list[list[Atom]], cutoff: float) -> list[FragmentIssue]:
    issues: list[FragmentIssue] = []
    for fragment_index, fragment in enumerate(fragments):
        for atom_index, (symbol, coord) in enumerate(fragment):
            nearest = min(
                (dist(coord, other_coord) for other_index, (_, other_coord) in enumerate(fragment) if other_index != atom_index),
                default=inf,
            )
            if nearest > cutoff:
                issues.append(
                    FragmentIssue(
                        fragment_index=fragment_index,
                        atom_index=atom_index,
                        symbol=symbol,
                        nearest_distance=nearest,
                    )
                )
    return issues


def build_plain_molecule(atoms: list[Atom], charge: int = 0, multiplicity: int = 1) -> str:
    lines = ["$molecule", f"{charge} {multiplicity}"]
    lines.extend(atom_line(symbol, coord) for symbol, coord in atoms)
    lines.append("$end")
    return "\n".join(lines)


def build_fragmented_molecule(
    symbols: list[str],
    coords: list[Coord],
    groups: list[list[int]],
    fragment_charges: list[int],
    fragment_multiplicities: list[int],
    charge: int = 0,
    multiplicity: int = 1,
) -> str:
    lines = ["$molecule", f"{charge} {multiplicity}"]
    for fragment_index, group in enumerate(groups):
        lines.append("--")
        lines.append(f"{fragment_charges[fragment_index]} {fragment_multiplicities[fragment_index]}")
        lines.extend(atom_line(symbols[idx], coords[idx]) for idx in group)
    lines.append("$end")
    return "\n".join(lines)
