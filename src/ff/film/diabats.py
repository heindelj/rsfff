"""The diabatic state of a fragment: asymptotic atomic charges and a dissociation channel per bond.

The ECE electrical heads (:mod:`rsfff.ff.film.electrical`) write every atomic charge as this
asymptotic charge plus bond charge flows that vanish as the bonds dissociate, so the diabatic state
has to say where a fragment's net charge goes. A full table of diabats is future work; until then
:func:`guess_diabatic_charges` makes a Lewis-structure guess from the covalent graph alone:

1. **Over-valent atoms** of the second period (N with four bonds, O with three) carry
   ``deg - valence`` positive charges: NH4+ -> N+, H3O+ -> O+, CH3NH3+ -> N+.
2. Every other atom has ``u = valence - degree`` free valences (P and S take the smallest of their
   expanded valences that fits). Free valences pair into pi bonds; the number of pi bonds is
   maximized over the graph.
3. The fragment's remaining charge sits on the free valences left unpaired, spread **equally over
   every atom that is left unpaired in some maximal pairing** -- the resonance average: acetate ->
   -1/2 on each O, OH- -> O-, CH3+ -> C+. A neutral leftover is a radical (OH., no charge).
4. Anything the rule cannot place (e.g. a charge with no free valence, nitro-type zwitterions) is
   spread uniformly over the fragment's heavy atoms, and :func:`guess_diabatic_charges` says so
   through its ``ok`` flag.

The result always sums exactly to the fragment charge.

**Per-bond channels.** One charge per atom fixes the homolytic channel of every bond, which is
wrong where the diabat dissociates heterolytically (CH3NH3+ -> CH3+ + NH3, NH4+ -> NH3 + H+, the
acetate O leaving as O-). A channel is the charge each side of a bond keeps once *that bond alone*
is broken, written as the asymptotic flow ``f_b``: the extra charge moved along the bond's stored
orientation ``a -> b`` relative to the per-atom charges. :func:`guess_diabat` guesses them:

* **dative bonds** of an over-valent cation centre (N+ with four bonds, O+ with three): the centre
  keeps the electron pair and the partner leaves as the cation (``f = +1`` away from the centre):
  NH4+ -> NH3 + H+, H3O+ -> H2O + H+, CH3NH3+ -> CH3+ + NH3 and CH3NH2 + H+;
* **split resonance charge**: a bond whose two sides would keep fractional parts of a resonance-
  averaged charge gives the whole charge to the smaller side (the leaving group): acetate's C-O ->
  CH3CO + O-;
* everything else is homolytic (``f = 0``), as are bonds in rings (breaking one separates nothing).

Bonds whose flows draw on the same source (the same cation centre, the same resonance group) form
a **group**: only one member's transfer is "spent" at a time (:mod:`rsfff.ff.film.electrical` uses
``t_b = p_b prod_{b' != b} (1 - p_b')`` within a group, ``p`` the channel progress), so breaking two
N-H bonds of NH4+ does not move two protons' worth of charge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = ["Diabat", "guess_diabat", "guess_diabatic_charges", "fragment_diabatic_charges",
           "fragment_diabats", "VALENCES"]

#: allowed valences, smallest first (the first one that fits the degree is used)
VALENCES: dict[int, tuple[int, ...]] = {
    1: (1,), 3: (1,), 5: (3,), 6: (4,), 7: (3,), 8: (2,), 9: (1,), 11: (1,), 14: (4,),
    15: (3, 5), 16: (2, 4, 6), 17: (1, 3, 5, 7), 35: (1,), 53: (1, 3, 5, 7),
}
_PERIOD2 = {5, 6, 7, 8, 9}


def _max_pairing(u: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> int:
    """Maximum number of pi bonds: a b-matching of free valences ``u`` over ``edges`` (brute force
    with memoization; the unsaturated subgraph of a monomer is small)."""

    @lru_cache(maxsize=None)
    def best(k: int, cap: tuple[int, ...]) -> int:
        if k == len(edges):
            return 0
        a, b = edges[k]
        out = best(k + 1, cap)
        m = min(cap[a], cap[b])
        for n in range(1, m + 1):
            c = list(cap)
            c[a] -= n
            c[b] -= n
            out = max(out, n + best(k + 1, tuple(c)))
        return out

    return best(0, tuple(u))


@dataclass
class Diabat:
    """One fragment's (or frame's) diabatic state.

    ``q``     : (n,) per-atom asymptotic charges (all bonds broken, homolytically).
    ``flow``  : (B,) asymptotic flow of each bond along its stored orientation ``a -> b``.
    ``group`` : (B,) atom index identifying the bond's exclusive group (cation centre or resonance
                group representative), ``-1`` for none.
    ``ok``    : the Lewis guess placed every charge.
    """

    q: np.ndarray
    flow: np.ndarray
    group: np.ndarray
    ok: bool = True


def guess_diabatic_charges(numbers, bonds, charge: float = 0.0) -> tuple[np.ndarray, bool]:
    """``(q (n,), ok)`` for one fragment: atomic numbers ``(n,)``, bonds ``(B, 2)`` local indices."""
    q, ok, _, _ = _lewis(numbers, bonds, charge)
    return q, ok


def _lewis(numbers, bonds, charge):
    """``(q, ok, cation_centres, resonance_groups)``."""
    z = [int(v) for v in np.asarray(numbers).reshape(-1)]
    n = len(z)
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    deg = np.bincount(b.reshape(-1), minlength=n) if b.size else np.zeros(n, dtype=np.int64)
    q = np.zeros(n)
    u = [0] * n
    centres: list[int] = []
    groups: list[list[int]] = []
    for i, zi in enumerate(z):
        vals = VALENCES.get(zi, (int(deg[i]),))
        fit = [v for v in vals if v >= deg[i]]
        if fit:
            u[i] = int(fit[0] - deg[i])
        elif zi in _PERIOD2:
            q[i] = float(deg[i] - vals[-1])            # over-valent: N(4) -> +1, O(3) -> +1
            centres.append(i)
    rest = float(charge) - q.sum()
    ok = True
    if abs(rest) > 1e-12:
        edges = tuple((int(a), int(c)) for a, c in b if u[a] > 0 and u[c] > 0)
        full = _max_pairing(tuple(u), edges)
        cand = []
        for i in range(n):
            if u[i] > 0:
                red = list(u)
                red[i] -= 1
                if _max_pairing(tuple(red), edges) == full:
                    cand.append(i)
        leftover = sum(u) - 2 * full
        if cand and leftover >= abs(rest) - 1e-12:
            q[cand] += rest / len(cand)
            if len(cand) > 1:
                groups.append(cand)
        else:
            heavy = [i for i in range(n) if z[i] > 1] or list(range(n))
            q[heavy] += rest / len(heavy)
            ok = False
    return q, ok, centres, groups


def _side(n: int, bonds: np.ndarray, k: int) -> np.ndarray | None:
    """Atoms on the ``b`` side of bond ``k`` (``bonds[k] = (a, b)``) once it is removed; ``None`` if
    the bond is in a ring (``a`` is still reachable)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for m, (a, b) in enumerate(bonds):
        if m != k:
            adj[a].append(b)
            adj[b].append(a)
    a0, b0 = int(bonds[k, 0]), int(bonds[k, 1])
    seen = {b0}
    stack = [b0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    if a0 in seen:
        return None
    mask = np.zeros(n, dtype=bool)
    mask[list(seen)] = True
    return mask


def guess_diabat(numbers, bonds, charge: float = 0.0) -> Diabat:
    """:class:`Diabat` of one fragment from its covalent graph (see the module docstring)."""
    numbers = np.asarray(numbers).reshape(-1)
    n = numbers.shape[0]
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    q, ok, centres, groups = _lewis(numbers, b, charge)
    flow = np.zeros(b.shape[0])
    group = -np.ones(b.shape[0], dtype=np.int64)
    member = {i: min(g) for g in groups for i in g}
    for k, (a, c) in enumerate(b):
        side_b = _side(n, b, k)
        if side_b is None:
            continue
        if a in centres or c in centres:
            centre = a if a in centres else c
            # the centre keeps the pair; its +1 leaves with the partner's side
            flow[k] = q[centre] if centre == a else -q[centre]
            group[k] = centre
            continue
        qb = float(q[side_b].sum())
        if abs(qb - round(qb)) > 1e-9:                 # a resonance charge split across the bond
            small_b = side_b.sum() <= n - side_b.sum()
            qs = qb if small_b else float(q.sum()) - qb
            target = math.copysign(math.ceil(abs(qs) - 1e-9), qs)
            extra = target - qs                         # charge moved onto the smaller side
            flow[k] = extra if small_b else -extra
            reps = {member[i] for i in range(n) if i in member}
            group[k] = min(reps) if reps else -1
    return Diabat(q, flow, group, ok)


def fragment_diabatic_charges(numbers, bonds, fragment_idx, fragment_charges) -> tuple[np.ndarray, bool]:
    """Per-atom asymptotic charges of a whole frame: :func:`guess_diabatic_charges` per fragment.
    ``bonds`` (B, 2) frame-local atom indices (never crossing fragments)."""
    numbers = np.asarray(numbers).reshape(-1)
    fi = np.asarray(fragment_idx, dtype=np.int64).reshape(-1)
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(numbers.shape[0])
    ok_all = True
    for f, qf in enumerate(np.asarray(fragment_charges, dtype=np.float64).reshape(-1)):
        atoms = np.nonzero(fi == f)[0]
        local = -np.ones(numbers.shape[0], dtype=np.int64)
        local[atoms] = np.arange(atoms.shape[0])
        inside = (local[b[:, 0]] >= 0) & (local[b[:, 1]] >= 0) if b.size else np.zeros(0, bool)
        q, ok = guess_diabatic_charges(numbers[atoms], local[b[inside]], float(qf))
        out[atoms] = q
        ok_all = ok_all and ok
    return out, ok_all


def fragment_diabats(numbers, bonds, fragment_idx, fragment_charges) -> Diabat:
    """:func:`guess_diabat` per fragment of a frame; ``bonds`` (B, 2) frame-local, never crossing
    fragments. Groups are frame-local atom indices."""
    numbers = np.asarray(numbers).reshape(-1)
    fi = np.asarray(fragment_idx, dtype=np.int64).reshape(-1)
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    q = np.zeros(numbers.shape[0])
    flow = np.zeros(b.shape[0])
    group = -np.ones(b.shape[0], dtype=np.int64)
    ok_all = True
    for f, qf in enumerate(np.asarray(fragment_charges, dtype=np.float64).reshape(-1)):
        atoms = np.nonzero(fi == f)[0]
        local = -np.ones(numbers.shape[0], dtype=np.int64)
        local[atoms] = np.arange(atoms.shape[0])
        inside = ((local[b[:, 0]] >= 0) & (local[b[:, 1]] >= 0)) if b.size else np.zeros(0, bool)
        d = guess_diabat(numbers[atoms], local[b[inside]], float(qf))
        q[atoms] = d.q
        flow[inside] = d.flow
        group[inside] = np.where(d.group >= 0, atoms[np.clip(d.group, 0, None)], -1)
        ok_all = ok_all and d.ok
    return Diabat(q, flow, group, ok_all)
