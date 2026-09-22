"""Geometric hydrogen-bond definitions for water, after Kumar, Schmidt and Skinner.

Implements the distance-angle descriptors of J. Chem. Phys. **126**, 204107 (2007) and the
two cutoffs that paper derives for the ``r``-``psi`` pair, which is the pair it argues is
special: of the six variables needed to fix a rigid water dimer, ``r`` and ``psi`` alone
carry almost all of the NBO ``sigma*_OH`` occupancy (rms 1.5e-3 against a mean occupancy of
2.1e-2, the best of every one- or two-variable map they tried).

Naming follows the *nuclear* convention throughout, not the NBO one: the **donor** is the
molecule whose O-H points at the other, and the **acceptor** is the molecule whose oxygen
receives it. The paper flags this collision explicitly (Sec. I) because Weinhold's orbital
picture calls the lone-pair molecule the donor.

Geometry (their Fig. 1), for a donor ``O_d-H`` and an acceptor ``O_a``
======================  ==================================================================
``R``                   ``|O_d - O_a|``
``r``                   ``|H - O_a|``, the intermolecular H...O distance
``beta``                at ``O_d``, between the donor's own O-H ray and ``O_d -> O_a``
``alpha``               at ``H``, between the extension of ``O_d -> H`` and ``H -> O_a``;
                        the complement of the usual O-H...O angle ``theta``, so ``alpha =
                        0`` is a linear hydrogen bond
``gamma``               at ``O_a``, between the extension of the acceptor's HOH bisector
                        and the intermolecular ray ``O_a -> H``
``psi``                 at ``O_a``, between ``O_a -> H`` and the acceptor's out-of-plane
                        normal -- the direction of its ``p_z`` lone-pair density, reported
                        folded into ``[0, 90]`` (see :func:`_fold_psi`)
======================  ==================================================================

``beta``/``alpha`` describe the *donor*'s orientation, ``gamma``/``psi`` the *acceptor*'s.
That distinction is the paper's main structural point: the acceptor-angle PMFs are shallow
and broad, so a hydrogen bond survives at any ``psi`` provided ``r`` shortens to compensate,
whereas a donor bent past ``beta ~ 40 deg`` is not hydrogen bonded at any distance.

Two cutoffs are provided for ``r``-``psi``, both from the paper and both reproduced here as
printed rather than refitted:

``"pmf"``
    The 0.82 kT equipotential of the 2D potential of mean force through its saddle point,
    their Eq. (3). SPC/E at 300 K.
``"occupancy"``
    ``N(r, psi) > 0.0085``, their Eqs. (4) and Fig. 10 -- an electronic-structure criterion,
    the minimum of the bimodal ``sigma*_OH`` occupancy distribution.

Both are cutoffs *of a liquid*, fitted to SPC/E configurations at ambient density. Applying
them to gas-phase clusters is a deliberate transfer: the functional forms are geometric and
carry over, but the cutoff values encode liquid packing statistics, so treat the resulting
counts as a consistent label, not as a measurement of the clusters.

Angles are degrees, distances Angstrom.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OCCUPANCY_CUTOFF",
    "dimer_geometry",
    "hbond_labels",
    "hbonds_per_molecule",
    "pmf_cutoff_radius",
    "sigma_star_occupancy",
]

#: Minimum of the liquid's bimodal ``sigma*_OH`` occupancy distribution (their Fig. 10).
OCCUPANCY_CUTOFF = 0.0085


def _fold_psi(psi: np.ndarray) -> np.ndarray:
    """``psi -> min(psi, 180 - psi)``.

    Both fits are stated for ``0 <= psi <= 90`` with the instruction to replace ``psi`` by
    its complement above 90 deg. That is not a convention: the acceptor's two ``p_z`` lobes
    are related by the molecular mirror plane, so the electronic structure genuinely is
    symmetric about ``psi = 90``, and the PMF in their Fig. 8 is plotted folded for the same
    reason.
    """
    psi = np.asarray(psi, dtype=float)
    return np.minimum(psi, 180.0 - psi)


def pmf_cutoff_radius(psi) -> np.ndarray:
    """Their Eq. (3): ``r_cut(psi)`` in Angstrom, the 0.82 kT PMF equipotential.

    A pair is hydrogen bonded when ``r < r_cut(psi)``. Ranges from 2.52 A head-on at
    ``psi = 0`` (into the out-of-plane lobe) to 1.99 A in the molecular plane.
    """
    p = _fold_psi(psi)
    return 2.52 - 0.011 * p + 0.000057 * p * p


def sigma_star_occupancy(r, psi) -> np.ndarray:
    """Their Eq. (4): the fitted NBO ``sigma*_OH`` occupancy ``N(r, psi)``, dimensionless.

    A separable exponential-in-``r`` times quadratic-in-``psi`` form, fitted to B3LYP/
    aug-cc-pVDZ NBO analyses of 10 000 SPC/E dimers. The 0.343 A decay length is short --
    ``N`` falls by a factor of ``e`` every 0.34 A -- which is why this classifier is so much
    more distance-driven than angle-driven.
    """
    r = np.asarray(r, dtype=float)
    p = _fold_psi(psi)
    return np.exp(-r / 0.343) * (7.1 - 0.050 * p + 0.00021 * p * p)


def _molecule_frame(positions: np.ndarray, o: int, hs: np.ndarray):
    """``(bisector, normal)`` unit vectors of a water molecule, both pointing out of ``O``.

    ``bisector`` is the HOH bisector as it points *toward* the hydrogens; ``gamma`` is
    measured from its extension, so the sign is flipped at the call site. ``normal`` is
    ``e1 x e2`` of the two O-H unit vectors and therefore has an arbitrary sign -- which is
    exactly why ``psi`` must be folded about 90 deg rather than taken as signed.
    """
    e = positions[hs] - positions[o]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    bisector = e.sum(axis=0)
    bisector /= np.linalg.norm(bisector)
    normal = np.cross(e[0], e[1])
    normal /= np.linalg.norm(normal)
    return bisector, normal


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


#: One row per candidate donor-H / acceptor-O pair.
GEOMETRY_DTYPE = np.dtype([
    ("donor_o", np.int64), ("h", np.int64), ("acceptor_o", np.int64),
    ("donor_frag", np.int64), ("acceptor_frag", np.int64),
    ("R", float), ("r", float),
    ("beta", float), ("alpha", float), ("gamma", float), ("psi", float),
])


def dimer_geometry(
    positions,
    atomic_numbers,
    fragment_idx,
    *,
    r_max: float = 3.0,
) -> np.ndarray:
    """Every inter-fragment ``H...O`` candidate within ``r_max``, with all six descriptors.

    Returns a structured array of :data:`GEOMETRY_DTYPE`. Every O-H of every molecule is
    tried against every oxygen of every *other* molecule, so a pair appears once per donor
    hydrogen; a doubly-donating molecule contributes two rows, and a doubly-accepting
    oxygen appears in two.

    ``r_max`` bounds the candidate set only -- it is not a hydrogen-bond criterion. It has
    to sit well outside both cutoffs (2.52 A at the most permissive angle) so that the
    classification is decided by the definition and not by this window; 3.0 A reaches into
    the first minimum of ``g_OH`` and past it. Fragments come from the partition, never
    from a distance rule: molecular identity is a graph property.
    """
    positions = np.asarray(positions, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    frag = np.asarray(fragment_idx, dtype=int)

    n_frag = int(frag.max()) + 1
    oxygen, hydrogens, frames = {}, {}, {}
    for f in range(n_frag):
        atoms = np.flatnonzero(frag == f)
        o = atoms[z[atoms] == 8]
        h = atoms[z[atoms] == 1]
        if o.size != 1 or h.size != 2:
            raise ValueError(
                f"fragment {f} has {o.size} oxygens and {h.size} hydrogens; these "
                f"descriptors are defined for water molecules only"
            )
        oxygen[f], hydrogens[f] = int(o[0]), h
        frames[f] = _molecule_frame(positions, int(o[0]), h)

    acceptor_frags = np.array(sorted(oxygen), dtype=int)
    acceptor_pos = positions[[oxygen[f] for f in acceptor_frags]]

    rows = []
    for fd in range(n_frag):
        o_d = oxygen[fd]
        for h in hydrogens[fd]:
            d = np.linalg.norm(acceptor_pos - positions[h], axis=1)
            for k in np.flatnonzero((d <= r_max) & (acceptor_frags != fd)):
                fa = int(acceptor_frags[k])
                o_a = oxygen[fa]
                bisector, normal = frames[fa]
                to_h = positions[h] - positions[o_a]        # the intermolecular O_a -> H ray
                rows.append((
                    o_d, int(h), o_a, fd, fa,
                    float(np.linalg.norm(positions[o_d] - positions[o_a])),
                    float(d[k]),
                    _angle(positions[h] - positions[o_d], positions[o_a] - positions[o_d]),
                    _angle(positions[h] - positions[o_d], positions[o_a] - positions[h]),
                    _angle(to_h, -bisector),
                    # Folded on the way out, not left raw. The normal's sign is set by which
                    # O-H the acceptor happened to list first, so a raw psi flips to its
                    # complement under a relabelling of two identical hydrogens or under a
                    # reflection of the whole cluster. Folding here makes the stored column
                    # the physical quantity; every consumer would otherwise have to remember
                    # to fold, and a PMF binned on the raw value would be smeared across
                    # both halves.
                    float(_fold_psi(_angle(to_h, normal))),
                ))
    return np.asarray(rows, dtype=GEOMETRY_DTYPE) if rows else np.empty(0, GEOMETRY_DTYPE)


def hbond_labels(geometry, *, definition: str = "occupancy") -> np.ndarray:
    """Boolean hydrogen-bond label per row of a :func:`dimer_geometry` array.

    ``definition`` is ``"occupancy"`` (Eq. 4 above :data:`OCCUPANCY_CUTOFF`; 3.4 H bonds per
    molecule in SPC/E) or ``"pmf"`` (inside the Eq. 3 equipotential; 3.2 per molecule). They
    disagree on a thin shell of near-cutoff geometries -- the paper's own estimate of the
    residual arbitrariness is that 0.2 H bonds per molecule.
    """
    r, psi = geometry["r"], geometry["psi"]
    if definition == "occupancy":
        return sigma_star_occupancy(r, psi) > OCCUPANCY_CUTOFF
    if definition == "pmf":
        return r < pmf_cutoff_radius(psi)
    raise ValueError(f"unknown r-psi definition {definition!r}; use 'occupancy' or 'pmf'")


def hbonds_per_molecule(geometry, n_fragments: int, *, definition: str = "occupancy") -> float:
    """``<n>``: hydrogen bonds *participated in* per molecule, the paper's Table I quantity.

    Each bond is counted twice, once for its donor and once for its acceptor, so this is
    ``2 * n_bonds / n_molecules`` -- the convention every number between 3 and 4 in the
    literature is quoted in.
    """
    return 2.0 * int(hbond_labels(geometry, definition=definition).sum()) / float(n_fragments)
