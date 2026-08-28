"""Harvest transition structures from confined, routing-weight-biased MD.

The mediator is fitted on 399 contested geometries and that is what limits work on it. This
script grows the corpus: put a seed ion in a few waters, hold them together with a weak
spherical wall, restrain the mediator toward a split membership, and keep every frame where
the model is being asked a question one fragmentation cannot answer.

    python scripts/run_reactive_md.py --n-waters 3 --ion h3o+ \
        --steps 20000 --target-frames 500 --out qchem_roundtrip/biased_sampling/h3o+_w3

    python scripts/run_reactive_md.py --geometry data/hydroxide_clusters/jp5b03893_si_002.xyz \
        --frame 12 --out .../oh-_w3_iso00        # charge inferred from the composition

**The dynamics are expected to blow up.** These clusters are well outside what the model was
fitted on -- w1 and w2 ions, two and three waters -- so the surface has cliffs, and a
thermostatted trajectory finds them. That is a property of the model and the run is not
designed to avoid it: every frame is checked, written the moment it passes, and a blowup
rewinds to a recent sane state, redraws velocities and continues. Nothing sampled before an
explosion is lost, and the run is judged on how many usable structures it produced.

A frame is kept when its membership is genuinely split (``--min-ambiguity``), the geometry is
one a proton transfer actually looks like (``--max-oh``, ``--max-oo``), no two atoms have
collapsed onto each other, the energy is finite and the forces are bounded. Consecutive keeps
are spaced by ``--min-gap`` so a slow crossing does not yield fifty copies of itself.

The geometry guards are not a nicety. The bias reaches a split membership by whichever route is
cheapest, and stretching one O-H is cheaper than compressing an O-O -- so left alone it produces
*stranded* protons rather than shared ones. Measured over the 5511 structures of the first,
unguarded harvest:

* the two oxygens flanking the most-stretched hydrogen are **more than 2.75 Angstrom apart in
  35% of frames**, and more than 3.0 in 15%. A proton "shared" across a 3.3 Angstrom O-O is not
  shared at all -- those are two separate minima with a barrier between them.
* a hydrogen sits **more than 1.45 Angstrom from any oxygen in 5.6%** of frames, which is a
  dissociated proton rather than a transferring one.

``--max-oo`` is therefore the guard that does the work and ``--max-oh`` is a backstop. Note that
an absolute O-H cutoff *cannot* be the main test: on a 2.6 Angstrom bond a genuinely shared
proton sits at 1.30, so anything tight enough to catch the pathology also throws away good
transition structures. Guarding costs throughput -- 49 structures/min unguarded against 10 with
both guards on -- because it rejects precisely the region the bias prefers.

Note ``--cv delta``, which restrains the geometric transfer coordinate instead of the
mediator's own weights. Biasing on ``logit`` samples where *this* model thinks the ambiguity
is; a ``delta`` run is the control that tells a real sampling gain apart from the model
steering toward its own opinion.

Output in ``--out``: ``transition_structures.xyz`` (extxyz, one frame per kept structure, with
the mediator weights and the competing fragmentations in the header), ``diagnostics.npz`` (per
step) and ``run.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch                                                              # noqa: E402
from ase import Atoms, units                                              # noqa: E402
from ase.io import read, write                                            # noqa: E402
from ase.md.langevin import Langevin                                      # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary  # noqa: E402
from ase.optimize import FIRE                                             # noqa: E402

from rsfff.md import (  # noqa: E402
    HarmonicBias,
    MediatedCalculator,
    enumerate_group,
    load_mediated_model,
)
from rsfff.md.calculator import snapshot_info                             # noqa: E402

class _Unstable(RuntimeError):
    """The trajectory left anything the model can describe. Recoverable, not fatal."""


DEFAULT_CHECKPOINT = "checkpoints/ion_mediator_v4_full/best.pt"


# ---------------------------------------------------------------------------------------
# The starting cluster
# ---------------------------------------------------------------------------------------

def _shared_proton_dimer(ion: str):
    """H5O2+ or H3O2- built directly, not carved out of a benchmark cluster.

    The two-oxygen case has no room to place a proton by the rules used for larger clusters.
    In a water dimer the O-O axis is already occupied by the donated hydrogen, so putting the
    new one there lands it ~0.9 Angstrom away (measured: 1571 eV/Angstrom at step 0), and
    every non-clashing direction points at the vacuum and enumerates ``M = 1``. A shared proton
    between two waters is not a perturbation of a dimer; it is its own geometry, so it is
    written out. Same construction as the scans in ``notebooks/mediator_plotting.ipynb``.
    """
    def unit(theta_deg, phi_deg, sign=1.0):
        t, ph = math.radians(theta_deg), math.radians(phi_deg)
        return np.array([sign * math.cos(t),
                         math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph)])

    if ion == "h3o+":                       # Zundel: O H H | O H H | bridging H
        r_oo, r_oh, theta = 2.42, 0.98, 112.0
        o1, o2 = np.zeros(3), np.array([r_oo, 0.0, 0.0])
        pos = [o1]
        pos += [o1 + r_oh * unit(theta, ph, -1.0) for ph in (60.0, 300.0)]
        pos += [o2]
        pos += [o2 + r_oh * unit(theta, ph, +1.0) for ph in (120.0, 240.0)]
        pos += [np.array([0.5 * r_oo, 0.0, 0.0])]
        return np.array(pos), ["O", "H", "H", "O", "H", "H", "H"]

    r_oo, r_oh, theta = 2.47, 0.97, 105.0   # H3O2-: O H | O H | bridging H
    o1, o2 = np.zeros(3), np.array([r_oo, 0.0, 0.0])
    pos = [o1, o1 + r_oh * unit(theta, 90.0, -1.0),
           o2, o2 + r_oh * unit(theta, 270.0, +1.0),
           np.array([0.5 * r_oo, 0.0, 0.0])]
    return np.array(pos), ["O", "H", "O", "H", "H"]


def _placement_directions(pos, sym, site: int, others: list[int]):
    """Unit vectors to try for a new proton on ``site``, best first.

    Toward each neighbouring oxygen, nearest first -- a hydronium wants to donate, and pointing
    the proton at an acceptor is what gives the mediator a competing decomposition to weigh.
    Then the lone-pair bisector as a fallback, because in a well-connected cluster *every* O-O
    axis is already occupied by a donated hydrogen and all the good directions clash. The
    bisector often faces outward and enumerates nothing, which is why it is last rather than
    first, but a cluster with no free axis needs it.
    """
    dirs = []
    for o in sorted(others, key=lambda o: np.linalg.norm(pos[o] - pos[site])):
        v = pos[o] - pos[site]
        dirs.append(v / np.linalg.norm(v))
    own = sorted((i for i, sy in enumerate(sym) if sy == "H"),
                 key=lambda h: np.linalg.norm(pos[h] - pos[site]))[:2]
    if own:
        b = -sum((pos[h] - pos[site]) / np.linalg.norm(pos[h] - pos[site]) for h in own)
        if np.linalg.norm(b) > 1e-6:
            dirs.append(b / np.linalg.norm(b))
    return dirs


def _protonate(pos, sym, site: int, direction):
    """Add a proton to ``site``, 1.0 Angstrom out along ``direction``."""
    return np.vstack([pos, pos[site] + 1.0 * np.asarray(direction)]), sym + ["H"]


def _clashes(pos, new_atom: int, cutoff: float = 0.9) -> bool:
    """Is the newly placed proton on top of something?

    The O-O axis of a hydrogen-bonded pair is exactly where the donated hydrogen already is,
    so the natural placement is also the one most likely to overlap. Unchecked this does not
    fail gracefully: the geometry is off any surface the model was fitted on and the first
    force is three orders of magnitude too large.
    """
    d = np.linalg.norm(pos - pos[new_atom], axis=-1)
    d[new_atom] = np.inf
    return bool(d.min() < cutoff)


def _deprotonate(pos, sym, hydrogen: int):
    """Remove one hydrogen, leaving a hydroxide."""
    mask = [i for i in range(len(sym)) if i != hydrogen]
    return pos[mask], [sym[i] for i in mask]


class Defect(NamedTuple):
    """Why a geometry was rejected, and whether it is still worth labelling.

    ``negative_space`` marks the rejects that are *useful*: a real, thermally reachable
    configuration that the model should learn is expensive. A stranded proton is exactly that
    -- the bias found it because nothing in the training set penalizes it, and the way to close
    that hole is to label it and put it in as repulsive data.

    A collapsed frame is not. Two nuclei 0.4 Angstrom apart is not a configuration the model
    needs a number for; it is off any surface Q-Chem will converge on, and its energy would
    dominate any loss it appeared in.
    """

    reason: str
    negative_space: bool


def geometry_defect(atoms, *, min_distance: float, max_oh: float,
                    max_oo: float) -> Defect | None:
    """Why this geometry is not worth keeping as a transition structure, or ``None``.

    Purely geometric -- it reads positions and nothing else. An earlier version asked the
    enumeration which host a contested proton would move to, which sounds more principled and
    is worse in two ways: the answer depends on the *held base*, so the same geometry is judged
    differently mid-trajectory than it is when re-read from a file, and a candidate whose
    envelope has already closed is not in the list to be measured at all. Re-classifying a
    harvest then disagreed with the guard that produced it -- 3 rejects out of 500 where the
    guard had rejected 35%.

    The three tests, in increasing order of what they are for:

    ``min_distance`` catches a collapsed frame -- two nuclei on top of each other. Not a
    chemical structure and not labelable; ``negative_space=False``.

    ``max_oh`` catches a **dissociated** proton: a hydrogen further than this from *any* oxygen
    is bound to nothing. A backstop, and it cannot be tightened much -- on a 2.6 Angstrom
    hydrogen bond a genuinely shared proton sits at 1.30, so a cut below ~1.4 throws away the
    structures the run exists to find.

    ``max_oo`` is the one that does the work. Take the most-stretched hydrogen and the two
    oxygens nearest it: a proton transfer happens on a *compressed* hydrogen bond, and across a
    3.3 Angstrom O-O the two protonation states are separate minima with a barrier between
    them. A proton parked in the middle of that is a stretched bond, not a transition
    structure. Measured over the first, unguarded harvest of 5511 frames: 35% exceed 2.75
    Angstrom and 15% exceed 3.0.
    """
    d = atoms.get_all_distances()
    np.fill_diagonal(d, np.inf)
    if d.min() < min_distance:
        return Defect(f"atoms {d.min():.2f} Angstrom apart", negative_space=False)

    z = atoms.get_atomic_numbers()
    oxy, hyd = np.flatnonzero(z == 8), np.flatnonzero(z == 1)
    if oxy.size < 1 or hyd.size < 1:
        return None

    d_oh = d[np.ix_(hyd, oxy)]                       # (H, O)
    nearest = d_oh.min(axis=1)
    if max_oh > 0 and nearest.max() > max_oh:
        return Defect(f"H stranded {nearest.max():.2f} Angstrom from any O",
                      negative_space=True)

    if max_oo > 0 and oxy.size >= 2:
        loose = int(nearest.argmax())                # the hydrogen furthest from its own oxygen
        near_two = np.argsort(d_oh[loose])[:2]
        span = float(d[oxy[near_two[0]], oxy[near_two[1]]])
        if span > max_oo:
            return Defect(f"most-stretched H spans a {span:.2f} Angstrom O-O",
                          negative_space=True)
    return None


def infer_charge(atoms: Atoms) -> int:
    """``+1`` for H3O+(H2O)n, ``-1`` for OH-(H2O)n, ``0`` for neutral water, from the formula.

    Read off the composition rather than the file, because none of the cluster sets records it:
    the CCDB hydronium files carry only an ASP model energy in kJ/mol and the hydroxide SI file
    carries only an isomer label. ``nH = 2 nO`` is neutral, one more is a hydronium, one fewer a
    hydroxide -- which is the whole space this model covers.
    """
    z = atoms.get_atomic_numbers()
    n_o, n_h = int((z == 8).sum()), int((z == 1).sum())
    if n_o + n_h != len(z):
        raise SystemExit("charge inference supports O/H clusters only; pass --charge")
    excess = n_h - 2 * n_o
    if excess not in (-1, 0, 1):
        raise SystemExit(
            f"composition H{n_h}O{n_o} implies a charge of {excess:+d}, which is not a single "
            f"H3O+ or OH- in water; pass --charge if that is deliberate"
        )
    return excess


def load_geometry(path: str, frame: int, charge: int | None) -> tuple[Atoms, int]:
    """A starting structure straight out of a cluster file.

    Nothing here looks at atom *order*, and that is deliberate. The two sets on disk disagree
    about where the ion sits -- the CCDB hydronium files put it **last** (``[O H H] x N`` then
    ``[O H H H]``) while the hydroxide SI file puts it **first** (``[O H]`` then the waters) --
    and the training corpus disagrees with both. Worse, ``Isomer 7j`` of the hydroxide file
    writes one water as ``H, O, H``, so a parser that gives each oxygen the hydrogens that
    follow it invents an H3O+/OH- pair inside a hydroxide cluster. Fragments come from
    :func:`rsfff.md.assign.rank_oh_fragment_assignments`, which minimizes the total O-H
    distance and never reads a position, so all three layouts are handled by the same code.
    """
    atoms = read(path, index=frame)
    if isinstance(atoms, list):                      # `index=":"`-style result
        atoms = atoms[0]
    atoms = Atoms(symbols=atoms.get_chemical_symbols(), positions=atoms.get_positions())
    return atoms, infer_charge(atoms) if charge is None else int(charge)


def build_cluster(n_waters: int, ion: str) -> tuple[Atoms, int]:
    """``n_waters`` waters **plus** one seed ion, from the MP2 benchmark cluster geometries.

    The count is waters *in addition to* the ion, so ``n_waters`` oxygens carry an intact water
    and one more carries the H3O+ or OH-. ``--n-waters 1 --ion h3o+`` is therefore H5O2+, the
    Zundel, and not a bare hydronium with nothing to react with. Counting the ion as one of the
    waters would make ``--n-waters 1`` a single fragment: ``M = 1``, no competing decomposition,
    and a bias with nothing to act on.

    Starting from an optimized neutral cluster and adding or removing one proton keeps the
    hydrogen-bond network intact, which matters more than the ion's own geometry -- the
    thermostat fixes a 1.0 Angstrom guess in a few hundred steps, but it will not reassemble a
    network that was never there.

    **The site is chosen by whether it produces a competitor, not by geometry alone.** Which
    oxygen to charge and which proton to move are both free choices, and most of them give a
    cluster the mediator has nothing to say about: an ion facing outward enumerates ``M = 1``,
    the run has no ambiguity to bias, and the guard rail aborts it 8000 steps later. So every
    (site, partner) pair is tried in order of preference and the first with a live competing
    decomposition is taken.
    """
    if ion not in ("h3o+", "oh-"):
        raise SystemExit(f"unknown ion {ion!r}, expected 'h3o+' or 'oh-'")
    if n_waters == 1:
        pos, sym = _shared_proton_dimer(ion)
        return Atoms(symbols=sym, positions=pos), (+1 if ion == "h3o+" else -1)

    n_oxygen = n_waters + 1
    source = ROOT / "benchmarks" / "structures" / f"w{max(n_oxygen, 4)}_mp2_avtz.xyz"
    if not source.exists():
        raise SystemExit(
            f"no benchmark cluster with {n_oxygen} oxygens ({source} missing); "
            f"benchmarks/structures holds w4 through w23"
        )
    at = read(str(source))
    pos, sym = at.get_positions(), at.get_chemical_symbols()

    if n_oxygen < 4:
        # Trim to a *connected* subset, grown outward from the most central oxygen. Trimming
        # by centrality alone picks whichever oxygens sit nearest the centre of mass, and in a
        # cyclic tetramer that is a diagonal pair 3.9 Angstrom apart -- two waters that are
        # not hydrogen bonded to each other, so no proton placement between them is ever in
        # range and the cluster enumerates M = 1.
        oxy = [i for i, s in enumerate(sym) if s == "O"]
        centroid = pos[oxy].mean(0)
        keep_o = [min(oxy, key=lambda i: np.linalg.norm(pos[i] - centroid))]
        while len(keep_o) < n_oxygen:
            rest = [o for o in oxy if o not in keep_o]
            keep_o.append(min(rest, key=lambda o: min(
                np.linalg.norm(pos[o] - pos[k]) for k in keep_o)))
        keep = []
        for o in keep_o:
            hs = sorted(
                (i for i, s in enumerate(sym) if s == "H"),
                key=lambda h: np.linalg.norm(pos[h] - pos[o]),
            )[:2]
            keep += [o] + hs
        keep.sort()
        pos, sym = pos[keep], [sym[i] for i in keep]

    oxy = [i for i, s in enumerate(sym) if s == "O"]
    centroid = pos[oxy].mean(0)
    charge = +1 if ion == "h3o+" else -1

    # Prefer the most solvated oxygen, then its closest neighbour; fall outward from there.
    candidates = []
    for site in sorted(oxy, key=lambda i: np.linalg.norm(pos[i] - centroid)):
        partners = sorted([o for o in oxy if o != site],
                          key=lambda o: np.linalg.norm(pos[o] - pos[site]))
        if charge > 0:
            candidates += [(site, v) for v in _placement_directions(pos, sym, site, partners)]
        else:
            # Which of the site's own hydrogens to remove. Taking the one furthest from any
            # other oxygen keeps a donated hydrogen bond intact rather than deleting it.
            own = sorted((i for i, s in enumerate(sym) if s == "H"),
                         key=lambda h: np.linalg.norm(pos[h] - pos[site]))[:2]
            candidates += sorted(
                ((site, h) for h in own),
                key=lambda sh: -min(np.linalg.norm(pos[sh[1]] - pos[o]) for o in partners),
            )

    fallback = None
    for site, other in candidates:
        p2, s2 = (_protonate(pos, sym, site, other) if charge > 0
                  else _deprotonate(pos, sym, other))
        if charge > 0 and _clashes(p2, len(s2) - 1):
            continue                       # the new proton landed on an existing atom
        atoms = Atoms(symbols=s2, positions=p2)
        if fallback is None:
            fallback = atoms                # first placement that at least does not overlap
        group = enumerate_group(
            torch.as_tensor(p2, dtype=torch.float64),
            torch.as_tensor(atoms.get_atomic_numbers()),
            charge,
        )
        if group.fragments.shape[0] > 1:
            return atoms, charge

    # Every placement was isolated. Hand back the first anyway rather than refusing: the
    # thermostat may well close a hydrogen bond in the first hundred steps, and the
    # no-competition guard will stop the run if it does not.
    if fallback is None:
        raise SystemExit(
            f"every placement of the {ion} on a {n_oxygen}-oxygen cluster overlapped an "
            f"existing atom. This should not happen; check benchmarks/structures/"
            f"w{max(n_oxygen, 4)}_mp2_avtz.xyz."
        )
    print(f"WARNING: no placement of the {ion} gave a competing decomposition at the starting "
          f"geometry; the run will abort on the no-competition guard unless dynamics closes a "
          f"hydrogen bond first.", flush=True)
    return fallback, charge


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--n-waters", type=int, default=3,
                   help="waters IN ADDITION to the ion, so the cluster has n+1 oxygens. "
                        "1 + h3o+ is H5O2+ (the Zundel), 1 + oh- is H3O2-.")
    p.add_argument("--ion", default="h3o+", choices=["h3o+", "oh-"])
    p.add_argument("--geometry",
                   help="start from this file instead of synthesising a cluster. Overrides "
                        "--n-waters/--ion. Any xyz/extxyz; atom order is irrelevant.")
    p.add_argument("--frame", type=int, default=0, help="frame index within --geometry")
    p.add_argument("--charge", type=int,
                   help="total charge. Default: inferred from the composition, +1 when "
                        "nH = 2 nO + 1 and -1 when nH = 2 nO - 1.")
    p.add_argument("--cv", default="logit", choices=["logit", "ambiguity", "delta"],
                   help="bias coordinate. 'logit' is the routing-weight one to use -- it does "
                        "not saturate, so one force constant works across the whole range. "
                        "'ambiguity' is the raw 1-sum(w^2) and needs a k that is unusable in "
                        "dynamics. 'delta' is the model-free control.")
    p.add_argument("--k-bias", type=float, default=0.03,
                   help="0 disables the bias exactly. Measured working scale: 0.03-0.1 Ha for "
                        "--cv logit, 0.3-1 Ha/Angstrom^2 for --cv delta. See "
                        "rsfff.md.bias.HarmonicBias for why --cv ambiguity needs ~10 Ha and "
                        "is unusable in dynamics even so.")
    p.add_argument("--target", type=float, default=0.0,
                   help="0 is the 50/50 membership for --cv logit and the transfer point for "
                        "--cv delta. For --cv ambiguity it is A0, whose maximum is 1-1/M -- "
                        "and do not target the maximum, the gradient vanishes there.")
    p.add_argument("--radius", type=float, default=0.0,
                   help="Angstrom, flat-bottom wall. 0 derives it from the starting geometry "
                        "as max|r_O - COM| + 1.5, which leaves every oxygen a comfortable "
                        "margin and still stops a water evaporating. A fixed radius across "
                        "cluster sizes is either slack for the small ones or a vice for the "
                        "big ones.")
    p.add_argument("--k-confine", type=float, default=0.02, help="Hartree/Angstrom^2")
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--timestep", type=float, default=0.25,
                   help="fs. The proton is the reaction coordinate, so no "
                        "hydrogen-mass repartitioning and no 1 fs shortcut.")
    p.add_argument("--friction", type=float, default=0.02, help="1/fs, Langevin")
    p.add_argument("--steps", type=int, default=20000,
                   help="step budget. The run stops at this, at --target-frames, or at "
                        "--max-restarts, whichever comes first.")
    p.add_argument("--target-frames", type=int, default=500,
                   help="stop once this many transition structures have been harvested")
    p.add_argument("--min-ambiguity", type=float, default=0.15,
                   help="keep a frame only if 1-sum(w^2) is at least this. 0.15 is a "
                        "membership of roughly 92/8 or more even -- decided frames are not "
                        "what this run is for and there is no point labelling them.")
    p.add_argument("--min-gap", type=int, default=20,
                   help="minimum steps between kept frames, so a slow crossing does not "
                        "produce fifty near-identical structures")
    p.add_argument("--check-every", type=int, default=1,
                   help="steps between sanity/harvest checks. 1 is free -- the observer only "
                        "reads results the force call already produced -- and a transition "
                        "can be over in a few tens of steps.")
    p.add_argument("--report-every", type=int, default=500, help="steps between progress lines")
    p.add_argument("--max-restarts", type=int, default=5000,
                   help="give up after this many blowups. High on purpose: a rewind costs "
                        "one force call and the instability is expected, so the step budget "
                        "should be what ends the run, not the restart count.")
    p.add_argument("--safe-history", type=int, default=40,
                   help="how many recent sane states to keep for rewinding")
    p.add_argument("--patience", type=int, default=3,
                   help="rewind one extra frame for every this many restarts, so repeatedly "
                        "diving into the same hole backs out of it")
    p.add_argument("--negative-fraction", type=float, default=0.25,
                   help="also keep rejected geometries, up to this many per accepted one, in "
                        "negative_structures.xyz with the rejection reason in the header. "
                        "These are the configurations the model thinks are cheap and the "
                        "reference method will say are expensive -- the label that stops the "
                        "bias walking there again. Only the *geometry* rejects qualify; a "
                        "diverged frame says nothing about the true surface. 0 disables.")
    p.add_argument("--min-distance", type=float, default=0.6,
                   help="Angstrom; a frame with two atoms closer than this is discarded")
    p.add_argument("--max-oh", type=float, default=1.45,
                   help="Angstrom; discard a frame in which any hydrogen is further than this "
                        "from its NEAREST oxygen. A backstop for a genuinely dissociated "
                        "proton, not the main guard -- a shared proton on a 2.6 A bond sits at "
                        "1.30 legitimately, so a tighter cut rejects good transition "
                        "structures. Rejects 5.6% of the first harvest. 0 disables it.")
    p.add_argument("--max-oo", type=float, default=2.75,
                   help="Angstrom; discard a frame in which the contested proton's donor and "
                        "acceptor oxygens are further apart than this. THIS is the real guard: "
                        "a transferring proton lives on a compressed hydrogen bond, and across "
                        "a 3.3 A O-O the two states are separate minima with nothing to "
                        "mediate. Rejects 35% of the first harvest. 0 disables it.")
    p.add_argument("--relax", type=int, default=0,
                   help="FIRE steps before dynamics; 0 (the default) skips it. Off by default "
                        "because an optimizer is the wrong tool on this surface: FIRE "
                        "accelerates along a consistent force direction and walks straight "
                        "into the cliffs the model has outside its training set -- measured, "
                        "|F| going from 1.4 to 10^5 eV/Angstrom within five steps. A "
                        "thermostat is far more forgiving. When enabled it is force-guarded "
                        "and stops the moment the geometry gets worse.")
    p.add_argument("--ramp", type=int, default=2000,
                   help="steps over which --k-bias is raised linearly from 0. Switching a "
                        "restraint on at full strength dumps 0.5*k*(cv-target)^2 into the "
                        "cluster in one step; ramping lets the proton follow it instead. 0 "
                        "applies the bias at full strength immediately.")
    p.add_argument("--max-deviation", type=float, default=1.0,
                   help="restraint goes linear past this deviation, capping the bias force at "
                        "k*max_deviation*|dcv/dR|. 0 means a pure harmonic (unbounded).")
    p.add_argument("--no-induction", action="store_true",
                   help="skip the coupled solve: ~1.5x faster, and a different surface")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-force", type=float, default=50.0,
                   help="eV/Angstrom; abort above this. Set for divergence, not for busy "
                        "dynamics: a thermal O-H stretch alone reaches ~5 eV/Angstrom and a "
                        "working bias adds several more, so a tight bound aborts healthy runs.")
    p.add_argument("--max-temperature", type=float, default=0.0,
                   help="K; abort above this. 0 derives it as 5x --temperature. This is the "
                        "better divergence signal -- a discontinuity pumps the thermostat "
                        "long before any single force looks anomalous.")
    p.add_argument("--out", default="logs/reactive_md")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.geometry:
        atoms, charge = load_geometry(args.geometry, args.frame, args.charge)
        source = f"{Path(args.geometry).name}[{args.frame}]"
    else:
        atoms, charge = build_cluster(args.n_waters, args.ion)
        if args.charge is not None and int(args.charge) != charge:
            raise SystemExit(
                f"--charge {args.charge:+d} contradicts a seeded {args.ion} cluster "
                f"({charge:+d}); drop one of them"
            )
        source = f"{args.ion} + {args.n_waters} waters"
    if args.radius <= 0.0:
        m, p_ = atoms.get_masses(), atoms.get_positions()
        com = (m[:, None] * p_).sum(0) / m.sum()
        oxy = atoms.get_atomic_numbers() == 8
        args.radius = float(np.linalg.norm(p_[oxy] - com, axis=1).max() + 1.5)
    print(f"cluster: {atoms.get_chemical_formula()}  charge {charge:+d}  "
          f"{len(atoms)} atoms  confinement radius {args.radius:.2f} A  ({source})", flush=True)

    model, cfg, _state = load_mediated_model(args.checkpoint)
    calc = MediatedCalculator(
        model, charge,
        bias=HarmonicBias(cv=args.cv, k=args.k_bias, target=args.target,
                          max_deviation=args.max_deviation or None),
        radius=args.radius, k_confine=args.k_confine,
        with_induction=not args.no_induction,
    )
    atoms.calc = calc

    if args.relax:
        # Relax on the *unbiased* surface: the initial proton placement is a guess, and
        # relaxing with the bias on would optimize the guess toward the bias instead.
        held, calc.bias = calc.bias, HarmonicBias(k=0.0)
        opt = FIRE(atoms, logfile=None, maxstep=0.05)
        best = (np.abs(atoms.get_forces()).max(), atoms.get_positions().copy())
        for _ in range(args.relax):
            opt.run(fmax=0.5, steps=opt.nsteps + 1)
            f = float(np.abs(atoms.get_forces()).max())
            if f > 5.0 * best[0] + 1.0:      # the optimizer has found a cliff, not a minimum
                atoms.set_positions(best[1])
                print(f"  relax aborted: |F| rose to {f:.1f} eV/Angstrom; keeping the best "
                      f"geometry seen ({best[0]:.2f})", flush=True)
                break
            if f < best[0]:
                best = (f, atoms.get_positions().copy())
        else:
            atoms.set_positions(best[1])
        calc.bias = held
        calc.reset_base()
        print(f"relaxed to |F|max {best[0]:.2f} eV/Angstrom", flush=True)

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    dyn = Langevin(
        atoms, timestep=args.timestep * units.fs,
        temperature_K=args.temperature, friction=args.friction / units.fs,
    )

    t_ceiling = args.max_temperature or 5.0 * args.temperature
    log: dict[str, list] = {k: [] for k in
                            ("step", "time_fs", "temperature", "energy_hartree", "bias_energy",
                             "confine_energy", "cv", "ambiguity", "occupancy",
                             "n_decompositions", "n_commits", "max_force", "harvested",
                             "negative")}
    traj_path = out / "transition_structures.xyz"
    negative_path = out / "negative_structures.xyz"
    for path in (traj_path, negative_path):
        if path.exists():
            path.unlink()
    state = {"t0": time.time(), "harvested": 0, "restarts": 0, "last_kept": -10**9,
             "negative": 0, "last_negative": -10**9}
    # Ring buffer of recent sane (positions, momenta). A blowup rewinds into this rather than
    # ending the run: the instability is expected, and everything sampled before it is still
    # good data.
    safe: deque = deque(maxlen=args.safe_history)

    def sane(r, t_now: float, fmax: float) -> Defect | None:
        """Why this frame is unusable, or ``None`` if it is fine.

        The divergence checks are all ``negative_space=False``: a frame the model could not
        even evaluate says nothing about where the true surface is steep, only that the
        trajectory left the region the model can describe at all.
        """
        if not np.isfinite(r["energy_hartree"]):
            return Defect("energy is not finite", negative_space=False)
        if t_now > t_ceiling:
            return Defect(f"{t_now:.0f} K over the {t_ceiling:.0f} K ceiling",
                          negative_space=False)
        if fmax > args.max_force:
            return Defect(f"|F|max {fmax:.1f} over {args.max_force} eV/Angstrom",
                          negative_space=False)
        return geometry_defect(atoms, min_distance=args.min_distance,
                               max_oh=args.max_oh, max_oo=args.max_oo)

    def record() -> None:
        r = calc.results
        w = np.asarray(r["weights"])
        fmax = float(np.abs(r["forces"]).max())
        t_now = float(atoms.get_temperature())
        step = dyn.get_number_of_steps()
        ambiguity = float(1.0 - (w**2).sum())

        bad = sane(r, t_now, fmax)
        if bad is None:
            safe.append((atoms.get_positions().copy(), atoms.get_momenta().copy()))

        # Harvest first, judge second: a frame that passed `sane` is worth keeping even if the
        # very next step diverges, and writing it now means a blowup can never cost us the
        # sampling that preceded it.
        keep = (bad is None
                and ambiguity >= args.min_ambiguity
                and step - state["last_kept"] >= args.min_gap)
        if keep:
            snap = atoms.copy()
            snap.info.update(snapshot_info(r), charge=charge, step=step,
                             restart=state["restarts"])
            write(str(traj_path), snap, format="extxyz", append=True)
            state["harvested"] += 1
            state["last_kept"] = step

        # Negative space. A rejected frame is not waste -- it is a configuration the model
        # currently thinks is cheap and that the reference method will say is expensive, which
        # is exactly the label that stops the bias walking there again. Kept to a quota
        # relative to the accepted set, because these are abundant (35% of the first,
        # unguarded harvest) and a training set swamped by repulsive geometries teaches the
        # model where *not* to be at the expense of where to be.
        elif (bad is not None and bad.negative_space and args.negative_fraction > 0
              and step - state["last_negative"] >= args.min_gap
              and state["negative"] < args.negative_fraction * max(state["harvested"], 1)):
            snap = atoms.copy()
            snap.info.update(snapshot_info(r), charge=charge, step=step,
                             restart=state["restarts"], rejection=bad.reason)
            write(str(negative_path), snap, format="extxyz", append=True)
            state["negative"] += 1
            state["last_negative"] = step

        log["step"].append(step)
        log["time_fs"].append(step * args.timestep)
        log["temperature"].append(t_now)
        log["energy_hartree"].append(r["energy_hartree"])
        log["bias_energy"].append(r["bias_energy"])
        log["confine_energy"].append(r["confine_energy"])
        log["cv"].append(r["collective_variable"])
        log["ambiguity"].append(ambiguity)
        log["occupancy"].append(r["occupancy"])
        log["n_decompositions"].append(r["n_decompositions"])
        log["n_commits"].append(r["n_commits"])
        log["max_force"].append(fmax)
        log["harvested"].append(state["harvested"])
        log["negative"].append(state["negative"])

        if bad is not None:
            raise _Unstable(f"step {step}: {bad.reason}")
        if step % args.report_every == 0:
            rate = step / max(time.time() - state["t0"], 1e-9)
            print(f"  step {step:7d}  T {t_now:6.1f} K  A {ambiguity:.3f}  "
                  f"cv {r['collective_variable']:+6.2f}  M {r['n_decompositions']}  "
                  f"k {calc.bias.k:.4f}  kept {state['harvested']:4d}"f"+{state['negative']:<4d} "
                  f"restarts {state['restarts']:3d}  {rate:5.1f} steps/s", flush=True)

    def ramp() -> None:
        """Raise the restraint to full strength over `--ramp` steps."""
        calc.bias.k = args.k_bias * min(1.0, dyn.get_number_of_steps() / args.ramp)

    if args.ramp > 0:
        calc.bias.k = 0.0
        dyn.attach(ramp, interval=1)
    dyn.attach(record, interval=args.check_every)

    print(f"harvesting up to {args.target_frames} structures with ambiguity >= "
          f"{args.min_ambiguity}, budget {args.steps} steps at {args.timestep} fs "
          f"({args.steps * args.timestep / 1000:.1f} ps)", flush=True)

    # --- the harvest loop -----------------------------------------------------------------
    # The dynamics are expected to blow up; that is a property of the model on clusters this
    # far outside its training set, not something to be tuned away. So a blowup is not the end
    # of the run -- it rewinds to a sane state, redraws velocities, and keeps going. What the
    # run is judged on is how many usable structures it produced before the budget ran out.
    try:
        while (dyn.get_number_of_steps() < args.steps
               and state["harvested"] < args.target_frames
               and state["restarts"] <= args.max_restarts):
            try:
                dyn.run(args.steps - dyn.get_number_of_steps())
            except _Unstable as exc:
                state["restarts"] += 1
                if not safe:
                    print(f"  unstable at the very first step ({exc}); nothing to rewind to",
                          flush=True)
                    break
                # Rewind further the more often we fail, so a repeated dive into the same hole
                # backs out of it instead of re-entering from one step away.
                back = min(len(safe), 1 + state["restarts"] // args.patience)
                pos, _mom = safe[-back]
                for _ in range(back - 1):
                    if len(safe) > 1:
                        safe.pop()
                atoms.set_positions(pos)
                MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
                Stationary(atoms)
                calc.reset_base()
                print(f"  restart {state['restarts']}: {exc}; rewound {back} frame(s), "
                      f"{state['harvested']} kept so far", flush=True)
    except KeyboardInterrupt:
        print("interrupted", flush=True)

    np.savez(out / "diagnostics.npz", **{k: np.asarray(v) for k, v in log.items()})
    (out / "run.json").write_text(json.dumps(
        {**vars(args), "charge": charge, "n_atoms": len(atoms),
         "formula": atoms.get_chemical_formula(),
         "harvested": state["harvested"], "negative": state["negative"],
         "restarts": state["restarts"],
         "steps_run": dyn.get_number_of_steps()}, indent=2))

    A = np.asarray(log["ambiguity"]) if log["ambiguity"] else np.zeros(1)
    elapsed = time.time() - state["t0"]
    print(f"\n{state['harvested']} transition structures -> {traj_path}")
    if state["negative"]:
        print(f"{state['negative']} negative-space structures -> {negative_path}")
    print(f"  {dyn.get_number_of_steps()} steps, {state['restarts']} restarts, "
          f"{elapsed:.0f} s ({state['harvested'] / max(elapsed / 60, 1e-9):.1f} structures/min)")
    print(f"  ambiguity over all sampled steps: mean {A.mean():.4f}  max {A.max():.4f}  "
          f"fraction >= {args.min_ambiguity}: {float((A >= args.min_ambiguity).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
