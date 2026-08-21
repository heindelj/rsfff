"""Turn the ``qchem_roundtrip`` AIMD trajectories and their EDA labels into extxyz.

Usage
-----
    python scripts/parse_aimd_eda.py [--root qchem_roundtrip] [--out-dir data/wb97mv_tzvpd]
                                     [--stems h3o_w2 ...] [--limit N] [--strict]
                                     [--monomer-stride 10] [--allow-partial]

Two kinds of file come out, because two kinds of trajectory went in.

**Microsolvated ions** (``h3o_w1``, ``h3o_w2``, ``oh_w1``, ``oh_w2``). Every 50th
frame of these was harvested into ALMO-EDA jobs -- one per placement of the
excess charge, so two decompositions for the ``w1`` systems and three for the
``w2`` ones. Those frames are written with the multi-fragmentation schema of
:mod:`rsfff.qcgen.multifrag`: one frame carrying the geometry, the AIMD forces
and *all* of its fragmentations' EDA labels.

**Bare ions** (``h3o``, ``oh``). Nothing to decompose, so these are written at
``--monomer-stride`` with the ordinary single-fragmentation schema: geometry,
energy and the true one-body forces. This is the H3O+/OH- counterpart of
``data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz``.

How a frame is associated with its labels
-----------------------------------------
Not by geometry matching, and not by file name arithmetic. The harvester wrote a
marker JSON for every EDA input it generated, naming the AIMD output, the step,
the frame ordinal and -- the part that matters -- the ``assignment_signature``,
which is the fragment partition *in AIMD atom indices*. That is the exact
permutation needed to undo the fragment-grouped atom order of the EDA
``$molecule`` block, so it is read rather than reconstructed.

Everything that could still be wrong is then checked per frame:

* the permuted element list must match the trajectory's,
* the EDA job's CT-allowed SCF energy must match the AIMD step's SCF energy
  (these are the same wavefunction; measured agreement is ~3e-9 Hartree),
* every fragmentation of a frame must agree with every other on that energy and
  on the supersystem Mulliken charges, which are properties of the wavefunction
  and not of the partition,
* the rotation onto the canonical frame must be proper and its residual tiny
  (:func:`rsfff.qcgen.multifrag.rotation_between` raises otherwise),
* ``rsfff.qcgen.qchem_eda.check_consistency`` must be clean.

A frame missing any of its fragmentations is dropped unless ``--allow-partial``:
the point of the format is that the decompositions are alternatives to be
compared, and a frame that offers only some of them is a silent thumb on the
scale. One frame in this dataset is affected -- ``w2_OH-`` frame 3450, whose
``state00`` job died on a Q-Chem scratch-file error.

``--strict`` turns every warning into an abort instead of a drop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qcgen.multifrag import (  # noqa: E402
    Fragmentation,
    MultiFragFrame,
    TrajectoryFrame,
    canonical_basis,
    fragment_formula,
    recenter,
    rotate_multipole,
    rotate_second_moments,
    rotation_between,
    same_level_of_theory,
    write_frames,
)
from qcgen.qchem_aimd import parse_aimd_output  # noqa: E402
from qcgen.qchem_eda import (  # noqa: E402
    QChemEDAParseError,
    check_consistency as check_eda,
    parse_eda_output,
    to_atomic_units as eda_to_atomic_units,
    unique_components,
)
from qcgen.qchem_out import KJMOL_PER_HARTREE  # noqa: E402

#: The AIMD SCF and the EDA job's CT-allowed SCF are the same wavefunction run
#: twice, but from different guesses -- superposition of atomic densities during
#: the dynamics, the frozen-ALMO determinant in the EDA job. At
#: ``SCF_CONVERGENCE = 8`` they land on slightly different points of a flat
#: surface: measured over all 999 jobs the gap is 3.1e-7 Hartree median, 2.4e-6
#: worst, with no second population.
#:
#: This is a *frame identity* test, not an energy tolerance, and the scale that
#: matters is how far apart neighbouring frames are: consecutive harvested frames
#: differ by ~2.5e-3 Hartree. So this sits ~40x above the worst honest
#: disagreement and ~25x below the closest possible confusion.
ENERGY_ATOL = 1e-4

#: Two fragmentations of one frame, by contrast, are the *same* SCF from the same
#: kind of guess and agree to 1e-10 Hartree at worst (measured over 400 frames).
#: Nothing about a partition can move the supersystem wavefunction, so this one
#: can afford to be strict.
CROSS_FRAGMENTATION_ATOL = 1e-8
#: Mulliken charges are a property of the wavefunction, so two fragmentations of
#: one frame must agree on them. Measured spread over this data is 1e-6 to 4e-6,
#: which is Q-Chem's 6-decimal printing plus SCF noise; a pairing error would show
#: up in the first or second decimal, so this is loose against noise and still
#: three orders of magnitude tighter than any real effect.
MULLIKEN_ATOL = 1e-4

#: Each frozen fragment is its own isolated SCF, so its Mulliken charges sum to
#: its formal charge *exactly* -- measured worst case 1e-6, i.e. print precision.
#: This is the sharp version of the "are the fragment blocks paired with the right
#: fragments" question: swap two blocks and an H2O's 0 lands on an H3O+'s +1.
FRAGMENT_CHARGE_ATOL = 1e-4

#: How far the frozen fragment dipoles may sum away from the relaxed supersystem
#: dipole, relative to ``sum |mu_f|``. Q-Chem's neutral-water default of 0.5 is the
#: wrong regime here: these are ionic, proton-shared clusters where charge transfer
#: delocalizes a few tenths of an electron across an O-O contact and so moves of
#: order 1 e*a0 of dipole. Measured over this data the ratio runs 0.21-0.94 (and the
#: two charge placements of one frame differ by ~the O-O distance in bohr, which is
#: exactly "one electron moved across the dimer"), so 0.94 is physics, not a defect.
#: FRAGMENT_CHARGE_ATOL above is the check that actually catches mispairing.
DIPOLE_RTOL = 1.25


def slug(text: str) -> str:
    """``wB97M-V`` -> ``wb97mv``; ``def2-TZVPD`` -> ``tzvpd``. As in parse_roundtrip.py."""
    text = re.sub(r"^def2[-_]?", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def label_slug(text: str) -> str:
    """``w1_H3O+`` -> ``w1_h3o+``: the system part of a file name.

    Unlike :func:`slug` this keeps ``+``/``-``, because for this data the charge
    *is* the distinction between two systems and dropping it would collide
    ``OH-`` with ``OH``.
    """
    return re.sub(r"[^a-z0-9+_-]", "", text.lower())


def marker_index(root: str) -> dict[str, dict[int, list[dict]]]:
    """``{aimd stem: {frame ordinal: [marker, ...]}}`` from the harvester's state files.

    Each marker also gets ``job_dir`` and ``output`` filled in, so the caller
    never has to rebuild a path from the stem.
    """
    index: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    eda_root = os.path.join(root, "eda")
    if not os.path.isdir(eda_root):
        return index
    for label in sorted(os.listdir(eda_root)):
        gen_dir = os.path.join(eda_root, label, "state", "generated")
        if not os.path.isdir(gen_dir):
            continue
        for name in sorted(os.listdir(gen_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(gen_dir, name)) as fh:
                marker = json.load(fh)
            if marker.get("source_calculation") != "aimd":
                continue
            marker["job_dir"] = os.path.join(eda_root, label)
            marker["label"] = label
            marker["output"] = os.path.join(
                eda_root, label, "outputs", name[: -len(".json")] + ".out"
            )
            stem = os.path.splitext(os.path.basename(marker["source_output"]))[0]
            index[stem][int(marker["frame_ordinal"])].append(marker)
    for stem in index:
        for ordinal in index[stem]:
            index[stem][ordinal].sort(key=lambda m: int(m["rank"]))
    return index


def permutation_from_signature(signature: str, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """``(eda -> aimd index, fragment index per AIMD atom)`` from an assignment signature.

    The signature is ``"0,1,2,3|4,5,6|7,8,9"``: one group of *AIMD* atom indices
    per fragment, in fragment order, which is also the order the EDA
    ``$molecule`` block lists them. So position ``j`` of an EDA-ordered array
    belongs to AIMD atom ``perm[j]``.
    """
    groups = [[int(i) for i in g.split(",")] for g in signature.split("|")]
    perm = np.array([i for g in groups for i in g], dtype=int)
    if sorted(perm.tolist()) != list(range(n_atoms)):
        raise QChemEDAParseError(
            f"assignment signature {signature!r} is not a permutation of {n_atoms} atoms"
        )
    fragment_idx = np.empty(n_atoms, dtype=int)
    for f, group in enumerate(groups):
        fragment_idx[group] = f
    return perm, fragment_idx


def build_fragmentation(marker, step, positions, relative_to):
    """Parse one EDA output and express everything in the frame's canonical basis.

    ``positions`` is the recentered AIMD geometry; the returned arrays are all in
    its atom order and its orientation.
    """
    rec = eda_to_atomic_units(parse_eda_output(marker["output"]))
    n = len(step.symbols)
    if rec.n_atoms != n:
        raise QChemEDAParseError(
            f"EDA job has {rec.n_atoms} atoms, AIMD frame has {n}"
        )
    if not rec.has_fragment_blocks:
        raise QChemEDAParseError(
            "no isolated-fragment multipole blocks; the job needs SCF_PRINT_FRGM = true"
        )

    perm, fragment_idx = permutation_from_signature(marker["assignment_signature"], n)
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)          # AIMD atom -> position in the EDA arrays

    if [rec.symbols[j] for j in inv] != list(step.symbols):
        raise QChemEDAParseError(
            "the assignment signature does not map the EDA element list onto the AIMD one"
        )
    if list(rec.fragment_idx[inv]) != list(fragment_idx):
        raise QChemEDAParseError(
            "the EDA job's own fragment partition disagrees with the marker's signature"
        )
    if abs(rec.energy - step.energy) > ENERGY_ATOL:
        raise QChemEDAParseError(
            f"EDA CT-allowed energy {rec.energy:.10f} != AIMD step energy "
            f"{step.energy:.10f}; this frame is not that frame"
        )

    rot, rmsd = rotation_between(rec.positions[inv], positions)

    dipoles = np.array([m["dipole"] for m in rec.fragment_multipoles])
    seconds = np.array(
        [unique_components(m["quadrupole"], "quadrupole") for m in rec.fragment_multipoles]
    )
    frag = Fragmentation(
        fragment_idx=fragment_idx,
        fragment_charges=list(rec.fragment_charges),
        fragment_mults=list(rec.fragment_mults),
        fragment_energies=rec.fragment_energies,
        fragment_dipoles=dipoles @ rot.T,
        fragment_second_moments=rotate_second_moments(seconds, rot),
        fragment_mulliken=np.concatenate(rec.fragment_mulliken)[inv],
        eda=rec.eda,
        rank=int(marker["rank"]),
        charge_fragment=int(marker["charge_fragment"]),
        excess_distance=float(marker.get("excess_distance", 0.0)),
        source=os.path.relpath(marker["output"], relative_to),
    )
    for f, (formal, mull) in enumerate(zip(rec.fragment_charges, rec.fragment_mulliken)):
        if abs(float(mull.sum()) - formal) > FRAGMENT_CHARGE_ATOL:
            raise QChemEDAParseError(
                f"fragment {f}'s Mulliken charges sum to {float(mull.sum()):.6f} but its "
                f"formal charge is {formal}; the per-fragment blocks are mispaired"
            )

    msgs = [
        f"eda: {m}"
        for m in check_eda(
            rec, atol=1e-3 / KJMOL_PER_HARTREE, dipole_rtol=DIPOLE_RTOL
        )
    ]
    return frag, msgs, rec, rot, rmsd


def build_frame(
    step, markers, aimd_path, root, expected, require_complete=True
) -> tuple[MultiFragFrame, list[str], float]:
    """Assemble one AIMD frame and every EDA decomposition of it."""
    if require_complete and len(markers) != expected:
        raise QChemEDAParseError(f"{len(markers)} of {expected} fragmentations present")
    positions = recenter(step.symbols, step.positions)

    frags, msgs, records, rots, rmsds = [], [], [], [], []
    for marker in markers:
        frag, frag_msgs, rec, rot, rmsd = build_fragmentation(
            marker, step, positions, root
        )
        frags.append(frag)
        msgs += frag_msgs
        records.append(rec)
        rots.append(rot)
        rmsds.append(rmsd)

    # The supersystem is one wavefunction: every fragmentation must agree on it.
    ref, ref_rot = records[0], rots[0]
    for rec, rot in zip(records[1:], rots[1:]):
        if abs(rec.energy - ref.energy) > CROSS_FRAGMENTATION_ATOL:
            raise QChemEDAParseError(
                f"fragmentations disagree on the supersystem energy "
                f"({rec.energy:.10f} vs {ref.energy:.10f})"
            )
    n = len(step.symbols)
    mulliken = []
    for marker, rec in zip(markers, records):
        perm, _ = permutation_from_signature(marker["assignment_signature"], n)
        inv = np.empty(n, dtype=int)
        inv[perm] = np.arange(n)
        mulliken.append(rec.mulliken_charges[inv])
    spread = float(np.abs(np.array(mulliken) - mulliken[0]).max())
    if spread > MULLIKEN_ATOL:
        raise QChemEDAParseError(
            f"fragmentations disagree on the supersystem Mulliken charges by {spread:.3g}; "
            f"they are a property of the wavefunction, not of the partition"
        )

    multipoles = {
        name: rotate_multipole(tensor, ref_rot) for name, tensor in ref.multipoles.items()
    }
    methods = {(rec.method, rec.basis) for rec in records}
    if len(methods) != 1:
        raise QChemEDAParseError(f"fragmentations disagree on method/basis: {sorted(methods)}")
    method, basis = methods.pop()

    frame = MultiFragFrame(
        symbols=list(step.symbols),
        positions=positions,
        forces=step.forces,
        energy=step.energy,
        mulliken=mulliken[0],
        multipoles=multipoles,
        fragmentations=frags,
        total_charge=ref.total_charge,
        multiplicity=ref.multiplicity,
        method=method,
        basis=canonical_basis(basis),
        config_type=markers[0]["label"],
        sample_id=step.ordinal,
        aimd_step=step.step,
        source=os.path.relpath(aimd_path, root),
    )
    return frame, msgs, max(rmsds)


def write_cluster_stem(stem, aimd_path, by_ordinal, root, out_dir, args) -> str | None:
    """Every harvested frame of one microsolvated trajectory -> one extxyz."""
    expected = max(len(v) for v in by_ordinal.values())
    ordinals = sorted(by_ordinal)
    if args.limit:
        ordinals = ordinals[: args.limit]
    print(
        f"{stem}: {len(ordinals)} harvested frame(s), {expected} fragmentation(s) each",
        file=sys.stderr,
    )

    record = parse_aimd_output(aimd_path, ordinals=ordinals)
    steps = {s.ordinal: s for s in record.steps}
    aimd_level = (record.method, record.basis)

    frames, dropped, worst_rmsd = [], 0, 0.0
    for k, ordinal in enumerate(ordinals):
        markers = by_ordinal[ordinal]
        try:
            step = steps[ordinal]
        except KeyError:
            dropped += 1
            print(f"  dropped frame {ordinal}: not present in the AIMD output", file=sys.stderr)
            continue
        if step.step != int(markers[0]["step"]):
            raise SystemExit(
                f"{stem}: frame {ordinal} is AIMD step {step.step} but the harvest marker "
                f"says {markers[0]['step']}; the two are out of sync"
            )
        try:
            frame, msgs, rmsd = build_frame(
                step, markers, aimd_path, root, expected,
                require_complete=not args.allow_partial,
            )
        except (QChemEDAParseError, OSError, ValueError) as exc:
            if args.strict:
                raise SystemExit(f"{stem}: frame {ordinal}: {exc}")
            dropped += 1
            print(f"  dropped frame {ordinal}: {exc}", file=sys.stderr)
            continue
        if msgs:
            for m in msgs:
                print(f"  frame {ordinal}: {m}", file=sys.stderr)
            if args.strict:
                raise SystemExit(f"{stem}: frame {ordinal} failed a consistency check")
        # The forces come from the trajectory and the labels from the EDA jobs, so
        # they have to be the same level of theory or the frame is a chimera. The
        # comparison is case-insensitive because the two templates in
        # qchem_roundtrip/templates/ spell the basis differently.
        if not same_level_of_theory(aimd_level, (frame.method, frame.basis)):
            raise SystemExit(
                f"{stem}: the AIMD run is {aimd_level[0]}/{aimd_level[1]} but its EDA "
                f"labels are {frame.method}/{frame.basis}"
            )
        worst_rmsd = max(worst_rmsd, rmsd)
        frames.append(frame)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(ordinals)}", file=sys.stderr)

    if not frames:
        print(f"{stem}: nothing usable", file=sys.stderr)
        return None

    label = frames[0].config_type
    name = f"{label_slug(label)}_{slug(frames[0].method)}_{slug(frames[0].basis)}.xyz"
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    write_frames(path, frames)
    print(
        f"  -> {path}: {len(frames)} frame(s)"
        f"{f', {dropped} dropped' if dropped else ''}"
        f", worst alignment RMSD {worst_rmsd:.2e} Angstrom",
        file=sys.stderr,
    )
    return path


def write_monomer_stem(stem, aimd_path, root, out_dir, args) -> str | None:
    """A bare-ion trajectory -> one single-fragmentation extxyz at ``--monomer-stride``."""
    record = parse_aimd_output(aimd_path, stride=args.monomer_stride)
    if args.limit:
        record.steps = record.steps[: args.limit]
    print(
        f"{stem}: no EDA jobs; {len(record.steps)} of {record.n_steps} frame(s) at "
        f"stride {args.monomer_stride}",
        file=sys.stderr,
    )
    config_type = fragment_formula(record.steps[0].symbols, record.total_charge)
    basis = canonical_basis(record.basis)

    frames = [
        TrajectoryFrame(
            symbols=list(step.symbols),
            positions=recenter(step.symbols, step.positions),
            forces=step.forces,
            energy=step.energy,
            total_charge=record.total_charge,
            multiplicity=record.multiplicity,
            method=record.method,
            basis=basis,
            config_type=config_type,
            sample_id=step.ordinal,
            aimd_step=step.step,
            source=os.path.relpath(aimd_path, root),
        )
        for step in record.steps
    ]
    unconverged = [s.step for s in record.steps if not s.converged]
    if unconverged:
        msg = f"{stem}: SCF unconverged at step(s) {unconverged[:5]}"
        if args.strict:
            raise SystemExit(msg)
        print(f"  {msg}", file=sys.stderr)

    name = f"{label_slug(config_type)}_{slug(record.method)}_{slug(basis)}.xyz"
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    write_frames(path, frames)
    print(f"  -> {path}: {len(frames)} frame(s)", file=sys.stderr)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse qchem_roundtrip AIMD trajectories and their EDA labels into extxyz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default="qchem_roundtrip", help="the round-trip bundle")
    ap.add_argument("--out-dir", default="data/wb97mv_tzvpd", help="where to write extxyz")
    ap.add_argument("--stems", nargs="*", default=None, help="AIMD stems (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames per stem")
    ap.add_argument(
        "--monomer-stride", type=int, default=10,
        help="keep every Nth frame of a trajectory that has no EDA jobs (default: 10)",
    )
    ap.add_argument(
        "--allow-partial", action="store_true",
        help="keep frames that are missing some of their fragmentations",
    )
    ap.add_argument("--strict", action="store_true", help="abort on any warning")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.root))
    aimd_dir = os.path.join(root, "aimd", "outputs")
    if not os.path.isdir(aimd_dir):
        print(f"error: no AIMD outputs under {aimd_dir}", file=sys.stderr)
        return 1

    index = marker_index(root)
    stems = args.stems
    if stems is None:
        stems = sorted(
            os.path.splitext(n)[0] for n in os.listdir(aimd_dir) if n.endswith(".out")
        )

    written = []
    for stem in stems:
        aimd_path = os.path.join(aimd_dir, f"{stem}.out")
        if not os.path.isfile(aimd_path):
            print(f"{stem}: no AIMD output, skipping", file=sys.stderr)
            continue
        by_ordinal = index.get(stem)
        if by_ordinal:
            written.append(
                write_cluster_stem(stem, aimd_path, by_ordinal, root, args.out_dir, args)
            )
        else:
            written.append(write_monomer_stem(stem, aimd_path, root, args.out_dir, args))

    return 0 if any(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
