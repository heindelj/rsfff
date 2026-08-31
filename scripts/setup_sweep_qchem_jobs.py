#!/usr/bin/env python3
"""Stage force and EDA jobs for a curated sweep, with a reaction-centred fragmentation.

    python scripts/setup_sweep_qchem_jobs.py                       # everything
    python scripts/setup_sweep_qchem_jobs.py --stride 10 --dry-run # count first

Four job sets, in three nested job dirs that the existing worker discovers on its own:

    force/transition_structures/     one per clean frame
    force/negative_space/            one per rejected frame
    eda/transition_structures/       three per clean frame (see below)

Why the EDA fragmentation is not the usual one
----------------------------------------------
Everywhere else in this corpus a cluster is split into **monomers** -- one fragment per water,
one per ion -- and every competing assignment of the shared proton is a separate job. That is
the right decomposition for fitting one-body and pair terms, and the wrong one for asking what
a proton transfer costs: with five monomers the EDA channels are sums over ten pairs, and the
one pair the transfer actually happens in is buried in them.

So here the cluster is split at the reaction instead. The two molecules sharing the proton are
kept as separate fragments and **everything else becomes a single fragment**:

    trimer, split A   [donor + shared H] [acceptor]        [environment]
    trimer, split B   [donor]            [shared H + acceptor]  [environment]

Those two are the same geometry read as the two protonation states, and the difference between
their EDA channels is the transfer, measured directly rather than inferred. The environment
appears once in each, identically, so it subtracts out.

    dimer             [donor + shared H + acceptor]        [environment]

and the dimer says what the *whole* reactive complex costs to solvate -- the quantity the
trimer pair cannot give, because in both of its splits the complex is already cut in two.

All three share **one atom ordering**: donor atoms, then the shared proton, then acceptor
atoms, then the environment. Only the ``--`` boundaries move. That is what makes the three
outputs comparable term by term; ordering each job by its own fragments would reshuffle the
atoms between them and leave every comparison to a permutation.

Charges follow the proton. The environment is always neutral -- the ion is by construction one
of the two molecules sharing the proton -- and this is asserted rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch                                                              # noqa: E402
from ase.io import read                                                   # noqa: E402

from rsfff.md import enumerate_group                                      # noqa: E402
from setup_ion_cluster_jobs import atom_line, ensure_job_dir, rem_block   # noqa: E402


@dataclass(frozen=True)
class Reaction:
    """One frame, cut at the proton that is moving.

    ``order`` is the atom permutation every job for this frame uses: donor, shared proton,
    acceptor, environment. ``n_donor``/``n_acceptor`` count the *own* atoms of each molecule,
    excluding the shared proton, so the fragment boundaries are arithmetic on those.
    """

    order: np.ndarray            # (N,) permutation into [donor | H* | acceptor | env]
    n_donor: int
    n_acceptor: int
    n_env: int
    q_donor_keeps: int           # charge of the donor side when it keeps the proton
    q_acceptor_keeps: int        # charge of the acceptor side when it takes the proton
    total_charge: int
    separation: float            # |r(H*-O_donor) - r(H*-O_acceptor)|, how shared it is


def find_reaction(atoms, charge: int) -> Reaction | None:
    """Locate the transferring proton and the two molecules it is shared between.

    The candidate chosen is the one whose moved proton is most nearly *equidistant* from its
    two hosts -- the most shared, hence the most transition-like. Picking the highest-weighted
    candidate instead would make the cut depend on the mediator, and these jobs exist to
    measure something the mediator should be fitted against rather than something it decided.
    """
    pos = np.asarray(atoms.get_positions())
    z = np.asarray(atoms.get_atomic_numbers())
    group = enumerate_group(torch.as_tensor(pos, dtype=torch.float64),
                            torch.as_tensor(z), charge)
    frags = group.fragments.numpy()
    if frags.shape[0] < 2:
        return None
    base = frags[0]
    oxygen_of = {int(base[o]): int(o) for o in np.flatnonzero(z == 8)}

    best = None
    for m in range(1, frags.shape[0]):
        moved = np.flatnonzero(frags[m] != base)
        if moved.size != 1:
            continue                                  # concerted; not a single transfer
        h = int(moved[0])
        donor_f, acceptor_f = int(base[h]), int(frags[m][h])
        if donor_f not in oxygen_of or acceptor_f not in oxygen_of:
            continue
        r_d = float(np.linalg.norm(pos[h] - pos[oxygen_of[donor_f]]))
        r_a = float(np.linalg.norm(pos[h] - pos[oxygen_of[acceptor_f]]))
        gap = abs(r_d - r_a)
        if best is None or gap < best[0]:
            best = (gap, h, donor_f, acceptor_f, m)
    if best is None:
        return None
    gap, h, donor_f, acceptor_f, m = best

    donor = [i for i in np.flatnonzero(base == donor_f) if i != h]
    acceptor = [i for i in np.flatnonzero(base == acceptor_f) if i != h]
    env = [i for i in range(len(z)) if base[i] not in (donor_f, acceptor_f)]

    q = group.atom_charge.numpy()
    q_donor_keeps = int(round(float(q[0][donor[0]])))          # decomposition 0: donor has H*
    q_acceptor_keeps = int(round(float(q[m][acceptor[0]])))    # decomposition m: acceptor has H*
    q_env = sum({int(base[i]): int(round(float(q[0][i]))) for i in env}.values())
    if q_env != 0:
        # The ion is by construction one of the two hosts, so the rest must be neutral water.
        # If it is not, the frame is something this fragmentation cannot describe.
        return None

    return Reaction(
        order=np.array(donor + [h] + acceptor + env, dtype=int),
        n_donor=len(donor), n_acceptor=len(acceptor), n_env=len(env),
        q_donor_keeps=q_donor_keeps, q_acceptor_keeps=q_acceptor_keeps,
        total_charge=int(charge), separation=gap,
    )


def molecule_block(symbols, coords, total_charge: int, blocks) -> str:
    """``$molecule`` with explicit ``--`` separated fragments.

    ``blocks`` is a list of ``(charge, [row indices])`` in the order they are written. Rows are
    already in the canonical order, so the indices are contiguous ranges.
    """
    lines = ["$molecule", f"{total_charge} 1"]
    for q, rows in blocks:
        lines += ["--", f"{q} 1"]
        lines += [atom_line(symbols[i], coords[i]) for i in rows]
    lines.append("$end")
    return "\n".join(lines)


def plain_block(symbols, coords, total_charge: int) -> str:
    return "\n".join(["$molecule", f"{total_charge} 1"]
                     + [atom_line(s, c) for s, c in zip(symbols, coords)] + ["$end"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transition", default="data/cluster_sweep/transition_structures.xyz")
    p.add_argument("--negative", default="data/cluster_sweep/negative_space.xyz")
    p.add_argument("--roundtrip-root", default=str(ROOT / "qchem_roundtrip"))
    p.add_argument("--stride", type=int, default=1,
                   help="take every Nth frame. The sweep is autocorrelated, so a stride is the "
                        "cheapest way to cut the job count without losing coverage.")
    p.add_argument("--method", default="wB97M-V")
    p.add_argument("--basis", default="def2-TZVPD")
    p.add_argument("--mem-total", type=int, default=64000)
    p.add_argument("--mem-static", type=int, default=10000)
    p.add_argument("--dry-run", action="store_true", help="count the jobs and write nothing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    rt = Path(args.roundtrip_root)
    dirs = {
        "force_ts": rt / "force" / "transition_structures",
        "force_neg": rt / "force" / "negative_space",
        "eda_ts": rt / "eda" / "transition_structures",
    }
    if not args.dry_run:
        for d in dirs.values():
            ensure_job_dir(d)

    force_rem = rem_block("force", args.method, args.basis, args.mem_total, args.mem_static)
    eda_rem = rem_block("eda", args.method, args.basis, args.mem_total, args.mem_static)
    counts, skipped = Counter(), Counter()
    manifest = []

    def stem_for(atoms, index: int) -> str:
        src = str(atoms.info.get("source", f"frame{index:06d}")).replace(":", "_")
        return f"{src}_s{int(atoms.info.get('step', 0)):05d}"

    # ---- force jobs, both sets ----------------------------------------------------------
    for key, path in (("force_ts", args.transition), ("force_neg", args.negative)):
        if not Path(path).exists():
            print(f"  {path} missing, skipping", flush=True)
            continue
        for i, atoms in enumerate(read(path, index=f"::{args.stride}")):
            q = int(atoms.info.get("charge", 0))
            text = plain_block(atoms.get_chemical_symbols(),
                               np.asarray(atoms.get_positions()), q) + "\n\n" + force_rem + "\n"
            counts[key] += 1
            if not args.dry_run:
                out = dirs[key] / "inputs" / f"{stem_for(atoms, i)}.in"
                if args.overwrite or not out.exists():
                    out.write_text(text)

    # ---- EDA, transition structures only ------------------------------------------------
    if Path(args.transition).exists():
        for i, atoms in enumerate(read(args.transition, index=f"::{args.stride}")):
            q = int(atoms.info.get("charge", 0))
            rx = find_reaction(atoms, q)
            if rx is None:
                skipped["no single transferring proton"] += 1
                continue
            if rx.n_env == 0:
                # H5O2+ and H3O2-: the reactive complex *is* the whole cluster, so there is no
                # environment to separate and neither the trimer nor the dimer means anything.
                skipped["no environment (the complex is the whole cluster)"] += 1
                continue

            sym = [atoms.get_chemical_symbols()[j] for j in rx.order]
            xyz = np.asarray(atoms.get_positions())[rx.order]
            nd, na, ne = rx.n_donor, rx.n_acceptor, rx.n_env
            donor = list(range(nd))
            shared = [nd]
            acceptor = list(range(nd + 1, nd + 1 + na))
            env = list(range(nd + 1 + na, nd + 1 + na + ne))

            jobs = {
                "pairA": [(rx.q_donor_keeps, donor + shared),
                          (rx.total_charge - rx.q_donor_keeps, acceptor), (0, env)],
                "pairB": [(rx.total_charge - rx.q_acceptor_keeps, donor),
                          (rx.q_acceptor_keeps, shared + acceptor), (0, env)],
                "dimer": [(rx.total_charge, donor + shared + acceptor), (0, env)],
            }
            stem = stem_for(atoms, i)
            for tag, blocks in jobs.items():
                assert sum(qq for qq, _ in blocks) == rx.total_charge, (stem, tag)
                assert sum(len(r) for _, r in blocks) == len(sym), (stem, tag)
                counts[f"eda_{tag}"] += 1
                if not args.dry_run:
                    out = dirs["eda_ts"] / "inputs" / f"{stem}_{tag}.in"
                    if args.overwrite or not out.exists():
                        out.write_text(
                            molecule_block(sym, xyz, rx.total_charge, blocks)
                            + "\n\n" + eda_rem + "\n")
            manifest.append(dict(stem=stem, charge=rx.total_charge, n_atoms=len(sym),
                                 n_donor=nd, n_acceptor=na, n_env=ne,
                                 separation=round(rx.separation, 4),
                                 source=str(atoms.info.get("source", ""))))

    print(f"{'(dry run) ' if args.dry_run else ''}job counts, stride {args.stride}:")
    for k in ("force_ts", "force_neg", "eda_pairA", "eda_pairB", "eda_dimer"):
        print(f"  {k:12s} {counts[k]:7d}")
    print(f"  {'TOTAL':12s} {sum(counts.values()):7d}")
    if skipped:
        print("skipped for EDA:")
        for reason, n in skipped.most_common():
            print(f"  {n:7d}  {reason}")
    if not args.dry_run:
        (rt / "sweep_jobs_manifest.json").write_text(json.dumps(
            {"method": args.method, "basis": args.basis,
             "counts": dict(counts), "skipped": dict(skipped),
             "jobs": manifest}, indent=2))
        print(f"\nmanifest -> {rt / 'sweep_jobs_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
