#!/usr/bin/env python3
"""Sanity-check and repair benchmark EDA fragment assignments.

The check marks an assigned fragment as suspicious when any atom in that
fragment has no same-fragment neighbor within the chosen cutoff. Repairs use the
same O/H rule as the AIMD EDA harvesting path: hydrogens are assigned to oxygen
fragments by the minimum total O-H distance subject to each fragment's charge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from benchmark_fragment_tools import (
    Atom,
    build_fragmented_molecule,
    build_plain_molecule,
    fragment_sanity_issues,
    minimized_oh_groups,
    ordered_atoms_from_groups,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUNDTRIP = REPO_ROOT / "qchem_roundtrip"


@dataclass
class QChemInput:
    path: Path
    charge: int
    multiplicity: int
    fragments: list[list[Atom]]
    fragment_charges: list[int]
    fragment_multiplicities: list[int]
    rem_block: str

    @property
    def symbols(self) -> list[str]:
        return [symbol for fragment in self.fragments for symbol, _ in fragment]

    @property
    def coords(self) -> list[tuple[float, float, float]]:
        return [coord for fragment in self.fragments for _, coord in fragment]

    @property
    def atoms(self) -> list[Atom]:
        return [atom for fragment in self.fragments for atom in fragment]


def parse_qchem_input(path: Path) -> QChemInput:
    lines = path.read_text().splitlines()
    try:
        molecule_start = lines.index("$molecule")
        molecule_end = lines.index("$end", molecule_start + 1)
        rem_start = lines.index("$rem", molecule_end + 1)
    except ValueError as exc:
        raise ValueError(f"{path}: expected $molecule, $end, and $rem blocks") from exc

    charge_mult = lines[molecule_start + 1].split()
    if len(charge_mult) != 2:
        raise ValueError(f"{path}: molecule charge/multiplicity line must contain two integers")
    charge, multiplicity = (int(charge_mult[0]), int(charge_mult[1]))

    fragments: list[list[Atom]] = []
    fragment_charges: list[int] = []
    fragment_mults: list[int] = []
    current: list[Atom] = []
    in_fragmented_section = False
    idx = molecule_start + 2
    while idx < molecule_end:
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        if line == "--":
            if current:
                fragments.append(current)
                current = []
            in_fragmented_section = True
            if idx >= molecule_end:
                raise ValueError(f"{path}: missing fragment charge/multiplicity after --")
            frag_charge_mult = lines[idx].split()
            idx += 1
            if len(frag_charge_mult) != 2:
                raise ValueError(f"{path}: fragment charge/multiplicity line must contain two integers")
            fragment_charges.append(int(frag_charge_mult[0]))
            fragment_mults.append(int(frag_charge_mult[1]))
            continue
        toks = line.split()
        if len(toks) < 4:
            raise ValueError(f"{path}: malformed atom line {line!r}")
        current.append((toks[0], (float(toks[1]), float(toks[2]), float(toks[3]))))
    if current:
        fragments.append(current)
    if not fragments:
        raise ValueError(f"{path}: no atoms found")
    if not in_fragmented_section:
        fragment_charges = [charge]
        fragment_mults = [multiplicity]
    if len(fragment_charges) != len(fragments):
        raise ValueError(f"{path}: fragment metadata count does not match fragment count")

    return QChemInput(
        path=path,
        charge=charge,
        multiplicity=multiplicity,
        fragments=fragments,
        fragment_charges=fragment_charges,
        fragment_multiplicities=fragment_mults,
        rem_block="\n".join(lines[rem_start:]) + "\n",
    )


def write_if_changed(path: Path, text: str) -> bool:
    old = path.read_text() if path.exists() else None
    if old == text:
        return False
    path.write_text(text)
    return True


def state_paths_for(root: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for state_dir in (root / "state").glob("*"):
        if state_dir.is_dir():
            paths.extend(path for path in state_dir.glob(f"{stem}*") if path.is_file())
    return paths


def repair_stem(
    eda_input: QChemInput,
    force_input_path: Path,
    *,
    delete_outputs: bool,
    fix: bool,
) -> tuple[bool, list[Path]]:
    groups = minimized_oh_groups(eda_input.symbols, eda_input.coords, eda_input.fragment_charges)
    eda_text = (
        build_fragmented_molecule(
            eda_input.symbols,
            eda_input.coords,
            groups,
            eda_input.fragment_charges,
            eda_input.fragment_multiplicities,
            eda_input.charge,
            eda_input.multiplicity,
        )
        + "\n\n"
        + eda_input.rem_block
    )

    force_input = parse_qchem_input(force_input_path)
    force_groups = minimized_oh_groups(force_input.symbols, force_input.coords, eda_input.fragment_charges)
    force_text = (
        build_plain_molecule(
            ordered_atoms_from_groups(force_input.symbols, force_input.coords, force_groups),
            force_input.charge,
            force_input.multiplicity,
        )
        + "\n\n"
        + force_input.rem_block
    )

    changed = False
    deleted: list[Path] = []
    if fix:
        changed = write_if_changed(eda_input.path, eda_text) or changed
        changed = write_if_changed(force_input_path, force_text) or changed
        if delete_outputs:
            for output_path in (
                eda_input.path.parents[1] / "outputs" / f"{eda_input.path.stem}.out",
                force_input_path.parents[1] / "outputs" / f"{force_input_path.stem}.out",
            ):
                if output_path.exists():
                    output_path.unlink()
                    deleted.append(output_path)
            for state_path in state_paths_for(eda_input.path.parents[1], eda_input.path.stem):
                state_path.unlink()
                deleted.append(state_path)
            for state_path in state_paths_for(force_input_path.parents[1], force_input_path.stem):
                state_path.unlink()
                deleted.append(state_path)
    return changed, deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roundtrip-root", type=Path, default=DEFAULT_ROUNDTRIP)
    parser.add_argument("--cutoff", type=float, default=1.5)
    parser.add_argument("--fix", action="store_true", help="rewrite bad inputs")
    parser.add_argument("--delete-outputs", action="store_true", help="delete matching EDA and force outputs while fixing")
    parser.add_argument("--only-with-output", action="store_true", help="only consider EDA inputs with an output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eda_root = args.roundtrip_root / "eda" / "benchmark_eda"
    force_root = args.roundtrip_root / "force" / "benchmark_force"
    bad_count = 0
    repaired_count = 0
    deleted_count = 0
    for input_path in sorted((eda_root / "inputs").glob("*.in")):
        if args.only_with_output and not (eda_root / "outputs" / f"{input_path.stem}.out").exists():
            continue
        eda_input = parse_qchem_input(input_path)
        issues = fragment_sanity_issues(eda_input.fragments, args.cutoff)
        if not issues:
            continue
        bad_count += 1
        max_nearest = max(issue.nearest_distance for issue in issues)
        print(
            f"BAD {input_path.name}: {len(issues)} atom(s) lack a same-fragment neighbor "
            f"within {args.cutoff:.3f} A; worst nearest={max_nearest:.3f} A"
        )
        force_input_path = force_root / "inputs" / input_path.name
        if not force_input_path.exists():
            raise FileNotFoundError(f"missing matching force input for {input_path.name}: {force_input_path}")
        changed, deleted = repair_stem(
            eda_input,
            force_input_path,
            delete_outputs=args.delete_outputs,
            fix=args.fix,
        )
        if changed:
            repaired_count += 1
        deleted_count += len(deleted)
        for path in deleted:
            print(f"  deleted {path}")

    action = "repaired" if args.fix else "would repair"
    print(f"{bad_count} bad EDA input(s); {action} {repaired_count}; deleted {deleted_count} file(s)")


if __name__ == "__main__":
    main()
