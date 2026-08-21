"""Shared helpers for water-cluster benchmark scripts.

The benchmark structures in ``benchmarks/structures`` are plain XYZ files ordered as
water monomers (``O H H`` repeated).  The unified force-field model needs a fragment
assignment and an intrafragment charge-flow graph, so this module builds those pieces
directly from that ordering and wraps the model as an ASE calculator.
"""

from __future__ import annotations

import json
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if importlib.util.find_spec("rsfff") is None:
    src_root = REPO_ROOT / "src"
    spec = importlib.util.spec_from_file_location(
        "rsfff", src_root / "__init__.py", submodule_search_locations=[str(src_root)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load rsfff package from {src_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rsfff"] = module
    spec.loader.exec_module(module)

import torch  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402

from rsfff.ff.v1 import load_v1_checkpoint  # noqa: E402
from rsfff.train.data import Batch  # noqa: E402
from rsfff.train.loss import compute_forces  # noqa: E402
from rsfff.train.train_eem import resolve_device  # noqa: E402


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_KJ_MOL = 2625.4996394799
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "water_staged" / "best.pt"


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    config_path: Path
    checkpoint_path: Path
    config_stage: str
    neighbor_types: tuple[int, ...]


def default_results_dir() -> Path:
    return REPO_ROOT / "benchmarks" / "results"


def structure_paths(paths: Iterable[str | Path] | None = None) -> list[Path]:
    if paths:
        out = [Path(p) for p in paths]
    else:
        out = sorted((REPO_ROOT / "benchmarks" / "structures").glob("*.xyz"))
    if not out:
        raise FileNotFoundError("no benchmark XYZ structures found")
    return out


def load_water_model(
    config_path: str | Path | None = None,
    *,
    stage: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    device: str | None = None,
) -> ModelBundle:
    """The benchmark model, rebuilt from a checkpoint alone.

    This goes through :mod:`rsfff.ff.v1`, the frozen copy of the unified pair model, because
    ``checkpoints/water_staged/best.pt`` is a v1 checkpoint and the live tree has moved to the
    fragment-expert architecture of ``docs/fff_v2.md``. See that package's docstring.

    ``config_path`` and ``stage`` are accepted and recorded so the CLI flags and the result JSON
    schema are unchanged, but they no longer *select* anything: a checkpoint embeds the full
    config it was trained under, which is both more robust and the only thing that still makes
    sense now that ``configs/water_staged.yaml`` lives under ``configs/archive/``. Passing a
    config that disagrees with the checkpoint used to silently build a different model; now it
    cannot.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if checkpoint_root is not None and checkpoint_path is None:
        root = Path(checkpoint_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        ckpt = root / DEFAULT_CHECKPOINT.parent.name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    dev = resolve_device(device or cfg.device, cfg.dtype)

    model, cfg, neighbor_types = load_v1_checkpoint(ckpt, device=dev, dtype=dtype)
    return ModelBundle(
        model=model,
        device=dev,
        dtype=dtype,
        config_path=Path(config_path) if config_path is not None else ckpt,
        checkpoint_path=ckpt,
        config_stage=stage or "archived-v1",
        neighbor_types=neighbor_types,
    )


def infer_ordered_water_fragments(symbols: list[str]) -> np.ndarray:
    if len(symbols) % 3:
        raise ValueError("water-cluster XYZ must contain a multiple of 3 atoms")
    frag = np.empty(len(symbols), dtype=np.int64)
    for i in range(0, len(symbols), 3):
        triplet = symbols[i : i + 3]
        if triplet != ["O", "H", "H"]:
            raise ValueError(
                "expected ordered water monomers as O H H repeated; "
                f"found {triplet!r} at atoms {i}-{i + 2}"
            )
        frag[i : i + 3] = i // 3
    return frag


def water_bond_graph(n_atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed O-H channels for each ordered water monomer."""
    edges: list[tuple[int, int]] = []
    for i in range(0, n_atoms, 3):
        edges.extend(((i, i + 1), (i + 1, i), (i, i + 2), (i + 2, i)))
    if edges:
        bond_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    else:
        bond_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    bond_batch = torch.zeros(bond_index.shape[1], dtype=torch.long, device=device)
    return bond_index, bond_batch


def atoms_to_batch(atoms, bundle: ModelBundle, *, requires_grad: bool) -> Batch:
    symbols = atoms.get_chemical_symbols()
    frag_np = infer_ordered_water_fragments(symbols)
    n_frag = int(frag_np.max()) + 1 if frag_np.size else 0
    pos = torch.tensor(atoms.get_positions(), dtype=bundle.dtype, device=bundle.device)
    pos.requires_grad_(requires_grad)
    nums = torch.tensor(
        [atomic_numbers[s] for s in symbols], dtype=torch.long, device=bundle.device
    )
    frag = torch.tensor(frag_np, dtype=torch.long, device=bundle.device)
    bond_index, bond_batch = water_bond_graph(len(atoms), bundle.device)
    return Batch(
        positions=pos,
        atomic_numbers=nums,
        batch_idx=torch.zeros(len(atoms), dtype=torch.long, device=bundle.device),
        n_systems=1,
        energy=torch.zeros(1, dtype=bundle.dtype, device=bundle.device),
        total_charge=torch.zeros(1, dtype=bundle.dtype, device=bundle.device),
        fragment_idx=frag,
        fragment_charge=torch.zeros(n_frag, dtype=bundle.dtype, device=bundle.device),
        fragment_two_s=torch.zeros(n_frag, dtype=bundle.dtype, device=bundle.device),
        fragment_to_batch=torch.zeros(n_frag, dtype=torch.long, device=bundle.device),
        n_fragments=n_frag,
        bond_index=bond_index,
        bond_batch=bond_batch,
    )


class RSFFFCalculator(Calculator):
    """ASE calculator for the trained water force field.

    ASE expects energies in eV and forces in eV/Angstrom.  The model returns Hartree
    and Hartree/Angstrom.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, bundle: ModelBundle):
        super().__init__()
        self.bundle = bundle

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        batch = atoms_to_batch(self.atoms, self.bundle, requires_grad=True)
        with torch.enable_grad():
            energy_h = self.bundle.model(batch).energy
            forces_h = compute_forces(energy_h, batch.positions, create_graph=False)
        self.results["energy"] = float(energy_h.detach().cpu()[0]) * HARTREE_TO_EV
        self.results["forces"] = forces_h.detach().cpu().numpy() * HARTREE_TO_EV


def model_energy_hartree(atoms, bundle: ModelBundle) -> float:
    batch = atoms_to_batch(atoms, bundle, requires_grad=False)
    with torch.no_grad():
        return float(bundle.model(batch).energy.detach().cpu()[0])


def model_report(atoms, bundle: ModelBundle, *, with_forces: bool = False) -> dict:
    """Evaluate total, binding, EDA channels, and optionally forces.

    Binding energy here is the model interaction energy at the supplied geometry:
    ``E_total - sum(fragment_energy)``.  For the unified model this is also the sum of
    the inter-fragment EDA channels in ``out.interaction``.
    """
    batch = atoms_to_batch(atoms, bundle, requires_grad=with_forces)
    ctx = torch.enable_grad() if with_forces else torch.no_grad()
    with ctx:
        out = bundle.model(batch)
        forces_h = (
            compute_forces(out.energy, batch.positions, create_graph=False)
            if with_forces
            else None
        )

    total_h = float(out.energy.detach().cpu()[0])
    fragment_sum_h = float(out.fragment_energy.detach().sum().cpu())
    eda_kj = {
        name: float(value.detach().cpu()[0]) * HARTREE_TO_KJ_MOL
        for name, value in sorted(out.interaction.items())
    }
    binding_h = total_h - fragment_sum_h
    report = {
        "total_energy_hartree": total_h,
        "fragment_energy_sum_hartree": fragment_sum_h,
        "binding_energy_kcal_mol": binding_h * HARTREE_TO_KCAL_MOL,
        "eda_kj_mol": eda_kj,
    }
    if forces_h is not None:
        forces = forces_h.detach().cpu().numpy() * HARTREE_TO_EV
        norms = np.linalg.norm(forces, axis=1)
        report.update(
            {
                "forces_ev_per_angstrom": forces,
                "max_force_ev_per_angstrom": float(norms.max()) if norms.size else 0.0,
            }
        )
    return report


def attach_singlepoint_results(atoms, report: dict) -> None:
    """Attach total and binding energies to an extxyz-friendly ASE object."""
    atoms.info["rsfff_total_energy_hartree"] = float(report["total_energy_hartree"])
    atoms.info["rsfff_binding_energy_kcal_mol"] = float(
        report["binding_energy_kcal_mol"]
    )
    for name, value in report["eda_kj_mol"].items():
        atoms.info[f"rsfff_eda_{name}_kj_mol"] = float(value)
    forces = report.get("forces_ev_per_angstrom")
    if forces is not None:
        atoms.calc = SinglePointCalculator(
            atoms,
            forces=np.asarray(forces, dtype=float),
        )


def reference_energy_hartree(atoms) -> float | None:
    """Extract the MP2/AVTZ frame energy from the benchmark XYZ comment."""
    for key in ("energy", "E"):
        if key in atoms.info:
            try:
                return float(atoms.info[key])
            except (TypeError, ValueError):
                pass
    for key, value in atoms.info.items():
        if value is True:
            try:
                return float(key)
            except (TypeError, ValueError):
                pass
    return None


def mp2_binding_energy_kcal_mol(
    atoms, monomer_energy_hartree: float
) -> float | None:
    energy = reference_energy_hartree(atoms)
    if energy is None:
        return None
    return (energy - (len(atoms) // 3) * monomer_energy_hartree) * HARTREE_TO_KCAL_MOL


def rmsd_kabsch(reference, candidate) -> float:
    """Mass-unweighted all-atom RMSD after optimal rotation and translation."""
    a = np.asarray(reference.get_positions(), dtype=float)
    b = np.asarray(candidate.get_positions(), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"RMSD shape mismatch: {a.shape} vs {b.shape}")
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    cov = b0.T @ a0
    u, _, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rot = u @ correction @ vt
    diff = b0 @ rot - a0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def oh_bond_lengths(atoms) -> list[dict[str, float | int]]:
    positions = atoms.get_positions()
    bonds = []
    for oxygen in range(0, len(atoms), 3):
        for hydrogen in (oxygen + 1, oxygen + 2):
            bonds.append(
                {
                    "oxygen_index": oxygen,
                    "hydrogen_index": hydrogen,
                    "length_angstrom": float(
                        np.linalg.norm(positions[oxygen] - positions[hydrogen])
                    ),
                }
            )
    return bonds


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
