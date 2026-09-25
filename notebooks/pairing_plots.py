"""Diagnostics for the pairing model against the film model and the Q-Chem scans.

The code behind ``notebooks/pairing_plots.ipynb``; the notebook only sets knobs and calls
into here (``%autoreload`` picks up edits). Three questions, in order:

1. **The scans.** ``qchem_roundtrip/pairing_scans`` holds rigid one-coordinate scans
   (``scripts/pairing_scans.py``): O-H stretches of H2O / H3O+ / OH-, an H-O-H bend, and the
   shared proton of H5O2+ / H3O2- walking between the oxygens. Every model is evaluated on
   exactly the frames Q-Chem labeled, so the curves are comparable point for point.
2. **The one-body decomposition.** The film model's fragment energy is
   ``E0 + Morse + angle + gated intra classical``; the pairing model's is
   ``E0(q) + E_pair + (1 - c) classical + c Pauli``. Both are pulled apart along the scans.
3. **The monomer AIMD sets.** Fragment-energy parity of both models on the thermal
   monomer frames (H2O, H3O+, OH-), the same one-body data the film was fitted to.

Models are loaded through :func:`rsfff.md.film_driver.load_film_model` (it dispatches on
``film.model`` in the checkpoint's config, so film and pairing checkpoints load the same
way), or built at their priors from a config when no checkpoint exists yet -- which is the
state of the pairing model at the time of writing, and is worth looking at: the priors were
calibrated to put the water O-H at the pyCMM well, and everything else is the physics of
the functional before any fit.

Fragmentations
--------------
A scan frame is one fragment for the pairing model (the total charge on it; the valence
prior then gives each oxygen of a Zundel 2.5). The film model needs a covalent topology, so
the proton-transfer scans are given to it twice, with the shared proton assigned to either
oxygen (``A`` / ``B``): its two diabats, which cross where the pairing model has one curve.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
for p in (ROOT / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ase.io import read as ase_read  # noqa: E402

from rsfff.ff.units import BOHR_ANG, KJMOL_PER_HARTREE  # noqa: E402
from rsfff.train.data import Batch, load_extxyz  # noqa: E402

__all__ = [
    "SCANS", "Scan", "ScanResult", "load_scan", "load_scans", "load_model",
    "build_prior_pairing_model", "evaluate_scan", "evaluate_all", "plot_stretches",
    "plot_bend", "plot_proton_transfer", "plot_bond_orders", "monomer_parity",
    "plot_monomer_parity", "savefig",
]

#: name -> (geometry stem, x label, kind). The stems are what scripts/pairing_scans.py wrote.
SCANS = {
    "h2o_stretch": ("pairing_scan_h2o_stretch", "r(O-H) ($\\AA$)", "stretch"),
    "h3o+_stretch": ("pairing_scan_h3o+_stretch", "r(O-H) ($\\AA$)", "stretch"),
    "oh-_stretch": ("pairing_scan_oh-_stretch", "r(O-H) ($\\AA$)", "stretch"),
    "h2o_bend": ("pairing_scan_h2o_bend", "H-O-H (deg)", "bend"),
    "h5o2+_pt": ("pairing_scan_h5o2+_pt", "proton offset from the O-O midpoint ($\\AA$)", "pt"),
    "h3o2-_pt": ("pairing_scan_h3o2-_pt", "proton offset from the O-O midpoint ($\\AA$)", "pt"),
}
SCAN_ROOT = ROOT / "qchem_roundtrip" / "pairing_scans"

MODEL_COLORS = {"qm": "#3c4450", "film": "#276ef1", "film:A": "#276ef1", "film:B": "#7a4cc2",
                "pairing": "#d64545", "pairing (priors)": "#e07a3f"}
TERM_COLORS = {"pairing": "#d64545", "pauli": "#e07a3f", "elst": "#276ef1", "disp": "#2a9d8f",
               "bonded": "#276ef1", "intra": "#e07a3f", "cross": "#999999"}


# ---------------------------------------------------------------------------
# the scans


@dataclass
class Scan:
    name: str
    kind: str
    xlabel: str
    coord: np.ndarray                 # (n,) the scanned coordinate (from the geometry header)
    group: np.ndarray                 # (n,) e.g. the O-O distance of a proton scan; 0 otherwise
    frames: list                      # ase.Atoms, in the geometry-file orientation
    charge: int
    energy: np.ndarray                # (n,) Q-Chem total energy, Hartree; nan where missing
    converged: np.ndarray             # (n,) bool
    info: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frames)

    def groups(self):
        return [g for g in np.unique(self.group)]


def load_scan(name: str, root: Path = SCAN_ROOT) -> Scan:
    """One scan: its geometry frames and whatever outputs have come back for them."""
    from rsfff.qcgen.qchem_force import parse_force_output

    stem, xlabel, kind = SCANS[name]
    frames = ase_read(root / "geoms" / f"{stem}.xyz", index=":")
    n = len(frames)
    energy = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    for k in range(n):
        out = root / "outputs" / f"{stem}_frame{k:04d}.out"
        if not out.exists():
            continue
        try:
            rec = parse_force_output(str(out))
        except Exception as exc:                          # noqa: BLE001 -- report, keep going
            print(f"{out.name}: {exc}")
            continue
        if rec.completed and rec.converged:
            energy[k] = rec.energy
            converged[k] = True
    coord = np.array([float(a.info.get("coord", np.nan)) for a in frames])
    group = np.array([float(a.info.get("r_oo", 0.0)) for a in frames])
    charge = int(frames[0].info.get("charge", 0))
    missing = int((~converged).sum())
    if missing:
        print(f"{name}: {missing}/{n} frames without a converged output")
    return Scan(name, kind, xlabel, coord, group, frames, charge, energy, converged,
                info=[dict(a.info) for a in frames])


def load_scans(names=None, root: Path = SCAN_ROOT) -> dict[str, Scan]:
    return {name: load_scan(name, root) for name in (names or SCANS)}


# ---------------------------------------------------------------------------
# models


def load_model(path, *, device: str = "cpu"):
    """A film or pairing checkpoint (``load_film_model`` dispatches on ``film.model``)."""
    from rsfff.md.film_driver import load_film_model

    model, cfg = load_film_model(path, device=device)
    return model, cfg


def build_prior_pairing_model(config_path=ROOT / "configs" / "water_pairing.yaml",
                              neighbor_types=(1, 8), *, atomic_states: bool = True):
    """The pairing model at its priors: no fit, only the calibrated functional.

    ``atomic_states`` turns on the charged atomic reference (``data.atomic_reference_states``
    of the config; defaults to the wB97M-V/def2-TZVPD file when the config does not set it),
    which is what makes the ion scans comparable to the neutral one.
    """
    from rsfff.mlip.reference_states import AtomicStateReference
    from rsfff.train.build_pairing import build_pairing_model
    from rsfff.train.config import load_config
    from rsfff.train.data import load_reference_energies

    cfg = load_config(str(config_path))
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)
    types = sorted(int(z) for z in neighbor_types)
    ref = load_reference_energies(str(ROOT / cfg.data.reference_energies), types).to(
        torch.get_default_dtype()
    )
    states = None
    if atomic_states:
        states_path = cfg.data.atomic_reference_states or "data/atomic_reference_states_wb97mv_tzvpd.json"
        states = AtomicStateReference.from_json(
            str(ROOT / states_path), types, dtype=torch.get_default_dtype()
        )
    torch.manual_seed(0)
    model = build_pairing_model(cfg.features, cfg.film, types, ref, states)
    return model.eval(), cfg


def is_pairing(model) -> bool:
    return hasattr(model, "network") and hasattr(model.network, "pairing_heads")


# ---------------------------------------------------------------------------
# batches from scan frames


def _fragmentation(atoms, how: str):
    """``(fragment_idx (N,), fragment_charges)`` for a scan frame.

    ``single``: everything is one fragment with the total charge. ``A`` / ``B``: the two
    proton-transfer diabats -- the last atom of the frame is the shared proton, assigned to
    the first (``A``) or second (``B``) oxygen; the fragment with the proton carries the
    charge for the cation, the fragment without the proton for the anion.
    """
    z = atoms.get_atomic_numbers()
    q_tot = int(atoms.info.get("charge", 0))
    n = len(z)
    if how == "single":
        return np.zeros(n, dtype=int), [q_tot]
    oxygens = np.flatnonzero(z == 8)
    if len(oxygens) != 2:
        raise ValueError(f"fragmentation {how!r} needs two oxygens, got {len(oxygens)}")
    # the halves as scripts/pairing_scans.py wrote them: [O1, H..., O2, H..., H_shared]
    frag = np.zeros(n, dtype=int)
    frag[oxygens[1]:n - 1] = 1
    frag[n - 1] = 0 if how == "A" else 1
    if q_tot > 0:
        charges = [q_tot, 0] if how == "A" else [0, q_tot]
    else:
        charges = [0, q_tot] if how == "A" else [q_tot, 0]
    return frag, charges


def scan_batch(frames, how: str = "single", dtype=None) -> tuple[Batch, np.ndarray, np.ndarray]:
    """One :class:`Batch` holding every frame of a scan, atoms reordered so fragments are
    contiguous within each frame.

    Returns the batch, the batch->frame atom permutation (per frame, as one flat array of
    frame-local indices) and the frame's first atom offset in the batch.
    """
    dtype = dtype or torch.get_default_dtype()
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    pos, z, bidx, fidx, f2b, charges, orders, offsets = [], [], [], [], [], [], [], []
    n_frag_total = 0
    n_atoms_total = 0
    for k, atoms in enumerate(frames):
        frag, q = _fragmentation(atoms, how)
        order = np.argsort(frag, kind="stable")
        pos.append(atoms.get_positions()[order])
        z.append(atoms.get_atomic_numbers()[order])
        bidx.append(np.full(len(order), k))
        fidx.append(frag[order] + n_frag_total)
        f2b.extend([k] * len(q))
        charges.extend(q)
        orders.append(order)
        offsets.append(n_atoms_total)
        n_frag_total += len(q)
        n_atoms_total += len(order)
    batch = Batch(
        positions=torch.tensor(np.concatenate(pos), dtype=dtype),
        atomic_numbers=torch.tensor(np.concatenate(z)),
        batch_idx=torch.tensor(np.concatenate(bidx)),
        n_systems=len(frames),
        energy=torch.zeros(len(frames), dtype=dtype),
        fragment_idx=torch.tensor(np.concatenate(fidx)),
        fragment_charge=torch.tensor(charges, dtype=dtype),
        fragment_two_s=torch.zeros(n_frag_total, dtype=dtype),
        fragment_to_batch=torch.tensor(f2b),
        n_fragments=n_frag_total,
    )
    return batch, np.concatenate(orders), np.array(offsets)


# ---------------------------------------------------------------------------
# evaluation


@dataclass
class ScanResult:
    """One model on one scan, everything in Hartree; ``terms`` are the one-body pieces."""

    label: str
    energy: np.ndarray                         # (n,) total model energy
    terms: dict[str, np.ndarray]               # name -> (n,)
    bond_order: dict[str, np.ndarray]          # pair label -> (n,) (pairing model only)
    comembership: dict[str, np.ndarray]        # pair label -> (n,)
    unpaired: np.ndarray | None = None         # (n, N) per-atom u
    formal_charge: np.ndarray | None = None    # (n, N) per-atom q^f from the electronic-state solve


def _pair_label(z, i, j):
    sym = {1: "H", 8: "O"}
    return f"{sym.get(int(z[i]), z[i])}{i}-{sym.get(int(z[j]), z[j])}{j}"


def _watch_pairs(atoms, kind: str):
    """Which atom pairs (frame indices) to track along the scan."""
    z = atoms.get_atomic_numbers()
    n = len(z)
    if kind == "stretch":
        return [(0, 1)]
    if kind == "bend":
        return [(0, 1), (1, 2)]
    oxygens = [int(i) for i in np.flatnonzero(z == 8)]
    return [(oxygens[0], n - 1), (oxygens[1], n - 1)]


def evaluate_scan(model, scan: Scan, *, how: str = "single", label: str | None = None,
                  chunk: int = 64) -> ScanResult:
    """Run one model over every frame of a scan (in chunks of ``chunk`` frames per batch).
    Missing Q-Chem labels do not matter here."""
    label = label or ("pairing" if is_pairing(model) else "film")
    n = scan.n
    pairing = is_pairing(model)
    z = scan.frames[0].get_atomic_numbers()
    n_at = len(z)
    pairs = _watch_pairs(scan.frames[0], scan.kind)
    pair_labels = [_pair_label(z, i, j) for i, j in pairs]
    energy = np.zeros(n)
    terms: dict[str, np.ndarray] = {}
    bond_order = {k: np.full(n, np.nan) for k in pair_labels}
    comember = {k: np.full(n, np.nan) for k in pair_labels}
    unpaired = np.full((n, n_at), np.nan)
    formal_charge = np.full((n, n_at), np.nan)

    def pool(x, index, m):
        return x.new_zeros(m).index_add_(0, index, x).numpy()

    for start in range(0, n, chunk):
        frames = scan.frames[start:start + chunk]
        m = len(frames)
        batch, order, offsets = scan_batch(frames, how)
        with torch.no_grad():
            out = model(batch)
        f2b = batch.fragment_to_batch
        energy[start:start + m] = out.energy.numpy()
        t = {
            "reference": pool(out.energy_ref, f2b, m),
            "one-body": pool(out.fragment_energy, f2b, m),
            "interaction": sum(v.numpy() for v in out.interaction.values()),
            "intra classical": pool(out.energy_intra, f2b, m),
            "induction": out.interaction["induction"].numpy() if "induction" in out.interaction else np.zeros(m),
        }
        if pairing:
            t["pairing"] = pool(out.energy_pairing, f2b, m)
            t["cross"] = out.interaction["cross"].numpy()
        else:
            t["bonded"] = pool(out.energy_bonded, f2b, m)
        for name, value in t.items():
            terms.setdefault(name, np.zeros(n))[start:start + m] = value
        pi = out.pair_index.numpy()
        n_tot = int(batch.positions.shape[0])
        keys = pi[0] * n_tot + pi[1]
        sub_index = out.sub_index.numpy() if pairing else None
        p_intra = out.p_intra.numpy()
        p_bo = out.bond_order.numpy() if pairing else None
        for k in range(m):
            off = offsets[k]
            inv = np.argsort(order[off:off + n_at])          # frame atom -> batch-local atom
            if pairing:
                unpaired[start + k, order[off:off + n_at]] = out.unpaired[off:off + n_at].numpy()
                formal_charge[start + k, order[off:off + n_at]] = out.formal_charge[off:off + n_at].numpy()
            for (i, j), name in zip(pairs, pair_labels):
                a, b = sorted((int(inv[i]) + off, int(inv[j]) + off))
                hit = np.flatnonzero(keys == a * n_tot + b)
                if hit.size == 0:
                    continue
                h = int(hit[0])
                comember[name][start + k] = float(p_intra[h])
                if pairing:
                    sub = np.flatnonzero(sub_index == h)
                    if sub.size:
                        bond_order[name][start + k] = float(p_bo[int(sub[0])])
    return ScanResult(
        label, energy, terms, bond_order if pairing else {}, comember,
        unpaired if pairing else None, formal_charge if pairing else None,
    )


def evaluate_all(models: dict, scans: dict[str, Scan]) -> dict[str, dict[str, ScanResult]]:
    """``{scan: {label: result}}``. Film models get both diabats on the proton scans."""
    results: dict[str, dict[str, ScanResult]] = {}
    for name, scan in scans.items():
        results[name] = {}
        for label, model in models.items():
            if scan.kind == "pt" and not is_pairing(model):
                for how in ("A", "B"):
                    results[name][f"{label}:{how}"] = evaluate_scan(model, scan, how=how, label=f"{label}:{how}")
            else:
                results[name][label] = evaluate_scan(model, scan, label=label)
    return results


# ---------------------------------------------------------------------------
# plots


def _color(label):
    base = label.split(":")[0]
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    return MODEL_COLORS.get(base, None)


def _rel(y, ref_index):
    return (y - y[ref_index]) * KJMOL_PER_HARTREE


def _ref_index(scan: Scan):
    """The frame every curve is referenced to: the Q-Chem minimum where labels exist."""
    if np.isfinite(scan.energy).any():
        return int(np.nanargmin(scan.energy))
    return 0


def savefig(fig, name, figdir=ROOT / "notebooks" / "figures", prefix="pairing"):
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{prefix}_{name}.{ext}", dpi=160, bbox_inches="tight")


def plot_stretches(scans, results, names=("h2o_stretch", "h3o+_stretch", "oh-_stretch"),
                   x_max=None, path=None):
    """Rows: one stretch each. Columns: relative energy, model - QM error, the one-body pieces."""
    names = [n for n in names if n in scans]
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 3.9 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        scan = scans[name]
        k0 = _ref_index(scan)
        x = scan.coord
        sel = np.ones_like(x, dtype=bool) if x_max is None else x <= x_max
        ax = row[0]
        if np.isfinite(scan.energy).any():
            ax.plot(x[sel], _rel(scan.energy, k0)[sel], "o", ms=3.5, color=MODEL_COLORS["qm"],
                    label="Q-Chem")
        for label, res in results[name].items():
            ax.plot(x[sel], _rel(res.energy, k0)[sel], "-", color=_color(label), label=label)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_ylabel(f"{name}\nenergy relative to the Q-Chem minimum (kJ/mol)")
        ax.set_xlabel(scan.xlabel)
        ax.legend(fontsize=8)

        ax = row[1]
        for label, res in results[name].items():
            if np.isfinite(scan.energy).any():
                err = (res.energy - scan.energy) * KJMOL_PER_HARTREE
                err = err - np.nanmean(err[sel])
                ax.plot(x[sel], err[sel], "-", color=_color(label), label=label)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_ylabel("model - Q-Chem, mean removed (kJ/mol)")
        ax.set_xlabel(scan.xlabel)
        ax.legend(fontsize=8)

        ax = row[2]
        for label, res in results[name].items():
            keys = [k for k in ("pairing", "bonded", "intra classical", "induction") if k in res.terms]
            for key in keys:
                y = res.terms[key] - res.terms[key][k0]
                ls = "-" if label.startswith("pairing") else "--"
                ax.plot(x[sel], (y * KJMOL_PER_HARTREE)[sel], ls, color=TERM_COLORS.get(key, None),
                        label=f"{label}: {key}")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_ylabel("one-body pieces, relative (kJ/mol)")
        ax.set_xlabel(scan.xlabel)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig


def plot_bend(scans, results, name="h2o_bend", path=None):
    scan = scans[name]
    k0 = _ref_index(scan)
    x = scan.coord
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    if np.isfinite(scan.energy).any():
        ax.plot(x, _rel(scan.energy, k0), "o", ms=3.5, color=MODEL_COLORS["qm"], label="Q-Chem")
    for label, res in results[name].items():
        ax.plot(x, _rel(res.energy, k0), "-", color=_color(label), label=label)
    ax.set_xlabel(scan.xlabel)
    ax.set_ylabel("energy relative to the Q-Chem minimum (kJ/mol)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for label, res in results[name].items():
        for key in ("pairing", "bonded", "intra classical"):
            if key in res.terms:
                y = (res.terms[key] - res.terms[key][k0]) * KJMOL_PER_HARTREE
                ax.plot(x, y, "-" if label.startswith("pairing") else "--",
                        color=TERM_COLORS.get(key), label=f"{label}: {key}")
    ax.set_xlabel(scan.xlabel)
    ax.set_ylabel("one-body pieces, relative (kJ/mol)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig


def plot_proton_transfer(scans, results, name="h5o2+_pt", path=None):
    """One panel per model (plus Q-Chem), one curve per O-O distance. Every curve is
    referenced to its own midpoint frame, so the panels compare *shapes* (a barrier or a
    well at the midpoint); the legend carries each model's absolute offset from Q-Chem at
    that midpoint, which is the constant the shape does not show."""
    scan = scans[name]
    labels = ["qm"] + list(results[name])
    groups = scan.groups()
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.0), sharey=True, squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        for g, color in zip(groups, cmap):
            sel = scan.group == g
            x = scan.coord[sel]
            mid = int(np.argmin(np.abs(x)))
            y = scan.energy[sel] if label == "qm" else results[name][label].energy[sel]
            text = f"O-O {g:.2f} $\\AA$"
            if label != "qm" and np.isfinite(scan.energy[sel][mid]):
                off = (y[mid] - scan.energy[sel][mid]) * KJMOL_PER_HARTREE
                text += f" (offset {off:+.0f})"
            ax.plot(x, (y - y[mid]) * KJMOL_PER_HARTREE, "o-" if label == "qm" else "-", ms=3,
                    color=color, label=text)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_title("Q-Chem" if label == "qm" else label)
        ax.set_xlabel(scan.xlabel)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(f"{name}: energy relative to each curve's midpoint (kJ/mol)")
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig


def plot_bond_orders(scans, results, names=None, path=None):
    """Bond order and co-membership of the watched pairs along each scan (pairing model)."""
    names = [n for n in (names or SCANS) if n in scans]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        scan = scans[name]
        drawn = False
        for label, res in results[name].items():
            if not res.bond_order:
                continue
            if scan.kind == "pt":
                for g in scan.groups():
                    sel = scan.group == g
                    for (pair, p), ls in zip(res.bond_order.items(), ("-", "--")):
                        ax.plot(scan.coord[sel], p[sel], ls, lw=1.2, label=f"{pair} (O-O {g:.2f})")
            else:
                for pair, p in res.bond_order.items():
                    ax.plot(scan.coord, p, "-", label=f"{label}: p {pair}")
                for pair, c in res.comembership.items():
                    ax.plot(scan.coord, c, ":", label=f"{label}: c {pair}")
                if res.unpaired is not None:
                    ax.plot(scan.coord, res.unpaired.sum(1), "-.", color="#3c4450", label="total unpaired")
            drawn = True
        ax.set_title(name)
        ax.set_xlabel(scan.xlabel)
        ax.set_ylim(-0.05, 1.05 if scan.kind != "stretch" else 2.1)
        if drawn:
            ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("bond order / co-membership")
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig


def plot_formal_charges(scans, results, names=None, path=None):
    """Formal charges of the oxygens (and the moving proton on the PT scans) along each scan."""
    names = [n for n in (names or SCANS) if n in scans]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        scan = scans[name]
        z = scan.frames[0].get_atomic_numbers()
        watch = [i for i, zi in enumerate(z) if zi == 8]
        if scan.kind == "pt":
            watch.append(len(z) - 1)                          # the shared proton is last
        elif scan.kind == "stretch":
            watch.append(1)                                   # the stretched hydrogen
        drawn = False
        for label, res in results[name].items():
            if res.formal_charge is None:
                continue
            groups = scan.groups() if scan.kind == "pt" else [None]
            for g in groups:
                sel = np.ones(scan.n, bool) if g is None else scan.group == g
                tag = "" if g is None else f" (O-O {g:.2f})"
                for i, ls in zip(watch, ("-", "--", "-.", ":")):
                    ax.plot(scan.coord[sel], res.formal_charge[sel, i], ls, lw=1.2,
                            label=f"{label}: q {'H' if z[i] == 1 else 'O'}{i}{tag}")
            drawn = True
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.set_title(name)
        ax.set_xlabel(scan.xlabel)
        if drawn:
            ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("formal charge $q^f$")
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig


# ---------------------------------------------------------------------------
# the monomer AIMD sets


MONOMER_SETS = {
    "H2O": "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz",
    "H3O+": "data/wb97mv_tzvpd/h3o+_wb97mv_tzvpd.xyz",
    "OH-": "data/wb97mv_tzvpd/oh-_wb97mv_tzvpd.xyz",
}


def monomer_parity(models: dict, sets=MONOMER_SETS, *, n_frames: int = 200, batch_size: int = 50):
    """``{species: {label: (ref, pred)}}`` fragment energies in Hartree on the AIMD monomers."""
    out: dict[str, dict[str, tuple]] = {}
    for species, rel in sets.items():
        path = ROOT / rel
        if not path.exists():
            print(f"{species}: {rel} not found; skipping")
            continue
        ds = load_extxyz(str(path), dtype=torch.get_default_dtype())
        idx = list(range(min(n_frames, len(ds))))
        out[species] = {}
        for label, model in models.items():
            ref, pred = [], []
            for start in range(0, len(idx), batch_size):
                batch = ds.flat_batch(idx[start:start + batch_size])
                with torch.no_grad():
                    o = model(batch, with_induction=False)
                ref.append(batch.fragment_energy.numpy())
                pred.append(o.fragment_energy.numpy())
            out[species][label] = (np.concatenate(ref), np.concatenate(pred))
    return out


def plot_monomer_parity(parity, path=None):
    species = list(parity)
    fig, axes = plt.subplots(1, len(species), figsize=(4.4 * len(species), 4.0), squeeze=False)
    for ax, sp in zip(axes.ravel(), species):
        for label, (ref, pred) in parity[sp].items():
            r = (ref - ref.mean()) * KJMOL_PER_HARTREE
            p = (pred - ref.mean()) * KJMOL_PER_HARTREE
            bias = float(np.mean(p - r))
            mae = float(np.mean(np.abs(p - r - bias)))
            ax.plot(r, p - bias, ".", ms=4, alpha=0.7, color=_color(label),
                    label=f"{label}: bias {bias:.1f}, MAE {mae:.1f} kJ/mol")
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "-", color="#bbbbbb", lw=0.8)
        ax.set_title(f"{sp} monomers ({len(ref)} frames)")
        ax.set_xlabel("Q-Chem fragment energy, centred (kJ/mol)")
        ax.set_ylabel("model, bias removed (kJ/mol)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        savefig(fig, path)
    return fig
