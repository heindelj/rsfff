"""Interpretability diagnostics for a trained film model.

    python scripts/film_diagnostics.py <checkpoint.pt> --mode sweep|jacobian|dimer|lambda

Writes PNGs and CSVs next to the checkpoint under ``diagnostics/``. The point of every mode is
the same: relate changes in feature space to changes in geometry, which the film design makes
possible because the network's outputs are *named physical parameters* rather than an energy.

sweep     Distort a single water along r_OH1, r_OH2 and theta_HOH; plot every bonded parameter
          and the permanent multipoles against the distortion, with the pyCMM constants as
          reference lines. Flat curves are the geometry-independence regularizer working;
          structure is the model saying the parameter genuinely varies (e.g. charge flux shows
          up here as dq/dr without ever being written as a term).
jacobian  d(theta)/d(internal coordinate) at equilibrium by autograd, as a heatmap -- the
          compact answer to "which geometric mode moves which parameter".
dimer     O-O separation scan of a water dimer: every env shift, the gate, and the per-channel
          energies -- verifies the smooth decay to the exact isolated limit.
lambda    A continuous state sweep C(lambda) between the two fragment relabelings of a dimer:
          total energy, force norm, and parameter movement along the path (fff_film.md §8.2E).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rsfff.ff.film import StateDescriptor
from rsfff.ff.film.bonded import DEFAULT_ANGLE_PRIOR, DEFAULT_BOND_PRIOR
from rsfff.ff.units import BOHR_ANG
from rsfff.train.build_film import build_film_model
from rsfff.train.data import Batch, load_reference_energies


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    neighbor_types = ckpt["neighbor_types"]
    reference = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(torch.get_default_dtype())
    model = build_film_model(config.features, config.film, neighbor_types, reference)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def water_batch(r1: float, r2: float, theta: float) -> Batch:
    """One water with the given O-H distances (Angstrom) and H-O-H angle (rad)."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            [r2 * math.cos(theta), r2 * math.sin(theta), 0.0],
        ],
        dtype=torch.get_default_dtype(),
    )
    return Batch(
        positions=positions,
        atomic_numbers=torch.tensor([8, 1, 1]),
        batch_idx=torch.zeros(3, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.zeros(3, dtype=torch.long),
        fragment_charge=torch.zeros(1),
        fragment_two_s=torch.zeros(1),
        fragment_to_batch=torch.zeros(1, dtype=torch.long),
        n_fragments=1,
    )


def dimer_batch(r_oo: float) -> Batch:
    """A near-linear hydrogen-bonded dimer at the given O-O distance (Angstrom)."""
    donor = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2392, 0.9281, 0.0]]
    )
    acceptor = torch.tensor(
        [[r_oo, 0.0, 0.0], [r_oo + 0.24, 0.93, 0.0], [r_oo + 0.24, -0.93, 0.0]]
    )
    positions = torch.cat((donor, acceptor)).to(torch.get_default_dtype())
    return Batch(
        positions=positions,
        atomic_numbers=torch.tensor([8, 1, 1, 8, 1, 1]),
        batch_idx=torch.zeros(6, dtype=torch.long),
        n_systems=1,
        energy=torch.zeros(1),
        fragment_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
        fragment_charge=torch.zeros(2),
        fragment_two_s=torch.zeros(2),
        fragment_to_batch=torch.zeros(2, dtype=torch.long),
        n_fragments=2,
    )


def bonded_scalars(out) -> dict[str, float]:
    b = out.parameters.bonded0
    return {
        "r_eq_1": float(b.r_eq[0]), "r_eq_2": float(b.r_eq[1]),
        "D_1": float(b.d[0]), "D_2": float(b.d[1]),
        "k_1": float(b.k[0]), "k_2": float(b.k[1]),
        "theta_eq": float(torch.arccos(b.cos_theta_eq[0])),
        "k_theta": float(b.k_theta[0]),
        "q_O": float(out.charges[0]),
        "q_H1": float(out.charges[1]),
        "mu_norm_O": float(out.mu[0].norm()) if out.mu is not None else 0.0,
    }


#: The pyCMM water constants, for reference lines.
_PYCMM = {
    "r_eq_1": DEFAULT_BOND_PRIOR[(1, 8)][0], "r_eq_2": DEFAULT_BOND_PRIOR[(1, 8)][0],
    "D_1": DEFAULT_BOND_PRIOR[(1, 8)][1], "D_2": DEFAULT_BOND_PRIOR[(1, 8)][1],
    "k_1": DEFAULT_BOND_PRIOR[(1, 8)][2], "k_2": DEFAULT_BOND_PRIOR[(1, 8)][2],
    "theta_eq": DEFAULT_ANGLE_PRIOR[8][0], "k_theta": DEFAULT_ANGLE_PRIOR[8][1],
}


def _plot_grid(rows, xlabel, x, out_png, title):
    names = [k for k in rows[0] if k != xlabel]
    n = len(names)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    for k, name in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        ax.plot(x, [row[name] for row in rows], lw=1.5)
        if name in _PYCMM:
            ax.axhline(_PYCMM[name], color="gray", ls="--", lw=1, label="pyCMM")
            ax.legend(fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def _write_csv(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


def mode_sweep(model, outdir: Path) -> None:
    r0 = DEFAULT_BOND_PRIOR[(1, 8)][0] * BOHR_ANG
    t0 = DEFAULT_ANGLE_PRIOR[8][0]
    sweeps = {
        "r_OH1": [(r, r0, t0) for r in torch.linspace(0.75, 1.6, 35).tolist()],
        "theta_HOH": [(r0, r0, t) for t in torch.linspace(1.2, 2.6, 35).tolist()],
    }
    with torch.no_grad():
        for label, grid in sweeps.items():
            rows = []
            xs = []
            for (r1, r2, theta) in grid:
                out = model(water_batch(r1, r2, theta), with_induction=False)
                x = r1 if label == "r_OH1" else theta
                xs.append(x)
                rows.append({label: x, "E_frag": float(out.fragment_energy[0]),
                             **bonded_scalars(out)})
            _write_csv(rows, outdir / f"sweep_{label}.csv")
            _plot_grid(rows, label, xs, outdir / f"sweep_{label}.png",
                       f"parameters vs {label} (isolated water)")


def mode_jacobian(model, outdir: Path) -> None:
    r0 = DEFAULT_BOND_PRIOR[(1, 8)][0] * BOHR_ANG
    t0 = DEFAULT_ANGLE_PRIOR[8][0]
    batch = water_batch(r0, r0, t0)
    pos = batch.positions.clone().requires_grad_(True)
    batch = Batch(**{**batch.__dict__, "positions": pos})
    out = model(batch, with_induction=False)
    b = out.parameters.bonded0

    # Internal-coordinate B-matrix rows at this geometry (unit displacement directions).
    with torch.no_grad():
        v1 = pos[1] - pos[0]
        v2 = pos[2] - pos[0]
        r1, r2 = v1.norm(), v2.norm()
        e1, e2 = v1 / r1, v2 / r2
        # d(theta)/dR rows for the angle
        cos_t = (e1 * e2).sum()
        sin_t = (1 - cos_t**2).sqrt()
        dt_d1 = (e2 - cos_t * e1) / (r1 * sin_t)
        dt_d2 = (e1 - cos_t * e2) / (r2 * sin_t)

    quantities = {
        "r_eq_1": b.r_eq[0], "D_1": b.d[0], "k_1": b.k[0],
        "cos_theta_eq": b.cos_theta_eq[0], "k_theta": b.k_theta[0],
        "q_O": out.charges[0], "q_H1": out.charges[1],
    }
    rows = []
    for name, q in quantities.items():
        (g,) = torch.autograd.grad(q, pos, retain_graph=True)
        rows.append({
            "parameter": name,
            "d/dr_OH1": float((g[1] * e1).sum() - (g[0] * e1).sum()) / 2.0,
            "d/dr_OH2": float((g[2] * e2).sum() - (g[0] * e2).sum()) / 2.0,
            "d/dtheta": float((g[1] * dt_d1).sum() + (g[2] * dt_d2).sum()),
        })
    _write_csv(rows, outdir / "jacobian.csv")

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 1.5))
    data = torch.tensor([[r["d/dr_OH1"], r["d/dr_OH2"], r["d/dtheta"]] for r in rows])
    scale = data.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    im = ax.imshow(data / scale, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["dr_OH1", "dr_OH2", "dtheta"])
    ax.set_yticks(range(len(rows)), [r["parameter"] for r in rows])
    for i in range(len(rows)):
        for j, key in enumerate(("d/dr_OH1", "d/dr_OH2", "d/dtheta")):
            ax.text(j, i, f"{rows[i][key]:.2e}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="row-normalized")
    ax.set_title("d(parameter) / d(internal coordinate) at equilibrium")
    fig.tight_layout()
    fig.savefig(outdir / "jacobian.png", dpi=150)
    plt.close(fig)
    print(f"wrote {outdir / 'jacobian.png'}")


def mode_dimer(model, outdir: Path) -> None:
    rows = []
    xs = torch.linspace(2.4, 9.0, 45).tolist()
    with torch.no_grad():
        for r_oo in xs:
            out = model(dimer_batch(r_oo))
            row = {
                "r_OO": r_oo,
                "E_int_kJmol": float(sum(out.interaction.values()) * 2625.5),
                "elst": float(out.interaction["elst"] * 2625.5),
                "pauli": float(out.interaction["pauli"] * 2625.5),
                "disp": float(out.interaction["disp"] * 2625.5),
                "induction": float(out.interaction["induction"] * 2625.5),
                "a_env_max": float(out.a_env.max()),
                "gate_max": float(out.parameters.gate.max()),
            }
            for name, shift in out.env_shift.items():
                row[f"env_{name}"] = float(shift.mean())
            rows.append(row)
    _write_csv(rows, outdir / "dimer_scan.csv")
    _plot_grid(rows, "r_OO", xs, outdir / "dimer_scan.png", "dimer O-O scan")


def mode_lambda(model, outdir: Path) -> None:
    batch = dimer_batch(2.9)
    species_idx = model.projector.species_index(batch.atomic_numbers)
    a = StateDescriptor.from_batch(
        batch, species_idx, model.projector.featurizer.n_species
    )
    b = a.permute_fragments(torch.tensor([1, 0]))
    rows = []
    xs = torch.linspace(0.0, 1.0, 21).tolist()
    for lam in xs:
        state = StateDescriptor.blend(a, b, lam)
        pos = batch.positions.clone().requires_grad_(True)
        out = model(Batch(**{**batch.__dict__, "positions": pos}), state)
        (g,) = torch.autograd.grad(out.energy.sum(), pos)
        theta = out.parameters.bonded0
        rows.append({
            "lambda": lam,
            "E_total": float(out.energy[0]),
            "force_norm": float(g.norm()),
            "u_mean": float(state.mixing_measure().mean()),
            "n_bonds": int(out.topology.bond_index.shape[1]),
            "bond_weight_mean": float(out.topology.bond_weight.mean()),
            "D_mean": float(theta.d.mean()),
        })
    _write_csv(rows, outdir / "lambda_sweep.csv")
    _plot_grid(rows, "lambda", xs, outdir / "lambda_sweep.png",
               "C(lambda) relabeling sweep (dimer)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument(
        "--mode", choices=("sweep", "jacobian", "dimer", "lambda", "all"), default="all"
    )
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    outdir = args.checkpoint.parent / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    modes = {
        "sweep": mode_sweep, "jacobian": mode_jacobian,
        "dimer": mode_dimer, "lambda": mode_lambda,
    }
    for name, fn in modes.items():
        if args.mode in (name, "all"):
            print(f"--- {name} ---")
            fn(model, outdir)


if __name__ == "__main__":
    main()
