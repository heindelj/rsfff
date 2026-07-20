"""Diagnostics for a trained (or fresh) MLIP+EEM model.

    python scripts/eem_diagnostics.py configs/h2o_h3o_eem.yaml \
        [--checkpoint checkpoints/h2o_h3o_eem/best.pt] [--eq-curves] [--frames 3]

Reports:
- E_elem(q) curves per element: single atoms across a charge grid (the constraint
  pins q for a 1-atom system -- direct evaluation of the chi0/eta0 parabola +
  MLIP(0) offset). Written as CSV next to the checkpoint (or ./ when none).
- H3O+ charge distribution and effective per-atom parameters (chi, eta).
- Molecular alpha decomposition: sum of atomic alphas vs the charge-flow term.
- Dipole in the stored frame and at the center of mass (labels/losses use the
  stored frame; COM values are for physical interpretation only).
- q and mu response to a finite applied field through the closed form.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# macOS: torch bundles its own libomp while conda's MKL loads llvm-openmp; two
# OpenMP runtimes in one process abort with "OMP: Error #15". Must be set before
# torch/numpy initialize libomp, hence here rather than in the shell.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from rsfff.mlip import build_eem_model  # noqa: E402
from rsfff.mlip.eem import charge_flow_polarizability
from rsfff.train.config import load_config
from rsfff.train.data import Batch, load_datasets, load_reference_energies

ATOMIC_MASSES = {1: 1.008, 8: 15.999}


def single_atom_batch(z: int, q: float) -> Batch:
    pos = torch.zeros(1, 3)
    return Batch(
        positions=pos, atomic_numbers=torch.tensor([z]),
        batch_idx=torch.zeros(1, dtype=torch.long), n_systems=1,
        energy=torch.zeros(1), forces=torch.zeros(1, 3),
        total_charge=torch.tensor([q]),
    )


def eq_curves(model, neighbor_types, out_dir: Path):
    from ase.data import chemical_symbols

    q_grid = torch.linspace(-1.5, 1.5, 61)
    lines = ["element,q,E_elem_total_Ha"]
    for z in neighbor_types:
        sym = chemical_symbols[z]
        for q in q_grid.tolist():
            e = model(single_atom_batch(z, q)).energy.item()
            lines.append(f"{sym},{q:.4f},{e:.10f}")
        for qi in (-1.0, 0.0, 1.0):
            e = model(single_atom_batch(z, qi)).energy.item()
            print(f"  E_elem({sym}, q={qi:+.0f}) = {e:.6f} Ha")
    path = out_dir / "e_elem_curves.csv"
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eq-curves", action="store_true")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    neighbor_types = dataset.unique_atomic_numbers
    e0 = load_reference_energies(config.data.reference_energies, neighbor_types)
    model = build_eem_model(
        config.features, config.mlip, config.eem, neighbor_types, e0
    )

    out_dir = Path(".")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        out_dir = Path(args.checkpoint).parent
        print(f"loaded {args.checkpoint} (epoch {state.get('epoch')})")
    else:
        print("no checkpoint: diagnostics on the *initialized* model")
    model.eval()

    if args.eq_curves:
        print("\nE_elem(q) curves (single atoms, direct evaluation):")
        eq_curves(model, neighbor_types, out_dir)

    h3o_indices = [
        i for i in range(len(dataset))
        if float(dataset.flat_batch([i]).total_charge[0]) > 0.5
    ][: args.frames]
    print(f"\nH3O+ diagnostics on {len(h3o_indices)} frames:")
    for idx in h3o_indices:
        batch = dataset.flat_batch([idx])
        out = model(batch)
        q = out.charges.detach()
        z = batch.atomic_numbers
        print(f"  frame {idx}: q(O) = {q[z == 8].item():+.4f}  "
              f"q(H) = {[round(v, 4) for v in q[z == 1].tolist()]}  "
              f"sum = {q.sum().item():+.6f}")
        print(f"    eta = {[round(v, 3) for v in out.eta.detach().tolist()]}  "
              f"lambda = {out.lam.item():+.4f}")

        mu = out.dipole[0].detach()
        masses = torch.tensor([ATOMIC_MASSES[int(v)] for v in z])
        com = (masses.unsqueeze(-1) * batch.positions).sum(0) / masses.sum()
        mu_com = mu - batch.total_charge[0] * com
        print(f"    mu(stored frame) = {mu.numpy().round(4)}  "
              f"mu(COM) = {mu_com.numpy().round(4)} e*A  "
              f"(target {batch.dipole[0].numpy().round(4) if batch.dipole is not None else 'n/a'})")

        aflow = charge_flow_polarizability(
            out.eta.detach(), batch.positions, batch.batch_idx, 1
        )[0]
        a_atoms = out.alpha[0].detach() - aflow
        print(f"    alpha iso: total {out.alpha[0].detach().trace().item()/3:.4f}  "
              f"= atomic {a_atoms.trace().item()/3:.4f} + charge-flow "
              f"{aflow.trace().item()/3:.4f}  "
              f"(target {batch.polarizability[0].trace().item()/3:.4f})"
              if batch.polarizability is not None else "")

        # q response to a finite field along z, through the closed form
        h = 1e-3
        f_plus = torch.zeros(1, 3); f_plus[0, 2] = h
        out_p = model(dataset.flat_batch([idx]), field=f_plus)
        dq_fd = (out_p.charges.detach() - q) / h
        print(f"    dq/dF_z (finite field): {[round(v, 4) for v in dq_fd.tolist()]}")


if __name__ == "__main__":
    main()
