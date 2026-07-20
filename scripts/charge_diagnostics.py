"""Charge-sanity diagnostics for a trained (or fresh) charge-aware model.

    python scripts/charge_diagnostics.py configs/h2o_h3o_charge.yaml \
        [--checkpoint checkpoints/h2o_h3o_charge/best.pt] [--eq-curves] [--frames 5]

Reports (docs/charge_aware_embedding.md, M3):
- E_elem(q) curves per element: single atoms evaluated across a charge grid
  (the constraint pins q for a 1-atom system, so this is a direct evaluation --
  no SCF freedom). Written as CSV next to the checkpoint (or ./ when none).
- H3O+ excess-charge distribution over the H atoms.
- q response to an applied field: dq/dF from the attach vs a finite field
  through the SCF (sign/magnitude sanity).
- alpha consistency: field route vs head route, plus dipoles of H3O+ at the
  center of mass (convention: labels/losses use the stored frame; COM values
  are reported here for physical interpretation only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rsfff.mlip import build_charge_model
from rsfff.train.config import load_config
from rsfff.train.data import Batch, load_datasets, load_reference_energies
from rsfff.train.loss import compute_polarizability

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
        # report the integer-charge points inline
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
    model = build_charge_model(config.features, config.charge, neighbor_types, e0)

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

    # --- per-frame charge/dipole/alpha diagnostics on H3O+ frames ---
    h3o_indices = [
        i for i in range(len(dataset))
        if float(dataset.flat_batch([i]).total_charge[0]) > 0.5
    ][: args.frames]
    print(f"\nH3O+ diagnostics on {len(h3o_indices)} frames:")
    for idx in h3o_indices:
        batch = dataset.flat_batch([idx])
        batch.positions.requires_grad_(True)
        field = torch.zeros(1, 3, requires_grad=True)
        out = model(batch, field=field)
        q = out.charges.detach()
        z = batch.atomic_numbers
        print(f"  frame {idx}: q(O) = {q[z == 8].item():+.4f}  "
              f"q(H) = {[round(v, 4) for v in q[z == 1].tolist()]}  "
              f"sum = {q.sum().item():+.6f}")

        # dipole at stored origin and at COM (convention note in the docstring)
        mu = out.dipole[0].detach()
        masses = torch.tensor([ATOMIC_MASSES[int(v)] for v in z])
        com = (masses.unsqueeze(-1) * batch.positions.detach()).sum(0) / masses.sum()
        mu_com = mu - batch.total_charge[0] * com
        print(f"    mu(stored frame) = {mu.numpy().round(4)}  "
              f"mu(COM) = {mu_com.numpy().round(4)} e*A")

        alpha_field = compute_polarizability(out.dipole, field, create_graph=False)[0]
        alpha_head = out.alpha_head[0].detach()
        print(f"    alpha iso: field route {alpha_field.trace().item()/3:.4f}  "
              f"head route {alpha_head.trace().item()/3:.4f}  "
              f"(target units e^2*A^2/Ha)")

        # q response to a finite field along z vs the attach-graph derivative
        dqdF = torch.autograd.grad(
            (out.charges * 1.0).sum(), field, retain_graph=True, allow_unused=True
        )
        h = 1e-3
        f_plus = torch.zeros(1, 3); f_plus[0, 2] = h
        out_p = model(dataset.flat_batch([idx]), field=f_plus)
        dq_fd = (out_p.charges.detach() - q) / h
        print(f"    dq/dF_z (finite field): {[round(v, 4) for v in dq_fd.tolist()]}")

    if out.scf_info is not None:
        print(f"\nlast SCF: iters={out.scf_info.n_iter} "
              f"residual={out.scf_info.residual:.2e} "
              f"min_tangent_eig={out.scf_info.min_tangent_eig:.4f}")


if __name__ == "__main__":
    main()
