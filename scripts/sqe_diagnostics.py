"""Inspect a trained monomer stack: charges, split charges, and channel compliances.

    python scripts/sqe_diagnostics.py [CHECKPOINT] [CONFIG]

Prints, per fragment state in the training set, the SQE charges and the learned per-bond
compliances, plus the free-atom audit (isolated-atom energies and polarizabilities through the
*trained* model, which §4.4.7 requires to stay pinned to the reference states).

What to look for:
  * charges: O negative and H positive in all three species; the excess charge of H3O+ spread
    over the hydrogens; OH- carrying its extra electron almost entirely on oxygen.
  * compliances: larger s means a softer channel. Within a single fragment, s should track the
    O-H distance -- a stretched bond is more compliant to charge flow, which is the beginning
    of the dissociation behavior Phase 3 has to get right. (The *ordering between* species is
    an empirical result, not something to expect a priori.)
  * transfers: q_i = q_i^(0) + sum_j p_ij should reconcile exactly, and every fragment's
    charges must sum to its formal charge to machine precision.
  * free-atom audit: how far training has pulled the isolated-atom predictions away from the
    reference data. These start *exact* (see rsfff.mlip.monomer) and drift by however much the
    finite ``atomic_ref_weight`` lets the molecular targets win; raise that weight to pin them
    harder. States flagged unbound are excluded from the loss, so large errors there are
    expected, not a regression.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from rsfff.mlip import (  # noqa: E402
    AtomicStateReference,
    DiabaticStateLibrary,
    build_monomer_model,
)
from rsfff.train.config import load_config  # noqa: E402
from rsfff.train.data import (  # noqa: E402
    load_atomic_reference_batch,
    load_datasets,
    load_reference_energies,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CKPT = os.path.join(REPO, "checkpoints", "monomer_h2o_h3o_oh", "best.pt")
DEFAULT_CONFIG = os.path.join(REPO, "configs", "monomer_h2o_h3o_oh.yaml")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    config_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG

    config = load_config(config_path)
    torch.set_default_dtype(
        torch.float64 if config.dtype == "float64" else torch.float32
    )

    library = DiabaticStateLibrary.from_yaml(config.data.diabatic_states)
    dataset = load_datasets(
        config.data.path, dtype=torch.get_default_dtype(), library=library
    )
    neighbor_types = dataset.unique_atomic_numbers
    e0 = load_reference_energies(config.data.reference_energies, neighbor_types)
    states = AtomicStateReference.from_json(
        config.data.atomic_reference_states, neighbor_types,
        dtype=torch.get_default_dtype(),
    )
    model = build_monomer_model(
        config.features, config.monomer, config.sqe, neighbor_types, e0, states
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch'] + 1}, val loss {ckpt['val_loss']:.4e})\n")

    from ase.data import chemical_symbols

    # One representative frame per label file (they are concatenated in config order).
    frames_per_file = len(dataset) // len(config.data.path)

    # --- per-species accuracy on the held-out split ---------------------------------------
    # Reported per species because the aggregate hides the interesting question: whether the
    # shared reference-state parameterization costs anything on any one fragment state.
    from rsfff.train.data import split_indices
    from rsfff.train.loss import compute_forces

    _, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    print("=" * 78)
    print(f"held-out accuracy ({len(val_idx)} frames), per fragment state")
    print("=" * 78)
    # The energy column is split deliberately. A rigid per-fragment-state offset and a
    # misshapen energy surface are very different failures: the offset is one number per state
    # (and is exactly what the atomic-reference anchors are fighting the molecular targets
    # over -- the model must get atomization energies right with the same parameters that
    # reproduce isolated atoms, which a model with free per-species constants never has to),
    # while the offset-removed column is what forces and dynamics actually see.
    print(f"{'species':>8}  {'n':>4}  {'E MAE (Ha)':>12}  {'E offset':>11}  {'E MAE -off':>11}"
          f"  {'F MAE':>10}  {'mu MAE':>10}  {'alpha MAE':>10}")
    for k, path in enumerate(config.data.path):
        name = os.path.basename(str(path)).replace(".extxyz", "")
        lo, hi = k * frames_per_file, (k + 1) * frames_per_file
        sel = val_idx[(val_idx >= lo) & (val_idx < hi)]
        if not len(sel):
            continue
        batch = dataset.flat_batch(sel)
        batch.positions.requires_grad_(True)
        with torch.enable_grad():
            out = model(batch)
            forces = compute_forces(out.energy, batch.positions, create_graph=False)
        e_err = (out.energy - batch.energy).detach()
        e_mae = float(e_err.abs().mean())
        e_off = float(e_err.mean())
        e_mae_shifted = float((e_err - e_err.mean()).abs().mean())
        f_mae = float((forces - batch.forces).detach().abs().mean())
        mu_mae = float((out.dipole - batch.dipole).detach().abs().mean())
        a_mae = float((out.alpha - batch.polarizability).detach().abs().mean())
        print(f"{name:>8}  {len(sel):>4}  {e_mae:12.3e}  {e_off:+11.3e}  {e_mae_shifted:11.3e}"
              f"  {f_mae:10.3e}  {mu_mae:10.3e}  {a_mae:10.3e}")
    print()
    print("=" * 78)
    print("per-fragment charges and channel compliances")
    print("=" * 78)
    for k, path in enumerate(config.data.path):
        batch = dataset.flat_batch(torch.tensor([k * frames_per_file]))
        out = model(batch)
        name = os.path.basename(str(path)).replace(".extxyz", "")
        q_total = float(out.charges.detach().sum())
        print(f"\n{name}  (Q = {float(batch.fragment_charge[0]):+.0f}, "
              f"sum q = {q_total:+.6f})")
        for i in range(batch.positions.shape[0]):
            sym = chemical_symbols[int(batch.atomic_numbers[i])]
            print(f"    {sym}{i}   q = {float(out.charges[i]):+.4f}   "
                  f"q0 = {float(out.baseline_charge[i]):+.4f}")
        for e in range(batch.bond_index.shape[1]):
            a, b = int(batch.bond_index[0, e]), int(batch.bond_index[1, e])
            sa = chemical_symbols[int(batch.atomic_numbers[a])]
            sb = chemical_symbols[int(batch.atomic_numbers[b])]
            r = float((batch.positions[a] - batch.positions[b]).norm())
            print(f"    {sa}{a}-{sb}{b}  r = {r:.3f} A   "
                  f"s = {float(out.compliance[e]):.4f}   "
                  f"p = {float(out.transfers[e]):+.4f}")
        mu = float(out.dipole[0].norm()) * 4.803204  # e*Angstrom -> Debye
        print(f"    |mu| = {mu:.3f} D    alpha_iso = "
              f"{float(out.alpha[0].trace() / 3):.3f} e^2 A^2/Ha")

    print("\n" + "=" * 78)
    print("free-atom audit: isolated atoms through the trained model (should match the")
    print("reference data exactly -- the monomer-preservation criterion of §4.4.7)")
    print("=" * 78)
    anchor = load_atomic_reference_batch(states, neighbor_types)
    out = model(anchor)
    print(f"{'state':>7}  {'q':>8}  {'E model':>14}  {'E ref':>14}  {'dE (kcal/mol)':>14}"
          f"  {'bound':>6}")
    for i, sym in enumerate(states.symbols):
        d = float(out.energy[i] - anchor.energy[i]) * 627.5094740631
        print(f"{sym:>7}  {float(out.charges[i]):+8.4f}  {float(out.energy[i]):14.8f}  "
              f"{float(anchor.energy[i]):14.8f}  {d:+14.4f}  "
              f"{str(bool(states.bound[i])):>6}")

    print(f"\n{'state':>7}  {'alpha_iso model':>16}  {'alpha_iso ref':>14}")
    for i, sym in enumerate(states.symbols):
        a_model = float(out.alpha[i].trace() / 3)
        a_ref = float(anchor.polarizability[i].trace() / 3)
        print(f"{sym:>7}  {a_model:16.4f}  {a_ref:14.4f}")


if __name__ == "__main__":
    main()
