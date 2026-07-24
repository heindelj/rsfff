"""Scan O-H dissociation of H2O / H3O+ / OH- through the diabatic mixture model.

    python scripts/dissociation_scan.py

Loads the frozen Phase-1 monomer stack, wraps it in a :class:`rsfff.mlip.MixtureModel`, and
pulls one O-H bond of each monomer apart from equilibrium to ~10 A. For each geometry it records
the total energy, the mixing weights c_K(r), the per-atom SQE charges, and the stretched-channel
compliance, and writes a figure to ``notebooks/figures/dissociation.png``.

What the plots should show (all *architectural*, on the untrained gate):
  * c_bound / c_dissoc cross smoothly over the envelope overlap window.
  * the stretched-channel compliance is driven to exactly 0 by the switch.
  * the charge on the leaving fragment collapses to an exact integer (the reference-state limit).
  * the energy is flat at the sum of the fragment reference predictions beyond the cutoff --
    there is no -1/r or -C6/r^6 tail yet (long-range electrostatics is Phase 3).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from ase.data import atomic_numbers as ATOMIC_NUMBER  # noqa: E402

from rsfff.mlip import (  # noqa: E402
    AtomicStateReference,
    DiabaticStateLibrary,
    EnvelopeConfig,
    MixtureModel,
    build_monomer_model,
    enumerate_diabats,
)
from rsfff.train.config import load_config  # noqa: E402
from rsfff.train.data import Batch, load_reference_energies  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Equilibrium geometries (Angstrom); atom 0 is O, atom 1 is the O-H being stretched.
GEOMETRY = {
    "oh-":  (["O", "H"], [[0, 0, 0.1078], [0, 0, -0.8621]]),
    "h2o":  (["O", "H", "H"],
             [[0, 0, 0.1178], [0, 0.7642, -0.4712], [0, -0.7642, -0.4714]]),
    "h3o+": (["O", "H", "H", "H"],
             [[0, 0, 0.0749], [0, 0.9436, -0.1997], [0.8172, -0.4718, -0.1997],
              [-0.8172, -0.4718, -0.1997]]),
}


def load_mixture(config_path=os.path.join(REPO, "configs", "mixture_h2o_h3o_oh.yaml")):
    """Build a :class:`MixtureModel` from the frozen checkpoint plus the envelope config.

    Returns ``(model, library, EnvelopeConfig)``. Shared with the test suite so the exact same
    wiring is exercised there.
    """
    raw = yaml.safe_load(open(config_path))
    mono_cfg = load_config(os.path.join(REPO, raw["monomer_config"]))
    library = DiabaticStateLibrary.from_yaml(os.path.join(REPO, mono_cfg.data.diabatic_states))
    neighbor_types = [1, 8]
    e0 = load_reference_energies(
        os.path.join(REPO, mono_cfg.data.reference_energies), neighbor_types
    )
    states = AtomicStateReference.from_json(
        os.path.join(REPO, mono_cfg.data.atomic_reference_states), neighbor_types
    )
    monomer = build_monomer_model(
        mono_cfg.features, mono_cfg.monomer, mono_cfg.sqe, neighbor_types, e0, states
    )
    ckpt = torch.load(os.path.join(REPO, raw["checkpoint"]), map_location="cpu",
                      weights_only=False)
    monomer.load_state_dict(ckpt["model_state"])
    monomer.eval()
    model = MixtureModel(monomer)
    model.eval()

    m = raw["mixture"]
    env = EnvelopeConfig(
        bound_envelope=m["bound_envelope"], channel_envelope=m["channel_envelope"],
        switch_r_on=float(m["switch_r_on"]), switch_r_off=float(m["switch_r_off"]),
        beta=float(m.get("beta", 0.0)), tau=float(m.get("tau", 1.0)),
    )
    return model, library, env


def single_system_batch(symbols, positions):
    n = len(symbols)
    return Batch(
        positions=torch.tensor(np.asarray(positions, dtype=float)),
        atomic_numbers=torch.tensor([ATOMIC_NUMBER[s] for s in symbols], dtype=torch.long),
        batch_idx=torch.zeros(n, dtype=torch.long), n_systems=1,
        energy=torch.zeros(1), forces=torch.zeros(n, 3),
    )


def scan(model, library, env, config_type, distances):
    """Stretch the O0-H1 bond over ``distances``; return per-r arrays."""
    symbols, pos0 = GEOMETRY[config_type]
    pos0 = np.array(pos0, dtype=float)
    unit = (pos0[1] - pos0[0]) / np.linalg.norm(pos0[1] - pos0[0])
    das = enumerate_diabats(library, symbols, config_type=config_type)
    rec = {k: [] for k in ("E", "c_bound", "c_diss", "q", "s_stretched")}
    for r in distances:
        pos = pos0.copy()
        pos[1] = pos0[0] + r * unit
        out = model(single_system_batch(symbols, pos), das, env)
        rec["E"].append(float(out.energy))
        rec["c_bound"].append(float(out.weights[0]))
        rec["c_diss"].append(float(out.weights[1]))
        rec["q"].append(out.charges.detach().numpy())
        # the stretched channel is the inter-fragment one (O0-H1)
        e = int(np.flatnonzero(out.is_inter.numpy())[0])
        rec["s_stretched"].append(float(out.compliance[e]))
    rec["q"] = np.array(rec["q"])
    return symbols, rec


def main():
    torch.set_default_dtype(torch.float64)
    model, library, env = load_mixture()
    distances = np.linspace(0.85, 10.0, 120)

    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    species = ["oh-", "h2o", "h3o+"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for col, ct in enumerate(species):
        symbols, rec = scan(model, library, env, ct, distances)
        d = distances
        # weights
        axes[0, col].plot(d, rec["c_bound"], label=f"c(|{ct}⟩)", lw=1.8)
        axes[0, col].plot(d, rec["c_diss"], label="c(dissociated)", lw=1.8, ls="--")
        axes[0, col].set_title(f"{ct}: mixing weights"); axes[0, col].legend(fontsize=8)
        # charges
        for i in range(len(symbols)):
            axes[1, col].plot(d, rec["q"][:, i], lw=1.8, ls="--" if i == 1 else "-",
                              label=f"{symbols[i]}{i}" + (" (leaving)" if i == 1 else ""))
        axes[1, col].axhline(0, color="k", lw=0.5)
        axes[1, col].set_title(f"{ct}: SQE charges"); axes[1, col].legend(fontsize=8)
        # compliance of the breaking bond
        axes[2, col].plot(d, rec["s_stretched"], lw=1.8, color="#b2182b")
        axes[2, col].set_title(f"{ct}: stretched-channel compliance × switch")
        axes[2, col].set_xlabel("O0–H1 distance [Å]")
    axes[0, 0].set_ylabel("mixing weight")
    axes[1, 0].set_ylabel("charge q [e]")
    axes[2, 0].set_ylabel(r"compliance s [e$^2$/Ha]")
    fig.tight_layout()
    out = os.path.join(REPO, "notebooks", "figures", "dissociation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # A short numeric summary of the endpoints.
    for ct in species:
        symbols, rec = scan(model, library, env, ct, np.array([0.97, 10.0]))
        q_far = rec["q"][-1]
        print(f"{ct:>5}: q(leaving H, r=10A) = {q_far[1]:+.6f}   "
              f"s_stretched(r=10A) = {rec['s_stretched'][-1]:.2e}   E = {rec['E'][-1]:.6f}")


if __name__ == "__main__":
    sys.exit(main())
