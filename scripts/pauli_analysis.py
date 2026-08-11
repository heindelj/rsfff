"""Evaluate a trained Pauli model: accuracy by cluster size, parameters, correction share.

    python scripts/pauli_analysis.py [checkpoint config [checkpoint config ...]]

Read-only with respect to the checkpoints. Mirrors ``scripts/dispersion_analysis.py`` but
feeds both the command-line report here and the Pauli half of notebooks/eda_rsff_plotting.ipynb.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsfff.ff.many_body import mbe_dataset  # noqa: E402
from rsfff.ff.pauli import DEFAULT_PAULI_PRIOR  # noqa: E402
from rsfff.ff.units import KJMOL_PER_HARTREE  # noqa: E402
from rsfff.train.config import load_config  # noqa: E402
from rsfff.train.data import load_extxyz  # noqa: E402
from rsfff.train.train_pauli import build_pauli_model  # noqa: E402

CLUSTERS = ("w2", "w3", "w4", "w5")
DATA_TEMPLATE = "data/eda_data/{tag}_wb97xv_qzvppd.xyz"
DEFAULT_RUNS = [
    ("environment-aware (full)",
     "checkpoints/eda_water_pauli/best.pt", "configs/eda_water_pauli.yaml"),
    ("intra-fragment (additive, same capacity)",
     "checkpoints/eda_water_pauli_intra/best.pt", "configs/eda_water_pauli_intra.yaml"),
]
BATCH = 200


def load_clusters(tags=CLUSTERS, dtype=torch.float64):
    return {tag: load_extxyz(DATA_TEMPLATE.format(tag=tag), dtype=dtype) for tag in tags}


#: Buffers added after some checkpoints were written. They are constants rebuilt from the
#: priors at construction, so a checkpoint that predates them loads correctly without one --
#: but everything else stays strict, so a genuine architecture mismatch still raises.
_BACKFILLABLE = {"pauli.params.quad_scale"}


def load_model(checkpoint: str, config: str):
    cfg = load_config(config)
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_pauli_model(cfg.features, cfg.pauli, state["neighbor_types"])
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    unknown = set(missing) - _BACKFILLABLE
    if unknown or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model built from {config}: "
            f"missing {sorted(unknown)}, unexpected {sorted(unexpected)}"
        )
    model.eval()
    return model, cfg, state


def predict(model, dataset, target="mod_pauli", n_frames=None, with_pairs=False):
    """Per-frame prediction plus the FF/correction split and per-atom parameters.

    ``with_pairs`` additionally returns the per-pair ``(r, e_ff, e_corr)`` arrays the
    switching-function diagnostic needs -- there is no other way to see *where in r* each
    piece of the model is actually contributing energy.
    """
    total = len(dataset) if n_frames is None else min(n_frames, len(dataset))
    out = {k: [] for k in ("pred", "ref", "ff", "corr")}
    atoms = {k: [] for k in ("Z", "q", "b", "mu", "quad")}
    pair = {k: [] for k in ("r", "e_ff", "e_corr")}
    for start in range(0, total, BATCH):
        batch = dataset.flat_batch(range(start, min(start + BATCH, total)))
        with torch.no_grad():
            o = model(batch)
        out["pred"].append(o.energy.numpy())
        out["ref"].append(batch.eda[target].numpy())
        out["ff"].append(o.energy_ff.numpy())
        out["corr"].append(o.energy_corr.numpy())
        atoms["Z"].append(batch.atomic_numbers.numpy())
        atoms["q"].append(o.q.numpy())
        atoms["b"].append(o.b.numpy())
        atoms["mu"].append(
            o.mu.norm(dim=-1).numpy() if o.mu is not None else np.zeros(len(o.q))
        )
        atoms["quad"].append(
            o.quad_s.norm(dim=-1).numpy() if o.quad_s is not None else np.zeros(len(o.q))
        )
        if with_pairs:
            pair["r"].append(o.r.numpy())
            pair["e_ff"].append(o.e_pair_ff.numpy() * KJMOL_PER_HARTREE)
            pair["e_corr"].append(o.e_pair_corr.numpy() * KJMOL_PER_HARTREE)
    res = {k: np.concatenate(v) * KJMOL_PER_HARTREE for k, v in out.items()}
    res.update({k: np.concatenate(v) for k, v in atoms.items()})
    if with_pairs:
        res.update({f"pair_{k}": np.concatenate(v) for k, v in pair.items()})
    return res


def report(label, model, datasets, target):
    print(f"\n=== {label} ===")
    print(f"{'cluster':<9}{'n':<7}{'MAE':<9}{'RMSE':<9}{'R^2':<10}{'<ref>':<9}"
          f"{'<|E_corr|/|E|>'}")
    errors, all_atoms = [], []
    for tag, ds in datasets.items():
        p = predict(model, ds, target)
        err = p["pred"] - p["ref"]
        share = np.abs(p["corr"]) / np.maximum(np.abs(p["ff"]) + np.abs(p["corr"]), 1e-30)
        r2 = np.corrcoef(p["pred"], p["ref"])[0, 1] ** 2
        print(f"{tag:<9}{len(err):<7}{np.abs(err).mean():<9.4f}"
              f"{np.sqrt((err**2).mean()):<9.4f}{r2:<10.5f}{p['ref'].mean():<9.2f}"
              f"{share.mean() * 100:.2f}%")
        errors.append(err)
        all_atoms.append(p)
    allerr = np.concatenate(errors)
    print(f"overall MAE {np.abs(allerr).mean():.4f} kJ/mol over {len(allerr)} frames")

    Z = np.concatenate([a["Z"] for a in all_atoms])
    print("learned per-atom parameters (prior in parentheses):")
    for z, sym in ((1, "H"), (8, "O")):
        m = Z == z
        q = np.concatenate([a["q"] for a in all_atoms])[m]
        b = np.concatenate([a["b"] for a in all_atoms])[m]
        mu = np.concatenate([a["mu"] for a in all_atoms])[m]
        quad = np.concatenate([a["quad"] for a in all_atoms])[m]
        print(f"  {sym}: q {q.mean():7.3f} +/- {q.std():.3f} ({DEFAULT_PAULI_PRIOR[z][0]:.3f}), "
              f"range {q.min():.3f}-{q.max():.3f}   "
              f"b {b.mean():5.3f} +/- {b.std():.3f} ({DEFAULT_PAULI_PRIOR[z][1]:.3f})   "
              f"|mu| {mu.mean():.4f}   |Q| {quad.mean():.4f}")
    return {tag: np.abs(e).mean() for tag, e in zip(datasets, errors)}


def strongest_frames(dataset, n: int, *, key: str = "int") -> np.ndarray:
    """Indices of the ``n`` most strongly interacting frames (most negative ``eda[key]``)."""
    values = dataset.flat_batch(range(len(dataset))).eda[key].numpy()
    return np.argsort(values)[:n]


def many_body_table(model, datasets, per_cluster: int, *, target="mod_pauli",
                    key: str = "int", tags=("w3", "w4", "w5"), progress_every: int = 0):
    """MBE of the strongest ``per_cluster`` frames of each cluster size, in kJ/mol.

    Unlike the dispersion table this keeps the force-field / correction split **per
    expansion order**, not just aggregated over everything beyond pairwise. For Pauli that
    distinction matters: the Slater form is a pair sum, so every ``E^(k>=3)`` it produces
    comes from the *environment dependence of the emitted multipoles* -- remove a
    neighboring molecule and the remaining monomers' charges, dipoles and quadrupoles all
    change. Splitting per order says whether that mechanism or the neural correction is
    supplying the non-additivity at each level.

    Returns flat arrays: ``tag``, ``n_fragments``, ``ref``, and for each of
    ``total``/``ff``/``corr`` the keys ``<c>_total``, ``<c>_two_body``, ``<c>_many_body``
    and ``<c>_e3``/``e4``/``e5``.
    """
    comps = ("total", "ff", "corr")
    rows: dict[str, list] = {k: [] for k in ("tag", "n_fragments", "ref")}
    for c in comps:
        for suffix in ("total", "two_body", "many_body", "e3", "e4", "e5"):
            rows[f"{c}_{suffix}"] = []

    k = KJMOL_PER_HARTREE
    for tag in tags:
        ds = datasets[tag]
        idx = strongest_frames(ds, per_cluster, key=key)
        res = mbe_dataset(model, ds, idx, progress_every=progress_every)
        n = len(idx)
        rows["tag"] += [tag] * n
        rows["n_fragments"] += res.n_fragments.tolist()
        rows["ref"] += (ds.flat_batch(idx).eda[target].numpy() * k).tolist()
        for c in comps:
            r = res if c == "total" else res.components[c]
            rows[f"{c}_total"] += (r.total.numpy() * k).tolist()
            rows[f"{c}_two_body"] += (r.two_body.numpy() * k).tolist()
            rows[f"{c}_many_body"] += (r.many_body.numpy() * k).tolist()
            for order, name in ((3, "e3"), (4, "e4"), (5, "e5")):
                vals = r.by_order.get(order)
                rows[f"{c}_{name}"] += (
                    (vals.numpy() * k).tolist() if vals is not None else [0.0] * n
                )
    return {kk: np.asarray(v) for kk, v in rows.items()}


def atom_parameters(model, datasets, n_frames=150, target="mod_pauli"):
    """Per-atom parameters pooled over cluster sizes, for the parameter figure."""
    parts = [predict(model, ds, target, n_frames=n_frames) for ds in datasets.values()]
    return {k: np.concatenate([p[k] for p in parts])
            for k in ("Z", "q", "b", "mu", "quad")}


def main(argv) -> int:
    runs = DEFAULT_RUNS
    if len(argv) >= 2:
        runs = [(f"run {i//2 + 1}", argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]

    datasets = load_clusters()
    table = {}
    for label, ckpt, conf in runs:
        if not os.path.exists(ckpt):
            print(f"\nskipping '{label}': {ckpt} not found")
            continue
        model, cfg, state = load_model(ckpt, conf)
        print(f"\n[{label}] epoch {state['epoch']}  val_loss {state['val_loss']:.4f}  "
              f"max_rank {cfg.pauli.max_rank}")
        table[label] = report(label, model, datasets, cfg.pauli.target)

    if len(table) > 1:
        print("\nMAE by cluster size (kJ/mol):")
        print(f"{'variant':<44}" + "".join(f"{t:<10}" for t in CLUSTERS))
        for label, row in table.items():
            print(f"{label:<44}" + "".join(f"{row[t]:<10.4f}" for t in CLUSTERS))
        base = next(iter(table.values()))
        for label, row in list(table.items())[1:]:
            print(f"\n'{label}' vs the full model:")
            for t in CLUSTERS:
                print(f"  {t}: {row[t]:.4f} -> {base[t]:.4f}  ({row[t] / base[t]:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
