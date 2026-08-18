"""Measure a staged unified fit against the failures that motivated its current shape.

``configs/water_staged.yaml`` fits three levels in sequence, and the things that went wrong
with it were *not* visible in the per-epoch MAEs. Each check below exists because a real
regression hid behind a healthy-looking log:

1. **One-body bias per fragment, per stage.** A pure constant, so ``ob_mae`` reports it while
   the correlation plot shows a slope (the constant times ``n_frag``, since the plot pools
   fragments to frames). Measured at +0.014 / +2.511 / -0.627 kJ/mol across frozen / pol / ct
   on the fit that motivated ``freeze_frozen_level``, identical to four digits across w2..w5.
2. **The internal/bond split.** The unlabeled gauge inside ``fragment_energy``. It moved by
   -62 then +109 kJ/mol while ``ob_mae`` stayed small, because the bond head chased it.
3. **Monomer polarizability eigenvalues.** Fitted almost exactly at the frozen stage (9.904
   a0^3 isotropic against a true 9.891) and then dismantled: 8.903 by the ct stage, with the
   two in-plane components down 1.8 a0^3 each. The isotropic average alone hides this.
4. **Three-body content of each interaction channel.** ``cls_elec`` must be exactly zero --
   ``eda_frz_elec`` is the Coulomb interaction of superimposed frozen monomer densities and is
   rigorously pairwise. ``pauli`` and ``disp`` must *not* be, since their parameters read
   ``h_env`` on inter pairs by design.
5. **Block norms.** Weight decay deleted five zero-initialized blocks and every equivariant
   ``equiv_reduce`` in one run, silently. A block reading exactly zero is the failure; a small
   one is a choice.

Usage:
    python scripts/staged_diagnostics.py [CONFIG] [--stages frozen,pol,ct] [--frames N]
"""

import argparse
import os
import sys

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rsfff.ff.many_body import mbe_dataset                      # noqa: E402
from rsfff.ff.units import KJMOL_PER_HARTREE as K               # noqa: E402
from rsfff.mlip.reference_states import AtomicStateReference    # noqa: E402
from rsfff.train.config import load_config, stage_config        # noqa: E402
from rsfff.train.data import (                                  # noqa: E402
    load_extxyz,
    load_reference_energies,
)
from rsfff.train.train_unified import build_unified_model       # noqa: E402

#: e^2 Angstrom^2 / Hartree per a0^3.
A0_CUBED = 0.2800852
#: Below this a block is reported as deleted rather than small. See `zero_init_readout`.
DEAD = 1.0e-6


def load_stage(cfg0, stage, checkpoint_root=None):
    """``(model, cfg)`` for one stage, from its own checkpoint."""
    cfg = stage_config(cfg0, stage)
    root = checkpoint_root or cfg.checkpoint_root
    path = os.path.join(root, f"{cfg0.run_name}_{stage.name}", "best.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    nt = state["neighbor_types"]
    ref = load_reference_energies(cfg.data.reference_energies, nt).to(
        torch.get_default_dtype()
    )
    states = None
    if cfg.data.atomic_reference_states:
        states = AtomicStateReference.from_json(
            cfg.data.atomic_reference_states, nt, dtype=torch.get_default_dtype()
        )
    model = build_unified_model(cfg, nt, ref, states)
    # Non-strict on purpose. The point of this script is comparing a fit against an earlier
    # one, and an earlier one predates whichever blocks were added since -- refusing to load
    # it would make the before/after comparison the tool exists for impossible. What is
    # missing gets printed, because a block silently left at its initialization is exactly the
    # kind of thing these checks are looking for elsewhere.
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    if missing or unexpected:
        blocks = sorted({k.rsplit(".", 1)[0] for k in list(missing) + list(unexpected)})
        print(f"  [{stage.name}] checkpoint predates this model: {len(missing)} tensors left "
              f"at initialization, {len(unexpected)} unused ({', '.join(blocks[:4])})")
    return model, cfg, state, set(missing)


def fragment_to_batch(batch):
    if batch.fragment_to_batch is not None:
        return batch.fragment_to_batch
    return batch.batch_idx.new_zeros(batch.n_fragments).scatter_(
        0, batch.fragment_idx, batch.batch_idx
    )


def validation_indices(dataset, cfg, limit):
    n = len(dataset)
    perm = np.random.default_rng(cfg.data.seed).permutation(n)
    n_val = max(1, int(round(n * cfg.data.holdout_fraction)))
    return np.sort(perm[:n_val]).tolist()[:limit]


def onebody_bias(model, datasets, cfg, limit, batch_size=16):
    """Per-fragment one-body residual, per cluster size.

    Reported per *fragment* and not per frame because that is where the constant lives: a
    frame-level number scales with ``n_frag`` and looks like a slope instead.
    """
    rows = {}
    for tag, ds in datasets.items():
        idx = validation_indices(ds, cfg, limit)
        per_frag = []
        for s in range(0, len(idx), batch_size):
            batch = ds.flat_batch(idx[s:s + batch_size])
            with torch.no_grad():
                out = model(batch)
            err = (out.fragment_energy - batch.fragment_energy) * K
            f2b = fragment_to_batch(batch)
            n = torch.bincount(f2b, minlength=batch.n_systems).to(err.dtype)
            frame = err.new_zeros(batch.n_systems).index_add_(0, f2b, err)
            per_frag.append((frame / n).numpy())
        v = np.concatenate(per_frag)
        rows[tag] = (float(v.mean()), float(np.abs(v).mean()))
    return rows


def monomer_split(model, anchor, take=64):
    """``(internal, bond, one-body bias, alpha eigenvalues)`` on the monomer anchor."""
    batch = anchor.flat_batch(list(range(min(take, len(anchor)))))
    with torch.no_grad():
        out = model(batch, with_polarizability=True)
    ev = torch.linalg.eigvalsh(out.polarizability / A0_CUBED).numpy()
    ref = (
        None if batch.polarizability is None
        else torch.linalg.eigvalsh(batch.polarizability / A0_CUBED).numpy()
    )
    return {
        "internal": float(out.energy_internal.mean()) * K,
        "bond": float(out.energy_bond.mean()) * K,
        "ob_bias": float((out.fragment_energy - batch.fragment_energy).mean()) * K,
        "alpha": ev.mean(0),
        "alpha_ref": None if ref is None else ref.mean(0),
    }


def three_body(model, dataset, limit, step=3):
    """Mean |3-body| per interaction channel, kJ/mol."""
    from types import SimpleNamespace

    class Pick(torch.nn.Module):
        def __init__(self, inner, name):
            super().__init__()
            self.inner, self.name = inner, name

        def forward(self, batch):
            return SimpleNamespace(energy=self.inner(batch).interaction[self.name])

    idx = list(range(0, min(len(dataset), limit * step), step))
    out = {}
    with torch.no_grad():
        names = list(model(dataset.flat_batch(idx[:1])).interaction)
    for name in names:
        res = mbe_dataset(Pick(model, name), dataset, idx, split_components=False)
        out[name] = float(np.abs(res.by_order[3].numpy() * K).mean())
    return out


def block_norms(model):
    """Readout and channel-reduction norms, so a deleted block is visible as a zero."""
    named = dict(model.named_parameters())
    out = {}
    for name, p in named.items():
        if name.endswith("equiv_reduce"):
            out[name] = float(p.norm())
    for prefix in sorted({k.rsplit(".", 2)[0] for k in named if k.endswith(".weight")}):
        layers = [
            k for k in named
            if k.startswith(prefix + ".") and k.endswith(".weight")
            and k[len(prefix) + 1].isdigit()
        ]
        if len(layers) < 2:
            continue
        last = sorted(layers, key=lambda s: int(s.split(".")[-2]))[-1]
        out[last] = float(named[last].norm())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?", default="configs/water_staged.yaml")
    ap.add_argument("--stages", default=None, help="comma-separated subset of stage names")
    ap.add_argument("--frames", type=int, default=120, help="validation frames per cluster")
    ap.add_argument("--checkpoint-root", default=None)
    args = ap.parse_args()

    cfg0 = load_config(args.config)
    torch.set_default_dtype(torch.float64 if cfg0.dtype == "float64" else torch.float32)
    wanted = args.stages.split(",") if args.stages else None
    stages = [s for s in cfg0.stages if wanted is None or s.name in wanted]

    paths = cfg0.data.path if isinstance(cfg0.data.path, list) else [cfg0.data.path]
    datasets = {}
    for p in paths:
        tag = next(
            (t for t in os.path.basename(p).replace("-", "_").split("_")
             if t.startswith("w") and t[1:].isdigit()),
            os.path.basename(p),
        )
        datasets[tag] = load_extxyz(p, dtype=torch.get_default_dtype())
    anchor = (
        load_extxyz(cfg0.data.monomer_path, dtype=torch.get_default_dtype())
        if cfg0.data.monomer_path else None
    )
    mbe_set = datasets.get("w3") or next(iter(datasets.values()))

    for stage in stages:
        try:
            model, cfg, state, untrained = load_stage(cfg0, stage, args.checkpoint_root)
        except FileNotFoundError as exc:
            print(f"\n=== {stage.name}: no checkpoint ({exc}) ===")
            continue

        print(f"\n{'=' * 78}\nstage {stage.name}   epoch {state.get('epoch', '?')}   "
              f"val_loss {state.get('val_loss', float('nan')):.6g}\n{'=' * 78}")

        print("\n[1] one-body residual, per fragment (kJ/mol) -- a constant here is the "
              "internal/bond gauge sliding")
        print(f"    {'cluster':>8s} {'bias':>10s} {'MAE':>10s}")
        for tag, (bias, mae) in onebody_bias(model, datasets, cfg, args.frames).items():
            print(f"    {tag:>8s} {bias:+10.4f} {mae:10.4f}")

        if anchor is not None:
            m = monomer_split(model, anchor)
            print("\n[2] monomer anchor (kJ/mol) -- watch internal+bond, not their sum")
            print(f"    internal {m['internal']:+10.2f}   bond {m['bond']:+10.2f}   "
                  f"one-body bias {m['ob_bias']:+8.4f}")
            print("\n[3] monomer polarizability eigenvalues (a0^3)")
            print(f"    predicted {np.array2string(m['alpha'], precision=3)}   "
                  f"iso {m['alpha'].mean():.3f}")
            if m["alpha_ref"] is not None:
                print(f"    reference {np.array2string(m['alpha_ref'], precision=3)}   "
                      f"iso {m['alpha_ref'].mean():.3f}")

        print("\n[4] mean |3-body| per channel (kJ/mol) -- elst must be 0, pauli/disp must not")
        for name, v in three_body(model, mbe_set, min(args.frames, 100)).items():
            flag = ""
            if name == "elst" and v > 1e-9:
                flag = "   <-- cls_elec is not two-body"
            if name in ("pauli", "disp") and v < 1e-12:
                flag = "   <-- environment residual is dead"
            print(f"    {name:>6s} {v:12.6f}{flag}")

        print("\n[5] blocks at or near zero (weight decay deletes; nothing warns)")
        # A zero-initialized readout that this checkpoint never contained reads zero because
        # it has not been trained yet, which is not the failure being looked for. Excluded and
        # counted rather than listed, so a genuinely deleted block still stands out.
        norms = {k: v for k, v in block_norms(model).items() if k not in untrained}
        skipped = len(block_norms(model)) - len(norms)
        dead = {k: v for k, v in norms.items() if v < DEAD}
        small = {k: v for k, v in norms.items() if DEAD <= v < 0.05}
        for k, v in sorted(dead.items()):
            print(f"    DEAD  {k:60s} {v:.3g}")
        for k, v in sorted(small.items()):
            print(f"    small {k:60s} {v:.3g}")
        if not dead and not small:
            print("    none")
        if skipped:
            print(f"    ({skipped} readouts not in this checkpoint, still at initialization)")


if __name__ == "__main__":
    main()
