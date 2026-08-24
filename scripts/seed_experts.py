"""Copy a trained expert's weights into the other experts of a multi-expert checkpoint.

Usage
-----
    python scripts/seed_experts.py \
        --from checkpoints/water_expert_full/best.pt \
        --out  checkpoints/seed_three_experts.pt \
        --source H2O --targets H3O HO

``checkpoints/water_expert_full/best.pt`` holds one expert, ``H2O``, fitted on the neutral
water clusters. The joint H2O/H3O+/OH- fit wants three, and starting the two new ones from
random weights throws away the closest thing to a prior that exists: hydronium and hydroxide
are water with a proton added or removed, they are made of the same two elements, and every
expert is built from the same config so the tensors have identical shapes.

So this is a **key rename**, not a conversion. Every ``experts.experts.<source>.*`` tensor is
written out again under each target's key, and everything else in the checkpoint -- the
featurizer, the reference energies, the optimizer state if there is one -- is carried through
untouched.

Two things it does not do, deliberately:

* It does not tie the copies. They are independent parameters from the first step, which is
  the point: the three experts must be free to diverge, and what the fit measures is *how
  far* they do.
* It does not touch ``train.init_from``. Point the config at the file this writes.

``rsfff.train.term_loop.warm_start`` will still skip anything the new model does not have and
report what it loaded, so a stale or partial checkpoint degrades to "some tensors loaded"
rather than to a crash. That makes it worth reading the counts this prints: if the source
prefix matched nothing, the run would start from scratch and say so only in one line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PREFIX = "experts.experts."


def seed(state: dict, source: str, targets: list[str]) -> tuple[dict, int]:
    """Return a new state dict with ``source``'s expert tensors copied under each target."""
    head = f"{PREFIX}{source}."
    donor = {k[len(head):]: v for k, v in state.items() if k.startswith(head)}
    if not donor:
        prefixes = sorted(
            {k[len(PREFIX):].split(".", 1)[0] for k in state if k.startswith(PREFIX)}
        )
        raise SystemExit(
            f"no tensors under {head!r}; the checkpoint holds expert(s) {prefixes or 'none'}"
        )

    out = dict(state)
    n_copied = 0
    for target in targets:
        if target == source:
            continue
        for name, value in donor.items():
            out[f"{PREFIX}{target}.{name}"] = value.clone()
            n_copied += 1
    return out, n_copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="H2O", help="the expert key to copy from")
    ap.add_argument(
        "--targets", nargs="+", default=["H3O", "HO"],
        help="expert keys to copy to (Hill order over the data's elements: HO, not OH)",
    )
    args = ap.parse_args(argv)

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    seeded, n_copied = seed(state, args.source, args.targets)

    if "model_state" in ckpt:
        ckpt = {**ckpt, "model_state": seeded}
    else:
        ckpt = seeded
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)

    n_donor = n_copied // max(len(set(args.targets) - {args.source}), 1)
    print(
        f"{args.src} -> {args.out}: copied {n_donor} tensors from {args.source!r} into "
        f"{', '.join(repr(t) for t in args.targets)} ({n_copied} written, "
        f"{len(seeded)} total in the checkpoint)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
