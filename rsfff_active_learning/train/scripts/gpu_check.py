#!/usr/bin/env python3
"""Does the film model train correctly on this GPU? Answers in about a minute, queues nothing.

    python scripts/gpu_check.py                       # on a GPU node, after sourcing the env
    python scripts/gpu_check.py --frames 32 --steps 20

Checks, in order, each printed as PASS / FAIL / WARN:

1. environment   torch, its CUDA build, the device, e3nn, rsfff's location, which neighbor
                 list backend is active.
2. neighbors     ``build_radius_graph`` on the GPU returns the same edges as on the CPU. A
                 CPU-only torch_cluster (``FORCE_ONLY_CPU=1``, as the NERSC install notes
                 build it) cannot take CUDA tensors; rsfff must fall back to plain torch.
3. placement     after ``.to(cuda)`` every parameter and buffer is on the GPU.
4. parity        one full-stage loss evaluation -- cluster EDA channels + induction CG solve
                 + forces (a second-order backward) + fragment, monomer-anchor and
                 regularizer streams -- from *identical* weights on CPU and on GPU, in
                 float64. Loss terms and every parameter gradient must agree to ``--rtol``.
                 This is the check that the GPU is computing the same fit, not just a fit.
5. descent       ``--steps`` Adam steps on one minibatch on the GPU; the loss must fall.
6. timing        seconds per training step on each device, and a rough epoch estimate for
                 the full w2-w5 set.

Exit status is non-zero if any required check failed.
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore", message="torch_cluster is not installed")

import torch  # noqa: E402
import yaml  # noqa: E402

from train_committee import DEFAULT_CONFIG, apply_subset, resolve_paths  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"[{status:4s}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def build(config, device):
    """Everything ``rsfff.train.train_film._train_once`` builds, on ``device``."""
    from rsfff.mlip.heads import env_parameters  # noqa: F401  (import check)
    from rsfff.train.build_film import build_film_model
    from rsfff.train.data import (fragment_view, load_cluster_datasets,
                                  load_reference_energies, split_indices_grouped)
    from rsfff.train.train_expert import load_anchor_datasets
    from rsfff.train.train_film import FilmStreams

    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    clusters = load_cluster_datasets(config.data.path, dtype=dtype,
                                     fragmentations=config.data.fragmentations)
    types = tuple(clusters.unique_atomic_numbers)
    train_idx, val_idx = split_indices_grouped(clusters._group_id,
                                               config.data.holdout_fraction, config.data.seed)
    fragments = fragment_view(clusters, train_idx)
    anchors = load_anchor_datasets(config.data.monomer_path, dtype=dtype)
    refs = load_reference_energies(config.data.reference_energies, types).to(dtype)

    def make():
        torch.manual_seed(config.train.seed)
        return build_film_model(config.features, config.film, types, refs).to(dtype=dtype)

    return clusters, train_idx, fragments, anchors, make, dtype


def streams_for(model, device, config, fragments, anchors):
    from rsfff.train.train_film import FilmStreams
    return FilmStreams(
        model, device, fragment_dataset=fragments,
        fragment_batch_size=config.film.fragment_batch_size,
        anchor_datasets=anchors, anchor_batch_size=config.film.anchor_batch_size,
        anchor_force_every=1, seed=config.data.seed,
    )


def loss_and_grads(model, streams, clusters, idx, config, device):
    """One training-mode loss with every term on (forces included) and its gradients."""
    from rsfff.train.train_film import film_fit
    model.train(True)
    model.zero_grad(set_to_none=True)
    batch = clusters.flat_batch(idx).to(device)
    batch.positions.requires_grad_(True)
    with torch.enable_grad():
        out = model(batch)
        loss, metrics, _ = film_fit(out, batch, config, training=True, with_forces=True)
        extra = streams.penalties(out, batch, config)
        terms = {"fit": loss.detach().clone()}
        terms.update({k: v.detach().clone() for k, v in extra.items()})
        for v in extra.values():
            loss = loss + v
        loss.backward()
    grads = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None)
             for n, p in model.named_parameters()}
    return float(loss.detach()), {k: float(v) for k, v in terms.items()}, grads, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--frames", type=int, default=48, help="frames per data file to load")
    ap.add_argument("--batch", type=int, default=16, help="cluster minibatch size")
    ap.add_argument("--steps", type=int, default=15, help="Adam steps for the descent check")
    ap.add_argument("--rtol", type=float, default=1e-7, help="loss terms, relative")
    ap.add_argument("--grad-rtol", type=float, default=1e-5,
                    help="worst gradient element relative to that tensor's largest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run with --device cpu as the 'GPU' (to test this script itself)")
    args = ap.parse_args()

    # --- 1. environment ----------------------------------------------------------------------
    import rsfff
    import rsfff.neighbors as nb
    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), python "
          f"{sys.version.split()[0]}, rsfff at {Path(rsfff.__file__).parent}")
    try:
        import e3nn
        print(f"e3nn {e3nn.__version__}; neighbor backend: {nb.BACKEND}")
    except ImportError:
        report("FAIL", "e3nn import")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            report("FAIL", "cuda available",
                   f"torch.cuda.is_available() is False (CUDA build {torch.version.cuda}). "
                   "A CPU-only torch in this env? See README 'Environment'.")
            return summary()
        props = torch.cuda.get_device_properties(dev)
        report("PASS", "cuda available", f"{torch.cuda.device_count()} device(s); "
               f"{props.name}, {props.total_memory / 2**30:.0f} GiB, sm_{props.major}{props.minor}")
    elif not args.allow_cpu:
        report("FAIL", "device", f"--device {args.device} is not a GPU (use --allow-cpu)")
        return summary()

    # --- 2. neighbor list on the device --------------------------------------------------------
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(60, 3, generator=g, dtype=torch.float64) * 8.0
    bidx = torch.zeros(60, dtype=torch.long)
    bidx[30:] = 1
    try:
        e_cpu = nb.build_radius_graph(pos, 5.0, bidx, context="gpu_check")
        e_dev = nb.build_radius_graph(pos.to(dev), 5.0, bidx.to(dev), context="gpu_check")
        a = sorted(map(tuple, e_cpu.t().tolist()))
        b = sorted(map(tuple, e_dev.cpu().t().tolist()))
        same = a == b and e_dev.device.type == dev.type
        report("PASS" if same else "FAIL", "neighbor list on device",
               f"{len(a)} edges, backend {nb.BACKEND}"
               + ("" if same else f"; cpu {len(a)} vs device {len(b)} edges"))
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "neighbor list on device",
               f"{type(exc).__name__}: {exc}. A CPU-only torch_cluster? Update rsfff "
               "(neighbors.py falls back to plain torch) or uninstall torch_cluster.")
        return summary()

    # --- config: the full stage, a small slice of the real data --------------------------------
    from rsfff.train.config import load_config, stage_config
    tree = yaml.safe_load(args.config.read_text())
    resolve_paths(tree, args.config.resolve().parent)
    scratch = Path(os.environ.get("SCRATCH", "/tmp")) / f"rsfff_gpu_check_{os.getpid()}"
    atexit.register(shutil.rmtree, scratch, True)
    from check_data import frames as _frames
    n_full = sum(1 for p in tree["data"]["path"] for _ in _frames(Path(p)))
    apply_subset(tree, args.frames, scratch)
    tree["device"] = "cpu"
    tmp = scratch / "config.yaml"
    tmp.write_text(yaml.safe_dump(tree, sort_keys=False))
    config = load_config(tmp)
    if config.stages:
        config = stage_config(config, config.stages[-1], "")
    print(f"config: {args.config} (stage '{config.run_name}'), {args.frames} frames/file, "
          f"dtype {config.dtype}, induction {config.film.induction}")

    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    clusters, train_idx, fragments, anchors, make, dtype = build(config, "cpu")
    model_cpu = make()
    model_dev = make()            # a second instance (the model does not deepcopy) ...
    model_dev.load_state_dict(model_cpu.state_dict())   # ... with provably identical weights
    model_dev = model_dev.to(dev)
    print(f"{len(clusters)} cluster frames, {len(fragments)} fragment views, "
          f"{sum(len(a) for a in anchors)} anchor frames; "
          f"{sum(p.numel() for p in model_cpu.parameters())} parameters")

    # --- 3. placement ---------------------------------------------------------------------------
    stray = [n for n, t in list(model_dev.named_parameters()) + list(model_dev.named_buffers())
             if t.device.type != dev.type]
    report("PASS" if not stray else "FAIL", "parameters/buffers on device",
           "" if not stray else f"{len(stray)} left behind, e.g. {stray[:3]}")

    # --- 4. parity --------------------------------------------------------------------------------
    idx = train_idx[: args.batch]
    s_cpu = streams_for(model_cpu, torch.device("cpu"), config, fragments, anchors)
    s_dev = streams_for(model_dev, dev, config, fragments, anchors)
    try:
        t0 = time.time()
        l_cpu, t_cpu, g_cpu, m_cpu = loss_and_grads(model_cpu, s_cpu, clusters, idx, config, "cpu")
        t_c = time.time() - t0
        loss_and_grads(model_dev, s_dev, clusters, idx, config, dev)   # warm-up (kernels, e3nn)
        s_dev = streams_for(model_dev, dev, config, fragments, anchors)  # same draws as CPU
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        l_dev, t_dev, g_dev, m_dev = loss_and_grads(model_dev, s_dev, clusters, idx, config, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_d = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report("FAIL", "loss + backward on device", f"{type(exc).__name__}: {exc}")
        return summary()
    report("PASS", "loss + backward on device", f"loss {l_dev:.6e}")

    worst_term = max(abs(t_dev[k] - t_cpu[k]) / max(abs(t_cpu[k]), 1e-30) for k in t_cpu)
    bad_terms = {k: (t_cpu[k], t_dev[k]) for k in t_cpu
                 if abs(t_dev[k] - t_cpu[k]) > args.rtol * max(abs(t_cpu[k]), 1e-12)}
    report("PASS" if not bad_terms else "FAIL", "loss terms CPU == GPU",
           f"{len(t_cpu)} terms, worst rel diff {worst_term:.1e}"
           + ("" if not bad_terms else f"; differing: {bad_terms}"))

    # Each tensor is compared against its own largest element, but never against less than
    # 1e-10 of the largest gradient anywhere. Some gradients are zero analytically and only
    # round-off survives -- the permanent charge head's output bias, whose uniform shift the
    # exact per-fragment charge projection removes, comes out ~1e-11 against a global ~1e8 --
    # and two devices' round-off relative to itself is O(1), which is not a disagreement.
    top = max((float(g.abs().max()) for g in g_cpu.values() if g is not None), default=0.0)
    floor = max(1e-10 * top, 1e-30)
    worst, worst_name, missing = 0.0, "", []
    for name, gc in g_cpu.items():
        gd = g_dev[name]
        if (gc is None) != (gd is None):
            missing.append(name)
            continue
        if gc is None:
            continue
        scale = max(float(gc.abs().max()), floor)
        rel = float((gd - gc).abs().max()) / scale
        if rel > worst:
            worst, worst_name = rel, name
    n_grads = sum(g is not None for g in g_cpu.values())
    ok = worst <= args.grad_rtol and not missing   # looser: CG tolerance, atomic sums
    report("PASS" if ok else "FAIL", "parameter gradients CPU == GPU",
           f"{n_grads} tensors, worst rel diff {worst:.1e} ({worst_name})"
           + (f"; grad present on one side only: {missing[:3]}" if missing else ""))
    if "cg_ind" in m_dev:
        report("PASS" if m_dev.get("cg_fail", 0) == 0 else "FAIL", "induction CG on device",
               f"{m_dev['cg_ind']:.0f} iterations (cpu {m_cpu.get('cg_ind', float('nan')):.0f}),"
               f" {m_dev.get('cg_fail', 0):.0f} failures")

    # --- 5. descent -----------------------------------------------------------------------------
    opt = torch.optim.Adam(model_dev.parameters(), lr=config.train.learning_rate)
    s_dev = streams_for(model_dev, dev, config, fragments, anchors)
    from rsfff.train.train_film import film_fit
    losses = []
    t0 = time.time()
    for _ in range(args.steps):
        batch = clusters.flat_batch(idx).to(dev)
        batch.positions.requires_grad_(True)
        model_dev.train(True)
        out = model_dev(batch)
        loss, _, _ = film_fit(out, batch, config, training=True, with_forces=True)
        for v in s_dev.penalties(out, batch, config).values():
            loss = loss + v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_dev.parameters(), config.train.grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_step = (time.time() - t0) / max(args.steps, 1)
    finite = all(math.isfinite(v) for v in losses)
    fell = finite and losses[-1] < 0.5 * losses[0]
    report("PASS" if fell else "FAIL", "loss falls on device",
           f"{losses[0]:.3e} -> {losses[-1]:.3e} over {args.steps} steps"
           + ("" if finite else " (non-finite!)"))

    # --- 6. timing ------------------------------------------------------------------------------
    n_train_full = int((1 - config.data.holdout_fraction) * n_full)
    steps_per_epoch = math.ceil(n_train_full / config.train.batch_size)
    scale = config.train.batch_size / max(len(idx), 1)
    print(f"\ntiming (batch {len(idx)}, forces on every step, float64):")
    print(f"  cpu     {t_c:7.2f} s/step  ({torch.get_num_threads()} threads)")
    print(f"  {dev.type:7s} {t_d:7.2f} s/step (parity step), {t_step:.2f} s/step (Adam loop)")
    if dev.type == "cuda":
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
    est = t_step * scale * steps_per_epoch
    print(f"  rough full-data epoch ({n_train_full} train frames, batch {config.train.batch_size}): "
          f"~{est / 60:.1f} min "
          f"(linear scaling from batch {len(idx)}; forces are every "
          f"{config.film.force_every}nd step in the real fit, so this is pessimistic)")
    return summary()


def summary() -> int:
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
          + (f"; FAILED: {', '.join(r[1] for r in failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
