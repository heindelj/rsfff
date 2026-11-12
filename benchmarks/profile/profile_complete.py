#!/usr/bin/env python
"""Phase timings of the complete (ion) model: where a training step and an MD step spend their time.

For rock-salt cubes of n^3 ions (random displacements, B frames per batch) it times, with the device synchronized
around every measurement:

  inference      precompute + forward, no gradients
  energy_train   forward + backward of an energy loss on a precomputed batch (what a cached training frame costs)
  forces         precompute + forward + dE/dx (one MD step's work)
  force_train    the same with create_graph=True, then the backward of a force loss (double backward)
  polar          forward + backward of the 7-field stencil polarizability on a precomputed batch
  md_api         CompletePotential.evaluate for 1 frame and for B frames (clusters; --periodic adds the cell of even n)

and splits the precompute + forward of the forces step by region (inclusive times: a region contains the regions it
calls, e.g. xpol_solve contains xpol_pairs and torch.linalg.solve; the backward is not split).

    python benchmarks/profile/profile_complete.py RUN_DIR_OR_MODEL_PT --device cpu --sizes 2,3 --batch 8 --reps 3
    python benchmarks/profile/profile_complete.py RUN_DIR --device cuda --sizes 2,3,4,5 --batch 16 --periodic \\
        --out benchmarks/profile/results/complete_gpu.json

RUN_DIR is a train_ionct --model complete run (it holds model.pt); pass --params if the checkpoint's parameter table is
not found from the working directory (run from the directory the training ran in, e.g. ~/research/rsfff).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict

import numpy as np
import torch

F_SET = 0.002          # the trainer's stencil field (a.u.)


class Clock:
    """Wall clock with the device synchronized, and inclusive per-region accumulators filled by :meth:`wrap`."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.acc = defaultdict(lambda: [0.0, 0])
        self.on = False

    def now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def wrap(self, owner, name, label=None):
        """Replace ``owner.name`` by a timed version (``label``: a string, or a function of the call's args)."""
        if not hasattr(owner, name):
            return False
        fn = getattr(owner, name)
        lab = label or name
        clock = self

        def timed(*a, **k):
            if not clock.on:
                return fn(*a, **k)
            t = clock.now()
            try:
                return fn(*a, **k)
            finally:
                rec = clock.acc[lab(a) if callable(lab) else lab]
                rec[0] += clock.now() - t
                rec[1] += 1

        setattr(owner, name, timed)
        return True

    def reset(self):
        self.acc.clear()


def rocksalt(n, d, symbols, jitter, rng):
    g = np.arange(n)
    I, J, K = np.meshgrid(g, g, g, indexing="ij")
    par = ((I + J + K) % 2).reshape(-1)
    pos = np.stack([I, J, K], -1).reshape(-1, 3) * d + rng.normal(scale=jitter, size=(n ** 3, 3))
    return [symbols[p] for p in par], pos


def tensors(frames, dev, zmap, grad):
    Z = torch.tensor([zmap[s] for sym, _ in frames for s in sym], device=dev)
    fr = torch.cat([torch.full((len(sym),), k, dtype=torch.long) for k, (sym, _) in enumerate(frames)]).to(dev)
    pos = torch.tensor(np.concatenate([X for _, X in frames]), dtype=torch.float64, device=dev, requires_grad=grad)
    return pos, Z, fr


def instrument(clock, model):
    """Timed versions of the forward's regions. Returns the names that were found."""
    import rsfff.ff.complete.channels as channels_mod
    import rsfff.ff.complete.gram as gram_mod
    import rsfff.ff.complete.model as model_mod
    import rsfff.ff.complete.tabulate as tab_mod
    import rsfff.ff.ionct.kernels as kernels_mod

    found = []
    for name in ("env_prepare", "env_raw_batch", "_env_extra", "diffuse_gated_invariants", "frozen_terms", "alpha_inverse",
                 "hyper_legendre", "xpol_solve", "xpol_pairs", "xpol_kappa", "xpol_gram_pairs", "xpol_perp_weights",
                 "flux_operator", "flux_solve", "xpol_solve_periodic"):
        if clock.wrap(model, name, f"model.{name}"):
            found.append(name)
    if getattr(model, "env", None) is not None:
        clock.wrap(model.env, "forward", "env network (SharedEnvironment.forward)")
    for mod, names in ((model_mod, ("xpol_overlap_map", "flux_pair_terms", "permanent_electrostatics",
                                    "induction_matrix_blocks", "solve_induction_blocks", "tt_damped_c6_energy",
                                    "slater_two_center_damp_unequal")),
                       (tab_mod, ("_basis_tensors",)), (gram_mod, ("gram_blocks",)),
                       (channels_mod, ("channels_from_pairs",)), (kernels_mod, ("all_pairs",))):
        for name in names:
            clock.wrap(mod, name, f"{mod.__name__.rsplit('.', 1)[-1]}.{name}")
    clock.wrap(tab_mod.RadialTable, "__call__", lambda a: f"table lookup: {type(a[0]).__name__}")
    clock.wrap(torch.linalg, "solve", "torch.linalg.solve (forward only)")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a train_ionct --model complete run directory, model.pt or tarball")
    ap.add_argument("--params", default=None, help="the ion parameter table (default: the checkpoint's)")
    ap.add_argument("--table-cache", default=None, help="directory of the radial-table cache (e.g. train/salts/cache)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", default="2,3,4", help="cube edges n: clusters of n^3 ions")
    ap.add_argument("--batch", type=int, default=16, help="frames per batch")
    ap.add_argument("--salt", default="Na,Cl", help="cation,anion")
    ap.add_argument("--spacing", type=float, default=2.82, help="nearest-neighbour distance (Angstrom)")
    ap.add_argument("--jitter", type=float, default=0.10, help="rms displacement (Angstrom)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--periodic", action="store_true", help="md_api also on the periodic cell (even n only)")
    ap.add_argument("--skip", default="", help="comma-separated steps to skip (e.g. force_train,polar)")
    ap.add_argument("--profiler", action="store_true", help="also print torch.profiler's top ops of one forces step")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the numbers as json")
    a = ap.parse_args()

    from rsfff.ff.complete import batched_precompute
    from rsfff.ff.ionct.params import ATOMIC_NUMBER
    from rsfff.md.ions import CompletePotential, load_complete_model

    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    model, tab, table, chan = load_complete_model(a.checkpoint, a.params, a.device, a.table_cache)
    pot = CompletePotential(model, tab, table, chan)
    clock = Clock(dev)
    instrument(clock, model)
    symbols = a.salt.split(",")
    skip = set(x for x in a.skip.split(",") if x)
    rng = np.random.default_rng(a.seed)
    info = {"torch": torch.__version__, "device": torch.cuda.get_device_name(dev) if cuda else platform.processor(),
            "threads": torch.get_num_threads(), "batch": a.batch, "reps": a.reps, "checkpoint": str(a.checkpoint),
            "xpol_form": getattr(model, "xpol_form", None), "ct_model": getattr(model, "ct_model", None)}
    print(json.dumps(info))
    results = {"info": info, "sizes": {}}

    def pre(pos, Z, fr, **kw):
        return batched_precompute(pos, Z, fr, tab, table, chan, inv_form=model.inv_form, **kw)

    for n in [int(x) for x in a.sizes.split(",")]:
        frames = [rocksalt(n, a.spacing, symbols, a.jitter, rng) for _ in range(a.batch)]
        N = n ** 3
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            bt_fixed = pre(*tensors(frames, dev, ATOMIC_NUMBER, False))
            bt_polar = pre(*tensors([f for f in frames for _ in range(7)], dev, ATOMIC_NUMBER, False))
        field = torch.zeros(7, 3, dtype=torch.float64, device=dev)
        for k in range(3):
            field[1 + 2 * k, k], field[2 + 2 * k, k] = F_SET, -F_SET
        field = field.repeat(a.batch, 1)

        def inference():
            with torch.no_grad():
                t0 = clock.now()
                bt = pre(*tensors(frames, dev, ATOMIC_NUMBER, False))
                t1 = clock.now()
                model(bt)
                return {"precompute": t1 - t0, "forward": clock.now() - t1}

        def energy_train():
            model.zero_grad(set_to_none=True)
            t0 = clock.now()
            o = model(bt_fixed)
            loss = sum((o[k] ** 2).sum() for k in ("e_pol", "e_ct", "e_pauli", "e_disp"))
            t1 = clock.now()
            loss.backward()
            return {"forward": t1 - t0, "backward": clock.now() - t1}

        def forces(train=False):
            model.zero_grad(set_to_none=True)
            pos, Z, fr = tensors(frames, dev, ATOMIC_NUMBER, True)
            t0 = clock.now()
            bt = pre(pos, Z, fr)
            t1 = clock.now()
            E = model(bt)["e_int"]
            t2 = clock.now()
            (g,) = torch.autograd.grad(E.sum(), pos, create_graph=train)
            t3 = clock.now()
            out = {"precompute": t1 - t0, "forward": t2 - t1, "dE/dx": t3 - t2}
            if train:
                ((g ** 2).sum() + (E ** 2).sum()).backward()
                out["backward"] = clock.now() - t3
            return out

        def polar():
            model.zero_grad(set_to_none=True)
            t0 = clock.now()
            o = model(bt_polar, field=field)
            e = torch.stack([o["e_pol"], o["e_ct"]], -1).reshape(a.batch, 7, 2)
            al = -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / F_SET ** 2
            t1 = clock.now()
            (al ** 2).sum().backward()
            return {"forward": t1 - t0, "backward": clock.now() - t1}

        numbers = [[ATOMIC_NUMBER[s] for s in sym] for sym, _ in frames]
        positions = [X for _, X in frames]

        def md_one():
            t0 = clock.now()
            pot.evaluate(numbers[:1], positions[:1])
            return {"total": clock.now() - t0}

        def md_batch():
            t0 = clock.now()
            pot.evaluate(numbers, positions)
            return {"total": clock.now() - t0}

        cell = np.eye(3) * n * a.spacing

        def md_periodic():
            t0 = clock.now()
            pot.evaluate(numbers[:1], positions[:1], cells=[cell])
            return {"total": clock.now() - t0, "cg_iters": float(pot.last.get("cg_iters", 0))}

        steps = [("inference", inference), ("energy_train", energy_train), ("forces", forces),
                 ("force_train", lambda: forces(True)), ("polar", polar), ("md_api_1frame", md_one),
                 ("md_api_batch", md_batch)]
        if a.periodic and n % 2 == 0:
            steps.append(("md_api_periodic_1frame", md_periodic))
        size = {"n_ions": N, "steps": {}}
        print(f"\n=== {N} ions x {a.batch} frames ({symbols[0]}{symbols[1]} cube, edge {n}) ===")
        for name, fn in steps:
            if name in skip or name.split("_1frame")[0] in skip:
                continue
            try:
                for _ in range(a.warmup):
                    fn()
                clock.reset()
                clock.on = name == "forces"
                runs = [fn() for _ in range(a.reps)]
                clock.on = False
            except Exception as err:                       # e.g. flux CT has no periodic path yet; out of memory
                clock.on = False
                size["steps"][name] = {"error": f"{type(err).__name__}: {err}"}
                print(f"{name:24s} FAILED  {type(err).__name__}: {str(err)[:140]}")
                if cuda:
                    torch.cuda.empty_cache()
                continue
            parts = {k: [r[k] for r in runs] for k in runs[0]}
            tot = [sum(v for k, v in r.items() if k != "cg_iters") for r in runs]
            rec = {"total_ms": 1e3 * float(np.mean(tot)), "min_ms": 1e3 * float(np.min(tot)),
                   "parts_ms": {k: (1e3 if k != "cg_iters" else 1.0) * float(np.mean(v)) for k, v in parts.items()}}
            size["steps"][name] = rec
            ptxt = "  ".join(f"{k} {v:.1f}" for k, v in rec["parts_ms"].items())
            print(f"{name:24s} {rec['total_ms']:9.1f} ms (min {rec['min_ms']:.1f})   {ptxt}")
            if name == "forces":
                base = 1e3 * float(np.mean([r["precompute"] + r["forward"] for r in runs]))
                rows = sorted(((1e3 * t / a.reps, c / a.reps, lab) for lab, (t, c) in clock.acc.items()), reverse=True)
                size["regions_ms"] = {lab: {"ms": ms, "calls": c} for ms, c, lab in rows}
                print(f"    regions of precompute + forward ({base:.1f} ms; inclusive, so they overlap):")
                for ms, c, lab in rows:
                    print(f"      {ms:9.2f} ms  {100 * ms / base:5.1f} %  x{c:<4.0f} {lab}")
        if cuda:
            size["peak_memory_MB"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
            print(f"peak GPU memory {size['peak_memory_MB']:.0f} MB")
        results["sizes"][str(n)] = size
        if a.profiler:
            from torch.profiler import ProfilerActivity, profile
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
            with profile(activities=acts) as prof:
                forces()
            print(prof.key_averages().table(sort_by="self_cuda_time_total" if cuda else "self_cpu_time_total",
                                            row_limit=30))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
