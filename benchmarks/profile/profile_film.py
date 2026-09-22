"""M0 of the torchff port: where does a FilmModel call spend its time?

Runs the trained film model on water clusters of increasing size and reports, per structure,

* forward-only wall time split by region (neighbor list, projector/features, parameter
  network, gates, elst / pauli / disp pair terms, bonded, coupled solve) -- from
  ``torch.profiler`` ``record_function`` scopes patched around the model's own call sites,
  so the model code is untouched;
* whole-call timings: forward, forward + forces (one backward), and a training step
  (energy + force loss with ``create_graph=True`` plus its backward) -- the thing the
  kernel port has to make faster;
* peak CUDA memory of the training step.

The region split covers the **forward only**: autograd runs the backward ops outside those
scopes, so the backward cost is reported as whole-call totals. The forward split still says
which terms dominate, and the backward of a pair term scales with its forward.

Usage (see benchmarks/profile/README.md)::

    python benchmarks/profile/profile_film.py --device cuda \
        --structures benchmarks/structures/w6_mp2_avtz.xyz benchmarks/structures/w21_mp2_avtz.xyz \
                     external/torchff-lib/examples/water_216.pdb
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[2]

from rsfff import neighbors  # noqa: E402
from rsfff.ff import backend as ff_backend  # noqa: E402
from rsfff.ff.film import model as film_model  # noqa: E402
from rsfff.ff.film import bonded as film_bonded  # noqa: E402
from rsfff.md.film_driver import load_film_model, make_batch, water_fragment_index  # noqa: E402

REGIONS = [
    "neighbor_list", "projector_features", "parameter_network", "gates",
    "elst_pairs", "pauli_pairs", "disp_pairs", "bonded", "coupled_solve",
]


# --------------------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------------------

def read_water(path: str):
    """``(positions (N,3) Angstrom, Z (N,))`` in ``O H H`` fragment order, molecules whole.

    PDB boxes (the torchff examples) are unwrapped so each H sits at the minimum image of
    its O: the profile wants a gas-phase cluster cut from bulk, not a periodic system.
    """
    from ase.io import read

    atoms = read(path)
    pos = atoms.get_positions().astype(float)
    z = atoms.get_atomic_numbers().astype(int)
    if path.endswith(".pdb") and atoms.cell.volume > 0:
        cell = np.diag(atoms.cell.lengths())
        o_atoms = np.flatnonzero(z == 8)
        h_atoms = np.flatnonzero(z == 1)
        d = pos[h_atoms, None, :] - pos[None, o_atoms, :]
        d -= np.round(d / cell.diagonal()) * cell.diagonal()
        owner = np.linalg.norm(d, axis=-1).argmin(axis=1)
        pos[h_atoms] = pos[o_atoms[owner]] + d[np.arange(h_atoms.size), owner]
    frag = water_fragment_index(pos, z)
    return pos, z, frag


# --------------------------------------------------------------------------------------
# region instrumentation (monkeypatches, model code untouched)
# --------------------------------------------------------------------------------------

def _scoped(name, fn):
    def wrapped(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapped


@contextlib.contextmanager
def instrumented(model):
    """Wrap the film model's call sites in ``record_function`` scopes; restore on exit."""
    saved = {}

    def patch(obj, attr, name):
        saved[(obj, attr)] = getattr(obj, attr)
        setattr(obj, attr, _scoped(name, getattr(obj, attr)))

    patch(film_model, "union_pairs", "neighbor_list")
    patch(film_model, "union_channels", "neighbor_list")
    patch(film_model, "slater_elec_pair_energy", "elst_pairs")
    patch(film_model, "slater_pauli_pair_energy", "pauli_pairs")
    # the dispersion and bonded leaves go through rsfff.ff.backend (torch or torchff kernels)
    patch(ff_backend, "tt_dispersion", "disp_pairs")
    patch(ff_backend, "bonded_energy", "bonded")
    patch(film_bonded.BondedTopology, "geometry", "bonded")  # torch path only, not nested
    patch(film_model, "coupled_response", "coupled_solve")
    # instance-level: nn.Module.__call__ looks up self.forward, so this shadows it
    for attr, name in (("projector", "projector_features"), ("network", "parameter_network")):
        sub = getattr(model, attr)
        saved[(sub, "forward")] = sub.forward
        sub.forward = _scoped(name, sub.forward)
    saved[(model, "_gates")] = model._gates
    model._gates = _scoped("gates", model._gates)
    try:
        yield
    finally:
        for (obj, attr), fn in saved.items():
            if attr == "forward" or attr == "_gates":
                try:
                    delattr(obj, attr)
                except AttributeError:
                    setattr(obj, attr, fn)
            else:
                setattr(obj, attr, fn)


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timeit(fn, device, repeats):
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.min(times))


def make_calls(model, pos, z, frag, device, *, with_induction, force_weight=1.0, n_frames=1):
    positions = np.tile(np.asarray(pos, dtype=float), (n_frames, 1))

    def batch(requires_grad):
        b = _batch_to(make_batch(positions, z, frag, n_frames=n_frames), device)
        if requires_grad:
            b.positions.requires_grad_(True)
        return b

    def forward():
        with torch.no_grad():
            return model(batch(False), with_induction=with_induction)

    def forward_forces():
        b = batch(True)
        out = model(b, with_induction=with_induction)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions)
        return out, g

    def train_step():
        model.zero_grad(set_to_none=True)
        b = batch(True)
        out = model(b, with_induction=with_induction)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=True)
        loss = out.energy.pow(2).mean() + force_weight * g.pow(2).sum(-1).mean()
        loss.backward()
        return loss

    return forward, forward_forces, train_step


def _batch_to(b, device):
    from dataclasses import fields, replace
    kw = {}
    for f in fields(b):
        v = getattr(b, f.name)
        if torch.is_tensor(v):
            kw[f.name] = v.to(device)
        elif isinstance(v, dict):
            kw[f.name] = {k: (x.to(device) if torch.is_tensor(x) else x) for k, x in v.items()}
    return replace(b, **kw)


def _activities(device):
    return [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])


def profile_regions(model, forward, device, repeats):
    with instrumented(model):
        forward()  # warm caches (pair enumerations, compiled paths)
        _sync(device)
        with profile(activities=_activities(device), record_shapes=False) as prof:
            for _ in range(repeats):
                forward()
                _sync(device)
    rows = {}
    for ev in prof.key_averages():
        if ev.key in REGIONS:
            cuda_us = getattr(ev, "device_time_total", None)
            if cuda_us is None:
                cuda_us = getattr(ev, "cuda_time_total", 0.0)
            rows[ev.key] = dict(
                cpu_ms=ev.cpu_time_total / 1e3 / repeats,
                cuda_ms=float(cuda_us) / 1e3 / repeats,
                calls=ev.count // repeats,
            )
    return rows, prof


def write_op_summary(model, fn, device, path, *, top=60):
    """One profiled call of ``fn``: per-op table (text, a few KB) and a gzipped chrome trace.

    Kept separate from :func:`profile_regions` so the trace holds a single call -- the
    5-repeat traces were ~75 MB each, which GitHub warns about; one call gzipped is ~1-3 MB
    and ``*_trace.json.gz`` is gitignored anyway. The text table is what to commit.
    """
    with instrumented(model):
        fn()
        _sync(device)
        with profile(activities=_activities(device), record_shapes=False) as prof:
            fn()
            _sync(device)
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    try:
        table = prof.key_averages().table(sort_by=sort_by, row_limit=top)
    except Exception:  # older torch spells the cuda column differently
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top)
    n_events = sum(ev.count for ev in prof.key_averages())
    with open(path.with_suffix(".txt"), "w") as fh:
        fh.write(f"# {path.stem}: one call, {n_events} op events, sorted by {sort_by}\n")
        fh.write(table)
    prof.export_chrome_trace(str(path) + "_trace.json.gz")
    return n_events


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints/water_film_full/best.pt"))
    ap.add_argument("--structures", nargs="+", default=[
        str(REPO_ROOT / "benchmarks/structures/w6_mp2_avtz.xyz"),
        str(REPO_ROOT / "benchmarks/structures/w12_mp2_avtz.xyz"),
        str(REPO_ROOT / "benchmarks/structures/w21_mp2_avtz.xyz"),
        str(REPO_ROOT / "external/torchff-lib/examples/water_216.pdb"),
    ])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-induction", action="store_true")
    ap.add_argument("--max-neighbors", type=int, default=1024,
                    help="override FilmModel.max_num_neighbors (12 A in bulk water is ~720)")
    ap.add_argument("--skip-train-step", action="store_true")
    ap.add_argument("--frames", type=int, default=1,
                    help="replicate each structure this many times in one batch (training batches "
                         "many frames, so per-call overhead is amortised; 1 = single frame)")
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks/profile/results"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--trace", action="store_true",
                    help="also write a per-op table (.txt) and a gzipped one-call chrome trace per structure")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, config = load_film_model(args.checkpoint, device=str(device))
    model.max_num_neighbors = int(args.max_neighbors)
    with_induction = not args.no_induction
    tag = args.tag or (torch.cuda.get_device_name(device).replace(" ", "_") if device.type == "cuda"
                       else platform.processor() or "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in args.structures:
        pos, z, frag = read_water(path)
        name = Path(path).stem
        n_atoms = int(z.size)
        print(f"\n== {name}: {n_atoms} atoms, {int(frag.max()) + 1} waters, device {device}", flush=True)
        neighbors.reset_cap_events()
        forward, forward_forces, train_step = make_calls(
            model, pos, z, frag, device, with_induction=with_induction, n_frames=args.frames
        )
        rec = dict(structure=name, n_atoms=n_atoms, n_waters=int(frag.max()) + 1, frames=args.frames,
                   induction=with_induction, device=str(device), tag=tag,
                   ff_backend=ff_backend.active_backend(torch.zeros(1, device=device)))

        # -- forward region split --------------------------------------------------------
        regions, prof = profile_regions(model, forward, device, args.repeats)
        out = forward()
        rec["n_pairs"] = int(out.pair_index.shape[1])
        rec["n_pairs_intra"] = int(out.is_intra.sum())
        if out.solver:
            rec["cg_iters"] = int(out.solver["ind"][0])
        rec["regions"] = regions
        if args.trace:
            rec["n_op_events_forward"] = write_op_summary(
                model, forward, device, out_dir / f"{tag}_{name}_forward"
            )

        # -- whole-call timings ----------------------------------------------------------
        for _ in range(2):
            forward_forces()
        rec["forward_ms"] = _timeit(forward, device, args.repeats)[0] * 1e3
        rec["forward_forces_ms"] = _timeit(forward_forces, device, args.repeats)[0] * 1e3
        if not args.skip_train_step:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                train_step()
                rec["train_step_ms"] = _timeit(train_step, device, args.repeats)[0] * 1e3
                if device.type == "cuda":
                    rec["train_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            except torch.cuda.OutOfMemoryError:
                rec["train_step_ms"] = float("nan")
                rec["train_oom"] = True
                torch.cuda.empty_cache()
        rec["cap_events"] = dict(neighbors.CAP_EVENTS)
        results.append(rec)
        print(_format_row(rec), flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    frames = f"_f{args.frames}" if args.frames != 1 else ""
    be = ff_backend.active_backend(torch.zeros(1, device=device))
    base = out_dir / f"{tag}_{'ind' if with_induction else 'noind'}{frames}_{be}_{stamp}"
    with open(base.with_suffix(".json"), "w") as fh:
        json.dump(dict(
            checkpoint=args.checkpoint, torch=torch.__version__, dtype=str(torch.get_default_dtype()),
            python=sys.version.split()[0], host=platform.node(), results=results,
        ), fh, indent=2)
    with open(base.with_suffix(".md"), "w") as fh:
        fh.write(_format_table(results))
    print(f"\nwrote {base.with_suffix('.md')} and .json")
    print(_format_table(results))


def _format_row(r):
    reg = r["regions"]
    key = "cuda_ms" if r["device"].startswith("cuda") else "cpu_ms"
    parts = " ".join(f"{k}={reg[k][key]:.1f}" for k in REGIONS if k in reg)
    return (f"  pairs={r['n_pairs']} fwd={r['forward_ms']:.1f}ms fwd+F={r['forward_forces_ms']:.1f}ms "
            f"train={r.get('train_step_ms', float('nan')):.1f}ms | forward split ({key}): {parts}")


def _format_table(results):
    if not results:
        return ""
    key = "cuda_ms" if results[0]["device"].startswith("cuda") else "cpu_ms"
    cols = ["structure", "ff_backend", "frames", "n_atoms", "n_pairs", "forward_ms", "forward_forces_ms", "train_step_ms",
            "train_peak_gb"] + REGIONS
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        vals = []
        for c in cols:
            if c in REGIONS:
                v = r["regions"].get(c, {}).get(key, float("nan"))
            else:
                v = r.get(c, "")
            vals.append(f"{v:.1f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    note = (f"\nForward split column unit: {key} per call (forward only; backward is in the "
            "whole-call columns). train_step = E+F loss with create_graph=True + backward.\n")
    return "\n".join(lines) + "\n" + note


if __name__ == "__main__":
    main()
