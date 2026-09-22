"""M0 of the torchff port: where does a FilmModel call spend its time?

Runs the trained film model on water clusters of increasing size and reports, per structure,

* forward-only wall time split by region (neighbor list, projector/features, parameter
  network, gates, elst / pauli / disp pair terms, bonded, coupled solve) -- from
  ``torch.profiler`` ``record_function`` scopes patched around the model's own call sites,
  so the model code is untouched;
* whole-call timings: forward, forward + forces (one backward), and a training step
  (energy + force loss with ``create_graph=True`` plus its backward) -- the thing the
  kernel port has to make faster;
* peak CUDA memory of the training step;
* the training step split three ways -- forward / first backward (dE/dR, ``create_graph``) /
  second backward (the loss) -- as synced wall time per phase, and the kernel time of every
  phase attributed back to the forward region whose autograd nodes ran it (see
  :func:`attribute_train_step`), so the backward-dominated training step can be read term
  by term, not just the forward. ``--loss energy`` runs the energy-only ablation (no double
  backward) for comparison.

The region columns of the main table are still **forward only**; the training-step tables
below it carry the backward.

Usage (see benchmarks/profile/README.md)::

    python benchmarks/profile/profile_film.py --device cuda \
        --structures benchmarks/structures/w6_mp2_avtz.xyz benchmarks/structures/w21_mp2_avtz.xyz \
                     external/torchff-lib/examples/water_216.pdb
"""

from __future__ import annotations

import argparse
import bisect
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


PHASES = ["forward", "first_backward", "second_backward"]


def make_calls(model, pos, z, frag, device, *, with_induction, force_weight=1.0, n_frames=1,
               loss="energy_forces"):
    """``loss="energy_forces"`` is the real training step (E+F, ``create_graph``, backward).
    ``loss="energy"`` is the ablation: energy-only loss, no ``create_graph`` -- the difference
    between the two is the price of the force loss, i.e. the double backward.
    """
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

    def train_step(marks=None):
        """One training step. ``marks`` (a list) receives ``(phase, seconds)`` per phase,
        timed with a device sync after each phase; the phases are also ``record_function``
        scopes named ``phase:<name>`` so the profiler can split the step the same way."""
        def mark(name, t0):
            if marks is not None:
                _sync(device)
                t1 = time.perf_counter()
                marks.append((name, t1 - t0))
                return t1
            return t0

        model.zero_grad(set_to_none=True)
        b = batch(True)
        t = time.perf_counter()
        with record_function("phase:forward"):
            out = model(b, with_induction=with_induction)
        t = mark("forward", t)
        if loss == "energy":
            with record_function("phase:second_backward"):
                total = out.energy.pow(2).mean()
                total.backward()
            mark("second_backward", t)
            return total
        with record_function("phase:first_backward"):
            (g,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=True)
        t = mark("first_backward", t)
        with record_function("phase:second_backward"):
            total = out.energy.pow(2).mean() + force_weight * g.pow(2).sum(-1).mean()
            total.backward()
        mark("second_backward", t)
        return total

    return forward, forward_forces, train_step


def time_phases(train_step, device, repeats):
    """Median wall time per phase of the training step (synced between phases)."""
    acc = {}
    for _ in range(repeats):
        marks = []
        _sync(device)
        train_step(marks=marks)
        for name, s in marks:
            acc.setdefault(name, []).append(s * 1e3)
    return {k: float(np.median(v)) for k, v in acc.items()}


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


# --------------------------------------------------------------------------------------
# training-step attribution: which forward region does each backward kernel belong to?
# --------------------------------------------------------------------------------------
#
# autograd tags every op it records with a per-thread sequence number, and the backward
# node that op creates (``MulBackward0`` ...) carries the same ``(fwd_thread, sequence_nr)``.
# So: forward ops inside a ``record_function`` region scope -> their sequence numbers ->
# the backward nodes with those numbers belong to that region. With ``create_graph=True``
# the ops run *inside* a backward node are recorded again (new sequence numbers, on the
# autograd thread), and the second backward's nodes point at them -- the same lookup, one
# hop further. Everything is summed as **self kernel time** per (phase, region), so a
# region's number is GPU-busy time, not the wall span; the gap to the phase wall time is
# dispatch/sync idle.

_EVAL_PREFIX = "autograd::engine::evaluate_function: "


def _own_device_us(ev, device):
    if device.type != "cuda":
        return ev.self_cpu_time_total
    if ev.is_user_annotation:
        return 0.0
    return sum(k.duration for k in ev.kernels)


def _walk(ev):
    yield ev
    for ch in ev.cpu_children:
        yield from _walk(ch)


def attribute_train_step(prof, device):
    """``{phase: {region: ms}}`` plus ``{phase: {backward node name: ms}}`` (second backward
    only, top nodes) from one profiled training step."""
    from torch.autograd import DeviceType

    events = [ev for ev in prof.events() if ev.device_type == DeviceType.CPU]
    events.sort(key=lambda e: e.time_range.start)
    phase_spans = [(ev.name[len("phase:"):], ev.time_range.start, ev.time_range.end)
                   for ev in events if ev.name.startswith("phase:")]
    region_spans = [(ev.name, ev.time_range.start, ev.time_range.end)
                    for ev in events if ev.name in REGIONS]

    def phase_of(ev):
        t = ev.time_range.start
        for name, s, e in phase_spans:
            if s <= t <= e:
                return name
        return None

    def forward_region_of(ev):
        t = ev.time_range.start
        for name, s, e in region_spans:
            if s <= t <= e:
                return name
        return "other"

    # per thread, (sequence_nr, region) of every recorded op, in creation order. A backward
    # node looks up its own sequence number; a node with no recorded op of its own (a custom
    # autograd.Function: the torchff kernels, _CoupledSolve) takes the region of the last op
    # recorded before it on the same thread, which is right because sequence numbers are
    # monotonic in time and the region scopes are contiguous.
    seq_region = {}                                     # thread -> [(seq, region), ...]
    node_region = {}                                    # id(backward node event) -> region
    per_phase = {p: {} for p in PHASES}
    node_time = {}                                      # second backward, by node name

    def lookup(thread, seq):
        lst = seq_region.get(thread)
        if not lst:
            return "other"
        i = bisect.bisect_right(lst, (seq, "￿")) - 1
        return lst[i][1] if i >= 0 else "other"

    for ev in events:                                   # time order: producers before users
        phase = phase_of(ev)
        if phase is None or ev.name.startswith("phase:") or ev.name in REGIONS:
            continue
        is_node = ev.name.startswith(_EVAL_PREFIX)
        if phase == "forward":
            region = forward_region_of(ev)
        else:
            node = ev if is_node else _enclosing_node(ev)
            if node is None:
                region = "other"                        # loss ops, grad accumulation
            elif node is ev:
                region = lookup(ev.fwd_thread, ev.sequence_nr)
                node_region[id(ev)] = region
            else:
                region = node_region.get(id(node), "other")
        if ev.sequence_nr >= 0 and not ev.fwd_thread:   # a recorded op (not a backward node)
            lst = seq_region.setdefault(ev.thread, [])
            if not lst or lst[-1][0] < ev.sequence_nr:
                lst.append((ev.sequence_nr, region))
        us = _own_device_us(ev, device)
        per_phase[phase][region] = per_phase[phase].get(region, 0.0) + us
        if phase == "second_backward" and ev.name.startswith(_EVAL_PREFIX):
            key = ev.name[len(_EVAL_PREFIX):]
            node_time[key] = node_time.get(key, 0.0) + sum(_own_device_us(c, device) for c in _walk(ev))
    ms = {p: {r: v / 1e3 for r, v in d.items()} for p, d in per_phase.items()}
    top = dict(sorted(((k, v / 1e3) for k, v in node_time.items()), key=lambda kv: -kv[1])[:25])
    return ms, top


def _enclosing_node(ev):
    p = ev.cpu_parent
    while p is not None:
        if p.name.startswith(_EVAL_PREFIX):
            return p
        p = p.cpu_parent
    return None


def profile_train_step(model, train_step, device):
    """One profiled training step -> (attribution, top second-backward nodes, profiler)."""
    with instrumented(model):
        train_step()
        _sync(device)
        with profile(activities=_activities(device), record_shapes=False) as prof:
            train_step()
            _sync(device)
    attrib, top_nodes = attribute_train_step(prof, device)
    return attrib, top_nodes, prof


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
    ap.add_argument("--loss", choices=["energy_forces", "energy"], default="energy_forces",
                    help="training-step loss: E+F with create_graph (default) or the energy-only "
                         "ablation (no double backward)")
    ap.add_argument("--no-train-split", action="store_true",
                    help="skip the per-phase timing and backward attribution of the training step")
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
            model, pos, z, frag, device, with_induction=with_induction, n_frames=args.frames,
            loss=args.loss,
        )
        rec = dict(structure=name, n_atoms=n_atoms, n_waters=int(frag.max()) + 1, frames=args.frames,
                   induction=with_induction, device=str(device), tag=tag,
                   ff_backend=ff_backend.active_backend(torch.zeros(1, device=device)),
                   loss=args.loss)

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
                if not args.no_train_split:
                    # -- training step: wall per phase + backward kernels attributed to regions
                    rec["train_phases_ms"] = time_phases(train_step, device, args.repeats)
                    attrib, top_nodes, prof = profile_train_step(model, train_step, device)
                    rec["train_attrib_ms"] = attrib
                    rec["train_second_backward_top_nodes_ms"] = top_nodes
                    if args.trace:
                        prof.export_chrome_trace(str(out_dir / f"{tag}_{name}_train_trace.json.gz"))
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
    lossflag = "_eonly" if args.loss == "energy" else ""
    base = out_dir / f"{tag}_{'ind' if with_induction else 'noind'}{frames}_{be}{lossflag}_{stamp}"
    with open(base.with_suffix(".json"), "w") as fh:
        json.dump(dict(
            checkpoint=args.checkpoint, torch=torch.__version__, dtype=str(torch.get_default_dtype()),
            python=sys.version.split()[0], host=platform.node(), results=results,
        ), fh, indent=2)
    with open(base.with_suffix(".md"), "w") as fh:
        fh.write(_format_table(results))
        fh.write(_format_train_split(results))
    print(f"\nwrote {base.with_suffix('.md')} and .json")
    print(_format_table(results))
    print(_format_train_split(results))


def _format_row(r):
    reg = r["regions"]
    key = "cuda_ms" if r["device"].startswith("cuda") else "cpu_ms"
    parts = " ".join(f"{k}={reg[k][key]:.1f}" for k in REGIONS if k in reg)
    phases = r.get("train_phases_ms")
    ph = (" | train phases: " + " ".join(f"{k}={v:.0f}" for k, v in phases.items())) if phases else ""
    return (f"  pairs={r['n_pairs']} fwd={r['forward_ms']:.1f}ms fwd+F={r['forward_forces_ms']:.1f}ms "
            f"train={r.get('train_step_ms', float('nan')):.1f}ms | forward split ({key}): {parts}{ph}")


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
    loss = results[0].get("loss", "energy_forces")
    what = ("E+F loss with create_graph=True + backward" if loss == "energy_forces"
            else "energy-only loss + backward (no create_graph; the --loss energy ablation)")
    note = (f"\nForward split column unit: {key} per call (forward only; backward is in the "
            f"whole-call columns). train_step = {what}.\n")
    return "\n".join(lines) + "\n" + note


def _format_train_split(results):
    """Per structure: phase wall times and the (phase x region) kernel-busy attribution."""
    out = []
    for r in results:
        attrib = r.get("train_attrib_ms")
        if not attrib:
            continue
        phases = [p for p in PHASES if p in r.get("train_phases_ms", {})]
        regions = REGIONS + ["other"]
        out.append(f"\n### {r['structure']} training step ({r.get('ff_backend')}, {r.get('frames')} frames): "
                   "wall per phase, kernel-busy ms by forward region\n")
        out.append("| region | " + " | ".join(phases) + " | total |")
        out.append("|---|" + "---|" * (len(phases) + 1))
        wall = r["train_phases_ms"]
        out.append("| **wall (synced)** | " + " | ".join(f"**{wall[p]:.1f}**" for p in phases)
                   + f" | **{sum(wall[p] for p in phases):.1f}** |")
        busy = {p: sum(attrib.get(p, {}).values()) for p in phases}
        out.append("| kernel busy | " + " | ".join(f"{busy[p]:.1f}" for p in phases)
                   + f" | {sum(busy.values()):.1f} |")
        for reg in regions:
            vals = [attrib.get(p, {}).get(reg, 0.0) for p in phases]
            if sum(vals) == 0.0:
                continue
            out.append(f"| {reg} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {sum(vals):.1f} |")
        top = r.get("train_second_backward_top_nodes_ms") or {}
        if top:
            out.append("\nsecond backward, top autograd nodes (kernel-busy ms, inclusive): "
                       + ", ".join(f"{k} {v:.1f}" for k, v in list(top.items())[:12]))
    if out:
        out.append("\nwall - kernel busy = launch/dispatch/sync idle in that phase. Region "
                   "attribution follows autograd sequence numbers (see attribute_train_step).\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()
