"""Non-variational (unrolled, fixed-K) vs variational (converged PCG) induction, by cluster size.

For water clusters of increasing size, times the film model (converged coupled solve + adjoint)
against the nonvariational model (``docs/fff_nonvariational.md``) at one or more ``K``, on one
or both backends (``torch`` reference / ``torchff`` kernels), for the three things that matter:

* ``energy``      -- one forward under ``no_grad`` (what a Monte Carlo or an evaluation costs);
* ``forces``      -- forward + one backward (an MD step);
* ``train_step``  -- energy + force loss with ``create_graph=True`` and its backward (the
  double backward the kernel port exists for), on ``--frames`` copies of the structure.

Per model it also reports the solve's own numbers: CG iterations for the film; for the
nonvariational model the physical residual at ``x_K`` and, with ``--accuracy``, the energy gap
and force difference to the converged solve of *its own* functional (the price of stopping
at ``K``). Peak CUDA memory of the training step is recorded on GPU.

Both models come from checkpoints so the timings reflect trained parameters (CG iteration
counts depend on them); ``--n-iter`` re-runs the nonvariational checkpoint at other ``K``
for timing (its accuracy columns are then those of an unrefitted ``K``, read them as such).

Results: ``results/<tag>.md`` (the table), ``results/<tag>.json`` (everything), and a figure
via ``bench_plot.py``. Usage::

    # Mac / CPU smoke
    python benchmarks/nonvariational/bench_solve.py --device cpu --backends torch \\
        --waters 4 8 16 32 --repeats 3

    # Perlmutter (interactive GPU node, conda env with torchff built)
    python benchmarks/nonvariational/bench_solve.py --device cuda --backends torch torchff \\
        --waters 8 16 32 64 128 216 512 1000 --n-iter 2 3 4 --frames 4 --accuracy

Clusters up to 216 waters are cut as spheres from ``external/torchff-lib/examples/water_216.pdb``
(the closest ``N`` molecules to the box centre); larger ones from the box replicated 2x2x2.
``--structures`` takes explicit xyz/pdb files instead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "profile"))

from profile_film import _batch_to, _fragment_order, _sync, read_water  # noqa: E402

from rsfff.ff import backend as ff_backend  # noqa: E402
from rsfff.ff.coupled_solve import pcg  # noqa: E402
from rsfff.ff.nonvariational import NonvariationalModel  # noqa: E402
from rsfff.ff.film.model import FilmModel  # noqa: E402
from rsfff.md.film_driver import load_film_model, make_batch, water_fragment_index  # noqa: E402

DEFAULT_FILM = (REPO_ROOT / "rsfff_active_learning/committees/film_committee_100k_excl/"
                "member_00/member_00_full/best.pt")
DEFAULT_NV = REPO_ROOT / "checkpoints/water_nonvariational_full/best.pt"
WATER_BOX = REPO_ROOT / "external/torchff-lib/examples/water_216.pdb"
HARTREE_KJ = 2625.4996


# --------------------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------------------

def cut_cluster(n_waters: int, box_path: Path = WATER_BOX):
    """The ``n_waters`` molecules closest to the centre of the (replicated) bulk box."""
    from ase.io import read

    cell = np.asarray(read(str(box_path)).cell.lengths(), dtype=float)
    pos, z, _ = read_water(str(box_path))          # unwrapped, molecules whole
    pos, z = _fragment_order(pos, z)
    mol = pos.reshape(-1, 3, 3)
    n_box = mol.shape[0]
    reps = 1
    while n_box * reps ** 3 < n_waters:
        reps += 1
    if reps > 1:
        shifts = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
        mol = np.concatenate([mol + (s * cell)[None, None, :] for s in shifts])
    centre = mol[:, 0].mean(axis=0)
    order = np.argsort(np.linalg.norm(mol[:, 0] - centre, axis=1))
    mol = mol[order[:n_waters]]
    pos = mol.reshape(-1, 3)
    pos -= pos.mean(axis=0)
    return pos, np.tile([8, 1, 1], n_waters)


def load_structure(path: str):
    pos, z, _ = read_water(path)
    return _fragment_order(pos, z)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------

def load_models(args, device):
    """``{label: model}`` in display order: the film, then the nonvariational at each K."""
    models = {}
    if args.film_checkpoint and Path(args.film_checkpoint).exists():
        film, _ = load_film_model(args.film_checkpoint, device=device,
                                  cg_check_every=args.cg_check_every)
        film.cg.update(rtol=args.cg_rtol, atol=args.cg_atol, maxiter=args.cg_maxiter)
        models["film (PCG)"] = film
    else:
        print(f"film checkpoint not found ({args.film_checkpoint}); skipping the film model")
    nv, _ = load_film_model(args.nv_checkpoint, device=device)
    if not isinstance(nv, NonvariationalModel):
        raise SystemExit(f"{args.nv_checkpoint} is not a nonvariational checkpoint")
    nv.cg.update(rtol=args.cg_rtol, atol=args.cg_atol, maxiter=args.cg_maxiter)
    trained_k = nv.n_iter
    for k in args.n_iter:
        label = f"nonvariational K={k}" + ("" if k == trained_k else " (unrefitted)")
        models[label] = (nv, k)
    return models, nv, trained_k


def _set_k(model, k):
    """Run a trained nonvariational model at another K: the iterate weights beyond the trained
    K are 1 (plain iterate); below it the trained ones are truncated."""
    import torch.nn as nn

    model.n_iter = int(k)
    raw = model.iterate_weights_raw
    if raw is None:
        return
    want = max(k - 1, 0)
    if raw.numel() != want:
        new = raw.new_zeros(want)
        n = min(want, raw.numel())
        new[:n] = raw.detach()[:n]
        model.iterate_weights_raw = nn.Parameter(new) if want else None


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------

def timeit(fn, device, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, float(np.min(times)) * 1e3


@dataclass
class Row:
    label: str
    backend: str
    n_waters: int
    n_atoms: int
    frames: int
    energy_ms: float = float("nan")
    forces_ms: float = float("nan")
    train_ms: float = float("nan")
    train_peak_mb: float = float("nan")
    cg_iter: float = float("nan")
    residual: float = float("nan")
    gap_kj: float = float("nan")          # E(x_K) - E(x*) per frame, kJ/mol
    gap_rel: float = float("nan")         # gap / |E_ind|
    force_rms_diff: float = float("nan")  # RMS |F_K - F*|, Ha/bohr... reported in kJ/mol/A
    extra: dict = field(default_factory=dict)


def _calls(model, pos, z, frag, device, n_frames):
    positions = np.tile(np.asarray(pos, dtype=float), (n_frames, 1))

    def batch(requires_grad):
        b = _batch_to(make_batch(positions, z, frag, n_frames=n_frames), device)
        if requires_grad:
            b.positions.requires_grad_(True)
        return b

    def energy():
        with torch.no_grad():
            return model(batch(False))

    def forces():
        b = batch(True)
        out = model(b)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions)
        return out, g

    def train_step():
        model.zero_grad(set_to_none=True)
        b = batch(True)
        out = model(b)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=True)
        total = out.energy.pow(2).mean() + g.pow(2).sum(-1).mean()
        total.backward()
        return total

    return batch, energy, forces, train_step


def accuracy(model: NonvariationalModel, batch_fn, device):
    """Energy gap and force difference between the unrolled state and the converged solve of
    the same functional, at the model's current parameters, on one frame."""
    from rsfff.ff.film.model import StateDescriptor
    from rsfff.ff.pairs import union_channels

    b = batch_fn(True)
    out = model(b)
    (f_k,) = torch.autograd.grad(out.energy.sum(), b.positions, retain_graph=True)
    e_k = out.energy.detach()
    e_ind = out.interaction["induction"].detach()

    # the converged level, differentiably, through the same film machinery
    species_idx = model.projector.species_index(b.atomic_numbers)
    state = StateDescriptor.from_batch(b, species_idx, model.projector.featurizer.n_species)
    ch_ind, chb_ind, _ = union_channels(b.positions, b.batch_idx, state.fragment_idx, 0.0)
    rp = model._response_parameters(out.parameters)
    gate_ind = out.gate["elst"] * (1.0 - out.p_intra)
    ref = model.converged_level(
        rp, positions=b.positions, batch=b, bond_index=ch_ind, bond_batch=chb_ind,
        pair_index=out.pair_index, gate=gate_ind,
    )
    e_star = out.energy - out.level_ind.energy + ref.energy
    (f_star,) = torch.autograd.grad(e_star.sum(), b.positions)
    gap = (e_k - e_star.detach()) * HARTREE_KJ
    df = (f_k - f_star).norm(dim=-1).pow(2).mean().sqrt() * HARTREE_KJ / 0.529177210903
    return dict(
        gap_kj=float(gap.mean()), gap_rel=float((gap / (e_ind.abs() * HARTREE_KJ + 1e-12)).mean()),
        force_rms_diff=float(df), cg_iter_ref=ref.n_iter,
    )


def run(args):
    device = torch.device(args.device)
    structures = []
    if args.structures:
        for p in args.structures:
            pos, z = load_structure(p)
            structures.append((Path(p).stem, pos, z))
    else:
        for n in args.waters:
            pos, z = cut_cluster(int(n))
            structures.append((f"w{n}", pos, z))

    models, nv, trained_k = load_models(args, device)
    rows: list[Row] = []
    for name, pos, z in structures:
        frag = water_fragment_index(pos, z)
        n_at = len(z)
        for backend_name in args.backends:
            ff_backend.set_backend(backend_name)
            for label, entry in models.items():
                model = entry[0] if isinstance(entry, tuple) else entry
                if isinstance(entry, tuple):
                    _set_k(model, entry[1])
                model.eval()
                batch_fn, energy, forces, train_step = _calls(model, pos, z, frag, device, args.frames)
                row = Row(label, backend_name, n_at // 3, n_at, args.frames)
                try:
                    row.energy_ms, _ = timeit(energy, device, args.repeats)
                    row.forces_ms, _ = timeit(forces, device, args.repeats)
                    out = energy()
                    if out.solver and "ind" in out.solver:
                        row.cg_iter = float(out.solver["ind"][0])
                    if isinstance(model, NonvariationalModel):
                        keep = model.with_residual
                        model.with_residual = True
                        try:
                            o = energy()
                            row.residual = float(o.solver["ind_residual"].max())
                        finally:
                            model.with_residual = keep
                        if args.accuracy:
                            try:
                                row.extra.update(accuracy(model, batch_fn, device))
                                row.gap_kj = row.extra["gap_kj"]
                                row.gap_rel = row.extra["gap_rel"]
                                row.force_rms_diff = row.extra["force_rms_diff"]
                            except RuntimeError as exc:   # OOM on the dense torch path
                                row.extra["accuracy_error"] = str(exc).splitlines()[0]
                    if not args.skip_train_step:
                        model.train()
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        row.train_ms, _ = timeit(train_step, device, args.repeats)
                        if device.type == "cuda":
                            row.train_peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
                        model.eval()
                except RuntimeError as exc:
                    row.extra["error"] = str(exc).splitlines()[0]
                    print(f"  {name} {backend_name} {label}: {row.extra['error']}", flush=True)
                rows.append(row)
                print(_fmt_row(name, row), flush=True)
            _set_k(nv, trained_k)
    ff_backend.set_backend("auto")
    return rows, trained_k


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------

def _f(x, nd=1):
    return "-" if x != x else f"{x:.{nd}f}"


def _fmt_row(name, r: Row):
    return (f"  {name:>6} {r.backend:>7} {r.label:<28} energy {_f(r.energy_ms):>8} ms  "
            f"forces {_f(r.forces_ms):>8} ms  train {_f(r.train_ms):>8} ms  "
            f"cg {_f(r.cg_iter, 0):>3}  resid {_f(r.residual, 4):>8}  gap {_f(r.gap_kj, 3):>7} kJ/mol")


def write_table(rows: list[Row], path: Path, args, trained_k):
    lines = [
        "# Non-variational vs variational induction, by cluster size", "",
        f"device `{args.device}` ({platform.node()}), torch {torch.__version__}, "
        f"torchff {'yes' if ff_backend.HAVE_TORCHFF else 'no'}, frames per training step {args.frames}, "
        f"median of {args.repeats}. Film: `{args.film_checkpoint}`; nonvariational (trained K={trained_k}): "
        f"`{args.nv_checkpoint}`. Times in ms per call on the whole batch.", "",
        "| waters | atoms | backend | model | energy | forces | train step | peak MB | CG it | residual | gap kJ/mol | gap/E_ind | dF rms kJ/mol/A |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.n_waters} | {r.n_atoms} | {r.backend} | {r.label} | {_f(r.energy_ms)} | {_f(r.forces_ms)} | "
            f"{_f(r.train_ms)} | {_f(r.train_peak_mb, 0)} | {_f(r.cg_iter, 0)} | {_f(r.residual, 4)} | "
            f"{_f(r.gap_kj, 3)} | {_f(r.gap_rel, 4)} | {_f(r.force_rms_diff, 3)} |"
        )
    # speedups relative to the film on the same structure and backend
    film = {(r.n_waters, r.backend): r for r in rows if r.label.startswith("film")}
    if film:
        lines += ["", "## Speedup over the film (same structure, same backend)", "",
                  "| waters | backend | model | energy | forces | train step |", "|---:|---|---|---:|---:|---:|"]
        for r in rows:
            ref = film.get((r.n_waters, r.backend))
            if ref is None or r is ref:
                continue
            lines.append(f"| {r.n_waters} | {r.backend} | {r.label} | x{_f(ref.energy_ms / r.energy_ms, 2)} | "
                         f"x{_f(ref.forces_ms / r.forces_ms, 2)} | x{_f(ref.train_ms / r.train_ms, 2)} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--film-checkpoint", default=str(DEFAULT_FILM))
    ap.add_argument("--nv-checkpoint", default=str(DEFAULT_NV))
    ap.add_argument("--waters", nargs="+", type=int, default=[8, 16, 32, 64, 128, 216])
    ap.add_argument("--structures", nargs="*", default=None, help="explicit xyz/pdb files instead of --waters")
    ap.add_argument("--n-iter", nargs="+", type=int, default=[3], help="K values for the nonvariational model")
    ap.add_argument("--backends", nargs="+", default=["torchff" if torch.cuda.is_available() else "torch"],
                    choices=["torch", "torchff"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=1, help="copies of the structure per training step")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--accuracy", action="store_true", help="gap and force difference to the converged solve")
    ap.add_argument("--skip-train-step", action="store_true")
    ap.add_argument("--cg-rtol", type=float, default=1e-9)
    ap.add_argument("--cg-atol", type=float, default=1e-12)
    ap.add_argument("--cg-maxiter", type=int, default=100)
    ap.add_argument("--cg-check-every", type=int, default=4)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks/nonvariational/results"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.device}_{'_'.join(args.backends)}_{time.strftime('%Y%m%d_%H%M%S')}"

    rows, trained_k = run(args)
    payload = dict(args=vars(args), trained_k=trained_k, host=platform.node(), torch=torch.__version__,
                   rows=[r.__dict__ for r in rows])
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1, default=str))
    write_table(rows, out_dir / f"{tag}.md", args, trained_k)
    print(f"\nwrote {out_dir / tag}.md / .json")
    if not args.no_plot:
        try:
            from bench_plot import plot
            plot(out_dir / f"{tag}.json", out_dir / f"{tag}.png")
            print(f"wrote {out_dir / tag}.png")
        except Exception as exc:  # matplotlib missing on a compute node is not a failure
            print(f"plot skipped: {exc}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
