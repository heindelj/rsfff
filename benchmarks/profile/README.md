# M0 — profile the film model before porting anything

`profile_film.py` times one FilmModel call on water clusters and splits the forward by
region (neighbor list, features, parameter network, gates, elst/pauli/disp pair terms,
bonded, coupled solve), then times forward, forward+forces, and a full training step
(E+F loss, `create_graph=True`, backward). Results land in `results/` as `.md` + `.json`.

## Local smoke test (Mac, CPU, small clusters)

```bash
conda activate rsfff
cd ~/dev/rsfff
python benchmarks/profile/profile_film.py --device cpu --repeats 3 \
    --structures benchmarks/structures/w6_mp2_avtz.xyz benchmarks/structures/w21_mp2_avtz.xyz
```

## Perlmutter (the numbers that matter)

```bash
cd /global/cfs/cdirs/m3196/heindelj/software/rsfff
git fetch && git checkout torchff && git submodule update --init
sbatch benchmarks/profile/perlmutter.sbatch
```

`perlmutter.sbatch` runs the default set (w6, w12, w21, and a 216-water cluster cut from
`external/torchff-lib/examples/water_216.pdb`) with and without induction, and a 1000-water
run with `--skip-train-step` guarded against OOM. Edit the account/env paths at the top if
they differ.

## Training-step split (post-M5)

The training step is 70-80 % backward, which the forward region split cannot see. Every run
now also times the step per phase (forward / first backward = dE/dR with `create_graph` /
second backward = the loss) with a sync between phases, and profiles one step attributing
each backward kernel to the forward region whose autograd node ran it (autograd sequence
numbers; custom `autograd.Function`s such as the torchff kernels and `_CoupledSolve` fall
back to the last op recorded before them). The `.md` gets one extra table per structure:
wall per phase vs kernel-busy per phase (the gap is dispatch/sync idle), then busy ms per
region and phase, plus the top autograd nodes of the second backward.

```bash
sbatch benchmarks/profile/perlmutter_train_split.sbatch
```

runs w6 + w21 at 128 frames four ways: torchff E+F, torchff `--loss energy` (no double
backward), torchff `--no-induction` (no solve anywhere), and torch E+F for reference. Local
smoke test of the same code path (CPU, seconds):

```bash
python benchmarks/profile/profile_film.py --device cpu --repeats 2 --frames 4 \
    --structures benchmarks/structures/w6_mp2_avtz.xyz
```

`--no-train-split` turns the extra profiling off; `--trace` additionally writes a one-step
`*_train_trace.json.gz`.

## Reading the table

- Region columns are **forward-only** CUDA (or CPU) time per call; autograd runs the
  backward outside those scopes, so compare them against `forward_ms`, not against the
  training step.
- `train_step_ms` is the target of the whole port. If `elst+pauli+disp+coupled_solve` is a
  small fraction of `forward_ms`, kernels will not move `train_step_ms` much and the plan's
  ordering should change (see the plan note in Obsidian).
- `cap_events` in the JSON is non-empty when the pair list was truncated by
  `--max-neighbors`; raise it if so (12 Å in bulk water needs ~720).
- `--trace` writes, per structure, a per-op table (`*_forward.txt`, commit this) and a
  one-call gzipped chrome trace (`*_trace.json.gz`, gitignored; open in Perfetto).
- `--frames 128` replicates a structure into one batch of 128 frames, i.e. the shape of a
  real training step (`batch_size: 128`), which amortises the per-call dispatch overhead
  that dominates single-frame timings.

## torch.compile and the CG sync stride

`--compile cg` runs the coupled solve's PCG iteration through `torch.compile` (dynamic shapes;
one graph, no recompiles across batch shapes on CPU) — training-safe, since the solve runs
under `no_grad` inside the implicit adjoint. `--compile network` / `all` also compiles the
parameter network's forward as one graph, but inductor does not support double backward, so
that is **inference only** (`--skip-train-step`; MD, sampling, evaluation). `--cg-check-every k`
tests convergence every k iterations instead of every one (converged frames take exact zero
steps, so results are unchanged; up to k-1 extra iterations against k× fewer host syncs).
The same switches in production: `RSFFF_COMPILE=cg|network|all` (read by
`load_film_model` and, `cg` only, by `train_film`) and `film.cg_check_every` in the config.

```bash
bash benchmarks/profile/run_train_split.sh main cg2 cg4 ccg ccg2 cinf
```
