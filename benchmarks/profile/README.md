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
