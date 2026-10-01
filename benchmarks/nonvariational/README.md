# Non-variational vs variational induction

`bench_solve.py` times the film model (converged PCG + adjoint) against the nonvariational
model (`docs/fff_nonvariational.md`: `K` unrolled iterations, learned mutual damping) on water
clusters of increasing size — energy, energy + forces, and the training step (E+F loss with
`create_graph=True`) — on the torch reference backend and/or the torchff kernels, and reports
the solve's own numbers next to the timings: CG iterations for the film; the physical residual
at `x_K`, the energy gap `E(x_K) − E(x*)` and the force difference to the converged solve of the
same functional for the nonvariational model (`--accuracy`). `bench_plot.py` draws the figure
from the JSON (importable from a notebook).

Both models are loaded from checkpoints (`film_committee_100k_excl` member 0 and
`checkpoints/water_nonvariational_full`), so CG iteration counts and the unrolled model's
widths are the trained ones. `--n-iter 2 3 4` re-runs the nonvariational checkpoint at other
`K` for timing; its accuracy at an unrefitted `K` is reported as such.

## Local smoke (Mac, CPU)

```bash
conda activate rsfff
python benchmarks/nonvariational/bench_solve.py --device cpu --backends torch \
    --waters 4 8 16 32 --n-iter 2 3 4 --repeats 3 --accuracy
```

On the torch backend the mutual operator materialises `(P, K, K)` tensors and the unrolled
loop differentiates through `K` of them, so CPU numbers say little about the kernels; the
film's solve runs under `no_grad` there. Use these to check that everything runs.

## Perlmutter (the numbers that matter)

```bash
cd /global/cfs/cdirs/m3196/heindelj/software/rsfff
git fetch && git checkout nonreactive && git submodule update --init
# rebuild torchff-lib: the nonvariational training step needs slater_elec_field_hvp
(cd external/torchff-lib && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation --no-deps -e .)
python -c "from torchff import slaterelec as se; print(se.HAVE_KERNELS, se.FIELD_DOUBLE_BACKWARD)"   # True True
sbatch benchmarks/nonvariational/perlmutter.sbatch
```

Also worth running once after the rebuild: `pytest external/torchff-lib/tests/test_slaterelec.py
tests/backend` — the field's double backward against the reference (gradgradcheck) and the
nonvariational model's parity across backends, including the force-loss gradient.

## What to read

- **forces** is the MD step. Film: ~10–15 PCG matvecs (field kernel) + an adjoint solve of the
  same size (the CT bond term consumes `x*`) + convergence-test syncs. Nonvariational: `K`
  field kernels forward, `K` field-VJP kernels back, no syncs.
- **train step** is where the field's double backward matters: the nonvariational loop needs
  `slater_elec_field_hvp` `K` times per step; the film needs it once (the adjoint's residual VJP).
- **gap / dF rms**: the price of stopping at `K` relative to the converged minimum of the
  *same* functional — not an error against the reference data (the fit absorbs it).
- Watch `residual` and `gap` grow with cluster size: if they do at fixed `K`, the model was
  fitted on clusters too small to see the series' tail, and larger training clusters or a
  larger `K` are needed.
