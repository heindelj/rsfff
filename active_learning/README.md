# Active learning for the water film model

An [easyAL](../../easyAL) loop that grows the training set for the range-separated force-field
functional by sampling with the model itself and labeling what it finds with Q-Chem, through
the existing `qchem_roundtrip/` job pool on Perlmutter.

    build      packmol packs (H2O)n into a spherical cavity        structures.extxyz
    optimize   relax every packing on the model's own surface      samples.extxyz
    dynamics   Langevin NVT from each minimum, inside a soft wall  samples.extxyz
    label      Q-Chem EDA + force from the round-trip job pool     labeled.extxyz
    train      refit the film model on everything labeled so far   model          (TODO)
    assess     measure the new model on its holdout                metrics.json   (TODO)

| file | contents |
|------|----------|
| `workflows.py`   | the loop wired up (`water_loop`, `loop_from_spec`) and the CLI |
| `common.py`      | reaches `qchem_roundtrip/scripts/qchem_roundtrip.py` and the Q-Chem output parsers; import from here |
| `build.py`       | `PackmolWaterClusters`: `(H2O)n` packed into a sphere, one packmol run per structure |
| `sampling.py`    | `MinimizeSample` and `DynamicsSample`: the two sampling stages |
| `model.py`       | the film checkpoint loaded once, and an ASE calculator with the wall folded in |
| `label_stage.py` | `QChemLabel`: eda+force jobs into the pool, merged back into training frames |
| `train_stage.py` / `assess_stage.py` | placeholders, with what they should become |
| `scripts/preflight.py` | check every prerequisite without queueing anything |
| `scripts/smoke_test.sh` | one full iteration on an interactive node, workers included |
| `templates/*_fast.in` | the cheap level of theory the smoke test can use |

`run` returns `"pending"` whenever it is waiting on the cluster — today, whenever the Q-Chem
jobs are not finished. Call it again until it converges; completed stages are skipped, so
re-running is free.

## Running it

```bash
conda install -c conda-forge packmol      # or: pip install packmol
pip install -e ../easyAL

# on Perlmutter, where submitting is just topping up the worker pool
python active_learning/workflows.py water \
    --root $SCRATCH/water_al \
    --checkpoint /path/to/checkpoints/water_film_full/best.pt \
    --submit "bash scripts/submit_workers.sh --target 16"

python active_learning/workflows.py water --root $SCRATCH/water_al --status
```

Nothing needs the repository to be the current directory: `--root` and `--checkpoint` can be
anywhere, the checkpoint carries its own reference energies, and the stages find the job pool
through `$RSFFF_QCHEM_ROOT` (falling back to `<repo>/qchem_roundtrip`). `$RSFFF_REPO` points at
the checkout when the loop runs from somewhere else with rsfff installed.

## Testing it on an interactive node

Two tools under `scripts/`, so a problem shows up in seconds rather than after a queue wait.

**`preflight.py`** checks everything the loop needs and queues nothing — the python
environment, `torch_cluster`, whether this `easyal` takes `sample=[...]`, whether the
installed `rsfff` actually contains `ff.film`, that packmol packs a dimer, that the checkpoint
loads and returns a finite energy and gradient, that the Q-Chem parsers load without pyscf,
and that the job pool's config and templates are there and its layout is writable. It prints a
line per check and exits non-zero if a required one failed.

```bash
python active_learning/scripts/preflight.py     --checkpoint checkpoints/water_film_full/best.pt --root $SCRATCH/water_al
```

**`smoke_test.sh`** runs one whole iteration inside an allocation, with the Q-Chem workers in
that same allocation and the label stage polling for them instead of stopping at `pending`:

```bash
salloc -A m3196 -N 1 -C cpu -q interactive -t 02:00:00
module load python qchem
conda activate rsfff                     # whatever has torch + rsfff + easyal
bash active_learning/scripts/smoke_test.sh --fast
```

It works in `$SCRATCH/rsfff_al_smoke` with its own copy of the pool's config and templates, so
the production pool is never touched and nothing it produces can be mistaken for training
data. One dimer, two MD frames, four Q-Chem jobs. `--fast` swaps in `templates/eda_fast.in`
and `force_fast.in` (wB97X-D/6-31G\*) so a pass takes a couple of minutes; without it the
production level of theory runs. Useful flags: `--waters`, `--frames`, `--workers`,
`--threads`, `--timeout`, `--reuse` (keep the tree and carry on from where it stopped).

It finishes with a digest — per-stage status, timing and metrics, the first labeled header,
and `dropped.json` if anything was thrown away — which is the part worth reading when
something is wrong.

The mechanism behind the waiting is `QChemLabel(wait_seconds=..., poll_seconds=...)`, on the
CLI as `--wait` and `--poll`. With `wait_seconds=0` (the default, and what a queued run wants)
the stage looks once and reports `Pending`; with a budget it keeps running the `sync` hooks
and checking until the jobs land. Nothing else differs between the two.

Driving it from a laptop instead, the hooks become the round trip:

```python
label=dict(
    submit=["bash scripts/sync_inputs_up.sh",
            "ssh perlmutter 'cd $REMOTE_DIR && bash scripts/submit_workers.sh --target 16'"],
    sync=["bash scripts/sync_outputs_down.sh"],
)
```

`submit` runs once, in the round-trip directory, after the inputs are written; `sync` runs
there before every completion check. With no hooks, do those steps by hand and run again.

From python, or from a task spec that a driver job can be pointed at:

```python
from workflows import water_loop, loop_from_spec
loop = water_loop("runs/water", initial_model="checkpoints/water_film_full/best.pt",
                  build=dict(sizes=(2, 30)), dynamics=dict(steps=4000),
                  label=dict(submit=["bash scripts/submit_workers.sh --target 16"]))
print(loop.run(max_iterations=10))     # "converged" | "pending" | "max_iterations"
```

```json
{"workflow": "water", "max_iterations": 10,
 "initial_model": "/path/checkpoints/water_film_full/best.pt",
 "stages": {"build": {"sizes": [2, 30]},
            "label": {"submit": ["bash scripts/submit_workers.sh --target 16"]}}}
```

## Why the sampling is two stages

They answer different questions and fail differently. The minimizer asks *where are this
model's minima*; a packing it cannot relax is a statement about the model. The dynamics asks
*what does this model do at temperature near those minima*; its failure modes are heating,
evaporation and a broken molecule. easyAL takes a list of `Sample` stages, so each gets its own
directory, contract, hashes and metrics, and either can be redone alone:

```python
loop.reset(0, "dynamics")     # redo the trajectories, keep the minima
```

Running the dynamics *from the minima* rather than from the packings is the point of the
ordering. A random packing carries tens of kcal/mol of strain; a trajectory started there
spends its first picosecond dumping that into the thermostat and samples nothing worth a
Q-Chem calculation.

## Stages

### `build` — `PackmolWaterClusters`

One packmol run per structure: `inside sphere 0 0 0 R` with `R` set from the cluster size at a
chosen density,

    R = (3 n V_w / 4 pi)^(1/3) + padding,    V_w = 30.0 A^3 at 1 g/cm^3

so a cluster starts near liquid density instead of as a gas that has to collapse. Packmol
writes molecules consecutively, so the output is already `O H H O H H ...`, which is the
fragment order `film_driver.water_fragment_index` requires — and the fragmentation is therefore
known without computing it, and is written onto the frame (`fragment_idx`, `n_fragments`,
`fragment_charges`, `fragment_multiplicities`). Every stage downstream wants it: the film model
takes a fixed fragmentation, and a Q-Chem EDA input is built out of one.

### `optimize` — `MinimizeSample`

`film_driver.optimize` (L-BFGS-B on the analytic gradient) to `gtol = 1e-3` Hartree/A. These
are trajectory starting points, not spectroscopy: a 300 K trajectory carries `|F|max` around
0.07 Hartree/A, so 1e-3 is already well inside the thermal noise the structure is about to be
handed. Tighten it if the minima are also going to a Hessian.

### `dynamics` — `DynamicsSample`

ASE `Langevin` at 300 K, 0.5 fs, friction 0.02/fs, with `confine.flat_bottom_sphere` around the
running center of mass so a cluster cannot simply evaporate. The wall lives in the calculator,
not the model, and the `model_energy` stored on a frame excludes it. A trajectory stops — and
everything before it is kept — when the energy is not finite, the temperature passes
`max_temperature`, `|F|max` passes `max_force`, or a water stops being intact.

### `label` — `QChemLabel`

Writes this iteration's frames as one extxyz file into the pool
(`<calculation>/geoms/<prefix>_iterNNN.xyz`, the same content for both calculations so the
per-frame stems match), calls the pool's own generator to expand them into one `.in` per frame,
runs the `submit` hooks once and the `sync` hooks every call, and reports itself `Pending`
until every job is done or failed. The workers are untouched: they claim the new inputs like
any others.

**A training frame needs both jobs.** The EDA decomposition and the analytic forces are
separate Q-Chem runs, and `scripts/parse_roundtrip.build_merged_frame` is what joins them. Its
cross-checks are the reason to want both: the two jobs are given the same geometry and must
come back in the same standard orientation (1e-8 A) with the same supersystem energy (1e-6 Ha).
A frame missing either, or failing those checks, is dropped and counted; `max_failed_fraction`
decides whether that stops the stage.

The merged geometry is Q-Chem's standard orientation, not the sampled one — forces and
multipoles live in that frame, so the coordinates have to. The link back to the sampled
structure is the stem, and it is checked by what survives a reorientation: the species sequence
and the sorted list of interatomic distances.

Labeled frames carry the schema `scripts/parse_roundtrip.py` writes (`energy`, `eda_*`,
`fragment_energies`, `fragment_dipoles`, `fragment_second_moments`, the molecular multipoles,
`forces`, `mulliken_charges`, `fragment_idx`, `method`, `basis`) plus the loop's lineage:
`sample_id` is the sampled frame's `source`, with `al_loop`, `al_iteration` and `al_stem`.

`common.py` loads the Q-Chem parsers out of `qcgen` **without importing that package** —
`qcgen/__init__.py` pulls in the pyscf compute backend, and a loop's environment has no reason
to carry pyscf. The three parser modules need numpy and each other, nothing more.

## Provenance

```
$SCRATCH/water_al/
    loop.json                     initial model + data (hashed), every stage's params, environment
    summary.json                  per iteration: stage statuses and metrics
    iter_000/
        build/      stage.json  structures.extxyz  scratch/{*.inp,*.xyz,*.log}
        optimize/   stage.json  samples.extxyz
        dynamics/   stage.json  samples.extxyz
        label/      stage.json  labeled.extxyz     scratch/{jobs.json,submitted.json,dropped.json}
        train/      stage.json  model
        assess/     stage.json  metrics.json
```

Each `stage.json` holds the stage's class and parameters, its status and timings, previous
attempts, its metrics and notes, and the sha256 of its input, the model it ran with, the
training set (`train`) and its output. A frame's history is also written on the frame itself:
`sample_id` reads `packmol/n012_1|min|md900`, and `al_stem` names the pair of Q-Chem jobs.
