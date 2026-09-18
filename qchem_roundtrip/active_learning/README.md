# Active learning for the water film model

An easyAL loop that grows the training set for the range-separated force-field functional by
sampling with the model itself and labeling what it finds with Q-Chem. It lives **inside** the
job-pool bundle, so it travels with it: the bundle is rsynced to Perlmutter and lives outside
any checkout there, and active learning is just another job type it can run.

    build      packmol packs (H2O)n into a spherical cavity        structures.extxyz
    optimize   relax every packing on the model's own surface      samples.extxyz
    dynamics   Langevin NVT from each minimum, inside a soft wall  samples.extxyz
    select     keep the frames the committee disagrees about       samples.extxyz
    label      Q-Chem EDA + force from the round-trip job pool     labeled.extxyz
    train      refit as a committee of independent fits            committee/
    assess     score the new committee on what it just learned     metrics.json

## The walk

The loop is **schedule-driven**: it walks up in cluster size, one size group per iteration,
and stops after five because the walk is finished, not because a threshold was met.

| iteration | sizes | structures packed / size | labeled / size | labeled |
|---|---|---|---|---|
| 0 | 2-5   | 50 | 250 | 1000 |
| 1 | 6-10  | 32 | 160 |  800 |
| 2 | 11-15 | 24 | 120 |  600 |
| 3 | 16-20 | 16 |  80 |  400 |
| 4 | 21-25 |  8 |  40 |  200 |
| | | | **total** | **3000** |

Small clusters first and most of them -- they are where the reference is cheap and where the
short-range behaviour is set -- then fewer of each as they grow and each label costs more. The
same number of every size within a group, so no size is quietly skipped.

`POOL_MULTIPLIER` (4) is how many candidates are sampled per label: selecting the most
uncertain 250 of 1000 is a choice, selecting 250 of 250 is not. The structures-per-size column
is that pool divided by the frames one trajectory yields (`steps // stride`), so changing the
trajectory length changes how many structures get packed, automatically.

Edit `SIZE_SCHEDULE` in `workflows.py` to change any of it.

| file | contents |
|------|----------|
| `workflows.py`   | the loop wired up (`water_loop`, `loop_from_spec`) and the CLI |
| `common.py`      | reaches `qchem_roundtrip/scripts/qchem_roundtrip.py` and the Q-Chem output parsers; import from here |
| `build.py`       | `PackmolWaterClusters`: `(H2O)n` packed into a sphere, one packmol run per structure |
| `sampling.py`    | `MinimizeSample` and `DynamicsSample`: the two sampling stages |
| `model.py`       | the film checkpoint loaded once, and an ASE calculator with the wall folded in |
| `label_stage.py` | `QChemLabel`: eda+force jobs into the pool, merged back into training frames |
| `committee.py` | load a committee, predict with it, and measure its disagreement |
| `train_stage.py` | `CommitteeTrain`: N members, one process each, warm started per index |
| `assess_stage.py` | `ValidationAssess`: what the new committee does on what it just learned |
| `scripts/nersc_env.sh` | the pool path and the conda prefix, for either side of the round trip |
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

## Production run on Perlmutter

`scripts/launch_production.sh` starts a run; everything after that is batch jobs.

```bash
module load python
source /global/cfs/cdirs/m3196/heindelj/rsfff_data/active_learning/scripts/nersc_env.sh
bash $RSFFF_QCHEM_ROOT/active_learning/scripts/launch_production.sh \
    --root $SCRATCH/water_al \
    --checkpoint $RSFFF_REPO/checkpoints/water_film_full/best.pt \
    --workers 16 \
    -- --train-config $RSFFF_REPO/configs/water_film.yaml
```

It runs `preflight.py`, writes every setting to `<root>/driver.env`, and queues two kinds of job:

- **driver** (`scripts/driver.slurm`, one CPU node, `--qos=premium`, `rsfff_al_<name>`): runs the loop until
  the label stage is waiting on Q-Chem (after polling `--wait`, default 15 min), then exits and
  requeues itself `--resubmit-delay` minutes later, so no node idles through the labeling. It
  stops at the end of the size walk (`driver_state/DONE`) or when a stage raises
  (`driver_state/FAILED`). `--dependency=singleton` keeps it to one driver per run, and 10 min
  before its wall clock it requeues immediately, so a stage longer than one driver carries on in
  the next. Four committee members train at once, 32 threads each (`--train-parallel`,
  `--train-threads`).
- **workers** (`qchem_al_worker`, `--qos=premium`, 24 h): submitted by the label stage's
  hooks through `scripts/al_workers.sh`, which is both the `submit` and the `sync` hook, so
  every check tops the pool back up to `--workers`. They use `config.al.json` (the pool's config
  with only eda and force enabled), and the target is capped at the outstanding eda/force jobs,
  so once the last input is claimed nothing new is queued to idle out. They will also run any
  other unfinished eda/force input in the pool.

```bash
squeue -u $USER -n rsfff_al_water_al,qchem_al_worker
cat $SCRATCH/water_al/driver_state/history          # every driver: start, exit, outcome
python3 $RSFFF_QCHEM_ROOT/active_learning/workflows.py water --root $SCRATCH/water_al --status
bash $RSFFF_QCHEM_ROOT/active_learning/scripts/launch_production.sh --resume $SCRATCH/water_al
```

Worker count, QOS and wall clocks can be changed in `driver.env` between drivers; they reach the
hooks through the environment, so the label stage's recorded parameters don't change. The
loop's own flags (`AL_EXTRA_ARGS`) are fixed for the life of the run. To stop: `scancel` the
driver (and the workers if you want them gone); `--resume` picks it up again.

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
source active_learning/scripts/nersc_env.sh
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

## Where the job pool lives

`scripts/nersc_env.sh` holds the two paths and sets whatever side you are on:

```bash
source active_learning/scripts/nersc_env.sh
```

On Perlmutter (`$NERSC_HOST` is set) it exports
`RSFFF_QCHEM_ROOT=/global/cfs/cdirs/m3196/heindelj/rsfff_data` — the bundle itself, so
`config.json`, `templates/`, `scripts/` and the `eda/`/`force/` trees sit directly under that
path — and activates the conda **prefix**
`/global/cfs/cdirs/m3196/heindelj/rsfff` (`conda activate <path>`, not a name). Nothing is
rsynced: the label stage writes straight into the pool.

On a laptop it exports `REMOTE=perlmutter` and
`REMOTE_DIR=/global/cfs/cdirs/m3196/heindelj/rsfff_data`, which is all that
`qchem_roundtrip/scripts/sync_*.sh` read. The label stage runs its hooks as subprocesses, so
they inherit both. Override any of `RSFFF_POOL_REMOTE`, `RSFFF_CONDA_PREFIX`, `REMOTE` or
`RSFFF_NERSC_ACCOUNT` by exporting it before sourcing.

Q-Chem's own scratch is separate from all of this — check `$QCSCRATCH` points somewhere on
`$SCRATCH` rather than CFS before running many workers.

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

### `select` — `SelectByCommittee`

Every frame that survives this stage is two Q-Chem jobs, so this is where the loop decides what
its compute is spent on. It ranks the pool by the committee's own disagreement and keeps the
schedule's budget **per size** -- the force spread grows with cluster size for reasons that
have nothing to do with how badly that size is described, so a global ranking would quietly
relabel pentamers all iteration.

`select_on="both"` (the default) divides each frame's `sigma_energy` and `sigma_forces` by its
median over the pool and takes the larger, so the two are compared on the scale the pool itself
sets rather than through a constant nobody can justify. In a test on a 12-frame pool this
picked the highest-force-spread trimer *and* a dimer whose force spread was mid-pack but whose
energy spread was the pool's largest -- which is the point of using both.

In iteration 0 there is one starting checkpoint, so there is no spread. The stage says so and
takes an even stride through each size's pool instead of pretending to a preference.

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

### `train` — `CommitteeTrain`

Four fits of the same data, differing in their initialization, each a separate
`rsfff.train.train_film` process on a config this stage writes out in full. A separate process
is not just for parallelism: `train_film` sets the global torch default dtype and builds
module-level state, and four models in one interpreter would share it.

```
iter_000/train/committee/
    committee.json                 members, checkpoints, split seed, what it warm started from
    member_00/
        config.resolved.yaml       exactly what was fitted -- rerunnable by hand
        train.log
        member_00/best.pt          (or member_00_<laststage>/best.pt for a staged config)
    member_01/ ...
```

A member that finished writes `done.json` and is never refitted, so a driver killed by the wall
clock loses only what was in flight. `parallel="auto"` runs one member per visible GPU, each
with its own `CUDA_VISIBLE_DEVICES`.

**Warm starting** points each member's `train.init_from` at the *same member index* of the
iteration before, so member 2 always continues member 2. That is what makes the size walk
affordable. It has a cost worth watching: members that share a history agree for reasons other
than the data, and a committee like that understates its own uncertainty. `warm_start=False`
for an iteration is the remedy.

### `assess` — `ValidationAssess`

Reports, on this iteration's labeled frames, the committee's fit error and its mean spread, and
`sigma_drop` -- this iteration's mean `sigma_forces` over the last one's. Those frames were
chosen *because* the previous committee disagreed about them, so after labeling and refitting
the spread on them should have fallen; a ratio near 1 means the new data taught the model
nothing it did not already have. Note these frames are in the training set, so the fit error
flatters the model; a real holdout assessment is the next thing to build here.

## Provenance

```
$SCRATCH/water_al/
    loop.json                     initial model + data (hashed), every stage's params, environment
    summary.json                  per iteration: stage statuses and metrics
    iter_000/
        build/      stage.json  structures.extxyz  scratch/{*.inp,*.xyz,*.log}
        optimize/   stage.json  samples.extxyz
        dynamics/   stage.json  samples.extxyz
        select/     stage.json  samples.extxyz
        label/      stage.json  labeled.extxyz     scratch/{jobs.json,submitted.json,dropped.json}
        train/      stage.json  committee/{committee.json,member_NN/...}
        assess/     stage.json  metrics.json
```

Each `stage.json` holds the stage's class and parameters, its status and timings, previous
attempts, its metrics and notes, and the sha256 of its input, the model it ran with, the
training set (`train`) and its output. A frame's history is also written on the frame itself:
`sample_id` reads `packmol/n012_1|min|md900`, and `al_stem` names the pair of Q-Chem jobs.
