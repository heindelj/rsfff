# Active learning for the water film model

An `easyal` loop that grows the training set for the range-separated force-field functional by
sampling with the model itself and labeling what it finds with Q-Chem.

```
build      packmol packs (H2O)n into a spherical cavity         structures.extxyz
optimize   relax every packing on the model's own surface       samples.extxyz
dynamics   Langevin NVT from each minimum, inside a soft wall    samples.extxyz
label      Q-Chem EDA + force on the selected frames            labeled.extxyz     (TODO)
train      refit the film model on everything labeled so far    model              (TODO)
assess     measure the new model on held-out structures         metrics.json       (TODO)
```

`build`, `optimize` and `dynamics` are implemented. The placeholders for the rest live in
`run_water.py`; `label` raises `Pending`, so a run today stops after `dynamics` with everything
it produced on disk, and resumes at `label` once that stage exists.

## Running it

```bash
conda install -c conda-forge packmol        # or: pip install packmol
pip install -e ../easyAL                    # the loop itself
python active_learning/run_water.py --root runs/water_al \
    --checkpoint checkpoints/water_film_full/best.pt --sizes 2 20 --per-size 2
```

`--help` lists every knob. All of them land in `stage.json`, so a run describes itself.

Nothing here needs the repository to be the current directory. `--root` and `--checkpoint` can
be anywhere, and `run_water.py` finds its own package from `__file__`, so on a cluster this is

```bash
python /path/to/rsfff/active_learning/run_water.py \
    --root $SCRATCH/water_al --checkpoint /path/to/checkpoints/water_film_full/best.pt
```

from any working directory, with `rsfff` pip-installed. A checkpoint carries its own
isolated-atom reference energies, so no part of loading a model reads `data/`.

## Why the sampling is two stages

They answer different questions and fail differently. The minimizer asks *where are this
model's minima*; a packing it cannot relax is a statement about the model. The dynamics asks
*what does this model do at temperature near those minima*; its failure modes are heating,
evaporation and a broken molecule. Two `Sample` stages (`easyal` takes a list) give each its own
directory, contract, hashes and metrics, and let either be redone alone:

```python
loop.reset(0, "dynamics")     # redo the trajectories, keep the minima
```

Running the dynamics *from the minima* rather than from the packings is the point of the
ordering. A random packing carries tens of kcal/mol of strain; a trajectory started there
spends its first picosecond dumping that into the thermostat and samples nothing worth an
EDA calculation.

## Stages

### `build` — `PackmolWaterClusters`

One packmol run per structure: `inside sphere 0 0 0 R` with `R` set from the cluster size at a
chosen density,

    R = (3 n V_w / 4 pi)^(1/3) + padding,    V_w = 30.0 A^3 at 1 g/cm^3

so a cluster starts near liquid density instead of as a gas that has to collapse. Packmol
writes molecules consecutively, so the output is already `O H H O H H ...`, which is the
fragment order `film_driver.water_fragment_index` requires. Every structure gets its own seed
(shifted by iteration, so iteration 1 packs different structures), its own `.inp` and its own
log in `scratch/`.

Frames carry `charge`, `multiplicity`, `n_waters`, `cavity_radius`, `packmol_seed`,
`packmol_converged` and a `source` tag that follows them through the rest of the loop.

### `optimize` — `MinimizeSample`

`film_driver.optimize` (L-BFGS-B on the analytic gradient) per structure, to `gtol = 1e-3`
Hartree/A by default. These are trajectory starting points, not spectroscopy: a 300 K
trajectory carries `|F|max` around 0.07 Hartree/A, so 1e-3 is already well inside the thermal
noise the structure is about to be handed, and the driver's own 1e-7 only buys iterations.
Tighten it if the minima are also going to a Hessian. Structures that do not
converge are dropped (`keep="all"` keeps them), and so is anything that stops being a set of
intact waters. Frames gain `model_energy`, `model_energy_per_water`, `relaxation_energy`,
`opt_max_force`, `opt_converged`, `opt_iterations`.

### `dynamics` — `DynamicsSample`

ASE `Langevin` at 300 K, 0.5 fs, friction 0.02/fs, with `confine.flat_bottom_sphere` around the
running center of mass so a cluster cannot simply evaporate. The wall is added in the
calculator, not in the model: it is a restraint on the sampling, and the `model_energy` stored
on a frame excludes it.

A trajectory is stopped — and everything it produced before that kept — when the energy is not
finite, the temperature passes `max_temperature`, `|F|max` passes `max_force`, or a water stops
being intact. The fragmentation is fixed at the start of each trajectory, as the film model
requires, which is exactly why that last check is worth making.

Frames gain `model_energy`, `temperature`, `time_fs`, `md_step`, `wall_energy`, `wall_radius`.

## Provenance

```
runs/water_al/
    loop.json                     initial model + data (hashed), every stage's params, environment
    summary.json                  per iteration: stage statuses and metrics
    iter_000/
        build/      stage.json  structures.extxyz  scratch/{*.inp,*.xyz,*.log}
        optimize/   stage.json  samples.extxyz
        dynamics/   stage.json  samples.extxyz
        label/      stage.json  labeled.extxyz
        train/      stage.json  model
        assess/     stage.json  metrics.json
```

Each `stage.json` holds the stage's class and parameters, its status and timings, previous
attempts, its metrics and notes, and the sha256 of its input, the model it ran with, the
training set (`train`) and its output. A frame's history is also written on the frame itself:
`source` reads `packmol/n012_1|min|md900`.
