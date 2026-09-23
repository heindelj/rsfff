# rsfff_active_learning

Active learning for the film water model, replacing `qchem_roundtrip`: easyAL runs the loop,
a cc_workers store owns every Q-Chem job and is where training data comes from. The code is
versioned on the rsfff branch `nonreactive`; data (`runs/`, `committees/`, checkpoints,
trajectories, the anchor set) is not -- see `.gitignore` -- and moves with `scripts/sync.sh`.
Everything runs on Perlmutter. Plan and progress: the **active learning** board in Obsidian
(`RSFFF/active_learning_board.md`, notes in `active_learning_board_notes/`).

```
build     PackmolBuild          (H2O)n packings for this iteration's sizes, not minimized
dynamics  UncertaintyDynamics   committee-biased Langevin, replicas batched per size    GPU
select    SelectUncertain       most uncertain per size; EDA subset marked
label     QChemStoreLabel       force (+eda2) specs into the store; waits; joins results   CPU jobs
train     CommitteeTrain        export from the store -> 4-member committee              4-GPU job
assess    CommitteeAssess       new vs old committee on the new labels, per size
```

```
rsfff_al/
    committee.py    K film models on a batch of replicas: E, F, sigma_E, grad sigma_E, sigma_F
    dynamics.py     biased BAOAB Langevin, guards + rewind, checkpoint/resume
    build.py        packmol packings
    select.py       candidate pool, per-size selection, EDA subset
    schedule.py     the size walk 2 -> 64 waters, labels and EDA fraction per size
    stages.py       build / dynamics / select stages
    store_io.py     frame -> Q-Chem specs; force (+eda2) results -> rsfff training frame
    label.py        the label stage
    dataset.py      training set out of the store (python -m rsfff_al.dataset STORE OUT)
    train_stage.py  the train stage
    assess.py       score_frames + the assess stage
train/
    train_committee.py, water_film_al.yaml, anchors/ (monomer set, atomic references)
    scripts/committee.slurm, gpu_check.py, check_data.py, check_committee.py
scripts/
    run_loop.py      the loop (--config configs/water_udd.yaml | --quick)
    driver.slurm     runs the loop until pending, requeues itself
    udd.py           one size outside the loop; calibrate_udd.sh + summarize_udd.py for AL1
    fake_qchem.py    model labels instead of Q-Chem, for dry runs (--quick) only
    check_export.py  does a store export reproduce the old training files?
    sync.sh, env_perlmutter.sh
configs/water_udd.yaml             the production loop
committees/film_committee_100k/    the starting committee
tests/                             pytest; test_legacy_equivalence needs RSFFF_REPO
```

## Sampling

`F = (1 - w) Fbar + w kappa grad sigma_E + F_wall`, `w = 0.2`. `grad sigma_E` is a weighted
sum of the members' own gradients (no second derivatives). `bias_mode="matched"` (default)
scales it to `|Fbar|` so the step is 80/20 by magnitude; `"raw"` keeps a constant kappa
(conservative; kappa = 1 is ~1% of the force for this committee). Guards: non-finite, |Fbar| >
0.5 Ha/A, T > 3x target, O-H > 1.3 A, intermolecular O-O < 2.0 / O-H < 1.15 / H-H < 1.0 A;
a failure rewinds 200 fs with new velocities (5 times, then the replica retires).

## Labels and training data

Selected frames become `force` and (per the EDA policy) `eda2` specs at wB97M-V/def2-TZVPD,
identical to what the qchem_roundtrip outputs adopt to -- so legacy data and AL labels are one
pool and nothing is computed twice. `training_frame` joins the two results into the schema
`scripts/parse_roundtrip.py` wrote (checked bitwise: `tests/test_legacy_equivalence.py`).
The train stage exports every done water job at that level from the store: frames with EDA
to `data.path`, force-only frames to `data.force_path` (a separate stream, total energy per
water + forces; rsfff commit 2ae0d86 on `nonreactive`), anchors always in.
Every AL frame carries `split_group = traj_id`, so validation holds out whole trajectories.

## Running

```bash
bash scripts/sync.sh up                                        # laptop -> Perlmutter
source /global/cfs/cdirs/m3196/heindelj/rsfff_active_learning/scripts/env_perlmutter.sh
python scripts/run_loop.py --root $SCRATCH/al_quick --quick   # every stage, CPU, fake labels
sbatch -A m3196 --export=ALL,AL_CONFIG=$RSFFF_AL/configs/water_udd.yaml $RSFFF_AL/scripts/driver.slurm
python scripts/run_loop.py --config configs/water_udd.yaml --status
touch $SCRATCH/rsfff_al/water_udd/STOP                         # end the driver chain
```

## Tests

```bash
python -m pytest tests -q                    # ~1 min on a CPU
RSFFF_REPO=~/dev/rsfff python -m pytest tests/test_legacy_equivalence.py
```
