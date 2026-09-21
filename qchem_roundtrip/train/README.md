# Committee training job

Train a committee of film models -- N independent fits of one config that differ only in
their initialization -- on a Perlmutter GPU node. It is a job of the `qchem_roundtrip` bundle
like `active_learning/`: it travels with the bundle sync, runs from wherever the bundle lives,
and needs nothing from a checkout except an installed `rsfff`.

The default fit is the **current `film` model's config** (`configs/water_film.yaml`, what
`checkpoints/water_film_full/best.pt` was fitted with), four times, **on every water cluster we
have labels for, as one dataset**:

| directory | frames | what |
|---|---|---|
| `data/clusters/original/` | 9,579 | w2-w5, the data `water_film_full` was fitted on |
| `data/clusters/benchmark/` | 81 | w4-w23 (no w15; 36 w13 isomers), the benchmark eda+force group |
| `data/monomer/` | | h2o monomer anchor set (one-body PES, multipoles, polarizability) |
| `data/atomic_references_wb97mv_tzvpd.json` | | isolated-atom reference energies |

All wB97M-V/def2-TZVPD ALMO-EDA + analytic forces, one schema (`scripts/parse_roundtrip.py`).

### No distinction between clusters

`configs/water_film_large.yaml` brought the benchmark clusters in as `data.large_path`: a
separate stream with its own minibatch every step, its own weight (`film.large_weight`), its
own holdout and its own `lg_*` metrics. Here there is none of that. `water_film.yaml` says

```yaml
data:
  path:
    - data/clusters/**/*.xyz
```

and `train_committee.py` expands the glob (sorted, so the seeded split is identical everywhere)
into one list. `rsfff.train.data.load_cluster_datasets` then treats every frame alike: one
geometry-grouped train/holdout split (`holdout_fraction` 0.1 over all 9,660), one shuffled
minibatch stream, one loss. A w23 frame is a frame like a w2 frame. The only asymmetry left is
the physics one: the EDA error is per frame and the force term averages over atoms, so a
large cluster carries more of each than a dimer does. At 81 of 9,660 frames the large
clusters are a small share of the fit; that is what the active-learning walk up in size is for.

### Adding clusters

Any file of neutral water clusters in a new `data/clusters/<group>/` is training data from
the next fit on; no config edit. From a round-trip eda/force group:

```bash
python scripts/collect_group.py <group> --eda-dir eda/<dir> --force-dir force/<dir>
python scripts/check_data.py            # or add the line to stage_data.sh and rerun it
```

`check_data.py` (also run by `train_committee.py` before every fit) is where "water, at one
level of theory" is enforced: O/H only, every fragment an intact H2O, neutral singlet, every
label present, one method/basis across all files, and **no geometry twice**.

That last check matters for the benchmark group. Its outputs hold 162 finished eda+force pairs,
but the 76 `*_mp2_avtz` ones are the same geometries as `*_wb97mv_def2-tzvpd` jobs, with
bit-identical energies, EDA terms and forces (the geometry files were renamed and the jobs
rerun). Keeping both would double-weight them and could put one copy in training and the other
in validation. `collect_group.py` keeps one frame per geometry (in Q-Chem's standard orientation),
preferring stems that still have a `geoms/` file, so the group contributes 81 frames. Those are
the same 81 geometries as the repository's `data/wb97mv_tzvpd_large/`. The `*_mp2_avtz` jobs for
w22 and w23 have no EDA section and are dropped either way.

`data/` is gitignored (the repo-wide `data/` rule); `data.sha256` is committed and pins all
25 files. `stage_data.sh --check` also fails on any file in `data/` that is not pinned, because
the glob would train on it.

```
train/
    README.md
    water_film.yaml            the fit; data paths relative to this directory
    data.sha256                what data/ must contain
    data/                      built by scripts/stage_data.sh (not in git)
        clusters/<group>/*.xyz     every file here is training data
        monomer/  atomic_references_wb97mv_tzvpd.json
    train_committee.py         N members, one process and one GPU each -> committee.json
    scripts/
        stage_data.sh          laptop: build data/ (copy + collect_group), check, pin  (--check: verify)
        collect_group.py       a round-trip eda/force group -> data/clusters/<group>/, deduplicated
        check_data.py          water only, one level of theory, labels present, no repeats
        env.sh                 source first: the active-learning env + TRAIN_DIR
        gpu_check.py           ~1 min: does the film model train *correctly* on this GPU?
        smoke.sh               the whole job, small, on an interactive GPU node
        submit.sh              queue a fit: 1 GPU node, 4 A100s, one member per GPU
        committee.slurm        the batch script submit.sh queues
        check_committee.py     load the result the way the AL loop will, and score it
    runs/<name>/               committee outputs (not synced, not in git)
```

## Getting it to Perlmutter

On the laptop, once (and again whenever the data changes). It needs the repo's `data/` and
the benchmark eda/force outputs, and numpy (no pyscf):

```bash
bash qchem_roundtrip/train/scripts/stage_data.sh
source qchem_roundtrip/active_learning/scripts/nersc_env.sh
bash qchem_roundtrip/scripts/sync_inputs_up.sh       # carries train/ (minus runs/) with the bundle
```

The GPU nodes also need the neighbor-list fix in `src/neighbors.py` (below), so pull the
checkout the env uses (`$RSFFF_REPO`, `/global/cfs/cdirs/m3196/heindelj/software/rsfff`).

## 1. Is the GPU training correctly? (`smoke.sh`, ~10 min)

```bash
salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node 4 -c 128 -t 00:30:00
module load python
source /global/cfs/cdirs/m3196/heindelj/rsfff_data/train/scripts/env.sh
bash $TRAIN_DIR/scripts/smoke.sh
```

It verifies the data pins, runs `gpu_check.py`, trains a 4-member committee on the first 64
frames of every cluster file (so every benchmark cluster, w4-w23) for one epoch per stage (through exactly the code path of the batch job), and loads it
back on the CPU with `active_learning/committee.py`.

`gpu_check.py` is the part that answers "does it train properly", check by check:

- **environment** -- torch has CUDA, the device, e3nn, which neighbor backend is active
- **neighbor list on device** -- same edges on GPU as on CPU
- **parameters/buffers on device** -- nothing left on the CPU after `.to(cuda)`
- **loss terms CPU == GPU**, **parameter gradients CPU == GPU** -- on a batch of the larger
  benchmark clusters (they sort first), one full-stage loss
  (EDA channels, the induction CG solve, forces as a second-order backward, the fragment,
  monomer-anchor and regularizer streams) from identical weights in float64 on both devices.
  They must agree to 1e-7 (terms) and 1e-5 (worst gradient element, relative to its tensor's
  largest) -- float64 differs only by summation order and the CG tolerance. This is the
  check that the GPU computes *the same fit*, not merely *a* fit
- **induction CG on device** -- same iteration count, no failures
- **loss falls on device** -- 15 Adam steps on one batch
- **timing** -- s/step on CPU and GPU, peak memory, and a rough full-data epoch time. It is
  measured on w10+ clusters, so it overestimates a w2-w5-dominated epoch. Use it to set the
  wall clock (`-t`) for the real run

Verified here on CPU only (no GPU in reach): the check itself passes CPU-vs-CPU with zero
difference, the committee trains and resumes, warm starting works, and the output loads
through `Committee.load`. The GPU run is the part still to do -- that is what `smoke.sh` is for.

## 2. The real fit (`submit.sh`)

```bash
source /global/cfs/cdirs/m3196/heindelj/rsfff_data/train/scripts/env.sh
bash $TRAIN_DIR/scripts/submit.sh --name film_committee            # 4 members, regular QOS, 12 h
bash $TRAIN_DIR/scripts/submit.sh --name film_committee -t 06:00:00   # after reading the timing
```

The batch job runs `gpu_check.py` first (about a minute; a failure stops it before four GPUs
spend the wall clock on a broken environment -- `GPU_CHECK=0` skips it), then
`train_committee.py`, then `check_committee.py` on the result. Everything after `--` goes to
`train_committee.py`:

```bash
bash $TRAIN_DIR/scripts/submit.sh --name film_c8 -- --members 8 --seed 100   # 2 rounds of 4
bash $TRAIN_DIR/scripts/submit.sh --name film_bs64 -- --set train.batch_size=64
bash $TRAIN_DIR/scripts/submit.sh --name film_ws -- --init-from $TRAIN_DIR/runs/film_committee
```

Account/QOS/time default from `RSFFF_GPU_ACCOUNT` (default `m3196`; if sbatch refuses a GPU
job on it, use the `m3196_g` form), `RSFFF_GPU_QOS` (`regular`) and `RSFFF_GPU_TIME`.
A member with `done.json` is never refitted, so resubmitting the same `--name` after a wall
clock kill trains only what is missing (a member cut off mid-fit restarts from scratch).

```
runs/film_committee/
    committee.json          members, checkpoints, val losses, seeds, config + data sha256, GPUs
    slurm-<job>.out         gpu_check, progress, check_committee table
    member_00/
        config.resolved.yaml   rerunnable by hand: python -m rsfff.train.train_film <it>
        train.log
        done.json
        member_00_isolated/best.pt
        member_00_full/best.pt   <- the member's checkpoint
```

`check_committee.py runs/<name>` loads the committee on the CPU (where the AL loop's sampling
runs) and reports per cluster size the committee-mean energy and force error against the labels
and the member spread (`sigma_E`, `sigma_F` -- the same numbers the select stage ranks on).
These frames include training frames, so it is a sanity check, not a holdout score. For
reference, the current `water_film_full` checkpoint gives E_MAE 0.8-1.2 kJ/mol per frame and
F_MAE 2.5-3.5 kJ/mol/A on the first 10 frames of each w2-w5 file. The benchmark rows are the
ones to watch: see `configs/water_film_large.yaml` for how badly w19+ induction extrapolated
from w2-w5 alone.

## Using the committee in the active-learning loop

`committee.json` is the manifest `active_learning/train_stage.py` writes, so the loop reads it
as-is: `--checkpoint $TRAIN_DIR/runs/film_committee` gives iteration 0 a real committee (a
spread to select on instead of the even stride) and warm starts member k from member k.

## Environment

`env.sh` sources `active_learning/scripts/nersc_env.sh` (the conda prefix
`/global/cfs/cdirs/m3196/heindelj/rsfff`). That env was built for the CPU driver; if
`gpu_check.py` says `torch.cuda.is_available() is False`, its torch is CPU-only and the GPU
job needs a CUDA torch. The simplest is NERSC's own, with rsfff on top:

```bash
module load pytorch                                  # CUDA torch for the A100s
python -m venv --system-site-packages $CFS/m3196/heindelj/rsfff_gpu
source $CFS/m3196/heindelj/rsfff_gpu/bin/activate
pip install e3nn ase pyyaml
pip install --no-deps -e /global/cfs/cdirs/m3196/heindelj/software/rsfff
cat > $CFS/m3196/heindelj/rsfff_gpu/setup.sh <<'SH'
module load pytorch
source $CFS/m3196/heindelj/rsfff_gpu/bin/activate
SH
export RSFFF_TRAIN_SETUP=$CFS/m3196/heindelj/rsfff_gpu/setup.sh   # env.sh sources it last
```

`RSFFF_TRAIN_SETUP` is exported into the batch job (`--export ALL`), so set it before
`submit.sh`.

### torch_cluster on GPU

The install notes build torch_cluster with `FORCE_ONLY_CPU=1`, which is right for the CPU
nodes and makes every CUDA call raise "Not compiled with CUDA support" -- the first neighbor
list of a GPU fit would kill it. `rsfff.neighbors.build_radius_graph` now probes that once and
falls back to `radius_graph_torch` (the same graph, on the device) for CUDA tensors, with one
warning. `gpu_check.py`'s neighbor check fails loudly if the checkout predates the fix.

## What differs between members, and what doesn't

Only `train.seed` (member k: `--seed + k`), i.e. the initialization. `data.seed` is shared, so
every member has the same train/holdout split and sees minibatches in the same order -- as in
the loop's train stage. That keeps the members' disagreement about the model rather than
about which frames each was shown; it also means it measures initialization variance only.
