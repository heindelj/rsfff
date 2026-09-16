# Active learning

Grows the training set toward larger clusters with a loop of four stages per iteration:

| stage    | template           | reads                                   | writes (standard location)               |
|----------|--------------------|-----------------------------------------|------------------------------------------|
| `build`  | `BuildStructures`  | `ctx.checkpoint`, `ctx.history(...)`    | `structures.extxyz`                      |
| `sample` | `SampleDynamics`   | `ctx.input("structures")`, checkpoint   | `trajectories/`, `candidates.extxyz`     |
| `label`  | `LabelFrames`      | `ctx.input("candidates")`               | `selected.extxyz`, `dataset/`            |
| `train`  | `TrainModel`       | `ctx.training_data()`, checkpoint       | `best.pt`, `config.yaml`                 |

To add a stage, subclass its template and implement `run(ctx)`, which only has to write
`ctx.output(key)`. For `label`, `run` is already written, so you implement `select(ctx, path)`
and, optionally, `evaluate(ctx, files)`. `run_campaign.py` has the four subclasses ready to
fill in and wires them into a campaign.

```bash
python -m active_learning.run_campaign active_learning/campaigns/water_large --iterations 3
python -m active_learning.run_campaign active_learning/campaigns/water_large --status
```

## Frames between stages

Frames are passed as extxyz files. Every frame needs `charge` and `multiplicity`. Frames in
`structures` and `candidates` also need a `structure_id`. Frames in `selected.extxyz` need
the EDA fragment definition, in the same format the Q-Chem generator reads:

```
Properties=species:S:1:pos:R:3:fragment_idx:I:1 charge=1 multiplicity=1 fragment_charges="0 0 1" fragment_multiplicities="1 1 1" structure_id=...
```

## Labeling (Q-Chem)

`LabelFrames.run` calls `active_learning/qchem.py`:

1. **`select`** writes `selected.extxyz`. It runs once, and the file is kept when the stage resumes.
2. **`qchem.write_jobs`** writes one EDA input and one force input per frame into
   `qchem_roundtrip/{eda,force}/al_<campaign>/iter_NNN/`. It uses `templates/{eda,force}.in`
   and the molecule mode from `config.json`, so the jobs match `eda/ion_clusters` and
   `force/ion_clusters` byte for byte. The Perlmutter workers already find nested job dirs.
3. **`qchem.wait_for_jobs`** raises `StagePending` until every output is back. The campaign
   records the stage as `pending` and stops. Then:
   ```bash
   bash qchem_roundtrip/scripts/sync_inputs_up.sh
   # on Perlmutter: bash scripts/submit_workers.sh --target N
   bash qchem_roundtrip/scripts/sync_outputs_down.sh
   python -m active_learning.run_campaign <campaign> --iterations 3   # resumes
   ```
   An output counts as finished only if its last lines contain Q-Chem's closing line.
   Crashed jobs can't be told apart from running ones, so they keep the stage pending. Once
   you've checked, pass `allow_unfinished=True` to the label stage and the parser will drop
   those frames.
4. **`qchem.parse_jobs`** runs `scripts/parse_roundtrip.py` and writes the merged
   EDA and force data to `dataset/`.
5. **`evaluate`** scores the model *before* it trains on the new frames. Its metrics are
   stored as `pre_*`. This is the error that shows whether active learning is converging.

## Provenance

Each stage directory gets a `stage.json` with:

- `stage`: the class and its constructor params. Put every knob in the params.
- `inputs`: sha256, size and frame count of everything the stage read through `ctx`:
  upstream outputs, the starting checkpoint, and files declared with `ctx.add_input`
  (Q-Chem templates, `config.json`, the parser, the base training YAML).
- `training_data`: the full, ordered, hashed training set the model was fit on.
- `outputs` / `extra_output_fingerprints`: hashes of what the stage wrote, including the
  Q-Chem job dirs.
- `metrics`, `runtime_params`, `notes`, and `attempts` (earlier pending or failed tries).
- `code`: the git commit and branch. If the tree is dirty, the stage dir also gets a
  `code.patch`, and new `.py` files are listed. Untracked files are *not* saved, so
  commit new code before a run you care about.
- `environment`: host, Python, conda env, and versions of torch, numpy, ase, e3nn and pyscf.

At the campaign level:

- `campaign.json` fixes the starting checkpoint and base data. If you later construct the
  campaign with different ones, it raises `ProvenanceError`.
- `events.jsonl` is an append-only log.
- `summary.json` has one row per stage with its status and metrics.

**Re-running** is safe. A completed stage is skipped unless one of these happened:

- a recorded input changed,
- one of its outputs changed on disk,
- its params changed,
- the training set gained or lost a file,
- you asked for it with `--rerun ITER:STAGE`.

A stage that re-runs has its old directory moved to `iter_NNN/_superseded/`, never deleted.
Downstream stages re-run only if the new outputs actually differ. Pass `--on-stale keep` or
`--on-stale error` to change this behavior.

`data/` is gitignored, so these hashes are the only record of which data a model saw.
`.gitignore` here keeps the bulk files under `campaigns/` out of git but keeps the JSON records.

## Tests

`tests/test_active_learning.py` tests the bookkeeping with toy stages. It also runs the whole
label stage, with the existing ion-cluster outputs standing in for Perlmutter.
