# Q-Chem Round-Trip Workers

This directory is a self-contained work bundle for Perlmutter Q-Chem labeling.
It contains the templates, scripts, config, and calculation folders needed by
the generator and worker.

Each calculation type has:

- `geoms/` - drop one or more extxyz `.xyz` files here.
- `inputs/` - generated `.in` files land here.
- `outputs/` - Q-Chem `.out` files are written here.

## Generate Inputs

From this repository:

```bash
python3 qchem_roundtrip/scripts/qchem_roundtrip.py \
  --root qchem_roundtrip \
  --config qchem_roundtrip/config.json \
  generate
```

The generator expands multi-frame extxyz files into one input per frame. A
single-frame file named `sample.xyz` becomes `inputs/sample.in`; a multi-frame
file becomes `inputs/sample_frame0000.in`, `inputs/sample_frame0001.in`, and so
on, with frame indices starting at 0.

Each frame header must include system charge and multiplicity:

```text
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
```

When a calculation uses fragmented Q-Chem molecule input, the same frame must
also include per-atom `fragment_idx` plus fragment charges and multiplicities:

```text
Properties=species:S:1:pos:R:3:fragment_idx:I:1 charge=0 multiplicity=1 n_fragments=2 fragment_charges="0 0" fragment_multiplicities="1 1"
```

The files in `data/eda_data/*.xyz` are examples of the intended format.
Both `.xyz` and `.extxyz` extensions are scanned.

## Interactive Worker Test

Inside an interactive Perlmutter allocation, from the remote `qchem_roundtrip`
directory:

```bash
module load python
module load qchem
QCHEM_THREADS=128 bash scripts/worker.sh
```

The worker regenerates inputs on every polling loop, atomically claims one
runnable input, runs Q-Chem from that calculation directory, and writes the
matching output into `outputs/`. If it finds no runnable work for 30 minutes, it
exits cleanly.

## Queued Workers

From the remote `qchem_roundtrip` directory:

```bash
sbatch scripts/worker.slurm
```

The Slurm wrapper loads both `python` and `qchem`, then starts the same worker
script used for interactive testing.

To keep a fixed pool of generic workers active or pending, use the pool
submitter from the remote `qchem_roundtrip` directory:

```bash
bash scripts/submit_workers.sh --target 8
```

That command counts current `qchem_worker` jobs submitted from this exact
roundtrip directory and submits only the missing number. It never cancels extra
jobs.

For an active feed where new inputs may appear over time, run the submitter in
watch mode:

```bash
bash scripts/submit_workers.sh --target 8 --watch --interval 300
```

Useful knobs:

```bash
export QCHEM_WORKER_TARGET=8
export QCHEM_THREADS=128
export QCHEM_IDLE_TIMEOUT_SECONDS=1800
export QCHEM_POLL_SECONDS=30
```

The Slurm account, queue, constraint, wall time, and node count live in
`scripts/worker.slurm`.

## Job Priority

Workers claim inputs by calculation priority first, then by input filename. The
priority is configured per calculation in `config.json`; higher numbers run
first:

```json
"force": {
  "priority": 20,
  "template": "templates/force.in",
  "molecule": {"mode": "plain"}
}
```

Missing priorities default to `0`. Existing running workers keep the script and
config they loaded when they started, so priority changes apply to newly started
workers after the updated bundle is synced.

## Rsync Setup

Add a normal SSH alias for Perlmutter if you do not already have one:

```sshconfig
Host perlmutter
    HostName perlmutter.nersc.gov
    User YOUR_NERSC_USERNAME
    IdentityFile ~/.ssh/nersc
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

NERSC's `sshproxy` certificate is usually valid for 24 hours. If SSH starts
asking for password+OTP again, renew the certificate:

```bash
sshproxy -u YOUR_NERSC_USERNAME
ssh-keygen -L -f ~/.ssh/nersc-cert.pub | grep Valid
```

Set your Perlmutter host alias and destination once per shell:

```bash
export REMOTE=perlmutter
export REMOTE_DIR=/pscratch/sd/j/$USER/qchem_roundtrip
```

If you generate inputs locally, push the bundle plus `inputs/`:

```bash
bash qchem_roundtrip/scripts/sync_inputs_up.sh
```

If you want Perlmutter to generate inputs from extxyz geometries, push
geometries instead and run the generator or worker remotely:

```bash
bash qchem_roundtrip/scripts/sync_geoms_up.sh
```

Pull outputs back to the laptop:

```bash
bash qchem_roundtrip/scripts/sync_outputs_down.sh
```

The rsync helpers intentionally sync only their target folders. They do not
delete remote outputs or local outputs.

## EDA Fragments

EDA inputs need fragment separators in the `$molecule` block. For `eda`, the
generator reads those fragments from `fragment_idx`, `fragment_charges`, and
`fragment_multiplicities` in each extxyz frame rather than from `config.json`.
