# Bonded model: an end-to-end example

Generate reference data with pyscf, fit one bonded model to four small species, and test it on scans it never saw.
Everything runs on a laptop: the level of theory is wB97X/def2-SVP, chosen for speed, not accuracy.

The model (`rsfff.ff.bonded`) is a conventional bonded force field (Morse bonds, cosine angles, bond-bond,
bond-angle, angle-angle and torsion terms) plus a split-charge / atomic-multipole / polarizability model of the
electrical properties. Its parameters are not constants: one small network turns each bond's geometric environment
into an 8-number latent, and every parameter is a bounded linear readout of those latents around a per-type table.

## Install

    pip install -e ~/dev/bb            # the network building blocks (import name: bb)
    pip install -e ~/dev/rsfff
    pip install pyscf matplotlib

## Run (from the rsfff checkout)

    WORKERS=8 bash examples/bonded/run.sh

or step by step:

    # 1. data: optimize, sample and label H2O, H3O+, OH- and H2O2
    python scripts/bonded_data.py all --species h2o h3o+ oh- h2o2 --out examples/bonded/data \
        --xc wb97x --basis def2-svp --grid 75 302 --wigner 40 --points 7 --raman --analytic-hessian --workers 8
    # 2. fit
    python -m rsfff.train.bonded examples/bonded/config.toml
    # 3. look
    python examples/bonded/analysis.py examples/bonded/run        # or open analysis.ipynb

The labelled data this produced is in `data/` already, so steps 2 and 3 run without step 1.
Measured on 2 cores: step 1 about 1 hour (290 frames, 13 SCF + gradient each), step 2 about 9 minutes.

## What each step does

1. **Data** (`scripts/bonded_data.py`, pyscf only). Per species: optimize; Hessian and frequencies; 40 normal-mode
   samples from the harmonic Wigner distribution at 600 K; rigid scans of every bond (0.75-1.6 r_e), every angle
   (+-35 degrees), the H3O+ umbrella and the H2O2 torsion. Each frame gets energy, forces, dipole, second moment,
   polarizability, dipole derivatives and polarizability derivatives, all from finite fields. `data/<name>.xyz` is
   the labelled set, `data/<name>_opt.xyz` the minimum (with its Hessian).
2. **Fit** (`config.toml`). `[terms]` lists the force-field terms, `[encoder]` the network sizes, `[data]` the files.
   `holdout = "scan"` keeps every scan frame out of training and model selection. The trainer first fits the
   constant-parameter model (tables alone), then trains the network on top of it.
3. **Analysis** (`analysis.py`): errors per species, and one panel per held-out scan (`run/scans.png`).

## What to expect (`expected/`, one seed)

Frames like the training frames (validation; the constant-parameter model in parentheses):

| species | E mae, kcal/mol | F rmse, kcal/mol/A | dipole, e bohr | alpha, bohr^3 | dmu/dR, e |
|---|---|---|---|---|---|
| H2O | 0.02 (0.75) | 0.6 (5.1) | 0.0008 (0.064) | 0.015 (0.35) | 0.008 (0.078) |
| H3O+ | 0.03 (0.23) | 0.6 (1.6) | 0.0009 (0.035) | 0.009 (0.22) | 0.007 (0.074) |
| OH- | 0.01 (0.86) | 0.4 (1.3) | 0.003 (0.054) | 0.008 (0.19) | 0.009 (0.25) |
| H2O2 | 0.09 (0.56) | 1.1 (6.9) | 0.0015 (0.038) | 0.036 (0.36) | 0.011 (0.084) |

Harmonic frequencies at the model's minimum (reference), cm^-1: H2O 1642 3891 3989 (1644 3893 3992);
H3O+ 755 1623 1623 3623 3735 3735 (767 1622 1623 3620 3731 3731); OH- 3564 (3565);
H2O2 370 1038 1360 1494 3854 3855 (353 1042 1355 1494 3854 3856).

Held-out scans (never trained on; they reach 60-110 kcal/mol, far beyond the samples):

| scan | frames | E mae, kcal/mol | largest error | energy range |
|---|---|---|---|---|
| H2O bend / stretch | 7 / 14 | 0.3 / 1.2 | 1.7 / 4.8 | 23 / 71 |
| H3O+ bend / stretch / umbrella | 21 / 21 / 13 | 0.4 / 1.8 / 0.07 | 1.3 / 7.5 / 0.2 | 21 / 63 / 3 |
| OH- stretch | 7 | 0.3 | 0.8 | 59 |
| H2O2 bend / stretch / torsion | 14 / 21 / 12 | 1.1 / 1.9 / 1.2 | 3.6 / 14 / 5.5 | 58 / 113 / 9 |

The O-H stretches, the bends and the umbrella extrapolate well. Two do not: the O-O stretch at 1.6 r_e, and the
H2O2 torsion, whose trans barrier the model puts at about half its height. The samples only cover torsion angles
near the minimum, so the barrier is not in the training data. With the scans in the training set instead
(`--set 'data.holdout=""'`), validation errors over samples and scans together are 0.06-0.15 kcal/mol and
0.2-1.9 kcal/mol/A.

## Your own molecule

    python scripts/bonded_data.py reference --xyz mol.xyz --name mol --charge 0 --out mydata --xc wb97x --basis def2-svp --analytic-hessian
    python scripts/bonded_data.py sample mydata/mol_opt.xyz --out mydata/mol_geoms.xyz --wigner 100 --scans
    python scripts/bonded_data.py label mydata/mol_geoms.xyz --out mydata/mol.xyz --raman --workers 8 --xc wb97x --basis def2-svp

The reference stage perceives the covalent bonds once, from your starting structure, and writes them into every
file; nothing later guesses bonds from distances. Then copy `config.toml`, point `[data]` at the new files, and add
`"improper"` to `[terms] bonded` if the molecule has planar three-coordinate centres.

## Limits

- **Molecules where every pair of atoms is within three bonds.** There are no intramolecular non-bonded terms, so
  anything larger than ethane- or methanol-sized is missing physics.
- Closed shells only have been run. Atom types are (element, number of neighbours): water's and peroxide's O-H share
  a table and are told apart by the network alone.
- OH- in def2-SVP has no diffuse functions: 46 of its 47 frames have a positive HOMO energy. Fine for an example;
  use a diffuse basis (the script's default is wB97M-V/def2-TZVPD) for real work.
- The split between charge-flow and atomic polarizability is constrained by the derivative labels but not pinned:
  no electrostatic-potential labels are fitted yet.
- `--backend gpu4pyscf` exists and has never been run.
