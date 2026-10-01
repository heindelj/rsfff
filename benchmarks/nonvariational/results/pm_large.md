# Non-variational vs variational induction, by cluster size

device `cuda` (nid003565), torch 2.14.0+cu130, torchff yes, frames per training step 1, median of 3. Film: `/global/cfs/cdirs/m3196/heindelj/software/rsfff/rsfff_active_learning/committees/film_committee_100k_excl/member_00/member_00_full/best.pt`; nonvariational (trained K=3): `/global/cfs/cdirs/m3196/heindelj/software/rsfff/checkpoints/water_nonvariational_full/best.pt`. Times in ms per call on the whole batch.

| waters | atoms | backend | model | energy | forces | train step | peak MB | CG it | residual | gap kJ/mol | gap/E_ind | dF rms kJ/mol/A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1536 | torchff | film (PCG) | 88.3 | 145.9 | 256.8 | 4028 | 20 | - | - | - | - |
| 512 | 1536 | torchff | nonvariational K=3 | 46.0 | 117.3 | 431.5 | 4059 | 3 | 0.0577 | - | - | - |
| 1000 | 3000 | torchff | film (PCG) | 153.6 | 261.6 | 423.0 | 16244 | 28 | - | - | - | - |
| 1000 | 3000 | torchff | nonvariational K=3 | 75.8 | 194.5 | 791.8 | 16279 | 3 | 0.2259 | - | - | - |

## Speedup over the film (same structure, same backend)

| waters | backend | model | energy | forces | train step |
|---:|---|---|---:|---:|---:|
| 512 | torchff | nonvariational K=3 | x1.92 | x1.24 | x0.60 |
| 1000 | torchff | nonvariational K=3 | x2.03 | x1.35 | x0.53 |
