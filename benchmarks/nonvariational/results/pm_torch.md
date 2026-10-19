# Non-variational vs variational induction, by cluster size

device `cuda` (nid003565), torch 2.14.0+cu130, torchff yes, frames per training step 4, median of 3. Film: `/global/cfs/cdirs/m3196/heindelj/software/rsfff/rsfff_active_learning/committees/film_committee_100k_excl/member_00/member_00_full/best.pt`; nonvariational (trained K=3): `/global/cfs/cdirs/m3196/heindelj/software/rsfff/checkpoints/water_nonvariational_full/best.pt`. Times in ms per call on the whole batch.

| waters | atoms | backend | model | energy | forces | train step | peak MB | CG it | residual | gap kJ/mol | gap/E_ind | dF rms kJ/mol/A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | torch | film (PCG) | 77.5 | 181.3 | 416.5 | 77 | 12 | - | - | - | - |
| 8 | 24 | torch | nonvariational K=3 | 66.1 | 172.2 | 436.8 | 91 | 3 | 0.0090 | - | - | - |
| 16 | 48 | torch | film (PCG) | 79.0 | 183.0 | 417.0 | 224 | 12 | - | - | - | - |
| 16 | 48 | torch | nonvariational K=3 | 66.9 | 174.7 | 434.1 | 276 | 3 | 0.0084 | - | - | - |
| 32 | 96 | torch | film (PCG) | 88.4 | 194.2 | 431.6 | 773 | 16 | - | - | - | - |
| 32 | 96 | torch | nonvariational K=3 | 67.5 | 175.3 | 441.9 | 1034 | 3 | 0.0261 | - | - | - |
| 64 | 192 | torch | film (PCG) | 93.5 | 202.9 | 455.1 | 2625 | 16 | - | - | - | - |
| 64 | 192 | torch | nonvariational K=3 | 68.5 | 179.0 | 449.7 | 3604 | 3 | 0.0240 | - | - | - |

## Speedup over the film (same structure, same backend)

| waters | backend | model | energy | forces | train step |
|---:|---|---|---:|---:|---:|
| 8 | torch | nonvariational K=3 | x1.17 | x1.05 | x0.95 |
| 16 | torch | nonvariational K=3 | x1.18 | x1.05 | x0.96 |
| 32 | torch | nonvariational K=3 | x1.31 | x1.11 | x0.98 |
| 64 | torch | nonvariational K=3 | x1.37 | x1.13 | x1.01 |
