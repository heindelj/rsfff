# Non-variational vs variational induction, by cluster size

device `cuda` (nid003565), torch 2.14.0+cu130, torchff yes, frames per training step 4, median of 5. Film: `/global/cfs/cdirs/m3196/heindelj/software/rsfff/rsfff_active_learning/committees/film_committee_100k_excl/member_00/member_00_full/best.pt`; nonvariational (trained K=3): `/global/cfs/cdirs/m3196/heindelj/software/rsfff/checkpoints/water_nonvariational_full/best.pt`. Times in ms per call on the whole batch.

| waters | atoms | backend | model | energy | forces | train step | peak MB | CG it | residual | gap kJ/mol | gap/E_ind | dF rms kJ/mol/A |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | torchff | film (PCG) | 51.6 | 97.3 | 186.8 | 43 | 12 | - | - | - | - |
| 8 | 24 | torchff | nonvariational K=2 (unrefitted) | 36.1 | 74.3 | 167.5 | 44 | 2 | 0.0092 | 0.333 | 0.0040 | 0.663 |
| 8 | 24 | torchff | nonvariational K=3 | 36.6 | 76.4 | 173.0 | 44 | 3 | 0.0090 | 0.292 | 0.0035 | 0.577 |
| 8 | 24 | torchff | nonvariational K=4 (unrefitted) | 37.4 | 78.5 | 179.1 | 45 | 4 | 0.0090 | 0.284 | 0.0034 | 0.559 |
| 16 | 48 | torchff | film (PCG) | 52.3 | 96.3 | 188.3 | 75 | 12 | - | - | - | - |
| 16 | 48 | torchff | nonvariational K=2 (unrefitted) | 36.4 | 75.0 | 168.9 | 77 | 2 | 0.0087 | 0.543 | 0.0028 | 0.497 |
| 16 | 48 | torchff | nonvariational K=3 | 37.0 | 77.0 | 175.2 | 77 | 3 | 0.0084 | 0.479 | 0.0024 | 0.431 |
| 16 | 48 | torchff | nonvariational K=4 (unrefitted) | 38.0 | 79.5 | 180.0 | 77 | 4 | 0.0084 | 0.471 | 0.0024 | 0.420 |
| 32 | 96 | torchff | film (PCG) | 59.9 | 103.6 | 196.8 | 153 | 16 | - | - | - | - |
| 32 | 96 | torchff | nonvariational K=2 (unrefitted) | 36.6 | 75.1 | 169.6 | 157 | 2 | 0.0264 | 2.977 | 0.0049 | 1.531 |
| 32 | 96 | torchff | nonvariational K=3 | 37.1 | 77.4 | 175.1 | 158 | 3 | 0.0261 | 2.626 | 0.0043 | 1.333 |
| 32 | 96 | torchff | nonvariational K=4 (unrefitted) | 38.0 | 79.7 | 180.5 | 158 | 4 | 0.0261 | 2.568 | 0.0042 | 1.297 |
| 64 | 192 | torchff | film (PCG) | 61.6 | 108.5 | 209.1 | 547 | 16 | - | - | - | - |
| 64 | 192 | torchff | nonvariational K=2 (unrefitted) | 37.3 | 79.9 | 192.3 | 553 | 2 | 0.0242 | 7.890 | 0.0054 | 2.065 |
| 64 | 192 | torchff | nonvariational K=3 | 38.5 | 83.7 | 208.6 | 553 | 3 | 0.0240 | 6.802 | 0.0047 | 1.731 |
| 64 | 192 | torchff | nonvariational K=4 (unrefitted) | 39.5 | 85.5 | 222.6 | 553 | 4 | 0.0240 | 6.580 | 0.0045 | 1.650 |
| 128 | 384 | torchff | film (PCG) | 67.3 | 125.9 | 220.4 | 2677 | 16 | - | - | - | - |
| 128 | 384 | torchff | nonvariational K=2 (unrefitted) | 41.4 | 95.5 | 284.1 | 2692 | 2 | 0.0387 | 34.409 | 0.0085 | 5.490 |
| 128 | 384 | torchff | nonvariational K=3 | 42.1 | 101.1 | 325.6 | 2693 | 3 | 0.0361 | 27.710 | 0.0068 | 4.219 |
| 128 | 384 | torchff | nonvariational K=4 (unrefitted) | 42.7 | 108.0 | 368.6 | 2695 | 4 | 0.0341 | 26.167 | 0.0064 | 3.827 |
| 216 | 648 | torchff | film (PCG) | 93.0 | 166.6 | 284.0 | 8339 | 20 | - | - | - | - |
| 216 | 648 | torchff | nonvariational K=2 (unrefitted) | 54.2 | 127.4 | 439.5 | 8365 | 2 | 0.0602 | 70.225 | 0.0096 | 7.604 |
| 216 | 648 | torchff | nonvariational K=3 | 55.1 | 140.3 | 520.2 | 8363 | 3 | 0.0454 | 54.035 | 0.0074 | 5.557 |
| 216 | 648 | torchff | nonvariational K=4 (unrefitted) | 55.7 | 153.8 | 606.7 | 8362 | 4 | 0.0361 | 50.078 | 0.0068 | 4.892 |

## Speedup over the film (same structure, same backend)

| waters | backend | model | energy | forces | train step |
|---:|---|---|---:|---:|---:|
| 8 | torchff | nonvariational K=2 (unrefitted) | x1.43 | x1.31 | x1.12 |
| 8 | torchff | nonvariational K=3 | x1.41 | x1.27 | x1.08 |
| 8 | torchff | nonvariational K=4 (unrefitted) | x1.38 | x1.24 | x1.04 |
| 16 | torchff | nonvariational K=2 (unrefitted) | x1.44 | x1.28 | x1.11 |
| 16 | torchff | nonvariational K=3 | x1.41 | x1.25 | x1.07 |
| 16 | torchff | nonvariational K=4 (unrefitted) | x1.38 | x1.21 | x1.05 |
| 32 | torchff | nonvariational K=2 (unrefitted) | x1.64 | x1.38 | x1.16 |
| 32 | torchff | nonvariational K=3 | x1.61 | x1.34 | x1.12 |
| 32 | torchff | nonvariational K=4 (unrefitted) | x1.58 | x1.30 | x1.09 |
| 64 | torchff | nonvariational K=2 (unrefitted) | x1.65 | x1.36 | x1.09 |
| 64 | torchff | nonvariational K=3 | x1.60 | x1.30 | x1.00 |
| 64 | torchff | nonvariational K=4 (unrefitted) | x1.56 | x1.27 | x0.94 |
| 128 | torchff | nonvariational K=2 (unrefitted) | x1.62 | x1.32 | x0.78 |
| 128 | torchff | nonvariational K=3 | x1.60 | x1.25 | x0.68 |
| 128 | torchff | nonvariational K=4 (unrefitted) | x1.58 | x1.17 | x0.60 |
| 216 | torchff | nonvariational K=2 (unrefitted) | x1.72 | x1.31 | x0.65 |
| 216 | torchff | nonvariational K=3 | x1.69 | x1.19 | x0.55 |
| 216 | torchff | nonvariational K=4 (unrefitted) | x1.67 | x1.08 | x0.47 |
