# ECE-fed bonded heads: PySCF pilot (b3lyp/def2-svpd), stretch scans held out

Train: 247 non-stretch frames of the 7 pilot molecules (minima, Wigner, torsion and inversion
scans), 10% validation. Test: all 144 `scan_stretch` frames. Loss: energy + forces + Hessian-vector
products (2 probes/step, scale 0.02 Ha/A^2), `bond_var_weight` 1e-3 unless noted. Single seed,
CPU, float64. Configs from `make_cfg.py` (`big=1` for the 200-epoch runs), evaluated with
`scripts/eval_bonding.py`. Units: val E kJ/mol/atom (trainer metric), F kcal/mol/A, H rel = relative
Frobenius error; stretch E is the absolute (atom-referenced) energy MAE in kcal/mol.

| run | epochs | val E | val F | val H | train F | train H | stretch E | stretch F | stretch H |
|---|---|---|---|---|---|---|---|---|---|
| latent (SOAP only, main) | 50 | 0.54 | 1.06 | 0.028 | 0.74 | 0.022 | 1.28 | 1.79 | 0.029 |
| ece + latents | 50 | 0.43 | 0.95 | 0.036 | 0.51 | 0.016 | 1.21 | 1.72 | 0.030 |
| ece only | 50 | 0.44 | 1.19 | 0.032 | 0.75 | 0.021 | 1.30 | 1.90 | 0.037 |
| latent | 200 | 0.32 | 0.82 | 0.019 | 0.31 | 0.011 | 0.98 | 1.32 | 0.022 |
| ece + latents | 200 | 0.18 | 0.68 | 0.026 | 0.13 | 0.006 | 0.94 | 1.29 | 0.022 |
| ece + latents, bond_var 0.05 | 200 | 0.36 | 0.92 | 0.025 | 0.12 | 0.006 | 1.14 | 1.59 | 0.026 |
| ece only | 200 | 0.28 | 1.04 | 0.025 | 0.15 | 0.007 | 1.34 | 1.82 | 0.030 |

(train = last epoch; val and stretch = best-validation checkpoint)
