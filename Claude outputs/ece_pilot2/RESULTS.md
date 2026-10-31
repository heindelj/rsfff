# pyscf-all (b3lyp/def2-svpd, 18 molecules): energies, Hessians, multipoles and response

Train: 612 non-stretch vacuum frames (10% val) + 719 probe-charge frames as a stream (10% val).
Held out: all 349 `scan_stretch` frames. Loss: E + F + Hessian-vector products (1 probe/step) +
dipole + traceless quadrupole + polarizability (all as energy derivatives, `train_bonding`
defaults for the scales), probe frames with the same labels. `field_features`, `couplings`,
`induction` on; 80 epochs, one seed, CPU float64. Configs: `make_cfg.py`; evaluation: `eval_all.sh`
(`scripts/eval_bonding.py`). Atom references: `atoms_b3lyp_def2svpd.json` (UKS atoms; F and S added).

Units: E kcal/mol (absolute, atom-referenced), F kcal/mol/A, H rel = relative Frobenius error,
mu = |error| in Debye, Theta MAE e*bohr^2, alpha MAE bohr^3 (mean alpha_iso ~27 bohr^3),
a_iso % = relative error of the isotropic polarizability, dE_int = error of the probe interaction
energy E(probes) - E(vacuum) (mean |dE_int| in the probe val set: 18 kcal/mol; 29 at 3 A, 10 at
5 A), dF_int its force change.

| split | model | E | F | H rel | mu | Theta | alpha | a_iso % | dE_int | dF_int |
|---|---|---|---|---|---|---|---|---|---|---|
| vacuum val (61) | latent | 0.49 | 0.71 | 0.020 | 0.066 | 0.043 | 0.15 | 0.95 | | |
| vacuum val (61) | ece + latents | 0.22 | 0.38 | 0.016 | 0.061 | 0.045 | 0.15 | 0.82 | | |
| probe val (72) | latent | 0.58 | 0.99 | | 0.126 | 0.123 | 0.71 | 2.50 | 0.43 | 0.61 |
| probe val (72) | ece + latents | 0.48 | 0.71 | | 0.123 | 0.125 | 0.70 | 2.47 | 0.44 | 0.57 |
| stretch (349, held out) | latent | 1.22 | 2.05 | 0.053 | 0.199 | 0.110 | 0.55 | 2.39 | | |
| stretch (349, held out) | ece + latents | 1.08 | 1.56 | 0.033 | 0.224 | 0.111 | 0.51 | 2.22 | | |

Worst cases (ece): probe-frame alpha for water / acetaldehyde / acetate (5-7%), stretch dipoles
of the carbonyl / carboxylate molecules (acetate 0.6 D, formamide 0.4, acetaldehyde 0.33,
formic acid 0.31), stretch alpha_iso for ammonium / hydroxide / HF (4-5%).
