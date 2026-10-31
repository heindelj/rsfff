# ECE electrical heads on pyscf-all (same split and training as RESULTS.md; 80 epochs, one seed)

Runs: `latent` (SOAP latents everywhere), `ece` (ECE bonded heads, latent electrical), `ece_elec`
(ECE bonded + ECE electrical heads with the dissociation envelopes, electrical heads also read the
permanent-family latents), `ece_elec_nolat` (electrical heads on the covalent graph only).
Units as in RESULTS.md (E kcal/mol, F kcal/mol/A, mu |error| D, Theta e*bohr^2, alpha bohr^3).

| split | run | E | F | H rel | mu | Theta | alpha | a_iso % | dE_int |
|---|---|---|---|---|---|---|---|---|---|
| vacuum val | latent | 0.49 | 0.71 | 0.020 | 0.066 | 0.043 | 0.151 | 0.95 | |
| vacuum val | ece | 0.22 | 0.38 | 0.016 | 0.061 | 0.045 | 0.148 | 0.82 | |
| vacuum val | ece_elec | 0.23 | 0.44 | 0.016 | 0.031 | 0.026 | 0.120 | 0.83 | |
| vacuum val | ece_elec_nolat | 1.47 | 0.67 | 0.019 | 0.084 | 0.044 | 0.143 | 1.11 | |
| probe val | latent | 0.58 | 0.99 | | 0.126 | 0.123 | 0.710 | 2.50 | 0.43 |
| probe val | ece | 0.48 | 0.71 | | 0.123 | 0.125 | 0.697 | 2.47 | 0.44 |
| probe val | ece_elec | 0.43 | 0.80 | | 0.098 | 0.120 | 0.687 | 2.38 | 0.35 |
| probe val | ece_elec_nolat | 1.45 | 1.21 | | 0.179 | 0.136 | 0.737 | 3.34 | 0.53 |
| stretch | latent | 1.22 | 2.05 | 0.053 | 0.199 | 0.110 | 0.552 | 2.39 | |
| stretch | ece | 1.08 | 1.56 | 0.033 | 0.224 | 0.111 | 0.511 | 2.22 | |
| stretch | ece_elec | 1.28 | 1.71 | 0.037 | 0.260 | 0.091 | 0.600 | 2.72 | |
| stretch | ece_elec_nolat | 2.32 | 1.86 | 0.039 | 0.314 | 0.117 | 0.624 | 2.73 | |

Stretch frames by bond and r / r_e (r_e = the scan's lowest-energy point): dipole |error| (D) and
alpha_iso error, `ece` vs `ece_elec`:

| bonds | r / r_e | n | mu ece | mu ece_elec | a_iso ece | a_iso ece_elec |
|---|---|---|---|---|---|---|
| X-H | < 1.15 | 148 | 0.060 | 0.035 | 0.93 % | 0.91 % |
| X-H | 1.15-1.3 | 25 | 0.122 | 0.129 | 2.62 % | 1.93 % |
| X-H | 1.3-1.5 | 50 | 0.318 | 0.309 | 5.06 % | 5.54 % |
| heavy-heavy | < 1.15 | 84 | 0.137 | 0.141 | 1.27 % | 1.25 % |
| heavy-heavy | 1.15-1.3 | 14 | 0.536 | 0.853 | 3.28 % | 5.59 % |
| heavy-heavy | 1.3-1.5 | 28 | 1.112 | 1.544 | 5.81 % | 10.90 % |

The heavy-heavy losses are where the labels leave the covalent diabat: along the RKS scans the
formic acid C-O(H) dipole grows 0.5 -> 4.1 D (charge separation, HCO+ ... OH-), and methylammonium's
C-N stretch heads for CH3+ + NH3 -- heterolytic channels, while the envelopes force every bond to
its homolytic limit (the per-atom diabatic charges put methylammonium's +1 on N).
