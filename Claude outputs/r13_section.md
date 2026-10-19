
## R13: dispersion and the complete potential (2026-10-18)

**Decision (Joe).** Leave out explicit induced octupoles and hexadecapoles; the missing polarization may later go into a generic delta-learned correction. First complete the potential, then revisit the polarization and CT forms.

**Dispersion** (`src/ff/ionct/dispersion.py`, `scripts/ion_frozen_fit.py --term disp`)
- Form: Tang-Toennies C6 + C8 per species, geometric combination rules, optional environment scale s_i on each ion's coefficients (s = 1 when isolated; network on the SOAP invariants). Fit to ALMO `disp`, same split as the CT model.
- **ALMO disp is not pairwise additive.** Whole clusters are 7–15 % below the sum of their pair values (walk Na₅Cl₅ about −13 %), so a pairwise form fits pairs and then misses clusters by ~10 %. An Axilrod–Teller term per species recovers most of it; an environment-scaled C6 gets all of it.

| form | test pairs MAE | test trimers | test whole clusters (bias) |
|---|---|---|---|
| pairwise C6 + C8 | 0.23 | 0.34 | 10.0 (−10.0) |
| + Axilrod–Teller, damped | 0.14 | 0.27 | 2.1 (−2.1) |
| **environment-scaled C6 (disp_nn)** | **0.012** | **0.021** | **0.71** (−0.47) |

  - Fitted constants (disp_nn): C6 Na 5.1, Cl 514; C8 Na 29, Cl 10 011; b Na 1.94, Cl 0.99.
  - Smooth on fine pair scans (Na–Cl 2.3–7 Å, Cl–Cl 2.9–7 Å). Beyond ~5 Å it is slightly over-attractive, because ALMO decays faster than C6/r⁶ there (≤ 0.02 kJ/mol).

**Frozen Pauli on its own** (`src/ff/ionct/frozen_pauli.py`, `--term pauli`)
- CMM monopole kernel, with or without an environment scale of q_P, fit to `mod_pauli`.
- mod_pauli is about 2 % non-additive: three-body −2 to −18 kJ/mol per cluster, higher orders ≈ −2.

| form | test all | test whole clusters |
|---|---|---|
| constants (pauli_pair) | 1.28 | 5.27 |
| environment-scaled (pauli_nn) | 0.31 | 7.2 |
| v14n Pauli kernel (shared with pol) | 0.76 | 10.7 |

**Complete potential** (`src/ff/ionct/potential.IonPotential`, `scripts/ionct_total.py`)
- E_int = elec[perm, fixed] + Pauli + disp + pol + CT. ALMO int equals cls_elec + mod_pauli + disp + pol + ct exactly.
- Forces come from autograd through everything. `precompute(..., differentiable=True)` keeps the graph through the geometry tensors and the SOAP features. Forces match finite differences (`tests/test_ionct_potential.py`; 51 ionct tests pass).
- The force set (`train/nacl/data/force`, nacl2–nacl20, ωB97M-V/def2-TZVPD): E_int = E − Σ free-ion energies agrees with ALMO int to 0.02 kJ/mol where the geometries coincide. 63 cluster frames are also EDA frames, so their energies were trained, but no forces were.

| combination | EDA test E_int all / whole clusters | force set E_int /ion (n = 4–20) | force error rms, n ≥ 4 (ref rms 25–40) |
|---|---|---|---|
| v14n (Pauli kernel + its own frozen Pauli) + disp_nn | 1.3 / 12.7 | 0.9–4.4 | 17–76 |
| v13 + pauli_pair + disp_nn | 1.6 / 6.7 | 0.6–2.4 | 14–21 |
| v13 + pauli_nn + disp_nn | — / 9.5 | 0.6–2.1 | 14–31 |
| v14 (Pauli kernel, constants) + disp_nn | 2.7 / 28 | 1.7–5.0 | 14–27 |

- **v14n's q_P network extrapolates badly in the big clusters.** In nacl14, its Pauli forces alone are 215 kJ/mol/Å rms. The shell degeneracy (q_P(Cl) high, q_P(Na) low) is not harmless.
- **The forces are poor in every combination.** R² is about 0.4 for v13 + pauli_nn + disp_nn on cluster frames (rms error 22 vs reference 29 kJ/mol/Å).
  - The electrostatic and Pauli forces are each ~120–145 kJ/mol/Å rms and cancel to a total of ~30, so a few % error in either short-range term is already 5–10 kJ/mol/Å.
  - A joint rescaling of the per-term forces (elec 0.70, Pauli 0.63, pol 0.48) cuts the error to 14. Rescaling any one term alone barely helps (≤ 1.6).
  - So the repulsive wall and the penetration electrostatics are both too steep together. Energies on near-equilibrium EDA geometries don't constrain that balance; forces do.
- **Na–Cl dimers beyond 3.4 Å are off by +4 to +7 kJ/mol.** This is ALMO's CT there (ionic–covalent mixing, −8 to −9 kJ/mol out to 4 Å), which the CT label mask excludes and the model doesn't reproduce: the known dimer limitation. Two-thirds of the force-set dimers are in that range.

**Next.**
1. Force training. It needs a lean differentiable precompute: the direct CT and Pauli forms only use L, A_off, the SOAP features and the pairs; the full precompute costs ~1–3 s per frame.
2. Revisit the polarization/CT forms with the complete potential and force data, with a generic delta correction for the rest of E_pol.

Runs: research/rsfff `train/nacl/ionct/r13/{disp_nn, disp_pair, pauli_nn, pauli_pair, total_*}` (total_* have total_eda.csv, total_force.csv, force_atoms.csv, ionct_total.png). Code: `Claude outputs/rsfff_r13.tgz`.
