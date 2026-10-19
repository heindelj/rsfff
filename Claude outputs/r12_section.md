
## R12: Pauli kernel on permanent + induced moments (2026-10-18)

**Goal (Joe).** Have exchange polarization come out of the Pauli kernel acting on the induced moments, rather than from separate terms. Drop the confinement and pseudo-potential networks. A full model (with dispersion) can then also use force data.

**Model** (`--pol-form pauli`, `kernels.pauli_kernel`). This is CMM's Slater Pauli kernel, applied to the Pauli multipoles p_i = [q_P,i, −κ_i x_i], where x is the induced dipole/quadrupole. The Pauli density moves with the electrons, against μ.

    E_P(x) = Σ_{i<j} p_jᵀ T_P(√(b_i b_j); r) p_i + ½ Σ_i x_iᵀ H_i x_i
    H_i    = Σ_j q_P,j f₁(b_ij r)/r [s0d I₃ + s2d (êêᵀ − I/3) ⊕ s0q I₅]

T_P is the damped tensor with the overlap factors f_n themselves, so it is exponentially short-ranged.

The expansion in x gives four pieces:
- **Frozen Pauli:** q_P q_P, fit to ALMO `mod_pauli` with `--w-pauli`.
- **Linear drive g:** exclusion / relief. This replaces the pseudo-potential.
- **Inter-site coupling** of the induced moments.
- **On-site stiffening H:** the second-order deformation density seen by the neighbours' Pauli monopole. This replaces the confinement c.

E_pol = min_x[½xᵀ(A + H_P)x + (L + g)·x]: the same solve as before, with α and C at their free-ion values. The constants are per species (q_P, b_P, κ_d, κ_q, s0d, s2d, s0q), with s2d bounded so H is positive definite. With `--pauli-nn` the network rescales each ion's q_P (exactly 1 when isolated).
- q_P, b_P are initialized from a pair fit to mod_pauli: Na 0.98, 1.36; Cl 6.58, 1.58 (rms 10 % in ln E).
- Tests (`tests/test_ionct_pauli.py`, 6):
  - CMM monopole limit;
  - exclusion sign;
  - padded assembly vs a pair-by-pair sum;
  - H positive definite, and the response reduced by H;
  - isolated ion;
  - gradients.
- 53 ionct tests pass. Trains in ~27 min on 1 core with direct CT.

**Results** (direct CT throughout; kJ/mol; Td/Oh = Cl-centred tet4/oct6 shells)

| run | pol network | test E_pol all / clusters | test E_CT all / clusters | mod_pauli test all / clusters | Td/Oh shells E_pol | α_pol err (bohr³/ion) |
|---|---|---|---|---|---|---|
| v13 | confinement + pseudo | 0.09 / 2.05 | 0.10 / 0.60 | — | 27.7 | 0.44 |
| v14 | none (constants) | 1.26 / **24.6** | 0.10 / 0.70 | 1.43 / 5.7 | 25.3 | 0.86 |
| v14n | q_P only | 0.45 / 2.03 | 0.16 / 1.05 | 0.76 / 10.7 | **3.1** | 0.53 |
| v14p0 | constants, no mod_pauli loss | 1.08 / 20.4 | 0.09 / 0.48 | (q_P runs away) | — | 0.68 |

- **Constants only (v14):** pairs fit, but E_pol in clusters is under-bound by about 3 kJ/mol per Na–Cl contact. That is +10 to +30 per cluster, growing with size.
  - Fitted constants: Cl κ_d 2.2, s0d 1.1, s2d 2.9. So the stiffening is almost all along the bond.
  - In a cluster, the stiffening from several neighbours adds up to an isotropic confinement, while the electrostatic drive partly cancels.
- **q_P network (v14n):** matches v13 on clusters, with one small head in place of two.
  - It fits the Td/Oh shells by a degeneracy. mod_pauli fixes only the product q_P,Cl·q_P,Na. In tet4 the network raises q_P(Cl) to 22.7 and lowers q_P(Na) to 0.41.
  - The relief then comes from Na⁺ polarized away from Cl⁻: −9 kJ/mol per Na. Cl⁻'s own moments are zero by symmetry.
  - That is the wrong ion, and mod_pauli in clusters degrades (10.7).
- **What the shells say about the missing physics.** Compare how fast each quantity falls with R (local exponent n of E ∝ R⁻ⁿ, 2.6 → 2.8 Å and 2.8 → 3.2 Å):

  | quantity | n (2.6–2.8) | n (2.8–3.2) |
  |---|---|---|
  | ALMO E_pol, tet4 | 5.2 | 5.7 |
  | ALMO E_pol, oct6 | 6.3 | 7.5 |
  | Na–Cl mod_pauli | 7.6 | 8.1 |

  - Any Pauli-driven mechanism decays at least as fast as the Pauli energy:
    - pairwise exchange polarization ∝ E_P: n ≈ 8;
    - breathing ∝ E_P²/H: n ≈ 8 if saturated, ≈ 16 if not (`kernels.breathing_kernel`, an FD-tested diagnostic).
  - The shells decay more slowly, and tet4 (the l = 3 field allowed by Td) more slowly than oct6 (l = 4 first). That is what **induced octupoles and hexadecapoles of Cl⁻** give (undamped R⁻⁸ and R⁻¹⁰, softened by penetration).
  - An order-of-magnitude estimate (α_l ~ α₁a^{2(l−1)}, a ≈ 3 bohr, undamped) gives ~10 kJ/mol per contact at 2.6 Å from l ≥ 3, the size of the residual. Not yet checked quantitatively.
  - So the l ≤ 2 truncation of the response, not exchange polarization, is the likely cause. In pairs the fit pushes this missing energy into the dipole; in clusters, where the dipoles cancel, it shows up as under-binding.

**Next.** Extend the induced response of Cl⁻ to l = 3 (and 4), with free-ion α₃, α₄ and the same Pauli kernel acting on them; or test the hypothesis first on the shells with damped point fields. Dispersion: rsfff's TT dispersion + `eda_disp`, for total energies and forces.

Runs: research/rsfff `train/nacl/ionct/{ionct_v14, ionct_v14n, ionct_v14p0}`. Code: `Claude outputs/rsfff_r12.tgz` (applied to ~/dev/rsfff).
