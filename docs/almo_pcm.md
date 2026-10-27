# ALMO-EDA in a polarizable continuum, and the same continuum in the force field

Branch `main`. Code: `src/qcgen/almo.py` (pyscf ALMO-EDA: fragments → frozen → SCF-MI → full, with a uniform field
and C-PCM), `src/ff/cpcm.py` (the force-field C-PCM, point for point pyscf's), `src/ff/complete/solvent.py` (the
continuum inside the complete model's induction solve), `scripts/almo_pcm_scan.py` (the NaCl scan),
`scripts/pcm_consistency.py` (QM/FF consistency on the frozen densities). Tests: `tests/test_qcgen_almo.py`,
`tests/test_ff_cpcm.py`, `tests/test_complete_solvent.py`.

## Why

Training labels in a continuum sample a wider range of field environments than gas-phase clusters without going to
the condensed phase, and the reaction field keeps ion pairs ionic where the vacuum ground state is a pair of neutral
radicals (NaCl beyond ~3.6 Å, where the vacuum ALMO-EDA labels the wrong state and those frames are masked today).
The force field is trained *with* the continuum on those frames and used without it, so the continuum must be the
same object on both sides — same cavity, same discretization, same definition of every EDA term in its presence.
The model's environment description (fields and overlaps, no dielectric conditioning) is what makes PCM-trained
parameters transfer out of the continuum.

## The decomposition in the continuum

Every state is a constrained minimum of one functional, `E[D] = E_KS[D] + E_field[D] + G_PCM[D]`, with the
surface charges equilibrated to the state's own density (fully variational, so the force field reproduces each
term by minimizing `E_FF + G_PCM` jointly under the matching constraint):

| state | constraint | continuum |
|---|---|---|
| fragments | each fragment alone | its *own* cavity (a solvated monomer) |
| frozen | antisymmetrized product of the fragment orbitals | complex cavity, charges re-equilibrated |
| SCF-MI | occupied orbitals of each fragment in its own AOs (ALMO) | in the Fock matrix |
| full | none | in the Fock matrix (pyscf's PCM SCF) |

`E_int = E_frz + E_pol + E_ct`, `E_frz = E_cav + E_cls_elec + E_disp + E_mod_pauli`:

* `cav = Σ_A (G_cplx[ρ_A] − G_own[ρ_A])`: each solvated fragment density moved into the complex cavity — the
  desolvation by the cavity change (0 in vacuum).
* `cls_elec`: Coulomb interaction of the frozen fragment densities and nuclei (vacuum, as in the quasiclassical
  ALMO-EDA) **plus** the screened cross term `G_cplx[Σρ_A] − Σ_A G_cplx[ρ_A]` (= `q_A·V_B` for C-PCM). Both pieces are
  returned (`cls_elec_vac`, `cls_elec_scr`). A force field with fixed moments and surface charges reproduces this
  exactly up to its representation of the fragment densities on the surface.
* `disp`: Q-Chem's orthogonal frozen decomposition (EDA2 option sets 1–5). The frozen density is split into
  mutually orthogonal fragment densities `P̃_A` that minimize the fragments' distortion `Σ_A (E_A[P̃_A] − E_A[P_A])`
  (`orthogonal_decomposition`: rotations between occupied orbitals of different fragments, exact gradient through
  the Fréchet derivative of `expm`, L-BFGS), and
  `E_disp = [E_xc(D_frz) − Σ_A E_xc(P̃_A)] − [E_x^HF(D_frz) − Σ_A E_x^HF(P̃_A)]` with Hartree-Fock exchange as the
  dispersion-free functional (DISP_FREE_X = HF, DISP_FREE_C = NONE, Q-Chem's defaults). `frozen_decomp="classical"`
  (EDA_CLS_DISP, undistorted fragment densities) and `disp_free="nlc"` (VV10 removed) are the alternatives; the
  classical variant is ~1.5 kJ/mol short per hydrogen bond and VV10-only misses most of it.
* `mod_pauli`: the rest of the frozen interaction.

The field enters as `h1 = +F·r` (an electron at potential φ has energy −φ; `φ = −F·r`) plus the nuclear term, the
same convention as `qcgen.pol_ref.Environment`, which can be passed as `env=` instead of `field=`.

### Validation against Q-Chem (vacuum)

| system | term | pyscf (this) | Q-Chem |
|---|---|---|---|
| NaCl 2.6 Å, ωB97M-V/def2-TZVPD | pol | −46.766 | −46.760 |
| NaCl 2.8 / 3.2 Å | pol | −41.52 / −31.27 | −41.53 / −31.26 |
| water dimer, ωB97X-V/def2-QZVPPD | frz / cls_elec / int | 0.731 / −32.926 / −13.405 | 0.730 / −32.926 / −13.405 |
| water dimer (def2-TZVPD) | disp | −8.64 | −8.67 (QZVPPD) |
| H₃O⁺(H₂O)₂, ωB97M-V/def2-TZVPD | frz / cls_elec / int | −78.727 / −238.340 / −227.208 | −78.733 / −238.343 / −227.213 |
| H₃O⁺(H₂O)₂ | pol / ct | −76.39 / −72.09 | −72.76 / −75.72 |

Fragment energies agree to 1e-7 Hartree. The pol/ct split differs for molecules because Q-Chem's EDA2 option set
1 uses **FERF (nMDQ uFERF) polarization** — each fragment's relaxation restricted to its field-response functions
— whereas this code does the full SCF-MI in the fragment AO space; the remainder (3.6 kJ/mol here, 0.6 for the
water dimer, 0.006 for NaCl) moves from ct to pol. For atoms the two are the same to 0.01 kJ/mol; for molecular
labels a FERF option would be needed. The terms close (`frz + pol + ct = E_full − ΣE_frag`) to 1e-9 in vacuum, with
a field, and in C-PCM; the EDA terms are translation invariant in a field; Gauss's law holds on the surface charges.

## The continuum (both sides)

pyscf's `solvent.pcm` defaults = Q-Chem's: C-PCM with `f = (ε−1)/ε`, SWIG switching, 302 Lebedev points per sphere,
Bondi radii × 1.2 with H = 1.1 Å. `rsfff.ff.cpcm` builds the same surface (Lebedev tables generated from pyscf,
`src/ff/_lebedev.py`), the same `S` and the same solve, and evaluates the solute potential the way pyscf does: smeared
by each point's Gaussian (`q erf(ζr)/r` for a point charge; a closed form for a Slater 1s shell,
`slater_shell_surface_potential`, which is what the complete model's frozen densities are). Oracle tests: surface,
ζ, F, S and q to 1e-12 against pyscf; the smeared Slater potential against quadrature to 1e-11; Born and Gauss limits;
autograd forces against finite differences.

The equilibrated charges minimize `q·V + qᵀSq/(2f)`, so adding them to an induction solve is one more block of an SPD
system, `H = [[A, Jᵀ], [J, S/f]]`, `g = [L, V₀]`, energy `½ gᵀy`: no nested solve, forces by autograd.

## The force field side (complete model, ions)

`CompleteModel.forward(bt, ..., solvent=solvent_terms(bt, tab, CPCMSpec(eps=...), eps=per_frame))`. Eliminating
the surface charges turns the joint system into the usual induction solve with a screened operator,
`A → A − f JᵀS⁻¹J`, `L → L − f JᵀS⁻¹V₀`, so `solve_induction_blocks` is reused unchanged; `V₀` is the smeared
surface potential of the nuclei + Slater shells (exact), `J` the shaped parents' potentials. Reported terms mirror
the QM definitions: `e_elst` gains the screened cross term `e_scr`, `e_cav` is the desolvation (each ion alone in
its own sphere as the reference), `e_pol = ½ L'·x` is the joint minimum relative to the frozen state in the same
cavity, and the reaction potential of the frozen + induced state at the atoms enters the CT gap like an external
potential (an electron moved from the solvated anion to the solvated cation loses the reaction field — what
suppresses CT in the continuum). ε = 1 (or `solvent=None`) is exactly the gas-phase model; per-frame ε lets solvated
and vacuum frames share a batch. The CT populations do not re-polarize the continuum (second order).

### QM/FF consistency on the frozen densities (`scripts/pcm_consistency.py`, NaCl 2.6 Å, ε = 78.36, 302 points)

| | FF (pf2 shells) | QM (solvated fragments) |
|---|---|---|
| G_cplx[Na⁺] / G_own[Na⁺] (kJ/mol) | −246.72 / −251.77 | −246.72 / −251.77 |
| G_cplx[Cl⁻] / G_own[Cl⁻] | −275.30 / −290.11 | −282.99 / −299.49 |
| e_cav | 19.86 | 21.56 |
| e_scr (cls_elec's screened cross term) | 400.86 | 403.12 |
| Σq (frozen state) | −0.030 | −0.022 |

Na⁺'s surface potential from the shells matches the QM density to 2e-6 au; Cl⁻'s to 3e-3 au rms (4 % of V): the
vacuum-fitted shells reproduce the vacuum anion density (0.330 vs 0.331 e beyond the 1.2 × Bondi sphere), but the
solvated anion is more compact (0.27 e beyond the sphere at ε = 78), which is a 3 % effect on its self-solvation and
1.7 / 2.3 kJ/mol on e_cav / e_scr at contact. The round trip (FF frozen charges as pyscf Gaussian MM charges,
radii = 1/ζ, around the QM frozen density) reproduces `q·V_QM` to 1e-10 — the export convention is right.

## The NaCl scan (`scripts/almo_pcm_scan.py`, ωB97M-V/def2-TZVPD, 1.9–8 Å, ε = 1, 2, 4, 78.36)

Results: `nacl_pcm_scan.json` / `.png` (Claude outputs). In vacuum the restricted full SCF stays on the ionic
solution along the whole scan (dipole ≈ eR to 8 Å) but CT has a hump at 3–4 Å (−7 → −9 → 0 kJ/mol) and even
turns +0.15 kJ/mol at 7 Å — the ionic/covalent near-degeneracy. In the continuum pol and CT decay monotonically
(ε = 78: CT −3.7 at 2.6 Å, −1.3 at 4 Å, −0.07 beyond 6 Å), the frozen term becomes the balance of screened
electrostatics against a 15–30 kJ/mol desolvation at contact, and the ion pair is cleanly ionic everywhere — the
behaviour the Stage D labels need.

## Open items

* FERF (nMDQ) polarization for molecular fragments, to match Q-Chem's pol/ct split (atoms are fine).
* Open-shell fragments (UKS SCF-MI).
* A solvated-fragment reference for the anion shells if the 2 kJ/mol frozen-level mismatch matters (or accept it:
  it is common-mode between e_cav and e_scr).
* The CT populations' own reaction field (self-consistent CT in the continuum).
* Cost: the per-frame cavity solve is dense (P ≈ 300 × atoms); 110 points per sphere (`CPCMSpec(ng=110)`, labels with
  `lebedev_order=17`) is the cheap option for training batches.
