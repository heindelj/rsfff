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
| NaCl 2.6 / 2.8 Å | mod_pauli | 61.45 / 34.89 | 61.45 / 34.89 |
| water dimer, ωB97X-V/def2-QZVPPD | frz / cls_elec / int | 0.731 / −32.926 / −13.405 | 0.730 / −32.926 / −13.405 |
| water dimer (def2-TZVPD) | disp | −8.64 | −8.67 (QZVPPD) |
| H₃O⁺(H₂O)₂, ωB97M-V/def2-TZVPD | frz / cls_elec / int | −78.727 / −238.340 / −227.208 | −78.733 / −238.343 / −227.213 |
| H₃O⁺(H₂O)₂ | pol / ct | −76.39 / −72.09 | −72.76 / −75.72 |

Fragment energies agree to 1e-7 Hartree. The terms close (`frz + pol + ct = E_full − ΣE_frag`) to 1e-9 in vacuum,
with a field, and in C-PCM; the EDA terms are translation invariant in a field; Gauss's law holds on the surface
charges.

### Polarization: FERF vs ALMO (`pol_model`)

Q-Chem's EDA2 option set 1 (Joe's labels) restricts each fragment's relaxation to its **fragment electric-field
response functions** (Horn & Head-Gordon 2015; "nMDQ uFERF" in Q-Chem 6: uncoupled responses to the monopole =
scaled nuclear charges, the dipole and the five traceless quadrupole operators, one SVD per operator, the left
singular vectors as virtuals, non-orthogonal between fragments), which gives pol a basis-set limit; the rest is CT.
`pol_model="ferf"` (default; `ferf_orders`, `ferf_coupled`, `ferf_svd_tol`) implements that, `pol_model="almo"` the
AO-block ALMO (option set 2). For atoms the two coincide (NaCl: 0.01 kJ/mol); for molecules the FERF space is the
smaller one. H₃O⁺(H₂O)₂ ωB97M-V/def2-TZVPD (kJ/mol):

| model | virtuals per water | pol |
|---|---|---|
| almo (full AO space) | all | −76.39 |
| ferf MDQ uncoupled (null vectors dropped only) | 45 = 9 × 5 | −75.20 |
| ferf DQ uncoupled | 40 | −73.71 |
| ferf MDQ uncoupled, relative singular values < 1e-2 dropped (the O 1s responses) | 37 | −71.25 |
| ferf DQ uncoupled, same | 32 | −68.98 |
| ferf D uncoupled | 15 | −60.12 |
| **Q-Chem 6.4 nMDQ uFERF** | ? | **−72.76** |

So Q-Chem's response space is evidently truncated further than "excluding the null vectors" (its value sits between
the 37- and 45-vector spaces; the core-orbital responses have relative singular values of 1e-3); the criterion is
not in the manual or the abstract of the uFERF paper. The remaining 2.4 kJ/mol here (0.6 for the water dimer) is
the uncertainty in the pol/ct split of molecular labels generated here vs Q-Chem; the sum pol + ct and every
other term agree. A Q-Chem run with `EDA2 = 2` on one frame would pin `pol_model="almo"` exactly, and
`eda_uferf = false, child_mp_orders = 232` the coupled DQ variant.

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

## Exchange polarization: the Hartree-product split (`hartree=True`)

One more variational state on the same domain (each fragment's occupied orbitals within occ ⊕ FERF): the
mutually polarized **Hartree product**, `H(x) = Σ_A E_A[P_A] + Σ_{A<B} Coulomb(A,B)` — each fragment's own
functional (xc intrafragment only) plus the full Coulomb coupling of nuclei and densities, no interfragment
antisymmetry (`HartreeProduct`, freeze-thaw solver `hartree_pol`; `H(x₀) − E_ISO` is cls_elec exactly). With
`A(x)` the antisymmetrized functional the SCF-MI minimizes:

* `pol_elec` = I_H = H(x_H) − H(x₀) ≤ 0: pure electrical induction (mutual, penetration and nonlinearity included,
  no Pauli counter-force) — what a classical induction model with the same response shapes represents;
* `pol_exch` = X_POL = POL − I_H, any sign, = `dP_onH` (P(x_H) − P(x₀), P = A − H: antisymmetrizing the electrically
  polarized orbitals) + `R_AS` (A(x_P) − A(x_H) ≤ 0, the further relaxation under A);
* the endpoint convention `pol_elec_act` = H(x_P) − H(x₀), `pol_exch_act` = POL − pol_elec_act: the electrical and
  Pauli energy changes of the *actual* ALMO deformation (`H_cost` = H(x_P) − H(x_H) ≥ 0).

Checks (`tests/test_qcgen_almo.py`): closures to 1e-9, signs, x_H a minimum of H under random in-domain rotations,
X_POL → 0 and I_H → −½ Σ α E² at 8 Å. The FERF restriction is essential: in the full AO space the Hartree product
collapses toward the partner nuclei (the cage script's XP probe).

`scripts/almo_xpol_scan.py` (ωB97M-V/def2-TZVPD, `docs/almo_pcm/xpol_scan.{json,png}`), kJ/mol:

| pair, R (Å) | POL | I_H | X_POL | dP_onH | R_AS | I_act | X_act |
|---|---|---|---|---|---|---|---|
| Na–Cl 2.2 | −58.7 | −385.7 | +327.0 | +708 | −381 | −115.0 | +56.3 |
| Na–Cl 2.6 | −46.8 | −174.2 | +127.4 | +246 | −119 | −88.8 | +42.1 |
| Na–Cl 3.0 | −36.3 | −78.7 | +42.3 | +71 | −29 | −57.3 | +21.0 |
| Na–Cl 3.6 | −22.1 | −28.9 | +6.8 | +9.2 | −2.3 | −27.0 | +4.9 |
| Na–Cl 4.5 | −9.4 | −9.8 | +0.4 | | | −9.8 | +0.3 |
| Cl–Cl 2.8 | −42.0 | −155.0 | +113.0 | +623 | −510 | **+9.3** | **−51.3** |
| Cl–Cl 3.2 | −29.6 | −33.8 | +4.2 | +26 | −22 | −18.7 | −10.9 |
| Cl–Cl 4.0 | −17.0 | −18.8 | +1.8 | | | −18.4 | +1.3 |
| Na–Na 2.8 | −3.3 | −3.4 | +0.04 | | | | |
| Cl⁻ in Na⁺₂ a90 2.6 | −89.5 | −256.8 | +167.4 | +309 | −141 | −148.9 | +59.4 |
| Cl⁻ in Na⁺₂ a180 2.6 | −19.7 | −55.5 | +35.8 | +75 | −39 | −22.6 | +2.9 |
| Cl⁻ in Na⁺₄ sq4 2.6 | −38.9 | −46.3 | +7.4 | +44 | −36 | −12.9 | −26.0 |
| Cl⁻ in Na⁺₄ tet4 2.6 | −34.1 | −125.0 | +90.9 | +241 | −150 | −16.4 | −17.8 |
| Cl⁻ in Na⁺₆ oct6 2.6 | −63.4 | −24.8 | **−38.6** | −8.7 | −30.0 | **+3.8** | **−67.2** |
| Cl⁻ in Na⁺₆ oct6 3.2 | −14.6 | −3.6 | −10.9 | −3.9 | −7.0 | +2.6 | −17.1 |

What it says. For Na⁺–Cl⁻ the pure electrical induction at contact is three to four times the ALMO polarization
(−174 vs −47 at 2.6 Å): the anion's density is pulled into the cation's attraction and the Pauli wall gives most of
it back — exchange polarization is large and positive, the confinement/stiffening picture. For Cl⁻–Cl⁻ the two
conventions disagree in sign, which is the point: the counterfactual electrical minimum again pulls density toward
the partner (I_H −155 at 2.8 Å, X_POL +113), but the *actual* ALMO deformation raises the Hartree energy (I_act
+9.3) — the anions polarize away from each other to relieve Pauli repulsion, against the electrostatics; along the
actual path the Cl–Cl polarization at contact is entirely exchange-driven (X_act −51), the negative-sign case.
Beyond ~3.6 Å both ions show a small positive X_POL (1–7 kJ/mol) and the conventions coincide; Na⁺ has essentially
no exchange polarization. In the octahedral cage (the case the breathing term was built for) the sign flips in
*both* conventions: the electrical induction is small (the fields cancel at the centre, I_H −25) and the ALMO
polarization of −63 is exchange-driven (X_POL −39, X_act −67, I_act +4): the Pauli pressure of six neighbours
contracts the anion, an l = 0 response no field can drive (still −11 of −15 at 3.2 Å). The other cages interpolate:
with the fields partly cancelling (a180, sq4, tet4) the electrical part shrinks and the Pauli part grows, and in
the endpoint convention sq4 and tet4 are already exchange-dominated (X_act −26 / −18 vs I_act −13 / −16) while
the open a90 shell is field-driven like the pair. For the force field: I_H (or I_act, see below) is the label for the dipole/quadrupole/
breathing response, X for the exclusion drive + stiffening — but at contact the counterfactual split asks the
stiffening to cancel hundreds of kJ/mol, so the endpoint convention, whose two pieces are the energies of one
deformation, is the better-conditioned fitting target; the counterfactual one is the cleaner statement that
electrostatics alone would over-polarize.

## Open items

* Q-Chem's FERF truncation criterion (the last 2.4 kJ/mol of the H₃O⁺(H₂O)₂ pol/ct split); atoms are unaffected.
* Open-shell fragments (UKS SCF-MI).
* A solvated-fragment reference for the anion shells if the 2 kJ/mol frozen-level mismatch matters (or accept it:
  it is common-mode between e_cav and e_scr).
* The CT populations' own reaction field (self-consistent CT in the continuum).
* Cost: the per-frame cavity solve is dense (P ≈ 300 × atoms); 110 points per sphere (`CPCMSpec(ng=110)`, labels with
  `lebedev_order=17`) is the cheap option for training batches.
