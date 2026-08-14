# Range-Separated Polarizable MLIP with EDA-Guided Fragmentation Mixing

## 1. Purpose and design philosophy

The goal is a reactive machine-learned interatomic potential (MLIP) that reproduces the Born–Oppenheimer ground-state surface `E(R; Q, S)` while remaining physically well-behaved across regions where the local charge/spin character changes (e.g. homolytic vs. heterolytic bond cleavage, ion-pair vs. radical-pair channels).

The model does **not** attempt to represent true diabatic states. Instead, it uses energy-decomposition-analysis (EDA) data as a training strategy to ensure the physical components of the force field remain well-defined. The total energy is partitioned into five conceptual components: **Electrostatics, Polarization, Pauli repulsion, Dispersion, and Bonding (deformation)**.

Reactivity is recovered through a smooth, learned mixing over candidate fragmentations combined with a self-consistent polarizable electrostatic solve. Charge Transfer (CT) is not treated as an explicit pairwise interaction; it emerges as the learned correction to the bond energy and the other interactions when the fragmentation constraints are lifted.

### Two mechanisms that must not be conflated

**Constraint lifting gives CT. Fragmentation mixing gives reactivity.** These are different and an earlier version of this document collapsed them.

Water has exactly one sensible fragmentation and a nonzero CT energy, which is enough to show that mixing cannot be what produces CT. ALMO-EDA defines CT as the relaxation obtained by lifting the constraint that MO coefficients stay within fragment blocks, at a *fixed* fragmentation. The model's analogue is the staged fit of §5.1: fit the polarization surface with inter-fragment charge transfer disallowed and features fragment-confined, then lift both. What CT *is*, in this decomposition, is the energy change that appears when a pair's bonding channel and classical channels are re-evaluated without those constraints.

Fragmentation mixing answers a different question — which molecule an atom belongs to — and is what carries the model through a bond breaking or forming.

Two physically distinct pieces of information are carried for every atom:

1. **Atomic environment** — the local geometric/chemical descriptor (invariant + equivariant channels).
2. **Electrostatic environment** — the range-separated electrostatic potential, field, and field gradient produced by the rest of the system, conditioned on net charge and spin.

Conservation of total charge is guaranteed *by construction* through a split-charge equilibration (SCEQ) polarization model, not through the learned features.

## 2. Physical variables and notation

| Symbol | Meaning |
| --- | --- |
| `R` | Nuclear coordinates |
| `Q, S` | Total (conserved) charge and spin of the system |
| `a_i(R)` | Atomic-environment features on atom `i` (irrep-resolved) |
| `φ, E, ∇E` | External electrostatic potential (l=0), field (l=1), field gradient (l=2) at each atom |
| `p, β` | Effective polarization / FF parameters and **pair-specific range-separation parameters** |
| `M*` | Converged atomic multipoles (monopole + higher moments) from the SCEQ solve |
| `θ` | All learnable network parameters |

The multipoles are carried in their natural irreps: **monopole = l=0, dipole = l=1, quadrupole = l=2 (symmetric traceless)**. All mixing across irrep orders must go through Clebsch–Gordan tensor products or gated (norm-based) nonlinearities — never a scalar MLP on concatenated components, which would break rotational equivariance.

## 3. Pipeline overview

```text
candidate fragmentations
        │  (approximate initial multipoles per fragmentation)
        ▼
approximate electrostatic environment  ──►  φ, E, ∇E
        │
        ├── atomic-environment features  a_i
        │
        ▼
   feature mixing  (permutation-invariant over fragmentations)
        │            → initial per-atom features  f_i^(0)
        ▼
   parameter network  →  effective FF parameters p & pair-specific damping β_ij
        │
        ▼
   SCEQ polarizable solve  (range-separated Coulomb)
        │            → converged multipoles  M*
        ▼
   correction features  =  { a_i , M* }
        │
        ▼
   pairwise correction MLIPs  →  ΔE_elst, ΔE_pol, ΔE_pauli, ΔE_disp, E_bonding
        │
        ▼
   E_total = E_FF(β_ij) + Σ ΔE_EDA + E_bonding

```

## 4. Stage-by-stage description

### 4.1 Candidate fragmentations → approximate electrostatic environment

For a given geometry, enumerate a small set of candidate fragmentations. Each fragmentation carries a crude, pre-allocated charge distribution, which is sufficient to seed the electrostatic environment.

### 4.2 Electrostatic features

From the approximate charge distribution, compute the electrostatic potential `φ`, field `E`, and field gradient `∇E` at each atom. These are the environment's action on the atom, expressed as a multipole expansion truncated at quadrupole-order response.

### 4.3 Feature mixing across fragmentations

Each atom participates in a variable number of candidate fragmentations. The mixing must be **permutation-invariant** over the fragmentation slot and handle a padded/variable count via masking.

> **Feature invariant.** *Atomic* features may and should depend on the fragmentation — that is how the model tells an H₃O⁺ from an H₂O, and how EDA data is usable at all. *Pair* features must be functions of the **mixture over** fragmentations, never of one selected fragmentation. Under a single fragmentation the mixture is that fragmentation and the two coincide, so water is the degenerate limit of the rule rather than an exception to it. Because mixing happens at the atomic level and pair features are built downstream of it, `β_ij` and `ΔE_ij` become functions of the mixture for free.
>
> One gap this exposes in the current code: `FlatLambdaSOAPFeaturizer`, which every force-field term uses, carries species and geometry only — no fragment charge, no multiplicity. Invisible on water; fatal for H₅O₂⁺, where the two fragmentations differ precisely in which fragment carries the charge. `rsfff.ff.unified.FragmentStateEmbedding` is the slot for it, identically zero at the neutral singlet so it stays inert until charged-fragment data arrives.

Use attention pooling where:

* **weights are invariant scalars** (built only from l=0 channels or invariant contractions of equivariant ones),
* **values are the (possibly equivariant) features**,
* the same scalar weight is applied across all irrep channels of a fragmentation and summed.

`w_a = softmax_a( g(invariants of f^(a)) / τ )`, then `f_i = Σ_a w_a f_i^(a)`.

The softmax partition of unity provides the smooth transition between regions where different fragmentations dominate. The learned combination of these features is what ultimately dictates the shift in energy (e.g., Charge Transfer).

### 4.4 Parameter network → effective FF & Range-Separation parameters

The mixed features `f_i` are mapped to the **effective polarization parameters** `p` (multipolar electronegativities, hardnesses, split-charge stiffnesses) and **pair-specific range-separation parameters** `β_ij`.

Allowing the range-separation to be pair-specific ensures that as the features change under feature-mixing, the classical FF interactions can be rigorously switched off or modulated at short range depending on whether the pair is acting strictly intermolecularly or transitioning into an intramolecular bond.

### 4.4.1 Range separation vs. routing

Two jobs rode on the old hard inter-fragment mask, and separating them is what makes the range separation learnable at all (implemented in `rsfff.ff.unified`):

* **Functional form** — is the classical multipole / Slater / Tang–Toennies form switched on for this pair? Learned per pair and per channel, with **no explicit same-fragment indicator**; it depends on the fragmentation only through the atomic features, per the invariant in §4.3.
* **Routing** — which training label does a pair's energy answer to? `eda_cls_elec` *is defined* over inter-fragment pairs and `fragment_energy` *is defined* over one fragment's atoms. This follows from what the labels mean and is not learned. A free network deciding it could move energy between the buckets at zero loss, and all four targets would stop being well-posed.

`r0` is per **atom** and per channel, combined across a pair as a geometric mean. A per-atom `r0` cannot distinguish an intramolecular O–H from an intermolecular one — it is the same hydrogen in both — and does not need to: the discrimination comes from `r`, which is what a range separation is for. One `r0` per channel because the channels are not descriptions of equal fidelity.

**Consequences of dropping the mask**, both real:

1. Every pair now carries the classical backbone, including covalent ones, so a range separation is *required* rather than optional — including for Pauli, which previously had none because bonded pairs were masked out of its pair list. At 0.96 Å the Slater Pauli form between O and H is ~1000 kJ/mol. See `rsfff.ff.range_priors` for the measured intra/inter distance gaps the priors sit in.
2. Intra-fragment pairs get real electrostatics routed into `fragment_energy`. This is a capability the per-term stack lacks entirely — its bond head dies at 4 Å, so two atoms of one fragment 8 Å apart interacted not at all.

**Routing is not a no-op on the total.** Relabelling a dimer as a single fragment changes the predicted energy, for two measurable reasons: the bond channel has no inter-fragment counterpart, so pairs that switch buckets gain a channel; and fragment-confined descriptors are partition-dependent themselves. Both are correct — relabelling asserts that two molecules are one. The first is where a CT energy should surface. What *is* exact is the accounting: every pair appears once, in one bucket, with no double counting and no gap.

### 4.5 SCEQ polarizable solve → converged multipoles

**Split-charge equilibration** provides:

* A **multipolar electronegativity** producing the *permanent* multipoles.
* **Charge-transfer variables** `q_AB` moving charge between pairs, conserving charge per channel by construction.

The stationarity condition `∂E_pol/∂M = 0` is solved with the explicit range-separated Coulomb interactions, utilizing the pair-specific damping parameters predicted by the network. The output is the converged multipole set `M*`. Implemented in `rsfff.ff.coupled_solve` and `rsfff.ff.polarization`; three things about it are load-bearing.

**The coupling operator is the frozen electrostatics channel's own.** `slater_elec_tensors` — point multipoles plus Slater penetration, under the same learned range separation `gate["elst"]` — not a separate polarization damper. Two consequences: `E_pol` is *exactly* zero when the multipoles do not move, so the label is a pure relaxation rather than the residue of two nearly-agreeing operators; and the polarization catastrophe is already handled, because the range separation switches off the short-range coupling that causes it. Measured on a water-like cluster, the smallest eigenvalue of the functional is `+0.13` gated and `−0.80` with the coupling ungated — in the second case the "solution" is a saddle and the energy is unbounded below.

**The change of variables is required, not stylistic.** The coupled form contains `½ μᵀ α⁻¹ μ`, and this codebase deliberately never inverts `α`. So rescale, exactly as SQE rescales `p = S v`:

```text
q = q0 + B S v      μ = α u      Θ = c w      x = (v, u, w)
```

Every block of the **matvec** is then polynomial in `(s, α, c)`, and the unpolarizable limits `α → 0`, `c → 0`, `s → 0` are well-conditioned rather than singular — the same argument `rsfff.mlip.sqe` records for choosing `S` over `S^½`, and load-bearing for force training at a closed channel. In these variables the charge block is `S L S + S`, which is symmetric; the asymmetric `L S + I` that `sqe_solve` solves is that system left-divided by `S`, and CG needs the symmetric one. The **preconditioner** may invert freely, including `α⁻¹`, because it runs entirely under `no_grad` and never enters a gradient. The rule: *inverse-free in the differentiated matvec, inverses allowed in the undifferentiated preconditioner.*

**The solver is preconditioned CG with the uncoupled response as the preconditioner**, adapted from `pyCMM/cmm/polarization.py`. That preconditioner captures all intra-fragment charge coupling exactly and leaves CG only the weak inter-fragment part, so a water cluster converges in 5–8 iterations. Two differences from pyCMM worth keeping: it enforces charge conservation with Lagrange multipliers, making its system an indefinite KKT saddle point that CG has no guarantee on, whereas SQE conserves charge by construction and is genuinely PSD; and every CG scalar here is **per frame**, so a frame's answer does not depend on which others shared its minibatch.

The matvec is *derived from* the energy — `A x = grad_E(x) − grad_E(0)`, `b = grad_E(0)` — so a sign error in the interaction tensor cannot make the solver and the reported energy disagree. That identity is pinned against autograd in `tests/test_ff_coupled_solve.py`, and it is why no dense Hessian is assembled anywhere.

### 4.6 Correction network → short-range EDA & Bonding energies

The final features `{ a_i , M* }` feed into pair-specific MLIP heads that map directly to the five energy components: Electrostatics, Polarization, Pauli, Dispersion, and Bonding.

Every pair is allowed to interact under the correction heads to predict the short-range pieces the force field cannot inherently capture. Because the FF backbone is rigorously range-separated (using the learned `β_ij`), its short-range contributions are switched off, and the correction heads seamlessly absorb the remainder of the interaction. The "bonding" head primarily handles the intramolecular deformation energy, while the other heads correct the classical counterparts.

### 4.7 Spin

Condition globally on total multiplicity. For homolytic/heterolytic discrimination, condition additionally on **atomic spin populations / spin-density moments** from the solve.

## 5. Training with EDA data

The EDA data acts as a robust training strategy to explicitly well-define the distinct physical components of the force field.

* **Primary objective:** Total energy and forces. Ultimately, the model must collapse multiple partitionings to a single, consistent total energy.
* **Auxiliary objective:** Match EDA components for *fixed, given fragmentations*. The EDA components differ across partitionings (especially in bonding/deformation energies).

### 5.1 Emergent Charge Transfer

**CT is not modeled as an explicit interaction term.** It emerges as the correction to the bond energy, alongside the changes to the classical interactions, when the fragmentation constraints are lifted — *not* from fragmentation mixing (see §1). The route is a staged fit that mirrors what ALMO-EDA actually does, and each stage relaxes exactly one constraint:

| stage | response parameters | solve | label |
| --- | --- | --- | --- |
| **frozen** | fragment-confined features, no field | grouped by fragment | `eda_cls_elec`, `fragment_energy` |
| **polarized** | *may* use environment-aware features, with field | grouped by fragment (no inter-fragment charge flow) | `eda_pol` = E_pol − E_frozen |
| **CT** | environment-aware, features un-grouped | inter-fragment channels open | `eda_ct` = E_ct − E_pol |

Only the **frozen** level is pinned: its multipoles are what `eda_cls_elec` means (frozen isolated-monomer densities) and its `E_internal` is what `fragment_energy` means, so its parameters cannot be environment-aware without breaking the 1-body label. That ceiling does **not** apply to the polarized level — environment-dependent response parameters are exactly what polarization *is*, and are how effective electrostatic interactions get absorbed into the response.

#### Every new term is a difference of one function

Implemented in `rsfff.ff.unified`. Each higher level is the *same* readout evaluated with one more constraint lifted, so it vanishes identically at the level below and adds no free parameters:

```text
ΔE_elst = W_elst(u(h_frag))                          → cls_elec
ΔE_pol  = W_elst(u(h_env)) − W_elst(u(h_frag))       → pol
             sum = W_elst(u(h_env))   ← one evaluation at inference

E_bond⁰   = W_bond(u(h_frag), φ=0)                       → fragment_energy
E_bond^pol= W_bond(u(h_frag), φ¹) − W_bond(u(h_frag),0)  → pol
E_bond^ct = W_bond(u(h_env),  φ²) − W_bond(u(h_frag),φ¹) → ct
             sum = W_bond(u(h_env), φ²)
```

The split is training-time bookkeeping and telescopes away at deployment. Three notes:

* **`cls_elec` is rigorously two-body** — the classical Coulomb interaction between frozen monomer densities — so its correction must read the fragment-confined stream. The environment-dependent part is not discarded; it becomes the polarization correction.
* **The frozen bond term takes `φ = 0`, not the frozen field.** `fragment_energy` is the *isolated* fragment, which feels nothing. So the whole field-dependent bond energy is polarization, including the part driven by the permanent field.
* **`EnvironmentResidual` is anchored**, `h_env = h_frag + g(h_full) − g(h_frag)`. For an isolated fragment `h_full == h_frag`, so every difference above is zero on a monomer — which is what these labels require. Without the subtraction a lone water has a spurious CT energy. This changes what `h_env` means, so checkpoints trained before it do not transfer their environment-dependent half.

#### Relaying the environment into the bonded region

The bond energy takes the external potential, field and field gradient as *features* (`rsfff.ff.environment`), built from the **converged** multipoles at that level and weighted by the electrostatic range separation. Deliberately **not** AMOEBA's route of switching on intramolecular electrostatics for the induced moments: intramolecular point multipoles are a poor model of bond response, it would need separate range separations for the permanent and induced parts, and the final model should make no distinction between them.

Because those features feed a head that is *not* variational in `M*`, the adjoint of §6.2 is mandatory here — see below.

#### Charge transfer needs a compliance scale

`ct` opens inter-fragment channels (`rsfff.ff.pairs.union_channels`), with the radius-derived ones enveloped to zero at the cutoff — without that a channel appears the instant two molecules come within range and the forces are discontinuous.

SQE has no structural reason to prefer an intramolecular channel over an intermolecular one: the electronegativity difference driving an O–H transfer is the same whether the two atoms share a molecule. At a shared `s_init` the model therefore opens covalent-strength channels across every hydrogen bond — measured on a freshly initialized w2, `ct` starts at **−52 kJ/mol against a true −8.2**, with only 0.0005 e of net charge actually crossing. `ct_compliance_scale` is the fix, and it is the charge-transfer analogue of a correction channel's `energy_scale`: at 0.1 an untrained w2 starts at −7.1 kJ/mol. Softplus is unbounded, so it sets where the head starts looking, not a ceiling.

### 5.2 Cross-fragmentation consistency (required)

* In a fraction of batches, present **≥2 fragmentations of the same geometry**.
* Add a **consistency loss** penalizing disagreement in total `E` and forces `F` across those fragmentations.

Without this term, the mixing network is under-constrained in the reactive window where two fragmentations are comparably dominant.

### 5.3 Monomer pretraining

Pretrain the parameter function and short-range network on **monomers embedded in many point-charge environments** to isolate response physics and validate against finite-field DFT polarizabilities.

## 6. Differentiability and the self-consistent solve

### 6.1 The subtlety: forces through `M*`

The total energy depends on both the converged multipoles and the structural coordinates. While the polarization energy is variational with respect to `M*` (`∂E_pol/∂M = 0`), the correction network `ΔE` is *not* variational in `M*`.

**Consequence:** The converged multipoles must carry their Jacobian `dM*/dR` into the correction network. Detaching `M*` yields analytic forces that do not equal `−dE/dR`.

### 6.2 Implicit differentiation

Differentiate `M*` via the **implicit function theorem** applied to the stationarity condition:

```text
dM*/dR = − (H_MM)⁻¹ ( ∂g/∂R + (∂g/∂p)(dp/dR) )

```

where `H_MM = ∂²E_pol/∂M²` is the polarization Hessian. This leaves the solver iterations off the computational graph while injecting the correct gradient.

**Implemented** in `rsfff.ff.coupled_solve._CoupledSolve`. Forward runs CG under `no_grad` and returns the state; backward receives `∂L/∂x`, solves the adjoint `A λ = ∂L/∂x` with the same CG and preconditioner (`A` is symmetric), and contributes `−λᵀ ∂R/∂θ` where `R(x, θ) = grad_E(x; θ)` is the residual the forward drove to zero. One extra CG solve per backward, and memory flat in the iteration count.

**Why it cannot be skipped here.** pyCMM detaches its solve entirely and recovers forces from stationarity: with `A x = −b` and `E = ½xᵀAx + bᵀx`, the derivative at fixed `x` is already the total derivative, because the missing term carries the factor `(A x + b) = 0`. That still holds and covers the *energy* for free — this arrangement gets it without special-casing, since `∂L/∂x` is then zero to CG tolerance and the adjoint exits immediately at no cost.

It does **not** cover the bond correction. The electrostatic environment features are built from `M*` and feed a head that is not variational in them, which is exactly the §6.1 failure. Measured in `tests/test_ff_coupled_solve.py`: with the correction inactive, detaching the solve and running the adjoint give *identical* gradients; with it active they differ by more than a part in a thousand and only the adjoint matches central differences.

## 7. Recurring failure modes to guard against

| Risk | Where it bites | Guard |
| --- | --- | --- |
| **Gauge leakage** | Polar / mid-range regions | Compact support on every correction channel (exactly zero past `r_off`, so only the classical form survives at long range), the per-channel EDA targets, and a linear penalty on mean `r0` |
| **Range separation asked to do routing's job** | Everywhere | Keep them separate (§4.4.1). Note that intra/inter H–H genuinely overlap in distance (intra reaches 1.729 Å, inter starts at 1.611), so no switch can separate *those* — routing keeps the EDA channels clean regardless |
| **Cross-fragmentation inconsistency** | Reactive window | Multi-fragmentation batches + total-E/F consistency loss |
| **Slot-order dependence** | Variable fragmentation count | Masked permutation-invariant pool, not fixed-width concat |
| **Equivariance breakage** | `E`/`∇E`/dipole/quadrupole channels | Correct irreps; CG products / gated nonlinearities only |
| **Wrong forces** | Everywhere with `M*` in ΔE | IFT Jacobian on `M*`; never detach (§6.2, implemented) |
| **Polarization catastrophe** | Coupled solve at short range | Reuse the electrostatic range separation as the coupling's gate — measured min eigenvalue `+0.13` gated against `−0.80` ungated. Watch `cg_pol`/`cg_ct`: a climbing iteration count is the early warning, well before a NaN. Note CG only reports negative curvature if it happens to probe that direction, so this is protected structurally, not detected reliably |
| **`pol` and `ct` corrections competing** | Both are "environment-dependent correction" | They are separated only by their labels and by acting on disjoint pair sets. Same degeneracy shape as the intra-classical/bond leak, which the fit did **not** police on its own — watch the classical/correction split (`pol_ff`, `ct_ff`) from epoch one, not the MAE |
| **`E_pol` losing its sign guarantee** | Once `g` trains the levels' response parameters apart | Exact only while the levels share parameters. Log `pol_ff`; `env_weight` is the lever |

## 8. What the design buys

* **Emergent Reactivity and CT:** Reactivity and charge transfer emerge from smooth fragmentation mixing and dynamically updating features, eliminating the need for rigid diabatic states or explicit CT pairwise terms.
* **Dynamic Physics Hand-off:** Pair-specific range-separation parameters allow the network to seamlessly switch off classical FF interactions as atomic pairs transition from intermolecular distances to intramolecular bonds.
* **Exact charge conservation** by construction through split-charge variables.
* **Data efficiency** from EDA: Auxiliary components regularize the short-range representation, while totals and forces govern the reactive interpolation.