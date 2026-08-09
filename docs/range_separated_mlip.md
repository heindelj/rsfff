# Range-Separated Polarizable MLIP with EDA-Guided Fragmentation Mixing

## 1. Purpose and design philosophy

The goal is a reactive machine-learned interatomic potential (MLIP) that reproduces the Born–Oppenheimer ground-state surface `E(R; Q, S)` while remaining physically well-behaved across regions where the local charge/spin character changes (e.g. homolytic vs. heterolytic bond cleavage, ion-pair vs. radical-pair channels).

The model does **not** attempt to represent true diabatic states. Instead it uses energy-decomposition-analysis (EDA) data as a supervision and parametrization aid, and it recovers reactivity through a smooth, learned mixing over candidate fragmentations combined with a self-consistent polarizable electrostatic solve.

Two physically distinct pieces of information are carried for every atom:

1. **Atomic environment** — the local geometric/chemical descriptor (invariant + equivariant channels).
2. **Electrostatic environment** — the range-separated electrostatic potential, field, and field gradient produced by the rest of the system, conditioned on net charge and spin.

Conservation of total charge is guaranteed *by construction* through a split-charge equilibration (SCEQ) polarization model, not through the learned features.

---

## 2. Physical variables and notation

| Symbol | Meaning |
|---|---|
| `R` | Nuclear coordinates |
| `Q, S` | Total (conserved) charge and spin of the system |
| `a_i(R)` | Atomic-environment features on atom `i` (irrep-resolved) |
| `φ, E, ∇E` | External electrostatic potential (l=0), field (l=1), field gradient (l=2) at each atom |
| `p` | Effective polarization / force-field parameters emitted by the parameter network |
| `M*` | Converged atomic multipoles (monopole + higher moments) from the SCEQ solve |
| `θ` | All learnable network parameters |

The multipoles are carried in their natural irreps: **monopole = l=0, dipole = l=1, quadrupole = l=2 (symmetric traceless)**. All mixing across irrep orders must go through Clebsch–Gordan tensor products or gated (norm-based) nonlinearities — never a scalar MLP on concatenated components, which would break rotational equivariance.

---

## 3. Pipeline overview

```
candidate fragmentations
        │  (approximate initial multipoles per fragmentation)
        ▼
approximate electrostatic environment  ──►  φ, E, ∇E  (range-separated)
        │
        ├── atomic-environment features  a_i
        │
        ▼
   feature mixing  (permutation-invariant over fragmentations)
        │            → initial per-atom features  f_i^(0)
        ▼
   parameter network  →  effective FF parameters  p
        │
        ▼
   SCEQ polarizable solve  (range-separated Coulomb)
        │            → converged multipoles  M*
        ▼
   correction features  =  { a_i , M* }
        │
        ▼
   short-range correction network  →  ΔE  (energy not captured by the polarizable model)
        │
        ▼
   E_total = E_pol(M*) + ΔE
```

---

## 4. Stage-by-stage description

### 4.1 Candidate fragmentations → approximate electrostatic environment

For a given geometry, enumerate a small set of candidate fragmentations. Each fragmentation carries a crude, pre-allocated charge distribution, which is sufficient to seed the electrostatic environment.

The seed choice is likely important near reactive transitions. We will have to explore this as we go.

### 4.2 Electrostatic features

From the approximate charge distribution, compute the **range-separated** electrostatic potential `φ`, field `E`, and field gradient `∇E` at each atom. These are the environment's action on the atom, expressed as a multipole expansion truncated at quadrupole-order response.

- `φ` enters as l=0, `E` as l=1, `∇E` as l=2.
- The range separation splits near-field (handled by the short-range network in the overlap region) from mid/long-range (explicit damped Coulomb). The crossover should be *learned*, not hard-coded at a fixed cutoff.

### 4.3 Feature mixing across fragmentations

Each atom participates in a variable number of candidate fragmentations. The mixing must be **permutation-invariant** over the fragmentation slot and handle a padded/variable count via masking — *not* a fixed-width concatenation, which invites slot-order dependence.

Use attention pooling where:
- **weights are invariant scalars** (built only from l=0 channels or invariant contractions of equivariant ones),
- **values are the (possibly equivariant) features**,
- the same scalar weight is applied across all irrep channels of a fragmentation and summed.

`w_a = softmax_a( g(invariants of f^(a)) / τ )`, then `f_i = Σ_a w_a f_i^(a)`.

The softmax partition of unity provides the smooth transition between regions where different fragmentations dominate — this is the reactivity mechanism and the differentiable gauge-fixing. A learnable temperature `τ` lets the mixing sharpen where a crossing is tight.

> **Note on the "identity during training" assumption.** Training on one fragmentation per sample does **not** teach the mixing network to be consistent across fragmentations — identity-on-one-input is a different function that merely agrees at one point. Cross-fragmentation smoothness must be trained in explicitly (Section 5).

### 4.4 Parameter network → effective FF parameters

The mixed features `f_i` are mapped to the **effective polarization parameters** `p`: multipolar electronegativities, hardnesses, and split-charge-transfer stiffnesses. Injecting the learning at the *parameter* level (rather than directly at the charges) keeps the physical solve in charge of conservation while letting the network absorb the short-range and environment dependence of the parameters. These effective parameters legitimately absorb some short-range interaction character.

### 4.5 SCEQ polarizable solve → converged multipoles

**Split-charge equilibration** provides two things:

- A **multipolar electronegativity** produces the *permanent* multipoles (intrinsic anisotropy: lone pairs, σ-holes).
- **Charge-transfer variables** `q_AB` move charge between pairs and are counted with opposite sign on each atom, so charge is conserved *per channel by construction*. Channels can be range-limited or closed as a bond breaks, keeping fragments at integer charge with no global re-solve. This is what makes clean ion-pair/neutral separation possible.

The stationarity condition `∂E_pol/∂M = 0` (a linear system for SCEQ, `H_MM M* = b`) is solved with the explicit range-separated Coulomb interactions. Damping (Thole / Tang–Toennies) should use **fixed or tightly-regularized** damping lengths so the explicit Coulomb term is a rigid backbone — a freely learnable damping length will fight the parameter and correction networks for the same mid-range energy.

The output is the converged multipole set `M*`.

### 4.6 Correction network → short-range energy

The final features are `{ a_i , M* }`: the atomic-environment descriptor plus the converged multipoles. These feed the short-range correction network that predicts `ΔE`, the portion of the energy not captured by the polarizable model (charge penetration, short-range anisotropy, exchange/dispersion residuals).

Conditioning on the *converged* multipoles (the settled output of the mutual-polarization solve) rather than only the raw external field gives the correction network the already-equilibrated, nonlocal information it needs to learn penetration and anisotropy corrections that depend on the actual charge distribution.

### 4.7 Spin

For a first version, condition globally on total multiplicity. For homolytic/heterolytic discrimination, condition additionally on **atomic spin populations / spin-density moments** from the solve — the spin-sector analog of the charge multipoles, which localize *where* radical character sits.

---

## 5. Training with EDA data

EDA supplies a decomposition of the interaction energy (frozen / polarization / charge-transfer) for a **given, fixed fragmentation**. Use it as follows:

- **Primary objective:** total energy and forces (always). These are the physical, basis-independent targets.
- **Auxiliary objective:** match EDA components as an *inductive bias / regularizer*, not as ground truth. EDA components are scheme- and basis-dependent gauge quantities; letting them define the loss fits a gauge choice.

### 5.1 Cross-fragmentation consistency (required)

EDA supervises the *within-fragmentation* decomposition, but **not** how to mix across fragmentations. To force the mixing network to learn a smooth transition rather than merely interpolate:

- In a fraction of batches, present **≥2 fragmentations of the same geometry**.
- Add a **consistency loss** penalizing disagreement in total `E` and forces `F` across those fragmentations (components may differ — they are gauge; totals may not).

Without this term the mixing network is under-constrained precisely in the reactive window where two fragmentations are comparably dominant.

### 5.2 Frame-convergence diagnostic

The compression removes the fragmentation gauge only if the canonical representation is insensitive to *which* fragmentations populate the atlas. Test directly: hold a system fixed, keep adding fragmentations to its decomposition, and confirm that `f_i` (and the resulting `E`/`F`) plateau. Drift means the atlas is under-sampled and is doing hidden gauge-fixing.

### 5.3 Monomer pretraining

Pretrain the parameter function and short-range network on **monomers embedded in many point-charge environments**. This isolates the response physics before entangling it with fragmentation ambiguity, and it has a physical ground truth: a monomer's learned response to an external field can be validated against finite-field DFT polarizabilities and dipole derivatives. Make "reproduces known polarizabilities" a hard gate — everything downstream inherits errors in the response.

---

## 6. Differentiability and the self-consistent solve

### 6.1 The subtlety: forces through `M*`

The total energy is

```
E_total = E_pol(M*; R, p)  +  ΔE(a, M*; θ)
```

Forces are the *total* derivative `F = −dE_total/dR`, and `M*` depends on `R`. Expanding:

```
dE_total/dR =  ∂E_pol/∂R                          (explicit)
             + (∂E_pol/∂M*)·(dM*/dR)              → 0   (Hellmann–Feynman)
             + ∂ΔE/∂R                             (explicit)
             + (∂ΔE/∂M*)·(dM*/dR)                 ≠ 0
             + (∂E/∂p)·(dp/dR)
```

- The polarization term `(∂E_pol/∂M*)(dM*/dR)` **vanishes** because `M*` is a stationary point of `E_pol` (`∂E_pol/∂M = 0`). This is the standard variational / Hellmann–Feynman cancellation.
- The correction term `(∂ΔE/∂M*)(dM*/dR)` does **not** vanish, because `ΔE` is *not* variational in `M*` — the multipoles are an input feature, so `∂ΔE/∂M* ≠ 0`.

**Consequence:** the converged multipoles must carry their Jacobian `dM*/dR` into the correction network. Detaching `M*` (treating the solve as a constant black box) drops the correction term and yields analytic forces that do not equal `−dE/dR`. Training on those forces corrupts the model.

### 6.2 The correct treatment: implicit differentiation

Differentiate `M*` via the **implicit function theorem** applied to the stationarity condition `g(M*, R, p) = ∂E_pol/∂M = 0`:

```
dM*/dR = − (H_MM)⁻¹ ( ∂g/∂R + (∂g/∂p)(dp/dR) )
```

where `H_MM = ∂²E_pol/∂M²` is the polarization Hessian.

- This leaves the **solver iterations** off the computational graph (no unrolling, no iteration history), while still injecting the correct gradient.
- For SCEQ, `H_MM` is exactly the hardness/Coulomb matrix already factorized to perform the solve, so the adjoint reuses that factorization — the correct gradient is nearly free.

The same implicit-differentiation applies to training-time parameter gradients `dM*/dθ` (through `p`).

### 6.3 Summary of the gradient rule

- Do **not** unroll the solver → use IFT.
- Do **not** detach `M*` → the correction network's dependence on `M*` is physical and contributes to forces.
- The Hellmann–Feynman shortcut applies **only** to the variational polarization energy, **not** to the non-variational correction network.

### 6.4 Fixed-point existence and branch selection

The joint (multipoles, parameters) fixed point is not guaranteed unique. Near a crossing, competing charge states can produce multiple stable fixed points (physically: charge-state bistability). For ground-state reactivity, make the charge solve a **total-Q-constrained equilibration** so the constraint collapses spurious basins to the physical ground state. Monitor solve conditioning in the reactive region (where it is worst); a small penalty on charge-transfer-variable magnitude ("charge-transfer reluctance") both regularizes the solve and encodes real chemistry.

---

## 7. Recurring failure modes to guard against

| Risk | Where it bites | Guard |
|---|---|---|
| **Gauge leakage** (short-range net absorbs electrostatic energy) | Polar / mid-range regions | Keep `E_pol` from `M*` a fixed non-learned contribution; check `ΔE` stays short-ranged in `R` |
| **Range-separation ambiguity** (params vs. explicit Coulomb vs. ΔE fight over mid-range) | ~mid-range window | Fix/regularize damping length; rigid Coulomb backbone |
| **Cross-fragmentation inconsistency** | Reactive window | Multi-fragmentation batches + total-E/F consistency loss |
| **Slot-order dependence** | Variable fragmentation count | Masked permutation-invariant pool, not fixed-width concat |
| **Equivariance breakage** | `E`/`∇E`/dipole/quadrupole channels | Correct irreps; CG products / gated nonlinearities only |
| **Wrong forces** | Everywhere with `M*` in ΔE | IFT Jacobian on `M*`; never detach |
| **Branch selection by seed** | Crossings | Geometry-responsive seed; constrained equilibration |

---

## 8. What the design buys

- **Exact charge conservation** by construction (split-charge variables), with the ability to localize/close transfer channels as bonds break.
- **Nonlocal, self-consistent electrostatics** feeding a local correction, so long-range charge redistribution is captured.
- **Reactivity through smooth fragmentation mixing** rather than explicit diabatic states — correct for avoided crossings (the regime of interest), at the cost of not representing true conical-intersection cusps.
- **Data efficiency** from EDA: components regularize the representation; totals and forces (plus cross-fragmentation consistency) carry the reactive interpolation.
- **Physically grounded pretraining** with a direct observable check (monomer polarizabilities).

### Boundary of validity

An infinitely smooth (softmax) mixing represents avoided crossings but rounds a true conical intersection into an average. Since the model targets ground-state reactivity through avoided crossings, this is an acceptable — arguably correct — limitation, not a defect to be discovered later on a system with a genuine seam.