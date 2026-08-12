# Range-Separated Polarizable MLIP with EDA-Guided Fragmentation Mixing

## 1. Purpose and design philosophy

The goal is a reactive machine-learned interatomic potential (MLIP) that reproduces the Born–Oppenheimer ground-state surface `E(R; Q, S)` while remaining physically well-behaved across regions where the local charge/spin character changes (e.g. homolytic vs. heterolytic bond cleavage, ion-pair vs. radical-pair channels).

The model does **not** attempt to represent true diabatic states. Instead, it uses energy-decomposition-analysis (EDA) data as a training strategy to ensure the physical components of the force field remain well-defined. The total energy is partitioned into five conceptual components: **Electrostatics, Polarization, Pauli repulsion, Dispersion, and Bonding (deformation)**.

Reactivity is recovered through a smooth, learned mixing over candidate fragmentations combined with a self-consistent polarizable electrostatic solve. Charge Transfer (CT) is not treated as an explicit pairwise interaction; rather, it emerges naturally as the learned correction to the bond energy and other interactions when features change across mixed partitionings.

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

Use attention pooling where:

* **weights are invariant scalars** (built only from l=0 channels or invariant contractions of equivariant ones),
* **values are the (possibly equivariant) features**,
* the same scalar weight is applied across all irrep channels of a fragmentation and summed.

`w_a = softmax_a( g(invariants of f^(a)) / τ )`, then `f_i = Σ_a w_a f_i^(a)`.

The softmax partition of unity provides the smooth transition between regions where different fragmentations dominate. The learned combination of these features is what ultimately dictates the shift in energy (e.g., Charge Transfer).

### 4.4 Parameter network → effective FF & Range-Separation parameters

The mixed features `f_i` are mapped to the **effective polarization parameters** `p` (multipolar electronegativities, hardnesses, split-charge stiffnesses) and **pair-specific range-separation parameters** `β_ij`.

Allowing the range-separation to be pair-specific ensures that as the features change under feature-mixing, the classical FF interactions can be rigorously switched off or modulated at short range depending on whether the pair is acting strictly intermolecularly or transitioning into an intramolecular bond.

### 4.5 SCEQ polarizable solve → converged multipoles

**Split-charge equilibration** provides:

* A **multipolar electronegativity** producing the *permanent* multipoles.
* **Charge-transfer variables** `q_AB` moving charge between pairs, conserving charge per channel by construction.

The stationarity condition `∂E_pol/∂M = 0` is solved with the explicit range-separated Coulomb interactions, utilizing the pair-specific damping parameters predicted by the network. The output is the converged multipole set `M*`.

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

Because we propose several partitionings of the system and compute those features at inference, **Charge Transfer (CT) is not modeled as an explicit interaction term.** Instead, the learned combination of features provides the CT energy. It emerges naturally as the learned correction to the bond energy, alongside the changes to the classical interactions that arise when the local features (and pair-specific damping parameters) update during the mixing process.

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

## 7. Recurring failure modes to guard against

| Risk | Where it bites | Guard |
| --- | --- | --- |
| **Gauge leakage** | Polar / mid-range regions | Ensure pair-specific damping properly shifts mid-range dominance between explicit Coulomb and MLIP heads |
| **Cross-fragmentation inconsistency** | Reactive window | Multi-fragmentation batches + total-E/F consistency loss |
| **Slot-order dependence** | Variable fragmentation count | Masked permutation-invariant pool, not fixed-width concat |
| **Equivariance breakage** | `E`/`∇E`/dipole/quadrupole channels | Correct irreps; CG products / gated nonlinearities only |
| **Wrong forces** | Everywhere with `M*` in ΔE | IFT Jacobian on `M*`; never detach |

## 8. What the design buys

* **Emergent Reactivity and CT:** Reactivity and charge transfer emerge from smooth fragmentation mixing and dynamically updating features, eliminating the need for rigid diabatic states or explicit CT pairwise terms.
* **Dynamic Physics Hand-off:** Pair-specific range-separation parameters allow the network to seamlessly switch off classical FF interactions as atomic pairs transition from intermolecular distances to intramolecular bonds.
* **Exact charge conservation** by construction through split-charge variables.
* **Data efficiency** from EDA: Auxiliary components regularize the short-range representation, while totals and forces govern the reactive interpolation.