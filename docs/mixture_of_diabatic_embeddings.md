# Latent-Space Diabatic Range-Separated MLIPs
## Architecture, Mathematical Formulation, and Ion-Pair Dissociation Testing Protocol
### Revision 2: Split-Charge Equilibration, Learned Channel Hardness, and Local Diabatic Gating

---

## 1. Executive Summary & Theoretical Foundations

Standard Machine-Learned Interatomic Potentials (MLIPs)—whether based on Atomic Cluster Expansions (ACE) or equivariant Message Passing Neural Networks (MPNNs)—rely on a monolithic, continuous latent representation of atomic environments. While highly successful near equilibrium geometries, this monolithic paradigm fails fundamentally when describing physical processes that involve **electronic state transitions, non-local charge transfer, or asymptotic dissociation into distinct charge and spin states**.

When an ion pair dissociates (e.g., $\mathrm{Na}^+ + \mathrm{Cl}^- \longrightarrow \mathrm{Na}^\bullet + \mathrm{Cl}^\bullet$), standard MLIPs are forced to smoothly interpolate between radically different electronic structures using a single set of atom-centered features. Consequently, they cannot rigorously capture:

1. **The Asymptotic Coulomb Tail:** Distinguishing between the long-range $-q_1 q_2 / r$ electrostatic attraction of closed-shell ions and the $-C_6 / r^6$ dispersion interaction of neutral radicals.
2. **Avoided Crossings (The Harpoon Mechanism):** The sharp physical transition where the ionic diabatic state crosses the neutral radical diabatic state as a function of separation distance.
3. **Integer Charge Localization at Dissociation vs. Many-Body Charge Transfer at Contact:** Conventional global Electronegativity Equalization (EEM/QEq) permits unphysical fractional charge transfer between arbitrarily distant fragments ("metallic" delocalization). The correct physics is two-sided: fragments in short-range contact genuinely exchange small amounts of charge (a dominant source of many-body cooperative polarization, e.g., in hydrogen-bonded networks), while dissociated fragments must localize exactly integer charge.
4. **Rigorous Energy Decomposition Analysis (EDA) Integration:** Classical and polarizable long-range force fields require well-defined unperturbed monomer reference states to compute induction and electrostatics without unphysical short-range catastrophe.

### The Diabatic Latent-Mixing + Split-Charge Solution
This document details a rigorous mathematical architecture and testing protocol for a **Latent-Space Diabatic Range-Separated MLIP**. By conditioning short-range atomic features on explicit reference molecular/atomic diabatic states—frozen monomer features corrected by a state-decorated atomic cluster expansion (ACE)—and blending them directly in latent space, we preserve the formal identity (valence, spin, and baseline response) of chemical fragments.

Four design decisions define the architecture:

- **A permanently frozen monomer stack anchors all asymptotics.** Fragment reference features, 1-body energies, and baseline response parameters are trained once on isolated fragments and frozen; every interaction effect enters as a state-decorated atomic cluster expansion (ACE) correction that vanishes *identically* at fragment isolation, with prediction heads constructed (via a subtraction trick) to be exactly zero there for any network weights. Dissociation limits, ionization-energy differences, and the gate's Coulomb bias are thereby consistent by construction, not by regularization.
- **State mixing occurs in latent space, locally per reactive center, prior to property prediction.** A single downstream network maps the blended representation to short-range atomic energies (capturing non-linear resonance stabilization, automatically confined to short range) and effective response tensors. Gating is factorized over reactive centers, guaranteeing size extensivity.
- **Charge conservation is structural, not constrained.** The long-range polarizable solve uses **Split-Charge Equilibration (SQE)**: atomic charges are parameterized as diabatic baseline charges plus antisymmetric transfers along an explicit **channel graph**. Total fragment charge is conserved identically for any channel values—no Lagrange multipliers. Every channel compliance is multiplied by a smooth pairwise switching function vanishing at the short-range cutoff, so inter-fragment charge transfer is structurally zero beyond $r_{\text{SR}}$ and asymptotic charges are exactly integer.
- **Charge-transfer physics is learned, and exists even among ground states.** Per-channel compliance (inverse bond hardness) is predicted by a network head from the blended latent features. Inter-fragment channels exist between *all* fragment pairs in short-range contact—not only those merged by an active diabat—so neighboring closed-shell molecules (e.g., two ground-state waters) exchange small amounts of charge that couple automatically into the global polarization solve. This is a deliberate mechanism for capturing **many-body charge transfer**: ambient compliances are small (trained against EDA charge-transfer energies), while channels between fragments merged by an active diabat become soft, enabling the large transfers of genuine bond formation. The formal diabatic charges enter only as baselines; deviations are governed by the learned compliance landscape, the electrostatic environment, and the gating weights.

Global spin is treated as a **bookkeeping constraint on valid combinations of diabatic states** (e.g., a global singlet restricts which local fragment-spin assignments may be simultaneously active); the model does not attempt to describe spin degrees of freedom explicitly. A simple damped dispersion model (e.g., Tang–Toennies-damped $C_6/r^6$ with fragment-state-conditioned coefficients) is included in the long-range component; it is conceptually standard and elided from the formalism below except where its diabatic conditioning matters.

---

## 2. Mathematical Formulation & Architecture

The total energy of the system is partitioned into a single long-range polarizable electrostatic/induction/dispersion component and a short-range quantum mechanical component:

$$E_{\text{total}} = E_{\text{LR}}^{\text{final}}\left(\boldsymbol{\chi}^{\text{final}},\ \boldsymbol{\alpha}_{\text{eff}}^{\text{final}},\ \{\kappa_{ij}\},\ \{q_i^{(0)}\}\right) + \sum_{i} E_{i,\text{SR}}\left(\mathbf{h}_i^{\text{final}}\right)$$

where $\mathbf{h}_i^{\text{final}}$ represents the smoothly blended atomic latent features formed over an active space of enumerated diabatic states, and $E_{\text{LR}}^{\text{final}}$ is obtained from a single global SQE solve over the channel graph.

```
       [ Input Coordinates & Formal Diabatic References {K} ]
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│  Short-Range MLIP (per Diabat)  │     │   Zeroth-Order Coulomb Estimate │
│  • Frozen Monomer Features      │     │  • Asymptotic Energy Ẽ^(K)(r)   │
│  • State-Decorated ACE Δh^(K)   │     │  • Long-Range Coordinate Bias   │
└────────────────┬────────────────┘     └────────────────┬────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                    [ Validity Envelope Pruning Ω_K(r) ]
                                     │
                    [ LOCAL Attention Gating per Reactive Center ]
                                     │
                                     ▼
        Blended Latents: h_i^final = h_{i,0}^final + Δh_i^final  (∑_{K_R} c_{K_R} · each)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
┌──────────────────────────────────────────┐     ┌──────────────────────────────────────────┐
│   Short-Range Energy (Frozen Baseline    │     │   Effective Response, EEM & Channel      │
│   + Isolation-Exact Correction)          │     │   Parameter Heads (same structure)       │
│  • 1-Body Baseline ∑ c_K E_1^(K) (frozen)│     │  • χ_i^final = ∑ c_K χ_{i,0}^(K) + Δχ_i  │
│  • ΔE via Subtraction Trick (≡0 at       │     │  • α_eff^final = ∑ c_K α_{i,0}^(K) + Δα_i│
│    isolation): Pauli/Penetration,        │     │  • Channel Compliance s_ij (→ κ_ij)      │
│    Non-Linear Resonance Stabilization    │     └────────────────────┬─────────────────────┘
└─────────────────────┬────────────────────┘                          │
                      │                                               ▼
                      │                          [ Channel Graph Assembly (all short-range
                      │                            contacts) & Pairwise Switch S(r_ij) ]
                      │                                               │
                      │                                               ▼
                      │                          [ Single Global SQE Solve over {p_ij} ]
                      │                          • q_i = q_i^(0) + ∑_j p_ij  (conservation
                      │                            structural — no Lagrange multipliers)
                      │                          • Damped Multipole Operators + Dispersion
                      ▼                                               ▼
            Short-Range Energy ∑ E_{i,SR}            Long-Range Energy E_LR^final
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                          Total Energy E_total & Analytical Forces
```

### 2.1 Hierarchical Featurization: Frozen Monomer Stack + State-Decorated ACE Correction

The featurization is a strict hierarchy: **atomic reference embeddings → frozen state-aware intra-fragment features → a state-decorated atomic cluster expansion (ACE) correction over frozen inputs → blended final features.** The division is not merely organizational—the frozen monomer stack carries the model's asymptotic guarantees by construction, while the correction layer carries all interaction physics and vanishes identically at fragment isolation.

For a given diabatic configuration $K$, every atom $i$ is assigned to a specific chemical fragment $F_a^{(K)}$ with a formal charge $Q_a^{(K)}$ and spin multiplicity $2S_a^{(K)} + 1$.

1. **Atomic Reference Embeddings:** Instead of a single generic element embedding $\mathbf{z}_Z$, each atom is initialized with a reference embedding conditioned on its formal diabatic fragment state:
   $$\mathbf{e}_i^{(K)} = \mathcal{E}\left(Z_i,\ \text{FragmentType}(i \in F_a^{(K)}),\ Q_a^{(K)},\ S_a^{(K)}\right)$$

2. **Frozen Intra-Fragment (Monomer) Features:** A state-aware intra-fragment network, message-passing only *within* the fragment's covalent graph, maps the reference embeddings and internal geometry to monomer features:
   $$\mathbf{h}_{i,0}^{(K)} = \mathcal{F}_{\text{mono}}\left(\{\mathbf{e}_j^{(K)}, \mathbf{r}_j\}_{j \in F_a^{(K)}}\right)$$
   $\mathcal{F}_{\text{mono}}$ is trained in Phase 1 on isolated (including distorted) fragments against rich auxiliary targets—energies, multipoles, response tensors—so that the frozen representation exposes the physically relevant directions downstream layers will need, then **frozen permanently**. All subsequent training phases treat $\mathbf{h}_{i,0}^{(K)}$ as fixed inputs. (If additional flexibility proves necessary, a small learned linear adapter of the frozen features may be trained in Phase 2; the monomer network itself is never revisited.)

3. **State-Decorated ACE Correction (the "Molecular Cluster Expansion"):** As fragments interact at short range, their internal electronic structure deforms. This is modeled as an additive, atom-centered correction:
   $$\mathbf{h}_i^{(K)} = \mathbf{h}_{i,0}^{(K)} + \Delta\mathbf{h}_i^{(K)}, \qquad \Delta\mathbf{h}_i^{(K)} = \text{ACE}_\nu\!\left(\left\{\mathbf{h}_{j,0}^{(K)},\ \mathbf{r}_{ij}\right\}_{r_{ij} < r_{\text{SR}}}\right)$$
   Concretely, the atomic density basis (radial functions × spherical harmonics over neighbors within $r_{\text{SR}}$, typically $4.5$–$5.0\ \text{Å}$) is decorated with channel weights derived from each neighbor's *frozen diabatic features* $\mathbf{h}_{j,0}^{(K)}$ rather than bare element identity; density products then deliver body-ordered ($\nu$-truncated) corrections at linear cost. Design commitments:
   - **Not a fragment-enumerated MBE.** A literal fragment-level many-body expansion requires $\mathcal{O}(S^2)$ dimer and $\mathcal{O}(S^3)$ trimer scan families as the fragment-state library of size $S$ grows, and tabulated terms cannot generalize to unsampled state combinations. The state-decorated ACE achieves genuine body-ordered structure (truncatable order, per-order attribution) with a *shared* parameterization: learned state embeddings interpolate across fragment-state pairs never explicitly sampled.
   - **Cross-diabat amortization.** The geometric density basis is state-independent and built once per configuration; evaluating $M$ active diabats costs one basis construction plus $M$ cheap channel contractions, not $M$ full feature passes.
   - **Exact receptive field.** A single-shot ACE (at most one message-passing layer) keeps the receptive field at exactly $r_{\text{SR}}$; deep message passing would grow it to $L \cdot r_{\text{SR}}$, silently violating the range separation and the frozen-feature asymptotics.
   - **The isolation zero.** Because every basis function carries the $r_{\text{SR}}$ cutoff, $\Delta\mathbf{h}_i^{(K)} \equiv 0$ *identically* for an isolated fragment—an exact architectural property, not a trained one. This zero anchors the monomer-preservation guarantees of Section 2.3.
   - **Low body order suffices.** The slowly-converging many-body physics (polarization, cooperative charge transfer) is carried by the global SQE solve and the ambient channels (Section 2.4); the ACE correction represents only the residual short-range physics (Pauli, penetration, short-range CT stabilization), whose body-order expansion converges rapidly. $\nu = 3$ is a defensible design point, not a compromise.

### 2.2 Local Diabatic Gating & Size-Extensive Latent Mixing

**Why gating must be local.** A single global softmax over enumerated system-wide diabatic states is neither size-extensive nor computationally tractable. If a condensed-phase system contains $N_c$ independent reactive centers, each with $m$ local states, the global state space is the product set of size $m^{N_c}$, and the mixing weight at one center would depend on the logits of arbitrarily distant, physically decoupled centers. Furthermore, a pooled global logit grows in magnitude with system size, silently rescaling the softmax temperature $\tau$ relative to its dimer-calibrated value. A global softmax factorizes into independent local softmaxes only if the logits are exactly additive over centers, which a nonlinear pooled MLP does not guarantee.

**The reactive-center factorization (mean-field ansatz).** We therefore define gating locally:

1. **Reactive Center Identification:** Continuous coordination-defect analysis (Step 1, Section 3) identifies localized reactive centers $R = 1, \dots, N_c$. Each center carries its own $k$-hop local active space of diabatic states $\{K_R\}$, enumerated over the atoms in its $k$-hop graph. Two provisional centers whose $k$-hop graphs overlap, or that are spanned by a common candidate diabat, are **merged into a single center** so that every diabatic state is wholly owned by exactly one center. Atoms belonging to no reactive center carry a single reference state ($c \equiv 1$).

2. **Local Logits with Local Coulomb Bias:** For each state $K_R$ of center $R$, pool only over the atoms of that center and subtract the *local* zeroth-order asymptotic estimate:
   $$z_{K_R} = \text{MLP}_{\text{gate}}\left(\sum_{i \in R} \mathbf{W}_{\text{pool}}\, \mathbf{h}_i^{(K_R)}\right) - \beta\, \tilde{E}^{(K_R)}(r), \qquad \tilde{E}^{(K_R)}(r) = E_{\text{monomer}}^{(K_R)} + \sum_{a < b \,\in R} \frac{Q_a^{(K_R)} Q_b^{(K_R)}}{r_{ab}}$$
   The zeroth-order bias is what gives the gate long-range geometric awareness beyond $r_{\text{SR}}$ (where the ACE correction vanishes identically and diabatic features reduce to frozen monomer features), guaranteeing Harpoon crossings are triggered at the correct asymptotic separations. Optionally, a small set of explicit long-range scalars (inter-fragment separations $r_{K_R}$, field estimates at fragment centroids) may be appended to the pooled features so that the learned component of the logit retains geometric sensitivity beyond $r_{\text{SR}}$ rather than delegating the entire crossing shape to $\beta\tilde{E}$.

3. **Local Validity-Weighted Softmax:** Mixing coefficients are computed independently per center:
   $$c_{K_R} = \frac{\Omega_{K_R} \exp\left(z_{K_R} / \tau\right)}{\sum_{J_R} \Omega_{J_R} \exp\left(z_{J_R} / \tau\right)}$$
   where $\Omega_{K_R} \in [0,1]$ is a $C^2$-continuous compact-support bump function guaranteeing zero force noise when states are pruned. Because each softmax runs over a bounded local state count, $\tau$ has a fixed, system-size-independent meaning, and the total computational cost is $\mathcal{O}(N_c)$.

4. **Global State Interpretation.** The implied global mixing weight of a product state $(K_1, K_2, \dots)$ is $\prod_R c_{K_R}$—a mean-field (Hartree-like) factorization over centers. Correlated multi-center states (e.g., genuine long-range electron transfer between distant redox partners) are handled by explicitly merging the participating centers into one active space, not by globalizing the softmax.

5. **Global Spin Bookkeeping.** A global spin (or charge) constraint acts as a filter on the *product* space: local states whose fragment spin assignments cannot participate in any spin-valid global product are pruned from their local active spaces before gating. This uses the diabatic spin labels purely combinatorially; no explicit spin dynamics are modeled.

6. **Latent Feature Blending:** All active local diabatic representations are collapsed into a single physical representation per atom, with the baseline and correction components tracked separately for the heads of Section 2.3:
   $$\mathbf{h}_{i,0}^{\text{final}} = \sum_{K_R} c_{K_R} \mathbf{h}_{i,0}^{(K_R)}, \qquad \Delta\mathbf{h}_i^{\text{final}} = \sum_{K_R} c_{K_R} \Delta\mathbf{h}_i^{(K_R)}, \qquad \mathbf{h}_i^{\text{final}} = \mathbf{h}_{i,0}^{\text{final}} + \Delta\mathbf{h}_i^{\text{final}} \quad (i \in R)$$
   *Why Latent Mixing?* Taking linear combinations of features rather than late energies grants downstream networks the expressivity to learn **non-linear resonance stabilization**. At avoided crossings ($c_1 \approx c_2 \approx 0.5$), the network maps the blended latent vector to an adiabatic ground-state energy that drops below the linear average of isolated diabatic curves, mimicking off-diagonal Hamiltonian coupling without solving a matrix equation. Crucially, this nonlinearity is *automatically confined to short range*: beyond $r_{\text{SR}}$, $\Delta\mathbf{h} \equiv 0$ and the head constructions of Section 2.3 make the energy exactly linear in $c$—the physically correct limit, since coupling matrix elements vanish with overlap. (At an exactly degenerate long-range seam, linear blending yields the average of the two states—but at degeneracy the average *is* the minimum, and the Coulomb bias breaks degeneracy elsewhere, so this is benign.)
   *Interpretation of the weights.* The $c_{K_R}$ live on the probability simplex (softmax is its natural unconstrained parameterization), and the gate is best understood as **amortized inference** of a variational mixing solution: a future self-consistent (MCSCF-like) extension would define $E_{\text{total}}(\{c\})$ and minimize over the simplex directly—at much greater cost—while the gating network learns to emit the minimizer in a single pass. To keep that door open, training includes evaluations at perturbed mixing coefficients (Section 4.3, Phase 3) so that $E(\{c\})$ has physically sensible curvature off the gate's output manifold, not merely correct values on it. Note that $c_{K_R}$ blend *representations*, and the nonlinear energy head deliberately blurs the distinction between a classical ensemble and a coherent superposition; $c = 0.5$ should not be read as a literal state population.

### 2.3 Property Prediction: Frozen Monomer Baselines + Isolation-Exact Correction Heads

Freezing monomer *features* alone is an incomplete freeze: if shared prediction heads were trained in Phases 2–3, the model's output for an isolated monomer would drift even with fixed inputs, silently invalidating Phase-1 calibration. This matters quantitatively—dissociation limits are differences of large monomer energies, and the Harpoon crossing position shifts by $\sim 1\ \text{Å}$ per $0.1\ \text{eV}$ of $IE - EA$ error—and it would desynchronize the gate (whose Coulomb bias uses the frozen $E_{\text{monomer}}^{(K)}$) from the energy head's actual asymptote. Monomer preservation is therefore enforced **architecturally**, not by regularization: every predicted quantity is a frozen, $c$-linear 1-body baseline plus a correction that vanishes *identically* at fragment isolation.

1. **Frozen 1-Body Baseline (Blended Linearly):** The Phase-1 monomer stack includes frozen heads for the 1-body energy $E_1^{(K)}(i)$ (a function of intra-fragment geometry, covering distorted monomers), baseline electronegativity $\boldsymbol{\chi}_{i,0}^{(K)}$, polarizability $\boldsymbol{\alpha}_{i,0}^{(K)}$, and the baseline charge distributions entering $q^{(0)}$. Baselines blend linearly with the mixing weights—exactly the weighted 1-body energies of the diabats:
   $$E_{i,\text{SR}}^{\text{base}} = \sum_{K_R} c_{K_R} E_1^{(K_R)}(i), \qquad \boldsymbol{\chi}_{i}^{\text{base}} = \sum_{K_R} c_{K_R}\, \boldsymbol{\chi}_{i,0}^{(K_R)}, \qquad \boldsymbol{\alpha}_{i}^{\text{base}} = \sum_{K_R} c_{K_R}\, \boldsymbol{\alpha}_{i,0}^{(K_R)}$$

2. **Isolation-Exact Correction Heads (the Subtraction Trick):** Corrections take the blended baseline and correction features as *separate* inputs, and subtract the head's own value at zero correction:
   $$\Delta E_{i,\text{SR}} = \text{MLP}_{\Delta E}\!\left(\left[\mathbf{h}_{i,0}^{\text{final}},\ \Delta\mathbf{h}_i^{\text{final}}\right]\right) - \text{MLP}_{\Delta E}\!\left(\left[\mathbf{h}_{i,0}^{\text{final}},\ \mathbf{0}\right]\right)$$
   and identically for $\Delta\boldsymbol{\chi}_i$ and $\Delta\boldsymbol{\alpha}_i$. Because $\Delta\mathbf{h} \equiv 0$ for an isolated fragment (Section 2.1), the correction is **exactly zero at isolation for any network weights**—full nonlinear expressivity at short range, machine-precision monomer preservation at long range. The subtracted term is constant per fragment-state and cacheable, so the overhead is negligible. This construction *replaces* the earlier soft training constraint "$\Delta\boldsymbol{\alpha}, \Delta\boldsymbol{\chi} \to 0$ as $r \to r_{\text{SR}}$" with an exact architectural property, and simultaneously guarantees exact linearity of the energy in $c$ beyond $r_{\text{SR}}$ (Section 2.2.6). The final quantities are
   $$E_{i,\text{SR}} = E_{i,\text{SR}}^{\text{base}} + \Delta E_{i,\text{SR}}, \qquad \boldsymbol{\chi}_i^{\text{final}} = \boldsymbol{\chi}_i^{\text{base}} + \Delta\boldsymbol{\chi}_i, \qquad \boldsymbol{\alpha}_{i,\text{eff}}^{\text{final}} = \boldsymbol{\alpha}_i^{\text{base}} + \Delta\boldsymbol{\alpha}_i$$
   The correction $\Delta E_{i,\text{SR}}$ carries the short-range EDA components (Pauli repulsion, charge penetration) and the non-linear resonance stabilization at avoided crossings; the baseline carries all 1-body physics.

3. **Channel Compliance (Learned Bond Hardness):** For every edge $(i,j)$ of the channel graph (Section 2.4), a pair head predicts a raw **compliance** (inverse hardness):
   $$s_{ij}^{\text{raw}} = \text{softplus}\!\left(\text{MLP}_{s}\left(\mathbf{h}_i^{\text{final}}, \mathbf{h}_j^{\text{final}}, e(r_{ij})\right)\right) \geq 0$$
   where $e(r_{ij})$ is a radial basis expansion. Predicting compliance $s = 1/\kappa$ rather than hardness $\kappa$ is deliberate: channel closure corresponds to $s \to 0$ (a bounded, well-conditioned limit) rather than $\kappa \to \infty$, and the pairwise switching function acts multiplicatively on $s$ (Section 2.4.3). No subtraction trick is needed here—the switch $S(r_{ij})$ already enforces exact shutdown of inter-fragment channels at $r_{\text{SR}}$, and intra-fragment compliances are legitimately monomer properties trained in Phase 1 and frozen with the rest of the monomer stack. Because the inputs are the *blended* features, the learned compliance is implicitly conditioned on the local mixing weights $c_{K_R}$—e.g., a channel between two fragments softens as a diabat merging them gains weight—without any explicit $c$-dependence in the head. Between ordinary ground-state fragments the head yields small but nonzero ambient compliances, the mechanism for many-body charge transfer (Section 2.4.3).

4. **What Trains When:** "Freeze the monomer stack" means, operationally: $\mathcal{F}_{\text{mono}}$, $E_1^{(K)}$, $\boldsymbol{\chi}_{i,0}^{(K)}$, $\boldsymbol{\alpha}_{i,0}^{(K)}$, intra-fragment compliances, and the $q^{(0)}$ charge distributions are all fixed after Phase 1. Phases 2–3 train only the ACE decoration weights, the correction heads $\text{MLP}_{\Delta E}, \text{MLP}_{\Delta\chi}, \text{MLP}_{\Delta\alpha}$, the inter-fragment compliance head $\text{MLP}_s$, and the gate. Light monomer replay in Phase 2 remains useful purely for conditioning the correction heads near the $\Delta\mathbf{h} \to 0$ boundary (where gradients are small); it carries no correctness burden.

### 2.4 Split-Charge Equilibration: Structural Charge Conservation & Learned Charge Transfer

Conventional EEM/QEq minimizes a globally quadratic $E(\{q_i\})$ subject to total-charge constraints enforced by Lagrange multipliers. Two pathologies follow: (i) fractional charge flows between arbitrarily distant fragments at dissociation, and (ii) blending diabats with *different fragment partitions* changes the number of constraints discretely, producing force discontinuities at validity-envelope boundaries. Split-Charge Equilibration (SQE) removes both by changing the coordinate system of the solve.

#### 2.4.1 Split-Charge Parameterization
Charge exists only as antisymmetric transfers along the edges of an explicit **channel graph** $\mathcal{G}$:
$$q_i = q_i^{(0)} + \sum_{j \in \mathcal{N}(i)} p_{ij}, \qquad p_{ij} = -p_{ji}$$
Every transfer $p_{ij}$ adds to atom $i$ exactly what it removes from atom $j$; therefore **the total charge of every connected component of $\mathcal{G}$ is conserved identically, for arbitrary $\{p_{ij}\}$**. There are no constraints and no Lagrange multipliers—the minimization over $\{p_{ij}\}$ is unconstrained. The former role of blended fragment-charge constraints is now played jointly by (i) the baseline charges $q^{(0)}$, (ii) the switched channel topology, and (iii) the learned compliances: the *distribution* of charge among fragments is a learned outcome of the solve rather than an imposed constraint value.

**Baseline charges.** $q_i^{(0)}$ are constructed on the **finest common refinement** of all active fragment partitions, by distributing each finest-level fragment's blended formal charge $\sum_{K_R} c_{K_R} Q_a^{(K_R)}$ over its atoms with a fixed per-fragment-type reference distribution (trained in Phase 1). Diabats that merge fragments do not define sub-fragment charges; any convention adopted for the baseline is immaterial because the open channel between the sub-fragments absorbs the difference (see 2.4.3), and the convention is irrelevant when the channel is closed.

#### 2.4.2 The SQE Energy Functional
Substituting $q(p)$ into the polarizable electrostatic energy and adding per-channel hardness penalties:
$$E_{\text{LR}}\left(\{p\}\right) = \sum_i \boldsymbol{\chi}_i^{\text{final}} \cdot \mathbf{q}_i(p) + \frac{1}{2}\sum_{i,j} \mathbf{q}_i(p)\, \mathbf{J}_{ij}\, \mathbf{q}_j(p) + \frac{1}{2}\sum_{(ij) \in \mathcal{G}} \kappa_{ij}\, p_{ij}^2 + E_{\text{disp}}$$
Here $\mathbf{q}_i$ collects the atomic charge (and, where used, higher on-site multipoles/induced dipoles governed by $\boldsymbol{\alpha}_{i,\text{eff}}^{\text{final}}$), $\mathbf{J}_{ij}$ contains Tang–Toennies-damped multipole interaction operators off-diagonal and effective hardness on-diagonal, $\kappa_{ij} = 1/s_{ij}$ is the channel hardness, and $E_{\text{disp}}$ is the damped dispersion model with fragment-state-conditioned coefficients. The functional remains quadratic in $\{p_{ij}\}$: the solve is a single sparse linear system per configuration, exactly as before, merely in transfer coordinates. SQE interpolates continuously between EEM ($\kappa \to 0$, "metallic") and fixed baseline charges ($\kappa \to \infty$, "insulating") on a per-channel basis, and additionally corrects EEM's well-known superlinear scaling of polarizability with system size.

#### 2.4.3 Channel Graph Construction, Switching & Ambient Charge Transfer
The channel graph is assembled from two edge classes:

- **Intra-fragment channels:** the covalent bond graph within each finest-partition fragment. Their compliances $s_{ij} = s_{ij}^{\text{raw}}$ are generally large (soft), granting near-full EEM flexibility inside a fragment.
- **Inter-fragment channels:** one channel for *every* pair of finest-partition fragments in short-range contact, between designated contact atoms (nearest heavy-atom pair; for polyatomic fragments a softmin-weighted multi-channel construction is the robust choice). The channel topology is purely distance-based—channels are not conditional on any diabat. The topology is a subgraph of the short-range neighbor list, so no new bookkeeping is introduced.

Every channel compliance is multiplied by an explicit $C^2$ pairwise switching function anchored at the short-range cutoff:
$$s_{ij}^{\text{eff}} = s_{ij}^{\text{raw}}\!\left(\mathbf{h}_i^{\text{final}}, \mathbf{h}_j^{\text{final}}, e(r_{ij})\right) \cdot S(r_{ij}), \qquad S \in C^2,\ S(r) = 1 \ (r \le r_{\text{on}}),\ S(r) = 0\ (r \ge r_{\text{SR}})$$
Compliance is a rapidly decaying pairwise quantity (charge-transfer matrix elements fall off roughly exponentially with overlap), so hard truncation via $S$ costs essentially nothing physically while making channel removal *exact and structural*: beyond $r_{\text{SR}}$ the channel does not exist, no learned suppression or numerical-threshold pruning is required, and asymptotic integer charges are guaranteed by topology.

Within the cutoff, the learned $s^{\text{raw}}$ produces two physical regimes from a single head:

1. **Ambient many-body charge transfer.** Between ordinary ground-state fragments (e.g., two hydrogen-bonded waters), $s^{\text{raw}}$ is small but nonzero. Small amounts of charge flow along the contact network and couple automatically into the global polarization solve—reproducing cooperative, non-additive many-body effects (dipole enhancement along hydrogen-bond chains, donor–acceptor asymmetry) that pure induction models miss. This is a deliberate design feature, not leakage: the flow is short-ranged by construction and its magnitude is trained against EDA charge-transfer energies (Section 4.2).
2. **Diabat-driven bond formation.** For a fragment pair merged by an active diabat, the blended features $\mathbf{h}^{\text{final}}$ shift toward the merged state as $c_{K_R}$ grows, and the compliance head—whose inputs are those blended features—softens the channel accordingly, enabling the large transfers of genuine bond formation. No explicit $\Omega$-gate on the compliance is needed: when the merging diabat is pruned ($\Omega_{K_R} \to 0$), the blended features relax smoothly to the unmerged states and the channel relaxes continuously to its ambient value rather than disappearing. Diabat pruning therefore never changes the structure of the solve at all—only the (smooth) feature inputs to $s^{\text{raw}}$.

#### 2.4.4 Continuity, Conditioning & Loop Redundancy
At the minimizer, each transfer scales with its compliance: for fixed electrostatic driving force $f_{ij}$, $p_{ij}^{\ast} = \mathcal{O}(s_{ij}^{\text{eff}} f_{ij})$, and the channel's total energy contribution is likewise $\mathcal{O}(s_{ij}^{\text{eff}})$. Consequently:

1. **Smooth closure at the cutoff.** As $r_{ij} \to r_{\text{SR}}$, $S(r_{ij}) \to 0$ with $C^2$ smoothness, so $p^{\ast} \to 0$ and the channel's energy and force contributions vanish continuously before the channel is dropped from the neighbor list. Channel removal is exact—zero energy and force error at the boundary by construction.
2. **Smooth behavior under diabat pruning.** Because channel existence is distance-based and compliance depends on diabats only through the smoothly blended features, pruning a state from the active space changes no solve structure whatsoever; $s^{\text{raw}}$ varies $C^2$-continuously with $\Omega$ through the blending weights. This replaces the earlier constraint-blending scheme, which could not avoid a discrete change in constraint count when a merged partition split.
3. **Conditioning.** The switching function bounds the dynamic range of active compliances: channels near the cutoff carry small $s^{\text{eff}}$ but equally small $p^{\ast}$, and the corresponding stiff rows can be Schur-eliminated or diagonally preconditioned without approximation concerns. No dead-zone regularization of the compliance head is required—the switch, not a learned value, is responsible for closure.
4. **Loop redundancy.** On cyclic channel topologies (ubiquitous now that the contact network is dense—e.g., water rings), circulating transfers change no $q_i$; the electrostatic part of $E$ is flat along these loop directions, but the $\kappa p^2$ terms regularize them, so the solve remains well-posed. The $\{p_{ij}\}$ are then not unique—only the charges are—and a spanning tree may be used when unique transfer coordinates are desired (e.g., for diagnostics). Merged multi-fragment diabats (e.g., an Eigen cation $\mathrm{H}_9\mathrm{O}_4^+$) require no special handling: the ambient contact channels among the constituent fragments already exist and simply soften as the merged state gains weight.

---

## 3. Implementation Guide: Step-by-Step Computational Workflow

```
[Step 1] Neighbor List & Coordination Defect Identification
   │     • Compute continuous coordination numbers n_i for reactive atoms.
   │     • Flag localized reactive centers where |n_i - n_val| > threshold.
   │     • Merge centers with overlapping k-hop graphs or shared candidate diabats.
   ▼
[Step 2] Local Graph Enumeration (k-Hop Active Space, per Center)
   │     • Enumerate chemically valid diabatic partitionings {K_R} per center.
   │     • Filter local states against global spin/charge bookkeeping constraints.
   │     • Compute geometric order parameters r_{K_R} and local Coulomb estimates Ẽ^(K_R)(r).
   ▼
[Step 3] Validity Envelope Evaluation & Pruning
   │     • Calculate polynomial bump envelopes Ω_{K_R}(r_{K_R}) for all candidate states.
   │     • Prune states where Ω_{K_R} < ε (guaranteed zero force/energy noise).
   ▼
[Step 4] Frozen Monomer Features, ACE Correction & Local Attention Gating
   │     • For active K_R, evaluate frozen monomer features h_{i,0}^{(K_R)}
   │       (frozen intra-fragment network on reference embeddings).
   │     • Build the state-independent ACE density basis once; contract per diabat
   │       with state-decoration weights to get corrections Δh_i^{(K_R)}
   │       (Δh ≡ 0 identically for isolated fragments).
   │     • Per center: predict logits z_{K_R} (incl. Coulomb bias -βẼ) and weights c_{K_R}.
   ▼
[Step 5] Latent Feature Blending & Property Prediction
   │     • Blend baseline and correction latents separately:
   │       h_{i,0}^final = ∑ c_{K_R} h_{i,0}^{(K_R)},  Δh_i^final = ∑ c_{K_R} Δh_i^{(K_R)}.
   │     • Evaluate frozen 1-body baselines ∑ c_{K_R} E_1^{(K_R)}, ∑ c_{K_R} χ_{i,0}^{(K_R)},
   │       ∑ c_{K_R} α_{i,0}^{(K_R)}; add subtraction-trick corrections ΔE, Δχ, Δα
   │       (exactly zero at isolation for any weights).
   │     • Assemble channel graph: intra-fragment bonds + one inter-fragment channel
   │       for every fragment pair in short-range contact (distance-based; no diabat
   │       condition).
   │     • Predict raw compliances s_ij^raw; apply switch: s_ij^eff = s_ij^raw · S(r_ij).
   │     • Build baseline charges q_i^(0) on the finest common partition from
   │       c-blended formal fragment charges.
   ▼
[Step 6] Single Global SQE Solve & Force Evaluation
   │     • Channels with S(r_ij) = 0 are simply absent (exact, structural removal);
   │       Schur-eliminate or precondition stiff near-cutoff rows for conditioning.
   │     • Solve the sparse quadratic problem over {p_ij} once (unconstrained —
   │       charge conservation is structural).
   │     • Recover q_i = q_i^(0) + ∑_j p_ij; evaluate E_LR^final incl. damped dispersion.
   │     • Aggregate E_total = E_LR^final + ∑ E_{i,SR}.
   │     • Backpropagate exact analytical spatial gradients F_i = -∇_i E_total
   │       (differentiating through the linear solve via the adjoint system).
   ▼
```

---

## 4. Experimental Testing Protocol: Ion-Pair Dissociation

To rigorously exercise the model's ability to distinguish ionic, radical, and covalently bound states—and to exercise the **overlapping-diabat and learned charge-transfer machinery**—we deploy a testing suite focusing on classic ion-pair dissociation curves.

### 4.1 Test Case Selection: Why Ion Pairs?
In gas-phase and solvent-separated ion pairs, the potential energy surface exhibits dramatic electronic transitions that serve as an ideal stress test for range-separated models. Crucially, each test case now includes a **merged-fragment (bound-pair) diabat** whose partition overlaps the dissociated partitions, so that the SQE channel machinery is exercised even in the simplest system:

| Test Case | Diabat 1 (Ionic) | Diabat 2 (Radical) | Diabat 3 (Merged Pair) | Physical Phenomena Tested |
| :--- | :--- | :--- | :--- | :--- |
| **Alkali Halide ($\mathrm{NaCl}$)** | $\lvert \mathrm{Na}^+ + \mathrm{Cl}^- \rangle$ ($Q=[+1,-1]$, $S=[0,0]$) | $\lvert \mathrm{Na}^\bullet + \mathrm{Cl}^\bullet \rangle$ ($Q=[0,0]$, $S=[\tfrac{1}{2},\tfrac{1}{2}]$) | $\lvert \mathrm{NaCl} \rangle$: one fragment, $Q_{\text{tot}}=0$, $S=0$ | **Harpoon Mechanism** (ionic/radical crossing at $r_c \approx 9.5\ \text{Å}$) **+ Learned CT:** partial covalency at equilibrium via the soft Na–Cl channel; smooth switch-off of the channel on dissociation; exact integer asymptotic charges. |
| **Organic Molecular Pair** | $\lvert \mathrm{CH}_3^+ + \mathrm{Cl}^- \rangle$ | $\lvert \mathrm{CH}_3^\bullet + \mathrm{Cl}^\bullet \rangle$ | $\lvert \mathrm{CH}_3\mathrm{Cl} \rangle$: one fragment | **Multi-Center Polarization** of polyatomic fragments + channel between polyatomic sub-fragments (contact-atom selection, intra-fragment redistribution of transferred charge). |
| **Proton Transfer Pair** | $\lvert \mathrm{H}_3\mathrm{O}^+ + \mathrm{OH}^- \rangle$ | $\lvert 2\,\mathrm{H}_2\mathrm{O} \rangle$ (Neutral) | $\lvert \mathrm{H}_5\mathrm{O}_2^+ \cdot \mathrm{OH}^- \rangle$-type merged states | **Solvation, Asymmetry & Hierarchical Overlap:** diabats with different fragment *counts*; finest-common-refinement baselines; recursive channel opening (Zundel/Eigen hierarchy) relevant to Grotthuss transport. |

The Na–Cl channel exists whenever the pair is within the switching range and closes smoothly as $S(r) \to 0$ at $r_{\text{SR}} \approx 4.5$–$5\ \text{Å}$. Within that range, the channel compliance transitions between two learned regimes: soft while the merged diabat $\lvert \mathrm{NaCl} \rangle$ carries weight (large covalent back-transfer) and small-but-ambient once only the ionic diabat remains active (residual contact CT). The merged diabat's validity envelope $\Omega_3(r)$ likewise has compact short-range support, and its pruning changes only the smooth feature inputs to the compliance head—never the structure of the solve.

### 4.2 Training Data Generation & EDA Decomposition
Training data must be generated using high-level electronic structure methods coupled with Energy Decomposition Analysis. **A single diabatization convention must be fixed across all phases** (e.g., ALMO-based fragment states throughout, or constrained DFT throughout); mixing conventions bakes inconsistencies into precisely the 3–6 Å hand-off window between the short-range network and the long-range solve.

1. **Scan Generation:** 1D and 2D potential energy surface scans across inter-fragment separations $r \in [1.5\ \text{Å}, 15.0\ \text{Å}]$, explicitly including distorted monomer geometries.
2. **Diabatic Reference Curves:** Constrained DFT or CASSCF/NEVPT2 energies forcing specific charge and spin distributions to isolate diabatic reference curves $E_{\text{QM}}^{(K)}(r)$; the merged-pair diabat is referenced to the adiabatic ground state in the strongly bound region where it dominates.
3. **EDA Partitioning:** Decompose QM interaction energies into long-range and short-range targets:
   $$E_{\text{target,LR}} = E_{\text{elstat}} + E_{\text{pol}} + E_{\text{disp}} + E_{\text{ct}}, \qquad E_{\text{target,SR}} = E_{\text{1-body}} + E_{\text{Pauli}} + E_{\text{pen}}$$
   The EDA polarization/CT split (well-defined in ALMO-EDA, where polarization is intra-fragment relaxation by construction) maps directly onto the channel structure, and **the fitting must respect that map**: when fitting $E_{\text{pol}}$, all inter-fragment compliances are clamped to zero, so polarization is carried entirely by $\boldsymbol{\alpha}_{\text{eff}}$ and intra-fragment channels; $E_{\text{ct}}$ is then fit with inter-fragment channels released, training the ambient compliances against the CT energy and the QM inter-fragment charge flow (e.g., ALMO CT charge). This assignment prevents double counting between induction and charge transfer and gives each channel class an unambiguous physical target. Because inter-fragment CT couples back into the multipole distribution, the released-channel fit is run with the (already-fit) polarization parameters frozen.
4. **Charge/Multipole References:** Fragment-resolved charges (e.g., from the chosen diabatization's population analysis) and molecular dipoles along each scan, used to train baselines, $\boldsymbol{\chi}$, and channel compliances jointly.

### 4.3 Step-by-Step Training Strategy

```
[Phase 1: Monomer Reference Training (the Frozen Stack)]
  └─► Train the intra-fragment featurization F_mono, 1-body energy heads E_1^(K)
      (over distorted monomer geometries), baseline electronegativities χ_{i,0}^(K),
      static polarizabilities α_{i,0}^(K), intra-fragment channel compliances, and
      per-fragment-type baseline charge distributions (for q^(0) construction)
      on isolated gas-phase fragments.
  └─► Target: monomer energies AND rich auxiliary targets — multipoles, static
      response tensors, formation energies — so the frozen features expose the
      directions downstream correction layers will need.
  └─► FREEZE the entire monomer stack permanently. All asymptotic guarantees
      (dissociation limits, IE−EA differences, Harpoon crossing position, gate/energy
      consistency of E_monomer^(K)) now rest on this validated, immutable component.

[Phase 2: ACE Correction, Response & Channel Training (Fixed Diabats)]
  └─► Train ONLY: ACE state-decoration weights, correction heads MLP_ΔE, MLP_Δχ,
      MLP_Δα (subtraction-trick form — exactly zero at isolation for any weights),
      and the inter-fragment compliance head MLP_s, on isolated diabatic states
      (c_K ∈ {0,1}). Optionally a small linear adapter of the frozen features.
  └─► Two-stage channel fit respecting the EDA pol/CT split (§4.2): (a) fit E_pol with
      inter-fragment compliances clamped to zero (intra-fragment channels + α_eff only);
      (b) release inter-fragment channels and fit ambient compliances to E_ct and QM
      inter-fragment charge flow, with polarization parameters frozen.
  └─► Merged-pair diabats: inter-fragment (sub-fragment) compliances trained against
      the large CT/covalency of the bound state along dissociation scans.
  └─► No closure constraints or dead-zone regularization needed anywhere: the ACE
      cutoff makes Δh ≡ 0 at isolation, the subtraction trick makes ΔE, Δχ, Δα ≡ 0
      there, and the pairwise switch S(r_ij) shuts inter-fragment channels at r_SR —
      all by construction. Light monomer replay is retained purely to condition the
      correction heads near the Δh → 0 boundary; it carries no correctness burden.

[Phase 3: Latent Mixing & Avoided Crossing Training]
  └─► Activate all diabatic states and local gating. Train MLP_gate, Coulomb scaling β,
      and temperature τ across dissociation trajectories; fine-tune the ACE decoration
      weights and correction heads at mixed c (this is where non-linear resonance
      stabilization is learned). The monomer stack remains frozen — the subtraction
      trick guarantees monomer predictions are untouched by any of this fine-tuning.
  └─► Target: total adiabatic ground-state QM energy E_total, QM charges/dipoles μ(r),
      and analytical QM forces across the avoided crossing AND across the channel-closure
      window (r ≈ 3–5 Å for NaCl), where gating, blending, and SQE interact.
  └─► Include evaluations at perturbed mixing coefficients c + δc (energy consistency
      regularizer) so E({c}) is meaningful off the gate's output manifold, preserving
      the future self-consistent (MCSCF-like) extension.
```

### 4.4 Evaluation Metrics & Quantitative Verification

The model must be evaluated against eight strict pass/fail quantitative criteria:

#### 1. Asymptotic Tail & Multipole Verification (The $-1/r$ vs $-1/r^6$ Test)
Plot the predicted interaction energy $\Delta E(r)$ for $r \in [8.0\ \text{Å}, 15.0\ \text{Å}]$ on a log-log scale against analytical asymptotics.
* **Ionic Diabat Target:** convergence to $-q_1 q_2 / r$ with slope $-1.0$.
* **Radical Diabat Target:** convergence to $-C_6 / r^6$ with slope $-6.0$, confirming zero spurious Coulomb leakage. Under SQE this is a *structural* guarantee (no channel exists at these separations); the test verifies the implementation honors it.

#### 2. Avoided Crossing & Weight Tracking
For the $\mathrm{NaCl}$ dissociation curve, monitor the local mixing weights $c_{\text{ion}}(r)$, $c_{\text{rad}}(r)$, $c_{\text{mol}}(r)$:
* **Criterion:** Near equilibrium ($r \approx 2.3\ \text{Å}$), $c_{\text{mol}} + c_{\text{ion}}$ must exceed $0.98$ (bound region dominated by molecular + ionic character). For $5\ \text{Å} < r < 8\ \text{Å}$, $c_{\text{ion}} > 0.98$ ($\Omega_{\text{mol}}$ has closed). At $r > 10.0\ \text{Å}$, $c_{\text{rad}} > 0.98$.
* **The Crossing Point:** The inflection where $c_{\text{ion}}(r_c) = c_{\text{rad}}(r_c) = 0.5$ must match the theoretical Harpoon distance $r_c \approx \frac{e^2}{IE(\mathrm{Na}) - EA(\mathrm{Cl})} \approx 9.4\ \text{Å}$ within $\pm 0.2\ \text{Å}$.

#### 3. Charge Trace & Dipole Staircase (New — Learned CT Verification)
Track the SQE charge $q_{\mathrm{Na}}(r)$ and molecular dipole $\mu(r)$ along the full dissociation curve:
* **Equilibrium region ($r \approx 2.3\ \text{Å}$):** the soft Na–Cl channel must pull $q_{\mathrm{Na}}$ *below* the ionic baseline of $+1$ toward the QM partial charge (NaCl's experimental dipole of $\sim 9.0\ \mathrm{D}$ implies $q_{\mathrm{Na}} \approx +0.79$ at $r_e$). Pass: $\mu(r)$ within $3\%$ of the QM dipole curve for $r < 4\ \text{Å}$.
* **Intermediate plateau ($r_{\text{SR}}$–$9\ \text{Å}$):** channel switched off ($S = 0$), ionic diabat dominant ⇒ $q_{\mathrm{Na}} = +1.000$ **exactly** (structural, machine precision).
* **Post-crossing ($r > 10\ \text{Å}$):** $q_{\mathrm{Na}} = 0.000$ exactly.
* **Transfer decay:** $p_{\mathrm{NaCl}}(r) \to 0$ smoothly through the switching window $[r_{\text{on}}, r_{\text{SR}}]$, with the ambient-CT contribution visible as a small residual transfer just inside $r_{\text{on}}$ after $\Omega_{\text{mol}}$ has closed.

#### 4. Strict $C^1$ Force Continuity Verification (Extended)
Numerical differentiation checks across **both** classes of smooth-structure boundaries: (a) diabatic validity-envelope cutoffs $r_{\text{cut}}$ where states are pruned, and (b) the **channel switch-off radius** $r_{\text{SR}}$ where $S(r_{ij})$ reaches zero and the channel leaves the solve.
* **Protocol:** Sample trajectories stepping across each boundary with $\Delta r = 10^{-5}\ \text{Å}$.
* **Metric:** $\Delta_{\text{force}} = \max_i \frac{\left|\mathbf{F}_{i,\text{anal}} - \mathbf{F}_{i,\text{num}}\right|}{\left|\mathbf{F}_{i,\text{anal}}\right| + \epsilon}$
* **Pass Criterion:** $\Delta_{\text{force}} < 10^{-6}$ across all pruning and switch-off boundaries, confirming that envelope weighting (for states) and the $C^2$ pairwise switch (for channels) completely suppress topological force spikes.

#### 5. Baseline-Convention Invariance (New)
Because the merged-pair diabat does not define sub-fragment charges, the baseline $q^{(0)}$ under Diabat 3 involves a convention (Section 2.4.1). Recompute the full NaCl curve under at least two distinct baseline conventions (e.g., inherit split-partition formal charges vs. symmetric split).
* **Pass Criterion:** total energies agree to within the linear-solve tolerance wherever Diabat 3 carries weight (the convention-dependent part of $q^{(0)}$ scales with $c_{\text{mol}}$, and the correspondingly soft channel absorbs it), and *exactly* wherever $c_{\text{mol}} = 0$ (the ambiguity vanishes from the baseline itself). Monitor the residual convention sensitivity through the window where $c_{\text{mol}}$ is small but nonzero and the channel has stiffened toward its ambient value—this cross-term must stay below the force-continuity tolerance.

#### 6. Ambient Many-Body Charge Transfer (New)
Verify the ground-state inter-fragment channels on hydrogen-bonded water clusters, where cooperative charge transfer is a well-characterized many-body effect:
* **Dimer:** reproduce the EDA CT energy along the water-dimer O···O scan within $0.1\ \text{kcal/mol}$, and the donor→acceptor charge flow (a few m$e$) in sign and magnitude against the reference population analysis.
* **Cooperativity:** for linear water chains and cyclic clusters $(\mathrm{H}_2\mathrm{O})_{n}$, $n = 2$–$6$, reproduce the non-additive enhancement of per-molecule dipole moments and total CT with chain length—the signature that channel flow couples correctly into the polarization solve. Pass: cluster dipoles within $3\%$ of QM.
* **Locality:** confirm that per-molecule charges in a large cluster are unchanged (to solve tolerance) when a molecule beyond $r_{\text{SR}}$ of its contact network is displaced—ambient CT must remain strictly short-ranged through the switch.

#### 7. Monomer Preservation Audit (New)
After Phases 2–3 are complete, re-run the full Phase-1 monomer validation suite (energies, multipoles, response tensors for every fragment state in the library) through the *final* model with each fragment isolated.
* **Pass Criterion:** predictions identical to the frozen Phase-1 model to machine precision. This is guaranteed architecturally (Δh ≡ 0 at isolation; subtraction-trick heads exactly zero there); the audit verifies the implementation honors the guarantee, and in particular that dissociation limits and the gate's E_monomer^(K) bias remain synchronized with the energy head's actual asymptotes.


---

## 5. Summary & Roadmap

By elevating the MLIP from an atom-centered monolithic feature space to a **diabatic, range-separated latent mixing architecture with a frozen monomer stack, state-decorated ACE corrections, and split-charge equilibration**, we bridge the gap between machine-learned quantum accuracy and classical empirical valence bond physics. The featurization hierarchy—frozen state-aware monomer features, corrected by a body-ordered cluster expansion that vanishes identically at isolation, blended per reactive center—makes monomer preservation, long-range linearity in the mixing weights, and dissociation asymptotics architectural properties rather than trained behaviors.

Local, per-reactive-center attention gating makes state mixing size-extensive and $\mathcal{O}(N_c)$, while a zeroth-order Coulomb bias guarantees rigorous Harpoon crossing behavior at asymptotic separations. Replacing constraint-based EEM with SQE turns charge conservation into a property of the coordinate system: charge flows only along an explicit short-range channel network, with **learned compliances** supplying the charge-transfer physics—from small ambient flows between ground-state neighbors that capture cooperative many-body polarization, to soft channels enabling bond formation when a merging diabat gains weight—and an explicit **$C^2$ pairwise switching function** enforcing exact, structural channel shutdown at the short-range cutoff. Formal diabatic charges are demoted from constraints to baselines; what was previously imposed is now learned.

The extended ion-pair protocol—with the bound pair included as an explicit overlapping diabat—exercises the full machinery in the minimal system: partial covalency through a soft channel at equilibrium, exact integer charges on the intermediate ionic plateau once the switch closes, the Harpoon crossing to radicals, and structural absence of long-range CT, complemented by water-cluster tests of ambient many-body charge transfer. This validates the core asymptotic, electrostatic, and charge-transfer machinery, paving the way for condensed-phase multi-electron redox reactions and Grotthuss proton transport networks. A natural future extension replaces amortized gating with self-consistent minimization of $E(\{c\})$ over the mixing simplex (an MCSCF-like mode), for which the perturbed-coefficient training in Phase 3 keeps the energy landscape well-defined.