# Latent-Space Diabatic Adiabaticization for Range-Separated MLIPs
## Architecture, Cover Discovery, Feature Mixing, and Training Protocol
### Revision 3: Learned Diabatic Covers, Overlap-Controlled Adiabaticization, and a Single Final System Evaluation

---

## 1. Executive Summary

Standard machine-learned interatomic potentials describe a system with one continuous set of atom-centered features. This is efficient, but it obscures processes in which the physically relevant electronic description changes qualitatively: homolytic and heterolytic bond cleavage, proton transfer, redox reactions, ion-pair neutralization, and transitions among distinct charge- or spin-localized states.

The present architecture introduces a library of physically interpretable **reference diabatic fragment states**, but it does not evaluate a complete potential for every possible diabat and then mix the resulting energies or force-field parameters. Instead, the diabatic states are used as a basis for **learned adiabaticization**:

1. The nuclear system is covered by several candidate sets of chemically valid fragment states. Each complete cover defines a candidate diabat.
2. A cheap proxy model ranks the covers and identifies a small active diabatic space.
3. Each active diabat supplies atom-centered reference features and inexpensive environment-conditioned feature corrections.
4. A learned adiabaticization network combines the active diabatic feature sets into one final set of **adiabatic atomic features**.
5. A single shared inference network maps those adiabatic features to all short-range and long-range model parameters.
6. The short-range MLIP and the global electrostatic, polarization, dispersion, and split-charge model are each evaluated only once.

The central object is therefore not a mixture of energies, charges, polarizabilities, or other predicted observables. It is a learned map

$$
\boxed{
\{\text{active diabatic atomic representations}\}
\longrightarrow
\{\text{one adiabatic atomic representation}\}.
}
$$

The architecture is designed to satisfy an exact **pure-state vertex condition**: when a reference diabat is the only active state, every atom must be represented by that diabat's own feature vector. Nonlinear mixing is allowed only when multiple diabats are active, and the nonlinear correction is multiplied by a learned electronic-overlap envelope that vanishes when the fragments involved in the transition cease to overlap.

A formally rigorous extension would treat the diabatic coefficients as variational degrees of freedom and minimize the final energy over the coefficient simplex, analogous to an MCSCF calculation. That extension defines the theoretical interpretation of the model, but the present implementation uses a learned one-shot coefficient inference network and performs no inner coefficient optimization.

Energy decomposition analysis (EDA) is essential to the training strategy. Pure diabatic fragment models and the shared property-inference heads are first trained on constrained, well-defined fragment states using energies, forces, multipoles, polarization, charge-transfer, and response targets. Adiabatic training then teaches the mixing network how combinations of these pretrained reference representations deform into the physical adiabatic representation.

---

## 2. Conceptual Separation of Responsibilities

The architecture separates four problems that should not be conflated.

### 2.1 Diabatic state definition

A diabatic state is a discrete electronic bookkeeping choice. For a system of atoms, a diabat specifies:

$$
K = \left\{
\mathcal P_K,
\{G_a,Q_a,S_a,\Gamma_a\}_{a\in\mathcal P_K},
S_{\mathrm{tot}}
\right\},
$$

where:

- $\mathcal P_K$ is a partition of the atoms into fragments;
- $G_a$ is the internal bonding or valence graph of fragment $a$;
- $Q_a$ is its integer formal charge;
- $S_a$ is its local spin;
- $\Gamma_a$ optionally identifies additional electronic character, such as an orbital occupation or excitation class;
- $S_{\mathrm{tot}}$ specifies the allowed total spin sector.

For an elongated O--H bond in water, distinct valid diabats include

$$
\mathrm{H_2O},\qquad
\mathrm{OH}^{\bullet}+\mathrm{H}^{\bullet},\qquad
\mathrm{OH}^{-}+\mathrm{H}^{+}.
$$

The environment does not define these states. It changes their relative relevance and the adiabatic representation inferred from them.

### 2.2 Cover discovery

Given only atoms, coordinates, total charge, and total spin, the system must be covered by compatible fragment states. This is a structured combinatorial inference problem.

### 2.3 Active-space inference

The number of valid covers can be large. A cheap proxy model identifies a small set of low-energy or chemically adjacent covers that are worth representing explicitly.

### 2.4 Adiabatic energy evaluation

The active diabats are converted into one final atomic representation. Only this final representation is sent through the expensive short-range and long-range models.

The proxy model therefore answers

> Which diabatic covers should be represented?

It does not answer

> What is the final energy of the system?

---

## 3. Candidate Fragment States and Complete Diabatic Covers

### 3.1 Fragment-state library

The model maintains a library of reference fragment states. A fragment-state entry is

$$
f = (A_f,G_f,Q_f,S_f,\Gamma_f),
$$

where $A_f$ is a subset of atoms and the remaining labels define its electronic state.

For an O/H chemistry library, possible entries include

$$
\mathrm{H},\ \mathrm{H}^{+},\ \mathrm{H}^{-},\
\mathrm{O},\ \mathrm{O}^{-},\ \mathrm{O}^{+},\
\mathrm{OH},\ \mathrm{OH}^{-},\ \mathrm{OH}^{+},\
\mathrm{H_2O},\ \mathrm{H_2O}^{+},\ \mathrm{H_2O}^{-},\
\mathrm{H_3O}^{+},\ldots
$$

with the appropriate spin and graph labels.

The library is not simply a list of equilibrium molecules. Each state is trained over distorted internal geometries and must remain evaluable throughout the geometric region in which it can participate in a reaction.

### 3.2 Fragment-state hypergraph

Each eligible fragment state is represented as a hyperedge spanning its atoms. Eligibility is controlled by permissive, smooth geometric criteria. The criteria should reject chemically impossible assignments, but should not prematurely remove states near bond-breaking or bond-forming regions.

A complete diabatic cover $K$ is a set of nonoverlapping fragment-state hyperedges satisfying

$$
\bigcup_{f\in K} A_f = \{1,\ldots,N\},
\qquad
A_f\cap A_g = \varnothing,
$$

along with global electronic constraints

$$
\sum_{f\in K}Q_f = Q_{\mathrm{tot}}
$$

and total-spin compatibility.

A box of O and H atoms is therefore not assigned one topology before the model begins. The cover generator proposes several complete, electronically valid interpretations.

### 3.3 Locality and factorization

In a large system, covers should be generated locally around ambiguous bonding, charge, or coordination regions. Provisional reactive centers are merged whenever:

- their candidate fragment assignments overlap;
- a candidate diabat spans both centers;
- they participate in a low-gap proton- or electron-transfer proposal.

Atoms outside reactive centers retain a single unambiguous reference assignment. This avoids construction of the full Cartesian product of independent local state spaces.

---

## 4. Proxy Model for Cover Discovery

### 4.1 Purpose

The proxy model should be substantially cheaper than the final potential. It needs high recall, not spectroscopic accuracy. Its purpose is to avoid discarding a physically relevant cover before the learned adiabaticization model sees it.

### 4.2 Proxy cover energy

For a candidate cover $K$, define

$$
\widetilde E_K
=
\sum_{a\in K}\widetilde E_{a}^{\mathrm{mono}}
+
\frac{1}{2}
\sum_{\substack{a\neq b\\R_{ab}<r_{\mathrm{proxy}}}}
\widetilde V_{ab}^{(K)}
+
\widetilde E_{\mathrm{LR}}^{(K)}.
$$

The terms are:

1. **Fragment self-energy** $\widetilde E_a^{\mathrm{mono}}$: a cheap approximation to the distorted isolated fragment-state energy.
2. **Short-range fragment-pair interaction** $\widetilde V_{ab}^{(K)}$: predicted by a small dimer MLIP, pair network, or inexpensive force-field representation.
3. **Cheap long-range state discrimination** $\widetilde E_{\mathrm{LR}}^{(K)}$: at minimum, formal-charge electrostatics and possibly fixed reference multipoles and polarizabilities.

The long-range term is required because a short-range dimer score cannot recognize state crossings whose dominant distinction is Coulombic, such as ionic and neutral-radical asymptotes.

### 4.3 Incremental evaluation

Candidate covers usually differ only within a small reactive region. The proxy score should therefore be evaluated incrementally:

- cache contributions from unchanged fragments;
- recompute only fragment self-energies and pair terms affected by a local cover edit;
- share geometric neighbor lists and radial bases across candidate covers.

This permits beam search, $k$-best set packing, or factor-graph inference without a full evaluation for every combinatorial partition.

### 4.4 Active cover set

A preliminary active set is

$$
\mathcal A_0
=
\left\{
K:\widetilde E_K-\widetilde E_{\min}<\Delta E_{\mathrm{screen}}
\right\}.
$$

To avoid missing an approaching crossing, augment this set with chemically adjacent states:

$$
\mathcal A
=
\mathcal A_0
\cup
\left\{
J:d_{\mathrm{edit}}(J,K)\leq 1
\text{ for some }K\in\mathcal A_0
\right\}.
$$

The edit graph includes elementary operations such as:

- homolytic bond cleavage or formation;
- heterolytic bond cleavage or formation;
- proton transfer;
- one-electron transfer;
- local spin recoupling;
- fragment merge or split.

Each candidate has a smooth validity factor $\Omega_K\in[0,1]$. A state enters or exits the active set through a compact-support envelope rather than a discontinuous top-$k$ change.

---

## 5. Reference Diabatic Feature Construction

### 5.1 Atomic reference embeddings

Each atom begins with an embedding conditioned on its element and its role in the reference fragment state:

$$
\mathbf e_i^{(K)}
=
\mathcal E\left(
Z_i,
G_a,
Q_a,
S_a,
\Gamma_a,
\mathrm{role}_i
\right),
\qquad i\in a.
$$

Bare element embeddings alone are insufficient because they do not distinguish oxygen in $\mathrm{H_2O}$, $\mathrm{OH}^{\bullet}$, and $\mathrm{OH}^{-}$.

### 5.2 Frozen fragment-state encoder

A fragment encoder produces reference atomic features

$$
\mathbf h_{i,0}^{(K)}
=
\mathcal F_{\mathrm{frag}}
\left(
\{\mathbf e_j^{(K)},\mathbf R_j\}_{j\in a}
\right).
$$

The fragment encoder is trained on isolated, distorted diabatic fragment states and then frozen or strongly regularized during later phases. It supplies a stable electronic vocabulary for the adiabaticization network.

### 5.3 Inexpensive environment-conditioned diabatic features

Each active diabat receives a short-range environment correction

$$
\mathbf h_i^{(K)}
=
\mathbf h_{i,0}^{(K)}
+
\Delta\mathbf h_{i,\mathrm{env}}^{(K)}.
$$

A state-decorated ACE or shallow equivariant neighborhood contraction is suitable:

$$
\Delta\mathbf h_{i,\mathrm{env}}^{(K)}
=
\operatorname{ACE}_{\nu}
\left(
\{\mathbf h_{j,0}^{(K)},\mathbf r_{ij}\}_{r_{ij}<r_{\mathrm{SR}}}
\right).
$$

The expensive geometric basis is built once. Each active cover changes only the state-decoration contractions. This is not a full system energy evaluation for each diabat.

The correction must vanish when the relevant neighboring fragments leave the short-range region:

$$
\Delta\mathbf h_{i,\mathrm{env}}^{(K)}\equiv 0
\quad\text{at fragment isolation}.
$$

### 5.4 Common mixing space

Before mixing, every diabatic feature is mapped into a shared latent space:

$$
\widehat{\mathbf h}_i^{(K)}
=
\mathcal P\left(\mathbf h_i^{(K)}\right).
$$

The same map $\mathcal P$ is used for every state. It may be identity after pretraining, or a small equivariant adapter. The common space is necessary so that feature differences and similarities across diabats are meaningful.

---

## 6. Learned Coefficients as Amortized Active-Space Inference

### 6.1 Local cover logits

For a reactive center $R$, pool each active cover into a state representation

$$
\mathbf g_K^{(R)}
=
\operatorname{Pool}_{i\in R}
\widehat{\mathbf h}_i^{(K)}.
$$

The coefficient network receives:

- the proxy cover score $\widetilde E_K$;
- the pooled state feature $\mathbf g_K^{(R)}$;
- validity information $\Omega_K$;
- geometric reaction descriptors;
- cheap electrostatic potentials and fields;
- formal charge and spin labels;
- edit-graph information.

The logits are

$$
z_K
=
\mathcal G
\left(
\mathbf g_K^{(R)},
\widetilde E_K,
\boldsymbol\xi_K
\right),
$$

and the current one-shot coefficients are

$$
c_K
=
\frac{
\Omega_K\exp(z_K/\tau)
}{
\sum_{J\in\mathcal A_R}
\Omega_J\exp(z_J/\tau)
}.
$$

The $c_K$ are latent inference coordinates, not literal state populations.

### 6.2 Long-range awareness

Most of the learned adiabaticization may be local, but state selection cannot always be purely short ranged. The gate should therefore receive inexpensive state-specific long-range descriptors such as

$$
\widetilde E_{K}^{\mathrm{Coul}}
=
\sum_{a<b}
\frac{Q_a^{(K)}Q_b^{(K)}}{R_{ab}},
$$

reference multipole interactions, and approximate donor--acceptor redox gaps. This allows long-range ionic/radical crossings to enter the active space even when no local coordination defect is present.

---

## 7. The Core Adiabaticization Map

This is the central component of the architecture.

### 7.1 Weighted diabatic mean

For each atom, compute the coefficient-weighted mean in the common mixing space:

$$
\boldsymbol\mu_i
=
\sum_{K\in\mathcal A_R}
 c_K\widehat{\mathbf h}_i^{(K)}.
$$

This term gives the correct pure-state limit automatically: if $c_L=1$, then

$$
\boldsymbol\mu_i=\widehat{\mathbf h}_i^{(L)}.
$$

The mean alone is not sufficient. It can discard information about how different the competing states are and cannot, by itself, represent nonlinear resonance stabilization.

### 7.2 Feature spread and state disagreement

Compute a permutation-invariant measure of the spread among active diabatic representations:

$$
\mathbf v_i
=
\sum_K c_K
\left(
\widehat{\mathbf h}_i^{(K)}-\boldsymbol\mu_i
\right)^{\odot 2}.
$$

For equivariant tensor channels, use invariant contractions of state differences to generate scalar spread descriptors while retaining equivariant state features for the correction network.

The spread distinguishes:

- a nearly pure state;
- a mixture of two very similar states;
- a mixture of electronically distinct ionic and radical states.

### 7.3 Pairwise nonlinear mixing correction

The adiabatic atomic feature is

$$
\boxed{
\mathbf z_i^{\mathrm{ad}}
=
\boldsymbol\mu_i
+
\Delta\mathbf z_{i}^{\mathrm{mix}}.
}
$$

The leading nonlinear correction is constructed pairwise over connected states in the diabat edit graph:

$$
\Delta\mathbf z_i^{\mathrm{mix}}
=
\sum_{K<J}
4c_Kc_J\,
\mathcal O_{KJ}\,
\boldsymbol\Psi_i^{KJ}.
$$

Here:

- $4c_Kc_J$ is zero at every pure-state vertex and maximal for an equal two-state mixture;
- $\mathcal O_{KJ}\in[0,1]$ is an electronic-overlap envelope;
- $\boldsymbol\Psi_i^{KJ}$ is a learned, symmetric equivariant correction.

A suitable symmetric input representation is

$$
\boldsymbol\Psi_i^{KJ}
=
\Psi\left(
\widehat{\mathbf h}_i^{(K)}+\widehat{\mathbf h}_i^{(J)},
\left(\widehat{\mathbf h}_i^{(K)}-\widehat{\mathbf h}_i^{(J)}\right)^{\odot 2},
\boldsymbol\mu_i,
\mathbf v_i,
\mathbf z_R,
\mathbf e_{KJ}
\right),
$$

where $\mathbf e_{KJ}$ identifies the chemical edit connecting the states and $\mathbf z_R$ is a shared center-level latent.

The pairwise form is analogous to retaining leading pair couplings among configuration-state functions. Higher-order corrections can be introduced later:

$$
\sum_{K<J<L}
c_Kc_Jc_L\,
\mathcal O_{KJL}\,
\Psi^{KJL},
$$

but they are not required in the first implementation.

### 7.4 Why the correction must vanish at pure-state vertices

If only state $L$ is active,

$$
c_L=1,\qquad c_{K\neq L}=0,
$$

then every pair factor $c_Kc_J$ is zero. Therefore,

$$
\boxed{
\mathbf z_i^{\mathrm{ad}}
=
\widehat{\mathbf h}_i^{(L)}
}
$$

exactly, for arbitrary network weights.

This is stronger than a training penalty. It ensures that the shared decoder sees exactly the pretrained diabatic feature whenever the active space has a single state.

### 7.5 The overlap envelope is not just a feature dot product

A normalized feature dot product is useful:

$$
\kappa_{KJ}
=
\frac{
\left\langle
W\mathbf g_K,
W\mathbf g_J
\right\rangle
}{
\|W\mathbf g_K\|\,\|W\mathbf g_J\|+\epsilon
}.
$$

It measures compatibility in the learned common space. It should not, however, be the sole definition of electronic overlap because:

1. learned latent vectors have gauge freedom and can be rescaled or rotated;
2. two separated states can have similar fragment features even when their electronic coupling is negligible;
3. two strongly coupled states can have dissimilar features precisely because their charge or spin assignments differ;
4. a dot product has no structural guarantee of vanishing at fragment separation.

The overlap envelope should combine a guaranteed geometric support factor with learned feature compatibility:

$$
\boxed{
\mathcal O_{KJ}
=
S_{KJ}^{\mathrm{geo}}(\mathbf R)
\times
\sigma\left[
 f_{\mathrm{ov}}
 \left(
 \kappa_{KJ},
 \|W\mathbf g_K-W\mathbf g_J\|^2,
 \mathbf d_{KJ},
 \Delta\widetilde E_{KJ},
 \mathbf e_{KJ}
 \right)
\right].
}
$$

The learned factor estimates whether the two state representations are capable of meaningful local mixing. The geometric factor enforces the asymptotic limit.

### 7.6 Geometric overlap support

Every edge $K\leftrightarrow J$ in the diabat edit graph has an associated set of transition contacts $\mathcal T_{KJ}$. Examples include:

- the bond being broken or formed;
- the donor--proton and proton--acceptor contacts in proton transfer;
- the donor--acceptor contact in short-range electron transfer;
- the fragment interface across which ionic and covalent states mix.

For each required contact $e$, define a $C^2$ switching function

$$
s_e(r)=
\begin{cases}
1, & r\leq r_{\mathrm{on}},\\
\text{$C^2$ transition}, & r_{\mathrm{on}}<r<r_{\mathrm{off}},\\
0, & r\geq r_{\mathrm{off}}.
\end{cases}
$$

For a transition requiring all contacts, use a smooth product:

$$
S_{KJ}^{\mathrm{geo}}
=
\prod_{e\in\mathcal T_{KJ}}s_e(r_e).
$$

For a transition that can proceed through any of several equivalent contacts, use a smooth OR:

$$
S_{KJ}^{\mathrm{geo}}
=
1-
\prod_{e\in\mathcal T_{KJ}}
\left[1-s_e(r_e)\right].
$$

More complex concerted edits can use learned soft-AND or soft-OR combinations, but the final result must retain the exact condition

$$
S_{KJ}^{\mathrm{geo}}=0
$$

when the relevant fragments no longer overlap.

### 7.7 Physical interpretation of the nonlinear correction

The weighted mean $\boldsymbol\mu_i$ represents the first-order interpolation among reference state descriptions. The correction $\Delta\mathbf z_i^{\mathrm{mix}}$ represents the electronic reorganization that is absent from any classical mixture of diabatic features:

- resonance stabilization;
- partial covalency;
- charge redistribution;
- spin-pairing effects represented within a fixed total-spin sector;
- nonlinear changes in polarizability and short-range repulsion;
- changes in charge-transfer compliance.

The correction is largest when:

- at least two coefficients are appreciable;
- the states differ meaningfully;
- the transition contacts have substantial geometric overlap;
- the learned compatibility network predicts strong mixing.

It vanishes when:

- the active space has only one state;
- one state dominates completely;
- the states are not connected by an allowed local edit;
- the transition fragments separate beyond the overlap range.

### 7.8 Center-level coherence

Atomic adiabaticization cannot be performed independently for each atom. A shared center representation is first formed:

$$
\mathbf z_R
=
\mathcal A_R
\left(
\{c_K,\mathbf g_K,\widetilde E_K,\boldsymbol\xi_K\}_{K\in\mathcal A_R}
\right).
$$

Every atomic correction $\boldsymbol\Psi_i^{KJ}$ receives $\mathbf z_R$. This ensures that all atoms in a reactive center make a consistent collective transition. For example, the oxygen and departing hydrogen cannot independently select incompatible homolytic and heterolytic characters.

### 7.9 Equivariance

The final feature should be an equivariant bundle

$$
\mathbf z_i^{\mathrm{ad}}
=
\left\{
\mathbf z_i^{(l=0)},
\mathbf z_i^{(l=1)},
\mathbf z_i^{(l=2)},\ldots
\right\}.
$$

Mixing occurs only among matching irreducible representations. Scalar coefficient and overlap factors can multiply equivariant corrections without breaking rotational covariance.

---

## 8. Pure-State, Isolation, and Asymptotic Conditions

### 8.1 Pure-state identity

For every reference diabat $K$, require

$$
\mathcal A
\left(
\{c_K=1,\widehat{\mathbf h}^{(K)}\}
\right)
=
\widehat{\mathbf h}^{(K)}.
$$

The pairwise residual construction enforces this exactly.

### 8.2 Pretrained decoder consistency

The shared decoder is pretrained so that each pure diabatic feature reproduces the corresponding reference model:

$$
\mathcal D
\left(
\widehat{\mathbf h}^{(K)}
\right)
\longrightarrow
\left
\{
E_{\mathrm{SR}}^{(K)},
\chi^{(K)},
\alpha^{(K)},
q_0^{(K)},
C_6^{(K)},
s^{(K)},\ldots
\right\}.
$$

When only one state is active, no downstream retraining is permitted to destroy this correspondence.

### 8.3 Nonlinear mixing must vanish with overlap

Because $\mathcal O_{KJ}$ contains compact geometric support,

$$
\Delta\mathbf z_i^{\mathrm{mix}}\rightarrow 0
$$

when the fragments involved in every competing state transition separate.

### 8.4 Asymptotic state sharpness

At long range, the nonlinear correction vanishes. The coefficient model must then approach the lowest relevant diagonal state rather than maintain an arbitrary soft hybrid. Training therefore includes:

- asymptotic coefficient targets or ranking losses;
- low-entropy regularization once all state couplings have vanished;
- exact formal-charge and dissociation-limit tests.

At an exact asymptotic degeneracy, the energy is invariant to the choice of diabatic combination. A deterministic tie-breaking convention may still be useful for stable observables.

---

## 9. One Shared Decoder and One Final Physical Evaluation

### 9.1 Shared electronic inference trunk

The final adiabatic feature is passed through one shared inference network:

$$
\mathbf u_i
=
\mathcal D_{\mathrm{shared}}
\left(
\mathbf z_i^{\mathrm{ad}}
\right).
$$

Small structured heads derive all final quantities from the same electronic latent:

$$
\begin{aligned}
\boldsymbol\theta_i^{\mathrm{SR}} &= D_{\mathrm{SR}}(\mathbf u_i),\\
\chi_i &= D_{\chi}(\mathbf u_i),\\
\alpha_i &= D_{\alpha}(\mathbf u_i),\\
q_{i}^{(0)} &= D_q(\mathbf u_i),\\
C_{6,i} &= D_{C_6}(\mathbf u_i),\\
s_{ij}^{\mathrm{raw}} &= D_s(\mathbf u_i,\mathbf u_j,e(r_{ij})).
\end{aligned}
$$

This is not output-level mixing. Every quantity is inferred once from the final adiabatic representation.

The heads must enforce their mathematical constraints structurally:

- positive-definite polarizabilities;
- nonnegative compliances and dispersion coefficients;
- symmetric pair parameters;
- equivariant multipoles;
- correct total charge;
- permutation invariance.

### 9.2 Short-range MLIP

The final short-range energy is evaluated once:

$$
E_{\mathrm{SR}}
=
\mathcal M_{\mathrm{SR}}
\left(
\{\boldsymbol\theta_i^{\mathrm{SR}},\mathbf R_i\}
\right).
$$

The MLIP may use an ACE, equivariant message-passing network, or another local model. It sees only the inferred adiabatic features, not separate diabatic energies.

### 9.3 Long-range model and SQE

The final long-range parameters are inserted into one global electrostatic, polarization, dispersion, and split-charge solve.

The split-charge parameterization is

$$
q_i=q_i^{(0)}+\sum_j p_{ij},
\qquad p_{ij}=-p_{ji}.
$$

Interfragment channel compliances are switched off smoothly with distance:

$$
s_{ij}^{\mathrm{eff}}
=
s_{ij}^{\mathrm{raw}}S(r_{ij}).
$$

The global response coordinates are obtained once:

$$
\mathbf p^*
=
\arg\min_{\mathbf p}
E_{\mathrm{LR}}
\left(
\mathbf p;
\{\chi_i,\alpha_i,q_i^{(0)},s_{ij},C_{6,i},\ldots\},
\mathbf R
\right).
$$

The total energy is

$$
\boxed{
E_{\mathrm{total}}
=
E_{\mathrm{SR}}
\left(\{\mathbf z_i^{\mathrm{ad}}\}\right)
+
E_{\mathrm{LR}}
\left(\mathbf p^*;\{\mathbf z_i^{\mathrm{ad}}\}\right).
}
$$

There is no complete short-range evaluation, parameter inference, or SQE solve for each active diabat.

---

## 10. MCSCF Interpretation and Present Approximation

The final physical model defines an energy for any allowed coefficient vector:

$$
E(\mathbf c)
=
E_{\mathrm{SR}}
\left[
\mathcal A(\{\mathbf h^{(K)}\},\mathbf c)
\right]
+
E_{\mathrm{LR}}
\left[
\mathcal A(\{\mathbf h^{(K)}\},\mathbf c)
\right].
$$

A rigorous variational extension would solve

$$
\boxed{
\mathbf c^*
=
\arg\min_{\mathbf c\in\Delta}
E(\mathbf c),
\qquad
\Delta=\left\{\mathbf c:c_K\geq0,\ \sum_Kc_K=1\right\}.
}
$$

This is analogous to optimizing configuration coefficients in MCSCF while the learned feature maps play the role of an adaptive electronic representation.

The present model does not perform this minimization. It uses

$$
\mathbf c=\mathcal G(\text{proxy scores, diabatic features, environment})
$$

as an amortized estimate of the variational solution. The coefficient network is therefore an inference accelerator rather than a fundamental definition of the energy.

To preserve the future variational interpretation, training should evaluate the energy at perturbed coefficient vectors and penalize pathological off-manifold behavior. The model should learn a smooth, locally stable $E(\mathbf c)$ even though only the inferred $\mathbf c$ is used during production.

---

## 11. EDA-Based Training Strategy

### Phase 1: Fragment-state library and pure-diat decoder pretraining

Train the reference fragment encoder and shared decoder on isolated, distorted diabatic fragments.

Targets include:

- diabatic fragment energies and forces;
- permanent multipoles;
- electrostatic potentials;
- static and anisotropic polarizabilities;
- reference electronegativities or hardness information;
- intra-fragment charge response;
- dispersion coefficients;
- spin- and charge-state energy differences.

EDA and constrained electronic-structure calculations make these states well defined. The shared decoder learns to interpret each reference diabatic feature before any state mixing is introduced.

After this phase, enforce the audit

$$
\mathcal D\left(\widehat{\mathbf h}^{(K)}\right)
=\text{reference properties of }K.
$$

### Phase 2: Proxy cover and environment-conditioned feature training

Train:

- the fragment self-energy proxy;
- the fragment-dimer proxy interaction model;
- cheap long-range cover corrections;
- the state-decorated environment feature contraction.

The proxy objective emphasizes active-space recall. A physically important state ranked slightly too low is more damaging than several unnecessary extra candidates.

For fixed pure diabats, train the environment-conditioned features and shared decoder against EDA-resolved interaction data:

- frozen electrostatics;
- polarization;
- charge transfer;
- Pauli repulsion;
- penetration;
- dispersion;
- total interaction energy and forces.

### Phase 3: Adiabaticization and coefficient inference

Activate multiple covers. Train:

- the coefficient network $\mathcal G$;
- the center-level state-set encoder;
- the overlap model $f_{\mathrm{ov}}$;
- the nonlinear pair correction $\Psi$;
- limited fine-tuning of the shared decoder and short-range MLIP.

Targets include:

- adiabatic total energies and forces;
- molecular and fragment-resolved multipoles;
- charge redistribution;
- polarizability and response changes;
- EDA component changes across crossings;
- reference diabatic weights when a trustworthy diabatization provides them.

Pure-state examples remain in every batch. Because the vertex identity is architectural, these examples primarily prevent decoder drift and preserve calibration.

### Phase 4: End-to-end robustness and active-set perturbation

Train against changes in the candidate set:

- randomly remove irrelevant high-energy states;
- insert redundant or nearly duplicate covers;
- vary the proxy screening window;
- permute state ordering;
- include graph-edit neighbors before they become energetically competitive;
- perturb inferred coefficients and evaluate energy curvature.

The final prediction should be insensitive to irrelevant candidate states and stable as covers enter or leave through their validity envelopes.

---

## 12. Computational Workflow

```text
[Input]
  Coordinates, elements, total charge, total spin
      |
      v
[1. Candidate fragment-state generation]
  Build eligible fragment-state hyperedges from the library
      |
      v
[2. Compatible-cover search]
  Generate k-best complete covers under charge/spin constraints
      |
      v
[3. Cheap proxy scoring]
  Fragment self energies + short-range fragment-dimer terms
  + cheap long-range state discrimination
      |
      v
[4. Active-space construction]
  Retain low-score covers + one-edit neighbors
  Apply smooth validity envelopes; merge overlapping centers
      |
      v
[5. Active-diat feature construction]
  Frozen fragment features + cheap state-decorated environment contractions
  Shared geometric basis constructed once
      |
      v
[6. Coefficient inference]
  Local state-set gate with proxy, geometric, and long-range descriptors
      |
      v
[7. Learned adiabaticization]
  Weighted mean + overlap-controlled nonlinear pair corrections
  Exact pure-state vertex identity
      |
      v
[8. Shared parameter inference]
  One adiabatic latent per atom -> all final physical parameters
      |
      +---------------------------+
      |                           |
      v                           v
[9a. One short-range MLIP]   [9b. One global LR/SQE solve]
      |                           |
      +-------------+-------------+
                    v
              Total energy and forces
```

---

## 13. Evaluation and Pass/Fail Tests

### 13.1 Cover recall

For every reaction trajectory, verify that the reference-dominant diabats are present in the active set before their crossing region is reached.

Primary metric:

$$
\mathrm{Recall@active\ set}>99.9\%
$$

on chemically relevant states, even if precision is lower.

### 13.2 Pure-state vertex identity

For every library state, force it to be the only active diabat and verify

$$
\mathbf z_i^{\mathrm{ad}}
=
\widehat{\mathbf h}_i^{(K)}
$$

and that all predicted energies and properties match the pretrained pure-diat model to machine precision.

### 13.3 Overlap shutdown

Along dissociation coordinates, verify that

$$
\mathcal O_{KJ}\rightarrow0,
\qquad
\Delta\mathbf z_i^{\mathrm{mix}}\rightarrow0
$$

smoothly and exactly at the geometric support boundary.

Numerically test analytical force continuity across every overlap cutoff.

### 13.4 Active-set invariance

Add irrelevant states with small validity or high proxy energy. The final energy, forces, and observables should remain unchanged within tolerance.

### 13.5 Avoided crossings

For NaCl, water bond dissociation, and organic ion-pair systems, verify:

- correct diabatic asymptotes;
- smooth coefficient transfer;
- correct adiabatic energy lowering;
- no long-range nonlinear resonance after overlap vanishes;
- correct ionic versus neutral long-range tails.

### 13.6 EDA component reconstruction

Test not only total energies but also:

- frozen electrostatics;
- polarization;
- charge transfer;
- Pauli and penetration terms;
- dispersion;
- fragment charge and dipole changes.

The purpose is to verify that the inferred adiabatic latent is electronically meaningful rather than merely an energy-fitting variable.

### 13.7 Water-cluster charge transfer

Use water dimers and clusters to verify that the final adiabatic features produce:

- correct donor--acceptor charge flow;
- cooperative dipole enhancement;
- short-range closure of interfragment transfer channels;
- stable behavior as proton-transfer covers become active.

### 13.8 One-evaluation audit

Profile the implementation and verify that per-diat work is restricted to:

- fragment feature lookup/evaluation;
- state-decoration contractions;
- proxy and gating calculations;
- the small adiabaticization block.

There must be exactly one final short-range energy evaluation and one global long-range response solve.

---

## 14. Initial Test Systems

### 14.1 Water bond dissociation

Active states:

$$
\mathrm{H_2O},
\qquad
\mathrm{OH}^{\bullet}+\mathrm{H}^{\bullet}
\text{ in the desired total-spin sector},
\qquad
\mathrm{OH}^{-}+\mathrm{H}^{+}.
$$

This test probes:

- homolytic versus heterolytic competition;
- environmental stabilization of charge-separated states;
- pure-state identity;
- overlap decay on bond dissociation;
- nonlinear adiabatic feature formation.

### 14.2 NaCl harpoon crossing

Active states:

$$
\mathrm{Na}^{+}+\mathrm{Cl}^{-},
\qquad
\mathrm{Na}^{\bullet}+\mathrm{Cl}^{\bullet},
\qquad
\mathrm{NaCl}.
$$

This test probes:

- long-range cover ranking;
- ionic/radical crossing;
- short-range nonlinear covalency;
- exact disappearance of resonance at separation;
- integer asymptotic charges.

### 14.3 Proton transfer in water clusters

Covers differ by the identity of the protonated and deprotonated fragments. This test probes:

- overlapping fragment covers;
- center merging;
- collective atomic adiabaticization;
- coupling of the final latent to global SQE and polarization;
- multiple simultaneously plausible proton locations.

---

## 15. Summary and Roadmap

The revised architecture treats diabatic states as a physically meaningful reference basis and the final potential as a learned adiabatic inference model.

The complete logic is:

1. Construct chemically valid fragment-state covers of the nuclear system.
2. Rank them with a cheap monomer, fragment-dimer, and long-range proxy model.
3. Retain a conservative active set of low-score covers and nearby chemical edits.
4. Build inexpensive state-conditioned atomic features for each active diabat.
5. Infer local diabatic coefficients in one pass.
6. Form a coefficient-weighted mean feature and add an overlap-controlled nonlinear correction.
7. Guarantee that every pure-state vertex exactly reproduces its reference diabatic feature.
8. Decode the single adiabatic atomic representation into all physical model parameters.
9. Evaluate one short-range MLIP and one global long-range/SQE model.

The single most important design choice is the form

$$
\mathbf z_i^{\mathrm{ad}}
=
\sum_Kc_K\widehat{\mathbf h}_i^{(K)}
+
\sum_{K<J}
4c_Kc_J\mathcal O_{KJ}\boldsymbol\Psi_i^{KJ}.
$$

It combines four desirable properties:

- exact recovery of every reference diabat at a simplex vertex;
- nonlinear expressivity near crossings;
- explicit dependence on state disagreement and chemical edit type;
- exact decay of nonlinear mixing when electronic overlap disappears.

A future variational mode will minimize the final energy over the coefficient simplex, providing an MCSCF-like treatment. The current model instead learns that minimizer through amortized inference, preserving the computational goal of a single full system evaluation.
