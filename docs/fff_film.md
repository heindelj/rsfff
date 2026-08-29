# State-Conditioned, Fragment-Projected MLIP

## Implementation guide and staged validation on nonreactive water

## 1. Purpose

The aim is to build a single-pass, energy-conserving MLIP in which a neural
network generates the parameters of physically motivated force-field terms.
The model should retain the useful fragment interpretation supplied by EDA
data while avoiding separate, topology-specific feature extractors or
independent expert networks.

The central architectural idea is:

> Compute a shared set of geometric primitives, project those primitives into
> fragment-internal and fragment-environment channels using the conditioned
> state, and use a shared FiLM-conditioned parameter network to generate one
> effective set of force-field parameters.

This architecture will first be validated on nonreactive water. In that phase,
fragment membership is known and discrete. Nevertheless, all interfaces needed
for later reactivity should be implemented from the beginning:

- a general atom-to-fragment assignment matrix;
- a fragment co-membership projector;
- local fragment-state conditioning;
- continuous, differentiable assignments;
- an explicit topology-mixing descriptor;
- optional low-rank hypernetwork adapters;
- full differentiation through features, conditioning, generated parameters,
  and any force-field SCF procedure.

The reactive phase can then replace the supplied discrete state by a smooth
geometry-dependent state without restructuring the model.

## 2. Design principles

### 2.1 Share the feature language, not necessarily every parameter response

All states should use the same elemental embeddings, radial basis, angular
basis, tensor-product rules, and hidden feature space. State conditioning
should determine how these features are partitioned and interpreted.

This is stronger and more useful than training separate expert networks. It
forces water, hydronium, and shared-proton environments to express their
chemistry in a common latent basis, which makes interpolation meaningful.

### 2.2 Apply fragment projection before nonlinear feature contraction

A late mask applied to an already constructed invariant feature is generally
not sufficient. If nonlinear contractions have already combined atoms across a
fragment boundary, the isolated-fragment contribution cannot in general be
recovered afterward.

The fragment projector should therefore be applied to pair or neighbor-density
contributions before power-spectrum, bispectrum, or higher-order ACE
contractions are formed.

### 2.3 Preserve vertex behavior architecturally

At a discrete fragment assignment, the projected features should reduce
exactly to fragment-internal and interfragment features. Environmental and
topology-mixing corrections should be constructed to vanish exactly in the
limits where they are not applicable.

Vertex behavior should not depend solely on the optimizer learning an
approximately correct limit.

### 2.4 Treat the state as a structured partition

A global learned expert identifier is not enough to describe which atoms
belong to which fragments. The primary state variable should be an
atom-to-fragment assignment. Learned keys can then describe the chemical role,
charge, spin, and other attributes of each fragment.

### 2.5 Produce one effective potential

At inference, the model should generate one set of effective force-field
parameters and evaluate the force field once. It should not evaluate every
vertex potential and average or diagonalize their energies.

### 2.6 Differentiate through all state and parameter dependence

If the generated parameters depend on coordinates,

\[
\theta(\mathbf R)=f_\theta(x(\mathbf R),c(\mathbf R)),
\]

then the force must be computed from the total derivative

\[
\mathbf F=-\frac{d}{d\mathbf R}
E_{\mathrm{FF}}\!\left(\mathbf R;\theta(\mathbf R)\right).
\]

This includes derivatives through the geometric features, state assignment,
FiLM modulation, generated force-field parameters, and variational or SCF
solutions. Holding the generated parameters fixed during force evaluation
would generally produce nonconservative forces.

## 3. State representation

### 3.1 Atom-to-fragment assignment

For atom \(i\) and fragment \(f\), define

\[
C_{if}\in[0,1],\qquad \sum_f C_{if}=1.
\]

For a nonreactive vertex, \(C\) is one-hot. In a future proton-transfer state,
the transferring proton may have fractional membership in two candidate
fragments.

The model should accept a continuous \(C\) from the beginning, even though the
first validation uses one-hot assignments.

### 3.2 Fragment co-membership projector

Define

\[
P_{ij}=\sum_f C_{if}C_{jf}.
\]

Then \(P_{ij}\) is one when atoms are in the same fragment, zero when they are
in different fragments, and continuous for a soft partition. Because it uses
co-membership rather than a fragment index, it is invariant to arbitrary
permutations of fragment labels.

For decompositions that require the identities of two fragments rather than
only same-versus-different membership, retain the directed block weights

\[
P_{ij}^{fg}=C_{if}C_{jg}.
\]

These can be combined with fragment attributes without making the model depend
on arbitrary fragment numbering.

### 3.3 Fragment attributes and local state keys

Each fragment can carry an attribute vector

\[
r_f = \operatorname{Embed}
\left(Q_f,S_f,\text{chemical role}_f,\ldots\right),
\]

where \(Q_f\) is fragment charge and \(S_f\) represents spin or multiplicity in
the form used by the model.

The local state key at atom \(i\) is

\[
k_i=\sum_f C_{if}r_f.
\]

Also provide a local mixing or assignment-uncertainty variable

\[
u_i=1-\sum_f C_{if}^2.
\]

This is zero at every vertex and positive when atom \(i\) is shared between
fragment assignments. For two equal memberships, it is convenient to use the
rescaled form \(\tilde u_i=2u_i\), which ranges from zero to one.

The conditioning vector can initially be

\[
c_i=[k_i,u_i].
\]

Geometry remains in the main feature network. It is not necessary for the
conditioning network itself to reproduce the full environmental encoding.

## 4. Generic primitives with fragment-projected features

### 4.1 Primitive neighbor contributions

For an ACE-like featurization, compute a generic pair contribution

\[
\psi_{ij}^{n\ell m}
=R_n(r_{ij})Y_{\ell m}(\widehat{\mathbf r}_{ij})e_{Z_j}.
\]

This operation depends on geometry and element identity but not on the chosen
fragmentation. It should be shared by every state.

### 4.2 Internal and environmental densities

For each central atom, construct

\[
\rho_{i,\mathrm{in}}^{n\ell m}
=\sum_j P_{ij}\psi_{ij}^{n\ell m},
\]

\[
\rho_{i,\mathrm{env}}^{n\ell m}
=\sum_j(1-P_{ij})\psi_{ij}^{n\ell m}.
\]

At a vertex these are exactly the contributions from the central atom's
fragment and its environment. At a soft state they vary continuously.

The following nonlinear feature blocks should be retained explicitly where
their cost is acceptable:

\[
\rho_{\mathrm{in}}\otimes\rho_{\mathrm{in}},\qquad
\rho_{\mathrm{in}}\otimes\rho_{\mathrm{env}},\qquad
\rho_{\mathrm{env}}\otimes\rho_{\mathrm{env}}.
\]

These distinguish intrinsic fragment geometry, fragment-environment response,
and nonlinear environmental correlations. At higher body order, it may be
preferable to down-select channels rather than enumerate every possible
internal/environmental word.

### 4.3 Recommended feature interface

The parameter network should receive structured blocks rather than a single
undifferentiated vector:

\[
x_i=
\left[
x_{i,\mathrm{in}},
x_{i,\mathrm{env}},
x_{i,\mathrm{cross}}
\right].
\]

The blocks can be embedded into a common hidden dimension before entering the
shared parameter function. Preserve their identity with separate initial
linear maps or explicit channel tags.

### 4.4 Relation to possible future message passing

Message passing is not required. The proposed projection acts naturally on the
current neighbor-density or ACE construction.

If message passing is added later, use the same projector at every interaction
layer:

\[
m_{i,\mathrm{in}}^{(l)}
=\sum_jP_{ij}M_l(h_i^{(l)},h_j^{(l)},e_{ij}),
\]

\[
m_{i,\mathrm{env}}^{(l)}
=\sum_j(1-P_{ij})M_l(h_i^{(l)},h_j^{(l)},e_{ij}).
\]

This prevents fragment provenance from being lost as information propagates.

## 5. Parameter-generation network

### 5.1 Baseline plus environmental response

Retain the valuable conceptual separation between the fragment baseline and
the environmental correction, but build both from shared primitive features:

\[
\theta_i
=\theta_i^{\mathrm{frag}}(x_{i,\mathrm{in}},c_i)
+\Delta\theta_i^{\mathrm{env}}
(x_{i,\mathrm{in}},x_{i,\mathrm{env}},x_{i,\mathrm{cross}},c_i).
\]

Construct the environmental correction so that

\[
\Delta\theta_i^{\mathrm{env}}
(x_{\mathrm{in}},0,0,c_i)=0.
\]

One implementation is

\[
\Delta\theta_i^{\mathrm{env}}
=g(a_{i,\mathrm{env}})\,
\widetilde{\Delta\theta}_i^{\mathrm{env}},
\]

where \(a_{i,\mathrm{env}}\) is a nonnegative measure of environmental density
and \(g(0)=0\). This gives exact isolated-fragment recovery while permitting
arbitrarily nonlinear geometry dependence when an environment is present.

### 5.2 FiLM conditioning

FiLM should be the first state-conditioning mechanism implemented. For hidden
layer \(l\),

\[
a_i^{(l)}=W_lh_i^{(l)}+b_l,
\]

\[
(\delta\gamma_i^{(l)},\beta_i^{(l)})=G_l(c_i),
\]

\[
h_i^{(l+1)}=operatorname{SiLU}
\left[
(1+\delta\gamma_i^{(l)})\odot a_i^{(l)}
+\beta_i^{(l)}
\right].
\]

Initialize the final layer of each \(G_l\) to zero. The initial model is then
an ordinary shared network, and state specialization develops during training.

Use separate FiLM generators for parameter families with distinct physical
roles, for example:

- bonded or short-range parameters;
- permanent multipoles;
- polarization or charge-response parameters;
- dispersion parameters;
- damping or overlap parameters.

The main hidden trunk can still be shared.

### 5.3 Why FiLM is the initial choice

The fragment projector already makes the features state dependent. FiLM only
needs to determine how the shared feature basis is used by a particular
fragment state. This provides strong parameter sharing, inexpensive per-atom
conditioning, smooth interpolation, and a common activation basis across all
vertices.

### 5.4 Optional low-rank hypernetwork adapters

Design the layer API so that a low-rank generated update can be introduced
later without changing the data or state representation:

\[
W_l(c_i)=W_l^{(0)}+
U_l\operatorname{diag}[a_l(c_i)]V_l^{\mathsf T}.
\]

The conditioning network generates only \(a_l\). The matrices \(U_l\) and
\(V_l\) are shared. This permits state-dependent mixing between hidden channels
without generating a complete dense network for every atom.

Do not implement full weight generation in the first version. Add a low-rank
adapter only when an ablation shows that FiLM has adequate features but cannot
represent the required state-dependent combinations of those features.

### 5.5 Physical output constraints

Predict unconstrained latent variables and transform them into valid physical
parameters afterward:

\[
\eta_i=f_{\mathrm{param}}(x_i,c_i),\qquad
\theta_i=T(\eta_i).
\]

Examples include:

- softplus or bounded maps for positive scalar quantities;
- Cholesky factors for positive-definite response tensors;
- explicit projection to enforce total fragment or system charge;
- symmetric tensor construction for multipoles or polarizabilities;
- smooth bounded damping parameters;
- a union set of candidate bonded terms with continuous amplitudes.

Interpolation should occur in a shared latent space before these constraints
are applied, unless a particular physical parameter has a more appropriate
known interpolation rule.

## 6. Energy construction and EDA supervision

Let the force-field energy be assembled from the physical terms already present
in the model:

\[
E_{\mathrm{model}}
=\sum_\alpha E_\alpha
(\mathbf R;\theta(\mathbf R,C)).
\]

For geometry \(g\) and fragmentation state \(t\), the EDA data provide
fragmentation-dependent component targets

\[
E_{g,t,\alpha}^{\mathrm{EDA}},
\]

while every valid fragmentation corresponds to the same total energy and
forces:

\[
\sum_\alpha E_{g,t,\alpha}^{\mathrm{EDA}}
=E_g^{\mathrm{QM}},
\qquad
-\nabla_{\mathbf R}E_g^{\mathrm{QM}}=\mathbf F_g^{\mathrm{QM}}.
\]

A useful training objective is

\[
\mathcal L=
w_E\mathcal L_E
+w_F\mathcal L_F
+\sum_\alpha w_\alpha\mathcal L_{\mathrm{EDA},\alpha}
+w_P\mathcal L_{\mathrm{property}}
+w_V\mathcal L_{\mathrm{vertex}}
+w_C\mathcal L_{\mathrm{consistency}}.
\]

The terms have the following roles:

- \(\mathcal L_E\): total energy accuracy;
- \(\mathcal L_F\): conservative force accuracy;
- \(\mathcal L_{\mathrm{EDA},\alpha}\): component-level physical supervision;
- \(\mathcal L_{\mathrm{property}}\): multipoles, response properties, or other
  parameter-related observables;
- \(\mathcal L_{\mathrm{vertex}}\): preservation of isolated or otherwise
  well-defined fragment states;
- \(\mathcal L_{\mathrm{consistency}}\): agreement of total energy and force
  across different valid fragmentation views of the same geometry.

Do not require EDA components to interpolate linearly between fragmentations.
Their decomposition is fragmentation dependent. Use EDA to anchor the discrete
states and total energies and forces to constrain continuous intermediate
representations.

## 7. Data representation

Each geometry should have a stable `geometry_id` and one or more state views.
A suggested logical schema is:

```text
geometry:
    geometry_id
    atomic_numbers
    coordinates
    total_energy
    forces
    optional total-system properties

state_view:
    geometry_id
    state_id
    atom_fragment_assignment C
    fragment_charge
    fragment_spin_or_multiplicity
    fragment_role
    EDA component targets
    optional fragment properties
```

Multiple state views of one geometry should share coordinate storage. Generic
pair primitives can be cached per geometry during training; the inexpensive
projection and nonlinear contractions are then performed for each state view.

For the first water study, the principal state uses each water molecule as a
neutral singlet fragment. If alternative legitimate EDA fragmentations are
available for the same cluster, retain them as additional state views. They
provide a direct nonreactive test of state conditioning and total-energy
consistency.

## 8. Phase I: nonreactive water validation

### 8.1 Scope

Begin with configurations for which the molecular identity of every water is
unambiguous. Do not learn a router in this phase. Supply the discrete assignment
matrix \(C\) from molecular connectivity or dataset metadata.

The main objectives are to verify that:

1. the new projected featurization reproduces isolated-fragment behavior;
2. the model learns environmental changes without corrupting fragment
   baselines;
3. FiLM conditioning is numerically stable;
4. EDA terms retain their intended interpretation;
5. the total energy and forces remain independent of arbitrary fragment labels;
6. analytic/autodifferentiated forces include all parameter-response terms.

### 8.2 Recommended training stages

#### Stage A: isolated water

Train or validate the fragment baseline on isolated water configurations over
the relevant intramolecular distortion range.

Required checks:

- \(x_{\mathrm{env}}=0\) exactly;
- the environmental correction is zero to numerical precision;
- predicted fragment parameters agree with their EDA/property targets;
- rotation, translation, and atom-permutation tests pass;
- analytic forces agree with finite differences.

#### Stage B: water dimers

Introduce intermolecular environments while keeping every water as a discrete
fragment. Dimers are particularly useful because internal and external feature
channels are easy to inspect.

Track errors as functions of O--O distance, hydrogen-bond angle, monomer
distortion, and interaction strength. Confirm that isolated-fragment behavior
is recovered continuously as the dimer separates.

#### Stage C: small water clusters

Train on trimers and larger clusters to test many-body environmental response.
Hold out complete cluster sizes and structural motifs rather than only random
frames. This distinguishes genuine transferability from interpolation within a
trajectory.

#### Stage D: alternative nonreactive fragmentation views

Where computationally meaningful, calculate EDA data for more than one
fragmentation of the same geometry. Train the fragmentation-specific EDA terms
while enforcing identical total energy and forces.

This is the first direct validation that state conditioning changes the
physical decomposition without changing the adiabatic target.

#### Stage E: synthetic continuous-state sweeps

For pairs of valid fragmentation views, form

\[
C(\lambda)=(1-\lambda)C_A+\lambda C_B,
\qquad \lambda\in[0,1].
\]

No intermediate EDA target is required. Apply total-energy and force targets
and inspect parameter continuity. This exercises the future reactive interface
without requiring reactive configurations.

Use these sweeps as a controlled diagnostic rather than assuming that every
soft partition has a unique physical EDA interpretation.

### 8.3 Important limitation of neutral-water-only training

If every training example has exactly the same fragment role, charge, spin, and
one-hot assignment pattern, FiLM conditioning is formally present but not
identified by the data; it can be absorbed into the shared weights. This is not
a failure, but it means that state conditioning has only been tested as software,
not learned as chemistry.

Alternative fragmentation views, ion-containing validation data, or the later
reactive dataset are required to demonstrate that the conditioning mechanism
learns distinct state responses.

## 9. Validation tests

### 9.1 Symmetry and invariance

- Translate and rotate configurations and verify invariant energies and
  equivariant forces.
- Permute identical atoms within a fragment.
- Permute fragment labels without changing \(C\)'s physical partition.
- Permute identical water molecules in a cluster.
- For a symmetric two-state example, exchange the two fragment roles and verify
  the corresponding output transformation.

### 9.2 Vertex preservation

- Verify exact zero environmental correction for isolated fragments.
- Compare the new fragment baseline against the previous implementation on a
  frozen validation set.
- Check that adding distant spectator fragments produces corrections that
  vanish with the intended cutoff or long-range limit.

### 9.3 Force consistency

- Compare autodifferentiated forces with central finite differences of the
  final scalar energy.
- Repeat while allowing all generated parameters to vary with geometry.
- If polarization is solved self-consistently, test gradients at tighter SCF
  tolerances until force errors converge.
- Run short NVE trajectories and measure energy drift.

### 9.4 State continuity

For fixed geometries, sweep \(\lambda\) between two state assignments and plot:

- total energy;
- force components;
- every major EDA term;
- representative charges and multipoles;
- polarizability or response eigenvalues;
- selected bonded and damping parameters;
- FiLM scales;
- the norm of \(\partial\theta/\partial\lambda\).

The total surface and forces should remain smooth. Individual EDA terms and
parameters need not be linear, but they should not oscillate without physical
or data-driven justification.

### 9.5 Generalization splits

Use validation splits that hold out:

- complete water-cluster sizes;
- hydrogen-bond network motifs;
- monomer distortion ranges;
- strongly compressed configurations;
- fragmentation views, when several are available;
- configurations close to the eventual proton-transfer regime.

## 10. Suggested ablations

The following sequence identifies which pieces provide real value:

1. Current separate fragment/environment featurizers.
2. Shared primitives with projected features but no state conditioning.
3. Projected features plus state key concatenation.
4. Projected features plus FiLM.
5. FiLM with separate modulators for each parameter family.
6. FiLM plus a low-rank adapter in the final parameter layer.

Compare not only total energy and force errors, but also EDA component errors,
parameter stability, state-sweep smoothness, and extrapolation to held-out
cluster motifs.

Do not add a hypernetwork solely because it lowers the training loss. It should
improve a topology- or state-specific validation failure without degrading
vertex preservation or parameter interpretability.

## 11. Minimal implementation skeleton

```python
def model(atomic_numbers, positions, state):
    # State data are supplied in the nonreactive phase and predicted later.
    C = state.atom_fragment_assignment
    fragment_attributes = state.fragment_attributes

    # Generic geometry/element primitives; independent of fragmentation.
    pair_primitives = primitive_encoder(atomic_numbers, positions)

    # Fragment-label-invariant soft partition.
    P = C @ C.transpose(-1, -2)

    # Project before nonlinear ACE/tensor-product contractions.
    density_in = aggregate(P, pair_primitives)
    density_env = aggregate(1.0 - P, pair_primitives)

    features_in = internal_contractions(density_in)
    features_env = environmental_contractions(density_env)
    features_cross = cross_contractions(density_in, density_env)

    # Local state key and mixing descriptor.
    role_embedding = embed_fragment_attributes(fragment_attributes)
    local_key = C @ role_embedding
    mixing = 1.0 - (C * C).sum(dim=-1, keepdim=True)
    conditioning = concatenate(local_key, mixing)

    # Shared FiLM-conditioned parameter function.
    features = embed_feature_blocks(
        features_in, features_env, features_cross
    )
    latent = film_parameter_trunk(features, conditioning)

    fragment_parameters = fragment_heads(latent, conditioning)
    environment_delta = environment_heads(latent, conditioning)
    environment_delta *= exact_environment_gate(density_env)

    raw_parameters = fragment_parameters + environment_delta
    parameters = impose_physical_constraints(raw_parameters, C, state)

    # One physical potential evaluation.
    energy, auxiliary_terms = force_field_energy(
        positions, parameters, state
    )

    # Differentiate this final scalar energy without detaching parameters.
    forces = -gradient(energy, positions)
    return energy, forces, auxiliary_terms, parameters
```

The actual implementation should avoid materializing a dense \(N\times N\)
projector for large systems. Compute \(P_{ij}\) only on neighbor-list edges:

```python
P_edge = (C[edge_i] * C[edge_j]).sum(dim=-1)
```

## 12. Software interfaces to establish now

Define stable interfaces for the following components:

```text
StateDescriptor
    atom_fragment_assignment
    fragment_attributes
    local_conditioning()
    edge_comembership(edge_index)
    mixing_measure()

PrimitiveEncoder
    geometry -> reusable pair/equivariant primitives

FragmentProjector
    primitives + StateDescriptor -> internal/environmental densities

FeatureContractor
    projected densities -> internal/environmental/cross feature blocks

ConditionedParameterNetwork
    feature blocks + local conditioning -> raw force-field parameters

PhysicalConstraintLayer
    raw parameters -> valid and conserved parameters

ForceFieldEvaluator
    geometry + effective parameters -> one scalar energy
```

The `ConditionedParameterNetwork` should expose a configuration such as

```text
conditioning_mode = "none" | "concatenate" | "film" | "low_rank"
```

even if only `film` is implemented initially. This makes later ablations and
capacity upgrades straightforward.

## 13. Transition to reactive training

After the water validation succeeds, the reactive extension should proceed in
the following order.

### 13.1 Add discrete reactive vertices

For \(\mathrm{H_5O_2^+}\), construct the two assignments

\[
A=\mathrm{H_3O_A^+/H_2O_B},\qquad
B=\mathrm{H_2O_A/H_3O_B^+}.
\]

They must use the same network and fragment-role embeddings under exchange of
the two oxygen atoms. They are two assignments of one shared expert template,
not two independently parameterized networks.

### 13.2 Train both fragmentations on the same geometries

Use EDA terms to anchor the two discrete decompositions. Apply the same total
energy and force targets to both views.

### 13.3 Introduce soft proton ownership

Interpolate the transferring proton's fragment memberships and supply the
resulting \(C\), \(P\), local key, and mixing descriptor to the already tested
architecture.

Initially use a physically transparent coordinate such as the difference in
the two O--H distances. Later replace it by a permutation-antisymmetric learned
router if necessary.

### 13.4 Add transition-only capacity only if required

If the single effective force-field parameterization cannot reproduce the
shared-proton stabilization, add either:

1. a low-rank parameter-network adapter multiplied by the mixing measure; or
2. a local scalar mixing correction

\[
E_{\mathrm{model}}
=E_{\mathrm{FF}}(\mathbf R;\theta_{\mathrm{eff}})
+u(\mathbf R)\Delta E_{\mathrm{mix}}(\mathbf R).
\]

Both corrections vanish at the vertices and retain a single force-field
evaluation. The second option is preferable to forcing charges, polarizabilities,
or bonded parameters to absorb energy that the physical force-field form cannot
naturally represent.

### 13.5 Learn the state router last

Once the conditioned potential works with supplied soft states, learn
\(C(\mathbf R)\). This separates two problems:

- whether the potential can represent the adiabatic surface given a state;
- whether the correct state can be inferred from geometry.

Trying to learn both from the beginning would make failure modes difficult to
identify and would introduce substantial non-identifiability.

## 14. Recommended first implementation

The first production prototype should contain:

1. the current generic radial/angular primitive calculation;
2. edge-wise fragment co-membership weights;
3. projected internal and environmental neighbor densities;
4. internal, environmental, and selected cross contractions;
5. a shared MLP parameter trunk;
6. two or three identity-initialized FiLM layers;
7. separate physical parameter heads and constraint transformations;
8. an exactly vanishing environmental residual for isolated fragments;
9. one force-field or polarization evaluation;
10. energy and force differentiation through the complete computation;
11. unit tests for fragment relabeling, vertex limits, and finite-difference
    forces;
12. configuration hooks, but not yet full implementations, for low-rank
    adapters and a learned state router.

This version is sufficiently general to expose the important architectural
questions on ordinary water while remaining simple enough to debug. The
reactive phase should primarily add new state data and a router, rather than
replace the representation or parameter network.

## 15. Decision summary

- Replace separate fragment and environment featurizers with shared geometric
  primitives and state-dependent projection.
- Do not apply fragment masks only after nonlinear invariant features have been
  formed.
- Keep internal, environmental, and cross feature channels distinguishable.
- Preserve isolated-fragment parameters using an exactly vanishing
  environmental residual.
- Implement FiLM first; reserve low-rank generated weights for demonstrated
  state-dependent interaction failures.
- Supply fragment assignments during nonreactive validation and learn the
  geometry-to-state router only after the conditioned potential is working.
- Use EDA to anchor physically meaningful vertex decompositions and total
  energies and forces to constrain the adiabatic surface and continuous state
  interpolation.
- Evaluate one effective potential and differentiate the final scalar energy
  through every generated quantity.

## References for the conditioning mechanisms

- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*,
  arXiv:1709.07871.
- Ha, Dai, and Le, *HyperNetworks*, arXiv:1609.09106.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
  arXiv:2106.09685. The application here is not language-model adaptation, but
  the low-rank parameterization provides a useful template for generated weight
  updates.
