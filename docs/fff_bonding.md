# The `bonding` branch: monomer models with explicit topology, couplings and field features

Branch `bonding` off `nonreactive` (4f1da5f). Plan and progress: Obsidian card
"Bonded Monomer Models (1-4 Topology)" / "monomer bonded plan" (also the project doc
`claude/monomer_bonded_plan.md`). This note records what the code does and its conventions.

## Scope

Monomers whose every atom pair is 1-2, 1-3 or 1-4 (the flexible set of Abdullah et al.,
arXiv:2504.14398, plus ethene, HF, H2O, NH3, CH4, OH-, OH., H3O+, NH4+ and HPO4.-), with
`exclude_through: 4` so the monomer energy is `sum E0 + bonded`. Four variants, two flags:

| | `couplings: false` | `couplings: true` |
|---|---|---|
| `field_features: false` | `configs/bonding_monomer.yaml` (A) | `..._coupled.yaml` |
| `field_features: true` | `..._fields.yaml` (C) | `..._coupled_fields.yaml` |

Trainer: `python -m rsfff.train.train_bonding <config>`.

## What is new

**Explicit covalent graph** (`Batch.covalent_bonds`, extxyz `bonds`). `BondedTopology.from_state`
takes it and enumerates bonds, angles, impropers, torsions, angle pairs and their index maps
(`src/ff/film/topology.py`, vectorized). Without it the legacy rule (every intra-fragment non-H-H
pair) is used, which the loader allows only for single-heavy-atom fragments. The loader keeps the
bonds attached to their atoms through the fragment sort and through `fragment_view`.
`BondedTopology.separation` gives the graph distance of any pair list.

**(Z, degree) typing** (`film.atom_typing: degree`). Prior-table rows per (element, covalent
degree): resonance-safe, and it separates C=O/C-O, sp2/sp3 C, amine/ammonium N, H3O+/H2O O.
Typed bond/angle/improper priors for the organic set (`ORGANIC_BOND_PRIOR`, `TYPED_*` in
`film/bonded.py`), element fallback. The degree-typed bond table is read in canonical
orientation (the element-typed path keeps its legacy lookup for checkpoint compatibility).
Nonbonded element priors added for C, P, S (response, q0, Pauli, dispersion). **Known gap:**
the nonbonded priors are still per element, so covalent Cl starts from the Cl- ion values.

**Torsions and couplings** (`film.torsions`, `film.couplings`; `src/ff/film/terms.py`):
Fourier torsions `sum_n K_n (1 + cos n phi)` (Chebyshev in cos phi, no acos, cosine-only), and
bond-bond, bond-angle, angle-angle, torsion-bond, torsion-angle, torsion-angle-angle. Couplings use
the Morse coordinate `y = 1 - exp(-beta (r - r_eq))` and `dc = cos theta - cos theta_eq`; the
bond/angle couplings are `2 rho sqrt(a1 a2)` with `|rho| < 0.95`, positive-definite per 2x2
block. Per-type tables (zero-init) + zero-init feature deviation (`delta_iso`, included in the
`bond_var` regularizer) + gated environment deviation; symmetry by orbit sum (both orderings of
each instance). A fresh head is exactly the uncoupled model.

**External sources** (`src/ff/external.py`, `Batch.external`): probe point charges and uniform
field / field gradient. `external_potential` returns the polytensor conjugate `ext_m` with
`E_ext = ext_m . M + c` (probe charges one-center Slater damped on the atom side, uniform
sources closed form); it enters the permanent elst (`interaction["external"]`) and the coupled
solve's right-hand side (`CoupledSystem.ext_m`, a constant shift of the coupling gradient), so
the induction channel stays a pure relaxation. Variational film solve only (the fixed-K model
raises).

**Response properties as energy derivatives** (`FilmModel.response_properties`):
`mu = -dE/dF`, `Theta = -3 dE/dG` (traceless), `alpha = -d2E/dF2`, w.r.t. a uniform F, G added
at zero, about a chosen origin. Runs the energy with the full adjoint (`stationary=False` in
`coupled_response`) so the second derivative sees `dx*/dF` under `create_graph`.
Verified: the derivative dipole equals the summed multipoles; without field features the
derivative alpha equals the closed-form `fragment_polarizability`; alpha matches finite fields.

**O(F^2) field features** (`film.field_features`; `src/ff/film/fields.py`): undamped
`(phi, E, gradE)` at each atom from the external sources -> linear invariants in the atom's frame
(`dphi` relative to the fragment mean; `E` on the lambda=1 internal features; `gradE` on the
lambda=2 features in Cartesian form) -> pairwise learned products -> `W2 tanh(W1 p)` with `W2`
zero-init, added to the bonded latent in the field-dressed evaluation only. The bonded energy
therefore has no first-order field dependence: the zero-field dipole is unchanged and alpha
gains `-d2E_b/dF2` (tests pin both). Its energy is booked in induction as
`E_b(theta) - E_b(theta_0)`. Sources: external only for now; other fragments' permanent
multipoles are the next source.

**Training streams** (`bonding.streams` in `src/train/train_bonding.py`): rsfff's loader wants a
label on every frame of a file or on none, so a set comes as several files: the main file
(`data.path`; epochs, checkpoints) plus streams `{name, path, weight, batch_size}`, each its own
minibatch every step and split with the same holdout fraction; their held-out parts enter the
validation loss. The ChemLab project `~/research/fff_bonding/cl` exports a set
(`python/scripts/export_dataset.py`) as `vacuum.xyz` (main), `vacuum_pol.xyz` (`pol`),
`probe.xyz` and `field.xyz`. `bonding.probe_path` still works (a stream named `probe`).

**ECE-fed bonded heads** (`film.bond_features: ece`; `src/ff/film/ece.py`): per-bond
features of the covalent graph from bb's edge cluster expansion (`bb.features.EdgeClusterExpansion`,
bb `docs/ece.md`): the bonded neighbours of each end expanded in the bond frame and combined up to
correlation order 2 (four-body: bond + one neighbour per end), i.e. the angles on each side, the
dihedral harmonics `cos m omega` between the ends, and the same-end angle-angle and out-of-plane
invariants. Both bond directions are merged symmetrically (`g+ + g-`, `(g+ - g-)^2`;
pseudoscalars only as `o+^2 + o-^2`, `o+ o-`), `asinh`-compressed, and embedded by a small MLP
(`ece_dim`). Every head then reads the embedding next to the latents: bond `b[ij]`; angle
`b[ia] + b[ak]`, `(b[ia] - b[ak])^2`; improper leg + symmetric partner pair; torsion
`[b[ab], b[bc], b[cd]]`, couplings their legs, angle pairs their three bonds, all inside the
existing orbit sums (reversal permutes the bond slots too), so the symmetry is still exact.

- *Isolated by construction*: the ECE is intramolecular and cutoff-free, so it is the same for
  `theta_0` and `theta`; the env deviation stays the gated joined-latent branch (zero in
  isolation, booked in induction). Nothing in the ECE path can double-count the environment.
- *Own coordinates*: `ece_own_bond: true` adds a Morse basis of the bond's own length, and an
  angle's legs carry the angle itself, so the parameters see their own coordinate (what makes the
  bounded forms exact for 1-4 monomers, and what can turn a parameter into a disguised energy).
  `bonding.bond_var_weight` (the `train_film` regularizer on `delta_iso`) and a stretch-scan
  hold-out are the guard.
- `ece_latents: false` removes the SOAP latents from the bonded/term heads: parameters are then a
  function of the ECE features and the atom types alone (cannot be combined with
  `field_features`, which shifts the latents).
- Off by default; a model without it has exactly its old parameter set.

**ECE electrical heads** (`film.electrical_features: ece`, needs `bond_features: ece`;
`src/ff/film/electrical.py`): the permanent charges, dipoles and quadrupoles, the atomic
polarizabilities and the SQE compliances built on the covalent bonds instead of the atom latents,
each as *free-atom value on the diabat + bond terms x an envelope that vanishes as the bond
dissociates*:

- charges `q_i = q_inf,i + sum_b sigma_ib e_b g_b` (antisymmetric bond flows; a Pauling
  `kappa (chi_a - chi_b)` start): the fragment charge is conserved exactly, every atom ends on its
  diabatic charge;
- dipoles / quadrupoles: sums over the atom's outgoing bonds of `e_b x` an equivariant bond-frame
  readout (`bb.heads.EdgeReadout`, axial and off-axis), zero for a free atom;
- polarizability `alpha_i = alpha0(Z, q_inf) exp(sum e_b s) expm(-sum e_b K)` (`K` traceless): positive
  definite always, the free atom / ion value at dissociation (`FREE_ATOM_ALPHA`, b3lyp/def2-svpd
  finite field, `scripts/free_atom_alpha.py`; linear in charge for a resonance-averaged `q_inf`);
- charge flows only along covalent bonds, with compliance `e_b softplus(...)`: the channel closes
  as the bond breaks (no fractional charge between separated atoms);
- envelope `e_b = exp(-gamma (r - r_ref))`, `r_ref` and `gamma = beta = sqrt(k/2D)` from the bond
  type's Morse prior times a learnable log-scale per quantity: a geometric electrical bond order;
- the response parameters keep a gated environment factor `exp(gate NN(z_joined))`, exactly 1
  isolated: the environment modifies the physical parameters rather than replacing them.

The limits hold by construction on a fixed diabatic topology (tests pull an atom to 40 A, and a
whole molecule apart). For the remaining fragment to become exactly the isolated fragment the ECE
neighbours are weighted by the Morse variable of the bond reaching them (`ece_neighbour_decay`,
default on); with `elec_latents: false` the electrical heads see only the covalent graph and the
partial-dissociation limit is exact too (the SOAP latents feel the departing atom out to their
cutoff, and the fragment conditioning keeps the diabat's composition).

**Diabatic state** (`src/ff/film/diabats.py`; `Batch.diabat_charge`, `diabat_flow`,
`diabat_group`): per-atom asymptotic charges plus a **dissociation channel per bond**.

- Atom charges (all bonds broken homolytically): a Lewis guess from the graph -- over-valent N/O
  carry `+(deg - valence)`; the remaining charge sits on the free valences left unpaired by a
  maximal pi pairing, averaged over every atom that can be left unpaired (acetate: -1/2 on each O;
  HPO4-: -1/3 on the three terminal O).
- Channels: the asymptotic flow `f_b` of each bond (charge moved along its stored orientation once
  that bond alone is broken, relative to the atom charges). Guess: dative bonds of an over-valent
  centre are heterolytic with the centre keeping the pair (NH4+ -> NH3 + H+, CH3NH3+ -> CH3+ + NH3);
  a bond that would split a resonance charge gives all of it to the smaller side (acetate C-O ->
  CH3CO + O-); everything else homolytic (and ring bonds, which separate nothing).
- Bonds drawing on one source (a cation centre, a resonance group) share an exclusive `group`.
- The heads use `q_asym = q_inf + sum_b sigma t_b f_b` with the channel progress
  `p_b = 1 - exp(-gamma ramp(r - r_ref - onset))` (exactly 0 in a bond's normal range, C2; `elec_channel_onset` 0.5 A and `elec_channel_width` 0.3 A by default, rates learnable per bond type) and
  `t_b = p_b prod_{b' in group, b' != b} (1 - p_b')`; charges are `q_asym` + enveloped flows and
  `alpha0 = alpha0(Z, q_asym)` (H+ has alpha 0). A proton leaving NH4+ ends as H+, CH3NH3+ separates
  into exactly CH3+ and NH3, two protons pulled from NH4+ leave NH2+ (tests).
- Headers `diabat_charges` (per atom), `diabat_flows` and `diabat_groups` (per bond, in the `bonds`
  order; groups are atom indices or -1) override the guess; the guess is a placeholder for a table
  of diabats.
- The bonded Morse terms are still referenced to neutral atoms, so a heterolytic channel's energy
  asymptote is not yet consistent with its electrical one.

**Hessian loss** (`bonding.hessian_weight`, `hessian_scale` [Ha/Ang^2], `hessian_probes`): the
loader reads a `hessian` header (`(3n, 3n)`, Ha/bohr^2, file atom order; permuted with the atoms,
stored in Ha/Ang^2) into `Batch.hessian`; the loss compares `H v = -d(F.v)/dx` for a few Gaussian
probes per step (one extra backward pass each, fixed seed at evaluation) and logs `h_rel`, the
relative error of `H v`. `scripts/eval_bonding.py` evaluates a checkpoint per molecule/kind
(optionally on the trainer's `--split val`): energies, forces, full Hessians, dipole / quadrupole /
polarizability as energy derivatives, and for probe frames (`--vacuum`) the interaction energy
and force change against the vacuum frame at the same geometry.

## Next

- Data (ChemLab `fff_bonding/cl`): `monomer-pilot` -- Wigner/rotor, scans, probe and field
  frames for the seven pilot molecules; Hessians at distorted geometries later.
- Train the four variants (B0 constant-parameter baseline first: heads off, tables only).
- Per-family `bond_var` weights; relative-energy target.
- Typed nonbonded priors (q0 for covalent Cl); inter-fragment permanent sources for the field
  features; external sources in the fixed-K solve; quadrupole polarizability in the film (R2).
