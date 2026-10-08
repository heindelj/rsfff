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

## Next

- Data (ChemLab campaign): vacuum + probe + uniform-field frames, Hessians, scans.
- Train the four variants (B0 constant-parameter baseline first: heads off, tables only).
- Hessian-vector-product loss; per-family `bond_var` weights; relative-energy target.
- Typed nonbonded priors (q0 for covalent Cl); inter-fragment permanent sources for the field
  features; external sources in the fixed-K solve; quadrupole polarizability in the film (R2).
