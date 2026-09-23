# The Pairing Model: Bonding as a Variational Spin-Pairing Energy

## Design note and first implementation (`rsfff.ff.pairing`)

## 1. Why

The film model (`docs/fff_film.md`) carries its covalent bonds as Morse + cosine-angle
terms on a topology assigned by the fragmentation, and switches its classical channels off
inside a fragment with per-element-pair Fermi functions of distance. The 1-body scans show
the cost: where those switches turn the intramolecular electrostatics, Pauli and dispersion
on (`r0(O,H) = 1.25` Å, `alpha = 40`), the Morse has to reproduce the negative of a large,
sharply windowed quantity, and the PES has a kink. Complement gating (switching the Morse
off with `1 - s`) would turn that cancellation into an agreement condition, but it still
leaves an assigned topology, a distance rule for bonding, and no route to reactivity.

This model removes all three. The bond is an explicit pairing energy between valence
sites, the amount of pairing (the bond order) is a variational state solved for at every
geometry, and that bond order *is* the fragment co-membership: it decides how much of every
classical channel acts between two atoms, and it is the only thing that does. Nothing in the
bonded description is switched by distance; the couplings are Slater overlaps that decay
exponentially on their own.

## 2. The functional

Over every pair within `pairing_cutoff` (4 Å; the taper below it sits where a bond's
coupling is ~1e-6 of its equilibrium value), with `p_ij` the bond order and
`u_i` the residual unpaired valence,

```
F(p) = sum_ij [ -J_ij p_ij + 1/2 kappa_ij p_ij^2 + T ( p ln p + (1 - p) ln(1 - p) ) ]
     + T sum_i [ u_i ln u_i - v_i ln v_i ]

subject to   sum_j p_ij + u_i = v_i    for every atom i.
```

- `J_ij` is the **bare pairing energy**: the Pauli operator (`slater_pauli_pair_energy`, a
  two-center Slater-damped multipolar contraction) applied to a per-atom *valence
  polytensor* `(q_v, mu_v, Theta_v)` with its own exponent `b`. A coarse-grained exchange
  integral: exponentially decaying, anisotropic through the dipole and quadrupole (that is
  where directional bonding and hence bond angles come from; there is no angle term).
- `kappa_ij = sqrt(kappa_i kappa_j)` is the **pairing hardness**, the promotion /
  localization cost of forming the bond. It plays the role `1/alpha` plays in polarization:
  unconstrained, `p = J/kappa` and `E = -J^2 / 2 kappa`, the Heitler-London `S^2` scaling at
  long range; saturated, `p = 1` and `E = -J + kappa/2`. The crossover gives the attractive
  branch its Morse-like shape without a Morse.
- `v_i` is the **valence capacity** (O: 2, H: 1; `DEFAULT_VALENCE`). Its charge dependence
  (§6) is not implemented yet.
- `T` is the barrier scale (0.002 Ha). Both entropic terms vanish as `T -> 0`; at finite `T`
  they keep `0 < p < 1` and `u > 0`, and they make the minimizer a smooth function of the
  parameters. On a bond `1 - p ~ exp(-(J - kappa)/T) ~ e^{-200}`: saturated to double
  precision, so the film accounting is recovered *exactly* on intact water.

`F` is strictly convex on a convex set: the minimizer is unique and smooth in every
parameter and every coordinate. That is the well-posedness statement.

### 2.1 Why the state variable is the pairing matrix, not per-atom spins

A collinear spin per atom cannot represent the H2 singlet (`<S1.S2> = -1/4` for a product
state against `-3/4`), and choosing spin signs on a bonded network is a discrete, frustrated
problem. Pairing fractions with per-atom capacity carry the same coarse-grained physics and
avoid both. "Collinear" here means one scalar per pair.

### 2.2 The solve (`bond_order.py`)

The equality constraints are handled in the dual. With multipliers `lambda_i` the
stationarity conditions decouple:

```
kappa_ij p_ij + T logit(p_ij) = J_ij - lambda_i - lambda_j       (scalar Newton per pair)
u_i = exp(-1 - lambda_i / T)
```

and the `N` marginal equations fix the multipliers. Their Jacobian is `-(D + A)` with
`A_ij = dp_ij/da` and `D_ii = sum_j A_ij + u_i/T`: symmetric, diagonally dominant, positive
definite, solved by a damped Newton method with a dense batched factorization per frame
(a frame is a few hundred atoms; revisit with CG if that changes).

Two numerical points decided the implementation:

1. **The marginal on a saturated atom is a balance of exponentially small quantities**: its
   bonds sit at `1 - e^{-200}`, and what they leave must equal the slack plus the
   `e^{-big}` bond orders of the pairs it does not form. So the residual is evaluated as
   `room_i = (v_i - n_sat) + sum_sat (1 - p)` against `fill_i = u_i + sum_unsat p`, each
   cancellation-free (`1 - p = sigmoid(-x)`), and Newton is taken on `ln fill - ln room`.
   The plain difference converges by one e-fold per step wherever the exponential dominates;
   the log form is exact there in one step. The line search and the convergence test read
   the plain imbalance `|fill - room|`, which is continuous across the saturated/unsaturated
   split (the log ratio is not, and stalls the line search where partners compete).
2. **Rows of atoms whose every term has underflowed** are regularized (`rho = sqrt(eps)`):
   nothing there is determined, and the floor keeps the step finite. Steps are trust-region
   limited to `20 T` so a wild trial cannot overflow the exponential slack.

Water clusters converge in 8-15 iterations; a competing hydrogen bond in ~10.

### 2.3 Derivatives

Nothing is unrolled. The Newton iteration runs under `no_grad` to convergence, then two
Newton steps are re-applied with the graph on, from the converged (detached) point. For a
Newton map `N(theta, lambda) = lambda - K^{-1} R`, `dN/dlambda = 0` at the solution (because
`R = 0` there), so one differentiable step from the solution has the exact first derivative
of the implicit function, and a second step -- whose input already has the exact first
derivative -- has the exact second derivative too. A force loss needs the second:
`d/dtheta [dE/dR]` runs through `dp/dR`. `tests/pairing/test_pairing_bond_order.py` checks
`gradcheck` and `gradgradcheck`; `test_pairing_model.py` checks the force-loss gradient
against central differences through the whole model. The same trick is used inside the
scalar per-pair equation.

### 2.4 Energy

The reported pairing energy is the minimized functional itself (barriers included), so it
is stationary in `p`. The per-atom part is referenced to the free atom (`u = v`), so an atom
without partners contributes exactly zero and the frozen isolated-atom references keep
their meaning: the film's per-fragment `E0` accounting is already per-atom.

## 3. Assembly (`model.py`)

```
E_f     = sum_i E0[Z_i] + E_pair(theta_0)
        + sum_{intra-assigned ij} [ (1 - c_ij) sum_a gate_a E_a + c_ij gate_pauli E_pauli ]
E_inter = sum_a sum_{inter-assigned ij} (1 - c_ij) gate_a E_a        -> eda_cls_elec / mod_pauli / disp
E_ind   = [coupled solve at theta] - [zero response] + [E_pair(theta) - E_pair(theta_0)]
E_cross = c_ij gate_pauli E_pauli and E_pair on inter-assigned pairs  (zero until a reaction)
E_total = sum_f E_f + sum_a E_inter^a + E_ind + E_cross
```

**The co-membership** `c_ij` is derived from the bond orders at `theta_0`:
`c = 1 - (1 - c12)(1 - c13)` with `c12 = p_ij` and `c13 = 1 - prod_k (1 - p_ik p_kj)` over
two-step paths of the candidate graph -- the classical 1-2 / 1-3 exclusions with fractional
bond orders (1-4 paths are not enumerated; add a third product when torsional fragments
appear). It replaces the film's `P_ij` in every role: the intra/inter split of the classical
channels, the induction gate `gate_elst (1 - c)`, and the SQE conductances
(`compliance x c`, on a channel graph that is now every pair within the pairing radius --
which also removes the long-range charge transfer between separated fragments).

**Channel weights.** Every classical channel acts between two atoms with weight `(1 - c)`;
the Pauli repulsion is additionally on within a bond at full strength -- it is the bond's
repulsive wall. Electrostatics inside a bond are off (the permanent multipoles are trained
on molecular multipole labels, not on atoms-in-bonds densities). With
`range_gate="bond_order"` the per-channel Fermi switches are gone and every channel carries
its cutoff taper only; `range_gate="fermi"` keeps the film's switches as an ablation.

**Environment dependence** follows the film exactly: the pairing family has an isolated
evaluation `theta_0` (fragment energies, co-membership) and an env-dressed one `theta`
(second solve), and `E_pair(theta) - E_pair(theta_0)` lands in the induction channel, the
analogue of the film's `E_bonded(theta) - E_bonded(theta_0)`.

**Bookkeeping.** The assigned `fragment_idx` decides which label a pair is compared against
and where its energy is pooled; it never enters the physics. `E_cross` is what keeps the
total an exact sum when the bond orders disagree with the assignment. On intact water it is
zero to ~1e-12 and every film invariant holds: an isolated fragment has exactly zero
induction and exactly zero inter-channel energy (`test_isolated_fragment_vertex`).

## 4. Priors (`heads.py`)

The valence polytensor head is `PauliMultipoleHeads` with the monopole prior equal to the
valence count and the dipole/quadrupole heads zero-initialized. Calibrated against the
pyCMM Pauli priors at rank 0: with `q_v(O) q_v(H) = 2`, `b_pair = 0.834 / bohr` and
`kappa = 0.2 Ha`, the O-H well sits at the pyCMM `r_eq = 1.812 bohr` with the pyCMM
`D = 0.1997 Ha`; the curvature comes out at 0.37 Ha/bohr^2 against Morse's 0.54, which the
network absorbs. At initialization a water is bound by ~0.39 Ha, every bond order is 1 to
1e-11, every non-bond below 1e-11, and the 1-3 H-H pair carries Pauli only (13 kJ/mol).

The pairing exponent is much softer than the Pauli one (0.83 vs ~2.1 / bohr), which the
calibration forces: with the same radial form for both, a minimum needs the attraction to
decay slower than the wall. Whether the two should share a radial basis (the physical
argument: both are overlaps of the same density) is an open question for the fit; the
heads keep them separate with the option of tying them later.

## 5. What is checked (`tests/pairing/`)

- valence marginals hold to 1e-9; a bond saturates; a 1-3 pair does not bond;
- a hydrogen bonded to an oxygen is squeezed out of an acceptor with `J > kappa/2`
  (`p < 1e-2` against 0.6 for a free hydrogen);
- atoms without partners carry zero pairing energy;
- `gradcheck` / `gradgradcheck` of the solve; finite-difference forces of the model;
  finite-difference gradient of a force loss with respect to a pairing parameter;
- the total energy is the exact sum of the four buckets; the isolated-fragment vertex;
- a stretched O-H drops its co-membership below 0.5 and costs energy;
- `film_fit` (the shared cluster loss) runs on a pairing output with the force term.

## 6. Not yet

- **Charge and multiplicity.** `v_i` is a species table. The plan: per-atom `(q_i, u_i)`
  conditioning of every head, `v_i` from element and SQE charge, and the multiplicity as
  the constraint `sum_i u_i >= 2S` (a high-spin state then has no pairing between its
  radical sites and is described by the nonbonded terms alone -- which is also the training
  signal for `J`, `(E_HS - E_LS)/2` along a stretch).
- **Hybridization.** One exponent per atom; the linear combination of an s-like and a
  p-like exponent is a head change.
- **Joint solve.** Bond orders are solved at `theta_0` and the gates derived from them
  before the induction solve (nested, both convex). A joint minimization where the gates
  depend on `p` and `J` on the induced state is non-convex and is left for later.
- **Kernels.** The pairing coupling goes through `rsfff.ff.backend.slater_pauli_pair_energy`,
  so the torchff Pauli kernel serves it unchanged once the port lands; the solve is torch.
- **Data.** Nothing here needs new labels: on intact water the model reproduces the film's
  accounting and trains on the same streams. Reactive validation needs the stretched /
  high-spin scans of the Obsidian plan.
