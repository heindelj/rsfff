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
- `v_i` is the **valence capacity**, `v_i(n_i)` of the atom's formal electron count, which
  is itself a variable of the same minimization (§2.4: O in H3O+ ends at 3, in OH- at 1,
  each O of a Zundel midpoint near 2.5), with the neutral value scaled by a
  zero-initialized readout of the latent (`exp` of a `valence` head).
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

### 2.4 Energy, formal charges and the atomic reference (`electronic_state.py`)

The reported pairing energy is the minimized functional itself (barriers included), so it
is stationary in every variable. The per-atom part is referenced to the free neutral atom
(`u = v0`, `q = 0`), so an atom without partners contributes exactly zero.

The capacity is **not** a fixed number per element: it is `v_i(n_i)` with `n_i` the atom's
*formal* electron count, solved jointly with the bond orders (module docstring of
`electronic_state.py`):

```
min_{p,u,n}  F(p, u) + sum_i [ E0_i(q_i) + B_i(n_i) ]           q_i = n0_i - n_i
s.t.  sum_j p_ij + u_i = v_i(n_i)     (atom)
      sum_{i in frame} n_i = N        (electron count = sum n0 - total charge)
      sum_{i in frame} u_i = 2S       (multiplicity, only when 2S > 0)
```

`v(n) = v0 + a (n - n0) + b (n - n0)^2` is `8 - n` for oxygen (2 at n = 6, 3 at 5, 1 at 7)
and `n (2 - n)` for hydrogen (peak 1 at n = 1). `E0(q)` is the charged atomic reference,
**piecewise linear** through `IP` and `-EA` (`chi = (IP + EA)/2`, `eta = IP - EA` from
`data.atomic_reference_states`), smoothed over `EPS_Q = 0.01 e` at the integer, with a
quadratic wall beyond one electron either way; `B(n) = T [n ln n + (c - n) ln (c - n)]` is
the shell barrier that keeps `0 < n < c`. The piecewise-linear form is essential: a
quadratic `E0` makes every fractional split of a formal charge cheaper than the integer one
and hydronium's charge smears over its hydrogens (`q_O = 0.49`); with the linear one the
formal charges stay at the integers unless the bonding pays, which is what `n_i` is for --
it is the Lewis-structure count, distinct from the SQE partial charge (bond polarity, solved
downstream and never fed into the capacity, which would put water's oxygen at ~1.6).

What comes out at the priors: water `q = 0`, hydronium `q_O = +1` with three bonds,
hydroxide `q_O = -0.99`, a Zundel midpoint splits the charge `0.59 / 0.39` at O-O 2.9 A and
localizes it by 0.25 A off center, a homolytic O-H stretch of water unpairs two electrons
with no formal charge, and the hydronium O-H stretch goes heterolytically (the leaving
hydrogen carries `q -> +0.8` and its capacity collapses). The solve is the dual Newton of
§2.2 on `(lambda_i, mu_frame, nu_frame)` with a bordered `(N + 2)^2` dense system per
frame; `n_i` is an inner scalar problem. Step acceptance is Armijo on the dual *or* a
residual decrease **that does not lower the dual** -- `n(mu)` is a staircase, and a
residual-only acceptance let the count multiplier cycle across an integer step. The loop
stops at `tol = 1e-8`; the two differentiable Newton steps that follow polish it to
roundoff and give exact first and second derivatives.

**Cost.** The inner problem is where the time goes: it is evaluated once per atom per
dual-state evaluation, and each of its iterations is a dozen tiny elementwise ops. Its
residual `h(n)` is a staircase too -- flat at slope ~`T` between the steps of `E0'` (at
`q = 0` and the walls at `q = +-1`), steep at them -- so plain Newton overshoots by whole
electrons from a plateau and crawls along the algebraic tail of the smoothed `|q|`, and
bisection needs forty halvings. The solve is a bracketed Newton iteration warm-started from
the previous evaluation's `n`, with the step at `q = 0` inverted in closed form (freeze the
smooth remainder `g = h + E0'` at the current point; `eta/2 q / sqrt(q^2 + eps^2) = g - chi`
gives `q`) whenever Newton would cross it or it is much the stiffer part, walls as hard
stops, and the bracket's own step (regula falsi, or bisection when the end residuals differ
by more than a decade) when a Newton step fails to shrink the residual: 5-15 residual
evaluations, against 63 for the bisection it replaced. The model's second and third
solves warm-start from the first (`warm=`), which at close parameters costs nothing beyond
the polish. Together: a 16-dimer training step went from 9.8 s to 1.3 s on one CPU core
(the film model: 0.43 s). Per state evaluation the solve is still a few thousand launches
with a host sync per Newton iteration, so on a GPU it is latency-bound whatever the batch
size; `bo_device: cpu` (the default) runs it on the host and moves the state back.

This is what makes a bond transferable across charge states: one O-H is ~-0.2 Ha in water,
hydronium and hydroxide alike (`test_charged_reference_removes_the_ionization_offset`).

**Limitation.** `E0` alone cannot *create* an ion pair from neutrals: the electrostatic
stabilization of separated ions lives in the induction solve. Two neutral fragments in one
frame therefore never autoionize, and the ion tests use one ion per frame. The fix is a
perturbative environment potential `phi_i q_i` in the count energetics, deliberately not
coupled into the bonding loop yet.

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
- `film_fit` (the shared cluster loss) runs on a pairing output with the force term;
- formal charges come out of the solve (hydronium `q_O = +1`, hydroxide `q_O = -1`, three
  and one bonds), a triplet constraint unpairs exactly two electrons, and the charged
  reference removes the ionization offset between the O-H bonds of water, H3O+ and OH-.

## 6. Not yet

- **Environment potential in the count solve.** `phi_i q_i` from the induction state, as a
  perturbative correction (the autoionization limitation of §2.4); then an SQE baseline
  from `q^f` so the partial charges of an ion are referenced to its formal charge.
- **Hybridization.** One exponent per atom; the linear combination of an s-like and a
  p-like exponent is a head change.
- **Joint solve.** Bond orders and formal counts are solved at `theta_0` and the gates
  derived from them before the induction solve (nested, both convex). A joint minimization
  where the gates depend on `p` and `J` on the induced state is non-convex and is left for
  later.
- **Kernels.** The pairing coupling goes through `rsfff.ff.backend.slater_pauli_pair_energy`,
  so the torchff Pauli kernel serves it unchanged once the port lands; the solve is torch.
- **Multiplicity.** A symmetric triplet (two equivalent bonds) splits the unpairing across
  both; the labels are for the state that unpairs one bond. Conditioning on `u_i` lets the
  heads break the symmetry, the prior cannot.
- **Data.** `scripts/pairing_scans.py` writes the rigid O-H stretches (H2O, H3O+, OH-), the
  H-O-H bend and the shared-proton scans (H5O2+, H3O2- at four O-O distances) as RKS,
  UKS-singlet and UKS-triplet sets under `qchem_roundtrip/pairing_scans*`;
  `scripts/aggregate_pairing_data.py` pools them with the monomer / cluster data into
  `data/pairing/`. RKS labels are unphysical beyond ~1.8 A on the stretches; the triplet
  minus singlet along a stretch is the training signal for `J`.

## 7. Two-stage architecture: topology, then parameters (`model.py`, `network.py`)

Nothing in the model reads the fragment assignment any more; the only inputs beyond the
geometry are the frame's total charge and multiplicity.

```
features on the total density  -> topology heads  -> (J, kappa, v0)   [state-free]
    -> electronic-state solve  -> p, c, q^f, u
features projected by c        -> parameter heads, FiLM-conditioned on [q^f, u]
    -> (J, kappa, v0, multipoles, Pauli, dispersion, SQE ...) -> second solve -> energies
```

**Stage 1** (`network.topology`): a small trunk on the *unprojected* Lambda features
(`projector.full_features`) emits a `PairingFamily` -- the couplings and capacities that
decide the Lewis structure. It is state-free by construction: it cannot know a fragment,
a charge or a spin, only the density around each atom. Its solve gives the bond orders,
the co-membership `c` (§3), the formal charges `q^f` and the unpaired counts `u`.

**Stage 2**: `projector.project(batch, c)` splits every feature block by the co-membership
(internal / environment / cross, the film's blocks with `c` in the role of `P_ij`), and
the parameter network is FiLM-conditioned on `[q^f_i, u_i]` in place of the film's
fragment key. Its `PairingFamily` is solved again (`theta_0` and `theta` as in §3) and
gives every energy. The stage-1 couplings are also what `coupling` reports.

So the answer to "what determines `n_i` without fragments" is: the stage-1 solve, from the
density alone, against the frame's electron count; and the answer to "why keep SQE
separate" is that `q^f` is the Lewis charge (integer-like, sets capacities) while the SQE
charge is bond polarity (continuous, sets electrostatics) -- the same atom carries both.
