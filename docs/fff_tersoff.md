# The Tersoff Model: an Explicit Bond Order with Emitted Parameters

## Design note and implementation plan (`rsfff.ff.tersoff`, branch `tersoff`)

## 1. Why

The `nonreactive` result says something general: starting from a completely uncoupled monomer
potential, the parameter network turns a simple functional form into an almost exact coupled
one. The functional form only has to be *locally* expressive; the network supplies the
environment dependence. The `pairing` branch makes the bonded description reactive by making
the bond order a variational state -- a strictly convex functional, a dual Newton solve, exact
implicit derivatives, formal charges solved jointly. That is the principled route, and it
costs a solve: dense per-frame factorizations, warm-start caches, fused inner kernels, and
still ~1.3 s against the film's 0.43 s on a 16-dimer step.

This branch asks the ReaxFF / Tersoff / REBO question: **is the solve necessary, or is a
closed-form bond order with network-emitted parameters enough?** It keeps everything the
pairing branch built -- the valence polytensor coupling `J_ij`, the pairing hardness `kappa`,
the capacity `v_i`, the co-membership `c_ij` that replaces every fragment gate, the two-stage
topology -> parameter network -- and replaces the one thing in the middle, the minimization,
with an explicit formula. It is therefore a strict ablation of the pairing model: same `J`,
same heads, same assembly, same tests, no solve. If it matches, the solve was unnecessary; if
it does not, the failure shows exactly what the constraint was buying.

The cost of the explicit form is that valence conservation becomes a property of the *formula*
instead of a constraint the network cannot violate. The network can cheat where the data do
not punish it, and it will do so at transition states. The plan below is built around the
diagnostics that catch that.

## 2. The functional forms

All quantities live on the candidate pairs within `pairing_cutoff` (4 A), exactly as in
`rsfff.ff.pairing`. `J_ij` is the bare pairing energy (`slater_pauli_pair_energy` on the
valence polytensor, `PairingHeads`), `kappa_ij = sqrt(kappa_i kappa_j)`, `v_i` the capacity.

### 2.1 Raw bond order

The unconstrained stationary point of the barrier-free pairing functional, ``kappa p = J``,
clipped to the unit interval:

```
b_ij = clip( J_ij / kappa_ij, 0, 1 )      smoothed over w = T / kappa (softplus corners)
```

`b = 1/2` at `J = kappa/2`, where the pairing model's attractive branch turns over, and `b = 1`
for `J >= kappa`. Because `J` is the Slater-overlap operator, the radial shape, the exponential
decay and the anisotropy (dipole and quadrupole ranks: bond angles, and later sigma vs pi) are
inherited unchanged; nothing is switched by distance. The pairing model's `T` is the smoothing
width, so `1 - p ~ exp(-(J - kappa)/T)` on a bond as there. (The first draft had a sigmoid
surrogate here; the clipped line *is* the stationary point, and it is what the water filling
below needs.)

### 2.2 Saturation: the explicit forms

The raw bond order does not know about capacity. At the pairing priors a hydrogen-bonded
`O...H` pair at 1.9 A has `J ~ 0.14 Ha` against `kappa/2 = 0.1`, so `b ~ 0.7`, and the 1-3
`H-H` pair of a water has `J ~ 0.13`, `b ~ 0.6`: **the unconstrained pairing model gives the
same** (`p = J/kappa`), and it is the valence constraint alone that squeezes them below 1e-2
(`test_valence_competition`). The saturation rule is therefore the whole model, and intact
water -- not a reaction -- is the first test: the covalent `O-H` must keep `p = 1` while the
1-3 pair and the hydrogen bond get `~0`, or the film accounting (elst/disp weighted by
`1 - c`) breaks.

Two rules, `tersoff_saturation`:

**`rebo` -- multiplicative (Tersoff / REBO).** Each atom scales all its bonds by one factor:

```
N_i = sum_j b_ij,        s_i = v_i / smoothmax(v_i, N_i),        p_ij = b_ij s_i s_j
```

`s_i = 1` below capacity, `v_i / N_i` above it, so `sum_j p_ij <= v_i` by construction. This
is the form the question asked for, and the ablation. Tersoff's own `(1 + zeta^n)^(-1/2n)`
has the same limits but is `2^(-1/2n)` *at* capacity, which is exactly where every bond of an
intact molecule sits; since `p = 1` has a meaning here (it is the co-membership), the factor
is one up to the capacity and bends only above it, over `tersoff_saturation_width` electrons.
Its weakness is *how* it shares: linearly. **Result at the priors:** it fails intact water
outright, not just the dimer -- with the 1-3 and hydrogen-bond raw orders above one half,
the water dimer's covalent bonds come out at `p = 0.3-0.46` and every non-bonded pair within
2 A carries `0.1-0.2` (`test_dimer_discriminator` records it). Whether the topology network
can rescue it by pushing `J` on those pairs below `kappa/2` is the "wider scope of simple
forms" question; at the priors the answer is that the raw order with the pairing priors is
far too soft for a linear sharing rule, and the pairing solve's competition was doing all
the work.

**`waterfill` -- per-atom energetic sharing (default).** The pairing functional restricted
to one atom, `min sum_j [-J_ij p_j + kappa p_j^2 / 2]` subject to `sum_j p_j <= v_i`,
`0 <= p_j <= 1`, has the water-filling solution

```
p_i(j) = clip( (J_ij - lambda_i) / kappa_ij, 0, 1 ),      lambda_i >= 0 fills the capacity
```

`lambda_i = 0` when the raw orders fit. Competition is energetic: a partner whose `J` falls
below `lambda_i` gets (smoothly) zero, so the dimer's hydrogen bond and the 1-3 pairs are
squeezed out the moment the covalent bonds saturate, and two equal partners split `v_i`
evenly with a transition width set by `kappa`, not `T`, which is what gives the Zundel
midpoint its `0.5 / 0.5`. The two ends are reconciled by the smaller grant,

```
p_ij = smoothmin( p_i(j), p_j(i) )
```

which is the dual solution whenever one end is the binding one and the even split of a shared
proton when both are. The smooth minimum `(x+y)/2 - sqrt(((x-y)/2)^2 + eps^2) + eps` is exact
on `x = y` (every bond of an intact molecule) and at most `eps = tersoff_reconcile_width`
(2e-3) above the true minimum elsewhere, so `sum_j p_ij <= v_i + g_i + deg_i eps` with `g_i`
the filling residual. No multiplicative factor is applied on top: any smooth `min(1, v/N)`
bends *at* capacity.

*The root.* `S_i(lambda) = sum_j clip((J_ij - lambda)/kappa_ij)` is monotone and, unsmoothed,
piecewise linear, so its root is found in a fixed number of steps without a convergence test:
Newton on the unsmoothed filling, exact on a linear piece, with a jump to the nearest
breakpoint on a flat one (the saturated partner that unsaturates first going up, the empty one
that fills first going down), a bracket as the safeguard, and the *bottom* of a flat root
interval when the capacity is met by saturated partners alone (`tersoff_fill_steps = 8`, no
graph). Then `tersoff_polish_steps = 4` differentiable Newton steps on the smoothed filling in
the pairing solve's log form (`ln fill - ln room`, `room` cancellation-free as
`(v - n_sat) + sum_sat (1 - p)`), each tried against its two bisections toward the current
point and the plain step, keeping the candidate with the smallest residual: the hard root
leaves a squeezed partner exactly on its breakpoint, where the smooth root is a balance of
exponential tails on which the plain step moves one `T` per iteration and the log form is
exact. At a root the Newton map has zero derivative in `lambda`, so these steps carry the
implicit first and second derivatives (`gradcheck` / `gradgradcheck` in
`tests/tersoff/test_tersoff_bond_order.py`). Water clusters and the ions converge to
`< 1e-4`; the tolerance for `converged` is `10 T`.

*What it does not do.* Each atom fills as if its partners were unconstrained. When both ends
of a pair are binding and disagree, the smaller grant wins and the other atom is left
*under* capacity: on the Zundel scan at the priors the shared proton's two orders sum to
0.86-0.95 between the midpoint and 0.15 A off it, where the pairing solve keeps them at 1.
That under-filling, together with the closed-form charge hand-over below, puts a hump of
~0.07 Ha on the explicit model's proton-transfer curve at the priors against ~0.03 Ha for the
pairing model (both untrained). `tersoff_dual_sweeps` (Jacobi sweeps of the coupled dual,
each atom refilling against `J - lambda_partner`) was written to close that gap and is left
in as **experimental**: one or two sweeps do shrink the under-filling, but the damped Jacobi
iteration is not monotone and can leave an atom on the wrong side of a breakpoint; `0` is
the default and the only tested setting. The proper rung above `waterfill` is the pairing
solve itself, which is the point of the ablation.

### 2.3 Energy

The same functional, evaluated at the explicit `p` instead of its minimizer:

```
E_pair = sum_ij [ -J_ij p_ij + kappa_ij p_ij^2 / 2 ]  +  sum_i E0_i(q_i)
```

The `kappa` term is kept so a saturated bond is `-J + kappa/2` and the pyCMM-calibrated
priors (`b_pair = 0.834 / bohr`, `kappa = 0.2 Ha`: the O-H `r_eq` and `D`) carry over
unchanged. The entropic barriers are dropped (they exist to make the *solve* smooth; the
explicit `p` is smooth by construction). `E_pair` is not stationary in `p`, so forces carry
`dp/dR`; that is ordinary autograd through an explicit graph and needs no implicit-function
machinery -- double backward for the force loss comes for free.

**Overbinding penalty.** With `rebo` and `waterfill` the capacity bound is built into `p`, so
no penalty is needed for what the question called "overbinding certain atoms". A ReaxFF-style
term on the *raw* coordination, `E_over = sum_i kappa_i softplus(sum_j b_ij - v_i)^2`, is kept
behind `tersoff.overbinding_penalty` (default off) as the ablation that tests whether a soft
penalty on the unsaturated bond order helps the network learn to keep `J` honest.

### 2.4 Formal charges and capacity, in closed form

The pairing model solves the formal electron count `n_i` jointly with the bond orders; that is
where hydronium's oxygen gets capacity 3 and hydroxide's gets 1. The explicit model needs the
same information without the solve (`formal_charge.py`):

```
q~_i   = zero-initialized readout of the family latent (per species; TersoffHeads.q_mlp)
q_i    = q~_i + w_i (Q_frame - sum_k q~_k) / sum_k w_k
v_i    = capacity(n0_i - q_i)          (the pairing branch's capacity polynomial, unchanged)
```

with the projection weights `w` by `tersoff_formal_charge`: `heavy_atoms` (one on heavy atoms,
zero on H), `uniform`, or the default **`overload`** -- the charge follows the bonding, once:
a first pass with the heavy-atom weights gives bond orders `p0`, and the weights of the second
pass are the atoms' over- (`Q > 0`) or under-coordination (`Q < 0`) in that state past half an
electron, `w_i = softplus((+-(sum_j p0_ij - v0_i) - 1/2) / width)`. That is ReaxFF's
over-coordination `Delta_i` used as a charge prior. At initialization (`q~ = 0`): H3O+ gives
`q_O = 0.996`, `v_O = 3`; OH- `q_O = -0.999`, `v_O = 1`; a Zundel frame splits `0.49 / 0.49`
at the midpoint and hands `0.92` of the charge -- and the third capacity -- to the oxygen the
proton sits on 0.1 A off it (the heavy-atom prior alone leaves `2.5 / 2.5` wherever the
proton is: `test_zundel_hands_the_charge_over`). The readout lets the network move charge
onto or off a hydrogen (a leaving proton) as the data demand. `u_i = v_i - sum_j p_ij`;
`(q_i, u_i)` condition the stage-2 network exactly as in the pairing model, and
`ChargedAtomicReference(q)` gives `E0_i(q_i)` so an O-H bond costs the same ~-0.2 Ha in
water, H3O+ and OH- (`test_charged_reference_removes_the_ionization_offset`). The
multiplicity constraint has no explicit analogue; a high-spin frame enters through the `u`
conditioning only.

This is the one place the explicit model is *less* principled than ReaxFF-with-EEM would be,
and the diagnostic is direct: `q_O` and `v_O` along the Zundel and Eigen scans and along the
hydronium O-H stretch (heterolytic: the proton must carry `q -> +1` and `v_H -> 0`).

### 2.5 Co-membership and assembly

Unchanged from `rsfff.ff.pairing.model`: `c = 1 - (1 - c12)(1 - c13)` from `p` over 1-2 and
1-3 paths (`comembership_from_bond_order`), every classical channel weighted `(1 - c)`, Pauli
ungated on bonds as the repulsive wall, `range_gate: bond_order`, four exact energy buckets,
`E_pair(theta) - E_pair(theta_0)` in the induction channel. The `nonreactive` branch's
`film.nonbonded: exclusions` is the *fixed-topology* limit of this accounting (`c` one-hot on
the covalent graph), which is the right first regression test.

## 3. Implementation (on the branch, 2026-09-25)

`pairing` is merged into `tersoff` (clean; the branch carries both the exclusions film and the
pairing package), and everything below subclasses `rsfff.ff.pairing`, so the two models stay
in lock-step and share tests.

| file | contents |
|---|---|
| `src/ff/tersoff/bond_order.py` | `raw_bond_order`, `saturate_rebo`, `saturate_waterfill` (root finder, sweeps, polish), `overbinding_penalty`, `explicit_state(...) -> ElectronicState` (the pairing container, so `PairingOutput`, `train_film` diagnostics and `pairing_plots.py` read it unchanged; `lam` = the filling multiplier, `mu`/`nu` zeros, `residual` the per-frame max filling residual), `tersoff_state_energy` (`-J p + kappa p^2/2`, `E0(q)`; no entropic terms). |
| `src/ff/tersoff/formal_charge.py` | `project_formal_charge` (heavy_atoms / uniform / weights), `overload_weights`. |
| `src/ff/tersoff/heads.py` | `TersoffHeads(PairingHeads)` + the zero-initialized `q_mlp`; `TersoffFamily(PairingFamily)` + `q_raw`. |
| `src/ff/tersoff/model.py` | `TersoffModel(PairingModel)`: overrides `_pairing` (coupling -> formal charge -> explicit state -> energy, with the optional overbinding penalty), stubs the warm-start cache; `forward` stashes the batch's atomic numbers for the projection and is otherwise inherited. |
| `src/train/build_tersoff.py` | `build_tersoff_model`: the pairing builder (now taking `model_cls` / `heads_cls` / their kwargs) with the `tersoff_*` config fields. |
| `src/train/build_pairing.py` | `MODEL_BUILDERS = {film, pairing, tersoff}`; `build_model` dispatches on it (the single point `train_film.py` and `md/film_driver.py` use). |
| `src/train/config.py` | `film.model: tersoff`; `tersoff_saturation`, `tersoff_saturation_width`, `tersoff_fill_steps`, `tersoff_formal_charge`, `tersoff_formal_charge_width`, `tersoff_reconcile_width`, `tersoff_dual_sweeps`, `tersoff_formal_charge_readout`, `tersoff_overbinding_penalty`. |
| `configs/water_tersoff.yaml`, `configs/ion_tersoff.yaml` | the pairing configs with the model swapped (same streams, labels, splits). |
| `tests/tersoff/` | `tersoff_helpers.py` (model builders, a hydrogen-bonded dimer, the ions, a Zundel frame with a movable proton), `test_tersoff_bond_order.py` (8: raw order, exact one-atom filling, the squeeze, the even split, the rebo bound, the capacity bound on a random graph, gradcheck/gradgradcheck for both rules), `test_tersoff_model.py` (15: intact water = film accounting, the dimer discriminator, capacity everywhere, vertex, exact bucket sum, FD forces for both rules, FD force-loss gradient, stretched bond, formal charges under both projections, the Zundel hand-over, the ionization offset, agreement with the pairing model on intact water, `film_fit` smoke, config round-trip). |

Cost: on an 8-water cluster on one CPU core, energy + forces take the same 0.17 s as the
warm-started pairing model; the explicit state is a few dozen scatters. The comparison that
matters is a training step on Perlmutter (M3).

## 4. Milestones

- **M0 -- scaffold: done** (merge, package, dispatch, configs, 23 tests).
- **M1 -- the dimer: decided at the priors.** `waterfill` reproduces the film accounting on
  intact water and the dimer exactly (`p(O-H) = 1`, `p(O...H) < 1e-4`); `rebo` fails intact
  water (§2.2). Still to do: a short `water_tersoff.yaml` fit with each rule to see whether
  the topology network closes `rebo`'s leak -- the direct measurement of how much scope the
  network buys a simple form.
- **M2 -- ions at the priors: done in tests, figures pending.** H3O+ / OH- capacities and
  bond orders match the pairing solve; the Zundel proton splits evenly at the midpoint and
  hands its charge over off it, with the under-filling hump of §2.2 as the known difference
  from `pairing`. Next: `notebooks/tersoff_plots.py` and the stretch / bend / PT figures
  against `notebooks/figures/pairing_*`.
- **M3 -- training**: `water_tersoff` -> `ion_tersoff` -> `tersoff_all` on the pairing data
  (`data/pairing`, `ion_pairing`; the RKS / UKS singlet / UKS triplet scans; nothing new to
  label for single bonds). Compare energy / force / EDA errors and the scan figures to
  `pairing_all` at equal parameter count. Timing target: film-model step time.
- **M4 -- pi bonds** (separate card once M3 is in hand). The directional coupling already
  exists: the valence polytensor's dipole and quadrupole ranks give `J` its angular
  dependence, so a double bond is a rank-2 contribution and needs no second radial `r0`
  the way ReaxFF's `BO_pi` does. Data: C-C stretches of ethane / ethylene / acetylene
  (does one `J` give 1, 2, 3?), the ethylene torsional barrier (the one thing a
  distance-only bond order cannot produce), formaldehyde and CO2 for heteroatom multiple
  bonds, benzene / allyl for fractional orders. Label with the existing `qchem_roundtrip`
  force template; a per-atom `hybridization` head (two exponents, s- and p-like -- the
  pairing branch's open item) is the model change if one exponent cannot fit the series.

## 5. What this branch does not do

- No joint bonding-induction solve (neither does `pairing`).
- No multiplicity constraint: `2S` enters only through the `u` conditioning.
- No electrostatic environment in the formal charge: the readout must learn autoionization
  from data (the pairing branch's `phi_i q_i` item applies here too).
- No kernels: `J` already goes through `slater_pauli_pair_energy`, so the torchff Pauli kernel
  serves it once the port lands; the explicit bond order is a handful of scatters.

## 6. Decision summary

- Same coupling, heads, co-membership, assembly and tests as `pairing`; the solve becomes a
  formula. The branch is an ablation, and it is built as one (subclass, shared tests).
- Raw bond order is the clipped `J / kappa` -- a function of the *coupling*, not of the
  distance -- so radial shape and anisotropy come from the polytensor overlap.
- Saturation is the model: `rebo` (asked for; linear sharing) and `waterfill` (energetic
  sharing, exact per atom) both implemented; at the priors intact water decides for
  `waterfill`, and `rebo` stays as the ablation.
- Capacity bound built into `p`; the overbinding penalty is an ablation, not a crutch.
- Formal charges in closed form: the charge follows the first pass's over-coordination
  (ReaxFF's `Delta_i` as a prior) plus a learned readout; the Zundel / Eigen / stretch scans
  are the honesty check.
- Single bonds on the existing pairing data first; pi bonds are a follow-up card on the
  polytensor's rank-2 coupling, with the ethylene torsion as the discriminating test.
