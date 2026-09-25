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

The unconstrained stationary point of the pairing functional, `kappa p + T logit(p) = J`,
has no closed form; its sigmoid surrogate with the same slope at `p = 1/2` is

```
b_ij = sigmoid( (J_ij - kappa_ij / 2) / T_b ),      T_b = T + kappa_ij / 4
```

`b -> 1` where `J >> kappa/2` (a bond), `b -> 0` where `J << kappa/2`, and the crossover at
`J = kappa/2` is where the pairing model's attractive branch turns over. Because `J` is the
Slater-overlap operator, the radial shape, the exponential decay and the anisotropy (dipole
and quadrupole ranks: bond angles, and later sigma vs pi) are inherited unchanged; nothing is
switched by distance. A plain sigmoid in `r` with emitted `(r0, w)` is kept as the ablation
`raw: radial` for the "how simple can it be" question, but it is not the default: it throws
away the directional information the polytensor already carries.

### 2.2 Saturation: the three explicit forms

The raw bond order does not know about capacity. At the pairing priors a hydrogen-bonded
`O...H` pair at 1.9 A has `J ~ 0.07 Ha` against `kappa/2 = 0.1`, so `b ~ 0.35`: **the
unconstrained pairing model gives the same** (`p = J/kappa`), and it is the valence constraint
alone that squeezes it below 1e-2 (`test_valence_competition`). The saturation rule is
therefore the whole model, and the water *dimer* -- not the reaction -- is the first test:
the covalent `O-H` must keep `p = 1` and the hydrogen bond must get `~0`, or the film
accounting (elst/disp weighted by `1 - c`) breaks on intact water.

Three rules, selected by `tersoff.saturation`, in increasing fidelity to the constraint:

**`rebo` -- multiplicative (Tersoff / REBO).** Each atom scales all its bonds by one factor:

```
N_i = sum_j b_ij,        s_i = ( 1 + (N_i / v_i)^m )^(-1/m),        p_ij = b_ij s_i s_j
```

`s_i -> 1` below capacity, `v_i / N_i` above it, so `sum_j p_ij <= s_i N_i <= v_i` by
construction (`m ~ 6-8`, smooth). This is what the question asked for and the cheapest
form. Its known weakness is *how* it shares: linearly. The water-dimer hydrogen with
`b = 1` on its bond and `0.35` on the acceptor gets `s_H = 0.78`, the covalent bond drops
to `0.78` and the hydrogen bond keeps `0.27`. Expect it to fail the dimer test at the
priors; whether the topology network can rescue it by lowering `J` on the acceptor pair is
exactly the "wider scope of simple forms" hypothesis, and the dimer test measures it.

**`waterfill` -- per-atom energetic sharing (default).** The pairing functional restricted
to one atom, `min sum_j [-J_ij p_j + kappa p_j^2 / 2]` subject to `sum_j p_j <= v_i`,
`0 <= p_j <= 1`, has the water-filling solution

```
p_i(j) = clip( (J_ij - lambda_i) / kappa_ij, 0, 1 ),     sum_j p_i(j) = min(v_i, sum_j b_ij)
```

with `lambda_i >= 0` the one scalar per atom that fills the capacity (a monotone 1-D root,
taken as `K = 3` Newton steps from `lambda = 0` -- fixed cost, unrolled, no convergence
test; the clip is the pairing model's smooth `T`-barrier so `p` is C2). Competition is
energetic: a partner with `J` below `lambda_i` gets exactly (smoothly) zero, so the dimer's
hydrogen bond is squeezed out the moment the covalent bond saturates, and two equal partners
split `v_i` evenly with a transition width set by `kappa`, not `T`, which is what gives the
Zundel midpoint its smooth `0.5 / 0.5`. The two ends are reconciled with a soft minimum,

```
p_ij = softmin_T( p_i(j), p_j(i) )
```

so both capacities hold: `sum_j p_ij <= sum_j p_i(j) <= v_i`. This is the explicit model's
analogue of the dual solve with the coupling `lambda_i + lambda_j` dropped; it is exact for an
isolated atom and for any pair whose two ends agree.

**`dual_k` -- K unrolled dual steps (ablation toward pairing).** `p_ij = sigmoid((J_ij -
lambda_i - lambda_j - kappa p_ij) / T)` with `lambda` updated by `K` diagonal-Jacobian Newton
steps of the marginal residual, then the `rebo` normalization as a final guarantee. Not
implemented first; it is the rung between `waterfill` and the solve if `waterfill` falls
short at transition states.

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
same information without the solve. Rule (`formal_charge.py`):

```
q~_i   = zero-initialized readout of the topology latent (per species)
q_i    = q~_i + w_i (Q_frame - sum_k q~_k) / sum_k w_k,     w_i = 1 on heavy atoms, 0 on H
v_i    = capacity(n0_i - q_i)          (the pairing branch's capacity polynomial, unchanged)
```

At initialization the frame charge sits on the heavy atoms: H3O+ gives `v_O = 3`, OH- gives
`v_O = 1`, a Zundel frame gives each oxygen `2.5` -- the same starting point the pairing
model's solve reaches at its priors -- and the readout lets the network move charge onto or
off a hydrogen (a leaving proton) as the data demand. `u_i = v_i - sum_j p_ij` is the
unpaired count. `(q_i, u_i)` condition the stage-2 network exactly as in the pairing model,
and `ChargedAtomicReference(q)` gives `E0_i(q_i)` so an O-H bond costs the same ~-0.2 Ha in
water, H3O+ and OH-. The multiplicity constraint has no explicit analogue; a high-spin frame
enters through `u` conditioning only (as it does on `pairing` today).

This is the one place the explicit model is *less* principled than ReaxFF-with-EEM would be,
and the diagnostic is direct: plot `q_O` and `v_O` along the Zundel and Eigen scans and along
the hydronium O-H stretch (heterolytic: the proton must carry `q -> +1` and `v_H -> 0`).

### 2.5 Co-membership and assembly

Unchanged from `rsfff.ff.pairing.model`: `c = 1 - (1 - c12)(1 - c13)` from `p` over 1-2 and
1-3 paths (`comembership_from_bond_order`), every classical channel weighted `(1 - c)`, Pauli
ungated on bonds as the repulsive wall, `range_gate: bond_order`, four exact energy buckets,
`E_pair(theta) - E_pair(theta_0)` in the induction channel. The `nonreactive` branch's
`film.nonbonded: exclusions` is the *fixed-topology* limit of this accounting (`c` one-hot on
the covalent graph), which is the right first regression test.

## 3. Implementation

### 3.0 Bring in the pairing scaffold

`tersoff` is branched from `nonreactive` (hard exclusions, latest committees). The pairing
package it needs lives on `pairing`, which already merged `nonreactive` up to `c5a3170`; the
only `nonreactive` commits it lacks are the exclusions commit and notebook/benchmark work.
**Step 0 is `git merge pairing` into `tersoff`** (expected conflicts: `src/train/config.py`
and `src/train/train_film.py`, both touched by the exclusions commit and by the pairing
dispatch; trivial to resolve). Everything below then builds on `src/ff/pairing` by subclassing,
so the two models stay in lock-step and share tests.

### 3.1 New package `src/ff/tersoff/`

| file | contents |
|---|---|
| `bond_order.py` | `raw_bond_order(J, kappa, T)` (sigmoid surrogate; `radial` ablation); `saturate_rebo(b, v, pairs, m)`; `saturate_waterfill(J, kappa, v, pairs, K, T)`; `softmin`; `explicit_state(J, kappa, v, sub_pairs, n_atoms, ...) -> ExplicitState(p, u, valence, lam)`; `overbinding_penalty(b, v)`. Pure torch, batched over frames by `index_add_`; no dense matrices. |
| `formal_charge.py` | `project_formal_charge(q_raw, species, batch_idx, total_charge)` (heavy-atom projection), `capacity_from_charge(tables, q)` reusing `pairing.electronic_state.capacity`. |
| `heads.py` | `TersoffHeads(PairingHeads)`: adds the zero-initialized `formal_charge` readout on the topology latent; everything else inherited (`q_v, b, mu_v, quad_v, kappa, v0`). |
| `model.py` | `TersoffModel(PairingModel)`: overrides `_pairing` to return `(ExplicitState, J, kappa, e_pair, e_atom)` from the closed form, drops the state cache / `bo_device` / warm-start plumbing (`_warm_start`, `_store_state` become no-ops), reports `bo_solver=None`. `forward` inherited: topology pass -> `c` -> projected features -> stage-2 heads -> `theta_0` and `theta` evaluations -> the same four buckets. |
| `__init__.py` | exports. |

`ElectronicState` is reused as the state container (`p, u, q, valence` filled; `n, lam, mu,
nu, n_iter, residual` carry the closed-form `lambda` or `None`) so `PairingOutput`,
`train_film.py` diagnostics (`bo_frac`, `qf_max`) and `notebooks/pairing_plots.py` read it
without change.

### 3.2 Training and dispatch

- `src/train/build_tersoff.py`: copy of `build_pairing.py` with the `tersoff:` config block
  (`raw: coupling|radial`, `saturation: rebo|waterfill`, `m`, `K`, `T`,
  `overbinding_penalty`, `formal_charge: heavy_atoms|readout`).
- `src/train/config.py`: `film.model: tersoff` (+ the `tersoff_*` fields next to the
  `pairing_*` ones). The dispatch already lives in one place, `build_pairing.build_model`
  (imported by `train_film.py` and `md/film_driver.py`); turn it into a
  `{film, pairing, tersoff}` registry there so nothing else changes.
- `configs/water_tersoff.yaml`, `configs/ion_tersoff.yaml`, `configs/tersoff_all.yaml`:
  the three pairing configs with the model swapped (same streams, labels, splits).

### 3.3 Tests (`tests/tersoff/`)

Copy `tests/pairing/pairing_helpers.py` and run the pairing model's invariants against the
explicit state, plus the ones that only make sense here:

1. **Intact water = film accounting**: `c = 1` on bonds and 1-3 pairs, `< 1e-6` elsewhere;
   `E_cross = 0`; isolated-fragment vertex (zero induction, zero inter channels).
2. **Water dimer (the discriminator)**: `p(O-H covalent) > 0.999`, `p(O...H) < 1e-2` at the
   priors under `waterfill`; recorded (not asserted) under `rebo`.
3. **Capacity**: `sum_j p_ij <= v_i + 1e-9` for every atom on every test frame, both rules.
4. **Formal charges**: H3O+ `q_O = +1`, `v_O = 3`, three bonds; OH- `q_O = -1`, one bond;
   Zundel midpoint `2.5 / 2.5` and a symmetric `p` split.
5. **Smoothness**: `gradcheck` / `gradgradcheck` of `explicit_state` (both rules) in `J`,
   `kappa`, `v`; central-difference forces through the whole model; finite-difference
   gradient of a force loss with respect to a pairing parameter.
6. **Stretched bond**: `c` drops below 0.5 and the energy rises; hydronium O-H stretch is
   heterolytic (`q_H -> +1`), water O-H homolytic (`u -> 2`, `q = 0`).
7. `film_fit` smoke on a `TersoffModel` output with the force term.

### 3.4 Diagnostics

`notebooks/tersoff_plots.py` (module; the notebook only calls it) extends
`notebooks/pairing_plots.py` with a model argument so every figure in
`notebooks/figures/pairing_*` is produced for `film`, `pairing` and `tersoff (rebo)`,
`tersoff (waterfill)` on the same axes: O-H stretches (H2O, H3O+, OH-), HOH bend, the
H5O2+ / H3O2- proton-transfer scans with `p_ij`, `sum_j p_Hj`, `q_O`, `v_O` along the
coordinate, and the dimer `p(O...H)` against O-O distance. Those plots, at the priors and
after training, are the deliverable of the viability test.

## 4. Milestones

- **M0 -- scaffold** (step 0 merge; package skeleton; `TersoffModel` running on
  `water_tersoff.yaml` with `rebo`; tests 1, 3, 5 pass). Expect intact water to reproduce
  the film accounting immediately, since `J >> kappa/2` on a bond gives `b = 1` to double
  precision.
- **M1 -- the dimer** (`waterfill`; tests 2, 5; dimer figure). Decides the default rule.
  If `rebo` leaks bond order into the hydrogen bond at the priors, train a short
  `water_tersoff.yaml` with each rule and check whether the topology network learns to close
  the leak; that is the direct measurement of how much scope the network buys a simple form.
- **M2 -- ions at the priors** (`formal_charge.py`; tests 4, 6; stretch and proton-transfer
  figures against the pairing figures). Success = same qualitative curves as `pairing`
  (smooth stretches, heterolytic vs homolytic asymptotes, Zundel hand-over) with no solve.
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
- Raw bond order is a sigmoid of the *coupling*, not of the distance, so radial shape and
  anisotropy come from the polytensor overlap.
- Saturation is the model: `rebo` (asked for; linear sharing) and `waterfill` (energetic
  sharing, exact per atom) both implemented; the water dimer decides.
- Capacity bound built into `p`; the overbinding penalty is an ablation, not a crutch.
- Formal charges in closed form with a heavy-atom prior; the Zundel / Eigen / stretch scans
  are the honesty check.
- Single bonds on the existing pairing data first; pi bonds are a follow-up card on the
  polytensor's rank-2 coupling, with the ethylene torsion as the discriminating test.
