# Plan: Charge-Aware Embeddings with Self-Consistent Charge Equilibration

## Goal

Test whether a charge-dependent atomic embedding, combined with an inner-loop
minimization of the total energy over atomic charges, can simultaneously
reproduce (i) energies/forces and (ii) response properties — dipoles, dipole
derivatives, and polarizabilities — for H2O and H3O+ using shared per-element
parameters. This validates the short-range embedding + parameter-prediction
half of the range-separated MLIP/FF architecture before any FF code is added.

## Scope

- **In scope:** charge-aware embedding, E(total; {q}) with charge SCF,
  response-parameter heads, loss design, integer-charge anchoring of E(q).
- **Out of scope (this phase):** long-range FF / Ewald, range separation,
  dissociation & piecewise-linear E(q) regularization, spin.

## Data

1. Existing: H2O and H3O+ configurations with energies, forces,
   polarizabilities, dipoles, dipole derivatives.
2. **To add (required):** isolated-species references at the same level of
   theory: H, H+, O, O-, OH, OH-, and vertical IP/EA of H2O (or H2O+ / H2O-
   single points). These anchor E(q) at integer charges.
3. Units and reference convention: all energies relative to free neutral
   atoms; document sign/units conventions in one place.

## Model components

### 1. Charge-aware species embedding
- Base embedding `z_i = f(element_i, q_i)` where `q_i` is a continuous
  per-atom charge variable. Implementation suggestion: element embedding
  concatenated with a small MLP encoding of q (or FiLM-style modulation of
  the element embedding by q). Must be smooth in q.
- **Free-atom limit constraint (architectural):** an atomic-energy head
  applied to the layer-0 embedding `z_i` alone must reproduce isolated
  atom/ion energies E_elem(q) at integer q (train on the isolated-species
  data; exact free-atom energies for the neutral atoms may be hard-coded as
  an offset). These embeddings when fed into the parameter predictor should also produce the free atom polarizabilities.

### 2. Environment update
- Existing featurization / message passing updates embeddings in the local
  environment: `h_i = MP(z_i, {z_j, r_ij})`. The q-dependence of z means the
  environment-updated features are charge-aware throughout.

### 3. Heads
- **Energy:** E_sr = sum_i E_atom(h_i) (atomic readout is fine for this
  phase; keep the option of a pairwise readout later).
- **Charges/multipoles:** the charge variables q_i themselves are the atomic
  charges (do not add a separate predicted-charge head). Optional atomic
  dipole head mu_i(h_i) for off-atom charge density; molecular dipole is
  mu = sum_i q_i r_i + sum_i mu_i.
- **Response:** atomic polarizability head alpha_i(h_i) (rank-2 symmetric tensor). Molecular polarizability assembled from atomic
  polarizabilities via
  non-interacting sum alpha_mol = sum_i alpha_i. This is an effective short-range tensor. The long-range interactions will be added later.

### 4. Charge SCF (inner loop)
- For each configuration: minimize E_total(R; {q}) over {q} subject to
  sum_i q_i = Q_total (known per system: 0 for H2O, +1 for H3O+).
- Implementation: projected gradient descent or L-BFGS on the
  charge-conservation hyperplane; fixed small iteration budget with
  convergence check. Initialize from element-default charges (phase 1) and
  from previous minibatch/MD values where applicable.
- **Gradients:** differentiate through the solution with the implicit
  function theorem (stationarity condition dE/dq = lambda * 1), not by
  unrolling. Verify force consistency numerically (see tests).

## Losses

Weighted sum; report each term separately:
- Energy MSE and force MSE (forces at the SCF solution; check
  Hellmann-Feynman-style simplification is valid at stationarity).
- Dipole MSE using mu = sum q_i r_i (+ atomic dipoles if enabled).
- Dipole-derivative MSE: d(mu)/dR via autodiff through the SCF — this
  supervises charge flux dq/dR and is the key signal tying q to physical
  charge.
- Polarizability MSE: preferred implementation computes alpha as the second
  derivative of E w.r.t. an applied uniform field E_ext (add -mu . E_ext to
  the energy and differentiate through the SCF); compare against the
  atomic-alpha assembly head. Consistency between the two is itself a
  diagnostic.
- Isolated-species energy loss at integer charges (strong weight).
- Light L2 regularization on q deviations from sensible ranges (weak; do
  not fight the SCF).

## Milestones & tests

1. **M0 — plumbing:** SCF converges on single molecules; gradient check of
   implicit differentiation vs. finite differences (energies, forces,
   d mu/dR) to <1e-5 relative.
2. **M1 — integer anchors:** free atoms/ions reproduced through the
   layer-0 head; E_elem(q) curves plotted per element.
3. **M2 — fit:** train on mixed H2O + H3O+ set. Targets: force RMSE in the
   usual MLIP range for the featurization used; dipole and dipole-derivative
   errors within a few percent; polarizability within ~5-10% (isotropic part
   first).
4. **M3 — charge sanity:** q response to an applied
   field has the right sign/magnitude; H3O+ excess charge distributes
   plausibly over H atoms.
5. **Ablation:** same model with q frozen at zero (no SCF) — quantifies what
   the charge variable buys, especially for dipole derivatives and the
   H2O/H3O+ joint fit.

## Deliverables

- Modules: `charge_embedding.py`, `charge_scf.py`, `response_heads.py`,
  `losses.py`, plus config + training script integrating with the existing
  featurization/MLIP code.
- Test suite covering M0 gradient checks and charge-conservation invariants
  (translation invariance of mu for neutral systems; correct origin
  dependence for charged systems — report dipoles of H3O+ at the center of
  mass and state the convention).
- Short report: metrics per milestone, E_elem(q) plots, ablation table.

## Known deferrals (do not implement now, keep interfaces open)

- Piecewise-linear / soft-min-over-charge-sector E(q) structure and
  dissociation data.
- Long-range FF continuation (the damped Coulomb kernel g and alpha, C6
  heads are the future interface points).
- Anisotropic polarizabilities, atomic dipole/quadrupole heads, spin.