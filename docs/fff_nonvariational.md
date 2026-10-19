# Non-variational polarization: a fixed number of iterations with learned mutual damping

Design note for the `nonvariational` model (`film.model: nonvariational`, branch `nonreactive`).
It is the film model of `docs/fff_film.md` with one thing replaced: the converged coupled
solve (`rsfff.ff.coupled_solve.pcg` + its adjoint) becomes a **fixed, unrolled** iteration whose
induced–induced coupling is evaluated with a **learned short-range damping**. The energy is
always the physical functional at whatever multipoles the iteration produced; forces are
ordinary autograd through the loop.

## 1. Why

The converged solve costs 10–15 matvecs plus an adjoint solve of the same size (the CT bond
term consumes `x*`, so the adjoint is never free), needs host syncs for convergence tests, and
under a force loss differentiates through an implicit function. A fixed-`K` loop is a static
graph: `K` matvecs forward, `K` back, no syncs, no tolerances, and second derivatives by plain
double backward. Energy conservation in MD is then exact up to the integrator, because the
energy is a closed-form function of `R` rather than the output of a solver at a tolerance.

The precedent is OPT (Simmonett, Pickard, Ponder, Brooks 2016: truncated perturbation series
with fitted global coefficients) and TCG (Aviat, Piquemal et al. 2017). Both recover the
converged polarization energy to ~1% at three terms and conserve energy better than CG at
production tolerance. What is new here is that the *short-range* part of the mutual coupling
is learned per site. That is the least physical part of a point-multipole model anyway (it is
where Thole/Slater damping is a fudge), so letting the network own it costs nothing.

## 2. The iteration, in the solver's variables

`coupled_solve` works in the rescaled state `x = (v, u, w)` with `q = q0 + B S v`,
`mu = mu0 + alpha u`, `Theta = quad0 + C w`, energy `E(x) = 1/2 x^T A x + b^T x + E0`, and
`A = M + P^T T P` where `M` is the fragment-internal block, `P: x -> delta m` the linear map to
the *induced* multipoles and `T` the gated Slater interaction on shells.

**Fields.** Let `F(m) = (g_q, g_mu, g_theta)` be the raw coupling gradient
(`coupled_solve._coupling_grad` before the pull-back) at multipoles `m`. Because it is affine,

    F(m(x)) = F(m(0)) + T_ss[b] · delta m(x),        delta m = m(x) - m(0),

where `F(m(0))` is the field of the permanent shells and nuclei (physical damping `b`,
one- and two-centre terms) and `T_ss[b] delta m` is the shell–shell operator on the induced
part: `slater_elec_field(pos, pairs, b, gate, delta m, m_nuc = 0)` — with zero nuclear moments
the one-centre terms drop out by themselves.

**The internal solve `Phi`.** Given fields `(g_q, g_mu, g_theta)`, the fragment-internal block
is solved exactly and inverse-free in the sectors that would need `alpha^-1`, `C^-1`:

    u = -(chivec + g_mu)            (mu0 form: -g_mu)
    w = -(chiquad + g_theta)        (quad0 form: -g_theta)
    (I + L S) v = -B^T (chi + eta q0 + g_q),   L = B^T eta B     (per fragment, dense, tiny)

This is `sqe_solve`'s system for the charges and the analytic uncoupled answer for the moments —
the CG preconditioner, now differentiable. `I + L S` is invertible for every `s >= 0`, so no
floor is needed and the closed-channel limit is exact.

**The loop.**

    x^(0)      = Phi( F(m(0)) )                              direct induction, exact per fragment
    x^(k+1)    = Phi( F(m(0)) + T_ss[b_ind] · P x^(k) )      k = 0 .. K-1
    x          = x^(0) + sum_{k=1..K} c_k (x^(k) - x^(k-1))  c_1 = 1 fixed, c_2.. learnable
    E_pol(x)   = coupled_energy(x) + slater_elec_pair_energy(m(x); b)      the PHYSICAL functional

Only `T_ss[b_ind]` differs from the physical operator: it is the same shell–shell tensor with
the induced-density widths `b_ind` in place of `b`. Direct induction, the permanent
electrostatics and the energy all keep `b`.

On `nonreactive` intra-fragment pairs are gated out of the induction (`gate_ind = gate (1 - p_intra)`),
so `x^(0)` is the *exact* isolated-fragment response and truncation touches intermolecular
many-body polarization only. Monomer polarizabilities are exact at every `K`. An isolated
fragment has `F(m(0)) = 0` (`chi = -eta q_perm`, `q0 = q_perm`, mu0 form) hence `x = 0` and
zero induction, as in the film.

## 3. Properties

* **Linear response.** `b_ind`, `alpha`, `c_k` depend on geometry and parameters only, never on
  fields or moments; `x` is linear in `F(m(0))` and in an external field.
* **Reciprocity.** With `Pi = P M^-1 P^T` (symmetric) the response is
  `sum_k c_k (Pi T~)^k Pi`, a palindrome of symmetric operators, hence symmetric. Requires the
  same `T~` at every iteration and *global* `c_k`; per-site or per-iteration relaxation weights
  would break it.
* **Long-range exactness.** `T~ - T = T_ss[b_ind] - T_ss[b]` is a difference of penetration
  terms, exponentially short-ranged. Beyond it the loop is the unmodified series through
  order `K`; under Ewald/PME the swap of widths lives entirely in the real-space part and costs
  nothing extra.
* **One-sided error.** For any `x`, `E(x) - E(x*) = 1/2 r^T A^-1 r >= 0` when `A` is PD: the
  scheme under-polarizes relative to the converged version of the same parameters and never
  over-polarizes. It stays finite even where `A` loses positive-definiteness (where CG reports
  `pd_fail`). The error is second order in the residual, which is why the physical functional
  and not `-1/2 mu.E0` is used. Forces are first order in the residual but exactly consistent
  with the energy that *is* the model.
* **Conservative forces.** `E(R, x_K(R))` is a closed form; autograd through the loop gives
  `-dE/dR` exactly, no Hellmann–Feynman shortcut and no adjoint.
* **Equivariance.** `b_ind` is an invariant scalar per site; all tensors are the existing
  equivariant Slater forms.

## 4. The learned widths

`FilmResponseHeads` grows one zero-initialised readout on the response-family latent,
`raw_i`, evaluated on the env-dressed latent only. The model forms

    s_i     = g(a_env,i) · softplus(raw_i + beta_Z)      (broaden mode; >= 0, exactly 0 in isolation)
    b_ind,i = b_i · exp(-s_i)

`g(a_env)` is the film's environment gate (exactly zero at `a_env = 0`), `beta_Z` a per-species
bias initialised at −3 so a fresh model starts with `b_ind ≈ b` and a nonzero gradient. Smaller
Slater exponent = broader induced density = weaker short-range mutual coupling, the direction
that keeps the spectrum of `Pi T~` tame. `induced_width_mode: free` drops the softplus and lets
the width move both ways. `T_ss[b_ind]` uses the usual combining rule `b_ij = sqrt(b_i b_j)`
inside the existing tensor code, so the isotropic and `r r^T` parts move consistently.

Expected effect: on ions part of the currently over-predicted reduction of `alpha` should move
into broadened `b_ind`, leaving `alpha` closer to physical confinement values. Watch
`env_b_ind` (the per-atom log shift) next to `env_alpha`.

## 5. Induced quadrupoles

`CoupledSystem` already carries the quadrupole sector (`cquad`, isotropic or axial); the film
never switched it on. `film.induced_quadrupoles: true` adds the isotropic `cquad` head
(`cquad_init`, `cquad_floor`) to `FilmResponseHeads`; `quad0 = quad_perm` as before. The
monomer polarizability anchor is unaffected (a uniform field does not drive quadrupoles).

## 6. Derivatives and cost

Plain autograd; `K` states of size `N × 9` are kept, negligible. The torchff field kernel has a
first-order VJP only, which is all MD needs; force *training* through the loop needs a second
derivative of the field, so in training mode the loop evaluates the torch reference field
(materialised `(P, K, K)` tensors, as the adjoint's one residual VJP does today). The
double-backward field kernel — the rank+1/rank+2 tensors contracted with two moment vectors —
is the one kernel this model asks torchff-lib for; it is the same kernel at every iteration.

At `K = 3`: 3 field evaluations forward and 3 back, versus ~10–15 CG matvecs plus an adjoint
solve of the same size today, and no host syncs.

## 7. Diagnostics and tests

`unrolled_response(..., with_residual=True)` reports the per-frame `max |A x + b|` of the
physical operator at `x_K`; `rsfff.ff.nonvariational.diagnose` compares `x_K` and `E(x_K)` with
`pcg` on a batch. Tests (`tests/nonvariational/`): the response matrix built column by column is
symmetric; `mu(2 E0) = 2 mu(E0)`; `b_ind = b`, large `K` reproduces `pcg` and the dense oracle;
`K = 0` reproduces the uncoupled fragment response; isolated fragment gives zero induction for
every `K`; far-apart fragments are unaffected by `b_ind`; finite-difference forces; rotation.

## 8. Training

Same data, same loss, same staged schedule as `configs/water_film.yaml`; `configs/water_nonvariational.yaml`
adds `model: nonvariational`, `n_iter: 3`, `iterate_weights: true`, `induced_width: true`,
`induced_quadrupoles: true`. `cg_ind` logs `K`, `cg_fail` is identically zero. Changing `K`
means refitting. Scan `K ∈ {2, 3, 4}` and take the smallest at the validation plateau. No
regulariser toward the exact minimum: being off it is the point.
