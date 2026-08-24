# Implementation Plan: Variational Polarizable Continuum Boundary Condition for an SQE-Polarizable Force Field

## 1. Overview and goals

We want to couple a polarizable continuum boundary condition to an existing SQE-polarizable force field (charges, induced dipoles, quadrupoles, multipolar Pauli repulsion, environment-dependent C6). The requirements are:

1. **Standard PCM behavior.** Under a specified dielectric $\varepsilon$, the stationary point of the model reproduces COSMO-type continuum solvation.
2. **Charge dynamics on the surface.** The surface polarization is a set of dynamical degrees of freedom propagated by an extended Lagrangian, run *adiabatically* (fictitious dynamics that tracks the stationary point — not physical dielectric relaxation, for now).
3. **Net charge on the continuum.** We can impose a specified total charge $Q_c$ on the continuum surface, for studying energetics of molecules in charged environments. Optionally the conjugate constant-potential ensemble.

The core observation is that SQE, the multipole response, and the continuum surface charge are **all convex quadratic problems with linear (Coulomb) coupling**, so the combined system has a single joint variational functional with a unique minimum, solvable with the existing CG + implicit-differentiation infrastructure.

We adopt **ddCOSMO** (domain-decomposition COSMO) as the continuum discretization because it is smooth, variational, matrix-free, linear-scaling, and has analytic gradients — all of which fit the existing solver and are essential for stable dynamics.

---

## 2. Variational functional

### 2.1 Base continuum functional (apparent surface charge form)

For a surface-charge representation $\mathbf{q} = \{q_i\}$ with solute potential $\boldsymbol{\varphi}$ sampled at the surface, the COSMO energy is the stationary point of

$$
G[\mathbf{q}] = \frac{1}{2 f(\varepsilon)}\, \mathbf{q}^\top \mathbf{S}\, \mathbf{q} + \mathbf{q}^\top \boldsymbol{\varphi},
\qquad
f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon}
$$

where $\mathbf{S}$ is the (regularized) surface Coulomb matrix. $\mathbf{S}$ is symmetric positive definite, so $G$ is strictly convex. Stationarity gives

$$
\frac{1}{f}\mathbf{S}\mathbf{q} = -\boldsymbol{\varphi}
\quad\Rightarrow\quad
\mathbf{q}^* = -f\, \mathbf{S}^{-1}\boldsymbol{\varphi},
\qquad
G[\mathbf{q}^*] = -\frac{f}{2}\,\boldsymbol{\varphi}^\top \mathbf{S}^{-1}\boldsymbol{\varphi}.
$$

### 2.2 ddCOSMO reformulation (the representation we implement)

Instead of surface point charges, ddCOSMO keeps the van der Waals cavity as a union of atom-centered spheres and expands the reaction potential on each sphere $j$ in real spherical harmonics up to $\ell_{\max}$ (typically 6–8):

$$
\text{unknowns } X_{j,\ell m}, \qquad \mathbf{L}\,\mathbf{X} = \mathbf{g}.
$$

- $\mathbf{L}$ = sparse ddCOSMO coupling operator; **never assembled explicitly**, only its action on a vector is needed (per-sphere spherical-harmonic operation + gather from overlapping neighbors). Matrix-free — ideal for the existing CG solver.
- $\mathbf{g}$ = RHS encoding the solute potential sampled at the per-sphere Lebedev quadrature points; this is where the SQE charges, induced dipoles, and quadrupoles enter.
- Underlying symmetric(izable) bilinear form → still variational → the Lagrangian machinery below carries over.
- Analytic gradient available (Lipparini et al., *JCTC* 2013).

The DOFs are smooth $\ell m$ coefficients rather than discrete tesserae, which is the root of the smoothness needed for dynamics (Section 4).

### 2.3 Joint functional with the force field

Because all pieces are quadratic with linear coupling, minimize a single functional over

$$
\big(\underbrace{\mathbf{p}}_{\text{split charges}},\; \underbrace{\boldsymbol{\mu}_{\text{ind}}}_{\text{induced dipoles}},\; \underbrace{\boldsymbol{\Theta}}_{\text{quadrupoles}},\; \underbrace{\mathbf{X}}_{\text{continuum}}\big)
$$

with the continuum potential contributing back to the solute (reaction field) and the solute potential forming $\mathbf{g}$. One joint linear solve yields self-consistent solute polarization plus reaction field. In practice this can be done as a single outer CG over the full coupled system, or as a nested/self-consistent loop; a single coupled solve is preferred for clean implicit-derivative gradients.

---

## 3. Extended Lagrangian dynamics (adiabatic)

Give the continuum DOFs a fictitious mass $m_q$ and propagate:

$$
\mathcal{L} = \tfrac{1}{2} m_q\, \dot{\mathbf{X}}^\top \dot{\mathbf{X}} - G[\mathbf{X}, \mathbf{R}]
\quad\Rightarrow\quad
m_q \ddot{\mathbf{X}} = -\frac{\partial G}{\partial \mathbf{X}} = -(\mathbf{L}\mathbf{X} - \mathbf{g}).
$$

- Run with a thermostat on $\mathbf{X}$, or with Niklasson-style dissipative XL propagation, so $\mathbf{X}$ tracks the stationary point adiabatically.
- **This is fictitious dynamics that stays near equilibrium — not physical Debye relaxation.** $m_q$ is an arbitrary numerical parameter chosen for adiabatic separation and stability, not a physical inertia. (Physical relaxation would require mapping $m_q$/friction to the Debye time; out of scope for now.)

### 3.1 Net charge constraint

Impose $\mathbf{1}^\top \mathbf{q} = Q_c$ (expressed in ddCOSMO via the appropriate projection of $\mathbf{X}$ onto total surface charge — the $\ell=0$ monopole contributions of each sphere) with a Lagrange multiplier $\lambda$:

$$
G[\mathbf{X}, \lambda] = \tfrac{1}{2}\mathbf{X}^\top \mathbf{L}\mathbf{X} - \mathbf{X}^\top\mathbf{g} + \lambda\big(\mathbf{c}^\top \mathbf{X} - Q_c\big),
$$

where $\mathbf{c}$ is the linear functional returning total surface charge. This adds **one bordered row/column** to the linear system:

$$
\begin{pmatrix} \mathbf{L} & \mathbf{c} \\ \mathbf{c}^\top & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \lambda \end{pmatrix}
=
\begin{pmatrix} \mathbf{g} \\ Q_c \end{pmatrix}.
$$

- $\lambda$ is a **uniform potential offset** = electrochemical potential of the surface.
- **Constant-charge (canonical):** fix $Q_c$, solve for $\lambda$.
- **Constant-potential (grand-canonical):** fix $\lambda$, drop the constraint row and add $-\lambda\,\mathbf{c}$ to the RHS; $Q_c$ then floats. This is the Legendre conjugate and is often the more physical choice for charged-environment energetics. Structurally identical to constant-potential electrode MD (Siepmann–Sprik; Reed–Lanning–Madden).
- **Recommendation:** implement both; expose $\lambda$ (or $Q_c$) as an input.

---

## 4. Smooth surfaces for stable dynamics

Two distinct sources of discontinuity must be handled.

### 4.1 Discrete tesserae (avoided by construction)

Classic tessellated PCM samples the surface with discrete points that switch on/off as atoms move → step discontinuities → delta-function forces → dynamics blow up. **ddCOSMO avoids this entirely** because its DOFs are smooth $\ell m$ coefficients, not surface points. This is the primary reason we use ddCOSMO.

### 4.2 Smoothly switched characteristic function

ddCOSMO still integrates over the *exposed* part of each sphere $j$ (the region not buried inside neighbors), defined by a characteristic function $\chi_j(\mathbf{r})$: 1 outside all other spheres, 0 inside. A sharp $\chi_j$ reintroduces a discontinuity when a quadrature point on sphere $j$ crosses the boundary of neighbor $k$. The fix is a smoothly switched $\chi$ with a regularization width $\eta$.

For a quadrature point at position $\mathbf{s}$ on sphere $j$ and a neighbor sphere $k$ of radius $R_k$ centered at $\mathbf{R}_k$, define the scaled signed distance

$$
t_{jk}(\mathbf{s}) = \frac{\lVert \mathbf{s} - \mathbf{R}_k\rVert - R_k}{\eta\, R_k}
\quad\text{(a common choice; normalizations vary by implementation).}
$$

The per-neighbor switch is a fixed polynomial with $C^2$ (or higher) continuity, e.g. the standard quintic used in ddCOSMO:

$$
w(t) =
\begin{cases}
0, & t \le 0, \\
6t^5 - 15t^4 + 10t^3, & 0 < t < 1, \\
1, & t \ge 1,
\end{cases}
$$

which satisfies $w(0)=w'(0)=w''(0)=0$ and $w(1)=1,\ w'(1)=w''(1)=0$. The full characteristic value at the point is the product over all overlapping neighbors:

$$
\chi_j(\mathbf{s}) = \prod_{k \ne j} w\big(t_{jk}(\mathbf{s})\big).
$$

- As long as $\eta > 0$, energy and gradient are continuous.
- This is the ddCOSMO analogue of the **SWIG** (Switching/Gaussian) scheme (Lange & Herbert, *JCP* 2010) used for tessellated ASC.

### 4.3 Analytic forces

The geometry gradient must include:

1. **Explicit-term derivatives** at fixed $\mathbf{X}$. By the envelope theorem / implicit function theorem, at the stationary point $\partial G/\partial\mathbf{X}=0$, so we differentiate only the explicit $\mathbf{R}$-dependence — **no backprop through the linear solve**. Solve, detach $\mathbf{X}$, differentiate explicit terms. (The existing implicit-differentiation wrapper already does this.)
2. **Switching derivatives** $\partial\chi/\partial\mathbf{R}$ via $\partial w/\partial t \cdot \partial t/\partial\mathbf{R}$.
3. **Radius derivatives** $\partial R_j/\partial\mathbf{R}$ — see Section 6, only nonzero if radii are made dynamical (Slater-width option).

**Sufficient quadrature:** use enough Lebedev points per sphere to resolve the ramp of width $\eta$; a smooth $\chi$ still needs adequate sampling across the switching region.

---

## 5. Calibration of the range parameter $\eta$

$\eta$ trades off force smoothness against cavity fidelity:

- Too small → force spikes near sphere intersections → tiny timestep required.
- Too large → "fuzzy" cavity → biased solvation energy.

**Procedure:**

1. Start from literature defaults ($\eta \sim 0.1$–$0.2$ in the switching-region units).
2. **Energy-conservation sweep:** run NVE with the extended-Lagrangian surface charges for a range of $\eta$ and timesteps; select the largest $\eta$ and timestep that give acceptable drift (see Section 7).
3. **Bias check:** for the chosen $\eta$, confirm that calibrated solvation energies (Section 7 Born tests, plus a small neutral-molecule set) are not materially shifted relative to the $\eta \to 0$ limit. If they are, either reduce $\eta$ or absorb the shift into the radius calibration.
4. Fix $\eta$ as a global hyperparameter; document the chosen value and the drift/timestep it was converged against.

Also converge $\ell_{\max}$ and the Lebedev order jointly, since all three interact with accuracy and smoothness.

---

## 6. Optional: Slater widths as the atomic radius

The natural choice is to set each sphere radius $R_j$ from the atom's Slater width, since the width already sets the charge spread the continuum "sees."

**Considerations:**

- **Self-energy calibration.** The diagonal self-energy of the surface operator (equivalently the effective sphere size / regularization) must be calibrated against reference solvation energies. Watch for **double-counting** between the electrostatic smearing (from finite-width charges) and the cavity definition — the continuum should not re-smear what the charge model already smears.
- **Radii are dynamical variables.** The Slater widths are NN outputs that depend on the environment, so $R_j = R_j(\text{NN inputs}(\mathbf{R}))$. This is the one genuinely new gradient path not present in vanilla ddCOSMO:

$$
\frac{\partial G}{\partial \mathbf{R}}\bigg|_{\text{radius path}}
= \sum_j \frac{\partial G}{\partial R_j}\,\frac{\partial R_j}{\partial(\text{NN inputs})}\,\frac{\partial(\text{NN inputs})}{\partial \mathbf{R}}.
$$

The $\partial R_j/\partial(\text{NN inputs})$ factor flows through the existing autodiff graph; the new pieces are $\partial G/\partial R_j$ (how the ddCOSMO energy depends on sphere radius, including via $\chi$ and via the quadrature grid) and its correct wiring into the NN backward pass.

- **Missing this path → discontinuous or wrong forces** when geometry (hence predicted widths) shifts. This is a common source of energy drift; test it explicitly.
- **Implementation order:** get fixed-radius ddCOSMO correct and validated first, then switch radii to Slater widths as a second step so the two error sources are separable.

---

## 7. Validation against Born charge calculations

The Born model is the canonical analytic check: a point charge $z$ (a.u.) at the center of a single sphere of radius $a$ in a dielectric $\varepsilon$ has solvation energy

$$
\Delta G_{\text{Born}} = -\frac{1}{2}\left(1 - \frac{1}{\varepsilon}\right)\frac{z^2}{a}.
$$

**Test ladder (each step gates the next):**

1. **Single-sphere monopole = Born.**
   - One sphere, central unit charge, sweep $\varepsilon$ (e.g. 2, 4, 20, 78.4, $\infty$) and radius $a$.
   - ddCOSMO energy must match $\Delta G_{\text{Born}}$ to the discretization tolerance. Convergence to the analytic value as $\ell_{\max}$ and Lebedev order increase is the key acceptance criterion.

2. **Charge scaling.** Verify $\Delta G \propto z^2$ and the correct $(1-1/\varepsilon)$ dielectric dependence across the swept $\varepsilon$.

3. **Off-center charge / Kirkwood multipoles.** Displace the charge, or use a dipole/quadrupole source, and compare against the Kirkwood analytic multipole expansion for a sphere. This exercises the higher-$\ell$ machinery that pure Born does not, and validates the quadrupole coupling from the force field.

4. **Two-sphere / diatomic.** Overlapping spheres exercise the switched characteristic function and neighbor coupling. No closed form, but check smoothness: scan the internuclear distance through the overlap region and confirm energy and forces are continuous (no kinks at the onset/offset of overlap).

5. **Net-charge constraint.**
   - Impose $Q_c$ on the continuum around a neutral solute and verify $\lambda$ (uniform offset) and total surface charge match the constrained analytic result for a sphere.
   - Constant-potential mode: fix $\lambda$, verify recovered $Q_c$ is the Legendre conjugate of the constant-charge run.

6. **Force/gradient tests.**
   - **Finite-difference** the analytic geometry gradient (atom positions) — must agree to ~$10^{-5}$–$10^{-6}$ relative.
   - Include the radius path (Section 6) in the finite-difference test once Slater-width radii are enabled.

7. **Energy conservation (NVE).**
   - Extended-Lagrangian surface charges, no thermostat, monitor total (physical + fictitious) energy drift over a long trajectory.
   - Drift that shrinks with timestep → integration; drift that does **not** shrink with timestep → a smoothness/gradient bug (sharp $\chi$, missing $\partial\chi/\partial\mathbf{R}$, or missing radius path). This is the most sensitive integration test.

8. **Adiabaticity.** Confirm the fictitious surface-charge DOFs stay cold relative to the nuclear DOFs (power-spectrum separation), i.e. $m_q$ and the thermostat give clean adiabatic tracking of the stationary point.

---

## 8. Suggested implementation sequence

1. **Fixed-radius equilibrium ddCOSMO** with matrix-free $\mathbf{L}$ action, plugged into the existing CG + implicit-diff solver; RHS $\mathbf{g}$ from SQE charges only.
2. **Born validation** (Section 7, steps 1–3). Gate.
3. **Add multipole sources** (induced dipoles, quadrupoles) to $\mathbf{g}$ and the reaction field back onto the solute; Kirkwood multipole check.
4. **Smooth $\chi$ + analytic forces**; two-sphere smoothness and finite-difference gradient tests (steps 4, 6).
5. **Net-charge constraint** (bordered system) + constant-potential mode; step 5 tests.
6. **Extended-Lagrangian dynamics**; energy conservation and adiabaticity (steps 7–8).
7. **Slater-width radii** as dynamical variables, with the extra gradient path; re-run finite-difference and energy-conservation tests. Calibrate self-energy and check for smearing double-counting.
8. **$\eta$, $\ell_{\max}$, Lebedev calibration** (Section 5) as a final joint sweep against energy conservation and a small solvation reference set.

---

## 9. Key references

- Cancès, Maday, Stamm, *J. Chem. Phys.* **139**, 054111 (2013) — ddCOSMO.
- Lipparini et al., *J. Chem. Theory Comput.* **9**, 3637 (2013) — ddCOSMO analytic forces.
- Lipparini et al., *J. Chem. Phys.* **141**, 184108 (2014) — ddCOSMO for large systems / MD.
- Stamm et al., *J. Chem. Phys.* **144**, 054101 (2016) — ddPCM (proper dielectric, if IEF form is needed later).
- Lange, Herbert, *J. Chem. Phys.* **133**, 244111 (2010) — SWIG smooth ASC.
- Lipparini, Scalmani, Mennucci, Cancès, Frisch, *J. Chem. Phys.* **133**, 014106 (2010) — symmetric variational IEF-PCM.
- Reed, Lanning, Madden, *J. Chem. Phys.* **126**, 084704 (2007) — constant-potential electrode formalism (net-charge/potential ensemble analogy).