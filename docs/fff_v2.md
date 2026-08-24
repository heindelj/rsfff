# Fragment Experts with a Two-Slot Parameterization

## 1. Objective

Every force-field parameter in this model is emitted by a network. The question that has been
hardest to answer is not what those networks should look like, it is **which part of what they emit
is a property of a molecule and which part is a property of that molecule's surroundings**. Those
are different physics, they are fitted against different labels, and when they share one input
vector nothing prevents the second from quietly doing the first's job. That failure has happened
here repeatedly and under several different names: a charge-transfer channel that was 96.7% neural,
a second one that was 100% descriptor swap, a bond readout that also defined the one-body zero.

This document specifies a model in which the separation is **analytic**. Two things are made
structurally true rather than encouraged by a loss:

1. Every parameterizer takes two inputs — a **fragment slot** and an **environment slot** — and the
   environment slot is *identically zero* for an isolated fragment. Evaluating a parameterizer with
   that slot zeroed therefore returns exactly what the model claims about the fragment on its own.
2. The one-body energy reads only the zeroed evaluation. So `fragment_energy` is exactly the
   isolated-fragment quantity its label is, at every point in training, at every geometry, at any
   separation from any neighbor.

Everything else in the design follows from wanting those two statements to be theorems.

The second commitment is to **fragment experts**. There is no universal parameterization network
keyed by element. A fragment of a given chemical composition is described by a network dedicated to
that composition, which is also the expert on how that fragment's parameters shift in the presence
of others. This is not a scalability story — it is a worse one, for now — but it is what lets the
one-body description be sharp enough that the constant/not-constant split above means something.

---

## 2. Fragment types and experts

For each fragment `f` the model needs its **type** `s`: the canonical composition, e.g. `"H2O"`.
The type selects an expert `E_s`, which owns a complete set of parameter-emitting heads:

| Head | Emits |
| --- | --- |
| response | `chi`, `eta`, permanent `mu_0` / `Theta_0`, `alpha`, `C`, Slater `Z`, `b`, charge-flux compliances |
| dispersion | `C6`, `b_disp` |
| Pauli | shell charge `q`, exponent `b`, `mu`, `Theta` |
| range separation | per-atom `r0` and per-channel `alpha` for each classical channel |
| bond | the per-atom energy of the electronic state |
| applicability | `v_f`, a per-fragment diagnostic (§8) |

Charge and multiplicity are **not** part of the type. They enter as a fragment-state input inside
the expert, so a single `"OH"` expert covers hydroxide and the OH radical and can be told which one
it is looking at. Splitting those into separate experts would multiply the expert count by the
charge/spin manifold for no gain — the two share almost all of their chemistry, and what differs is
exactly what an input slot is for.

Water has one expert, so dispatch is a fast path. With more than one type present, atoms are grouped
by their fragment's type, each expert runs on its own group, and the results are scattered back.

---

## 3. The two slots

Both slots are per-atom λ-SOAP descriptors (λ = 0, 1, 2) built from **one** neighbor search and
**one** set of spherical harmonics, differing only by which edges are allowed to contribute:

```text
h_i  =  power spectrum of the density over edges with  frag(i) == frag(j)      fragment slot
eta_i = power spectrum of the density over edges with  frag(i) != frag(j)      environment slot
```

The two edge sets are complementary halves of the same graph. `h` is the descriptor a fragment would
have if the rest of the system were deleted; `eta` is a description of everything that was deleted,
as seen from atom `i`.

### Why complementary masks and not a residual

The previous model carried one stream, `h_env = h_frag + g(h_full) - g(h_frag)`: an environment
residual added into the fragment vector, anchored so that it vanishes for an isolated fragment. That
anchoring works — the subtraction is exact, not merely small — but it makes "isolated" a property of
an arithmetic identity between two evaluations of `g` rather than a property of the input. Two
consequences followed. The environment could never be inspected as an object of its own; and the
notion of "the parameter this fragment would have alone" required knowing which of two vectors to
pass, which is a convention the code had to enforce by hand, in different places, differently.

With complementary masks, `eta = 0` for an isolated fragment because there are no cross-fragment
edges to sum. Not approximately, not by construction of `g`, not at initialization only. There is
nothing to enforce.

### What the environment slot should eventually carry

For water the density channels are keyed by element, which is enough. When more fragment types exist
the cross-fragment density should be keyed by `(element, neighbor fragment type)`, so that an expert
can distinguish "a hydrogen belonging to a nearby water" from "a hydrogen belonging to a nearby
hydronium". The machinery for a learned, state-aware density channel already exists in
`FlatStateSOAPFeaturizer`; this is the place it goes.

---

## 4. Parameterizers

Every parameterizer takes exactly two inputs, and combines them in its first layer:

```text
P_s(h, eta)  =  f( W_h · h  +  W_env · eta  +  b )        W_env zero-initialized
```

which gives two evaluations of every quantity:

```text
theta   =  P_s( h , eta )      the in-medium parameter
theta_0 =  P_s( h , 0   )      the isolated-fragment parameter
```

`theta_0 = theta` for an isolated fragment identically, since `eta` is then zero. `theta_0` is
cheap: it is the same forward pass with one term dropped.

### The split is analytic, not a training schedule

An earlier version of this plan proposed fitting `theta_0` on monomers, freezing it, and letting
only `W_env` train on clusters. That is wrong, and the reason is worth stating because it is not
obvious.

**`C6`, `b_disp`, the Pauli multipoles and `r0` have no monomer-side label at all.** They appear
only in inter-fragment channels; the intra-fragment classical remainder that survives the range
separation is well under a kJ/mol per fragment, so monomer data cannot constrain them even in
principle. Freezing them after a monomer stage would pin them at noise and then require `W_env` to
carry the entire dispersion and Pauli parameterization — the environment weights would *become* the
model, which is exactly the pathology this design exists to prevent.

So `theta_0` and `theta` share `W_h`, and cluster data supervises both. The guarantee does not come
from the optimizer:

> `theta(h, 0) = theta_0(h)` holds identically, whatever the optimizer does, and no `eta` enters
> `fragment_energy` anywhere. Therefore `fragment_energy` is exactly an isolated-fragment quantity
> at every point in training.

What the `W_h` / `W_env` split buys, without any freeze, is that **the environment sector is a named
set of tensors**. The penalty below attaches to it, the diagnostics read it, and the ablation
"switch off the environment entirely" is one line rather than a second model.

### The environment penalty

The classical channels are fitted against interaction labels, which constrain the *in-medium*
parameters. Nothing in those labels says how much of `C6` should be a property of the water and how
much a property of its neighbors. That decision is made explicitly:

```text
L_env  =  sum over quantities q of   w_q * || theta_q(h, eta)  -  theta_q(h, 0) ||^2
```

in log space for the positive-definite quantities (`C6`, `b`, `r0`, `alpha`). This pushes every
explanation into `theta_0` unless the data genuinely demands otherwise, and its per-quantity terms
are directly readable: *how much of `C6` is environmental* is a number reported each epoch.

It replaces a penalty on `||h_env - h_frag||`, a feature-space norm with no physical interpretation
and hence no defensible weight.

---

## 5. The one-body sector

```text
E_f  =  sum_{i in f} E0[Z_i]                             fixed atomic reference
     +  E_internal( f ; theta_0 )                        response / SQE solve
     +  sum_{intra pairs} gate(theta_0) * E_class(theta_0)
     +  sum_{i in f} E_bond( h_i , M_i , Phi_i^intra ; theta_0 )
```

No `eta` appears. Every parameter is the isolated evaluation, including on the intra-fragment
classical channels — a change from the previous model, which let Pauli and dispersion read the
environment stream on intra pairs on the argument that the range separation had already switched
that contribution below a kJ/mol. That argument is about magnitude; the claim being made here is
about kind, and a small environment dependence in a term whose label has none is still the wrong
functional form.

`Phi^intra` is the electrostatic potential, field and field gradient from **intra-fragment pairs
only**. The field an atom feels from its own molecule is part of what its bonds are worth; the field
from a neighboring molecule is not.

### The target is an atomization energy

`E_f` is fitted against the isolated-fragment SCF energy, which is the atomization energy up to the
fixed shift `sum_i E0[Z_i]`. That shift is the only external information the one-body sector
receives. In particular:

* **No free-atom anchoring.** The previous bond head evaluated `net(x) - net(ref)` so that a lone
  atom's energy was exactly `E0`. That cancels the readout's final-layer bias identically, which
  removes the only direction that moves the one-body constant without also reshaping its geometry
  dependence — and it cost a measured constant −5.23 kJ/mol per fragment that nothing in the loss
  could cheaply remove.
* **No per-species offset.** `species_offset` and its `tanh(||h||)` gate existed solely to hand that
  direction back. With the anchoring gone the bias is free again and the gadget is unnecessary.
* **No free-atom polarizability penalty.** Pinning `alpha_i` to a tabulated free-atom value is a
  constraint at a geometry the expert does not claim to describe. The tabulated `chi_0`/`eta_0`/
  `alpha` remain as *initialization*.

The consequence is explicit and accepted: **the model no longer has an exact isolated-atom limit.**
A water expert is not asked about a bare oxygen. Saying so is the applicability diagnostic's job
(§8), not an anchoring's.

---

## 6. The interaction sector

```text
E_inter  =  sum_{inter pairs, c}  gate_c(theta) * E_class^c(theta)         c in {elst, pauli, disp}
```

fitted against `eda_cls_elec`, `eda_mod_pauli`, `eda_disp`. Same classical forms as before — damped
Slater multipole electrostatics, a Slater Pauli overlap term, Tang–Toennies-damped `C6` — and the
same learned Fermi range separation deciding per pair and per channel how much of each classical
form is switched on.

**There are no neural pair corrections.** No per-pair energy readout, no per-pair `r0` deviation.
The neural content of this model lives entirely in the parameters an expert emits; the classical
forms are then evaluated as written. This is what the "experts emit parameters" thesis actually
commits to, and a pair readout sitting on top of it is a second, unlabeled model competing for the
same energy — which is how a dispersion leak once put −39 kJ/mol per fragment of "intramolecular
dispersion" between bonded atoms.

One consequence worth noting: the previous model needed a hand-written special case in which the
electrostatic channel was re-scored on the fragment-confined stream, so that the environment-driven
part of `cls_elec` could be rebooked as polarization. That was the `theta` vs `theta_0` pattern
applied to one channel by hand. Here every channel has both evaluations by construction, and the
rebooking is uniform.

---

## 7. Induction and charge transfer

```text
E_ind  =  [ coupled solve with theta(h, eta) ]  -  [ the same functional at frozen multipoles ]
       +  sum_i [ E_bond( h_i, M_i^ind, Phi^ind ; theta )
                  - E_bond( h_i, M_i, Phi^intra ; theta_0 ) ]
```

fitted against `eda_pol + eda_ct` as one label.

The first line is a genuine relaxation: the same energy functional, minimized with the response
parameters read in-medium and the multipoles free to move against each other, minus its own value at
the unrelaxed multipoles. It is negative by the variational principle when nothing else differs.

The second line is the part this design is really about. **The environment-dependence of the bonding
term is the charge-transfer mechanism.** It is the same weights, read at the in-medium parameters
and at the relaxed electronic state, minus the same weights read at the isolated parameters and the
frozen state. It is not a channel bolted on with its own readout — the two previous attempts at that
both ended up with a network wearing the label — and its size is directly measurable as
`theta - theta_0` propagated through `E_bond`.

The fragment still supplies the *graph* of allowed charge flux (its own intra-fragment channels);
the environment supplies the *parameters* on it. No charge crosses a fragment boundary in this
model. Explicit inter-fragment charge flow returns when reactivity does.

---

## 8. Mediation: what happens when an atom is shared

### The unit of competition is an atom, not a fragmentation

Competing fragmentations are not independent hypotheses about a system. For two decompositions `A`
and `B` of one geometry, let

```text
D  =  { i : frag_A(i) != frag_B(i) }
```

In every H3O+/OH− frame in `data/wb97mv_tzvpd`, `|D| = 1` and the atom is a hydrogen.
`H3O+ | H2O | H2O` versus `H2O | H3O+ | H2O` is one proton changing address. Whatever else
competition between fragmentations might mean in general, in the data this model has it means
**a bond is being relabeled**, and that is the structure to exploit.

Moving atom `i` from `f` to `g` does the following, and nothing else:

| Changes | Does not change |
| --- | --- |
| pairs `(i, j in f)` flip intra → inter | any pair with both atoms outside `f ∪ g` |
| pairs `(i, j in g)` flip inter → intra | `sum_i E0[Z_i]`, which is fragmentation-invariant |
| the compositions of `f` and `g`, hence **which experts describe them** | the nuclei, the total charge, `E_total`, the forces |
| `Q_f` and `Q_g` | |
| the SQE channel graph inside `f` and `g` | |
| `h` and `eta` for every atom of `f ∪ g` | |

The *decision* is atom-local; the *consequence* is confined to `f ∪ g`. That is what bounds the cost
of doing this properly, and it is why a mediator that sees one atom and its two candidate hosts is
the right size of object.

### Why this is not the applicability score's job

`v_f` asks "is this expert appropriate for this fragment". At a proton-transfer geometry both
decompositions are asking a well-posed question of a well-posed expert — a slightly-too-long H3O+
and an H2O with a slightly-too-close proton are each perfectly describable. Ranking them by pooled
per-fragment viability answers a question adjacent to the one that matters. What matters is what
becomes of **the bond being relabeled**, and no quantity pooled over a whole fragment is addressed
to it.

`v_f` is therefore demoted (§11): it survives as a per-fragment diagnostic and an optional mediator
input, and the softmax-over-fragmentations loss it carried is replaced by what follows.

### When a swap is live

Two stages, and the separation is deliberate: enumeration is discrete and cheap, mediation is
learned and smooth.

**Enumeration.** Atom `i`, currently in `f`, is a candidate for sharing with `g` when all of:

1. `Omega_ig > 0`, a C²-continuous validity bump (`rsfff.mlip.switch.validity_bump`) on the contact
   geometry — for a transferring hydrogen, the `H···g` distance.
2. The bank holds experts for **both** post-swap compositions. This is the whole of "we have seen
   this reaction before": the expert keys *are* the reaction database, so there is no second
   registry to fall out of sync with the model, and `ExpertBank.assign` already refuses an unknown
   composition rather than answering with the wrong network.
3. The post-swap charges and multiplicities are ones those two experts were trained on.

**Mediation.** Every atom then carries a membership over its candidate hosts, a partition of unity:

```text
pi_ig  =  softmax_over g ( M( h_i , eta_i , H_f , H_g , rho_ifg )  +  log Omega_ig )
sum_g pi_ig  =  1
```

`H_f` and `H_g` are the pooled fragment descriptors of the two hosts and `rho` the contact geometry.
`M` is **one universal network, not one per expert**: it learns the logic of chemical competition
and never a fragment's chemistry. That is the division of labour the deferred router existed for,
relocated to the object that can actually act on it.

`M` must read only combinations symmetric under swapping the two candidate hosts, so the same
physical swap enumerated from either end returns the same weights. `AdiabaticCorrection` already
does this by reading `h^K + h^J` and `(h^K − h^J)^2`; the same trick applies here.

An atom with one candidate has `pi = 1` and costs nothing, so a neutral water cluster runs the
model of §5–§7 untouched.

### What the weights do

One rule decides where each quantity is mixed:

> **Mix at the lowest level at which the quantity means the same thing to both experts.**

| Quantity | Mixed at | Because |
| --- | --- | --- |
| pair routing, `b_ij = sum_F w_F I_F(ij)` | the accounting | `sum_F w_F = 1`, so every pair still appears exactly once across the buckets |
| `C6`, `b_disp`, `r0`, Pauli multipoles, response parameters | the **parameter** | a `C6` is a `C6`: both experts emit the same physical number, and pairs already combine per-atom values across experts by geometric mean |
| `E_bond` | the **output** | `FragmentBondEnergy` is per-expert and the two networks share no input space, so there is nothing to mix upstream of it |
| the response solve | one solve on the union graph | a channel only one assignment opens carries its weight as a compliance scale |

The routing row is the one most easily left out of this design and the one that would break it. It
is where energy crosses the boundary between `fragment_energy` and the EDA channels, so a hard
`is_intra` under a soft `pi` is not merely inconsistent — it is a discontinuity in the accounting
itself.

The solve row has a precedent to reuse rather than invent: the diabatic-mixture stack solved exactly
this with `finest_common_refinement` plus a compliance switch, where a partially-open channel is
rescaled by `S` and not `sqrt(S)`, on pain of training NaNs at the closed limit.

Note what the rule does **not** permit. Blending the experts' *features* and decoding once — what
the diabatic mixture did, and the better answer wherever it is available — is unavailable here by
construction: the decoder is per-composition, and that is the architecture rather than an accident.
Mixing therefore happens downstream of the experts, and the price is that it happens separately for
each kind of quantity.

### Invariants

Non-negotiable, and each one testable:

1. **Vertex identity.** When every `pi` is one-hot the model reduces *exactly* — not approximately —
   to the single-fragmentation model of §5–§7. This is what makes the mediator an addition rather
   than a second model, and it is the property the old `4 c_K c_J` prefactor bought structurally.
2. **Partition of unity.** `sum_g pi_ig = 1`, so the accounting identity of §6 survives.
3. **Swap symmetry.** The weights do not depend on which host was enumerated first.
4. **C² continuity** as a candidate opens or closes, carried by `Omega`.
5. **`fragment_energy` stays a vertex quantity.** It is defined per fragmentation and nowhere else:
   a fragment with a half-owned atom has no isolated-fragment SCF energy to be compared against. The
   mixture produces a **total**, and the fragment-view stream of §9 is untouched by all of this.

### Training the mediator

There is a label problem here and it has to be stated plainly: **a mixture has no ALMO-EDA label.**
Every EDA channel is defined relative to a choice of fragments, so `eda_cls_elec` for a 60/40
mixture of two decompositions is not a quantity Q-Chem computed, or could compute.

What *is* fragmentation-invariant is the total energy and the forces. Every frame of a `group_id`
carries the same `E_total` and the same forces, because it is the same geometry and the same
wavefunction. That gives the split:

| Supervises | With |
| --- | --- |
| the pure vertices | the four EDA channels, per fragmentation, exactly as §9 already does |
| the mixture | `E_total` and forces — the only labels defined where `pi` is not one-hot |
| the mediator, as a shaping prior | `Delta E_ind` between the candidate assignments |

The third row needs care. `|E_pol + E_ct|` picks the chemically obvious assignment in 398 of 399
frames, with a best-versus-second gap of 165–476 kJ/mol, so it is an excellent teacher of *which*
assignment is better. It cannot teach *how much of each to keep*: it is a ranking signal and the
degree of mixing is a magnitude. Only the total-energy residual through the crossover can set that,
and it can — precisely because both decompositions carry the same `E_total` label there while the
model gives them different answers. The size of that disagreement is a direct measurement of how
badly each pure assignment is doing, and removing it is what `pi` is for.

So `Delta E_ind` shapes the mediator early and keeps it honest; `E_total` and the forces decide it.

### Open

* **Charges of candidate fragments.** Moving an atom between fragments means deciding what charge
  goes with it, and an expert needs `(Q_f, 2S_f)` to answer at all. For this corpus the rule is "a
  transferring hydrogen carries +1", which covers every H3O+/OH− frame. A general rule is not
  decided and probably should not be invented ahead of data that needs one.
* **Cost.** Each live candidate re-featurizes and re-evaluates `f ∪ g`. Bounded, but not free; the
  trigger's tightness is what controls it.
* **A flat direction.** Intra classical energy is already degenerate with `E_bond`. Letting `pi`
  move energy across the intra/inter boundary hands the mediator a second route into that
  degeneracy, and the `r0` barrier of §6 may not be enough on its own.
* **Concerted swaps** (`|D| > 1`). Out of scope; the corpus contains none.

---

## 9. Training

### Two data streams, one fit

The corpus carries isolated-fragment labels for every fragment of every cluster —
`fragment_energies`, `fragment_dipoles`, `fragment_second_moments` — so each cluster frame can be
exploded into one frame per fragment. For the water set that is roughly 34k monomer geometries,
sampled exactly where clusters actually go, against the 499 in the dedicated monomer set.

Each optimizer step draws from both streams:

| Stream | `eta` | Supervises |
| --- | --- | --- |
| fragment views | `= 0` | `fragment_energy`; fragment dipole and quadrupole; on the dedicated monomer subset, molecular polarizability and true one-body forces |
| clusters | live | `eda_cls_elec`, `eda_mod_pauli`, `eda_disp`, `eda_pol + eda_ct`, cluster forces |

plus `L_env` from §4.

Both rows describe a *single* fragmentation, which is all the water corpus has. Where competing
fragmentations exist the cluster row splits: the EDA channels supervise the pure vertices only, and
`E_total` and the forces carry the mixture, for the reason in §8 — those are the only labels that
survive `pi` leaving one-hot. The fragment-view row is unaffected either way, since
`fragment_energy` is defined at the vertices and nowhere else.

There is no freeze and no ordering requirement between them, for the reason given in §4: the
interaction labels are the *only* constraint on several of the `theta_0` quantities, so they must
reach `W_h`. A cheap warm-start pass over the fragment-view stream alone — no pair list, no coupled
solve — is worth doing to get the one-body and response sectors into the right basin before the
clusters arrive, but it is a warm start and nothing depends on it.

### What to watch

| Quantity | Reading |
| --- | --- |
| per-quantity environment share | `Δlog C6`, `Δlog b_disp`, `Δlog r0`, `Δ||mu_Pauli||`, `ΔE_bond`. **The number this design exists to produce.** If a channel's explanation lives here rather than in `theta_0`, `L_env` is too loose. |
| `\|\|eta\|\|` | how much environment the fit asks for. Exactly zero means weight decay deleted `W_env`, not that the model declined the environment — `W_env` is zero-initialized and must be decay-exempt. |
| `ind_ff` vs the `E_bond` difference | the split of induction between a real relaxation and a parameter shift. Both are physical; the ratio is the thing to keep honest. |
| `internal` vs `e_bond` | the unlabeled split inside `fragment_energy`. Only the sum is fitted, so this is a flat direction — watch its spread, not its value. |
| `v_f` | a diagnostic only (§8); watch that it is not silently decaying to a constant. |
| `pi` occupancy | the fraction of shared atoms whose membership is genuinely split. Exactly one-hot everywhere means the mediator is off or the trigger never opens; broadly split everywhere means it is hedging rather than deciding. |
| `E_total` at the crossover | the residual the mediator exists to remove, measured where `pi` is furthest from one-hot. It is the only place the mixture has a label. |

---

## 10. Deferred

These are specified here so that adding them is an addition rather than a retrofit. The router and
the smooth occupancies that used to head this list have moved into §8, which is where the mediator
now does their job at the granularity the data actually has.

**Ambiguity correction.** The mediator of §8 is deliberately **linear**: it keeps a fraction of each
expert's answer alive and nothing more. It therefore cannot represent physics that is in neither
conventional fragmentation. Where two assignments are simultaneously live, `A = 4 pi_1 pi_2` gates a
correction `E_amb = A * U( z_1 , z_2 , rho_12 , eta )`, which may depend on the environment
arbitrarily because that is precisely what it is for.

This is second and not first on purpose. A correction that vanishes at both vertices is
unidentifiable until the linear mixture is already carrying the crossover: with `pi` untrained,
`U` would simply absorb whatever the mixture was failing to do, and there would be no way to tell
which of the two was wrong. Fit the linear mediator, measure what `E_total` still misses through the
crossover, and only then decide whether a correction is warranted.

**Concerted and multi-atom swaps.** §8 mediates one atom's membership at a time. A relabeling with
`|D| > 1` — a concerted double proton transfer, say — needs the candidate enumeration to range over
sets rather than atoms, at which point the fragmentation-level `p_F = softmax(S_F / T)` of the old
router becomes the right object after all. Nothing in §8 forecloses that; it is the general case of
which one-atom mediation is the specialization the corpus supports.

**Slot mixing.** Cross-contractions between the fragment and environment slots' λ≥1 blocks —
`mu_h · v_env`, `Theta_h : G_env` — which see the *relative orientation* of a fragment and its
surroundings rather than each in isolation. Keeping the slots separate now is what makes this an
addable module instead of something already smeared into one vector.

---

## 11. What was removed from the previous draft of this document, and why

**The linear electrostatic response of the bonding energy** (`E_bond,elec = R_f : X_elec`, with `R`
nonlinear in geometry but the field entering only linearly). The motivation was to keep the
electrostatics pairwise additive. It is not pairwise additive here — the coupled response solve is
explicitly not — so the constraint bought nothing and cost an entire mechanism, a separate
electrostatic descriptor, and a parallel treatment for the viability score. Environment dependence
of bonding is now carried by `theta - theta_0`, unrestricted in form (§7).

**The universal parameterization network.** Replaced by per-composition experts (§2).

**Applicability as the arbiter between fragmentations.** `v_f` was to be softmaxed across the
decompositions of a geometry and fitted to `-|E_pol + E_ct|`, and the implementation in
`rsfff.train.loss.applicability_loss` does exactly that. It answers the wrong question: competition
is not a property of either fragment, and a score pooled over a whole fragment is not addressed to
the bond that is actually being relabeled. Replaced by the mediator (§8), which sees the shared atom
and both candidate hosts. `v_f` survives as a diagnostic.

Two earlier positions on that head are both superseded and worth recording, because the pair of
them is what showed the framing was wrong. The draft before this one had `V_s(h_f, eta_f)` and was
narrowed to the fragment slot alone on the argument that viability is a property of the fragment.
The implementation then widened it back to the joined slot on the argument that competition needs
`eta` — which is true, and is exactly why the quantity belongs to a third party rather than to
either expert.

**Free-atom anchoring, the per-species offset, and the free-atom polarizability penalty** (§5).

**Neural pair corrections** (§6).

---

## Guiding principle

The environment is allowed to do arbitrary, nonlinear, effective many-body work — quenched `C6`,
softened Slater exponents, environment-shifted bonding, charge transfer. Nothing about its *form* is
restricted.

What is restricted is its **address**. It enters through one named set of weights, on one input slot
that is identically zero when there is nothing around, and it is denied entry to the one-body sector
altogether. That is what makes "what is a property of this molecule" a question the model answers by
construction rather than a question we try to infer from a fit.
