# Hull sampling plan (complete model, environment invariants)

Opened 2026-11-02. Code, results and figure: `dev/rsfff/Claude outputs/hull_coverage/` (`hullkit.py` is the reusable part; `hull_analysis.png` is the summary figure).

## Idea

The 42 environment invariants of the complete model come from frozen components only: the other ions' frozen densities projected onto each ion's own probe densities. So any geometry can be placed in the network's input space without a trained model or a label. That gives three uses:
- measure which invariants carry independent information;
- measure whether a structure lies inside the region the training set covers;
- grow that region on purpose, by choosing the candidates that extend the training set's convex hull the most.

**Definitions used throughout:**
- The unit is one ion. The heads are per ion, and with no species embedding (`species_emb False` in m26) the network input is I = feats / feat_scale, the same function for every species.
- The hull is taken in the k leading PCA components of the training ions, whitened (k = 6 → 90 % of the variance; k = 10 → 97.7 %).
- Distance to the hull is solved as NNLS with the simplex constraint added as a stiff extra row, min ‖Pᵀλ − x‖² + M²(1ᵀλ − 1)², λ ≥ 0. A query costs about 0.5 ms against the hull vertices at k = 6.
- Vertices: the training hull has 675 at k = 6 and 3116 at k = 10, out of about 9.5k unique training ions.
- Danskin gives the gradient for biasing MD: ∇ₓd = (x − p\*)/d.

## Findings on existing data (m26 five_salts_polarfilt, 6572 frames, 17953 ions)

### 1. Data geometry

- TwoNN intrinsic dimension is about 1.8, because the set is mostly dimers and trimers from the MBE (n=2: 2334 frames; n=3: 4047; ≥4: 191).
- PCA: 90 % of the variance in 6 components, 99 % in about 12, 99.9 % in 24.
- Heavy tails: the scaled |S1|², |S2|² reach 80–90 against a 99th percentile below 2.

### 2. Redundancy

- **shell, dip, quad and donor are nearly the same features.** Thirty pairs of invariants have Spearman ρ > 0.98, almost all of them the same invariant on two of these four probes (0.98–0.997).
  - Part of this is by construction: the donor probe *is* the outer shell for cations, and the acceptor probe *is* the outer shell for anions.
- **Information imbalance,** Δ(rest → all 42) when one probe group is dropped. The floor (same nearest neighbour) is 0.0007.

  | dropped group | Δ |
  |---|---|
  | shell | 0.0007 (floor) |
  | dip | 0.0007 (floor) |
  | quad | 0.0007 (floor) |
  | donor | 0.0007 (floor) |
  | breath | 0.0017 |
  | acceptor | 0.0043 |
  | both frontier probes | 0.0045 |
  | all Coulomb (V) | 0.0050 |

- **Greedy forward selection** (single invariants, Δ(selected → all)):

  | selected so far | Δ |
  |---|---|
  | shell:V1.S1 | 0.36 |
  | + acceptor:V0_pen | 0.10 |
  | + acceptor:V1.S1 | 0.046 |
  | + donor:V0_pen | 0.029 |
  | + shell:\|V1\|² | 0.014 |
  | + breath:\|V2\|² | 0.005 |
  | 8 invariants | 0.002 |
  | 16 invariants | 0.0009 |

  - By probe group, quad + acceptor + breath (21 invariants) reach 0.0012.
- **Reading:**
  - The six labelled probes behave like about three radial probes: a compact one (any of shell/dip/quad/donor), a diffuse one (the acceptor, i.e. the Fukui f⁺ of the cations), and the l = 0 breathing shape.
  - Of the frontier pair, donor is redundant and acceptor is not, because it is the most diffuse probe.
  - V1·S1 (field–overlap alignment) is the most informative single invariant.
- **Caveat:** this measures redundancy where the data live (low-dimensional, dimer-dominated). It says nothing yet about what each head needs, and it must be repeated on the widened set.

### 3. Hull distance vs error on held-out frames (val + test, frame score = maximum over its ions)

- Fraction outside: at k = 6, 7.5 % of held-out ions and 15 % of frames; at k = 10, 27 % and 42 %.
- Median |error| per ion (kJ/mol), inside → outside the hull, k = 6:

  | | pol | Pauli | CT | total int. |
  |---|---|---|---|---|
  | trimers (1696) | 0.06 → 0.16 | 0.09 → 0.21 | 0.09 → 0.23 | 0.19 → 0.34 |
  | ≥ 4 ions (41: 19 in, 22 out) | 0.32 → 1.24 | 0.51 → 1.54 | 0.34 → 0.30 | 0.89 → 1.33 |

  - k = 10 gives the same picture.
- Being outside the hull doubles or triples the error for pol and Pauli. CT separates less clearly.
- **The continuous distance correlates weakly** with error (Spearman 0.1–0.4, and the many zeros dilute it).
  - The 5-nearest-neighbour distance tracks the Pauli error better (ρ ≈ 0.5).
  - So: hull membership is the extrapolation flag, and density inside the hull is a second signal.
- Out-of-hull dimers are few (about 20 of 753), but their CT and total errors are much larger (1.75 vs 0.22 and 2.1 vs 0.32 kJ/mol per ion).

### 4. NaCl MD pool (sample/nacl/round1–3, 12–30 ions, every 5th frame: 2823 frames, 37k ions)

- 40 % of pool ions lie outside the training hull (k = 6).
- **By Na–Cl coordination number** (r < 3.3 Å), fraction outside:

  | CN | 1 | 2 | 3 | 4 | 5 | 0 |
  |---|---|---|---|---|---|---|
  | outside | 0.14 | 0.01 | 0.33 | 0.94 | 0.99 | 0.68 |

  - CN 0 means no counter-ion within 3.3 Å.
  - Condensed-phase coordination is essentially absent from training, which is section 3.1 of the handoff, now measured.
- **The old film committee's σ_E does not track feature novelty:** frame-level Spearman between its σ_E and the hull distance is −0.17.
- **The pool contains broken frames:** like-ion contacts of 0.56–0.98 Å, and ions hundreds to thousands of Å away. Their invariants are 10⁴ in whitened units, and a pure hull criterion would pick them first. Any selector needs a physical guard.

### 5. Greedy hull extension vs alternatives

- **Setup:**
  - Guard applied first: like > 2.4 Å, unlike > 1.8 Å, every ion within 6 Å of another; 2567 of 2823 frames pass.
  - Selection draws from 500 frames of half the trajectories; coverage is measured on 500 frames of the other half.
- **Fraction of held-out pool ions outside the hull of train ∪ selected:**

  | frames selected | 0 | 5 | 10 | 20 | 40 | 80 |
  |---|---|---|---|---|---|---|
  | greedy (farthest frame, hull updated after each pick) | 0.41 | 0.12 | 0.08 | 0.06 | 0.045 | 0.038 |
  | random (3 seeds) | 0.41 | 0.32–0.35 | 0.26–0.28 | 0.15–0.21 | 0.12–0.17 | 0.08–0.11 |
  | committee σ_E | 0.41 | 0.41 | 0.34 | 0.23 | 0.22 | 0.21 |

- Ten greedy frames cover what random needs about 80 for.
- Greedy picks are physical: 16–30-ion clusters, closest unlike contacts 2.2–2.55 Å, like contacts 3.4–3.6 Å (closest like-ion pair per frame).
- **Caveat:** the metric is the same quantity greedy optimizes. It is evaluated on held-out trajectories, but whether the coverage gain becomes an error gain is what H3 tests.

### 6. Salt overlap (transferability precursor)

Fraction of the column salt's training ions that lie inside the row salt's training hull (k = 6, common whitener):

| hull \ ions | NaCl | LiF | LiI | KBr | CsI |
|---|---|---|---|---|---|
| NaCl | ≈1 | 0.01 | 0.32 | 0.41 | 0.24 |
| LiF | 0.38 | ≈1 | 0.23 | 0.24 | 0.09 |
| LiI | 0.06 | 0.05 | ≈1 | 0.07 | 0.17 |
| KBr | 0.09 | 0.00 | 0.13 | ≈1 | 0.27 |
| CsI | 0.09 | 0.01 | 0.04 | 0.07 | ≈1 |

Fraction inside the union of the other four salts' hulls:

| NaCl | LiF | LiI | KBr | CsI |
|---|---|---|---|---|
| 0.87 | 0.09 | 0.61 | 0.79 | 0.31 |

- Today the salts largely occupy different regions, with LiF almost entirely its own.
- The invariants are absolute: small ions at short distances give large overlaps. A "train on salts A, predict salt B" claim is therefore currently an extrapolation claim for LiF and CsI.

## Plan

### H0. Tooling: `hullkit` into rsfff
- Move `hullkit.py` into the library, e.g. `src/coverage/hull.py`. Functions: whitener, `hull_distance`, `extreme_points`, MaxVol + grade, kNN distance, TwoNN, information imbalance, greedy selection.
- Add a `coverage` CLI that takes a cache (or extxyz + parameter table) and writes per-ion distances.
- Store the hull (whitener + vertices) next to each checkpoint, so every model carries its own domain of applicability.
- Speed: the vertex reduction is a one-off cost (84 s at k = 6, 270 s at k = 10, CPU). Updates are incremental: only new points can become vertices, and distances only shrink as the hull grows (lazy greedy).

### H1. Feature set decision (before the expansion)
- **Supervised ablations** with `train.ionct`, same data, same seeds:
  - (a) all 42;
  - (b) without donor (frontier question);
  - (c) three probes: quad, or shell, + acceptor + breath, i.e. 21 invariants;
  - (d) the greedy top 8–10.
- Compare per-term MAEs, the ≥ 4-ion clusters and α. Keep the smallest set that loses nothing.
- **Consider replacing labelled probes with a small radial probe basis:** two or three exponents per species, e.g. compact and diffuse tied to the species' shells, plus l = 0. The redundancy says that is what the network uses.
- **Missing-feature check:** look for pairs of ions with near-identical I but large differences in residual (per-channel Δ^ov from the CT inversions, per-ion pol/Pauli where MBE allows). If such pairs exist, an invariant is missing.
- **Repeat sections 1–2 after the first hull round:** condensed-phase environments may un-redundant some probes.

### H2. Hull space
- Choose k (and the feature set) by the in/out error separation on held-out data, not by variance. Fewer features → a larger usable k before everything is "outside".
- Keep the raw-I hull (the network's own coordinates) as the reference. Report slog(I) as a robustness check, since the |S|² tails stretch the hull.
- Inside the hull, add a density term (kNN distance) to flag holes.

### H3. Candidate generation and selection (one round per size or system group)
- **Pools:**
  - MD with the current complete model: NaCl first, then the other salts, 20–60 ions.
  - Bulk and surface cuts at thermal geometries (handoff 3.1).
  - PCM ε and applied fields as extra knobs, which move the field invariants without distorting geometry.
- **Biased sampling:** hull-distance-biased MD, F = F_model + κ Jᵀ(x − p\*)/d. Use a softmax over the ions of a frame, warm-started NNLS, the same guards as the AL loop, and an energy window.
  - This needs no committee, and the gradient costs one featurizer backward pass.
- **Selection:**
  1. Physical guard (min contacts, no detached ions, energy window).
  2. Greedy hull extension with lazy updates, k frames per round.
  3. A small share (about 20 %) by kNN density inside the hull, for holes.
  4. A random 10 % as a control.
- **Labels:** one ALMO-EDA per frame (frz/pol/ct with `cls_elec`, `mod_pauli`) plus forces on the same frames (handoff 3.1, 3.4).

### H4. Loop and stopping
- Per round, retrain and recompute the hull. Track:
  - the fraction of new-MD ions outside;
  - error vs hull distance on a fixed held-out set;
  - the in/out error ratio.
- Stop a system when under about 5 % of ions from production-length MD are outside, and the in/out error ratio is close to 1.

### H5. Transferability experiment
- **Design:** training sets drawn from different salt combinations, sampled to matched coverage of a shared region of I-space. Matching uses the same hull or density over the intersection.
- **Compare** the fitted heads as functions on probe points in the intersection: the parameter factors c_α, the Pauli and C6 scales, and the CT gap terms.
- **Key figure:** error vs hull distance with within-salt and cross-salt test ions together. If they fall on one curve, error is set by coverage, not chemistry.
- **Mixed-salt clusters stay test-only** (handoff 3.2).
- **Prerequisite:** today's overlap is small (section 6). Either sample the overlap deliberately, or decide that the invariants should be normalized per species (open question 2).

## Open questions for Joe

1. Hull per head or one per ion? Every head reads all 42 today. If H1 gives heads different subsets, the hull should follow the head.
2. Should the invariants be made species-relative, e.g. divided by each probe's self-overlap or scaled by the ion's radius? That would let Li⁺ and Cs⁺ environments overlap. It trades the "same function of the physical environment" interpretation for coverage.
3. Is the stopping criterion acceptable, and is the k-per-round budget set by Q-Chem cost for 20–60-ion ALMO-EDA?
