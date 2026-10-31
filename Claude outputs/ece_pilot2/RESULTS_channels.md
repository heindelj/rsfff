# Per-bond dissociation channels (pyscf-all)

Diabat guess: dative bonds of over-valent cation centres heterolytic (NH4+ -> NH3 + H+, H3O+ -> H2O + H+,
CH3NH3+ -> CH3+ + NH3 / CH3NH2 + H+); acetate C-O -> CH3CO + O-; everything else homolytic.

Run 1 (all stretch frames held out, 80 epochs, channel onset 0.15 A): vs the homolytic ECE electrical
model, in-distribution unchanged (val mu 0.034 vs 0.031 D, alpha_iso 0.77 vs 0.83 %), stretch worse
for the cations' X-H bonds: NH4+ / H3O+ N-H / O-H at 1.4-1.5 r_e put ~0.3-0.5 e on the leaving H while
the labels still look homolytic there (mu error 0.84 / 0.72 D vs 0.08 / 0.10). No training frame lies
beyond the onset, so the transfer rate was the untrained prior (gamma = Morse beta).

Run 2 (stretch frames with r <= 1.3 r_e moved into training, test on the 78 frames beyond; 40
epochs; onset 0.15 A, per-bond-type learnable rates):

| test (r > 1.3 r_e) | homolytic | channels |
|---|---|---|
| all: mu (D) / alpha_iso % / F | 0.669 / 4.64 / 3.45 | 0.704 / 5.11 / 3.27 |
| acetate | 1.46 / 4.49 | 1.13 / 3.32 |
| methylammonium | 1.05 / 2.22 | 0.97 / 2.80 |
| ammonium | 0.21 / 5.00 | 1.29 / 11.68 |
| hydronium | 0.25 / 3.33 | 1.54 / 10.21 |

Same frames, models trained without any stretch data: mu 0.75 (homolytic) / 0.95 D (channels),
alpha_iso 7.5 / 8.1 %. Validation (in range) and probe frames are unchanged by the channels
(val mu 0.060 both; probe dE_int 0.42 / 0.41 kcal/mol).

Reading: the channels help where the heterolysis is under way inside the scanned range (acetate's O-,
CH3NH3+ C-N), and hurt for the protons of NH4+ / H3O+, whose transfer has not started by 1.5 r_e;
data to 1.3 r_e does not pin the rate down. The default onset is now 0.5 A (width 0.3 A,
`film.elec_channel_onset/width`) -- not yet retrained. Long scans of the charged molecules (2.5-3 r_e)
are what the channel rates need.
