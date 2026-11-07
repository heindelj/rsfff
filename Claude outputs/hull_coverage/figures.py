import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
# (a) feature selection curve
s = json.load(open('select_ii.json'))['single']
a = ax[0, 0]; y = [v for _, v in s]
a.plot(range(1, len(y) + 1), y, '-o', color=C[0], lw=2, ms=4)
a.axhline(2 / 2500, color=INK2, ls=':', lw=1); a.text(1, 2 / 2500 * 0.6, 'floor: same nearest neighbour', ha='left', color=INK2, fontsize=8)
a.set_yscale('log'); a.set_ylim(4e-4, 0.6); a.set_xlabel('number of invariants selected (greedy)'); a.set_ylabel('information imbalance Δ(selected → all 42)')
for i, (n, v) in enumerate(s[:6]):
    a.annotate(n, (i + 1, v), xytext=(6, 2), textcoords='offset points', fontsize=7, color=INK2)
a.set_title('(a) ~8–10 of 42 invariants reproduce the full neighbourhoods', loc='left', fontsize=9.5)
# (b) error inside vs outside the hull, held-out frames, k=6
r = pd.DataFrame(json.load(open('hull_err_k6.json'))['by_subset']).set_index('subset')
a = ax[0, 1]; terms = ['pol', 'pauli', 'ct', 'int']; x = np.arange(len(terms)); w = 0.36
for j, (sub, lab) in enumerate([('n=3', 'trimers'), ('n>=4', '≥4 ions')]):
    inn = [r.loc[sub, f'med_in_{t}'] for t in terms]; out = [r.loc[sub, f'med_out_{t}'] for t in terms]
    a.bar(x + (j - 0.5) * w * 1.02 - w / 4, inn, w / 2, color=C[0] if j == 0 else '#86b6ef', label=f'{lab}, inside')
    a.bar(x + (j - 0.5) * w * 1.02 + w / 4, out, w / 2, color=C[1] if j == 0 else '#f3a483', label=f'{lab}, outside')
a.set_xticks(x, ['pol', 'Pauli', 'CT', 'total int.']); a.set_ylabel('median |error| per ion (kJ/mol)')
a.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left'); a.set_ylim(0, 2.05); a.grid(axis='x', visible=False)
a.set_title('(b) held-out frames: error inside vs outside the training hull', loc='left', fontsize=9.5)
# (c) pool ions outside the training hull, by coordination
qs = pd.read_pickle('pool_sub_d_k6.pkl'); g = qs.groupby('cn').d.agg(n='count', f=lambda d: (d > 1e-5).mean())
g = g[g.n >= 20]
a = ax[1, 0]; a.bar(g.index, g.f, color=C[0], width=0.6)
for cn, row in g.iterrows(): a.text(cn, row.f + 0.02, f'n={int(row.n)}', ha='center', fontsize=7, color=INK2)
a.set_xlabel('Na–Cl coordination number of the ion (r < 3.3 Å)'); a.set_ylabel('fraction outside training hull')
a.set_ylim(0, 1.08); a.grid(axis='x', visible=False)
a.set_title('(c) NaCl MD pool (12–30 ions): CN ≥ 4 is unseen', loc='left', fontsize=9.5)
# (d) greedy selection coverage on held-out trajectories
gd = json.load(open('greedy_demo.json')); a = ax[1, 1]
for k_, col, lab in [('greedy_hull', C[0], 'greedy hull extension'), ('sigma_E', C[1], 'committee σ_E (old film committee)')]:
    v = np.array(gd[k_]); a.plot(v[:, 0], v[:, 1], '-o', color=col, lw=2, ms=4, label=lab)
R = np.array([np.array(gd[f'random{s}'])[:, 1] for s in range(3)]); m = np.array(gd['random0'])[:, 0]
a.fill_between(m, R.min(0), R.max(0), color=C[2], alpha=0.2, lw=0); a.plot(m, R.mean(0), '-o', color=C[2], lw=2, ms=4, label='random (3 seeds)')
a.set_xlabel('pool frames added to the training hull'); a.set_ylabel('fraction of held-out pool ions outside')
a.legend(frameon=False, fontsize=7.5); a.set_ylim(0, 0.45)
a.set_title('(d) greedy hull extension covers held-out trajectories fastest', loc='left', fontsize=9.5)
fig.text(0.01, 0.005, 'm26 five_salts_polarfilt features (42 sq invariants / feat_scale); hull in the 6 leading whitened PCs of the training ions.',
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.015, 1, 1)); fig.savefig('hull_analysis.png', dpi=160)
