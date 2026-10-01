"""Figure 1 (v3): next-season wins above the points-based expectation by quintile of UPR+ and of the official rating.
Adjusted means: outcome residualised on the controls (points won, log rank, age, log charted volume, season), plus the
grand mean; 95% CI from the player-clustered SE of each quintile mean.
usage: python fig_quint.py <width_pt> <height_pt>"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from common import OUT, LATEX

W_PT, H_PT = float(sys.argv[1]), float(sys.argv[2])
D = pd.read_csv(OUT + 'panel_v3.csv').dropna(subset=['conv_next', 'tpw', 'rank0', 'age', 'uprp', 'upr_off'])
stats = {}
for k in ['uprp', 'upr_off']:
    D['q'] = D.groupby('t')[k].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    f = smf.ols('conv_next ~ C(q) + tpw + np.log(rank0) + age + np.log(n_ch) + C(t)', data=D).fit(
        cov_type='cluster', cov_kwds={'groups': D.pid})
    V = f.cov_params().values
    rows = []
    for q in range(5):
        X = f.model.data.orig_exog.copy()
        for j in range(1, 5):
            X[f'C(q)[T.{j}]'] = 1.0 if q == j else 0.0
        a_ = X.mean().values                                   # marginal mean over the sample's covariates
        m = a_ @ f.params.values; se = np.sqrt(a_ @ V @ a_)
        rows.append((100 * m, 100 * 1.96 * se, int((D.q == q).sum())))
    stats[k] = np.array(rows)
    print(k, np.round(stats[k], 2).tolist(), 'Q5-Q1 %.2f' % (stats[k][4, 0] - stats[k][0, 0]))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'pdf.fonttype': 42})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
fig, ax = plt.subplots(figsize=(W_PT / 72.27, H_PT / 72.27))
x = np.arange(1, 6)
ax.axhline(0, color=MUTED, lw=0.6, zorder=1); ax.grid(axis='y', color=GRID, lw=0.4, zorder=0)
for k, off, col, lab in [('uprp', -0.12, BLUE, 'UPR+'), ('upr_off', 0.12, ORANGE, 'Official UPR')]:
    s = stats[k]
    ax.errorbar(x + off, s[:, 0], yerr=s[:, 1], fmt='o', ms=4, color=col, ecolor=col, elinewidth=1, capsize=0,
                mec='white', mew=0.6, zorder=3, label=lab)
    ax.plot(x + off, s[:, 0], color=col, lw=1, alpha=0.5, zorder=2)
ax.text(5 - 0.12 - 0.13, stats['uprp'][4, 0], 'UPR+', color=INK, ha='right', va='center', fontsize=6.5)
ax.text(5 + 0.12, stats['upr_off'][4, 0] - stats['upr_off'][4, 1] - 0.12, 'Official', color=INK, ha='center', va='top', fontsize=6.5)
ax.set_xticks(x, ['Q1\nlowest', 'Q2', 'Q3', 'Q4', 'Q5\nhighest'])
ax.set_xlim(0.55, 5.45)
ax.set_xlabel('Quintile of the signal (three prior seasons)', labelpad=1.5, color=INK)
ax.set_ylabel('Next-season wins vs. points (pp)', labelpad=1.5, color=INK)
for s_ in ['top', 'right']:
    ax.spines[s_].set_visible(False)
for s_ in ['left', 'bottom']:
    ax.spines[s_].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK)
ax.legend(loc='upper left', frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2, handlelength=1.0)
fig.subplots_adjust(left=0.165, right=0.985, top=0.965, bottom=0.215)
fig.savefig(LATEX + 'figure1.pdf')
fig.savefig(OUT + 'figure1_preview.png', dpi=300)
