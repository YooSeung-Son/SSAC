"""R1 (big points decide matches) and R5 (is point-level clutch real?) on Match Charting Project data.

Per point i: residual r_i = outcome - baseline, baseline = that player's serve (or return) point win
rate in the same match. Clutch WPA of a player in a match = sum_i I_i * r_i with I_i the Markov leverage.
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

from common import OUT
P = pd.read_parquet(OUT + 'mcp_points.parquet')
M = pd.read_parquet(OUT + 'mcp_matches.parquet')
P = P[P.match_id.isin(M.match_id)].copy()
P = P.merge(M[['match_id', 'pid1', 'pid2', 'date', 'Surface', 'complete', 'exp1', 'winner', 'tpw1']], on='match_id')
grp = P.groupby(['match_id', 'Svr'])
P['b'] = grp.srv_won.transform('mean')                     # server's own match baseline
P['r'] = P.srv_won - P.b                                    # server residual
P['Ic'] = P.imp - grp.imp.transform('mean')                # leverage centred within match-server (baseline uses same points)
P['d1'] = np.where(P.Svr == 1, P.r, -P.r)                   # residual from player 1's perspective
q85 = P.imp.quantile(0.85)
P['big'] = P.imp >= q85
res = {'n_points': int(len(P)), 'n_matches': int(P.match_id.nunique()), 'big_threshold': float(q85),
       'big_share_points': float(P.big.mean()), 'big_share_leverage': float(P.loc[P.big, 'imp'].sum() / P.imp.sum())}

# ---------------- R1: what explains the match outcome beyond points? ----------------
mm = P.groupby('match_id').apply(lambda d: pd.Series({
    'wpa_big': (d.imp * d.d1)[d.big].sum(), 'wpa_oth': (d.imp * d.d1)[~d.big].sum(),
    'raw_big': d.d1[d.big].sum(), 'raw_oth': d.d1[~d.big].sum()}), include_groups=False)
Mx = M.set_index('match_id').join(mm)
C = Mx[Mx.complete].copy()
C['won1'] = (C.winner == 1) * 1.
C['resid'] = C.won1 - C.exp1
r_big = pearsonr(C.resid, C.wpa_big)[0]; r_oth = pearsonr(C.resid, C.wpa_oth)[0]
X = np.c_[np.ones(len(C)), C.wpa_big, C.wpa_oth]
beta, *_ = np.linalg.lstsq(X, C.resid, rcond=None)
r2_all = 1 - ((C.resid - X @ beta) ** 2).sum() / ((C.resid - C.resid.mean()) ** 2).sum()
r2_big = pearsonr(C.resid, C.wpa_big)[0] ** 2; r2_oth = pearsonr(C.resid, C.wpa_oth)[0] ** 2
fewer = ((C.tpw1 < 0.5) & (C.won1 == 1)) | ((C.tpw1 > 0.5) & (C.won1 == 0))
res['R1'] = dict(n=len(C), corr_big=r_big, corr_other=r_oth, r2_big=r2_big, r2_other=r2_oth, r2_both=r2_all,
                 won_with_fewer_points=float(fewer.mean()), sd_resid=float(C.resid.std()))
print('R1', {k: round(v, 3) if isinstance(v, float) else v for k, v in res['R1'].items()})
print('big points: %.1f%% of points, %.1f%% of total leverage' % (100 * res['big_share_points'], 100 * res['big_share_leverage']))

# ---------------- player-match clutch table ----------------
def player_match(P, col='srv_won'):
    """clutch WPA per player-match; noise variance from the binomial with centred leverage."""
    r = P[col] - P.b
    e = P.Ic * r                                   # centred weights: identical sum, correct null variance
    v = P.Ic ** 2 * P.b * (1 - P.b)
    df = pd.DataFrame({'match_id': P.match_id, 'Svr': P.Svr, 'e': e, 'v': v, 'w': P.imp, 'bp': P.is_bp, 'tb': P.in_tb})
    s = df.groupby(['match_id', 'Svr']).agg(e=('e', 'sum'), v=('v', 'sum'), w=('w', 'sum'))
    s = s.reset_index()
    a = s.pivot(index='match_id', columns='Svr', values=['e', 'v', 'w'])
    out = []
    for pl, other in [(1, 2), (2, 1)]:
        out.append(pd.DataFrame({'match_id': a.index, 'side': pl,
                                 'e_srv': a[('e', pl)].values, 'e_ret': -a[('e', other)].values,
                                 'v': (a[('v', 1)] + a[('v', 2)]).values, 'w': (a[('w', 1)] + a[('w', 2)]).values}))
    o = pd.concat(out)
    o['e'] = o.e_srv + o.e_ret
    return o

PM = player_match(P)
mi = M.set_index('match_id')
PM['pid'] = np.where(PM.side == 1, mi.pid1.reindex(PM.match_id).values, mi.pid2.reindex(PM.match_id).values)
PM['date'] = mi.date.reindex(PM.match_id).values
PM['surface'] = mi.Surface.reindex(PM.match_id).values
PM = PM.dropna(subset=['pid']).sort_values(['pid', 'date'])
PM.to_parquet(OUT + 'mcp_player_match.parquet')


def summarize(PM, min_m=20):
    """true between-player SD of the clutch rate, split-half reliability (odd/even matches)."""
    g = PM.groupby('pid')
    n = g.size(); keep = n.index[n >= min_m]
    Q = PM[PM.pid.isin(keep)].copy()
    Q['half'] = Q.groupby('pid').cumcount() % 2
    agg = Q.groupby(['pid', 'half'])[['e', 'v', 'w']].sum().unstack('half')
    rate = lambda h: agg[('e', h)] / agg[('w', h)]
    sh = pearsonr(rate(0), rate(1))[0]
    tot = Q.groupby('pid')[['e', 'v', 'w']].sum()
    c = tot.e / tot.w; nv = tot.v / tot.w ** 2
    tau2 = c.var() - nv.mean()
    return dict(players=len(keep), split_half=sh, spearman_brown=2 * sh / (1 + sh), tau2=tau2,
                true_sd_pp=100 * np.sqrt(max(tau2, 0)), noise_sd_pp=100 * np.sqrt(nv.mean()),
                var_ratio=c.var() / nv.mean())

obs = summarize(PM)
print('R5 observed', {k: round(v, 4) if isinstance(v, float) else v for k, v in obs.items()})
# bootstrap CI over players for tau2 / true SD
rng = np.random.default_rng(0)
tot = PM.groupby('pid')[['e', 'v', 'w']].sum(); nm = PM.groupby('pid').size()
tot = tot[nm >= 20]
c = (tot.e / tot.w).values; nv = (tot.v / tot.w ** 2).values
bs = []
for _ in range(2000):
    i = rng.integers(0, len(c), len(c)); bs.append(c[i].var(ddof=1) - nv[i].mean())
bs = np.array(bs)
obs['tau2_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
print('tau2 95% CI', obs['tau2_ci'], 'true SD pp CI', [100 * np.sqrt(max(x, 0)) for x in obs['tau2_ci']])

# permutation null: shuffle point outcomes within match x server (keeps every baseline, breaks leverage link)
perm = []
key = (P.match_id.astype('category').cat.codes.values.astype(np.int64) * 4 + P.Svr.values.astype(np.int64))
for k in range(40):
    u = rng.random(len(P))
    order = np.lexsort((u, key))
    sw = np.empty(len(P), dtype=np.int64)
    sorted_pos = np.lexsort((np.arange(len(P)), key))      # original positions grouped by key
    sw[sorted_pos] = P.srv_won.values[order]
    Pp = P.assign(srv_perm=sw)
    s = summarize(player_match(Pp, 'srv_perm').assign(
        pid=lambda d: np.where(d.side == 1, mi.pid1.reindex(d.match_id).values, mi.pid2.reindex(d.match_id).values),
        date=lambda d: mi.date.reindex(d.match_id).values).dropna(subset=['pid']).sort_values(['pid', 'date']))
    perm.append(s)
pf = pd.DataFrame(perm)
obs['perm_split_half_max'] = float(pf.split_half.max()); obs['perm_split_half_mean'] = float(pf.split_half.mean())
obs['perm_var_ratio_max'] = float(pf.var_ratio.max()); obs['perm_var_ratio_mean'] = float(pf.var_ratio.mean())
obs['perm_p_split_half'] = float((pf.split_half >= obs['split_half']).mean())
print('perm null: split-half mean %.3f max %.3f | var ratio mean %.3f max %.3f | p=%.3f' %
      (pf.split_half.mean(), pf.split_half.max(), pf.var_ratio.mean(), pf.var_ratio.max(), obs['perm_p_split_half']))
res['R5'] = obs
# sensitivity: serve-side vs return-side, by surface, min matches
sens = {}
for lab, sub in [('serve_only', PM.assign(e=PM.e_srv)), ('return_only', PM.assign(e=PM.e_ret)),
                 ('min40', None)] + [(f'surf_{s}', PM[PM.surface == s]) for s in ['Hard', 'Clay', 'Grass']]:
    sens[lab] = summarize(PM, 40) if lab == 'min40' else summarize(sub, 20 if not lab.startswith('surf') else 15)
    print(lab, {k: round(v, 4) if isinstance(v, float) else v for k, v in sens[lab].items()})
res['R5_sens'] = sens
json.dump(res, open(OUT + 'mcp_clutch.json', 'w'), indent=1, default=float)
