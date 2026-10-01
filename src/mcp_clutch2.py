"""R5 robustness: remove population-level score-state effects before measuring player clutch.

baseline_adj: logit P(server wins) = logit(server's match (or set) rate) + population effects of
score state (game score, tiebreak state) and leverage decile. Clutch = leverage-weighted deviation from
baseline_adj, re-centred within match x server so it stays relative to the player's own level.
Null: outcomes simulated from baseline_adj (keeps every population effect, no player clutch).
"""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from common import OUT
P = pd.read_parquet(OUT + 'mcp_points.parquet')
M = pd.read_parquet(OUT + 'mcp_matches.parquet')
P = P[P.match_id.isin(M.match_id)].copy().reset_index(drop=True)
mi = M.set_index('match_id')
P['set_no'] = P.Set1 + P.Set2
logit = lambda p: np.log(p / (1 - p))


def state_cols(P):
    g = P.Pts.astype(str).str.strip()
    gm = np.where(P.in_tb, 'TB', g)
    gm = pd.Series(gm).replace({'AD-40': '40-30', '40-AD': '30-40'}).values   # advantage == 40-30 / 30-40
    tb = np.where(P.in_tb, np.clip(P.Pts.astype(str).str.split('-').str[0].apply(lambda x: int(x) if str(x).isdigit() else 0)
                                   - P.Pts.astype(str).str.split('-').str[1].apply(lambda x: int(x) if str(x).isdigit() else 0), -2, 2), 9)
    dec = pd.qcut(P.imp.rank(method='first'), 10, labels=False)
    X = pd.get_dummies(pd.DataFrame({'gs': gm, 'tbd': tb.astype(str), 'dec': dec.astype(str)}), drop_first=True).astype(float)
    return X


def adjusted_baseline(P, level='match'):
    keys = ['match_id', 'Svr'] + (['set_no'] if level == 'set' else [])
    g = P.groupby(keys).srv_won
    k, n = g.transform('sum'), g.transform('size')
    b0 = ((k + 1) / (n + 2)).clip(0.05, 0.95)
    X = state_cols(P)
    fit = sm.GLM(P.srv_won.values, sm.add_constant(X.values), family=sm.families.Binomial(), offset=logit(b0).values).fit()
    eta = logit(b0).values + sm.add_constant(X.values) @ fit.params
    b = 1 / (1 + np.exp(-eta))
    return b, fit, list(X.columns)


def player_match(P, y, b, keys=('match_id', 'Svr')):
    r = y - b
    r = r - pd.Series(r).groupby([P[k] for k in keys]).transform('mean').values       # relative to own level
    Ic = P.imp.values - P.groupby(list(keys)).imp.transform('mean').values
    e = P.imp.values * r
    v = Ic ** 2 * b * (1 - b)
    df = pd.DataFrame({'match_id': P.match_id.values, 'Svr': P.Svr.values, 'e': e, 'v': v, 'w': P.imp.values})
    a = df.groupby(['match_id', 'Svr'])[['e', 'v', 'w']].sum().unstack('Svr')
    out = []
    for pl, other in [(1, 2), (2, 1)]:
        out.append(pd.DataFrame({'match_id': a.index, 'side': pl, 'e_srv': a[('e', pl)].values, 'e_ret': -a[('e', other)].values,
                                 'v_srv': a[('v', pl)].values, 'v_ret': a[('v', other)].values,
                                 'w': (a[('w', 1)] + a[('w', 2)]).values}))
    o = pd.concat(out)
    o['e'] = o.e_srv + o.e_ret; o['v'] = o.v_srv + o.v_ret
    o['pid'] = np.where(o.side == 1, mi.pid1.reindex(o.match_id).values, mi.pid2.reindex(o.match_id).values)
    o['date'] = mi.date.reindex(o.match_id).values
    o['surface'] = mi.Surface.reindex(o.match_id).values
    return o.dropna(subset=['pid']).sort_values(['pid', 'date'])


def summarize(PM, min_m=20, col='e', vcol='v'):
    n = PM.groupby('pid').size(); keep = n.index[n >= min_m]
    Q = PM[PM.pid.isin(keep)].copy(); Q['half'] = Q.groupby('pid').cumcount() % 2
    agg = Q.groupby(['pid', 'half'])[[col, 'w']].sum().unstack('half')
    sh = pearsonr(agg[(col, 0)] / agg[('w', 0)], agg[(col, 1)] / agg[('w', 1)])[0]
    tot = Q.groupby('pid')[[col, vcol, 'w']].sum()
    c = tot[col] / tot.w; nv = tot[vcol] / tot.w ** 2
    return dict(players=len(keep), split_half=sh, var_ratio=c.var() / nv.mean(),
                true_sd_pp=100 * np.sqrt(max(c.var() - nv.mean(), 0)), noise_sd_pp=100 * np.sqrt(nv.mean()))


res = {}
rng = np.random.default_rng(1)
for level in ['match', 'set']:
    b, fit, cols = adjusted_baseline(P, level)
    if level == 'match':
        dec = [c for c in cols if c.startswith('dec_')]
        res['pop_leverage_effects'] = {c: float(fit.params[1 + cols.index(c)]) for c in dec}
        print('population leverage-decile effects (logit, vs lowest decile):', {c: round(res['pop_leverage_effects'][c], 3) for c in dec})
        P['b_adj'] = b
    PM = player_match(P, P.srv_won.values, b, ('match_id', 'Svr') if level == 'match' else ('match_id', 'Svr', 'set_no'))
    o = summarize(PM)
    o_srv = summarize(PM.assign(e=PM.e_srv, v=PM.v_srv)); o_ret = summarize(PM.assign(e=PM.e_ret, v=PM.v_ret))
    # parametric null keeps population state effects
    nulls = []
    for k in range(40):
        ysim = (rng.random(len(P)) < b).astype(float)
        nulls.append(summarize(player_match(P, ysim, b, ('match_id', 'Svr') if level == 'match' else ('match_id', 'Svr', 'set_no'))))
    nf = pd.DataFrame(nulls)
    o.update(null_sh_mean=float(nf.split_half.mean()), null_sh_max=float(nf.split_half.max()),
             null_vr_mean=float(nf.var_ratio.mean()), null_vr_max=float(nf.var_ratio.max()),
             p_sh=float((nf.split_half >= o['split_half']).mean()), serve=o_srv, ret=o_ret)
    res[f'R5_adj_{level}'] = o
    print(level, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k not in ('serve', 'ret')})
    print('   serve side', {k: round(v, 3) if isinstance(v, float) else v for k, v in o_srv.items()})
    print('   return side', {k: round(v, 3) if isinstance(v, float) else v for k, v in o_ret.items()})
    if level == 'match':
        PM.to_parquet(OUT + 'mcp_player_match_adj.parquet')
        # does clutch track ability? (confound check)
        tot = PM.groupby('pid')[['e', 'w']].sum(); n = PM.groupby('pid').size(); tot = tot[n >= 20]
        M2 = M.copy()
        ab = pd.concat([M2[['pid1', 'tpw1']].rename(columns={'pid1': 'pid', 'tpw1': 'tpw'}),
                        M2[['pid2', 'tpw1']].assign(tpw=lambda d: 1 - d.tpw1).rename(columns={'pid2': 'pid'})[['pid', 'tpw']]])
        ab = ab.groupby('pid').tpw.mean()
        c = tot.e / tot.w
        res['clutch_vs_ability_r'] = float(pearsonr(c, ab.reindex(c.index))[0])
        print('corr(clutch, points-won share) = %.3f' % res['clutch_vs_ability_r'])
for lab, sub in [('min40', 40)]:
    PMa = pd.read_parquet(OUT + 'mcp_player_match_adj.parquet')
    res['R5_adj_min40'] = summarize(PMa, 40); print('min40', res['R5_adj_min40'])
PMa = pd.read_parquet(OUT + 'mcp_player_match_adj.parquet')
for s in ['Hard', 'Clay', 'Grass']:
    res[f'R5_adj_{s}'] = summarize(PMa[PMa.surface == s], 15); print(s, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f'R5_adj_{s}'].items()})
P[['match_id', 'Pt', 'b_adj']].to_parquet(OUT + 'mcp_points_badj.parquet')
json.dump(res, open(OUT + 'mcp_clutch2.json', 'w'), indent=1, default=float)
