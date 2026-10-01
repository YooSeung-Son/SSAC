"""R2/R3 (official UPR diagnostics), R6 (UPR+ vs official), R7 (next-season wins vs points), R8/R9 (rank)."""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from common import OUT, R, COMP, rankings, rank_at

res = {}
rng = np.random.default_rng(7)
# ------------------------------------------------------------------ tour-level official components
L = pd.read_parquet(OUT + 'long_all.parquet')
L['yr'] = L.date.dt.year
T = L[(L.src == 'tour') & L.tourney_level.isin(['G', 'M', 'A', 'F']) & ~L.team & ~L['round'].astype(str).str.startswith('Q')
      & ~L.wo & ~L.ret & L.yr.between(2012, 2025)].copy()
T['bps_k'], T['bps_n'] = T.s_bpSaved, T.s_bpFaced
T['bpc_k'], T['bpc_n'] = T.o_bpFaced - T.o_bpSaved, T.o_bpFaced
T['tbw_k'], T['tbw_n'] = T.tb_won, T.tb_n
T['dsw_k'], T['dsw_n'] = T.won * T.deciding, T.deciding * 1
T = T[T.has_stats]
T['spw'] = T.spw_k / T.spw_n; T['rpw'] = T.rpw_k / T.rpw_n; T['tpw'] = (T.spw_k + T.rpw_k) / (T.spw_n + T.rpw_n)
T = T.sort_values(['pid', 'date', 'mid'])


def rates(df):
    g = df.groupby('pid')
    out = pd.DataFrame({k: g[k + '_k'].sum() / g[k + '_n'].sum().replace(0, np.nan) for k in COMP})
    out['upr'] = 100 * out[COMP].sum(1, min_count=4)
    for k in COMP:
        out[k + '_n'] = g[k + '_n'].sum()
    out['n'] = g.size(); out['tpw'] = g.tpw.mean(); out['spw'] = g.spw.mean(); out['rpw'] = g.rpw.mean()
    return out


# R2: split-half reliability & stabilisation of each official component (player-seasons, >=20 tour matches)
S = T[T.yr.between(2015, 2025)].copy()
S['ps'] = S.pid.astype(str) + '_' + S.yr.astype(str)
cnt = S.groupby('ps').size(); S = S[S.ps.isin(cnt.index[cnt >= 20])]
S['half'] = S.groupby('ps').cumcount() % 2
H = {h: rates(S[S.half == h].assign(pid=lambda d: d.ps)) for h in (0, 1)}
F = rates(S.assign(pid=lambda d: d.ps))
r2 = {}
for k in COMP + ['upr']:
    a, b = H[0][k], H[1][k]; ok = a.notna() & b.notna()
    sh = pearsonr(a[ok], b[ok])[0]
    d = dict(split_half=sh)
    if k in COMP:
        p = F[k].mean(); n = F[k + '_n']
        obs = F[k].var(); noise = (F[k] * (1 - F[k]) / n).mean()
        tau2 = max(obs - noise, 1e-9)
        per_match = (F[k + '_n'] / F.n).mean()
        n50 = p * (1 - p) / tau2
        d.update(signal_share=max(obs - noise, 0) / obs, events_for_rel50=n50, matches_for_rel50=n50 / per_match,
                 events_per_match=per_match)
    r2[k] = d
r2['season_matches_median'] = float(F.n.median())
res['R2'] = r2
print('R2', json.dumps({k: {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v for k, v in r2.items()}, indent=0))

# R3: the official UPR re-counts ordinary skill
X = F.dropna(subset=['upr'])
r3 = {f'corr_{s}': pearsonr(X.upr, X[s])[0] for s in ['tpw', 'spw', 'rpw']}
fit = smf.ols('upr ~ spw + rpw', data=X).fit()
r3['r2_skill'] = fit.rsquared
resid = {}
for h in (0, 1):
    Hh = H[h].dropna(subset=['upr'])
    f = smf.ols('upr ~ spw + rpw', data=Hh).fit()
    resid[h] = Hh.upr - f.fittedvalues
ok = resid[0].index.intersection(resid[1].index)
r3['split_half_residual'] = pearsonr(resid[0][ok], resid[1][ok])[0]
r3['split_half_raw'] = r2['upr']['split_half']
res['R3'] = r3
print('R3', {k: round(v, 3) for k, v in r3.items()})

# ------------------------------------------------------------------ UPR+ from charted matches
PM = pd.read_parquet(OUT + 'mcp_player_match_adj.parquet')
PM['yr'] = pd.to_datetime(PM.date).dt.year
tot = PM.groupby('pid')[['e', 'v', 'w']].sum(); nm = PM.groupby('pid').size(); tot = tot[nm >= 20]
TAU2 = max((tot.e / tot.w).var() - (tot.v / tot.w ** 2).mean(), 1e-9)
res['tau2'] = TAU2; res['true_sd_pp'] = 100 * np.sqrt(TAU2)
# per-match leverage sum -> match win probability scale of 1 SD of clutch
res['mean_leverage_per_match'] = float(PM.w.mean())
res['wp_per_sd_per_match'] = float(np.sqrt(TAU2) * PM.w.mean())
print('tau2 %.3e  true SD %.3f pp ; 1 SD = %.2f pp match-win probability per match' % (TAU2, 100 * np.sqrt(TAU2), 100 * res['wp_per_sd_per_match']))


def upr_plus(sub):
    g = sub.groupby('pid')[['e', 'v', 'w']].sum()
    c = g.e / g.w; nv = g.v / g.w ** 2
    return pd.DataFrame({'uprp': c * TAU2 / (TAU2 + nv), 'uprp_raw': c, 'uprp_rel': TAU2 / (TAU2 + nv),
                         'n_ch': sub.groupby('pid').size()})


# R6-1: split-half on the SAME charted matches: official UPR (from charted points) vs UPR+
Pt = pd.read_parquet(OUT + 'mcp_points.parquet')
Mm = pd.read_parquet(OUT + 'mcp_matches.parquet').set_index('match_id')
Pt['set_no'] = Pt.Set1 + Pt.Set2
need = (Mm.bo + 1) // 2
rows = []
for side in (1, 2):
    sv = Pt[Pt.Svr == side]; rt = Pt[Pt.Svr != side]
    a = pd.DataFrame({'bps_k': sv[sv.is_bp].groupby('match_id').srv_won.sum(), 'bps_n': sv[sv.is_bp].groupby('match_id').size(),
                      'bpc_k': (1 - rt[rt.is_bp].srv_won).groupby(rt[rt.is_bp].match_id).sum(),
                      'bpc_n': rt[rt.is_bp].groupby('match_id').size()})
    tbl = Pt[Pt.in_tb].groupby(['match_id', 'set_no']).tail(1)
    a['tbw_k'] = (tbl.PtWinner == side).groupby(tbl.match_id).sum(); a['tbw_n'] = tbl.groupby('match_id').size()
    a = a.reindex(Mm.index).fillna(0)
    fin = Pt.groupby('match_id').apply(lambda d: ((d.Set1 == d.Set2) & (d.Set1 == need[d.name] - 1)).any(), include_groups=False)
    dec = (fin.reindex(Mm.index).fillna(False) & Mm.complete)
    a['dsw_n'] = dec * 1; a['dsw_k'] = (dec & (Mm.winner == side)) * 1
    a['pid'] = Mm['pid1' if side == 1 else 'pid2']; a['date'] = Mm.date
    rows.append(a.reset_index())
OF = pd.concat(rows).dropna(subset=['pid']).sort_values(['pid', 'date'])
OF = OF.merge(PM[['match_id', 'pid', 'e', 'v', 'w']], on=['match_id', 'pid'], how='inner')
n = OF.groupby('pid').size(); OF = OF[OF.pid.isin(n.index[n >= 20])]
OF['half'] = OF.groupby('pid').cumcount() % 2
hh = {}
for h in (0, 1):
    q = OF[OF.half == h].groupby('pid').sum(numeric_only=True)
    hh[h] = pd.DataFrame({'upr': 100 * sum(q[k + '_k'] / q[k + '_n'].replace(0, np.nan) for k in COMP), 'uprp': q.e / q.w})
okp = hh[0].dropna().index.intersection(hh[1].dropna().index)
sh_off = pearsonr(hh[0].loc[okp, 'upr'], hh[1].loc[okp, 'upr'])[0]
sh_plus = pearsonr(hh[0].loc[okp, 'uprp'], hh[1].loc[okp, 'uprp'])[0]
bs = []
idx = np.array(okp)
for _ in range(2000):
    s = rng.choice(idx, len(idx))
    bs.append(pearsonr(hh[0].loc[s, 'uprp'], hh[1].loc[s, 'uprp'])[0] - pearsonr(hh[0].loc[s, 'upr'], hh[1].loc[s, 'upr'])[0])
res['R6_reliability'] = dict(players=len(okp), split_half_official=sh_off, split_half_uprplus=sh_plus,
                             diff=sh_plus - sh_off, diff_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
print('R6 reliability (same charted matches)', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['R6_reliability'].items()})

# ------------------------------------------------------------------ panel: year-end t, window t-2..t
SM = pd.read_parquet(OUT + 'sack_markov.parquet'); SM['yr'] = SM.date.dt.year
rk = rankings()
pl = pd.read_csv(R + 'atp_players.csv', dtype={'dob': str}).set_index('player_id')
pl['dob'] = pd.to_datetime(pl.dob, format='%Y%m%d', errors='coerce')
eloend = L.sort_values('date').groupby(['pid', 'yr']).elo_post.last()
rowsP = []
for t in range(2014, 2025):
    up = upr_plus(PM[PM.yr.between(t - 2, t)]); up = up[up.n_ch >= 10]
    off = rates(T[T.yr.between(t - 2, t)])
    sk = SM[SM.yr.between(t - 2, t)].groupby('pid').agg(tpw_all=('tpw', 'mean'), conv_past=('conv', 'mean'), n_all=('won', 'size'))
    nx = SM[SM.yr == t + 1].groupby('pid').agg(conv_next=('conv', 'mean'), conv_next_sum=('conv', 'sum'), n_next=('won', 'size'),
                                                wins_next=('won', 'mean'))
    r0, r1 = rank_at(rk, f'{t}-12-31'), rank_at(rk, f'{t + 1}-12-31')
    for pid in up.index:
        rowsP.append(dict(pid=pid, t=t, uprp=up.uprp[pid], uprp_raw=up.uprp_raw[pid], uprp_rel=up.uprp_rel[pid], n_ch=up.n_ch[pid],
                          upr_off=off.upr.get(pid, np.nan), n_tour=off.n.get(pid, 0),
                          tpw=sk.tpw_all.get(pid, np.nan), conv_past=sk.conv_past.get(pid, np.nan), n_all=sk.n_all.get(pid, 0),
                          conv_next=nx.conv_next.get(pid, np.nan), conv_next_sum=nx.conv_next_sum.get(pid, np.nan),
                          n_next=nx.n_next.get(pid, 0), rank0=r0.get(pid, np.nan), rank1=r1.get(pid, np.nan),
                          elo0=eloend.get((pid, t), np.nan),
                          age=(pd.Timestamp(f'{t}-12-31') - pl.dob.get(pid, pd.NaT)).days / 365.25 if pd.notna(pl.dob.get(pid, pd.NaT)) else np.nan,
                          name=f"{pl.name_first.get(pid, '')} {pl.name_last.get(pid, '')}"))
D = pd.DataFrame(rowsP)
D['d_lrank'] = np.log(D.rank0) - np.log(D.rank1)           # >0 = ranking improved
for k in ['uprp', 'upr_off', 'conv_past']:
    D[k + '_z'] = (D[k] - D[k].mean()) / D[k].std()
D.to_csv(OUT + 'panel_uprplus.csv', index=False)
print('panel rows', len(D), 'players', D.pid.nunique(), '| with next-season matches', int((D.n_next >= 10).sum()))

ctl = ' + tpw + np.log(rank0) + age + np.log(n_ch) + C(t)'


def eff(y, x, d, extra=''):
    ev = [v.strip() for v in extra.replace('+', ' ').split() if v.strip()]
    d = d.dropna(subset=[y, x, 'tpw', 'rank0', 'age'] + ev).copy()
    f = smf.ols(f'{y} ~ {x}' + ctl + extra, data=d).fit(cov_type='cluster', cov_kwds={'groups': d.pid})
    return dict(b=float(f.params[x]), lo=float(f.conf_int().loc[x, 0]), hi=float(f.conf_int().loc[x, 1]), p=float(f.pvalues[x]), n=int(f.nobs))


R7d = D[D.n_next >= 10]
res['R7'] = {x: eff('conv_next', x, R7d) for x in ['uprp_z', 'upr_off_z', 'conv_past_z']}
res['R7_both'] = eff('conv_next', 'uprp_z', R7d, ' + upr_off_z')
for k, v in res['R7'].items():
    print('R7 next-season wins vs points per match, per SD', k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
print('R7 UPR+ controlling official UPR', {a: round(b, 4) if isinstance(b, float) else b for a, b in res['R7_both'].items()})
# bootstrap the R6 predictive comparison (difference of per-SD effects), clustered by player
pids = R7d.pid.unique(); diffs = []
base = R7d.dropna(subset=['conv_next', 'uprp_z', 'upr_off_z', 'tpw', 'rank0', 'age'])
for _ in range(300):
    s = rng.choice(pids, len(pids)); d = pd.concat([base[base.pid == p] for p in s])
    try:
        b1 = smf.ols('conv_next ~ uprp_z' + ctl, data=d).fit().params['uprp_z']
        b2 = smf.ols('conv_next ~ upr_off_z' + ctl, data=d).fit().params['upr_off_z']
        diffs.append(b1 - b2)
    except Exception:
        pass
res['R6_predict_diff'] = dict(diff=res['R7']['uprp_z']['b'] - res['R7']['upr_off_z']['b'],
                              ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
print('R6 predictive diff (UPR+ - official) per SD', res['R6_predict_diff'])
# quintiles (Figure 1 data)
for k in ['uprp', 'upr_off']:
    R7d[k + '_q'] = R7d.groupby('t')[k].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
qt = {k: R7d.groupby(k + '_q').conv_next.agg(['mean', 'sem', 'size']) for k in ['uprp', 'upr_off']}
for k, v in qt.items():
    print('quintiles', k, (v['mean'] * 100).round(2).tolist())
res['R7_quintiles'] = {k: v.reset_index().to_dict('list') for k, v in qt.items()}

# R8: ranking change; R9: ranks 51-150
R8d = D.dropna(subset=['rank1', 'rank0'])
res['R8'] = {x: eff('d_lrank', x, R8d) for x in ['uprp_z', 'upr_off_z']}
R9d = R8d[R8d.rank0.between(51, 150)]
res['R9'] = {x: eff('d_lrank', x, R9d) for x in ['uprp_z', 'upr_off_z']}
for lab in ['R8', 'R9']:
    for k, v in res[lab].items():
        print(lab, 'log-rank change per SD', k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()},
              ' -> %+.1f%% ranking' % (100 * (np.exp(v['b']) - 1)))
q = R9d.copy(); q['top'] = (q.uprp >= q.groupby('t').uprp.transform('median')) * 1
res['R9_halves'] = q.groupby('top').agg(n=('pid', 'size'), d_lrank=('d_lrank', 'mean'), rank0=('rank0', 'median'),
                                       rank1=('rank1', 'median')).reset_index().to_dict('list')
print('R9 halves', res['R9_halves'])
cases = R9d.sort_values('uprp', ascending=False)[['name', 't', 'rank0', 'rank1', 'uprp', 'uprp_rel', 'n_ch', 'conv_next', 'tpw']].head(15)
print(cases.to_string())
json.dump(res, open(OUT + 'panel.json', 'w'), indent=1, default=float)
