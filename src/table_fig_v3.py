"""Table 1 (official components vs UPR+) and Figure 1 data (adjusted quintiles) for the MCP abstract."""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import pearsonr
from common import OUT, COMP

TAU2 = json.load(open(OUT + 'panel.json'))['tau2']
# ---------------- reliability on the SAME charted matches (career, players >= 20 charted matches) ----------------
Pt = pd.read_parquet(OUT + 'mcp_points.parquet')
Mm = pd.read_parquet(OUT + 'mcp_matches.parquet').set_index('match_id')
PM = pd.read_parquet(OUT + 'mcp_player_match_adj.parquet')
Pt['set_no'] = Pt.Set1 + Pt.Set2
need = (Mm.bo + 1) // 2
fin = Pt.groupby('match_id').apply(lambda d: ((d.Set1 == d.Set2) & (d.Set1 == need[d.name] - 1)).any(), include_groups=False)
dec = (fin.reindex(Mm.index).fillna(False) & Mm.complete)
tbl = Pt[Pt.in_tb].groupby(['match_id', 'set_no']).tail(1)
rows = []
for side in (1, 2):
    sv = Pt[Pt.Svr == side]; rt = Pt[Pt.Svr != side]
    a = pd.DataFrame(index=Mm.index)
    a['bps_k'] = sv[sv.is_bp].groupby('match_id').srv_won.sum(); a['bps_n'] = sv[sv.is_bp].groupby('match_id').size()
    a['bpc_k'] = (1 - rt[rt.is_bp].srv_won).groupby(rt[rt.is_bp].match_id).sum(); a['bpc_n'] = rt[rt.is_bp].groupby('match_id').size()
    a['tbw_k'] = (tbl.PtWinner == side).groupby(tbl.match_id).sum(); a['tbw_n'] = tbl.groupby('match_id').size()
    a['dsw_n'] = dec * 1; a['dsw_k'] = (dec & (Mm.winner == side)) * 1
    a['pts_k'] = np.where(side == 1, Mm.tpw1, 1 - Mm.tpw1)
    a = a.fillna(0); a['pid'] = Mm['pid1' if side == 1 else 'pid2']; a['date'] = Mm.date
    rows.append(a.reset_index())
OF = pd.concat(rows).dropna(subset=['pid'])
OF = OF.merge(PM[['match_id', 'pid', 'e', 'v', 'w']], on=['match_id', 'pid'], how='inner').sort_values(['pid', 'date'])
n = OF.groupby('pid').size(); OF = OF[OF.pid.isin(n.index[n >= 20])]
OF['half'] = OF.groupby('pid').cumcount() % 2


def comp_rates(Q):
    g = Q.groupby('pid').sum(numeric_only=True)
    r = pd.DataFrame({k: g[k + '_k'] / g[k + '_n'].replace(0, np.nan) for k in COMP})
    r['upr'] = 100 * r[COMP].sum(1, min_count=4)
    r['uprp'] = g.e / g.w
    for k in COMP:
        r[k + '_n'] = g[k + '_n']
    r['v'] = g.v; r['w'] = g.w; r['m'] = Q.groupby('pid').size(); r['skill'] = Q.groupby('pid').pts_k.mean()
    return r


H0, H1, A = comp_rates(OF[OF.half == 0]), comp_rates(OF[OF.half == 1]), comp_rates(OF)
T1 = {}
for k in COMP + ['upr', 'uprp']:
    ok = H0[k].notna() & H1[k].notna()
    d = dict(split_half=pearsonr(H0.loc[ok, k], H1.loc[ok, k])[0], skill_r=pearsonr(A[k].dropna(), A.loc[A[k].notna(), 'skill'])[0])
    if k in COMP:
        obs = A[k].var(); noise = (A[k] * (1 - A[k]) / A[k + '_n']).mean(); tau2 = max(obs - noise, 1e-9)
        p = A[k].mean(); per_m = (A[k + '_n'] / A.m).mean()
        d['m50'] = p * (1 - p) / tau2 / per_m
    elif k == 'uprp':
        d['m50'] = float(((A.v / A.w ** 2) * A.m).median() / TAU2)
    else:
        sb = d['split_half']; rel = 2 * sb / (1 + sb)            # matches for the full UPR via Spearman-Brown
        d['m50'] = float(A.m.median() * (1 / max(rel, 1e-6) - 1)) if rel > 0 else np.inf
    T1[k] = d
print('reliability (charted, >=20 matches):', {k: {a: round(b, 3) for a, b in v.items()} for k, v in T1.items()})
print('players', len(A), 'median charted matches', A.m.median())

# ---------------- predictive columns (panel; official components from tour-level window) ----------------
D = pd.read_csv(OUT + 'panel_uprplus.csv')
L = pd.read_parquet(OUT + 'long_all.parquet'); L['yr'] = L.date.dt.year
T = L[(L.src == 'tour') & L.tourney_level.isin(['G', 'M', 'A', 'F']) & ~L.team & ~L['round'].astype(str).str.startswith('Q')
      & ~L.wo & ~L.ret & L.has_stats].copy()
T['bps_k'], T['bps_n'] = T.s_bpSaved, T.s_bpFaced
T['bpc_k'], T['bpc_n'] = T.o_bpFaced - T.o_bpSaved, T.o_bpFaced
T['tbw_k'], T['tbw_n'] = T.tb_won, T.tb_n
T['dsw_k'], T['dsw_n'] = T.won * T.deciding, T.deciding * 1
comp = []
for t in sorted(D.t.unique()):
    g = T[T.yr.between(t - 2, t)].groupby('pid')
    c = pd.DataFrame({k: g[k + '_k'].sum() / g[k + '_n'].sum().replace(0, np.nan) for k in COMP}); c['t'] = t
    comp.append(c.reset_index())
D = D.merge(pd.concat(comp), on=['pid', 't'], how='left')
for k in COMP:
    D[k + '_z'] = (D[k] - D[k].mean()) / D[k].std()
ctl = ' + tpw + np.log(rank0) + age + np.log(n_ch) + C(t)'


def eff(y, x, d):
    d = d.dropna(subset=[y, x, 'tpw', 'rank0', 'age']).copy()
    f = smf.ols(f'{y} ~ {x}' + ctl, data=d).fit(cov_type='cluster', cov_kwds={'groups': d.pid})
    return [float(f.params[x]), float(f.conf_int().loc[x, 0]), float(f.conf_int().loc[x, 1]), float(f.pvalues[x]), int(f.nobs)]


Dn = D[D.n_next >= 10]; Dr = D.dropna(subset=['rank0', 'rank1'])
zmap = {'bps': 'bps_z', 'bpc': 'bpc_z', 'tbw': 'tbw_z', 'dsw': 'dsw_z', 'upr': 'upr_off_z', 'uprp': 'uprp_z'}
for k, z in zmap.items():
    T1[k]['win_next'] = eff('conv_next', z, Dn)
    T1[k]['rank_next'] = eff('d_lrank', z, Dr)
    print(k, 'win/SD %.2f pp [%.2f, %.2f] p=%.3f' % tuple([100 * x for x in T1[k]['win_next'][:3]] + [T1[k]['win_next'][3]]),
          '| rank/SD %+.1f%% [%+.1f, %+.1f] p=%.3f' % tuple([100 * (np.exp(x) - 1) for x in T1[k]['rank_next'][:3]] + [T1[k]['rank_next'][3]]))
json.dump(T1, open(OUT + 'table1_v3.json', 'w'), indent=1, default=float)

# ---------------- Figure 1: adjusted quintile means of next-season wins above points expectation ----------------
fig = {}
for k in ['uprp', 'upr_off']:
    d = Dn.dropna(subset=['conv_next', k, 'tpw', 'rank0', 'age']).copy()
    d['q'] = d.groupby('t')[k].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    f = smf.ols('conv_next ~ C(q)' + ctl, data=d).fit(cov_type='cluster', cov_kwds={'groups': d.pid})
    # adjusted means: predicted at sample covariate values, varying the quintile only
    mm = []
    for q in range(5):
        dq = d.assign(q=q); pr = f.get_prediction(dq).summary_frame()
        mm.append(pr['mean'].mean())
    ci = []
    base = [c for c in f.params.index if c.startswith('C(q)')]
    cov = f.cov_params()
    for q in range(5):
        if q == 0:
            ci.append(np.nan)
        else:
            ci.append(1.96 * np.sqrt(cov.loc[f'C(q)[T.{q}]', f'C(q)[T.{q}]']))
    raw = d.groupby('q').conv_next.agg(['mean', 'sem', 'size'])
    fig[k] = dict(adj_mean=mm, ci_vs_q1=ci, raw_mean=raw['mean'].tolist(), raw_sem=raw['sem'].tolist(), n=raw['size'].tolist(),
                  q5_minus_q1=[float(f.params['C(q)[T.4]']), float(f.conf_int().loc['C(q)[T.4]', 0]), float(f.conf_int().loc['C(q)[T.4]', 1]),
                               float(f.pvalues['C(q)[T.4]'])])
    print(k, 'adjusted quintile means (pp):', np.round(100 * np.array(mm), 2), '| raw:', np.round(100 * raw['mean'].values, 2),
          '| Q5-Q1 %.2f pp [%.2f, %.2f] p=%.4f' % tuple([100 * x for x in fig[k]['q5_minus_q1'][:3]] + [fig[k]['q5_minus_q1'][3]]))
json.dump(fig, open(OUT + 'fig1_v3.json', 'w'), indent=1, default=float)
Dn.to_csv(OUT + 'panel_v3.csv', index=False)
