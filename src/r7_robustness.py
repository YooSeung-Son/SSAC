"""R7 robustness quoted in the abstract: UPR+ beyond past wins-above-expectation, tour-only outcomes,
players outside the top 20, ranks 51-150, and the season-scale conversion of the quintile gap."""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import OUT

D = pd.read_csv(OUT + 'panel_uprplus.csv')
d = D[D.n_next >= 10].dropna(subset=['conv_next', 'uprp_z', 'conv_past_z', 'tpw', 'rank0', 'age'])
ctl = ' + tpw + np.log(rank0) + age + np.log(n_ch) + C(t)'
res = {}


def fit(formula, data, x='uprp_z'):
    f = smf.ols(formula + ctl, data=data).fit(cov_type='cluster', cov_kwds={'groups': data.pid})
    return dict(b=float(f.params[x]), lo=float(f.conf_int().loc[x, 0]), hi=float(f.conf_int().loc[x, 1]), p=float(f.pvalues[x]), n=int(f.nobs))


res['with_past_conversion'] = fit('conv_next ~ uprp_z + conv_past_z', d)
res['past_conversion_alone'] = fit('conv_next ~ conv_past_z', d, 'conv_past_z')
res['corr_uprp_past_conversion'] = float(d[['uprp', 'conv_past']].corr().iloc[0, 1])
SM = pd.read_parquet(OUT + 'sack_markov.parquet'); SM['yr'] = SM.date.dt.year
tour = SM[SM.src == 'tour'].groupby(['pid', 'yr']).agg(cn=('conv', 'mean'), nn=('won', 'size')).reset_index()
d2 = d.merge(tour.assign(t=lambda x: x.yr - 1)[['pid', 't', 'cn', 'nn']], on=['pid', 't'], how='left')
d2 = d2[d2.nn >= 10]
res['tour_only_outcome'] = fit('cn ~ uprp_z', d2)
res['rank_above_20'] = fit('conv_next ~ uprp_z', d[d.rank0 > 20])
res['rank_51_150'] = fit('conv_next ~ uprp_z', d[d.rank0.between(51, 150)])
q = json.load(open(OUT + 'fig1_v3.json'))['uprp']['q5_minus_q1'][0]
res['median_next_season_matches'] = float(d.n_next.median())
res['q5_q1_wins_per_season'] = float(q * d.n_next.median())
for k, v in res.items():
    print(k, v if not isinstance(v, dict) else {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
json.dump(res, open(OUT + 'r7_robustness.json', 'w'), indent=1)
