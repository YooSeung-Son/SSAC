import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / 'out') + '/'                    # derived tables, JSON results
R = str(ROOT / 'data' / 'tennis_atp') + '/'      # Jeff Sackmann ATP files (matches, rankings, players)
D_MCP = str(ROOT / 'data' / 'mcp') + '/'         # Match Charting Project (men)
LATEX = str(ROOT / 'latex') + '/'
COMP = ['bps', 'bpc', 'tbw', 'dsw']


def load():
    A = pd.read_parquet(OUT + 'long_analysis.parquet')
    A['year'] = A.date.dt.year
    return A


def counts(A):
    """Per-player opportunity / success counts for the four ATP Under Pressure components."""
    A = A[~A.ret & ~A.wo]
    st = A[A.has_stats]
    c = pd.DataFrame({
        'bps_n': st.groupby('pid').s_bpFaced.sum(), 'bps_k': st.groupby('pid').s_bpSaved.sum(),
        'bpc_n': st.groupby('pid').o_bpFaced.sum(),
        'bpc_k': st.groupby('pid').apply(lambda d: (d.o_bpFaced - d.o_bpSaved).sum(), include_groups=False),
        'tbw_n': A.groupby('pid').tb_n.sum(), 'tbw_k': A.groupby('pid').tb_won.sum(),
        'dsw_n': A.groupby('pid').deciding.sum(), 'dsw_k': A[A.deciding].groupby('pid').won.sum(),
        'n_matches': A.groupby('pid').size(),
    }).fillna(0)
    return c


def beta_binom_fit(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    m = n > 0
    k, n = k[m], n[m]

    def nll(t):
        a, b = np.exp(t)
        return -(betaln(k + a, n - k + b) - betaln(a, b)).sum()
    p0 = k.sum() / n.sum()
    r = minimize(nll, np.log([p0 * 20, (1 - p0) * 20]), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8))
    return np.exp(r.x)


def upr(c):
    """Raw ATP-style UPR (sum of four percentages) and Beta-Binomial empirical-Bayes shrunk version."""
    c = c.copy()
    for k in COMP:
        c[k + '_raw'] = np.where(c[k + '_n'] > 0, c[k + '_k'] / c[k + '_n'].where(c[k + '_n'] > 0), np.nan)
        a, b = beta_binom_fit(c[k + '_k'], c[k + '_n'])
        c[k + '_eb'] = (c[k + '_k'] + a) / (c[k + '_n'] + a + b)
        c[k + '_prior'] = a / (a + b); c[k + '_m'] = a + b
    c['upr_raw'] = 100 * c[[k + '_raw' for k in COMP]].sum(1, min_count=4)
    c['upr_eb'] = 100 * c[[k + '_eb' for k in COMP]].sum(1)
    return c


def rankings():
    rk = pd.concat([pd.read_csv(R + f) for f in ['atp_rankings_10s.csv', 'atp_rankings_20s.csv', 'atp_rankings_current.csv']])
    rk['date'] = pd.to_datetime(rk.ranking_date.astype(str), format='%Y%m%d')
    return rk.drop_duplicates(['date', 'player'])


def rank_at(rk, when):
    """Rank list from the last ranking date on or before `when`."""
    d = rk.date[rk.date <= pd.Timestamp(when)].max()
    return rk[rk.date == d].set_index('player')['rank']
