"""Match Charting Project (men) -> point table with leverage, match table with Markov expectation."""
import sys, re, unicodedata, numpy as np, pandas as pd
sys.setrecursionlimit(20000)
from markov import FMT, point_importance, matchw

from common import D_MCP as D
from common import OUT
from common import R

m = pd.read_csv(D + 'charting-m-matches.csv', encoding='latin-1', dtype=str)
m = m[m['Best of'].isin(['3', '5'])].copy()
m['bo'] = m['Best of'].astype(int)
m['fmt'] = m['Final TB?'].str.strip().map(FMT).fillna('tb7')
m['date'] = pd.to_datetime(m.Date, format='%Y%m%d', errors='coerce')
m = m[m.date.notna()]
P = pd.concat([pd.read_csv(D + f, encoding='latin-1', low_memory=False,
                           usecols=['match_id', 'Pt', 'Set1', 'Set2', 'Gm1', 'Gm2', 'Pts', 'TbSet', 'Svr', 'PtWinner'])
               for f in ['charting-m-points-to-2009.csv', 'charting-m-points-2010s.csv', 'charting-m-points-2020s.csv']])
P = P[P.match_id.isin(m.match_id)].dropna(subset=['Set1', 'Set2', 'Gm1', 'Gm2', 'Pts', 'Svr', 'PtWinner'])
for c in ['Pt', 'Set1', 'Set2', 'Gm1', 'Gm2', 'Svr', 'PtWinner']:
    P[c] = pd.to_numeric(P[c], errors='coerce')
P = P.dropna(subset=['Set1', 'Set2', 'Gm1', 'Gm2', 'Svr', 'PtWinner'])
P = P[P.Svr.isin([1, 2]) & P.PtWinner.isin([1, 2])]
P = P.sort_values(['match_id', 'Pt']).reset_index(drop=True)
P = P.merge(m[['match_id', 'bo', 'fmt']], on='match_id')

GAME = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}


def parse_pts(s):
    """server-first score -> (server pts, returner pts, is_tiebreak_score)."""
    s = str(s).strip()
    if '-' not in s:
        return None
    x, y = s.split('-', 1)
    if x in GAME and y in GAME and not (x == '0' and y == '0' and False):
        sx, ry = GAME[x], GAME[y]
        if x == 'AD': sx, ry = 4, 3
        if y == 'AD': sx, ry = 3, 4
        return sx, ry, False
    if x.isdigit() and y.isdigit():
        return int(x), int(y), True
    return None


rows = []
cache = {}
for t in P.itertuples(index=False):
    r = parse_pts(t.Pts)
    if r is None:
        rows.append((np.nan, False, np.nan, np.nan)); continue
    sx, rx, numeric = r
    s1, s2, g1, g2 = int(t.Set1), int(t.Set2), int(t.Gm1), int(t.Gm2)
    need = (t.bo + 1) // 2
    final = (s1 == need - 1 and s2 == need - 1)
    sfmt = t.fmt if final else 'tb7'
    at = 12 if sfmt == 'tb12' else 6
    if sfmt == 'mtb' and final:
        in_tb = True
    elif sfmt == 'adv':
        in_tb = False
    else:
        in_tb = (g1 == at and g2 == at) and (numeric or str(t.Pts).strip() == '0-0')
    if in_tb and not numeric and str(t.Pts).strip() != '0-0':
        rows.append((np.nan, False, np.nan, np.nan)); continue
    srv = int(t.Svr)
    a, b = (sx, rx) if srv == 1 else (rx, sx)
    if not in_tb and (a > 4 or b > 4):
        rows.append((np.nan, False, np.nan, np.nan)); continue
    key = (s1, s2, g1, g2, srv, a, b, in_tb, t.bo, t.fmt)
    if key not in cache:
        try:
            cache[key] = abs(point_importance(0.64, 0.64, s1, s2, g1, g2, srv, a, b, in_tb, t.bo, t.fmt))
        except RecursionError:
            cache[key] = np.nan
    # break point: returner one point from winning the game (not in tiebreak)
    bp = (not in_tb) and ((rx >= 3 and rx > sx) or (rx == 3 and sx < 3))
    rows.append((cache[key], in_tb, bp, sx * 10 + rx))
P['imp'] = [x[0] for x in rows]; P['in_tb'] = [x[1] for x in rows]; P['is_bp'] = [x[2] for x in rows]
P = P[P.imp.notna()].copy()
P['is_bp'] = P.is_bp.astype(bool)
P['srv_won'] = (P.Svr == P.PtWinner).astype(int)
print('points', len(P), 'matches', P.match_id.nunique(), 'distinct states', len(cache))

# ---------------- match table ----------------
g = P.groupby(['match_id', 'Svr']).srv_won.agg(['sum', 'size']).unstack('Svr')
M = m.set_index('match_id').loc[g.index].copy()
M['sv1_k'], M['sv1_n'] = g[('sum', 1)], g[('size', 1)]
M['sv2_k'], M['sv2_n'] = g[('sum', 2)], g[('size', 2)]
M = M[(M.sv1_n >= 20) & (M.sv2_n >= 20)]
last = P.groupby('match_id').tail(1).set_index('match_id')
M['winner'] = last.PtWinner.reindex(M.index)
# completeness: the last point's winner must have been one set away and the point must end that set
lw = last.reindex(M.index)
sets_w = np.where(lw.PtWinner == 1, lw.Set1, lw.Set2)
M['complete'] = sets_w == (M.bo + 1) // 2 - 1
pts1 = M.sv1_k + (M.sv2_n - M.sv2_k); pts2 = M.sv2_k + (M.sv1_n - M.sv1_k)
M['tpw1'] = pts1 / (pts1 + pts2)
M['p1'] = ((M.sv1_k + 0.5) / (M.sv1_n + 1)).clip(0.3, 0.95)
M['p2'] = ((M.sv2_k + 0.5) / (M.sv2_n + 1)).clip(0.3, 0.95)
M['exp1'] = [matchw(round(a, 3), round(b, 3), 0, 0, bo, f) for a, b, bo, f in zip(M.p1, M.p2, M.bo, M.fmt)]
print('matches with outcome', int(M.complete.sum()), 'of', len(M))

# ---------------- player ids (Sackmann) ----------------
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z]', '', s.lower())

pl = pd.read_csv(R + 'atp_players.csv', dtype={'dob': str})
pl['key'] = (pl.name_first.fillna('') + pl.name_last.fillna('')).map(norm)
L = pd.read_parquet(OUT + 'long_all.parquet', columns=['pid', 'date'])
act = L.assign(yr=L.date.dt.year).groupby(['pid', 'yr']).size()
cand = pl.groupby('key').player_id.apply(list).to_dict()


def pid_for(name, yr):
    c = cand.get(norm(name), [])
    if len(c) <= 1:
        return c[0] if c else np.nan
    return max(c, key=lambda i: act.get((i, yr), 0))

M['yr'] = M.date.dt.year
M['pid1'] = [pid_for(n, y) for n, y in zip(M['Player 1'], M.yr)]
M['pid2'] = [pid_for(n, y) for n, y in zip(M['Player 2'], M.yr)]
print('id match rate', M.pid1.notna().mean().round(3), M.pid2.notna().mean().round(3))
print('unmatched examples', pd.concat([M.loc[M.pid1.isna(), 'Player 1'], M.loc[M.pid2.isna(), 'Player 2']]).value_counts().head(10).to_dict())
M = M.reset_index()
P = P[P.match_id.isin(M.match_id)]
P.to_parquet(OUT + 'mcp_points.parquet'); M.to_parquet(OUT + 'mcp_matches.parquet')
print(M.groupby('yr').size().to_dict())
