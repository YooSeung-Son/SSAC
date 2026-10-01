"""Build player-match long table for the Under Pressure study.

Elo (overall + surface) and rolling serve/return strength are computed sequentially
over every ATP tour, qualifying and Challenger match from 2000 on; the analysis sample is
tour-level main-draw singles 2019-2025 (team events, Olympics, Next Gen Finals excluded).
"""
import re, numpy as np, pandas as pd

from common import R
from common import OUT
Y0, Y1 = 2000, 2026

ROUND_ORD = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'R128': 10, 'R64': 11, 'R32': 12, 'R16': 13,
             'QF': 14, 'SF': 15, 'BR': 16, 'F': 17, 'RR': 12, 'ER': 9}

frames = []
for y in range(Y0, Y1 + 1):
    t = pd.read_csv(f'{R}atp_matches_{y}.csv', low_memory=False); t['src'] = 'tour'
    c = pd.read_csv(f'{R}atp_matches_qual_chall_{y}.csv', low_memory=False); c['src'] = 'qc'
    frames += [t, c]
m = pd.concat(frames, ignore_index=True)
m = m[m.score.notna()].copy()
m['date'] = pd.to_datetime(m.tourney_date.astype(int).astype(str), format='%Y%m%d')
m['rord'] = m['round'].map(ROUND_ORD).fillna(12)
m = m.sort_values(['date', 'tourney_id', 'rord', 'match_num'], kind='mergesort').reset_index(drop=True)
m['wo'] = m.score.str.contains('W/O|Walkover|unfinished|Def', case=False, regex=True)
m['ret'] = m.score.str.contains('RET|DEF', regex=True)

team = m.tourney_name.str.contains('Davis Cup|ATP Cup|Atp Cup|United Cup|Laver Cup|Olympics|Next ?Gen', case=False, regex=True)
m['analysis'] = ((m.src == 'tour') & m.tourney_level.isin(['G', 'M', 'A', 'F']) & ~team
                 & ~m['round'].str.startswith('Q') & m.date.dt.year.between(2019, 2025) & ~m.wo)
m['team'] = team

# ---------------- Elo (538-style K) ----------------
def kf(n):
    return 250.0 / (n + 5) ** 0.4

elo, elo_s, nm, nm_s = {}, {}, {}, {}
E = np.zeros((len(m), 6))  # w_elo, l_elo, w_selo, l_selo, w_n, l_n
EP = np.zeros((len(m), 2))  # post-match overall Elo of winner, loser
for i, (w, l, s, wo, tm) in enumerate(zip(m.winner_id.values, m.loser_id.values, m.surface.fillna('Hard').values,
                                         m.wo.values, team.values)):
    s = 'Hard' if s == 'Carpet' else s
    ew, el = elo.get(w, 1500.), elo.get(l, 1500.)
    sw, sl = elo_s.get((w, s), 1500.), elo_s.get((l, s), 1500.)
    E[i] = ew, el, sw, sl, nm.get(w, 0), nm.get(l, 0)
    if wo:
        EP[i] = ew, el
        continue
    pw = 1 / (1 + 10 ** ((el - ew) / 400))
    ps = 1 / (1 + 10 ** ((sl - sw) / 400))
    elo[w] = ew + kf(nm.get(w, 0)) * (1 - pw); elo[l] = el - kf(nm.get(l, 0)) * (1 - pw)
    elo_s[(w, s)] = sw + kf(nm_s.get((w, s), 0)) * (1 - ps); elo_s[(l, s)] = sl - kf(nm_s.get((l, s), 0)) * (1 - ps)
    nm[w] = nm.get(w, 0) + 1; nm[l] = nm.get(l, 0) + 1
    EP[i] = elo[w], elo[l]
    nm_s[(w, s)] = nm_s.get((w, s), 0) + 1; nm_s[(l, s)] = nm_s.get((l, s), 0) + 1
m[['w_elo', 'l_elo', 'w_selo', 'l_selo', 'w_nprev', 'l_nprev']] = E
m[['w_elo_post', 'l_elo_post']] = EP

# ---------------- score parsing ----------------
SET = re.compile(r'(\d+)-(\d+)(\((\d+)(?:-\d+)?\))?')

def parse(score):
    """Return (#sets, #TB won by winner, #TB won by loser)."""
    sets = SET.findall(score.split('RET')[0])
    tw = tl = 0
    for a, b, paren, _ in sets:
        a, b = int(a), int(b)
        if paren or {a, b} == {7, 6}:
            if a > b: tw += 1
            else: tl += 1
    return len(sets), tw, tl

P = m.score.map(parse)
m['nsets'] = [p[0] for p in P]; m['tb_w'] = [p[1] for p in P]; m['tb_l'] = [p[2] for p in P]
m['deciding'] = (~m.ret) & (~m.wo) & (m.nsets == m.best_of)

# ---------------- long table ----------------
cols_s = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
base = ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'date', 'round', 'best_of', 'src',
        'analysis', 'ret', 'wo', 'team', 'nsets', 'deciding', 'score', 'match_num']
m['mid'] = np.arange(len(m))
rows = []
for side, opp, won in [('w', 'l', 1), ('l', 'w', 0)]:
    d = m[base + ['mid']].copy()
    d['pid'] = m[f'{"winner" if side == "w" else "loser"}_id']; d['oid'] = m[f'{"winner" if opp == "w" else "loser"}_id']
    d['pname'] = m[f'{"winner" if side == "w" else "loser"}_name']
    d['won'] = won
    d['rank'] = m[f'{"winner" if side == "w" else "loser"}_rank']; d['orank'] = m[f'{"winner" if opp == "w" else "loser"}_rank']
    d['age'] = m[f'{"winner" if side == "w" else "loser"}_age']
    d['elo'] = m[f'{side}_elo']; d['oelo'] = m[f'{opp}_elo']; d['selo'] = m[f'{side}_selo']; d['oselo'] = m[f'{opp}_selo']
    d['nprev'] = m[f'{side}_nprev']; d['elo_post'] = m[f'{side}_elo_post']
    for c in cols_s:
        d['s_' + c] = m[f'{side}_{c}']; d['o_' + c] = m[f'{opp}_{c}']
    d['tb_won'] = m[f'tb_{side}']; d['tb_n'] = m.tb_w + m.tb_l
    rows.append(d)
L = pd.concat(rows, ignore_index=True).sort_values(['mid', 'won'], ascending=[True, False]).reset_index(drop=True)

# serve / return points won in the match
L['spw_n'] = L.s_svpt; L['spw_k'] = L.s_1stWon + L.s_2ndWon
L['rpw_n'] = L.o_svpt; L['rpw_k'] = L.o_svpt - (L.o_1stWon + L.o_2ndWon)
ok = (L.s_svpt > 0) & (L.o_svpt > 0) & L.spw_k.notna() & L.rpw_k.notna() & ~L.wo
L['has_stats'] = ok

# ---------------- rolling 52-week serve / return strength (pre-match, shrunk) ----------------
# prior: 60 % serve points won, 40 % return points won, 600 pseudo-points
PRI_N = 600.
L = L.sort_values(['pid', 'date', 'mid']).reset_index(drop=True)
out_s = np.full(len(L), np.nan); out_r = np.full(len(L), np.nan)
for pid, g in L.groupby('pid', sort=False):
    idx = g.index.values; dts = g.date.values.astype('datetime64[D]').astype(np.int64)
    sk = np.where(g.has_stats, g.spw_k, 0.).astype(float); sn = np.where(g.has_stats, g.spw_n, 0.).astype(float)
    rk = np.where(g.has_stats, g.rpw_k, 0.).astype(float); rn = np.where(g.has_stats, g.rpw_n, 0.).astype(float)
    csk, csn, crk, crn = [np.concatenate([[0.], np.cumsum(a)]) for a in (sk, sn, rk, rn)]
    lo = np.searchsorted(dts, dts - 364, side='left')   # window start (inclusive)
    hi = np.searchsorted(dts, dts, side='left')          # strictly before this tournament
    s_k = csk[hi] - csk[lo]; s_n = csn[hi] - csn[lo]; r_k = crk[hi] - crk[lo]; r_n = crn[hi] - crn[lo]
    out_s[idx] = (s_k + 0.60 * PRI_N) / (s_n + PRI_N)
    out_r[idx] = (r_k + 0.40 * PRI_N) / (r_n + PRI_N)
L['roll_spw'] = out_s; L['roll_rpw'] = out_r
opp = L[['mid', 'pid', 'roll_spw', 'roll_rpw']].rename(columns={'pid': 'oid', 'roll_spw': 'oroll_spw', 'roll_rpw': 'oroll_rpw'})
L = L.merge(opp, on=['mid', 'oid'], how='left')
L = L.sort_values(['mid', 'won'], ascending=[True, False]).reset_index(drop=True)

L.to_parquet(OUT + 'long_all.parquet')
A = L[L.analysis].copy()
A.to_parquet(OUT + 'long_analysis.parquet')
print('all player-match rows', len(L), 'analysis rows', len(A), 'matches', A.mid.nunique(),
      'players', A.pid.nunique())
print(A.groupby(A.date.dt.year).mid.nunique())
print('analysis: ret', A.ret.mean(), 'stats', A.has_stats.mean(), 'rank missing', A['rank'].isna().mean())
