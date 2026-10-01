"""Markov points-based win expectation for every Sackmann match with serve stats (2012-2026)."""
import sys, numpy as np, pandas as pd
sys.setrecursionlimit(20000)
from markov import matchw

from common import OUT
L = pd.read_parquet(OUT + 'long_all.parquet')
L = L[(L.date.dt.year >= 2012) & L.has_stats & ~L.wo & ~L.ret & ~L.team].copy()


def final_fmt(name, level, yr, bo):
    if level != 'G' or bo != 5:
        return 'tb7'
    n = str(name).lower()
    if 'australian' in n:
        return 'tb10' if yr >= 2019 else 'adv'
    if 'roland' in n or 'french' in n:
        return 'tb10' if yr >= 2022 else 'adv'
    if 'wimbledon' in n:
        return 'tb10' if yr >= 2022 else ('tb12' if yr >= 2019 else 'adv')
    if 'us open' in n:
        return 'tb10' if yr >= 2022 else 'tb7'
    return 'tb7'


L['fmt'] = [final_fmt(a, b, c, d) for a, b, c, d in zip(L.tourney_name, L.tourney_level, L.date.dt.year, L.best_of)]
L['ps'] = (L.spw_k / L.spw_n).clip(0.2, 0.97)                 # own serve points won
L['po'] = (1 - L.rpw_k / L.rpw_n).clip(0.2, 0.97)             # opponent's serve points won
rd = lambda x: round(round(x / 0.004) * 0.004, 3)
L['p_mk'] = [matchw(rd(a), rd(b), 0, 0, int(bo), f) for a, b, bo, f in zip(L.ps, L.po, L.best_of, L.fmt)]
L['tpw'] = (L.spw_k + L.rpw_k) / (L.spw_n + L.rpw_n)
L['conv'] = L.won - L.p_mk
L[['mid', 'pid', 'oid', 'date', 'won', 'p_mk', 'tpw', 'conv', 'src', 'tourney_level', 'best_of', 'elo', 'oelo', 'selo', 'oselo', 'rank',
   'orank', 'age', 'analysis']].to_parquet(OUT + 'sack_markov.parquet')
print(len(L), 'rows; mean conv %.4f, sd %.3f; corr(won, p_mk) %.3f' % (L.conv.mean(), L.conv.std(), np.corrcoef(L.won, L.p_mk)[0, 1]))
