"""Markov model of a tennis match (iid points, serve-dependent).

p1 / p2: probability that player 1 / player 2 wins a point on his own serve.
fmt (deciding-set format): 'tb7' TB to 7 at 6-6, 'tb10' TB to 10 at 6-6, 'adv' advantage set,
'tb12' TB to 7 at 12-12, 'mtb' match tiebreak to 10 instead of a deciding set.
All probabilities are P(player 1 wins ...).
"""
from functools import lru_cache

FMT = {'1': 'tb7', '0': 'tb7', 'N': 'tb7', 'V': 'tb7', 'A': 'adv', 'T': 'tb10', 'W': 'tb12', 'S': 'mtb'}


@lru_cache(maxsize=None)
def hold(p, s, r):
    """P(server wins game) from server points s, returner points r (0,1,2,3=40,...)."""
    if s >= 4 and s - r >= 2:
        return 1.0
    if r >= 4 and r - s >= 2:
        return 0.0
    if s >= 3 and r >= 3:
        d = p * p / (p * p + (1 - p) * (1 - p))
        if s == r:
            return d
        return p + (1 - p) * d if s > r else p * d
    return p * hold(p, s + 1, r) + (1 - p) * hold(p, s, r + 1)


def _tb_server(k, first):
    return first if k % 4 in (0, 3) else 3 - first


@lru_cache(maxsize=None)
def tbw(p1, p2, a, b, first, target):
    """P(player 1 wins tiebreak) at points a (pl.1) - b (pl.2); `first` served point 0."""
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    if min(a, b) >= target - 1:
        # past target-1 all: each pair of points has one serve by each player -> closed form
        qa, qb = p1, 1 - p2
        d = qa * qb / (qa * qb + (1 - qa) * (1 - qb))
        if a == b:
            return d
        q = p1 if _tb_server(a + b, first) == 1 else 1 - p2
        return q + (1 - q) * d if a > b else q * d
    srv = _tb_server(a + b, first)
    q = p1 if srv == 1 else 1 - p2       # P(player 1 wins this point)
    return q * tbw(p1, p2, a + 1, b, first, target) + (1 - q) * tbw(p1, p2, a, b + 1, first, target)


def _set_over(g1, g2):
    return (g1 >= 6 and g1 - g2 >= 2) or (g2 >= 6 and g2 - g1 >= 2) or (g1 == 7 and g2 == 6) or (g2 == 7 and g1 == 6)


@lru_cache(maxsize=None)
def setw(p1, p2, g1, g2, server, fmt):
    """P(player 1 wins set) at games g1-g2, `server` to serve the next game."""
    if fmt == 'mtb':
        return tbw(p1, p2, 0, 0, server, 10)
    if fmt == 'adv':
        if (g1 >= 6 and g1 - g2 >= 2):
            return 1.0
        if (g2 >= 6 and g2 - g1 >= 2):
            return 0.0
        if min(g1, g2) >= 5:
            h1, h2 = hold(p1, 0, 0), 1 - hold(p2, 0, 0)     # pl.1 wins a game on own / opponent serve
            d = h1 * h2 / (h1 * h2 + (1 - h1) * (1 - h2))
            if g1 == g2:
                return d
            h = h1 if server == 1 else h2
            return h + (1 - h) * d if g1 > g2 else h * d
    else:
        at = 12 if fmt == 'tb12' else 6
        if g1 == at and g2 == at:
            return tbw(p1, p2, 0, 0, server, 10 if fmt == 'tb10' else 7)
        if fmt == 'tb12':
            if (g1 >= 6 and g1 - g2 >= 2) or g1 == 13:
                return 1.0
            if (g2 >= 6 and g2 - g1 >= 2) or g2 == 13:
                return 0.0
        else:
            if _set_over(g1, g2):
                return 1.0 if g1 > g2 else 0.0
    h = hold(p1, 0, 0) if server == 1 else 1 - hold(p2, 0, 0)   # P(pl.1 wins this game)
    return h * setw(p1, p2, g1 + 1, g2, 3 - server, fmt) + (1 - h) * setw(p1, p2, g1, g2 + 1, 3 - server, fmt)


@lru_cache(maxsize=None)
def matchw(p1, p2, s1, s2, bo, fmt):
    """P(player 1 wins match) at sets s1-s2 with a new set about to start (first server averaged)."""
    need = (bo + 1) // 2
    if s1 >= need:
        return 1.0
    if s2 >= need:
        return 0.0
    f = fmt if (s1 == need - 1 and s2 == need - 1) else 'tb7'
    ps = 0.5 * (setw(p1, p2, 0, 0, 1, f) + setw(p1, p2, 0, 0, 2, f))
    return ps * matchw(p1, p2, s1 + 1, s2, bo, fmt) + (1 - ps) * matchw(p1, p2, s1, s2 + 1, bo, fmt)


def _after_game(p1, p2, s1, s2, g1, g2, next_server, bo, fmt, set_fmt):
    """P(pl.1 wins match) right after a game ended with games g1-g2."""
    if set_fmt == 'adv':
        over = (g1 >= 6 and g1 - g2 >= 2) or (g2 >= 6 and g2 - g1 >= 2)
    elif set_fmt == 'tb12':
        over = (g1 >= 6 and g1 - g2 >= 2) or (g2 >= 6 and g2 - g1 >= 2) or g1 == 13 or g2 == 13
    else:
        over = _set_over(g1, g2)
    if over:
        return matchw(p1, p2, s1 + (g1 > g2), s2 + (g2 > g1), bo, fmt)
    ps = setw(p1, p2, g1, g2, next_server, set_fmt)
    return ps * matchw(p1, p2, s1 + 1, s2, bo, fmt) + (1 - ps) * matchw(p1, p2, s1, s2 + 1, bo, fmt)


def point_importance(p1, p2, s1, s2, g1, g2, server, a, b, in_tb, bo, fmt):
    """Importance of the next point = P(pl.1 wins match | pl.1 wins point) - P(... | pl.1 loses point).
    a, b: points of player 1 and player 2 in the current game or tiebreak (game points as 0,1,2,3,4...)."""
    need = (bo + 1) // 2
    final = (s1 == need - 1 and s2 == need - 1)
    set_fmt = fmt if final else 'tb7'
    if set_fmt == 'mtb' and final:
        in_tb = True
    if in_tb:
        target = 10 if (final and set_fmt in ('tb10', 'mtb')) else 7
        k = a + b
        first = server if k % 4 in (0, 3) else 3 - server
        w = tbw(p1, p2, a + 1, b, first, target); l = tbw(p1, p2, a, b + 1, first, target)
        if set_fmt == 'mtb' and final:
            return w - l
        return (w - l) * (matchw(p1, p2, s1 + 1, s2, bo, fmt) - matchw(p1, p2, s1, s2 + 1, bo, fmt))
    ps = p1 if server == 1 else p2
    sx, rx = (a, b) if server == 1 else (b, a)
    dh = hold(ps, sx + 1, rx) - hold(ps, sx, rx + 1)            # server's game win prob swing
    nxt = 3 - server
    win_srv = _after_game(p1, p2, s1, s2, g1 + (server == 1), g2 + (server == 2), nxt, bo, fmt, set_fmt)
    lose_srv = _after_game(p1, p2, s1, s2, g1 + (server == 2), g2 + (server == 1), nxt, bo, fmt, set_fmt)
    imp_srv = dh * (win_srv - lose_srv)                          # in pl.1 win-prob units, server-won minus server-lost
    return imp_srv if server == 1 else -imp_srv
