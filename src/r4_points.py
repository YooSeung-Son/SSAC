"""R4: does the official ratio predict break-point / tiebreak-point outcomes beyond ordinary skill? UPR+?
Cross-fit: each player's charted matches split into odd/even halves; features from one half,
point outcomes (break points, tiebreak points) from the other; 5-fold CV logistic by match."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss

from common import OUT
P = pd.read_parquet(OUT + 'mcp_points.parquet')
M = pd.read_parquet(OUT + 'mcp_matches.parquet').set_index('match_id')
B = pd.read_parquet(OUT + 'mcp_points_badj.parquet')
P = P.merge(B, on=['match_id', 'Pt'])
P['spid'] = np.where(P.Svr == 1, M.pid1.reindex(P.match_id).values, M.pid2.reindex(P.match_id).values)
P['rpid'] = np.where(P.Svr == 1, M.pid2.reindex(P.match_id).values, M.pid1.reindex(P.match_id).values)
P['date'] = M.date.reindex(P.match_id).values
P = P.dropna(subset=['spid', 'rpid'])
# clutch residual relative to own level (as in UPR+)
P['r'] = P.srv_won - P.b_adj
P['r'] = P.r - P.groupby(['match_id', 'Svr']).r.transform('mean')
P['e'] = P.imp * P.r
# half assignment per player-match (by date order of each player's matches)
pm = pd.concat([pd.DataFrame({'match_id': M.index, 'pid': M.pid1, 'date': M.date}),
                pd.DataFrame({'match_id': M.index, 'pid': M.pid2, 'date': M.date})]).dropna().sort_values(['pid', 'date'])
pm['half'] = pm.groupby('pid').cumcount() % 2
half = pm.set_index(['match_id', 'pid']).half
P['h_s'] = half.reindex(list(zip(P.match_id, P.spid))).values
P['h_r'] = half.reindex(list(zip(P.match_id, P.rpid))).values
TAU = 1.84e-5


def feats(Q):
    """per-player features from a set of points (Q holds the 'feature' half for both roles)."""
    sv = Q.groupby('spid'); rv = Q.groupby('rpid')
    f = pd.DataFrame({'spw_k': sv.srv_won.sum(), 'spw_n': sv.size()}).join(
        pd.DataFrame({'rpw_k': (1 - Q.srv_won).groupby(Q.rpid).sum(), 'rpw_n': rv.size()}), how='outer').fillna(0)
    bp_s = Q[Q.is_bp].groupby('spid').srv_won.agg(['sum', 'size']); bp_r = (1 - Q[Q.is_bp].srv_won).groupby(Q[Q.is_bp].rpid).agg(['sum', 'size'])
    f['bps'] = ((bp_s['sum'] + 0.62 * 30) / (bp_s['size'] + 30)).reindex(f.index).fillna(0.62)
    f['bpc'] = ((bp_r['sum'] + 0.38 * 30) / (bp_r['size'] + 30)).reindex(f.index).fillna(0.38)
    tb = Q[Q.in_tb]
    tw = pd.concat([pd.DataFrame({'pid': tb.spid, 'w': tb.srv_won}), pd.DataFrame({'pid': tb.rpid, 'w': 1 - tb.srv_won})]).groupby('pid').w.agg(['sum', 'size'])
    f['tbw'] = ((tw['sum'] + 0.5 * 40) / (tw['size'] + 40)).reindex(f.index).fillna(0.5)
    f['spw'] = (f.spw_k + 0.63 * 200) / (f.spw_n + 200); f['rpw'] = (f.rpw_k + 0.37 * 200) / (f.rpw_n + 200)
    es = Q.groupby('spid').agg(e=('e', 'sum'), w=('imp', 'sum')); er = Q.groupby('rpid').agg(e=('e', 'sum'), w=('imp', 'sum'))
    vs = (Q.imp ** 2 * Q.b_adj * (1 - Q.b_adj)).groupby(Q.spid).sum(); vr = (Q.imp ** 2 * Q.b_adj * (1 - Q.b_adj)).groupby(Q.rpid).sum()
    cs = es.e / es.w; cr = -er.e / er.w
    f['up_s'] = (cs * TAU / (TAU + vs / es.w ** 2)).reindex(f.index).fillna(0)
    f['up_r'] = (cr * TAU / (TAU + vr / er.w ** 2)).reindex(f.index).fillna(0)
    return f


lg = lambda p: np.log(p / (1 - p))
out = {}
for kind, sel in [('break_points', P.is_bp), ('tiebreak_points', P.in_tb)]:
    preds = {k: [] for k in ['skill', 'skill+official', 'skill+UPR+']}; ys = []
    for fh in (0, 1):
        F = feats(P[(P.h_s == fh) & (P.h_r == fh)])
        T = P[sel & (P.h_s == 1 - fh) & (P.h_r == 1 - fh)].copy()
        T = T[T.spid.isin(F.index) & T.rpid.isin(F.index)]
        X = pd.DataFrame({'sk': lg(F.spw.reindex(T.spid).values) - lg(1 - F.rpw.reindex(T.rpid).values)})
        if kind == 'break_points':
            X['off1'] = lg(F.bps.reindex(T.spid).values); X['off2'] = lg(F.bpc.reindex(T.rpid).values)
        else:
            X['off1'] = lg(F.tbw.reindex(T.spid).values); X['off2'] = lg(F.tbw.reindex(T.rpid).values)
        X['up1'] = F.up_s.reindex(T.spid).values * 100; X['up2'] = F.up_r.reindex(T.rpid).values * 100
        y = T.srv_won.values; g = T.match_id.values
        sets = {'skill': ['sk'], 'skill+official': ['sk', 'off1', 'off2'], 'skill+UPR+': ['sk', 'up1', 'up2']}
        for k, cols in sets.items():
            pr = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, g):
                lr = LogisticRegression(C=1e4, max_iter=1000).fit(X.iloc[tr][cols], y[tr]); pr[te] = lr.predict_proba(X.iloc[te][cols])[:, 1]
            preds[k].append(pr)
        ys.append(y)
    y = np.concatenate(ys)
    rng = np.random.default_rng(3); bidx = rng.integers(0, len(y), (1000, len(y)))
    base = np.concatenate(preds['skill'])
    o = {}
    for k, v in preds.items():
        p = np.concatenate(v)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)); ll0 = -(y * np.log(base) + (1 - y) * np.log(1 - base))
        d = (ll - ll0)[bidx].mean(1)
        o[k] = dict(n=len(y), logloss=float(ll.mean()), brier=brier_score_loss(y, p), d_logloss_vs_skill=float((ll - ll0).mean()),
                    ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    out[kind] = o
    print(kind, json.dumps({k: {a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items()} for k, v in o.items()}))
json.dump(out, open(OUT + 'r4_points.json', 'w'), indent=1)
