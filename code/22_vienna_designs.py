"""Vienna (Table IV): (1) leave-one-day-out with 4G features rebuilt per fold so that no training target uses the
held-out day's LTE; (2) day-level clusters (G = 5) for all intervals, plus segment-level for comparison; one-sided bounds;
(3) own-unit share and nearest-training distance per design.
usage: python 22_vienna_designs.py <vienna_targets.csv.gz> <phone_data_lte.parquet> <results dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import t as tdist
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
D = pd.read_csv(sys.argv[1], parse_dates=['t15']); OUT = Path(sys.argv[3])
y = D.target_p90.to_numpy(); ly = np.log1p(y); W = np.sqrt(D.n.to_numpy()); SEG = D.sess.to_numpy(); DAY = D.camp.astype(str).to_numpy()
XY = D[['x_m', 'y_m']].to_numpy(); XYc = XY - XY.mean(0); D['xr'] = XYc[:, 0]; D['yr'] = XYc[:, 1]; COORDS = ['xr', 'yr']
TIMES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
FEAT = [c for c in D.columns if (c[:2] in ('r1', 'r2', 'r5')) or c == 'nearest_4g_m']; TW = [c for c in FEAT if 'sameh3' in c]; NT = [c for c in FEAT if 'sameh3' not in c]
LEAN = [f'r250_{k}_median' for k in ('rsrp', 'rsrq', 'sinr', 'dl')] + ['nearest_4g_m']
LAT0 = 48.19; KX = 111320 * np.cos(np.radians(LAT0)); KY = 110540
l = pd.read_parquet(sys.argv[2]); l = l[(l.operator == 'A') & l.latitude.notna() & l.longitude.notna()].copy()
l['camp'] = (l.time - pd.Timedelta(hours=12)).dt.date.astype(str); l['x'] = l.longitude.astype(float) * KX; l['y'] = l.latitude.astype(float) * KY
l['hour3'] = (l.time.dt.hour // 3).astype(int); l = l.sort_values('time'); l['lsess'] = (l.time.diff().dt.total_seconds() > 600).cumsum()
for c in ['dl_throughput_mbps', 'rsrp_dbm', 'rsrq_db', 'sinr_db', 'rssi_dbm']: l[c] = l[c].astype(float)
M = {'dl': 'dl_throughput_mbps', 'rsrp': 'rsrp_dbm', 'rsrq': 'rsrq_db', 'sinr': 'sinr_db', 'rssi': 'rssi_dbm'}
def build(Tc, L):
    """same feature definitions as v_prep.py, for targets Tc from LTE pool L"""
    L = L.reset_index(drop=True); tree = cKDTree(L[['x', 'y']].to_numpy()); XYt = Tc[['x_m', 'y_m']].to_numpy(); dn, _ = tree.query(XYt, k=1)
    rows = [{'nearest_4g_m': float(d)} for d in dn]
    for r in (100, 250, 500):
        nb = tree.query_ball_point(XYt, r)
        for f, idx, h3 in zip(rows, nb, Tc.hour3):
            q = L.iloc[idx]; p = f'r{r}_'; f[p + 'obs_count'] = len(q); f[p + 'trace_count'] = q.lsess.nunique() if len(q) else 0
            for k, c in M.items():
                v = q[c].dropna(); f[p + k + '_median'] = v.median() if len(v) else np.nan; f[p + k + '_p90'] = v.quantile(.9) if len(v) else np.nan
            for k in ('dl', 'rsrp'):
                v = q[M[k]].dropna(); f[p + k + '_p10'] = v.quantile(.1) if len(v) else np.nan
            v = q.dl_throughput_mbps.dropna(); f[p + 'zero_dl_share'] = float((v < .1).mean()) if len(v) else np.nan
            v = q.rsrp_dbm.dropna(); f[p + 'weak_rsrp_share'] = float((v < -110).mean()) if len(v) else np.nan
            w = q[q.hour3 == h3]; p2 = p + 'sameh3_'; f[p2 + 'obs_count'] = len(w); f[p2 + 'trace_count'] = w.lsess.nunique() if len(w) else 0
            for k in ('dl', 'rsrp', 'rsrq', 'sinr'):
                v = w[M[k]].dropna(); f[p2 + k + '_median'] = v.median() if len(v) else np.nan; f[p2 + k + '_p90'] = v.quantile(.9) if len(v) else np.nan
    return pd.DataFrame(rows, index=Tc.index)[FEAT]
raw = lambda p: np.maximum(0, np.expm1(p))
def mk(): return Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)), ('m', ExtraTreesRegressor(160, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=1))])
def fitpred(X, feats, tr, te):
    m = mk(); m.fit(X.iloc[tr][feats], ly[tr], m__sample_weight=W[tr]); return m.predict(X.iloc[te][feats])
MODELS = {'coords': COORDS, 'plain': NT + COORDS, 'full': FEAT + COORDS, 'lean': LEAN + COORDS, 'loc_time': COORDS + TIMES}
def run(folds, fixed_loco=False):
    P = {k: np.full(len(y), np.nan) for k in MODELS}
    for tr, te in folds:
        X = D
        if fixed_loco:
            d_out = DAY[te[0]]; X = D.copy()
            for c in np.unique(DAY[tr]):
                idx = D.index[(DAY == c)]; X.loc[idx, FEAT] = build(D.loc[idx], l[(l.camp != c) & (l.camp != d_out)]).values
            print('  rebuilt training features without day', d_out, flush=True)
        for k, f in MODELS.items(): P[k][te] = fitpred(X, f, tr, te)
    return P
def contrib(pa, pb, G):
    sst = np.sum((y - y.mean()) ** 2); return pd.Series((y - raw(pb)) ** 2 - (y - raw(pa)) ** 2).groupby(G).sum() / sst
def tint(c):
    g = len(c); se = np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2)); s = float(c.sum())
    return {'d': s, 'G': g, 'ci95': [s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se], 'up95': s + tdist.ppf(.95, g - 1) * se}
def blocks(bs):
    g = (D.x_m // bs).astype(int).astype(str) + '_' + (D.y_m // bs).astype(int).astype(str); return list(GroupKFold(5).split(D, y, g))
def buffered(base, buf):
    out = []
    for tr, te in base:
        d, _ = NearestNeighbors(n_neighbors=1).fit(XY[te]).kneighbors(XY[tr]); out.append((tr[d[:, 0] > buf], te))
    return out
def diag(folds):
    tot_d = pd.Series(DAY).value_counts(); tot_s = pd.Series(SEG).value_counts(); od = np.zeros(len(y)); os_ = np.zeros(len(y)); nn = np.zeros(len(y))
    for tr, te in folds:
        cd = pd.Series(DAY[tr]).value_counts(); cs = pd.Series(SEG[tr]).value_counts()
        od[te] = [cd.get(u, 0) / tot_d[u] for u in DAY[te]]; os_[te] = [cs.get(u, 0) / tot_s[u] for u in SEG[te]]
        nn[te] = NearestNeighbors(n_neighbors=1).fit(XY[tr]).kneighbors(XY[te])[0][:, 0]
    return {'own_day': float(od.mean()), 'own_seg': float(os_.mean()), 'nn_m': float(np.median(nn))}
designs = {'b500': blocks(500), 's5': list(GroupKFold(5).split(D, y, SEG)), 'loco_fixed': [(np.flatnonzero(DAY != c), np.flatnonzero(DAY == c)) for c in np.unique(DAY)]}
designs['buf250'] = buffered(designs['b500'], 250)
res = {}
for dn, f in designs.items():
    P = run(f, fixed_loco=(dn == 'loco_fixed')); row = diag(f)
    for k in MODELS: row[k] = float(r2_score(y, raw(P[k])))
    for a in ('plain', 'full', 'lean'):
        row[f'd_{a}|day'] = tint(contrib(P[a], P['coords'], DAY)); row[f'd_{a}|seg'] = tint(contrib(P[a], P['coords'], SEG))
    res[dn] = row; print(dn, {k: (round(v, 3) if isinstance(v, float) else (round(v['d'], 3), [round(z, 3) for z in v['ci95']], round(v['up95'], 3)) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
    np.save(OUT / f'v3_preds_{dn}.npy', P)
(OUT / 'v3_main.json').write_text(json.dumps(res, indent=1, default=float))
