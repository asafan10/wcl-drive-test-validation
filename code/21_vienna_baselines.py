"""Vienna: 4G-only models, training-mean baseline, skill by distance, coverage, concurrent-LTE upper bound.
usage: python 21_vienna_baselines.py <vienna_targets.csv.gz> <phone_data_lte.parquet> <results dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
D = pd.read_csv(sys.argv[1], parse_dates=['t15']); OUT = Path(sys.argv[3])
y = D.target_p90.to_numpy(); ly = np.log1p(y); W = np.sqrt(D.n.to_numpy()); TR = D.sess.to_numpy(); CA = D.camp.to_numpy()
XY = D[['x_m', 'y_m']].to_numpy(); XYc = XY - XY.mean(0); D['xr'] = XYc[:, 0]; D['yr'] = XYc[:, 1]; COORDS = ['xr', 'yr']
feat = [c for c in D.columns if (c[:2] in ('r1', 'r2', 'r5')) or c == 'nearest_4g_m']; NT = [c for c in feat if 'sameh3' not in c]
LEAN = [f'r250_{k}_median' for k in ('rsrp', 'rsrq', 'sinr', 'dl')] + ['nearest_4g_m']
# concurrent LTE (same phone, same session, same tile and 15-min interval as the target's 5G samples)
LAT0 = 48.19; KX = 111320 * np.cos(np.radians(LAT0)); KY = 110540
l = pd.read_parquet(sys.argv[2]); l = l[(l.operator == 'A') & l.latitude.notna()].copy(); l = l.sort_values('time')
g5t = D[['sess', 't15']].copy()
l['x'] = l.longitude.astype(float) * KX; l['y'] = l.latitude.astype(float) * KY
l['tile'] = (l.x // 100).astype(int).astype(str) + '_' + (l.y // 100).astype(int).astype(str); l['t15'] = l.time.dt.floor('15min')
for c in ['rsrp_dbm', 'rsrq_db', 'sinr_db', 'dl_throughput_mbps']: l[c] = l[c].astype(float)
agg = l.groupby(['tile', 't15']).agg(**{f'cc_{k}_{s}': (k, (lambda v, q=q: v.quantile(q))) for k in ('rsrp_dbm', 'rsrq_db', 'sinr_db', 'dl_throughput_mbps') for s, q in (('med', .5), ('p90', .9))}).reset_index()
D = D.merge(agg, on=['tile', 't15'], how='left'); assert len(D) == len(y)
CC = [c for c in D.columns if c.startswith('cc_')]
raw = lambda p: np.maximum(0, np.expm1(p))
def mk(): return Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)), ('m', ExtraTreesRegressor(160, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=1))])
def et(feats, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: m = mk(); m.fit(D.iloc[tr][feats], ly[tr], m__sample_weight=W[tr]); p[te] = m.predict(D.iloc[te][feats])
    return p
def ridge(feats, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)), ('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-2, 3, 12)))]).fit(D.iloc[tr][feats], ly[tr], r__sample_weight=W[tr]); p[te] = m.predict(D.iloc[te][feats])
    return p
def trmean_raw(folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: p[te] = np.log1p(np.mean(y[tr]))  # unweighted, consistent with unweighted R2/SSE
    return p
def contrib(pa, pb):
    sst = np.sum((y - y.mean()) ** 2); return pd.Series((y - raw(pb)) ** 2 - (y - raw(pa)) ** 2).groupby(TR).sum() / sst
def tint(c):
    g = len(c); se = np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2)); s = float(c.sum())
    return {'d': s, 'ci95': [s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se], 'ci90': [s - tdist.ppf(.95, g - 1) * se, s + tdist.ppf(.95, g - 1) * se]}
def blocks(bs):
    g = (D.x_m // bs).astype(int).astype(str) + '_' + (D.y_m // bs).astype(int).astype(str); return list(GroupKFold(5).split(D, y, g))
def buffered(base, buf):
    out = []
    for tr, te in base:
        d, _ = NearestNeighbors(n_neighbors=1).fit(XY[te]).kneighbors(XY[tr]); out.append((tr[d[:, 0] > buf], te))
    return out
Ds = {'b500': blocks(500), 'b1000': blocks(1000), 's5': list(GroupKFold(5).split(D, y, TR)), 'loco': [(np.flatnonzero(CA != c), np.flatnonzero(CA == c)) for c in np.unique(CA)]}
Ds['buf250'] = buffered(Ds['b500'], 250); Ds['buf500'] = buffered(Ds['b500'], 500)
res = {'coverage': {r: float((D[f'r{r}_obs_count'].fillna(0) > 0).mean()) for r in (100, 250, 500)}, 'cc_matched': float(D[CC[0]].notna().mean())}
rows = []
for dn, f in Ds.items():
    P = {'coords': et(COORDS, f), 'plain4g_only': et(NT, f), 'lean4g_only': et(LEAN, f), 'lean4g_ridge': ridge(LEAN, f), 'plain4g_coords': et(NT + COORDS, f), 'train_mean_raw': trmean_raw(f), 'cc_coords': et(CC + COORDS, f), 'cc_only': et(CC, f)}
    dist = np.zeros(len(y))
    for tr, te in f: dist[te] = NearestNeighbors(n_neighbors=1).fit(XY[tr]).kneighbors(XY[te])[0][:, 0]
    row = {'nn_m': float(np.median(dist))}
    for k, p in P.items(): row[k] = {'R2': float(r2_score(y, raw(p))), 'MAE': float(mean_absolute_error(y, raw(p))), 'rho': float(spearmanr(y, p).statistic) if k != 'train_mean_raw' else float('nan')}
    for k in ('plain4g_only', 'lean4g_only', 'lean4g_ridge', 'coords', 'cc_only'): row[f'vs_mean|{k}'] = tint(contrib(P[k], P['train_mean_raw']))
    row['cc_vs_coords'] = tint(contrib(P['cc_coords'], P['coords'])); row['plain_vs_coords'] = tint(contrib(P['plain4g_coords'], P['coords']))
    res[dn] = row; print(dn, {k: (round(v['R2'], 3) if isinstance(v, dict) and 'R2' in v else '') for k, v in row.items()}, flush=True)
    rows.append(pd.DataFrame({'design': dn, 'trace': TR, 'y': y, 'dist': dist, **{k: raw(p) for k, p in P.items()}}))
pd.concat(rows).to_csv(OUT / 'v2_preds.csv.gz', index=False)
(OUT / 'v2_main.json').write_text(json.dumps(res, indent=1, default=float))
