"""Cork: 4G checks (Sections V-B to V-D). Modes
  parts s0 s1  : trace x5 random partitions, per-trace SSE contributions for coords / full 4G / plain 4G
  only4g       : 4G-only models (no coordinates) vs coordinates and training mean in all designs; per-target predictions and
                 nearest-training distances for skill-by-distance; absolute R2, MAE, Spearman
  misc         : 4G coverage shares; missingness indicators; dense-coverage subset; permutation importance with trace-level
                 intervals (20 permutations); no-static intervals; concurrent LTE-anchor upper bound"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
R = Path(__file__).resolve().parent; OUT = R.parent / 'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(R)); import cork_common as st; st.cv = st
cv, dt, y, TR = st.cv, st.dt, st.y, st.TRACE; ly = np.log1p(y); leg = list(cv.legacy); W = cv.W
TW = [c for c in leg if 'sameh3' in c]; NT = [c for c in leg if 'sameh3' not in c]
LEAN = [f'r250_{m}_median' for m in ('rsrp_dbm', 'rsrq_db', 'snr_db', 'cqi', 'dl_mbps')] + ['nearest_4g_measurement_m']
XY = dt[['x_m', 'y_m']].to_numpy(); raw = lambda p: np.maximum(0, np.expm1(p)); G = len(np.unique(TR))
def et_model(ind=False): return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=ind, keep_empty_features=True)), ('m', ExtraTreesRegressor(160, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=1))])
def et(feats, folds, ind=False, X=None):
    X = dt if X is None else X; p = np.full(len(y), np.nan)
    for tr, te in folds: m = et_model(ind); m.fit(X.iloc[tr][feats], ly[tr], m__sample_weight=W[tr]); p[te] = m.predict(X.iloc[te][feats])
    return p
def ridge(feats, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)), ('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-2, 3, 12)))]).fit(dt.iloc[tr][feats], ly[tr], r__sample_weight=W[tr]); p[te] = m.predict(dt.iloc[te][feats])
    return p
def trmean(folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: p[te] = np.average(ly[tr], weights=W[tr])
    return p
def trmean_raw(folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: p[te] = np.log1p(np.mean(y[tr]))  # unweighted, consistent with unweighted R2/SSE
    return p
def contrib(pa, pb, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask; yy = y[m]; sst = np.sum((yy - yy.mean()) ** 2)
    return pd.Series((yy - raw(pb[m])) ** 2 - (yy - raw(pa[m])) ** 2).groupby(TR[m]).sum() / sst
def tint(c, two=True):
    g = len(c); se = np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2)); s = float(c.sum())
    return {'d': s, 'ci95': [s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se], 'ci90': [s - tdist.ppf(.95, g - 1) * se, s + tdist.ppf(.95, g - 1) * se]}
def blocks(bs):
    g = dt.group500 if bs == 500 else (dt.x_m // bs).astype(int).astype(str) + '_' + (dt.y_m // bs).astype(int).astype(str); return list(GroupKFold(5).split(dt, y, g))
def buffered(base, buf):
    out = []
    for tr, te in base:
        d, _ = NearestNeighbors(n_neighbors=1).fit(XY[te]).kneighbors(XY[tr]); out.append((tr[d[:, 0] > buf], te))
    return out
def designs():
    D = {'b500': blocks(500), 'b1000': blocks(1000), 'b2000': blocks(2000), 't5': list(GroupKFold(5).split(dt, y, TR)), 'loto': st.LOTO}
    for b in (250, 500, 1000): D[f'buf{b}'] = buffered(D['b500'], b)
    return D
def nn_dist(folds):
    d = np.zeros(len(y))
    for tr, te in folds: d[te] = NearestNeighbors(n_neighbors=1).fit(XY[tr]).kneighbors(XY[te])[0][:, 0]
    return d
mode = sys.argv[1]; res = {}
if mode == 'parts':
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); ug = np.unique(TR); C = []
    for s in range(s0, s1):
        perm = np.random.default_rng(s).permutation(ug); grp = {g: i % 5 for i, g in enumerate(perm)}; f = np.array([grp[t] for t in TR])
        folds = [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(5)]
        pc, pg, pn = et(cv.coords, folds), et(leg + cv.coords, folds), et(NT + cv.coords, folds)
        C.append(pd.DataFrame({'seed': s, 'full': contrib(pg, pc), 'plain': contrib(pn, pc)})); print(s, flush=True)
    pd.concat(C).rename_axis('trace').reset_index().to_csv(OUT / f'r5_parts_{s0}_{s1}.csv', index=False); sys.exit()
if mode == 'only4g':
    D = designs(); rows = []
    for dn, f in D.items():
        P = {'coords': et(cv.coords, f), 'plain4g_only': et(NT, f), 'lean4g_only': et(LEAN, f), 'lean4g_ridge': ridge(LEAN, f), 'plain4g_coords': et(NT + cv.coords, f), 'train_mean': trmean(f), 'train_mean_raw': trmean_raw(f)}
        dist = nn_dist(f); row = {'nn_m': float(np.median(dist))}
        for k, p in P.items():
            row[k] = {'R2': float(r2_score(y, raw(p))), 'MAE': float(mean_absolute_error(y, raw(p))), 'rho': float(spearmanr(y, p).statistic) if not k.startswith('train_mean') else float('nan')}
        for k in ('plain4g_only', 'lean4g_only', 'lean4g_ridge', 'coords'): row[f'vs_mean|{k}'] = tint(contrib(P[k], P['train_mean_raw']))
        res[dn] = row; print(dn, {k: (round(v['R2'], 3) if isinstance(v, dict) and 'R2' in v else v) for k, v in row.items() if not k.startswith('vs_')}, flush=True)
        rows.append(pd.DataFrame({'design': dn, 'trace': TR, 'y': y, 'dist': dist, **{k: raw(p) for k, p in P.items()}}))
    pd.concat(rows).to_csv(OUT / 'r5_only4g_preds.csv.gz', index=False)
if mode == 'misc':
    D = designs(); M = {k: D[k] for k in ('b500', 't5', 'loto')}
    res['coverage'] = {r: float((dt[f'r{r}_obs_count'].fillna(0) > 0).mean()) for r in (100, 250, 500)}
    res['missing_share'] = {'all': float(dt[leg].isna().mean().mean()), 'plain': float(dt[NT].isna().mean().mean()), 'tw': float(dt[TW].isna().mean().mean())}
    dense = (dt['r100_obs_count'].fillna(0) > 0).to_numpy(); res['dense_n'] = int(dense.sum())
    for dn, f in M.items():
        pc = et(cv.coords, f); pn = et(NT + cv.coords, f); pi = et(NT + cv.coords, f, ind=True); pg = et(leg + cv.coords, f)
        res[f'ind|{dn}'] = tint(contrib(pi, pc)); res[f'dense|{dn}|plain'] = tint(contrib(pn, pc, dense)); res[f'dense|{dn}|full'] = tint(contrib(pg, pc, dense))
        print(dn, 'ind', res[f'ind|{dn}']['d'], 'dense plain', res[f'dense|{dn}|plain']['d'], flush=True)
    # permutation importance with intervals (fit once per fold, 20 permutations of the time-window block in the test fold)
    for dn, f in M.items():
        base = np.full(len(y), np.nan); perm = np.zeros((20, len(y))); rng = np.random.default_rng(7)
        for tr, te in f:
            m = et_model(); m.fit(dt.iloc[tr][leg + cv.coords], ly[tr], m__sample_weight=W[tr]); X = dt.iloc[te][leg + cv.coords]; base[te] = m.predict(X)
            for r in range(20):
                Xp = X.copy(); Xp[TW] = Xp[TW].to_numpy()[rng.permutation(len(te))]; perm[r, te] = m.predict(Xp)
        c = sum(contrib(base, perm[r]) for r in range(20)) / 20   # positive = shuffling lowers R2
        res[f'pimp|{dn}'] = tint(c); print('pimp', dn, res[f'pimp|{dn}'], flush=True)
    # no-static intervals
    keep = ~pd.Series(TR).str.contains('/Static/').to_numpy()
    for dn, f in M.items():
        fk = [(tr[keep[tr]], te[keep[te]]) for tr, te in f]; fk = [(a, b) for a, b in fk if len(a) and len(b)]
        pc = et(cv.coords, fk); pg = et(leg + cv.coords, fk); pn = et(NT + cv.coords, fk)
        res[f'nostatic|{dn}|full'] = tint(contrib(pg, pc, keep)); res[f'nostatic|{dn}|plain'] = tint(contrib(pn, pc, keep))
    # concurrent LTE-anchor upper bound: KPIs of the target's own 5G samples (LTE anchor in the UCC 5G corpus)
    raw5 = pd.read_csv(R.parent / 'data' / 'ucc_prepared.csv.gz', parse_dates=['timestamp'])
    b = raw5[(raw5.operator.astype(str) == 'B') & raw5.generation.isin(['4G', '5G'])]; b = b[b[['latitude', 'longitude', 'dl_mbps']].notna().all(axis=1)]
    lat_ref = float(b.latitude.median()); nr = b[b.generation == '5G'].copy()
    nr['x_m'] = nr.longitude * 111320 * np.cos(np.deg2rad(lat_ref)); nr['y_m'] = nr.latitude * 110540
    nr['tile100'] = np.floor(nr.x_m / 100).astype(int).astype(str) + '_' + np.floor(nr.y_m / 100).astype(int).astype(str); nr['t15'] = nr.timestamp.dt.floor('15min')
    agg = nr.groupby(['tile100', 't15']).agg(**{f'cc_{k}_{s}': (k, (lambda v, q=q: v.quantile(q))) for k in ('rsrp_dbm', 'rsrq_db', 'snr_db', 'cqi') for s, q in (('med', .5), ('p90', .9))}).reset_index()
    X = dt.copy(); X['t15'] = pd.to_datetime(X.t15); X = X.merge(agg, on=['tile100', 't15'], how='left'); assert len(X) == len(dt)
    CC = [c for c in X.columns if c.startswith('cc_')]; res['cc_matched'] = float(X[CC[0]].notna().mean())
    for dn, f in M.items():
        pc = et(cv.coords, f, X=X); pcc = et(CC + cv.coords, f, X=X); pco = et(CC, f, X=X)
        res[f'concurrent|{dn}'] = {'coords': float(r2_score(y, raw(pc))), 'cc+coords': float(r2_score(y, raw(pcc))), 'cc_only': float(r2_score(y, raw(pco))), 'd': tint(contrib(pcc, pc))}
        print('concurrent', dn, res[f'concurrent|{dn}'], flush=True)
(OUT / f'r5_{mode}.json').write_text(json.dumps(res, indent=1, default=float))
