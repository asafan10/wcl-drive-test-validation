"""Cork: model comparison under the validation designs (Table III, Sections V-B and V-C, Fig. 1).
modes: rand36 s0 s1 | area | blocks | nostatic | lean | perm | refit s0 s1"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
R = Path(__file__).resolve().parent; OUT = R.parent / 'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(R)); import cork_common as st; st.cv = st
cv, dt, y, TR = st.cv, st.dt, st.y, st.TRACE; ly = np.log1p(y); leg = list(cv.legacy)
TW = [c for c in leg if 'sameh3' in c]; NT = [c for c in leg if 'sameh3' not in c]
LEAN = [f'r250_{m}_median' for m in ('rsrp_dbm', 'rsrq_db', 'snr_db', 'cqi', 'dl_mbps')] + ['nearest_4g_measurement_m']
XY = dt[['x_m', 'y_m']].to_numpy(); W = cv.W
raw = lambda p: np.maximum(0, np.expm1(p))
r2 = lambda p, idx=None: r2_score(y, raw(p)) if idx is None else r2_score(y[idx], raw(p)[idx])

def designs(mask=None):
    D = {'spatial': list(GroupKFold(5).split(dt, y, dt.group500)), 'trace5': list(GroupKFold(5).split(dt, y, TR)), 'loto': st.LOTO}
    return D

def et(feats, folds, w=None, X=None):
    X = dt if X is None else X; w = W if w is None else w; p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = cv.fit(cv.et(3, 1., 160, 42), X.iloc[tr][feats], ly[tr], w[tr]); p[te] = m.predict(X.iloc[te][feats])
    return p
def knn(folds, k=10):
    p = np.full(len(y), np.nan)
    for tr, te in folds: p[te] = KNeighborsRegressor(min(k, len(tr)), weights='distance').fit(XY[tr], ly[tr]).predict(XY[te])
    return p
def gp_fit(Z, t):
    k = Ck(1.0) * RBF(length_scale=0.3, length_scale_bounds=(0.02, 5)) + WhiteKernel(0.5, (1e-3, 5))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Z, t)
Zc = (XY - XY.mean(0)) / 1000.
def gp(folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: p[te] = gp_fit(Zc[tr], ly[tr]).predict(Zc[te])
    return p
def rk(folds):
    """regression kriging: ridge on lean 4G features, GP on residuals over coordinates"""
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        lm = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-2, 3, 12)))]).fit(dt.iloc[tr][LEAN], ly[tr])
        res = ly[tr] - lm.predict(dt.iloc[tr][LEAN]); g = gp_fit(Zc[tr], res)
        p[te] = lm.predict(dt.iloc[te][LEAN]) + g.predict(Zc[te])
    return p

def cluster_t(pa, pb, idx=None):
    """small-sample cluster interval for R2(a)-R2(b): sum of per-trace SSE contributions, t(G-1)"""
    from scipy.stats import t as tdist
    ya = y; sst = np.sum((ya - ya.mean()) ** 2)
    c = pd.Series((ya - raw(pb)) ** 2 - (ya - raw(pa)) ** 2).groupby(TR).sum() / sst
    G = len(c); se = np.sqrt(G / (G - 1) * np.sum((c - c.mean()) ** 2)); q = tdist.ppf(.975, G - 1)
    return float(c.sum()), float(c.sum() - q * se), float(c.sum() + q * se), c

mode = sys.argv[1]; res = {}
if mode == 'rand36':
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); rows = []; D = designs()
    for s in range(s0, s1):
        drop = set(np.random.default_rng(1000 + s).choice(NT, 36, replace=False)); keep = [c for c in leg if c not in drop]
        row = {'seed': s}
        for dn, f in D.items(): row[dn] = r2(et(keep + cv.coords, f))
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'r4_rand36_{s0}_{s1}.csv', index=False); sys.exit()

if mode == 'main':
    """three designs: coords, g4c, g4c_notime, lean ET, regression kriging; cluster-t intervals; interaction; held-out permutation importance"""
    D = designs(); P = {}
    gi = {g: np.flatnonzero(TR == g) for g in np.unique(TR)}; ug = np.unique(TR); brng = np.random.default_rng(5)
    BI = [np.concatenate([gi[g] for g in brng.choice(ug, len(ug))]) for _ in range(2000)]
    for dn, f in D.items():
        P[dn] = {'coords': et(cv.coords, f), 'g4c': et(leg + cv.coords, f), 'g4c_notime': et(NT + cv.coords, f), 'lean': et(LEAN + cv.coords, f), 'rk': rk(f), 'knn': knn(f), 'gp': gp(f)}
        res[dn] = {k: r2(v) for k, v in P[dn].items()}; print(dn, res[dn], flush=True)
        for a in ['g4c', 'g4c_notime', 'lean', 'rk']:
            d0, lo, hi, _ = cluster_t(P[dn][a], P[dn]['coords'])
            b = np.array([r2(P[dn][a], i) - r2(P[dn]['coords'], i) for i in BI])
            res[f'd|{dn}|{a}'] = {'d': d0, 'ct95': [lo, hi], 'boot95': np.percentile(b, [2.5, 97.5]).tolist(), 'boot90': np.percentile(b, [5, 95]).tolist()}
    for a in ['g4c', 'g4c_notime']:
        for other in ['trace5', 'loto']:
            _, _, _, cs = cluster_t(P['spatial'][a], P['spatial']['coords']); _, _, _, co = cluster_t(P[other][a], P[other]['coords'])
            c = cs - co; from scipy.stats import t as tdist; G = len(c); se = np.sqrt(G / (G - 1) * np.sum((c - c.mean()) ** 2)); q = tdist.ppf(.975, G - 1)
            b = np.array([(r2(P['spatial'][a], i) - r2(P['spatial']['coords'], i)) - (r2(P[other][a], i) - r2(P[other]['coords'], i)) for i in BI])
            res[f'inter|{other}|{a}'] = {'d': float(c.sum()), 'ct95': [float(c.sum() - q * se), float(c.sum() + q * se)], 'boot95': np.percentile(b, [2.5, 97.5]).tolist()}
    # held-out permutation importance for g4c: permute a feature group in the test fold, 10 repeats
    rng = np.random.default_rng(7)
    for dn, f in D.items():
        base = P[dn]['g4c']; drops = {'coords': [], 'time_window_4g': [], 'plain_4g': []}
        for rep in range(10):
            q = {k: np.full(len(y), np.nan) for k in drops}
            for tr, te in f:
                m = cv.fit(cv.et(3, 1., 160, 42), dt.iloc[tr][leg + cv.coords], ly[tr], W[tr])
                for k, cols in [('coords', cv.coords), ('time_window_4g', TW), ('plain_4g', NT)]:
                    X = dt.iloc[te][leg + cv.coords].copy(); X[cols] = X[cols].to_numpy()[rng.permutation(len(te))]; q[k][te] = m.predict(X)
            for k in drops: drops[k].append(r2(base) - r2(q[k]))
            if rep == 2: break
        res[f'pimp|{dn}'] = {k: float(np.mean(v)) for k, v in drops.items()}; print(dn, res[f'pimp|{dn}'], flush=True)
    np.save(OUT / 'r4_main_preds.npy', P, allow_pickle=True)

if mode == 'area':
    """held-out areas: 500 m / 1 km block folds; training targets within a buffer of any test target are removed"""
    nn_all = NearestNeighbors().fit(XY)
    for bs in (500, 1000):
        grp = (dt.x_m // bs).astype(int).astype(str) + '_' + (dt.y_m // bs).astype(int).astype(str)
        base = list(GroupKFold(5).split(dt, y, grp))
        for buf in (0, 250, 500, 1000):
            f = []; dist = np.zeros(len(y))
            for tr, te in base:
                if buf > 0:
                    d, _ = NearestNeighbors(n_neighbors=1).fit(XY[te]).kneighbors(XY[tr]); tr = tr[d[:, 0] > buf]
                dd, _ = NearestNeighbors(n_neighbors=1).fit(XY[tr]).kneighbors(XY[te]); dist[te] = dd[:, 0]; f.append((tr, te))
            P = {'coords': et(cv.coords, f), 'g4c': et(leg + cv.coords, f), 'g4c_notime': et(NT + cv.coords, f), 'lean': et(LEAN + cv.coords, f), 'rk': rk(f), 'knn': knn(f), 'gp': gp(f)}
            row = {k: r2(v) for k, v in P.items()}; row['median_nn_m'] = float(np.median(dist)); row['min_train'] = int(min(len(a) for a, _ in f))
            for a in ['g4c', 'g4c_notime', 'lean', 'rk']:
                d0, lo, hi, _ = cluster_t(P[a], P['coords']); row[f'd_{a}'] = [d0, lo, hi]
                d0, lo, hi, _ = cluster_t(P[a], P['knn']); row[f'dk_{a}'] = [d0, lo, hi]
            res[f'{bs}|{buf}'] = row; print(bs, buf, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

if mode == 'blocks':
    for bs in (1000, 2000):
        grp = (dt.x_m // bs).astype(int).astype(str) + '_' + (dt.y_m // bs).astype(int).astype(str); f = list(GroupKFold(5).split(dt, y, grp))
        trs = pd.Series(TR).value_counts(); share = np.zeros(len(y))
        for tr, te in f:
            c = pd.Series(TR[tr]).value_counts(); share[te] = [c.get(t, 0) / max(trs[t] - 1, 1) for t in TR[te]]
        P = {'coords': et(cv.coords, f), 'g4c': et(leg + cv.coords, f), 'g4c_notime': et(NT + cv.coords, f)}
        row = {k: r2(v) for k, v in P.items()}; row['own_share'] = float(share.mean())
        for a in ['g4c', 'g4c_notime']: d0, lo, hi, _ = cluster_t(P[a], P['coords']); row[f'd_{a}'] = [d0, lo, hi]
        res[f'block{bs}'] = row; print(bs, row, flush=True)

if mode == 'nostatic':
    keep = ~pd.Series(TR).str.contains('/Static/').to_numpy(); idx = np.flatnonzero(keep)
    sub = dt.iloc[idx].reset_index(drop=True); ys = y[idx]; lys = ly[idx]; ws = W[idx]; trs = TR[idx]
    D = {'spatial': list(GroupKFold(5).split(sub, ys, sub.group500)), 'trace5': list(GroupKFold(5).split(sub, ys, trs)), 'loto': [(np.flatnonzero(trs != t), np.flatnonzero(trs == t)) for t in np.unique(trs)]}
    for dn, f in D.items():
        row = {}
        for name, feats in [('coords', cv.coords), ('g4c', leg + cv.coords), ('g4c_notime', NT + cv.coords)]:
            p = np.zeros(len(ys))
            for tr, te in f: m = cv.fit(cv.et(3, 1., 160, 42), sub.iloc[tr][feats], lys[tr], ws[tr]); p[te] = m.predict(sub.iloc[te][feats])
            row[name] = float(r2_score(ys, raw(p)))
        row['d_g4c'] = row['g4c'] - row['coords']; row['d_g4c_notime'] = row['g4c_notime'] - row['coords']; res[dn] = row; print(dn, row, flush=True)
    res['n_targets'] = int(len(ys)); res['n_traces'] = int(len(np.unique(trs)))

if mode == 'perm':
    D = designs(); rng = np.random.default_rng(0); PERM = [rng.permutation(len(dt)) for _ in range(20)]
    for dn in ['spatial', 'trace5', 'loto']:
        f = D[dn]
        for name, feats in [('g4c', leg), ('g4c_notime', NT)]:
            real = r2(et(feats + cv.coords, f)); sh = []
            for idx in PERM:
                X = dt.copy(); X[feats] = dt[feats].to_numpy()[idx]; sh.append(r2(et(feats + cv.coords, f, X=X)))
            res[f'{dn}|{name}'] = {'real': real, 'perm': sh, 'p_upper': float((1 + sum(s >= real for s in sh)) / 21)}; print(dn, name, real, min(sh), max(sh), flush=True)

if mode == 'refit':
    """Poisson(1) trace-weight refit bootstrap: weights enter training and the R2 evaluation"""
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); D = designs(); rows = []; ug = np.unique(TR)
    for b in range(s0, s1):
        wt = dict(zip(ug, np.random.default_rng(100 + b).poisson(1., len(ug)))); tw = np.array([wt[t] for t in TR], float); ok = tw > 0
        row = {'rep': b}
        for dn in ['spatial', 'trace5']:
            f = [(tr[ok[tr]], te[ok[te]]) for tr, te in D[dn]]; f = [(a, c) for a, c in f if len(a) and len(c)]
            for name, feats in [('coords', cv.coords), ('g4c', leg + cv.coords), ('g4c_notime', NT + cv.coords)]:
                p = et(feats, f, w=W * tw); m = ~np.isnan(p) & ok
                row[f'{dn}|{name}'] = r2_score(y[m], raw(p)[m], sample_weight=tw[m])
        rows.append(row); print(b, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'r4_refit_{s0}_{s1}.csv', index=False); sys.exit()

(OUT / f'r4_{mode}.json').write_text(json.dumps(res, indent=1, default=float))
