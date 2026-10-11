"""Vienna: (2) audit of day-clustered intervals (per-day contributions printed; segment-level SE for comparison);
(1) retransformation robustness on naive / cross-fitted smearing / log scales, day clusters.
usage: python 24_vienna_scales.py <vienna_targets.csv.gz> <results dir> <out.json>"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
D = pd.read_csv(sys.argv[1]); y = D.target_p90.to_numpy(); ly = np.log1p(y); DAY = D.camp.astype(str).to_numpy(); SEG = D.sess.to_numpy(); XY = D[['x_m', 'y_m']].to_numpy()
def blocks(bs):
    g = (D.x_m // bs).astype(int).astype(str) + '_' + (D.y_m // bs).astype(int).astype(str); return list(GroupKFold(5).split(D, y, g))
F = {'b500': blocks(500), 's5': list(GroupKFold(5).split(D, y, SEG)), 'loco_fixed': [(np.flatnonzero(DAY != c), np.flatnonzero(DAY == c)) for c in np.unique(DAY)]}
F['buf250'] = F['b500']  # same test folds
def fid(folds):
    f = np.full(len(y), -1)
    for k, (tr, te) in enumerate(folds): f[te] = k
    return f
def scales(p, fa):
    s = np.zeros(len(y)); e = np.exp(ly - p)
    for k in np.unique(fa): s[fa == k] = e[fa != k].mean()
    return {'naive': np.maximum(0, np.expm1(p)), 'smear': np.maximum(0, np.exp(p) * s - 1), 'log': p}
def contrib(pa, pb, sc, G):
    t = ly if sc == 'log' else y; sst = np.sum((t - t.mean()) ** 2); return pd.Series((t - pb) ** 2 - (t - pa) ** 2).groupby(G).sum() / sst
def tint(c):
    g = len(c); se = float(np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2))); s = float(c.sum())
    return {'d': s, 'se': se, 'G': g, 'ci95': [s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se], 'up95': s + tdist.ppf(.95, g - 1) * se, 'per': {str(k): float(v) for k, v in c.items()}}
res = {}
P = {dn: np.load(f'{sys.argv[2]}/v3_preds_{dn}.npy', allow_pickle=True).item() for dn in F}
for dn in F:
    fa = fid(F[dn]); Sx = {k: scales(P[dn][k], fa) for k in ('coords', 'plain', 'full')}
    for sc in ('naive', 'smear', 'log'):
        r = {'plain|day': tint(contrib(Sx['plain'][sc], Sx['coords'][sc], sc, DAY)), 'full|day': tint(contrib(Sx['full'][sc], Sx['coords'][sc], sc, DAY))}
        if sc == 'naive': r['plain|seg'] = tint(contrib(Sx['plain'][sc], Sx['coords'][sc], sc, SEG)); r['full|seg'] = tint(contrib(Sx['full'][sc], Sx['coords'][sc], sc, SEG))
        res[f'{dn}|{sc}'] = r
        print(dn, sc, 'plain', round(r['plain|day']['d'], 3), [round(z, 3) for z in r['plain|day']['ci95']], 'up', round(r['plain|day']['up95'], 3), '| full', round(r['full|day']['d'], 3), 'up', round(r['full|day']['up95'], 3))
        if sc == 'naive':
            print('   day SE', round(r['plain|day']['se'], 4), 'seg SE', round(r['plain|seg']['se'], 4), 'ratio', round(r['plain|day']['se'] / r['plain|seg']['se'], 2), 'per-day', {k[5:]: round(v, 3) for k, v in r['plain|day']['per'].items()})
json.dump(res, open(sys.argv[3], 'w'), indent=1)
# coordinates-only R2 on each scale (Table V)
co = {}
for dn in F:
    Sx = scales(P[dn]['coords'], fid(F[dn]))
    for sc in ('naive', 'smear', 'log'):
        t = ly if sc == 'log' else y; co[f'{dn}|{sc}'] = float(1 - np.sum((t - Sx[sc]) ** 2) / np.sum((t - t.mean()) ** 2))
json.dump(co, open(str(sys.argv[3]).replace('.json', '_coords.json'), 'w'), indent=1)
