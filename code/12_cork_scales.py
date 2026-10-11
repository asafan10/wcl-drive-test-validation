"""Cork: back-transformation robustness of the plain/full 4G changes (Table V, Section V-F).
Three scales: naive expm1; cross-fitted Duan smearing (factor for a test fold from out-of-fold residuals of the other folds); log scale.
modes: main  -> designs 500 m blocks, trace x5 (standard partition), LOTO from r4_main_preds.npy
       parts s0 s1 -> trace x5 random partitions (as r5 parts), per-trace contributions on the three scales"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import GroupKFold
R = Path(__file__).resolve().parent; OUT = R.parent / 'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(R)); import cork_common as st; st.cv = st
cv, dt, y, TR = st.cv, st.dt, st.y, st.TRACE; ly = np.log1p(y); leg = list(cv.legacy); W = cv.W
NT = [c for c in leg if 'sameh3' not in c]
def fold_id(folds):
    f = np.full(len(y), -1)
    for k, (tr, te) in enumerate(folds): f[te] = k
    return f
def scales(p, fid):
    out = {'naive': np.maximum(0, np.expm1(p))}
    s = np.zeros(len(y)); e = np.exp(ly - p)
    for k in np.unique(fid): s[fid == k] = e[fid != k].mean()
    out['smear'] = np.maximum(0, np.exp(p) * s - 1); out['log'] = p; return out
def contrib(pa, pb, sc):
    t = ly if sc == 'log' else y; sst = np.sum((t - t.mean()) ** 2)
    return pd.Series((t - pb) ** 2 - (t - pa) ** 2).groupby(TR).sum() / sst
def r2(p, sc):
    t = ly if sc == 'log' else y; return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
def tint(c):
    g = len(c); se = np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2)); s = float(c.sum())
    return {'d': s, 'ci95': [s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se], 'up95': s + tdist.ppf(.95, g - 1) * se, 'lo95': s - tdist.ppf(.95, g - 1) * se}
def et(feats, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds: m = cv.fit(cv.et(3, 1., 160, 42), dt.iloc[tr][feats], ly[tr], W[tr]); p[te] = m.predict(dt.iloc[te][feats])
    return p
mode = sys.argv[1]
if mode == 'main':
    P = np.load(OUT / 'r4_main_preds.npy', allow_pickle=True).item()
    D = {'spatial': list(GroupKFold(5).split(dt, y, dt.group500)), 'trace5': list(GroupKFold(5).split(dt, y, TR)), 'loto': st.LOTO}
    res = {}
    for dn, f in D.items():
        fid = fold_id(f); S = {k: scales(P[dn][k], fid) for k in ('coords', 'g4c', 'g4c_notime')}
        for sc in ('naive', 'smear', 'log'):
            res[f'{dn}|{sc}'] = {'coords': r2(S['coords'][sc], sc), 'plain': tint(contrib(S['g4c_notime'][sc], S['coords'][sc], sc)), 'full': tint(contrib(S['g4c'][sc], S['coords'][sc], sc))}
            r = res[f'{dn}|{sc}']; print(dn, sc, round(r['coords'], 3), 'plain', round(r['plain']['d'], 3), round(r['plain']['up95'], 3), 'full', round(r['full']['d'], 3), round(r['full']['up95'], 3), flush=True)
    (OUT / 'r10_scale_main.json').write_text(json.dumps(res, indent=1, default=float))
if mode == 'parts':
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); ug = np.unique(TR); C = []
    for s in range(s0, s1):
        perm = np.random.default_rng(s).permutation(ug); grp = {g: i % 5 for i, g in enumerate(perm)}; fa = np.array([grp[t] for t in TR])
        folds = [(np.flatnonzero(fa != k), np.flatnonzero(fa == k)) for k in range(5)]
        Sx = {k: scales(et(fs, folds), fa) for k, fs in (('coords', cv.coords), ('full', leg + cv.coords), ('plain', NT + cv.coords))}
        row = {'seed': s}
        for sc in ('naive', 'smear', 'log'):
            row[sc] = {'coords': r2(Sx['coords'][sc], sc), 'plain': contrib(Sx['plain'][sc], Sx['coords'][sc], sc).to_dict(), 'full': contrib(Sx['full'][sc], Sx['coords'][sc], sc).to_dict()}
        C.append(row); print(s, flush=True)
    (OUT / f'r10_parts_{s0}_{s1}.json').write_text(json.dumps(C, default=float))
if mode == 'rand36':
    # random-drop control on three scales: gap = (R2 drop - R2 coords) under 500 m blocks minus the same under LOTO
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); P = np.load(OUT / 'r4_main_preds.npy', allow_pickle=True).item()
    D = {'spatial': list(GroupKFold(5).split(dt, y, dt.group500)), 'loto': st.LOTO}; FI = {k: fold_id(f) for k, f in D.items()}
    base = {dn: scales(P[dn]['coords'], FI[dn]) for dn in D}; rows = []
    for s in range(s0, s1):
        drop = set(np.random.default_rng(1000 + s).choice(NT, 36, replace=False)); keep = [c for c in leg if c not in drop]
        Sx = {dn: scales(et(keep + cv.coords, f), FI[dn]) for dn, f in D.items()}; row = {'seed': s}
        for sc in ('naive', 'smear', 'log'): row[sc] = (r2(Sx['spatial'][sc], sc) - r2(base['spatial'][sc], sc)) - (r2(Sx['loto'][sc], sc) - r2(base['loto'][sc], sc))
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'r11_rand36_{s0}_{s1}.csv', index=False)
