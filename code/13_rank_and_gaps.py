"""Spearman change (plain 4G + coordinates minus coordinates) with 95 % cluster-bootstrap and leave-one-cluster-out jackknife intervals (Cork: traces;
Vienna: days), coordinates-only R2 per back-transformation scale (Cork), and the 500 m-block minus LOTO gaps on three
scales for full 4G and for plain 4G (time-window features removed).
Needs: results/r4_main_preds.npy (10_cork_designs.py main), data/vienna_targets.csv.gz and results/v3_preds_*.npy (22).
Output: results/r12_rank.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
R = Path(__file__).resolve().parent; OUT = R.parent / 'results'; sys.path.insert(0, str(R))
import cork_common as st
y, ly, TR, dt = st.y, np.log1p(st.y), st.TRACE, st.dt

def fold_id(folds, n):
    f = np.full(n, -1)
    for k, (tr, te) in enumerate(folds): f[te] = k
    return f
def scales(p, fid, t_log):
    s = np.zeros(len(p)); e = np.exp(t_log - p)
    for k in np.unique(fid): s[fid == k] = e[fid != k].mean()
    return {'naive': np.maximum(0, np.expm1(p)), 'smear': np.maximum(0, np.exp(p) * s - 1), 'log': p}
def r2(p, sc, yy, lyy):
    t = lyy if sc == 'log' else yy; return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
def boot_drho(t, pa, pb, G, B=2000, seed=3):
    ug = np.unique(G); idx = {u: np.flatnonzero(G == u) for u in ug}; rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        i = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); out.append(spearmanr(t[i], pa[i]).statistic - spearmanr(t[i], pb[i]).statistic)
    d = spearmanr(t, pa).statistic - spearmanr(t, pb).statistic; return {'d': float(d), 'ci95': np.percentile(out, [2.5, 97.5]).tolist()}

def jack_drho(t, pa, pb, G):
    """leave-one-cluster-out jackknife interval for the change in Spearman rho (t(G-1) quantile)"""
    from scipy.stats import t as tdist
    d = spearmanr(t, pa).statistic - spearmanr(t, pb).statistic; ug = np.unique(G)
    th = np.array([spearmanr(t[G != u], pa[G != u]).statistic - spearmanr(t[G != u], pb[G != u]).statistic for u in ug])
    g = len(ug); se = np.sqrt((g - 1) / g * np.sum((th - th.mean()) ** 2)); q = tdist.ppf(.975, g - 1)
    return {'d': float(d), 'ci95': [float(d - q * se), float(d + q * se)]}

res = {}
P = np.load(OUT / 'r4_main_preds.npy', allow_pickle=True).item()
D = {'spatial': list(GroupKFold(5).split(dt, y, dt.group500)), 'trace5': list(GroupKFold(5).split(dt, y, TR)), 'loto': st.LOTO}
for dn, f in D.items():
    res[f'cork|{dn}|rho'] = boot_drho(ly, P[dn]['g4c_notime'], P[dn]['coords'], TR)
    res[f'cork|{dn}|rho_jack'] = jack_drho(ly, P[dn]['g4c_notime'], P[dn]['coords'], TR)
    Sc = scales(P[dn]['coords'], fold_id(f, len(y)), ly)
    for sc in ('naive', 'smear', 'log'): res[f'cork|{dn}|coords|{sc}'] = r2(Sc[sc], sc, y, ly)
for a in ('g4c', 'g4c_notime'):
    for sc in ('naive', 'smear', 'log'):
        v = {}
        for dn in ('spatial', 'loto'):
            fi = fold_id(D[dn], len(y)); v[dn] = r2(scales(P[dn][a], fi, ly)[sc], sc, y, ly) - r2(scales(P[dn]['coords'], fi, ly)[sc], sc, y, ly)
        res[f'cork|gap_loto|{a}|{sc}'] = v['spatial'] - v['loto']
V = pd.read_csv(R.parent / 'data' / 'vienna_targets.csv.gz'); vly = np.log1p(V.target_p90.to_numpy()); DAY = V.camp.astype(str).to_numpy()
for dn in ('b500', 's5', 'loco_fixed', 'buf250'):
    Pv = np.load(OUT / f'v3_preds_{dn}.npy', allow_pickle=True).item()
    res[f'vienna|{dn}|rho'] = boot_drho(vly, Pv['plain'], Pv['coords'], DAY)
    res[f'vienna|{dn}|rho_jack'] = jack_drho(vly, Pv['plain'], Pv['coords'], DAY)
json.dump(res, open(OUT / 'r12_rank.json', 'w'), indent=1, default=float)
for k, v in res.items(): print(k, round(v, 4) if not isinstance(v, dict) else (round(v['d'], 3), [round(z, 3) for z in v['ci95']]))
