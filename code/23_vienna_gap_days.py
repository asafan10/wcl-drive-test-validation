"""Vienna: 500 m-block vs segment x5 design gap with day (and segment) clusters.
usage: python 23_vienna_gap_days.py <vienna_targets.csv.gz> <results dir> <out.json>"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import t as tdist
D = pd.read_csv(sys.argv[1]); y = D.target_p90.to_numpy(); DAY = D.camp.astype(str).to_numpy(); SEG = D.sess.to_numpy()
raw = lambda p: np.maximum(0, np.expm1(p)); sst = np.sum((y - y.mean())**2)
P = {k: np.load(f'{sys.argv[2]}/v3_preds_{k}.npy', allow_pickle=True).item() for k in ('b500', 's5')}
def c(P, a, G): return pd.Series((y - raw(P['coords']))**2 - (y - raw(P[a]))**2).groupby(G).sum() / sst
def tint(v):
    g = len(v); se = np.sqrt(g/(g-1)*np.sum((v-v.mean())**2)); s = float(v.sum()); q = tdist.ppf(.975, g-1)
    return {'d': s, 'ci95': [s-q*se, s+q*se], 'per': {str(k): round(float(x), 4) for k, x in v.items()}}
out = {}
for a in ('full', 'plain', 'lean'):
    for gn, G in (('day', DAY), ('seg', SEG)):
        out[f'{a}|{gn}'] = tint(c(P['b500'], a, G) - c(P['s5'], a, G))
for k, v in out.items(): print(k, round(v['d'], 3), [round(z, 3) for z in v['ci95']], v['per'] if '|day' in k else '')
json.dump(out, open(sys.argv[3], 'w'), indent=1)
