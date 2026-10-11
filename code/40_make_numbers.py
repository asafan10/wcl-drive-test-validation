"""Compute every number quoted in the letter from the result files, write results/letter_numbers.json, and compare it
with results/letter_numbers_reference.json (the values printed in the submitted letter).
usage: python 40_make_numbers.py"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
R = Path(__file__).resolve().parent; ROOT = R.parent; RES = ROOT / 'results'; sys.path.insert(0, str(R))
import cork_common as st

sg = lambda x: ('0.000' if abs(x) < .0005 else f'{x:+.3f}').replace('-', '−')        # signed, 3 decimals
f3 = lambda x: f'{x:.3f}'.replace('-', '−')
f2 = lambda x: ('0.00' if abs(x) < .005 else f'{x:.2f}').replace('-', '−')
iv = lambda lo, hi: f'[{sg(lo)}, {sg(hi)}]'
iv2 = lambda lo, hi: f'[{f2(lo)}, {f2(hi)}]'
J = lambda n: json.load(open(RES / n))
cat = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / pat)))])
def tint(c):
    g = len(c); se = np.sqrt(g / (g - 1) * np.sum((c - c.mean()) ** 2)); s = float(np.sum(c))
    return s, s - tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.975, g - 1) * se, s + tdist.ppf(.95, g - 1) * se, s - tdist.ppf(.95, g - 1) * se
N = {}

# ---------------------------------------------------------------- Cork data (Section III-A)
dt, y, TR = st.dt, st.y, st.TRACE
N['n5_max'] = str(int(dt.n5.max())); N['n5_med'] = f'{dt.n5.median():.0f}'

# ---------------------------------------------------------------- Cork designs (Table III)
m, bl, ar = J('r4_main.json'), J('r4_blocks.json'), J('r4_area.json')
XY = dt[['x_m', 'y_m']].to_numpy()
def blocks(bs):
    g = dt.group500 if bs == 500 else (dt.x_m // bs).astype(int).astype(str) + '_' + (dt.y_m // bs).astype(int).astype(str)
    return list(GroupKFold(5).split(dt, y, g))
def buffered(base, buf):
    out = []
    for tr, te in base:
        d, _ = NearestNeighbors(n_neighbors=1).fit(XY[te]).kneighbors(XY[tr]); out.append((tr[d[:, 0] > buf], te))
    return out
D = {'spatial': blocks(500), 'b1000': blocks(1000), 'b2000': blocks(2000), 't5': list(GroupKFold(5).split(dt, y, TR)), 'loto': st.LOTO}
for b in (250, 500, 1000): D[f'buf{b}'] = buffered(D['spatial'], b)
tot = pd.Series(TR).value_counts()
for k, f in D.items():                           # NN distance and own-trace share per design
    nn = np.zeros(len(y)); own = np.zeros(len(y))
    for tr, te in f:
        nn[te] = NearestNeighbors(n_neighbors=1).fit(XY[tr]).kneighbors(XY[te])[0][:, 0]
        c = pd.Series(TR[tr]).value_counts(); own[te] = [c.get(u, 0) / max(tot[u] - 1, 1) for u in TR[te]]  # share of the other targets of the same trace
    N[f't_{k}_nn'] = f'{np.median(nn):.0f}'; N[f't_{k}_own'] = f'{100 * own.mean():.0f}'
def row(key, coords, dp, df):
    N[f't_{key}_c'] = f3(coords); N[f't_{key}_dp'] = sg(dp[0]); N[f't_{key}_dpi'] = iv(dp[1], dp[2]); N[f't_{key}_df'] = sg(df[0]); N[f't_{key}_dfi'] = iv(df[1], df[2])
for dn in ('spatial', 'loto'):
    a, b = m[f'd|{dn}|g4c_notime'], m[f'd|{dn}|g4c']; row(dn, m[dn]['coords'], [a['d'], *a['ct95']], [b['d'], *b['ct95']])
for bs in (1000, 2000):
    x = bl[f'block{bs}']; row(f'b{bs}', x['coords'], x['d_g4c_notime'], x['d_g4c'])
for buf in (250, 500, 1000):
    x = ar[f'500|{buf}']; row(f'buf{buf}', x['coords'], x['d_g4c_notime'], x['d_g4c'])
P5 = cat('r5_parts_*.csv').drop_duplicates(['seed', 'trace']); N['np5'] = str(P5.seed.nunique()); N['np'] = N['np5']
for col, k in (('plain', 'dp'), ('full', 'df')):
    s, lo, hi, up, _ = tint(P5.groupby('trace')[col].mean()); N[f't_t5_{k}'] = sg(s); N[f't_t5_{k}i'] = iv(lo, hi)
    if col == 'plain': N['t5avg_plain_up'] = sg(up)
C10 = list({r['seed']: r for r in sum((json.load(open(f)) for f in sorted(glob.glob(str(RES / 'r10_parts_*.json')))), [])}.values())
N['t_t5_c'] = f3(np.mean([r['naive']['coords'] for r in C10]))
for dn in ('spatial', 'loto'):                   # one-sided 95 % bounds, plain 4G
    x = m[f'd|{dn}|g4c_notime']; lo, hi = x['ct95']; se = (hi - lo) / (2 * tdist.ppf(.975, 19)); N[f'up_{dn}'] = sg(x['d'] + tdist.ppf(.95, 19) * se)
# skill against the training mean (Table III) and 4G-only models
p = pd.read_csv(RES / 'r5_only4g_preds.csv.gz')
dmap = {'b500': 'spatial', 'b1000': 'b1000', 'b2000': 'b2000', 't5': 't5', 'loto': 'loto', 'buf250': 'buf250', 'buf500': 'buf500', 'buf1000': 'buf1000'}
for dn, g in p.groupby('design'):
    r2 = lambda c: 1 - ((g.y - g[c]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(); rm = r2('train_mean_raw')
    N[f't_{dmap[dn]}_sk'] = f2((r2('coords') - rm) / (1 - rm))
    if dn == 'buf250': N['sk4_buf250'] = f2((r2('lean4g_ridge') - rm) / (1 - rm))
for k in ('b1000', 'b2000', 'buf250', 'buf500', 'buf1000'): N[f'sk_{k}'] = N[f't_{k}_sk']
o = J('r5_only4g.json')
for d in ('loto', 'buf250', 'buf500', 'buf1000'):
    x = o[d]; N[f'o_{d}_coords'] = f3(x['coords']['R2']); N[f'o_{d}_plain'] = f3(x['plain4g_only']['R2']); N[f'o_{d}_ridge'] = f3(x['lean4g_ridge']['R2']); N[f'o_{d}_tm'] = f3(x['train_mean_raw']['R2'])
g = p[p.design == 'buf250']; sst = ((g.y - g.y.mean()) ** 2).sum()
c = (((g.y - g.coords) ** 2) - ((g.y - g.lean4g_ridge) ** 2)).groupby(g.trace).sum() / sst; s, lo, hi, up, low = tint(c)
N['ridge250'] = sg(s); N['ridge250_i'] = iv(lo, hi); N['ridge250_up'] = sg(up)
N['ridge250_lo'] = sg(low); N['ridge250_p1'] = f'{1 - tdist.cdf(s / ((hi - lo) / (2 * tdist.ppf(.975, len(c) - 1))), len(c) - 1):.3f}'
N['r_loto_rk'] = f3(m['loto']['rk'])
# coverage, dense subset, concurrent LTE anchor (Cork)
mi = J('r5_misc.json')
for r in (100, 250, 500): N[f'cov_{r}'] = f"{100 * mi['coverage'][str(r)]:.0f}"
N['dense_n'] = str(mi['dense_n'])
for d in ('b500', 't5'): N[f'dense_{d}_plain'] = sg(mi[f'dense|{d}|plain']['d'])
N['dense_b500_full'] = sg(mi['dense|b500|full']['d']); N['dense_b500_full_i'] = iv(*mi['dense|b500|full']['ci95'])
N['dense_t5_full'] = sg(mi['dense|t5|full']['d'])
N['ns_t5_full'] = sg(mi['nostatic|t5|full']['d']); N['ns_t5_full_i'] = iv(*mi['nostatic|t5|full']['ci95'])
N['cc_loto'] = sg(mi['concurrent|loto']['d']['d']); N['cc_loto_i'] = iv(*mi['concurrent|loto']['d']['ci95'])

# ---------------------------------------------------------------- design dependence (Section V-C, Fig. 1)
for k in ('g4c', 'g4c_notime'): x = m[f'inter|loto|{k}']; N[f'in_loto_{k}'] = sg(x['d'])
N['in_loto_g4c_ct'] = iv(*m['inter|loto|g4c']['ct95'])
P4 = np.load(RES / 'r4_main_preds.npy', allow_pickle=True).item()
raw = lambda q: np.maximum(0, np.expm1(q)); sstc = np.sum((y - y.mean()) ** 2)
con = lambda a, b: pd.Series((y - raw(b)) ** 2 - (y - raw(a)) ** 2).groupby(TR).sum() / sstc
for a, col, k in (('g4c', 'full', 'gap5_full'), ('g4c_notime', 'plain', 'gap5_plain')):
    s, lo, hi, _, _ = tint(con(P4['spatial'][a], P4['spatial']['coords']) - P5.groupby('trace')[col].mean()); N[k] = sg(s)
    if col == 'full': N['gap5_full_i'] = iv(lo, hi)
cvm = (dt['r500_obs_count'].fillna(0) > 0).to_numpy(); N['cv500_n'] = str(int(cvm.sum())); N['cv500_pct'] = f'{100 * cvm.mean():.0f}'
N['cv500_tr'] = str(len(np.unique(TR[cvm]))); ups = []           # plain 4G on targets with 4G within 500 m
yy = y[cvm]; sst5 = np.sum((yy - yy.mean()) ** 2)
for dn, k in (('spatial', 'b500'), ('trace5', 't5'), ('loto', 'loto')):
    c5 = pd.Series((yy - raw(P4[dn]['coords'][cvm])) ** 2 - (yy - raw(P4[dn]['g4c_notime'][cvm])) ** 2).groupby(TR[cvm]).sum() / sst5
    s5, _, _, u5, _ = tint(c5); N[f'cv500_{k}'] = sg(s5); ups.append(u5)
N['cv500_up_min'], N['cv500_up_max'] = sg(min(ups)), sg(max(ups))
rf = cat('r4_refit_*.csv').drop_duplicates('rep'); N['rf_n'] = str(len(rf))
x = (rf['spatial|g4c'] - rf['spatial|coords']) - (rf['trace5|g4c'] - rf['trace5|coords']); N['rf_in_g4c'] = sg(x.mean()); N['rf_in_g4c_i'] = iv(x.quantile(.025), x.quantile(.975))
N['rd_n'] = str(len(cat('r4_rand36_*.csv').drop_duplicates('seed')))

# ---------------------------------------------------------------- per-sample re-evaluation (Table II)
rv, r2s = J('reval.json'), J('reval2.json')
for city in ('cork', 'vienna'):
    for dn in ('random', 'temporal_within_trace', 'spatial_500m', 'trace_or_day_out'):
        x = rv[city][dn]; k = f'{city}_{dn}'
        N[f'{k}_own'] = f"{100 * x['own_unit']:.0f}"; N[f'{k}_adj'] = f"{100 * x['adj_10s']:.0f}"
        N[f'{k}_rl'] = f3(x['R2|radio+loc']); N[f'{k}_ro'] = f3(x['R2|radio_only']); N[f'{k}_lo'] = f3(x['R2|loc_only'])
    N[f'{city}_random_rl'] = f3(r2s[city]['random_seeds|radio+loc']['mean']); N[f'{city}_random_ro'] = f3(r2s[city]['random_seeds|radio_only']['mean'])
    x = r2s[city]['random_seeds|radio+loc']; N[f'{city}_rs_rl_rng'] = f"{f2(x['min'])}–{f2(x['max'])}"
    N[f'{city}_uo_rl_i'] = iv2(*r2s[city]['unit_out|radio+loc']['boot95'])
N['vienna_uo_ro_i'] = iv2(*r2s['vienna']['unit_out|radio_only']['boot95'])
N['cork_n'] = f"{rv['cork']['n_samples']:,}"
v = list(rv['vienna']['trace_or_day_out']['per_unit|radio+loc'].values()); N['vienna_day_rl_min'] = f3(min(v)); N['vienna_day_rl_max'] = f3(max(v))

# ---------------------------------------------------------------- Vienna (Section V-E, Table IV)
V = pd.read_csv(ROOT / 'data' / 'vienna_targets.csv.gz')
N['v_targets'] = f'{len(V):,}'; N['v_tiles'] = f'{V.tile.nunique():,}'; N['v_sess'] = str(V.sess.nunique()); N['v_camp'] = str(V.camp.nunique())
N['v_maxsess'] = str(V.sess.value_counts().max()); N['v_minsess'] = str(V.sess.value_counts().min())
N['v_nplain'] = str(len([c for c in V.columns if c.startswith('r') and c[1] in '125' and 'sameh3' not in c]) + 1); N['v_ntw'] = str(len([c for c in V.columns if 'sameh3' in c]))
v3, v2 = J('v3_main.json'), J('v2_main.json')
for k, x in v3.items():
    N[f'v3_{k}_c'] = f3(x['coords']); N[f'v3_{k}_lt'] = f3(x['loc_time']); N[f'v3_{k}_nn'] = f"{x['nn_m']:.0f}"; N[f'v3_{k}_own'] = f"{100 * x['own_day']:.0f}"
    N[f'v3_{k}_lean'] = sg(x['lean'] - x['coords'])
    for a, t in (('plain', 'dp'), ('full', 'df')): d = x[f'd_{a}|day']; N[f'v3_{k}_{t}'] = sg(d['d']); N[f'v3_{k}_{t}i'] = iv(*d['ci95'])
ups = [x['d_plain|day']['up95'] for x in v3.values()]; N['v3_up_min'] = f2(min(ups)); N['v3_up_max'] = f2(max(ups))
N['v_max_r2'] = f3(max(x[mm] for x in v3.values() for mm in ('coords', 'plain', 'full', 'lean', 'loc_time')))
N['v2_loco_tm'] = f3(v2['loco']['train_mean_raw']['R2']); N['v2_buf250_tm'] = f3(v2['buf250']['train_mean_raw']['R2'])
ccs = [v2[d]['cc_vs_coords']['d'] for d in ('b500', 'b1000', 's5', 'loco', 'buf250', 'buf500')]; N['v2_cc_min'] = sg(min(ccs)); N['v2_cc_max'] = sg(max(ccs))
gd = J('v3_gap_day.json'); N['v_gap_day'] = sg(gd['full|day']['d']); N['v_gap_day_i'] = iv(*gd['full|day']['ci95'])

# ---------------------------------------------------------------- back-transformation (Section V-F, Table V)
cm, vr, vc, rk = J('r10_scale_main.json'), J('v_r10.json'), J('v_r10_coords.json'), J('r12_rank.json')
for sc in ('naive', 'smear', 'log'):
    Mt = pd.DataFrame([r[sc]['plain'] for r in C10]).mean(); s, lo, hi, up, _ = tint(Mt)
    srcs = {'cb500': (cm[f'spatial|{sc}']['plain']['d'], cm[f'spatial|{sc}']['plain']['up95']), 'ct5': (s, up), 'cloto': (cm[f'loto|{sc}']['plain']['d'], cm[f'loto|{sc}']['plain']['up95'])}
    for d in ('b500', 's5', 'loco_fixed', 'buf250'): x = vr[f'{d}|{sc}']['plain|day']; srcs[f'v{d}'] = (x['d'], x['up95'])
    for k, (dd, uu) in srcs.items(): N[f'bt_{k}_{sc}'] = f'{sg(dd)} ({sg(uu)})'; N[f'bt_{k}_{sc}_up'] = sg(uu)
    N[f'bc_cb500_{sc}'] = f2(rk[f'cork|spatial|coords|{sc}']); N[f'bc_cloto_{sc}'] = f2(rk[f'cork|loto|coords|{sc}'])
    N[f'bc_ct5_{sc}'] = f2(np.mean([r[sc]['coords'] for r in C10]))
    for d in ('b500', 's5', 'loco_fixed', 'buf250'): N[f'bc_v{d}_{sc}'] = f2(vc[f'{d}|{sc}'])
rd = cat('r11_rand36_*.csv').drop_duplicates('seed')
for sc in ('naive', 'smear', 'log'):
    N[f'rd_{sc}_lo'] = sg(rd[sc].min()); N[f'rd_{sc}_hi'] = sg(rd[sc].max()); N[f'twgap_{sc}'] = sg(rk[f'cork|gap_loto|g4c_notime|{sc}'])
for sc in ('smear', 'log'): N[f'fullgap_{sc}'] = sg(rk[f'cork|gap_loto|g4c|{sc}'])
from scipy.stats import spearmanr                # coordinates-only Spearman rho per design (pooled out-of-fold, log scale)
rc = [spearmanr(np.log1p(y), P4[d]['coords']).statistic for d in ('spatial', 'trace5', 'loto')]
rv = [spearmanr(np.log1p(V.target_p90.to_numpy()), np.load(RES / f'v3_preds_{d}.npy', allow_pickle=True).item()['coords']).statistic for d in ('b500', 's5', 'loco_fixed', 'buf250')]
N['rhoc_cork_min'], N['rhoc_cork_max'], N['rhoc_vie_min'], N['rhoc_vie_max'] = f2(min(rc)), f2(max(rc)), f2(min(rv)), f2(max(rv))
for k, key in (('cb500', 'cork|spatial'), ('ct5', 'cork|trace5'), ('cloto', 'cork|loto'), ('vb500', 'vienna|b500'), ('vs5', 'vienna|s5'), ('vloco_fixed', 'vienna|loco_fixed'), ('vbuf250', 'vienna|buf250')):
    x = rk[f'{key}|rho']; N[f'rho_{k}'] = sg(x['d']); N[f'rho_{k}_i'] = iv(*x['ci95'])

json.dump(N, open(RES / 'letter_numbers.json', 'w'), indent=1, ensure_ascii=False)
ref = json.load(open(RES / 'letter_numbers_reference.json'))
bad = {k: (ref[k], N.get(k)) for k in ref if N.get(k) != ref[k]}
print(f'{len(ref) - len(bad)} of {len(ref)} letter numbers reproduced')
for k, (a, b) in bad.items(): print(f'  MISMATCH {k}: letter {a}  computed {b}')
