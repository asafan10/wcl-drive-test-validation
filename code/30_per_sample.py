"""Re-evaluation of published-style per-sample 5G throughput prediction under different validation splits.
usage: python 30_per_sample.py <ucc_prepared.csv.gz> <phone_data_5g.parquet> <out.json>
Datasets: Cork (UCC 5G, operator B, 15 driving download traces; static sessions excluded) and Vienna (operator A, NSA serving beam, 24 coverage segments, 5 days).
Model: histogram gradient boosting (as GDBT/XGBoost in the literature) on per-sample features.
Feature sets: 'radio+loc' = location + speed + same-sample radio KPIs; '+history' adds the previous sample's throughput (1-step lag).
Splits: random 70/30 (Lumos5G, Minovski), temporal within trace (last 30 % of each trace; Basit), 500 m spatial blocks (5-fold),
        leave-one-trace-out (Cork) / leave-one-day-out (Vienna).
Diagnostic: share of test samples with a training sample from the same independence unit (Cork: trace; Vienna: measurement day, noon to noon UTC, holding one or two logging runs) within 10 s, and within 50 m."""
import sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial import cKDTree
fc, fv, out = sys.argv[1:4]
def load_cork():
    d = pd.read_csv(fc); d = d[(d.generation == '5G') & (d.operator == 'B') & ~d.trace_id.str.contains('/Static/')].copy()  # driving traces only
    d['t'] = pd.to_datetime(d.timestamp); d = d.sort_values(['trace_id', 't'])
    d['trace'] = d.trace_id; d['unit'] = d.trace_id
    d = d.rename(columns={'dl_mbps': 'dl', 'rsrp_dbm': 'rsrp', 'rsrq_db': 'rsrq', 'snr_db': 'snr', 'speed_kmh': 'speed'})
    kpis = ['rsrp', 'rsrq', 'snr', 'cqi', 'rssi_dbm']; return d, kpis
def load_vienna():
    g = pd.read_parquet(fv); d = g[(g.beam_type == 'Serving beam') & g.dl_throughput_mbps.notna() & g.latitude.notna()].copy()
    d = d.sort_values('time'); d['t'] = d.time; d['trace'] = (d.t.diff().dt.total_seconds() > 600).cumsum()
    d['unit'] = (d.t - pd.Timedelta(hours=12)).dt.date.astype(str)
    d = d.rename(columns={'dl_throughput_mbps': 'dl', 'rsrp_dbm': 'rsrp', 'rsrq_db': 'rsrq', 'sinr_db': 'snr', 'rssi_dbm': 'rssi', 'pathloss_db': 'pathloss'})
    for c in ['latitude', 'longitude', 'dl', 'rsrp', 'rsrq', 'snr', 'rssi', 'pathloss']: d[c] = d[c].astype(float)
    kx = 111320 * np.cos(np.radians(48.19)); x = d.longitude * kx; y = d.latitude * 110540
    dt = d.t.diff().dt.total_seconds(); dist = np.sqrt(x.diff() ** 2 + y.diff() ** 2); sp = (dist / dt * 3.6).where((dt > 0) & (dt < 10))
    d['speed'] = sp.where(d.trace == d.trace.shift()).clip(upper=200)
    return d, ['rsrp', 'rsrq', 'snr', 'rssi', 'pathloss']
res = {}
for name, loader, lat0 in [('cork', load_cork, 51.9), ('vienna', load_vienna, 48.19)]:
    d, kpis = loader(); d = d.dropna(subset=['dl', 'latitude', 'longitude']).reset_index(drop=True)
    kx = 111320 * np.cos(np.radians(lat0)); d['x'] = d.longitude * kx; d['y'] = d.latitude * 110540
    d['lag_dl'] = d.groupby('trace').dl.shift(1)
    d['ts'] = (d.t - d.t.min()).dt.total_seconds()
    d['blk'] = (d.x // 500).astype(int).astype(str) + '_' + (d.y // 500).astype(int).astype(str)
    y = d.dl.to_numpy(); TR = d.trace.astype(str).to_numpy(); n = len(d)
    rng = np.random.default_rng(0); perm = rng.permutation(n); cut = int(.7 * n)
    pos = d.groupby('trace').cumcount(); size = d.groupby('trace').dl.transform('size')
    tmask = (pos >= .7 * size).to_numpy()
    designs = {'random': [(perm[:cut], perm[cut:])],
               'temporal_within_trace': [(np.flatnonzero(~tmask), np.flatnonzero(tmask))],
               'spatial_500m': list(GroupKFold(5).split(d, y, d.blk)),
               'trace_or_day_out': [(np.flatnonzero(d.unit.astype(str).to_numpy() != u), np.flatnonzero(d.unit.astype(str).to_numpy() == u)) for u in d.unit.astype(str).unique()]}
    fsets = {'radio+loc': ['x', 'y', 'speed'] + kpis, 'radio_only': ['speed'] + kpis, 'loc_only': ['x', 'y'], '+history': ['x', 'y', 'speed'] + kpis + ['lag_dl']}
    R = {'n_samples': int(n), 'n_traces': int(len(np.unique(TR))), 'n_units': int(d.unit.nunique())}
    for dn, folds in designs.items():
        # diagnostic: same-trace training sample within 10 s / 50 m
        near_t = np.zeros(n, bool); near_s = np.zeros(n, bool); covered = np.zeros(n, bool)
        for tr, te in folds:
            covered[te] = True
            UN = d.unit.astype(str).to_numpy(); trs = pd.DataFrame({'trace': UN[tr], 'ts': d.ts.to_numpy()[tr], 'x': d.x.to_numpy()[tr], 'y': d.y.to_numpy()[tr]})
            for trc, grp in pd.DataFrame({'trace': UN[te], 'i': te}).groupby('trace'):
                g2 = trs[trs.trace == trc]
                if len(g2) == 0: continue
                tt = np.sort(g2.ts.to_numpy()); q = d.ts.to_numpy()[grp.i.to_numpy()]
                k = np.searchsorted(tt, q); lo = np.abs(q - tt[np.clip(k - 1, 0, len(tt) - 1)]); hi = np.abs(tt[np.clip(k, 0, len(tt) - 1)] - q)
                near_t[grp.i.to_numpy()] = np.minimum(lo, hi) <= 10
                dd, _ = cKDTree(g2[['x', 'y']].to_numpy()).query(d[['x', 'y']].to_numpy()[grp.i.to_numpy()]); near_s[grp.i.to_numpy()] = dd <= 50
        UNa = d.unit.astype(str).to_numpy(); tot = pd.Series(UNa).value_counts(); own = np.zeros(n)
        for tr, te in folds:
            c = pd.Series(UNa[tr]).value_counts(); own[te] = [c.get(u, 0) / tot[u] for u in UNa[te]]
        row = {'own_unit': float(own[covered].mean()), 'adj_10s': float(near_t[covered].mean()), 'adj_50m': float(near_s[covered].mean())}
        for fs, cols in fsets.items():
            p = np.full(n, np.nan)
            for tr, te in folds:
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr]); p[te] = m.predict(d.iloc[te][cols])
            ok = ~np.isnan(p); row[f'R2|{fs}'] = float(r2_score(y[ok], p[ok])); row[f'MAE|{fs}'] = float(mean_absolute_error(y[ok], p[ok]))
            if dn == 'trace_or_day_out' and name == 'vienna':
                U = d.unit.astype(str).to_numpy(); row[f'per_unit|{fs}'] = {u: float(r2_score(y[U == u], p[U == u])) for u in np.unique(U)}
        R[dn] = row; print(name, dn, {k: (round(v, 3) if isinstance(v, float) else {u: round(w, 3) for u, w in v.items()}) for k, v in row.items()}, flush=True)
    res[name] = R
json.dump(res, open(out, 'w'), indent=1)
