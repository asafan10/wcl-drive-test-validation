"""Per-sample re-evaluation, additions (Table II): random-split seeds, unit-level bootstrap intervals, persistence baseline.
Re-evaluation of published-style per-sample 5G throughput prediction under different validation splits.
usage: python 31_per_sample_seeds.py <ucc_prepared.csv.gz> <phone_data_5g.parquet> <out.json>
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
from sklearn.metrics import r2_score as _r2
for name, loader, lat0 in [('cork', load_cork, 51.9), ('vienna', load_vienna, 48.19)]:
    d, kpis = loader(); d = d.dropna(subset=['dl', 'latitude', 'longitude']).reset_index(drop=True)
    kx = 111320 * np.cos(np.radians(lat0)); d['x'] = d.longitude * kx; d['y'] = d.latitude * 110540
    d['lag_dl'] = d.groupby('trace').dl.shift(1); y = d.dl.to_numpy(); n = len(d); U = d.unit.astype(str).to_numpy()
    fsets = {'radio+loc': ['x', 'y', 'speed'] + kpis, 'radio_only': ['speed'] + kpis}
    R = {}
    # (i) random 70/30 over 10 seeds
    for fs, cols in fsets.items():
        v = []
        for seed in range(10):
            perm = np.random.default_rng(seed).permutation(n); cut = int(.7 * n); tr, te = perm[:cut], perm[cut:]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr]); v.append(float(_r2(y[te], m.predict(d.iloc[te][cols]))))
        R[f'random_seeds|{fs}'] = {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
    # (ii) unit-out per-unit R2 range and unit-level bootstrap of pooled R2
    for fs, cols in fsets.items():
        p = np.full(n, np.nan)
        for u in np.unique(U):
            tr, te = np.flatnonzero(U != u), np.flatnonzero(U == u)
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr]); p[te] = m.predict(d.iloc[te][cols])
        per = [float(_r2(y[U == u], p[U == u])) for u in np.unique(U)]
        ug = np.unique(U); gi = {g: np.flatnonzero(U == g) for g in ug}; rng = np.random.default_rng(5)
        bs = [float(_r2(y[i], p[i])) for i in (np.concatenate([gi[g] for g in rng.choice(ug, len(ug))]) for _ in range(1000))]
        R[f'unit_out|{fs}'] = {'R2': float(_r2(y, p)), 'per_unit_min': min(per), 'per_unit_max': max(per), 'boot95': np.percentile(bs, [2.5, 97.5]).tolist()}
    # (iii) persistence baseline for one-step forecasting: previous sample's throughput
    ok = d.lag_dl.notna().to_numpy(); R['persistence_R2'] = float(_r2(y[ok], d.lag_dl.to_numpy()[ok])); R['persistence_MAE'] = float(np.mean(np.abs(y[ok] - d.lag_dl.to_numpy()[ok])))
    R['n_units'] = int(len(np.unique(U))); res[name] = R; print(name, json.dumps(R), flush=True)
json.dump(res, open(out, 'w'), indent=1)
