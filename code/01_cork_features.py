"""Step 1. Cork (UCC, operator B): 5G targets per 100 m tile and 15-minute interval, and 97 features from the 4G drive tests.

Target: 90th percentile of 5G downlink throughput of a (tile, 15-min) cell with at least five 1-s samples; each target is
labelled with its 5G trace (the trace that contributes most samples).
Features (from 4G samples within 100, 250 and 500 m of the target): counts, medians, percentiles and shares of
throughput, RSRP, RSRQ, SNR, CQI, RSSI and speed, plus the distance to the nearest 4G sample (61 plain features), and
the same statistics restricted to 4G samples in the target's three-hour time-of-day window (36 time-window features).
Input: data/ucc_prepared.csv.gz (step 0).  Output: data/cork_features.csv.gz, data/cork_target_trace.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / 'data' / 'ucc_prepared.csv.gz', parse_dates=['timestamp'], dtype={'operator': str})
df = df[(df.operator == 'B') & (df.generation.isin(['4G', '5G']))].copy()
df = df[df[['latitude', 'longitude', 'dl_mbps']].notna().all(axis=1)]
lat_ref = float(df.latitude.median()); df['x_m'] = df.longitude * 111320 * np.cos(np.deg2rad(lat_ref)); df['y_m'] = df.latitude * 110540
lte = df[df.generation == '4G'].reset_index(drop=True); nr = df[df.generation == '5G'].reset_index(drop=True)
nr['tile100'] = np.floor(nr.x_m / 100).astype(int).astype(str) + '_' + np.floor(nr.y_m / 100).astype(int).astype(str)
nr['t15'] = nr.timestamp.dt.floor('15min')
tar = nr.groupby(['tile100', 't15'], sort=True).agg(n5=('dl_mbps', 'size'), latitude=('latitude', 'median'), longitude=('longitude', 'median'),
                                                     x_m=('x_m', 'median'), y_m=('y_m', 'median'),
                                                     target_p90=('dl_mbps', lambda s: float(s.quantile(.9)))).reset_index()
tar = tar[tar.n5 >= 5].reset_index(drop=True)
tar['hour'] = tar.t15.dt.hour; tar['hour3'] = 3 * (tar.hour // 3); tar['dow'] = tar.t15.dt.dayofweek
for unit, period in [('hour', 24), ('dow', 7)]:
    tar[unit + '_sin'] = np.sin(2 * np.pi * tar[unit] / period); tar[unit + '_cos'] = np.cos(2 * np.pi * tar[unit] / period)
tar['group500'] = np.floor(tar.x_m / 500).astype(int).astype(str) + '_' + np.floor(tar.y_m / 500).astype(int).astype(str)
tree = cKDTree(lte[['x_m', 'y_m']].to_numpy(float)); h3 = 3 * (lte.timestamp.dt.hour // 3)
metrics = ['dl_mbps', 'rsrp_dbm', 'rsrq_db', 'snr_db', 'cqi', 'rssi_dbm', 'speed_kmh']
rows = []
for r in tar.itertuples():
    vals = {}; dst, _ = tree.query([r.x_m, r.y_m]); vals['nearest_4g_measurement_m'] = float(dst)
    for rad in [100, 250, 500]:
        ids = tree.query_ball_point([r.x_m, r.y_m], rad); sub = lte.iloc[ids]
        vals[f'r{rad}_obs_count'] = len(sub); vals[f'r{rad}_trace_count'] = sub.trace_id.nunique()
        for m in metrics:
            s = sub[m].dropna(); vals[f'r{rad}_{m}_median'] = float(s.median()) if len(s) else np.nan; vals[f'r{rad}_{m}_p90'] = float(s.quantile(.9)) if len(s) else np.nan
            if m in ('dl_mbps', 'rsrp_dbm'): vals[f'r{rad}_{m}_p10'] = float(s.quantile(.1)) if len(s) else np.nan
        vals[f'r{rad}_zero_dl_share'] = float((sub.dl_mbps < 1).mean()) if len(sub) else np.nan
        vals[f'r{rad}_weak_rsrp_share'] = float((sub.rsrp_dbm < -110).mean()) if len(sub) else np.nan
        same = sub.iloc[np.where(h3.iloc[ids].to_numpy() == r.hour3)[0]] if ids else sub
        vals[f'r{rad}_sameh3_obs_count'] = len(same); vals[f'r{rad}_sameh3_trace_count'] = same.trace_id.nunique()
        for m in ['dl_mbps', 'rsrp_dbm', 'rsrq_db', 'snr_db', 'cqi']:
            s = same[m].dropna(); vals[f'r{rad}_sameh3_{m}_median'] = float(s.median()) if len(s) else np.nan; vals[f'r{rad}_sameh3_{m}_p90'] = float(s.quantile(.9)) if len(s) else np.nan
    rows.append(vals)
dt = pd.concat([tar, pd.DataFrame(rows)], axis=1)
dt.to_csv(ROOT / 'data' / 'cork_features.csv.gz', index=False, compression='gzip')
tr = nr.groupby(['tile100', 't15']).trace_id.agg(lambda s: s.mode().iloc[0]).rename('trace').reset_index()
m = dt[['tile100', 't15']].merge(tr, on=['tile100', 't15'], how='left'); assert m.trace.notna().all()
m.to_csv(ROOT / 'data' / 'cork_target_trace.csv', index=False)
print('targets', len(dt), '| tiles', dt.tile100.nunique(), '| traces', m.trace.nunique(), '| 4G features', dt.shape[1] - tar.shape[1])
