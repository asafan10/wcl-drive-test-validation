"""Vienna 4G/5G drive-test replication: targets and features.
usage: python 20_vienna_targets.py <phone_data_5g.parquet> <phone_data_lte.parquet> <out.csv.gz>
- 5G: operator A, serving-beam rows with DL throughput; sessions split at gaps > 10 min; measurement day (camp) = date of (time - 12 h), i.e. noon to noon UTC
- target: p90 of 5G DL per 100 m tile x 15-min interval x session, >= 5 samples
- 4G features: operator-A LTE samples from OTHER measurement days only (legacy-archive emulation), within 100/250/500 m:
  plain: counts, median/p90 of DL, RSRP, RSRQ, SINR, RSSI; p10 of DL and RSRP; zero-DL and weak-RSRP shares; nearest distance
  time-window: counts and median/p90 of DL, RSRP, RSRQ, SINR from samples in the target's 3-hour time-of-day window (UTC)"""
import sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
f5, fl, out = sys.argv[1:4]
LAT0 = 48.19; KX = 111320 * np.cos(np.radians(LAT0)); KY = 110540
g = pd.read_parquet(f5); s = g[(g.beam_type == 'Serving beam') & g.dl_throughput_mbps.notna() & g.latitude.notna() & g.longitude.notna()].copy()
s = s.sort_values('time').reset_index(drop=True); s['sess'] = (s.time.diff().dt.total_seconds() > 600).cumsum()
s['camp'] = (s.time - pd.Timedelta(hours=12)).dt.date.astype(str)
s['x'] = s.longitude.astype(float) * KX; s['y'] = s.latitude.astype(float) * KY
s['tile'] = (s.x // 100).astype(int).astype(str) + '_' + (s.y // 100).astype(int).astype(str); s['t15'] = s.time.dt.floor('15min')
T = s.groupby(['tile', 't15', 'sess']).agg(n=('dl_throughput_mbps', 'size'), target_p90=('dl_throughput_mbps', lambda v: float(np.quantile(v.astype(float), .9))),
                                          x_m=('x', 'median'), y_m=('y', 'median'), camp=('camp', 'first')).reset_index()
T = T[T.n >= 5].reset_index(drop=True)
T['hour'] = T.t15.dt.hour + T.t15.dt.minute / 60; T['hour3'] = (T.t15.dt.hour // 3).astype(int); T['dow'] = T.t15.dt.dayofweek
T['hour_sin'] = np.sin(2 * np.pi * T.hour / 24); T['hour_cos'] = np.cos(2 * np.pi * T.hour / 24); T['dow_sin'] = np.sin(2 * np.pi * T.dow / 7); T['dow_cos'] = np.cos(2 * np.pi * T.dow / 7)
T['group500'] = (T.x_m // 500).astype(int).astype(str) + '_' + (T.y_m // 500).astype(int).astype(str)
l = pd.read_parquet(fl); l = l[(l.operator == 'A') & l.latitude.notna() & l.longitude.notna()].copy()
l['camp'] = (l.time - pd.Timedelta(hours=12)).dt.date.astype(str); l['x'] = l.longitude.astype(float) * KX; l['y'] = l.latitude.astype(float) * KY
l['hour3'] = (l.time.dt.hour // 3).astype(int); l['lsess'] = (l.sort_values('time').time.diff().dt.total_seconds() > 600).cumsum()
for c in ['dl_throughput_mbps', 'rsrp_dbm', 'rsrq_db', 'sinr_db', 'rssi_dbm']: l[c] = l[c].astype(float)
M = {'dl': 'dl_throughput_mbps', 'rsrp': 'rsrp_dbm', 'rsrq': 'rsrq_db', 'sinr': 'sinr_db', 'rssi': 'rssi_dbm'}
rows = []
for camp, Tc in T.groupby('camp'):
    L = l[l.camp != camp].reset_index(drop=True); tree = cKDTree(L[['x', 'y']].to_numpy())
    XY = Tc[['x_m', 'y_m']].to_numpy(); dn, _ = tree.query(XY, k=1)
    for r in (100, 250, 500):
        nb = tree.query_ball_point(XY, r)
        for i, (idx, h3) in enumerate(zip(nb, Tc.hour3)):
            q = L.iloc[idx]; f = rows[len(rows) - len(Tc) + i] if r > 100 else {'_i': Tc.index[i], 'nearest_4g_m': float(dn[i])}
            p = f'r{r}_'; f[p + 'obs_count'] = len(q); f[p + 'trace_count'] = q.lsess.nunique() if len(q) else 0
            for k, c in M.items():
                v = q[c].dropna(); f[p + k + '_median'] = v.median() if len(v) else np.nan; f[p + k + '_p90'] = v.quantile(.9) if len(v) else np.nan
            for k in ('dl', 'rsrp'):
                v = q[M[k]].dropna(); f[p + k + '_p10'] = v.quantile(.1) if len(v) else np.nan
            v = q.dl_throughput_mbps.dropna(); f[p + 'zero_dl_share'] = float((v < .1).mean()) if len(v) else np.nan
            v = q.rsrp_dbm.dropna(); f[p + 'weak_rsrp_share'] = float((v < -110).mean()) if len(v) else np.nan
            w = q[q.hour3 == h3]; p2 = p + 'sameh3_'; f[p2 + 'obs_count'] = len(w); f[p2 + 'trace_count'] = w.lsess.nunique() if len(w) else 0
            for k in ('dl', 'rsrp', 'rsrq', 'sinr'):
                v = w[M[k]].dropna(); f[p2 + k + '_median'] = v.median() if len(v) else np.nan; f[p2 + k + '_p90'] = v.quantile(.9) if len(v) else np.nan
            if r == 100: rows.append(f)
F = pd.DataFrame(rows).set_index('_i').sort_index(); D = T.join(F)
D.to_csv(out, index=False, compression='gzip')
plain = [c for c in F.columns if 'sameh3' not in c]; tw = [c for c in F.columns if 'sameh3' in c]
print('targets', len(D), 'tiles', D.tile.nunique(), 'sessions', D.sess.nunique(), 'campaigns', D.camp.nunique())
print('targets per session', D.sess.value_counts().sort_index().to_dict())
print('features plain', len(plain), 'time-window', len(tw), 'missing share', round(float(D[list(F.columns)].isna().mean().mean()), 3))
print('share of targets with other-campaign LTE within 100 m', round(float((D.r100_obs_count > 0).mean()), 3), 'median nearest 4G m', round(float(D.nearest_4g_m.median()), 1))
print('target p90 Mbps', D.target_p90.describe().round(1).to_dict())
