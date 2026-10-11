"""Step 0. Build the 1-s sample table from the public UCC 4G and 5G drive-test datasets.

Inputs (download them yourself; they are not redistributed here):
  --lte  folder that contains 'Dataset/' from the UCC 4G LTE dataset  (Raca et al., MMSys 2018)
  --nr   folder that contains '5G-production-dataset/' from https://github.com/uccmisl/5Gdataset
         (unzip 5G-production-dataset.zip; Raca et al., MMSys 2020)
Output: data/ucc_prepared.csv.gz

Rules (reproduce the table used in the letter exactly):
  * 4G dataset: every CSV under Dataset/<mobility>/; keep rows with NetworkMode == 'LTE' and State == 'D' (downloading).
  * 5G dataset: the Download/ sessions only (Driving and Static); keep rows with NetworkMode == '5G' and State == 'D'.
  * RSRP outside the 3GPP range [-140, -44] dBm (e.g. the logger's -200 dBm sentinel) is treated as missing.
  * Rows that share a timestamp within a session are averaged (radio KPIs, speed, position, throughput);
    the operator is taken from the first such row. Throughput is converted from kbit/s to Mbit/s.
usage: python 00_prepare_ucc.py --lte <dir> --nr <dir>
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
COLS = {'RSRP': 'rsrp_dbm', 'RSRQ': 'rsrq_db', 'SNR': 'snr_db', 'CQI': 'cqi', 'RSSI': 'rssi_dbm', 'Speed': 'speed_kmh',
        'Longitude': 'longitude', 'Latitude': 'latitude'}

def one(path, rel, source, mode, generation, mobility):
    r = pd.read_csv(path, dtype={'Operatorname': str})
    for k in list(COLS) + ['DL_bitrate']: r[k] = pd.to_numeric(r[k], errors='coerce')
    r.loc[(r.RSRP < -140) | (r.RSRP > -44), 'RSRP'] = np.nan  # outside the 3GPP RSRP range (e.g. the -200 dBm 'no value' sentinel)
    r = r[(r.NetworkMode == mode) & (r.State == 'D')].copy()
    r['dl_mbps'] = r.DL_bitrate / 1000.0
    if r.empty: return None
    a = r.groupby('Timestamp', sort=False).agg(operator=('Operatorname', 'first'), dl_mbps=('dl_mbps', 'mean'), **{v: (k, 'mean') for k, v in COLS.items()}).reset_index()
    a['timestamp'] = pd.to_datetime(a.Timestamp, format='%Y.%m.%d_%H.%M.%S')
    a['trace_id'] = f'{source}:{rel}'; a['generation'] = generation; a['mobility'] = mobility
    return a[['trace_id', 'timestamp'] + list(COLS.values()) + ['dl_mbps', 'operator', 'mobility', 'generation']]

def main(lte, nr):
    out = []
    for f in sorted(glob.glob(os.path.join(lte, 'Dataset', '*', '*.csv'))):
        rel = os.path.relpath(f, lte).replace(os.sep, '/'); mob = rel.split('/')[1]
        x = one(f, rel, 'ucc_lte', 'LTE', '4G', mob)
        if x is not None: out.append(x)
    for f in sorted(glob.glob(os.path.join(nr, '5G-production-dataset', 'Download', '*', '*.csv'))):
        rel = os.path.relpath(f, nr).replace(os.sep, '/'); mob = rel.split('/')[2].lower()
        x = one(f, rel, 'ucc_5g', '5G', '5G', mob)
        if x is not None: out.append(x)
    d = pd.concat(out, ignore_index=True)
    (ROOT / 'data').mkdir(exist_ok=True); d.to_csv(ROOT / 'data' / 'ucc_prepared.csv.gz', index=False, compression='gzip')
    b = d[d.operator == 'B']
    print('rows', len(d), '| operator B: 4G', int((b.generation == '4G').sum()), ' 5G', int((b.generation == '5G').sum()),
          '| 5G sessions', b[b.generation == '5G'].trace_id.nunique())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--lte', required=True); ap.add_argument('--nr', required=True)
    a = ap.parse_args(); main(a.lte, a.nr)
