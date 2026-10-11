"""Prints the Cork data statistics quoted in Section III-A of the letter (sample counts, traces, targets, missingness,
LTE-anchor RSRP and SNR medians).  usage: python 03_cork_data_stats.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent; sys.path.insert(0, str(R)); import cork_common as st
d = pd.read_csv(R.parent / 'data' / 'ucc_prepared.csv.gz', parse_dates=['timestamp'], dtype={'operator': str}); b = d[d.operator == 'B']
g5, g4 = b[b.generation == '5G'], b[b.generation == '4G']
print('5G samples', len(g5), '| traces', g5.trace_id.nunique(), '(driving', g5[g5.mobility == 'driving'].trace_id.nunique(), ', static', g5[g5.mobility == 'static'].trace_id.nunique(), ')',
      '| years', sorted(g5.timestamp.dt.year.unique()))
print('4G samples', len(g4), '| years', sorted(g4.timestamp.dt.year.unique()))
print('median RSRP 5G-corpus LTE anchor %.0f dBm vs 4G set %.0f dBm; median SNR %.0f vs %.0f dB' % (g5.rsrp_dbm.median(), g4.rsrp_dbm.median(), g5.snr_db.median(), g4.snr_db.median()))
dt = st.dt; per = pd.Series(st.TRACE).value_counts()
print('targets', len(dt), '| tiles', dt.tile100.nunique(), '| targets per trace %d-%d' % (per.min(), per.max()), '| samples per target %d-%d, median %.0f' % (dt.n5.min(), dt.n5.max(), dt.n5.median()))
print('missing 4G feature entries %.0f %%' % (100 * dt[st.legacy].isna().mean().mean()))
