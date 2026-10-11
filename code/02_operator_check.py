"""Operator check: (1) MCC-MNC codes leaked in Operatorname per operator label; (2) 5G anchor eNB locations
(median position of the 10 strongest-RSRP samples per NODEHEX, eNBs with >=200 samples) vs 4G serving-cell
site locations of operators A and B; A is subsampled to B's site count (200 draws) as a density-matched null.
usage: python 02_operator_check.py <UCC 4G Dataset dir> <UCC 5G 5G-production-dataset dir>
Prints the values quoted in Section III-A (network code; anchor-to-site medians 263 m vs 302-618 m)."""
import sys, glob
import numpy as np, pandas as pd
d4, d5 = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_csv(f, dtype=str).assign(file=f) for f in glob.glob(f'{d4}/*/*.csv')], ignore_index=True)
d['op'] = d.file.str.extract(r'/([AB])_', expand=False)
codes = d[d.Operatorname.str.match(r'^\d{4,}$', na=False)]; print(pd.crosstab(codes.Operatorname, codes.op))
g5 = pd.concat([pd.read_csv(f, usecols=['Longitude', 'Latitude', 'RSRP', 'NODEHEX', 'Operatorname'], dtype=str) for f in glob.glob(f'{d5}/**/*.csv', recursive=True) if '__MACOSX' not in f], ignore_index=True)
print('5G Operatorname:', g5.Operatorname.value_counts().to_dict())
for c in ['Longitude', 'Latitude', 'RSRP']: g5[c] = pd.to_numeric(g5[c], errors='coerce')
g5 = g5.dropna(subset=['Longitude', 'Latitude', 'RSRP']); g5 = g5[~g5.NODEHEX.isin(['-', '0'])]
cnt = g5.NODEHEX.value_counts(); g5 = g5[g5.NODEHEX.isin(cnt[cnt >= 200].index)]
est = g5.sort_values('RSRP', ascending=False).groupby('NODEHEX').head(10).groupby('NODEHEX')[['Longitude', 'Latitude']].median().to_numpy()
for c in ['ServingCell_Lon', 'ServingCell_Lat']: d[c] = pd.to_numeric(d[c], errors='coerce')
kx = 111320 * np.cos(np.radians(51.9)); ky = 110540
S = {op: d[d.op == op].dropna(subset=['ServingCell_Lon', 'ServingCell_Lat'])[['ServingCell_Lon', 'ServingCell_Lat']].round(5).drop_duplicates().to_numpy() for op in 'AB'}
dist = lambda s: np.sqrt(((est[:, None, 0] - s[None, :, 0]) * kx) ** 2 + ((est[:, None, 1] - s[None, :, 1]) * ky) ** 2).min(1)
for op in 'AB': m = dist(S[op]); print(op, 'sites', len(S[op]), 'eNBs', len(m), 'median %.0f m' % np.median(m))
rng = np.random.default_rng(0); meds = [np.median(dist(S['A'][rng.choice(len(S['A']), len(S['B']), replace=False)])) for _ in range(200)]
print('A subsampled to B site count: median %.0f-%.0f m' % (min(meds), max(meds)))
