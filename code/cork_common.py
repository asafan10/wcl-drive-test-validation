"""Shared Cork data and helpers (imported by the Cork analysis scripts).
dt: targets with 97 4G features (data/cork_features.csv.gz); y: p90 5G throughput (Mbit/s); W: sqrt sample-count weights;
legacy: the 97 4G feature names; coords: target coordinates; TRACE: 5G trace of each target; LOTO: leave-one-trace-out folds."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
ROOT = Path(__file__).resolve().parent.parent
_dt = pd.read_csv(ROOT / 'data' / 'cork_features.csv.gz')
times = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']; coords = ['latitude', 'longitude']
skip = {'tile100', 't15', 'n5', 'latitude', 'longitude', 'x_m', 'y_m', 'target_p90', 'hour', 'hour3', 'dow', 'group500'} | set(times)
legacy = [c for c in _dt.columns if c not in skip]
y = _dt.target_p90.to_numpy(); W = np.sqrt(_dt.n5.to_numpy())
dt = _dt.copy(); dt['t15'] = pd.to_datetime(dt.t15)
tm = pd.read_csv(ROOT / 'data' / 'cork_target_trace.csv', parse_dates=['t15'])
TRACE = dt.merge(tm, on=['tile100', 't15'], how='left').trace.to_numpy()
LOTO = list(LeaveOneGroupOut().split(dt, y, TRACE))

def et(leaf, mf, n, seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('m', ExtraTreesRegressor(n_estimators=n, min_samples_leaf=leaf, max_features=mf, random_state=seed, n_jobs=1))])

def fit(m, X, yy, w):
    if isinstance(m, Pipeline): m.fit(X, yy, m__sample_weight=w)
    else: m.fit(X, yy, sample_weight=w)
    return m

