#!/usr/bin/env bash
# Runs the full pipeline. Edit the three data paths first (see README.md).
set -euo pipefail
LTE=raw/lte                       # folder that contains Dataset/ (UCC 4G)
NR=raw/nr                         # folder that contains 5G-production-dataset/ (UCC 5G)
VIE=raw/vienna                    # folder with phone_data_5g.parquet and phone_data_lte.parquet
PY=${PY:-python}
cd "$(dirname "$0")"
$PY code/00_prepare_ucc.py --lte $LTE --nr $NR
$PY code/01_cork_features.py
$PY code/02_operator_check.py $LTE/Dataset $NR/5G-production-dataset
$PY code/03_cork_data_stats.py
# Cork (the 'parts', 'refit' and 'rand36' runs are long; split their seed ranges to run in parallel)
$PY code/10_cork_designs.py main
$PY code/10_cork_designs.py blocks
$PY code/10_cork_designs.py area
$PY code/10_cork_designs.py refit 0 100
$PY code/10_cork_designs.py rand36 0 30
$PY code/11_cork_4g_checks.py only4g
$PY code/11_cork_4g_checks.py misc
$PY code/11_cork_4g_checks.py parts 0 100
$PY code/12_cork_scales.py main
$PY code/12_cork_scales.py parts 0 100
$PY code/12_cork_scales.py rand36 0 30
# per-sample re-evaluation (Table II)
$PY code/30_per_sample.py data/ucc_prepared.csv.gz $VIE/phone_data_5g.parquet results/reval.json
$PY code/31_per_sample_seeds.py data/ucc_prepared.csv.gz $VIE/phone_data_5g.parquet results/reval2.json
# Vienna
$PY code/20_vienna_targets.py $VIE/phone_data_5g.parquet $VIE/phone_data_lte.parquet data/vienna_targets.csv.gz
$PY code/21_vienna_baselines.py data/vienna_targets.csv.gz $VIE/phone_data_lte.parquet results
$PY code/22_vienna_designs.py data/vienna_targets.csv.gz $VIE/phone_data_lte.parquet results
$PY code/23_vienna_gap_days.py data/vienna_targets.csv.gz results results/v3_gap_day.json
$PY code/24_vienna_scales.py data/vienna_targets.csv.gz results results/v_r10.json
# both cities, numbers and figure
$PY code/13_rank_and_gaps.py
$PY code/40_make_numbers.py
$PY code/41_figure1.py
