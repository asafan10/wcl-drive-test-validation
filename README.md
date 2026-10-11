# Validation design in drive-test 5G throughput prediction — code and results

Code and results for the letter

> A. Hafez, [Co-author], and A. Saafan, "Validation Design in Drive-Test 5G Throughput Prediction: Inflated Accuracy and
> No Reliable Evidence of Gain From Legacy 4G Data," submitted to *IEEE Wireless Communications Letters*, 2026.

Every number, table and figure in the letter can be regenerated from the public datasets with the scripts below.
`code/40_make_numbers.py` recomputes all 277 numbers quoted in the letter from `results/` and checks them against
`results/letter_numbers_reference.json` (the printed values); with the shipped results it reports
`277 of 277 letter numbers reproduced`.

## Data (public, not redistributed here)

| Dataset | Source | Used for |
|---|---|---|
| UCC 4G LTE drive tests (Raca et al., ACM MMSys 2018) | doi: 10.5281/zenodo.1219679 | Cork 4G archive |
| UCC 5G production dataset (Raca et al., ACM MMSys 2020) | https://github.com/uccmisl/5Gdataset (unzip `5G-production-dataset.zip`); doi: 10.5281/zenodo.3751194 | Cork 5G targets |
| Vienna 4G/5G drive-test dataset (Wiedner et al., 2026; CC BY 4.0) | doi: 10.5281/zenodo.18322065 (`phone_data_5g.parquet`, `phone_data_lte.parquet`) | Vienna replication |

Place them, for example, as `raw/lte/Dataset/...`, `raw/nr/5G-production-dataset/...` and `raw/vienna/*.parquet`.

## Environment

Python 3.13 with the versions in `requirements.txt` (`pip install -r requirements.txt`). Use **scikit-learn 1.8**:
`GroupKFold` assigns groups to folds differently in 1.9, which changes the spatial folds and therefore the numbers.
All randomness is seeded. Everything runs on a CPU.

## Run order

`run_all.sh` runs everything in order (edit the three data paths at its top). Steps:

| Step | Script | Output | Letter |
|---|---|---|---|
| 0 | `00_prepare_ucc.py --lte <dir> --nr <dir>` | `data/ucc_prepared.csv.gz` (1-s samples) | — |
| 1 | `01_cork_features.py` | `data/cork_features.csv.gz`, `data/cork_target_trace.csv` (746 targets, 97 4G features) | III |
| 2 | `02_operator_check.py`, `03_cork_data_stats.py` | printed | III-A |
| 3 | `10_cork_designs.py main / blocks / area / refit 0 100 / rand36 0 30` | `r4_*` | Table III, V-B, V-C, Fig. 1 |
| 4 | `11_cork_4g_checks.py only4g / misc / parts 0 100` | `r5_*` | Table III, V-B, V-C, V-D |
| 5 | `12_cork_scales.py main / parts 0 100 / rand36 0 30` | `r10_*`, `r11_*` | Table V, V-C |
| 6 | `30_per_sample.py`, `31_per_sample_seeds.py` | `reval.json`, `reval2.json` | Table II |
| 7 | `20_vienna_targets.py` | `data/vienna_targets.csv.gz` | V-E |
| 8 | `21_vienna_baselines.py`, `22_vienna_designs.py` | `v2_*`, `v3_*` | Table IV, V-E |
| 9 | `23_vienna_gap_days.py`, `24_vienna_scales.py` | `v3_gap_day.json`, `v_r10*.json` | V-E, Table V |
| 10 | `13_rank_and_gaps.py` | `r12_rank.json` | Table V, V-C |
| 11 | `40_make_numbers.py`, `41_figure1.py` | `letter_numbers.json`, `fig1.png/pdf` | all |

The long steps are the 100 random trace partitions (steps 4 and 5) and the 100 trace-resampled refits (step 3);
each takes several hours on two cores and can be split into seed ranges run in parallel (e.g. `parts 0 50` and
`parts 50 100`). `results/` already contains all model outputs. To check the letter's numbers without re-running the
models, run only the quick data steps 0, 1 and 7 (a few minutes; they rebuild `data/`, which is not redistributed
because it is derived from the third-party datasets), then step 11.

## Methods in brief

* **Targets.** 90th percentile of 5G downlink throughput per 100 m tile and 15-minute interval with at least five 1-s
  samples (Cork: 746 targets, 20 traces; Vienna: 1,805 targets, 24 segments on 5 measurement days).
* **Models.** Extra-Trees on log(1 + p90) with sqrt(sample count) weights; coordinates only, coordinates + all 97 4G
  features, coordinates + 61 plain 4G features, and a lean set; regression kriging, kNN-IDW and a GP as references.
  Per-sample predictors (Table II) use histogram gradient boosting on throughput in Mbit/s.
* **Designs.** 500 m / 1 km / 2 km spatial blocks, 5 folds of whole traces (100 random partitions), leave-one-trace-out,
  held-out areas (250 / 500 / 1000 m buffers); Vienna: blocks, segment folds, leave-one-day-out with 4G features
  rebuilt per fold, 250 m buffer.
* **Inference.** R² differences are sums of per-cluster contributions with t(G − 1) intervals (Cork: traces;
  Vienna: days); one-sided 95 % bounds; Poisson-weighted trace refits; cluster bootstrap for Spearman changes.

## Licences

Code: MIT (see `LICENSE`). Result files in `results/`: CC BY 4.0. The datasets keep their own licences.

## How to cite

Please cite the letter above and this archive ([DOI assigned by Zenodo]).
