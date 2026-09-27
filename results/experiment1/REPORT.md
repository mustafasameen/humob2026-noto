# Experiment 1 results

Reproduction passed before creating `codex/experiment-1-baselines`: `final` on `may_jun` = **0.2029103543624607**, or **0.202910** to six decimals. The CSV score column is `combined`.

Completed **88 successful runs**: 21 final references, 21 linear baselines, six rank-selection trials, and 40 fits using the selected ranks. No model run failed. Initial setup/network failures are preserved in `../setup/initial-attempts.md`.

## Rank selection

Ranks were selected separately using only `may_jun`; every trial is retained under `tuning/`. BTMF selected **10** and TRMF selected **5**.

| Method | Rank | may_jun Combined NRMSE |
|---|---:|---:|
| BTMF | 5 | 0.223190 |
| BTMF | 10 | 0.222901 |
| BTMF | 20 | 0.228706 |
| TRMF | 5 | 0.874862 |
| TRMF | 10 | 0.887705 |
| TRMF | 20 | 0.888367 |

## Scores by window

Lower is better. All methods use the existing scorer and the same target dates.

| Window | final | Linear | BTMF (10) | TRMF (5) |
|---|---:|---:|---:|---:|
| may_jun | 0.202910 | 0.218958 | 0.222901 | 0.874862 |
| late_jan | 0.205483 | 0.224974 | 0.245631 | 0.655290 |
| april | 0.203035 | 0.227494 | 0.222580 | 0.852279 |
| roll_0429 | 0.203527 | 0.219099 | 0.232998 | 0.866523 |
| roll_0513 | 0.207422 | 0.223110 | 0.223419 | 0.881167 |
| roll_0527 | 0.209406 | 0.224228 | 0.225992 | 0.880950 |
| roll_0610 | 0.210707 | 0.225117 | 0.230004 | 0.882873 |
| roll_0624 | 0.209962 | 0.224232 | 0.230936 | 0.860475 |
| roll_0708 | 0.206443 | 0.222227 | 0.225712 | 0.864062 |
| roll_0722 | 0.204250 | 0.216633 | 0.224448 | 0.846225 |
| roll_0805 | 0.202873 | 0.216929 | 0.221136 | 0.837814 |
| roll_0819 | 0.206070 | 0.221804 | 0.221899 | 0.845523 |
| shift_0422 | 0.205014 | 0.218963 | 0.225419 | 0.874396 |
| shift_0506 | 0.210041 | 0.224290 | 0.231156 | 0.884380 |
| shift_0520 | 0.208498 | 0.223441 | 0.223573 | 0.883500 |
| shift_0603 | 0.210723 | 0.226644 | 0.231078 | 0.883932 |
| shift_0617 | 0.209395 | 0.225268 | 0.228935 | 0.878341 |
| shift_0701 | 0.208411 | 0.222752 | 0.229248 | 0.860639 |
| shift_0715 | 0.203922 | 0.219555 | 0.224707 | 0.852266 |
| shift_0729 | 0.203168 | 0.216226 | 0.223779 | 0.827501 |
| shift_0812 | 0.203164 | 0.216233 | 0.224750 | 0.836854 |

## Pooled comparisons

Differences are **baseline minus final**; positive means the baseline is worse. Pooled scores average window-date rows, and confidence intervals cluster overlapping rows by calendar date, using the original bootstrap (4,000 repetitions, seed 0).

| Pool | Method | Score | final | Difference | 95% CI |
|---|---|---:|---:|---:|---|
| pooled_all | linear | 0.221500 | 0.206512 | +0.014987 | [+0.013151, +0.016941] |
| pooled_roll | linear | 0.221461 | 0.206720 | +0.014741 | [+0.012938, +0.016594] |
| pooled_shift | linear | 0.221464 | 0.206907 | +0.014556 | [+0.012499, +0.016750] |
| pooled_all | btmf | 0.226437 | 0.206512 | +0.019925 | [+0.016970, +0.022981] |
| pooled_roll | btmf | 0.226267 | 0.206720 | +0.019547 | [+0.016566, +0.022775] |
| pooled_shift | btmf | 0.226955 | 0.206907 | +0.020048 | [+0.016969, +0.023387] |
| pooled_all | trmf | 0.862611 | 0.206512 | +0.656099 | [+0.646400, +0.665440] |
| pooled_roll | trmf | 0.862666 | 0.206720 | +0.655946 | [+0.641786, +0.669583] |
| pooled_shift | trmf | 0.864446 | 0.206907 | +0.657539 | [+0.642495, +0.672208] |

The all-window pool includes the rank-selection window and is descriptive, not a held-out estimate. Rolling and shifted pools exclude `may_jun`.

## Findings and scope

- linear: rolling-window score 0.221461; difference from final +0.014741 (95% CI [+0.012938, +0.016594]).
- btmf: rolling-window score 0.226267; difference from final +0.019547 (95% CI [+0.016566, +0.022775]).
- trmf: rolling-window score 0.862666; difference from final +0.655946 (95% CI [+0.641786, +0.669583]).

TRMF performs poorly with these fixed transdim settings on this daily completion task. This is a result for the documented configuration, not a claim about optimally tuned TRMF. No regularization, iteration count, scaling or lag tuning was performed after seeing scores.

## Protocol and verification

- Full 366-day calendar; absent dates, NA dates and targets masked; observed zero flows retained. Keys use `combine.eval_box_context`.
- Linear blends ordinary anchor means over the first-to-last target-date span, matching the submitted interpolation's default span convention, without day-type factors.
- Both factor methods use fixed daily lags [1, 2, 7] and seed 1000. The weekly lag is a declared adaptation of upstream dataset-specific seasonality, not a universal upstream default.
- BTMF uses 1,000 burn-in and 200 retained iterations. TRMF uses 200 iterations, all three lambdas 500 and eta 0.03. Initialization and other priors follow the upstream notebooks.
- Transdim is pinned to `eed7007ff49f83b12280844b209323e7383cc471`; its MIT licence is included. Mask/diagnostic adaptations are detailed in `../../src/humob26/baselines/README.md`.
- Per-window intervals use unchanged `paired_bootstrap_components`; pooled intervals use unchanged `date_cluster_bootstrap`.
- 31 tests passed, one optional integration test skipped. The actual documented data/forecast reproduction ran separately and passed.
- Synthetic equivalence checks matched both upstream algorithms bitwise where masks agree. All 88 run records passed date coverage, anchor separation and score aggregation checks.
- Existing model, evaluation, window, metric, bootstrap, data, combine and evidence modules are unchanged.

## Artifacts

- `rank_scores.csv`, `selection.json`: all rank trials and frozen selection.
- `window_scores.csv`, `comparisons.csv`, `pooled_scores.csv`: aggregate results and confidence intervals.
- Each run directory: `run.json`, `packages.txt`, `output.log`, `window.json`, daily/window score CSVs and baseline comparison CSVs.
- `../setup/`: setup commands, package versions, release metadata/checksum, source provenance and verification logs.
- Run command: `.venv/bin/python scripts/experiment1.py`; install dependencies with `.venv/bin/pip install -e ".[test,baselines]"`.
- Downloaded release: `forecasts-timesfm3/timesfm3_mlx_forecasts.tar.gz`, extracted into `forecasts/timesfm3_mlx/`. Raw data and NPZ forecast/context files are not committed.
