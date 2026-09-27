# Experiment 1

This package adds independent baselines. The submitted model, evaluation CLI,
validation windows, metrics and bootstrap implementations are unchanged.

## Upstream source and licence

BTMF and TRMF are adapted from `imputer/BTMF.ipynb` and `imputer/TRMF.ipynb`
in [xinychen/transdim](https://github.com/xinychen/transdim/tree/eed7007ff49f83b12280844b209323e7383cc471),
commit `eed7007ff49f83b12280844b209323e7383cc471`.
The MIT licence and Xinyu Chen's copyright notice are retained in
`TRANSDIM_LICENSE`. These are the transdim implementations, including their
original numerical update order, rather than newly derived algorithms.

Adaptations:

- Extract algorithm cells into importable Python; omit notebook datasets,
  plotting and IPython magic.
- Remove dense hidden-ground-truth arguments and diagnostic MAPE/RMSE. Training
  never receives hidden target values. Progress reports iteration counts only.
- BTMF always uses the upstream NaN-mask path; TRMF uses a NaN observation mask
  instead of `value != 0`, preserving observed zero flows.
- Fix the unused large-matrix BTMF fallback's reference to a notebook-global
  `sparse_mat`, using the passed precision-weighted data and mask. Our 1,395-pair
  matrices and ranks <=20 use the upstream vectorized branch, not this fallback.
- No changes to factor updates, priors, regularization or averaging.

## Fixed settings

Rank candidates are 5, 10 and 20, selected separately per method using only
`may_jun` Combined NRMSE. All six trials are retained; ties choose the smaller
rank. The selected rank is frozen before any other window is fitted.

The upstream notebooks use dataset-specific seasonal lags, not one universal
lag list. For daily HuMob data we fix `[1, 2, 7]` days (weekly seasonality),
without tuning it. This adaptation was declared before rank selection.

- BTMF: `option="factor"`, 1,000 burn-in iterations, 200 posterior samples;
  `W`/`X` initialized with `0.01 * randn`, following the non-random missing
  examples. Upstream prior hyperparameters remain unchanged.
- TRMF: 200 iterations; `lambda_w=lambda_x=lambda_theta=500`, `eta=0.03`;
  `W`/`X`/`theta` initialized with `0.1 * rand`, as upstream.
- Seed 1000 for every independent fit, as in upstream examples. No scaling or
  data-dependent hyperparameter choices. Negative predictions are clipped at 0.
- Linear: ordinary per-pair anchor means, observed absent pairs counted as zero,
  NA days excluded. Linear blend from before mean on the first target date to
  after mean on the last target date. Uses actual calendar offsets and the
  submitted interpolation's default target-span convention, no day-type factors.

The factorization input is pairs-by-days on the full 366-day calendar,
2023-11-01 through 2024-10-31. Keys come from `combine.eval_box_context`; missing
calendar dates, NA dates and hidden targets are NaN. Observed zeros stay zero.

## Run and output

```bash
.venv/bin/pip install -e '.[test,baselines]'
.venv/bin/python scripts/experiment1.py
```

Prerequisite: the documented reproduction in `results/check/window_scores.csv`
must give `final/may_jun = 0.202910` to six decimal places. This was verified
before creating the experiment branch; see `results/setup/`.

The script runs references, linear interpolation, all six tuning trials, then
the chosen ranks on the remaining 20 windows. At most two CPU subprocesses run
at once with one BLAS thread each. It reuses successful runs but refuses to
overwrite failed or incomplete attempts. To run a separate attempt directly:

```bash
.venv/bin/python -m humob26.baselines.run run --method btmf --rank 5 \
  --window may_jun --out results/another-attempt
```

Each run has `run.json` (exact command, package versions, source and input
hashes, settings, status, timing), `packages.txt`, `output.log`, `window.json`,
`daily_scores.csv`, `window_scores.csv` and `comparisons.csv`. Failures retain
tracebacks and `failure.csv`. Per-day CSVs retain the SSE and RMSE components.

Per-window comparisons call `bootstrap.paired_bootstrap_components` with its
unchanged defaults (4,000 repetitions, seed 0). Pooled comparisons call
`bootstrap.date_cluster_bootstrap`, preserving the (window, date) ordering used
by `evaluate`. Reported differences are baseline minus final (positive is worse).
Pooled scores average all window-date rows, so dates repeated across overlapping
windows contribute repeatedly; the bootstrap clusters those repeated dates.

`rank_scores.csv` reports every trial. `selection.json` freezes the selections.
The summary includes all 21 required windows and separately pools the nine
rolling windows, nine shifted windows and all 21 windows. The all-window pool
includes the tuning window and must not be interpreted as a held-out estimate.
No January or stable windows are silently substituted for the requested set.
