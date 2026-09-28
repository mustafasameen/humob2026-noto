# Experiment 2: municipality recovery

This analysis uses the existing nearest-centroid mapping and the original final
submission. It adds `humob26.recovery` and `scripts/experiment2.py`; no model,
validation-window, metric, bootstrap or submission implementation is changed.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test,recovery]'
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment2.py \
  --data data/raw/humob2026-dataset.tsv \
  --forecasts forecasts/timesfm3_mlx \
  --out results/experiment2/run_001
```

The output directory must be new. A failed attempt is retained and cannot be
overwritten. Choose `run_002`, etc. for a new attempt. The supplied forecast
directory must contain the paper's TimesFM-3 forecasts `fc_may_jun.npz` and `fc_test_gap.npz` (MLX
build). They are derived from the challenge data, so they are not distributed with the code; the authors
share them on request with users of the challenge data. Forecasts made with `python -m humob26 forecast`
differ slightly and do not pass the reproduction checks below.

The runner prepares fresh caches from the raw data and regenerates the two
contexts, runs the documented May–June evaluation and checks 0.202910 to six
decimals, then regenerates the submission and checks it:

- full-grid intermediate MD5: `be03ccbf5aaebd81a275b3a52953504f`
- submission with every value rounded to six decimals (`submission.rounded_digest`):
  `f96f76dfcf777b36fff319e9292e8fe1`

The submission's in-box values are written at full precision, so its exact MD5
(`f19bfbd37dd859e0c0e928ffd0f83c5a` on the reference build) can differ in the
last digits between machines; the rounded digest does not. Any mismatch stops
the run before the recovery analysis. Output paths are scoped to the run folder.

## Definitions

A municipality's numerator is the sum of within-cell flows for evaluation-region
cells assigned to it by `evidence.cell_to_muni`. This is a nearest-centroid
approximation, not a municipal boundary polygon assignment.

The denominator is the sum over **every raw OD row** with `keep_oob=True`, including
outside-to-grid and grid-to-outside flows. Observed days use their own totals.
After confirming these totals are constant to numerical tolerance, reconstructed
days use their mean. Model output totals are not used as denominators.

The outside-grid bucket's observed numerator is the within-bucket
`-1_-1 -> -1_-1` entry. Cross-boundary movement remains in the denominator.
The submission omits all outside-grid entries, so the bucket has an observed
anchor comparison but no reconstructed share or halfway date.

Anchor means use observed days from January 22–31 and April 1–30. The ratio is
April / January. Independently resample days within each anchor 4,000 times
with replacement (seed 0); use the 2.5th and 97.5th percentiles of ratios of the
resampled means. Date indices are shared across municipality columns. Missing
days are never treated as observed zero flow. A zero January denominator makes
the ratio undefined; any undefined resampled denominator makes its CI undefined
and is counted in the output.

For each municipality, the first available reconstructed day with
`(share - January mean) / (April mean - January mean) >= 0.5` is the halfway
date. This handles increases and decreases. The check does not require sustained
crossing. An unchanged anchor level, an unreached threshold and an absent
reconstruction have distinct statuses. February 2 and March 5 are excluded by
the original submission and remain blank; no extra forecasts are generated.

## Outputs and provenance

- `municipality_summary.csv`: observed anchor means and counts, April/January
  ratio and bootstrap CI, halfway level, first date, and explicit status fields.
- `daily_shares.csv`: one row per day/group on the 366-day calendar, with shares,
  numerators, denominators, source and missing-status fields. This includes all
  raw observed dates; the figure shows November–April.
- `daily_totals.csv`: raw all-row totals and outside-bucket diagnostics.
- `cell_to_municipality.csv`: assignment of all 1,476 evaluation cells.
- `municipality_shares.png` and `.pdf`: observed/reconstructed time series.
- `REPORT.md`: method notes and results.
- `run.json`, `commands.jsonl`, `packages.txt`, logs, `verification.json`, and
  `checksums.json`: source/input hashes, commands, versions, timing, checks and
  output hashes. Failed runs retain `failure.log` and `failure.csv`.

The raw TSV, regenerated TSVs, NPZ files and run folders stay local (Git
ignores them). Confidence intervals
express uncertainty from observed anchor-day resampling; they do not measure
uncertainty in the reconstructed path.

## Independent audit

Run the independent arithmetic audit with:

```bash
.venv/bin/python scripts/verify_experiment2.py \
  data/raw/humob2026-dataset.tsv results/experiment2/run_001
```
