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
directory must contain `fc_may_jun.npz` and `fc_test_gap.npz` from the
[forecasts-timesfm3 release](https://github.com/mustafasameen/humob2026-noto/releases/tag/forecasts-timesfm3),
asset `timesfm3_mlx_forecasts.tar.gz` (SHA-256
`9d269e31d397ef36b8e6fc97f53365b9b6f3f8eb7cf37e825549ef328f659826`).

The runner prepares fresh caches from the raw data and regenerates the two
contexts, runs the documented May–June evaluation and checks 0.202910 to six
decimals, then regenerates the submission. It requires both published MD5s:

- submission: `f19bfbd37dd859e0c0e928ffd0f83c5a`
- full-grid intermediate: `be03ccbf5aaebd81a275b3a52953504f`

By default, any mismatch stops the run before the recovery analysis. Output paths are scoped
to the run folder; the original commands and model defaults are preserved.

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
- `REPORT.md`: method notes and results. `FINDINGS.md`: short draft text.
- `run.json`, `commands.jsonl`, `packages.txt`, logs, `verification.json`, and
  `checksums.json`: source/input hashes, commands, versions, timing, checks and
  output hashes. Failed runs retain `failure.log` and `failure.csv`.

The raw TSV, regenerated TSVs and NPZ files remain local under the existing Git
ignore rules. Derived CSVs and figures are committed. Confidence intervals
express uncertainty from observed anchor-day resampling; they do not measure
uncertainty in the reconstructed path.

## Current run status

`run_001` preserves the strict checksum failure. `run_002` explicitly used
`--allow-submission-md5-mismatch` to produce provisional reconstruction outputs.
May–June reproduces 0.202910 and the full-grid MD5 matches, but the final TSV
MD5 is `e5032f9314ddfb61a0e76db9310932a8`. Its cause is unresolved.
The option does not relax the May–June or full-grid checks; it labels the CSVs,
report and figure as provisional. Observed anchor comparisons use only raw data.
An original author submission is needed to investigate the difference.

After the author changed the submission writer to LF line endings (`06a2eaa`),
the downloaded dataset was restored and the strict pipeline rerun as `run_004`.
The dataset SHA-256 matched the prior run exactly. The May–June score was again
0.202910, the full-grid MD5 again matched, and both generated TSVs were
byte-identical to `run_002`. The final submission MD5 still differs, so the
reconstruction and halfway dates remain provisional. See the run index and
`run_004` failure/provenance files for the exact command, versions and hashes.

Run the independent arithmetic audit with:

```bash
.venv/bin/python scripts/verify_experiment2.py \
  data/raw/humob2026-dataset.tsv results/experiment2/run_002
```

The checked-in findings are in `results/experiment2/FINDINGS.md`; the run index
and additional execution records are in that directory's `README.md`.
