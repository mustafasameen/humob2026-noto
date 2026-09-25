# Reproducing the paper's artifacts

Every command below assumes `prepare` has already been run once:

```
python -m humob26 prepare --data /path/to/humob2026-dataset.tsv --cache ./cache
```

and that a directory of forecast files (see the README's "Forecasts"
section) is available at `./forecasts`, with one `ctx_<window>.npz` /
`fc_<window>.npz` pair per window under its public name (`may_jun`,
`test_gap`, ...).

## Submission file

```
python -m humob26 submit --cache ./cache --forecasts ./forecasts \
    --out submission.tsv --validator /path/to/humob2026_validator_official.py
```

Expected output: `submission.tsv` (the file to upload) and
`submission.fullgrid.tsv` (the intermediate full-grid build); the
validator prints `Validation passed!`. See the README for the expected
md5 of each file.

## Validation table

```
python -m humob26 evaluate --cache ./cache --forecasts ./forecasts \
    --out ./scores
```

Expected output: `./scores/daily_scores.csv` (window, date, arm, Combined,
NRMSE diag, NRMSE off-diagonal), `./scores/window_scores.csv` (the same,
averaged per window), and `./scores/comparisons.csv` (mean per-day
difference, percent change, and a 95% paired bootstrap CI for each of the
default cumulative-build-up arm pairs, plus a pooled rolling-window CI
where all nine `roll_*` windows are present).

## Forecaster comparison

```
python -m humob26 compare-slot --cache ./cache --contexts ./forecasts \
    --reference ./forecasts/reference \
    --arms candidate_a=./forecasts/candidate_a --arms candidate_b=./forecasts/candidate_b \
    --mean ensemble=reference,candidate_a,candidate_b \
    --out ./bakeoff
```

Expected output: `./bakeoff/bakeoff_windows.csv` (per window, per arm:
Combined score, percent better than the reference, diagonal/off-diagonal
NRMSE), `./bakeoff/bakeoff_summary.csv` (the per-arm verdict: pooled
rolling-window CI at the Bonferroni level for the number of arms, no
significant loss on the no-harm windows, PASS/FAIL), and
`./bakeoff/bakeoff_gates.csv` (every shape/finite/quantile-order/offset
gate checked along the way).

## Shock-origin check

```
python -m humob26 compare-slot --cache ./cache --contexts ./forecasts \
    --reference ./forecasts/reference --arms joint=./forecasts/joint \
    --shock-origin-joint-arm joint --out ./bakeoff
```

Expected output (in addition to the forecaster-comparison files above):
`./bakeoff/shock_origin.csv` (the three earliest-January windows: reference
and joint-mode Combined scores), `./bakeoff/shock_origin_summary.csv` (the
pooled per-day difference and its 95% CI on the two windows added since the
reference window, with a KEEP/REVERT verdict), and
`./bakeoff/shock_origin_legs.csv` (the same three windows' forward/backward
context lengths, mean backward blend weight, and the joint gain split into
"forward leg only", "backward leg only" and "both").

## Data description

```
python -m humob26 describe --data /path/to/humob2026-dataset.tsv --out ./describe.txt
```

Expected output: a text report (`./describe.txt`) covering days observed per block; the count and
median positive-day span of evaluation-region pairs ever positive in training and in each anchor; the
diagonal's squared-mass share in its top 5 cells; and the mean diagonal/off-diagonal level of each
anchor. No numbers from a run of this command are recorded in this document; run it to see them.

## Sensitivity sweeps

See `EXTENDING.md` for how to define and run a `sweep` grid.
