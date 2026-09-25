# Additional experiments

Four experiments that extend the paper's results. Each runs on a CPU; a GPU only makes the TimesFM-3
runs faster.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" && pytest -q
python -m humob26 prepare --data data/raw/humob2026-dataset.tsv --cache cache/
python -m humob26 contexts --cache cache/ --out forecasts/timesfm3_mlx/
```

Put the TimesFM-3 forecast files (`fc_<window>.npz`) in `forecasts/timesfm3_mlx/`, next to the context
files the last command wrote. The paper's forecasts come from the MLX build; if you produce your own with
`python -m humob26 forecast` instead, the scores below differ slightly.

**Reproduction check.** Before anything else:

```bash
python -m humob26 evaluate --cache cache/ --forecasts forecasts/timesfm3_mlx/ \
    --windows may_jun --out results/check/
```

In `results/check/window_scores.csv`, arm `final` on `may_jun` must score **0.202910**. If it does not,
stop: the setup differs from the paper's.

**Rules.** Change only what each experiment names, keep every run (failures too) with its command,
package versions and output CSV, and put code on a branch with a pull request.

## 1. Established methods on the same windows

Run established completion methods through the paper's windows, scorer and bootstrap.

- **Per-pair linear interpolation:** a straight line from each pair's before-anchor mean to its
  after-anchor mean (no day-of-week factors).
- **BTMF** (Bayesian temporal matrix factorization) and **TRMF** (temporal regularized matrix
  factorization) from https://github.com/xinychen/transdim (note its licence).

For each window, build the pairs-by-days matrix on the full calendar, with the unobserved days and the
window's target missing:

```python
import numpy as np
from humob26 import combine, data, windows
from humob26.calendar import daterange
days, na = data.load_eval_box(None, cache="cache/eval_box.npz")
all_dates = sorted(d for d in days if d not in na)
w = windows.get_window("may_jun", days, na)
keys, train, calib = combine.eval_box_context(days, w, all_dates)
cal = daterange("20231101", "20241031")        # the full calendar: temporal models need real time
M = data.build_matrix(days, cal, keys)         # NaN on days not in the file (the Feb-Mar gap)
for i, d in enumerate(cal):
    if d in na or d in w.target:
        M[i] = np.nan                          # missing days and the window's hidden target
```

Fill the target rows with the method, clip at zero, turn each day into
`metric.DayFlows(d, keys[m], row[m])` for the positive entries, and score with
`metric.score({d: days[d] for d in w.target}, pred)`.

- **Settings:** the method papers' defaults; the rank may be chosen from {5, 10, 20} on `may_jun` only.
  Report every rank run.
- **Windows:** `may_jun`, `late_jan`, `april`, the nine `roll_*` and the nine `shift_*` windows.
- **Report:** the combined score per window and pooled, and the paired difference against `final`: per
  window with `bootstrap.paired_bootstrap_components`, pooled with `bootstrap.date_cluster_bootstrap`, as
  `evaluate` does. Code goes in `src/humob26/baselines/` with a script.

## 2. Recovery by municipality

Both anchors are observed, so the change between late January and April can be measured without a
model; the reconstruction then shows how it unfolded in between.

1. Assign each evaluation-region cell to its municipality with `humob26.evidence.cell_to_muni`.
2. For each municipality and observed day, its share = the sum of its cells' within-cell flows divided
   by the day's total over all rows, including the bucket outside the grid (the same every day; read it
   with `data.iter_days(path, keep_oob=True)`).
3. Report each municipality's mean share over the late-January anchor (January 22 to 31) and over April,
   their ratio, and a 95% bootstrap interval for the ratio (resample days, 4,000). Report the bucket
   outside the grid the same way.
4. Regenerate the submission:
   `python -m humob26 submit --cache cache/ --forecasts forecasts/timesfm3_mlx/ --out results/submission.tsv`
   (md5 `f19bfbd37dd859e0c0e928ffd0f83c5a` with the paper's forecasts). Compute the same shares for each
   day of February and March, and report the day on which each municipality's reconstructed share first
   closes half of its late-January-to-April change.

Deliver a CSV, a figure (daily shares by municipality from November to April, observed and
reconstructed) and two or three sentences of findings.

## 3. How much history TimesFM-3 needs

PyTorch build, weights `google/timesfm-3.0-pytorch` on Hugging Face. Run the full contexts first; this
run is the reference and also shows how closely the PyTorch build reproduces the MLX forecasts:

```bash
python -m humob26 forecast --model timesfm3 --backend torch --device cuda \
    --weights /path/to/timesfm-3.0-pytorch --contexts forecasts/timesfm3_mlx/ \
    --out forecasts/timesfm3_torch/
```

(`--device cpu` without a GPU.) Then cut the forward context to the last 14, 28 and 56 days and the
backward context to the first 30, 60 and 120 days, one cut at a time (6 runs):

```bash
python -m humob26 forecast --model timesfm3 --backend torch --device cuda \
    --weights /path/to/timesfm-3.0-pytorch --contexts forecasts/timesfm3_mlx/ \
    --context-days-fwd 28 --out forecasts/timesfm3_fwd28/
python -m humob26 compare-slot --cache cache/ --contexts forecasts/timesfm3_mlx/ \
    --reference forecasts/timesfm3_torch/ --arms fwd28=forecasts/timesfm3_fwd28/ --out results/context/
```

Report, for each cut, the gain over the full contexts (%) on the rolling windows (pooled, with its
interval), `may_jun`, `late_jan`, `april` and `january`. Keep `forecasts/timesfm3_torch/`.

## 4. Sensitivity of the model's constants

`python -m humob26 sweep` (see `EXTENDING.md`) varies one setting at a time around the submitted
configuration. The grid must contain `"submitted": {}`; `--assert-reference` stops the run if that entry
does not reproduce the check above.

```bash
python -m humob26 sweep --cache cache/ --forecasts forecasts/timesfm3_mlx/ --grid grid_rank.json \
    --assert-reference results/check/window_scores.csv --out results/sweep_rank/
```

| setting | values |
|---|---|
| `rts_rank` | 4, 8, 12 (submitted), 24 |
| `anchor_rts_blend` | 0.3, 0.5 (submitted), 0.7 |
| `fm_diagonal_weight` | 0.25, 0.5 (submitted), 1.0 |
| `single_tau` | 7.588 (Kumamoto only), 26.88 (Hurricane Maria only); unset is the submitted prior |

Example grid (`grid_rank.json`):

```json
{"submitted": {}, "rank4": {"rts_rank": 4}, "rank8": {"rts_rank": 8}, "rank24": {"rts_rank": 24}}
```

Report one table per setting: the combined score on each validation window and the pooled
rolling-window difference from the submitted configuration, with its interval.
