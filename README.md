# Look Both Ways: Reconstructing Two Hidden Months of Post-Earthquake Mobility

Code for the paper by Mustafa Sameen, Mrunal Vibhute and Xilei Zhao (University of Florida) at the
ACM SIGSPATIAL International Workshop on the Human Mobility Prediction Challenge (HuMob '26). The
Python package is `humob26`.

A gap-filling pipeline for the HuMob Challenge 2026: given daily
origin-destination (OD) flow counts on a 100 x 70 grid of 2 km cells for
Nov 2023 - Jan 2024 and Apr - Oct 2024, predict the flow for every day of
the intervening Feb 1 - Mar 31 2024 gap.

## Method

Every prediction is built from a before-anchor (the observed level just
before the gap) and an after-anchor (the observed level just after it),
blended across the gap, with a rank-reduced state-space smoother and a
foundation-model forecast folded in on the cell-to-self ("diagonal") pairs,
where the gap concerns how much activity a cell retains rather than how it
splits between origins and destinations.

- **Two-anchor interpolation.** Per OD pair, a day-type-adjusted level from
  each anchor window, blended with a weight that moves from the before-
  anchor to the after-anchor across the gap. Day-type factors (weekday and
  Japanese public holidays) are estimated over as long a window as
  possible and shrunk toward the pooled cross-cell shape, either by a
  fixed amount or, on the diagonal, by an empirical-Bayes rule that shrinks
  a cell's own factor in proportion to its own sampling variance. The
  diagonal's blend weight is an exponential-recovery curve averaged over a
  literature-bracketed prior on its time constant; the off-diagonal's is a
  plain square-root recovery.
- **State-space smoother.** The OD matrix is reduced to a small number of
  spatial factors on observed days only, each factor is bridged across the
  gap with a fixed-interval (Rauch-Tung-Striebel) smoother, and the
  reconstruction is mapped back through the same day-type factors.
- **Foundation-model diagonal.** A general-purpose time-series forecaster
  is run forward from the pre-gap series and backward from the reversed
  post-gap series; the two forecasts are blended by inverse distance to
  the gap and averaged evenly with the interpolation-plus-smoother
  diagonal. `evaluate` and `submit` read a forecaster's output from a
  directory of files (see "Forecasts" below); `forecast` produces that
  output by running a pluggable adapter (see "Forecasters").
- **Scope.** Every pooled quantity (day-type factors, holiday multiplier,
  smoother basis) is estimated on evaluation-box pairs only. Predictions
  for pairs with an end outside the evaluation box use the interpolation
  and smoother built on the whole grid, since nothing else ever sees them.

Nine named arms cover the organisers' own published baselines, each model
component alone, and the full stack: `april_mean`, `novdec_mean` (flat
per-pair means over the after- and before-anchor, the organisers'
construction), `after_mean_dow` (the after-anchor mean with the
interpolation's day-type and holiday factors), `anchor` (interpolation
alone), `anchor_rts` (interpolation + smoother), `anchor_rts_fullgrid` (the
same, built on the whole grid and restricted back to the evaluation box),
`anchor_rts_eb` (`anchor_rts` with the empirical-Bayes interpolation
variant), `anchor_rts_fm` (`anchor_rts` with the foundation-model
diagonal), and `final` (the empirical-Bayes interpolation variant +
smoother + foundation-model diagonal: the full method).

## External recovery records

`humob26.evidence` tests whether public administrative records improve the
path between the anchors (Section 5.4 of the paper); it is not part of the
`final` model. `external/ishikawa_per_muni.csv` holds the per-municipality
shelter-evacuee counts parsed from the Ishikawa Prefecture disaster
headquarters bulletins
(https://www.pref.ishikawa.lg.jp/saigai/202401jishin-taisakuhonbu.html),
one row per bulletin date and municipality. `scripts/external_records.py`
reruns the test; see its docstring for the arguments.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

Requires Python >= 3.10, numpy and pandas.

## Data access

The dataset is distributed by the HuMob Challenge 2026 organisers via
Zenodo, access by request: https://doi.org/10.5281/zenodo.20709796. Place
the downloaded TSV file anywhere on disk; every command below takes its
path explicitly. Nothing in this repository bundles, caches a copy of, or
redistributes the dataset itself.

If you use this dataset, cite the dataset and the data descriptor of its
earlier version:

> Yabe, T., Tsubouchi, K., Shimizu, T. HuMob Challenge 2026. Zenodo (2026).
> https://doi.org/10.5281/zenodo.20709796

> Yabe, T., Tsubouchi, K., Shimizu, T., Sekimoto, Y., Sezaki, K., Moro, E.,
> Pentland, A. YJMob100K: City-scale and longitudinal dataset of anonymized
> human mobility trajectories. *Scientific Data* **11**, 397 (2024).
> https://doi.org/10.1038/s41597-024-03237-9

## Reproducing a submission

```
python -m humob26 prepare  --data /path/to/humob2026-dataset.tsv --cache ./cache
python -m humob26 contexts --cache ./cache --out ./contexts
# ... run a forecaster on ./contexts, producing fc_<window>.npz files (see "Forecasts") ...
python -m humob26 evaluate --cache ./cache --forecasts ./forecasts --out ./scores
python -m humob26 submit   --cache ./cache --forecasts ./forecasts --out submission.tsv \
    --validator /path/to/humob2026_validator_official.py
```

`submit` also writes an intermediate full-grid file (by default
`<out>.fullgrid.tsv`) covering every pair on the whole grid; the final
submission is that file with every evaluation-box pair replaced by the
`final` arm's predictions at full precision. Rebuilding from the same
dataset and forecasts reproduces the submitted file:

| file                     | expected md5                      |
|--------------------------|------------------------------------|
| `submission.tsv`         | `f19bfbd37dd859e0c0e928ffd0f83c5a` |
| `submission.tsv`, values rounded to six decimals (`humob26.submission.rounded_digest`) | `f96f76dfcf777b36fff319e9292e8fe1` |
| `submission.fullgrid.tsv`| `be03ccbf5aaebd81a275b3a52953504f` |

The in-box values are written at full precision, so a machine whose
linear-algebra library rounds differently can change their last digits and
the file's exact md5; the rounded digest and the full-grid file do not change.

## Forecasts

`contexts` exports one `ctx_<window>.npz` file per window, each holding:

| array      | shape                  | meaning                                              |
|------------|------------------------|-------------------------------------------------------|
| `keys`     | `(n_series,)`          | evaluation-box OD-pair keys this window forecasts      |
| `is_diag`  | `(n_series,)`          | whether each key is a diagonal (cell-to-self) pair     |
| `fwd`      | `(n_series, n_fwd)`    | the calendar-regular series observed before the gap    |
| `bwd`      | `(n_series, n_bwd)`    | the reversed series observed after the gap             |
| `f_off`    | `(n_target,)`          | each target day's step count ahead of `fwd`'s last day |
| `b_off`    | `(n_target,)`          | each target day's step count ahead of `bwd`'s last day (backwards) |
| `T`        | `(n_target,)`          | the target dates                                       |
| `E`, `S`   | scalar                 | the last forward-context date and first backward-context date |

A forecaster consumes `fwd` and `bwd` and must produce a matching
`fc_<window>.npz` with `fwd` and `bwd` arrays of shape `(n_series, h)` for
some forecast horizon `h >= max(f_off)` (respectively `max(b_off)`):
column `f_off[i] - 1` of `fwd` is the forecast for target day `i` counting
forward from `E`, and column `b_off[i] - 1` of `bwd` is the forecast for
target day `i` counting backward from `S`. `evaluate` and `submit` read
only the diagonal series of `fc_<window>.npz` and blend the two directions
by inverse distance to the gap.

## Forecasters

```
python -m humob26 forecast --model timesfm3 --backend mlx --weights /path/to/weights \
    --contexts ./contexts --out ./forecasts
```

`forecast` runs one registered adapter (`timesfm3`, `chronos2`, or
`patchtst_fm`; each needs that model's own framework installed, via this
package's `[timesfm-mlx]`/`[timesfm-torch]`/`[chronos]`/`[patchtst]` extras
or an environment that already has it) over every exported context,
checking it on a synthetic planted series first. `compare-slot` scores a
candidate forecaster against a reference one with everything else held
fixed, `sweep` re-scores a grid of model-constant overrides, and `describe`
reports summary statistics of the data. See `docs/REPRODUCE.md` for the
exact commands and `docs/EXTENDING.md` for how to add a new forecaster.

## Citation

Mustafa Sameen, Mrunal Vibhute, and Xilei Zhao. 2026. Look Both Ways: Reconstructing Two Hidden Months of
Post-Earthquake Mobility. In ACM SIGSPATIAL International Workshop on the Human Mobility Prediction
Challenge (HuMob '26), November 3, 2026, Riverside, CA, USA. ACM, New York, NY, USA.

```bibtex
@inproceedings{sameen2026lookbothways,
  author    = {Sameen, Mustafa and Vibhute, Mrunal and Zhao, Xilei},
  title     = {Look Both Ways: Reconstructing Two Hidden Months of Post-Earthquake Mobility},
  booktitle = {ACM SIGSPATIAL International Workshop on the Human Mobility Prediction Challenge (HuMob '26)},
  year      = {2026},
  address   = {Riverside, CA, USA},
  publisher = {ACM}
}
```
