# Additional experiments

Four experiments extend the paper's results. Each runs on a CPU; a GPU only makes the TimesFM-3
runs faster.

| experiment | question | instructions |
|---|---|---|
| 1 | How do established completion methods (per-pair linear interpolation, BTMF, TRMF) score on the same windows? | `src/humob26/baselines/README.md` |
| 2 | How far did each municipality's share recover between late January and April, and when does the reconstruction close half of that change? | `docs/EXPERIMENT2.md` |
| 3 | How much history does TimesFM-3 need? | `docs/EXPERIMENT3.md` |
| 4 | How sensitive is the model to its constants? | `docs/EXPERIMENT4.md` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,baselines,recovery]" && pytest -q
python -m humob26 prepare --data data/raw/humob2026-dataset.tsv --cache cache/
python -m humob26 contexts --cache cache/ --out forecasts/timesfm3_mlx/
```

Put the TimesFM-3 forecast files (`fc_<window>.npz`) in `forecasts/timesfm3_mlx/`, next to the context
files the last command wrote. The paper's forecasts come from the MLX build; forecasts produced with
`python -m humob26 forecast` instead give slightly different scores.

**Reproduction check.** Before anything else:

```bash
python -m humob26 evaluate --cache cache/ --forecasts forecasts/timesfm3_mlx/ \
    --windows may_jun --out results/check/
```

In `results/check/window_scores.csv`, arm `final` on `may_jun` must score **0.202910**. If it does not,
the setup differs from the paper's.

## Settings varied in experiment 4

| setting | values |
|---|---|
| `rts_rank` | 4, 8, 12 (submitted), 24 |
| `anchor_rts_blend` | 0.3, 0.5 (submitted), 0.7 |
| `fm_diagonal_weight` | 0.25, 0.5 (submitted), 1.0: the forecast's share of the diagonal average |
| `single_tau` | 7.588 (Kumamoto only), 26.88 (Hurricane Maria only); unset is the submitted prior |
