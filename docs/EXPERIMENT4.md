# Experiment 4: sensitivity of model constants

Run `scripts/experiment4.py` with the prepared cache and released MLX forecasts. The script runs the unchanged `humob26 sweep` command for each of the four documented settings, always including `submitted: {}` and asserting the reference scores. It selects the `final` rows of `evaluate/window_scores.csv` for that assertion because `evaluate` also emits other model arms.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/experiment4.py --out results/experiment4/run_NEW
```

Use a fresh output directory for each attempt. The result contains the full command and package log, source and input SHA-256 hashes, one CSV per grid entry, four combined-score tables, pooled rolling-window differences with confidence intervals, a report, and a checksums manifest. The bootstrap uses the original 4,000 repetitions, seed 0. No evaluation window, metric, or model default is modified.

If the reference evaluation was already completed successfully in a previous attempt, provide its `reference` directory with `--reference-existing`. The script copies it, verifies the May–June result and records the source path and hash. Failed attempts are preserved under `results/experiment4`.

`fm_diagonal_weight` is the forecast's share of the diagonal average (0.5 averages the forecast and the interpolation-smoother blend equally; 1.0 uses the forecast alone).
