# Experiment 4: sensitivity of model constants

Run `scripts/experiment4.py` from this branch with the prepared cache and released MLX forecasts. The script runs the unchanged `humob26 sweep` command for each of the four documented settings, always including `submitted: {}` and asserting the reference scores. It selects the `final` rows of `evaluate/window_scores.csv` for that assertion because `evaluate` also emits other model arms.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/experiment4.py --out results/experiment4/run_NEW
```

Use a fresh output directory for each attempt. The result contains the full command and package log, source and input SHA-256 hashes, one CSV per grid entry, four combined-score tables, pooled rolling-window differences with confidence intervals, a report, and a checksums manifest. The bootstrap uses the original 4,000 repetitions, seed 0. No evaluation window, metric, or model default is modified.

If the reference evaluation was already completed successfully in a previous attempt, provide its `reference` directory with `--reference-existing`. The script copies it, verifies the May–June result and records the source path and hash. Failed attempts are preserved under `results/experiment4`.

The `fm_diagonal_weight` values in this repository multiply **both** diagonal components. Changing 0.5 to 0.25 halves their sum; changing it to 1.0 doubles their sum. This explains why that sweep has much larger score changes than the other settings. A direct May–June `evaluate --set fm_diagonal_weight=0.25` check matched the sweep score to floating-point precision.
