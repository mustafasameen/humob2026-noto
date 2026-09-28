# Task 2 run index

- `run_001`: strict attempt; May–June check passed, final submission MD5 failed. Failure CSV/log retained.
- `run_002`: explicitly allowed final-file mismatch; complete provisional analysis, CSVs, figure and REPORT.md. Observed comparisons are independent of the submission.
- `independent-audit.log`: independent raw-dictionary audit of 3,092 shares, bootstrap ratios, first crossings and exclusions; passed. Final checksum mismatch remains unresolved.
- `tests-final.log`: 39 passed, 1 skipped.
- `FINDINGS.md`: three sentences for review; reconstructed dates explicitly provisional.
- `diagnostic_default_threads.log`: submitting without thread limits produced the same mismatching final MD5. This does not identify its cause.
- `run_003`: strict attempt stopped because the downloaded dataset was unavailable at that time; failure record retained.
- `checksum_after_lf/`: submission regenerated after author commit `06a2eaa`; its final MD5 still differed.
- `run_004`: fresh strict attempt with the restored downloaded dataset and current `origin/main`. The May–June score was 0.202910; the full-grid MD5 matched; the final submission MD5 remained `e5032f9314ddfb61a0e76db9310932a8` instead of `f19bfbd37dd859e0c0e928ffd0f83c5a`. Its generated submission and full-grid files are byte-identical to `run_002`. The restored dataset SHA-256 is `13c37f6a6fe42e6e0ca6f673e364ee5077cea5e2bbfb13ef759bb8d3178ef240`, identical to the earlier input. This strict attempt stopped before calculating new recovery results; `run_002` remains the provisional analysis.

## Commands and environment

Worktree: `humob2026-task2`, branch `codex/experiment-2-recovery`, originally based on main `a0c0c7f061f92a5da340c0417fd6828825fa7baf` and now containing author commit `06a2eaa9c5a03ad354b03712a50415204019e6fc`.
The raw data path is a symlink to the existing downloaded input. Released forecasts were copied from the existing local setup. Inputs and complete package versions are recorded in each run's `run.json` and `packages.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test,recovery]' > results/experiment2/install.log 2>&1
.venv/bin/python -m pytest -q > results/experiment2/tests-initial.log 2>&1
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/humob26-mpl-task2 .venv/bin/python scripts/experiment2.py --data data/raw/humob2026-dataset.tsv --forecasts forecasts/timesfm3_mlx --out results/experiment2/run_001 > results/experiment2/run_001-console.log 2>&1
.venv/bin/python -m humob26 submit --cache results/experiment2/run_001/cache --forecasts results/experiment2/run_001/forecasts --out results/experiment2/diagnostic_default_threads/submission.tsv > results/experiment2/diagnostic_default_threads.log 2>&1
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/humob26-mpl-task2 .venv/bin/python scripts/experiment2.py --data data/raw/humob2026-dataset.tsv --forecasts forecasts/timesfm3_mlx --out results/experiment2/run_002 --allow-submission-md5-mismatch > results/experiment2/run_002-console.log 2>&1
.venv/bin/python scripts/verify_experiment2.py data/raw/humob2026-dataset.tsv results/experiment2/run_002 > results/experiment2/independent-audit.log 2>&1
.venv/bin/python -m pytest -q > results/experiment2/tests-final.log 2>&1
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/humob26-mpl-task2 .venv/bin/python scripts/experiment2.py --data data/raw/humob2026-dataset.tsv --forecasts forecasts/timesfm3_mlx --out results/experiment2/run_004 > results/experiment2/run_004-console.log 2>&1
```

Both strict attempts exited 1 at the final submission MD5 check; the other completed commands exited 0. Both test runs passed. The diagnostic used the same virtual environment as both initial runs; no packages changed between them. Each run preserves subprocess commands, durations and exit codes. Raw/generated TSVs and NPZs stay local under existing ignore rules. Their hashes remain in the manifests. `artifact-checksums.json` also records the diagnostic files and post-run reports; the original per-run manifests are unchanged.

No original model, evaluation, window, metric, bootstrap or submission code was modified. The PNG was visually inspected for readable labels and explicit provisional status.
