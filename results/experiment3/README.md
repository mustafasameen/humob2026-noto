# Experiment 3 run index

- `run_001`: successful CPU run of full-context PyTorch reference and six one-sided cuts across the four named and nine rolling windows, followed by `compare-slot` versus full PyTorch and MLX. Local NPZ outputs are retained under each named run and exposed through `forecasts/timesfm3_*` links. The NPZs and 1.3 GB checkpoint are excluded from Git; their hashes remain in `run.json`, `weights-checksums.json`, and `checksums.json`.
- `download.log`: initial checkpoint attempt failed when the laptop ran out of disk space; `download-retry.log` records the successful pinned revision. `install.log`, `tabulate-install.log`, `packages-after-report.txt` record package setup. The initial `run_001/packages.txt` captures the inference environment; `tabulate` was added after forecasting began to format the report.
- `planted-gate.log`: official PyTorch checkpoint passed the repo's synthetic forward and reverse checks. `tests.log` and `tests-final.log`: unit tests. `independent-verification.log`: forecast-leg and saved-score audit. `percentage-intervals.log`: direct date-cluster bootstrap on percentage gains.
- `context_lengths.csv`: context geometry; `run_001/scores_*.csv`: one per-cut score table; `run_001/pooled_gain_percent.csv`: pooled percent gain and both ordinary and six-arm-corrected intervals; `run_001/REPORT.md` and `FINDINGS.md`: result text.
- `torch_full_mayjun_check.log` preserves an early check that lacked a context file; `_002.log` and CSV show the corrected successful check. `early_*` folders preserve additional May–June spot checks.

Commands:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]' '/private/tmp/humob-timesfm-upstream[torch]' > results/experiment3/install.log 2>&1
.venv/bin/python scripts/download_timesfm3.py > results/experiment3/download.log 2>&1
.venv/bin/python scripts/download_timesfm3.py > results/experiment3/download-retry.log 2>&1
.venv/bin/python scripts/experiment3.py --out results/experiment3/run_001 > results/experiment3/run_001-console.log 2>&1
.venv/bin/python scripts/verify_experiment3.py results/experiment3/run_001 > results/experiment3/independent-verification.log 2>&1
.venv/bin/python scripts/experiment3_percent_intervals.py results/experiment3/run_001 cache > results/experiment3/percentage-intervals.log 2>&1
.venv/bin/python -m pytest -q > results/experiment3/tests-final.log 2>&1
```

The cloned TimesFM source commit is in `upstream-commit.txt`, and the model revision is in `weights-revision.json`. A direct May–June check with full PyTorch produced 0.20291035383556955. The released MLX score is 0.2029103543624606. The model's six prediction arrays were compared for exact equality where the corresponding leg was not cut; the observations are saved in `one-sided-cut-check.txt` and `run_001/leg_verification.csv`. `artifact-checksums.json` covers every committed output and local input, including failures. The post-run score splits and report appendix are reproduced with `.venv/bin/python scripts/package_experiment3.py`, which also refreshes checksum manifests.
