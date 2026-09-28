from pathlib import Path
import pandas as pd,hashlib,json
root=Path('results/experiment3')
run=root/'run_001'
score=pd.read_csv(run/'comparison/bakeoff_windows.csv')
for arm in ('fwd14','fwd28','fwd56','bwd30','bwd60','bwd120'):
    score[score.arm==arm].to_csv(run/f'scores_{arm}.csv',index=False)
pct=pd.read_csv(run/'pooled_gain_percent.csv')
lines=['','## Pooled percentage gain with date-cluster bootstrap','',
       'A positive gain means the cut improves on full PyTorch context. The familywise intervals use the same six-arm Bonferroni endpoints as `compare-slot`; the unadjusted 95% intervals are also shown. There are 170 unique rolling dates, 4,000 resamples, and seed 0.','',
       '| Cut | Gain (%) | 95% CI (%) | Familywise 95% CI (%) |','|---|---:|---|---|']
for r in pct.itertuples():
    lines.append(f'| {r.arm} | {r.pooled_gain_percent:.4f} | [{r.ci95_lo:.4f}, {r.ci95_hi:.4f}] | [{r.familywise95_lo:.4f}, {r.familywise95_hi:.4f}] |')
lines+=['','A cut leaves a window unchanged when its context is already shorter than the requested limit; see `context_lengths.csv`. Full-context PyTorch reproduces May–June as 0.2029103538 versus 0.2029103544 with released MLX forecasts. The largest absolute score difference over the 13 windows is about 0.0000058 on April. The `compare-slot` verdict column tests a broader no-harm rule; `FAIL` there means the cut did not meet that rule, not that the experiment failed.','']
report_path=run/'REPORT.md'
base=report_path.read_text().split('\n## Pooled percentage gain with date-cluster bootstrap')[0].rstrip()
report_path.write_text(base+'\n'+'\n'.join(lines))
(root/'FINDINGS.md').write_text('''# Draft findings

Against full PyTorch contexts, all six shorter settings have negative pooled rolling-window gain; the largest losses are forward 14 days at -0.470% (95% date-cluster CI -0.568% to -0.366%) and forward 28 days at -0.163% (-0.212% to -0.110%).
With a familywise correction across six comparisons, forward 14/28 and backward 30/120 days show clear losses; the forward 56 and backward 60 intervals include zero.
Full-context PyTorch closely matches the released MLX forecasts: both score 0.202910 on May–June to six decimals, and their largest absolute window-score difference across the 13 windows is about 0.0000058 (April).
''')
(root/'README.md').write_text('''# Experiment 3 run index

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
''')
(run/'checksums.json').write_text(json.dumps({str(f.relative_to(run)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(run.rglob('*')) if f.is_file() and not f.is_symlink() and f.name!='checksums.json'},indent=2)+'\n')
(root/'artifact-checksums.json').write_text(json.dumps({str(f.relative_to(root)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(root.rglob('*')) if f.is_file() and not f.is_symlink() and f.name!='artifact-checksums.json'},indent=2)+'\n')
