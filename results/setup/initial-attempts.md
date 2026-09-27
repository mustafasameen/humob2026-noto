# Initial setup attempts

- SSH clone (sandbox and network-enabled): exit 128, GitHub Repository not found.
- `gh auth status`: exit 127, gh is not installed.
- HTTPS ls-remote in sandbox: exit 128, DNS failure.
- HTTPS clone with network access: succeeded.
- `python3 -m venv .venv`: succeeded.
- Initial `.venv/bin/pip install -e ".[test]"`: failed because sandbox DNS could not reach PyPI.
- Retried installation with network access: succeeded. Installed versions are in versions.txt.
- Anonymous GitHub releases API: HTTP 404. Existing Git HTTPS credential allowed release access.
- Downloaded release forecasts-timesfm3 / timesfm3_mlx_forecasts.tar.gz; verified SHA-256 against release.json and extracted 26 fc_*.npz files to forecasts/timesfm3_mlx/.
- Dataset discovered in Downloads; linked at data/raw/humob2026-dataset.tsv. No raw data is committed.

Full test, prepare, contexts and evaluation commands, exit codes and output are in commands.jsonl and numbered logs.

Reproduction passed on main commit a0c0c7f061f92a5da340c0417fd6828825fa7baf, before creating codex/experiment-1-baselines: final/may_jun combined = 0.2029103543624607 (0.202910 at six decimals). The CSV column is named combined, not score.

Inspected docs/EXPERIMENTS.md completely, CLI evaluation/Pipeline, windows, bootstrap, metric, data, combine, evidence, foundation, interpolation and anchor helpers. No existing source behavior or validation windows were modified.
