# Experiment 3: TimesFM-3 PyTorch context length

The `timesfm3` adapter's PyTorch backend uses Google's official `timesfm3.torch.TimesFM3Forecaster`; its tests check the output and the direction of each context cut, and `forecast` runs the synthetic forward and backward checks before any window.

Install the `timesfm-torch` extra (`timesfm==3.0.2` with PyTorch), download `google/timesfm-3.0-pytorch` with `scripts/download_timesfm3.py`, and run:

```bash
python scripts/experiment3.py --out results/experiment3/run_NEW \
  --cache cache --contexts forecasts/timesfm3_mlx \
  --weights models/timesfm-3.0-pytorch --device cpu
```

The runner first keeps the full PyTorch forecasts as the reference. It then runs six one-sided cuts: forward 14, 28, 56 days and backward 30, 60, 120 days. It uses the four named windows and all nine rolling windows, the original `compare-slot` scorer and its 4,000-replicate bootstrap. A separate comparison measures PyTorch full context against the paper's MLX forecasts. Each run retains its command, log, package versions, input/source and forecast hashes, and failed attempts. The model weights and generated NPZ files stay local.

The checkpoint comes from Google's [official model repository](https://huggingface.co/google/timesfm-3.0-pytorch) and is subject to its noncommercial weights license. The exact downloaded revision and checksums are recorded in `results/experiment3/weights-revision.json` and `weights-checksums.json`.

The run writes `REPORT.md` with the pooled and per-window gains. On windows whose observed block is shorter than a cut, the cut leaves the forecast unchanged. Forecast files, the checkpoint and run folders stay local (Git ignores them).

The percentage intervals are computed by `scripts/experiment3_percent_intervals.py` using the same calendar-date cluster resampling as `compare-slot`; the report gives both ordinary 95% intervals and familywise-corrected intervals across six cuts. `scripts/verify_experiment3.py` independently checks that each one-sided cut leaves the other forecast leg unchanged and validates every saved window score percentage. Both run after the experiment:

```bash
python scripts/verify_experiment3.py results/experiment3/run_NEW
python scripts/experiment3_percent_intervals.py results/experiment3/run_NEW cache
```
