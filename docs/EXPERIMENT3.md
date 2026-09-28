# Experiment 3: TimesFM-3 PyTorch context length

The existing `forecast` CLI had no PyTorch implementation for TimesFM-3. This branch adds an adapter using Google's official `timesfm3.torch.TimesFM3Forecaster` and tests its output and the direction of each context cut. The mandatory synthetic forward and backward forecast checks passed with the pinned official weights.

Install `timesfm==3.0.2` with PyTorch, download `google/timesfm-3.0-pytorch`, and run:

```bash
python scripts/experiment3.py --out results/experiment3/run_NEW \
  --cache cache --contexts forecasts/timesfm3_mlx \
  --weights models/timesfm-3.0-pytorch --device cpu
```

The runner first keeps the full PyTorch forecasts as the reference. It then runs six one-sided cuts: forward 14, 28, 56 days and backward 30, 60, 120 days. It uses the four named windows and all nine rolling windows from the experiment instructions, the original `compare-slot` scorer and its 4,000-replicate bootstrap. A separate comparison measures PyTorch full context against the released MLX forecasts. Each run retains its command, log, package versions, input/source and forecast hashes, and failed attempts. The model weights and generated NPZ files stay local.

The checkpoint comes from Google's [official model repository](https://huggingface.co/google/timesfm-3.0-pytorch) and is subject to its noncommercial weights license. The exact downloaded revision and checksums are recorded in `results/experiment3/weights-revision.json` and `weights-checksums.json`.

The completed results are in `results/experiment3/run_001/REPORT.md`. `results/experiment3/context_lengths.csv` shows which cuts actually shorten each window. The full PyTorch reference also remains locally available at `forecasts/timesfm3_torch/` through a link to the numbered run; the six cut directories have matching `forecasts/timesfm3_*` links. These forecast files and the checkpoint are excluded from Git, while their SHA-256 hashes and the score CSVs are committed.

The percentage intervals are computed by `scripts/experiment3_percent_intervals.py` using the same calendar-date cluster resampling as `compare-slot`; the report gives both ordinary 95% intervals and familywise-corrected intervals across six cuts. `scripts/verify_experiment3.py` independently checks that each one-sided cut leaves the other forecast leg unchanged and validates every saved window score percentage. The final audit passed 156 leg comparisons and 78 window/arm scores.
