# Draft findings

Against full PyTorch contexts, all six shorter settings have negative pooled rolling-window gain; the largest losses are forward 14 days at -0.470% (95% date-cluster CI -0.568% to -0.366%) and forward 28 days at -0.163% (-0.212% to -0.110%).
With a familywise correction across six comparisons, forward 14/28 and backward 30/120 days show clear losses; the forward 56 and backward 60 intervals include zero.
Full-context PyTorch closely matches the released MLX forecasts: both score 0.202910 on May–June to six decimals, and their largest absolute window-score difference across the 13 windows is about 0.0000058 (April).
