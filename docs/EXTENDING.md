# Adding a forecaster

The diagonal foundation-model slot is pluggable. Adding a new model means
writing one adapter module and, usually, nothing else.

## 1. The interface

Implement `humob26.forecasters.base.Forecaster`:

```python
from humob26.forecasters.base import Forecaster, register

@register("my_model")
class MyModelAdapter(Forecaster):
    name = "my_model"

    def __init__(self, model, ...):
        self._model = model

    @classmethod
    def from_pretrained(cls, weights_dir, **kwargs):
        # import the framework HERE, not at module level, so importing
        # this file never requires it to be installed.
        from my_model_framework import load
        return cls(load(weights_dir))

    def predict(self, contexts, horizon):
        # contexts: a list of 1-D arrays, one per series, not necessarily
        # the same length. Return (median, q10, q90), each (len(contexts), horizon).
        ...

    # Optional: only if the model can forecast several series in one call
    # sharing information across them.
    def predict_joint(self, matrix, horizon):
        ...

    # Optional: only if the model can condition on context on both sides
    # of a gap in one pass. mask is (n, L); return (median, q10, q90),
    # each (n, L) -- the whole series, not just the masked span.
    def infill(self, series, mask):
        ...
```

Heavy imports (torch, the model's own package) belong inside `__init__`
or `from_pretrained`, never at module top level: importing the adapter
module registers the class without requiring the framework, and only
`forecast --model my_model` needs it actually installed.

## 2. The planted test

Before `forecast` runs a model on any real window, it checks it on a
synthetic weekly-sine-plus-trend series (`humob26.forecasters.planted`):
forward from a 92-step context, backward (time-reversed) from a 214-step
one, and, for a model with `infill`, an interior 60-step gap between two
contexts. The gate is `median RMSE < 10% of the series' standard
deviation`, and a new adapter must pass it before being trusted on data:

```python
from humob26.forecasters import planted
results = planted.check_directional(my_adapter)
# each result: (n_context, reversed, ok, worst_rmse_over_sd)
```

A model that fails this on real (non-degenerate) contexts almost always has
a wiring bug -- a transposed context, the wrong horizon, unconverted units
-- not a genuine modelling limitation.

## 3. compare-slot

Once a model has an adapter and has been run with `forecast` to produce a
`fc_<window>.npz` per window, it can be scored against the reference
forecaster with no code changes:

```
python -m humob26 compare-slot --cache ./cache --contexts ./forecasts \
    --reference ./forecasts/reference --arms my_model=./forecasts/my_model \
    --out ./bakeoff
```

`compare-slot` holds everything else fixed (the base construction, the
off-diagonal, the windows) and swaps only the diagonal forecast source, so
the resulting verdict isolates that one model's effect. See
`REPRODUCE.md`'s "Forecaster comparison" section for the full output.

# Running a sweep

`sweep` re-scores a small grid of `--set key=value` overrides against the
submitted configuration, using the same arm construction `evaluate` uses.
The overridable settings (see `config.Settings`): `rts_rank`,
`anchor_rts_blend`, `daytype_shrink`, `eb_enabled`, `anchor_before_days`,
`fm_diagonal_weight`, `tau_range` (as `lo,hi`), `single_tau` (bypasses the
tau-averaged recovery curve for one fixed time constant), and `curve_span`
(`target`, the default, runs the recovery curve over each window's target days;
`anchors` runs it from the day after the before-anchor to the day before the
after-anchor, so a window whose anchors do not touch its target, such as
`late_jan`, `april` or `january`, sits where it would in the gap). `evaluate`
takes the same `--set` overrides, e.g. `--set curve_span=anchors`.

A grid is a JSON object of `{run_name: {setting: value, ...}}`; a
`"submitted"` entry with no overrides is the reference point every other
row's pooled rolling-window difference is measured against, and (with
`--assert-reference`) the row that must reproduce a known-good per-window
score table exactly:

```json
{
  "submitted": {},
  "rank_8": {"rts_rank": 8},
  "no_eb": {"eb_enabled": false},
  "wider_tau": {"tau_range": [4.0, 30.0]}
}
```

```
python -m humob26 sweep --cache ./cache --forecasts ./forecasts \
    --grid grid.json --out ./sweep --assert-reference ./scores/window_scores_final.csv
```

Each `run_name` writes its own `./sweep/sweep_<run_name>.csv` (per-window
mean Combined score); rows after `"submitted"` also print the pooled
`roll_*`-window difference against it, with a 95% CI from the same
calendar-date cluster bootstrap `evaluate`'s pooled comparison uses.
