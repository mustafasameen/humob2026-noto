"""Bidirectional forecasting context export, and reading back a forecast.

A foundation time-series forecaster is run on this task's diagonal series
the way any general-purpose forecaster has to be, given the shape of the
record: forward from the contiguous block observed before the target, and
backward (series reversed) from the contiguous block observed after it,
because a single window can be closer to one side of the gap than the
other. Not-a-number days inside a context are linearly interpolated so the
forecaster sees a regular calendar; the target days themselves never enter
either context.

This phase of the pipeline only builds and exports that context, and reads
back a forecast that some external forecaster already produced from it (as
a pair of .npz files -- see `read_forecast`). Producing the forecast itself
is a separate, later phase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .calendar import daterange, to_date
from .config import N_EVAL_CELLS
from .data import build_matrix
from .metric import DayFlows

# The record has two contiguous observed blocks around the gap; a context
# is drawn from whichever block contains its anchor date, however far that
# date sits from the block's own edge.
BLOCKS = (
    tuple(daterange("20231101", "20240131")),
    tuple(daterange("20240401", "20241031")),
)


def block_of(d):
    for b in BLOCKS:
        if b[0] <= d <= b[-1]:
            return b
    return None


def _calendar_context(days, obs_set, cal_days, keys):
    """(len(keys) x len(cal_days)) matrix, calendar-regular: a not-observed
    or not-a-number day is linearly interpolated from its neighbours so the
    forecaster sees a complete series."""
    obs = [d for d in cal_days if d in obs_set]
    M = np.full((len(cal_days), keys.size), np.nan)
    if obs:
        B = build_matrix(days, obs, keys)
        idx = {d: i for i, d in enumerate(cal_days)}
        M[[idx[d] for d in obs]] = np.nan_to_num(B)
    x = np.arange(len(cal_days)); ok = ~np.isnan(M[:, 0])
    for j in range(keys.size):
        M[:, j] = np.interp(x, x[ok], M[ok, j])
    return M.T.astype(np.float32)


def build_context(days, na, all_dates, window):
    """Bidirectional forecasting context for one validation window.

    Returns a dict with the same fields the exported .npz carries: `keys`
    (the OD pairs with positive mass in the window's own anchors),
    `is_diag`, `fwd`/`bwd` context matrices, `f_off`/`b_off` (each target
    day's step count ahead of the forward/backward context's edge), `T`
    (the target dates) and the edge dates `E` (last forward-context day)
    and `S` (first backward-context day).
    """
    obs_set = set(all_dates)
    tset = set(window.target)
    support = {}
    for d in list(window.before) + list(window.after):
        if d in days and d not in na:
            for k, v in zip(days[d].key, days[d].val):
                if v > 0:
                    support[int(k)] = 1
    keys = np.array(sorted(support), dtype=np.int64)
    is_diag = (keys // N_EVAL_CELLS) == (keys % N_EVAL_CELLS)

    T = list(window.target)
    before_obs = [d for d in all_dates if d < T[0] and d not in tset]
    E = before_obs[-1]
    fwd_cal = [d for d in block_of(E) if d <= E]
    after_obs = [d for d in all_dates if d > T[-1] and d not in tset]
    S = after_obs[0]
    bwd_cal = [d for d in block_of(S) if d >= S]
    fwd = _calendar_context(days, obs_set, fwd_cal, keys)
    bwd = _calendar_context(days, obs_set, bwd_cal, keys)[:, ::-1].copy()   # reversed in time
    f_off = np.array([(to_date(t) - to_date(E)).days for t in T])          # steps ahead of E
    b_off = np.array([(to_date(S) - to_date(t)).days for t in T])          # steps "ahead" of S, backwards
    if (set(fwd_cal) | set(bwd_cal)) & tset:
        raise AssertionError("a target day leaked into a forecasting context")
    return {"keys": keys, "is_diag": is_diag, "fwd": fwd, "bwd": bwd,
            "f_off": f_off, "b_off": b_off, "T": np.array(T), "E": np.array(E), "S": np.array(S)}


def save_context(path, ctx):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ctx)


def read_forecast(ctx_path, forecast_path):
    """Inverse-distance blend of a forward and a backward forecast into one
    per-day diagonal prediction.

    `ctx_path` is the exported context (for `keys`, `T`, `f_off`, `b_off`);
    `forecast_path` holds the forecaster's own `fwd`/`bwd` arrays, one
    column per forecast step ahead of the context's edge. The weight on the
    backward forecast for a target day is its step count ahead of the
    backward edge divided by the sum of both step counts, so a forecast
    that has to reach further across the gap is automatically
    down-weighted relative to one that does not.
    """
    ctx = np.load(ctx_path, allow_pickle=True)
    fc = np.load(forecast_path)
    keys, T = ctx["keys"], [str(t) for t in ctx["T"]]
    fo, bo = ctx["f_off"], ctx["b_off"]
    out = {}
    for i, d in enumerate(T):
        wb = fo[i] / (fo[i] + bo[i])
        v = np.clip((1 - wb) * fc["fwd"][:, fo[i] - 1] + wb * fc["bwd"][:, bo[i] - 1], 0, None)
        m = v > 0
        out[d] = DayFlows(d, keys[m], v[m].astype(float))
    return out
