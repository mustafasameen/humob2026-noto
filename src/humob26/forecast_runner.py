"""Drives a registered forecaster adapter over every exported context file,
writing forecasts in the standard fwd/fwd_q10/fwd_q90/bwd/bwd_q10/bwd_q90
format `foundation.read_forecast` consumes.

Before touching any real window, the adapter is checked on the planted
series (see `forecasters.planted`): a forecaster that fails this on real
data almost always has a wiring bug (wrong horizon, transposed context,
unconverted units), not a modelling limitation, so this is a hard gate, not
a warning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .forecasters import planted


def run_planted_gate(forecaster, check_infill=False):
    """Raise SystemExit with a clear message if the adapter fails the
    planted-series check; otherwise return the printable result lines."""
    lines = []
    ok_all = True
    for n_context, reversed_, ok, worst in planted.check_directional(forecaster):
        lines.append(f"planted {'reversed' if reversed_ else 'forward '} n={n_context}: "
                     f"worst RMSE/sd {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    if check_infill:
        ok, worst = planted.check_infill(forecaster)
        lines.append(f"planted interior-gap infill: worst RMSE/sd {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    if not ok_all:
        raise SystemExit("planted-series gate failed:\n" + "\n".join(lines))
    return lines


def _truncate_context(rows, n_days):
    if n_days is None:
        return rows
    return [r[-n_days:] if len(r) > n_days else r for r in rows]


def _top_joint_rows(ctx, k):
    diag = np.where(ctx["is_diag"])[0]
    level = np.concatenate([ctx["fwd"][diag], ctx["bwd"][diag]], axis=1).mean(axis=1)
    return diag[np.argsort(-level, kind="stable")[:k]]


def forecast_window(forecaster, ctx, joint_top=None, infill=False, diagonal_only=False,
                    context_days_fwd=None, context_days_bwd=None):
    """One context file -> a dict in the standard forecast format, plus
    `mv_rows` if `joint_top` selected any rows for a joint call."""
    keys = ctx["keys"]
    diag = ctx["is_diag"]
    n = keys.size
    hf, hb = int(ctx["f_off"].max()), int(ctx["b_off"].max())
    out = {"fwd": np.zeros((n, hf), np.float32), "fwd_q10": np.zeros((n, hf), np.float32),
          "fwd_q90": np.zeros((n, hf), np.float32), "bwd": np.zeros((n, hb), np.float32),
          "bwd_q10": np.zeros((n, hb), np.float32), "bwd_q90": np.zeros((n, hb), np.float32)}
    rows = np.where(diag)[0] if diagonal_only else np.arange(n)

    if infill:
        # One combined series per row: forward context, an interior gap the
        # width of the shorter horizon's calendar span, the reversed
        # backward context -- reconstructed in a single pass.
        fo, bo = ctx["f_off"], ctx["b_off"]
        span = set((fo + bo).tolist())
        if len(span) != 1:
            raise ValueError("f_off + b_off is not constant across target days; infill needs one gap width")
        gap = min(span) - 1
        before = ctx["fwd"][rows]
        after = ctx["bwd"][rows][:, ::-1]
        xs = np.concatenate([before, np.zeros((rows.size, gap), np.float32), after], axis=1)
        mask = np.zeros(xs.shape, bool)
        mask[:, before.shape[1]:before.shape[1] + gap] = True
        med, lo, hi = forecaster.infill(xs, mask)
        rec_med = med[:, before.shape[1]:before.shape[1] + gap]
        rec_lo = lo[:, before.shape[1]:before.shape[1] + gap]
        rec_hi = hi[:, before.shape[1]:before.shape[1] + gap]
        out["fwd"][rows] = rec_med[:, :hf]
        out["fwd_q10"][rows] = rec_lo[:, :hf]
        out["fwd_q90"][rows] = rec_hi[:, :hf]
        out["bwd"][rows] = rec_med[:, ::-1][:, :hb]
        out["bwd_q10"][rows] = rec_lo[:, ::-1][:, :hb]
        out["bwd_q90"][rows] = rec_hi[:, ::-1][:, :hb]
        return out

    for leg, horizon in (("fwd", hf), ("bwd", hb)):
        n_ctx = context_days_fwd if leg == "fwd" else context_days_bwd
        contexts = _truncate_context([ctx[leg][i] for i in rows], n_ctx)
        med, lo, hi = forecaster.predict(contexts, horizon)
        out[leg][rows], out[f"{leg}_q10"][rows], out[f"{leg}_q90"][rows] = med, lo, hi

    if joint_top:
        top = _top_joint_rows(ctx, joint_top)
        for leg in ("fwd", "bwd"):
            n_ctx = context_days_fwd if leg == "fwd" else context_days_bwd
            contexts = _truncate_context(ctx[leg][top], n_ctx) if n_ctx else ctx[leg][top]
            med, lo, hi = forecaster.predict_joint(np.asarray(contexts), out[leg].shape[1])
            out[leg][top], out[f"{leg}_q10"][top], out[f"{leg}_q90"][top] = med, lo, hi
        out["mv_rows"] = top
    return out


def run_all(forecaster, contexts_dir, out_dir, joint_top=None, infill=False, diagonal_only=False,
           context_days_fwd=None, context_days_bwd=None, sort_quantiles=False):
    """Forecast every ctx_*.npz in `contexts_dir`, writing fc_<name>.npz
    into `out_dir`. Returns a list of (window_name, n_series) printed
    summaries."""
    from .forecasters.chronos2 import sort_quantiles as _sort

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for p in sorted(Path(contexts_dir).glob("ctx_*.npz")):
        name = p.stem[4:]
        ctx = np.load(p, allow_pickle=True)
        result = forecast_window(forecaster, ctx, joint_top, infill, diagonal_only,
                                 context_days_fwd, context_days_bwd)
        if sort_quantiles:
            for leg in ("fwd", "bwd"):
                result[leg], result[f"{leg}_q10"], result[f"{leg}_q90"] = _sort(
                    result[leg], result[f"{leg}_q10"], result[f"{leg}_q90"])
        np.savez_compressed(out_dir / f"fc_{name}.npz", **result)
        summaries.append((name, int(ctx["keys"].size)))
    return summaries
