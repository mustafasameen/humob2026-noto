"""Two-anchor interpolation across the prediction gap.

Every prediction in this module is built the same way: take a day-type
adjusted level from the anchor window before the gap and one from the
anchor window after it, and blend them with a weight that moves from 1
(all before) to 0 (all after) as the target day moves across the gap. The
two variants here differ only in how the day-type factor itself is
estimated:

  plain      one set of factors, shrunk toward the pooled shape by a fixed
             amount everywhere (`anchors.daytype_factors`).
  EB-shrunk  the diagonal's factors use the empirical-Bayes shrinkage
             instead (`anchors.daytype_factors_eb`); the off-diagonal
             keeps the fixed shrink, because its pairs are too sparse for
             the empirical-Bayes variance estimate to be reliable (it
             would over-shrink almost everything to the pooled shape).

Both blend the diagonal with the recovery-weight curve and the
off-diagonal with a plain square-root curve: the recovery-weight prior
describes the fraction of people away from home (a diagonal quantity), and
says nothing about how flow between cells resumes, so it is only applied
where the evidence for it applies.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .anchors import (anchor_levels, daytype_factors, daytype_factors_eb,
                      gap_weight, holiday_multiplier)
from .calendar import day_type, is_isolated_holiday, to_date
from .config import DAYTYPE_SHRINK, GRID_N_CELLS, N_EVAL_CELLS
from .metric import DayFlows


def _diag_mask(keys):
    """Which of `keys` are diagonal pairs, working out whether `keys` lives
    in evaluation-box or full-grid space from the largest key present."""
    nc = N_EVAL_CELLS if keys.max() < N_EVAL_CELLS * N_EVAL_CELLS else GRID_N_CELLS
    return (keys // nc) == (keys % nc)


def curve_bounds(target_dates, before_dates, after_dates, curve_span="target"):
    """First and last calendar day of the recovery curve on a window (see
    config.Settings.curve_span)."""
    tgt = sorted(target_dates)
    if curve_span == "anchors" and before_dates and after_dates:
        return (to_date(max(before_dates)) + dt.timedelta(days=1),
                to_date(min(after_dates)) - dt.timedelta(days=1))
    if curve_span not in ("target", "anchors"):
        raise ValueError(f"unknown curve_span {curve_span!r}")
    return to_date(tgt[0]), to_date(tgt[-1])


def interp_two_anchor(days, target_dates, before_dates, after_dates, keys,
                      calib_dates, na_dates=(), shape="sqrt",
                      shrink=DAYTYPE_SHRINK, shape_off=None, recovery_table=None,
                      curve_span="target"):
    """Anchor blend with long-window day-type factors and holiday handling.

    `shape` governs the diagonal's blend weight, `shape_off` the
    off-diagonal's (defaults to `shape`). Passing `shape="tau_avg"` uses
    the recovery-weight curve (`recovery_table` overrides the default one,
    for a sensitivity run); `"sqrt"` a plain concave recovery.
    """
    fac = daytype_factors(days, calib_dates, keys, shrink=shrink)
    hol = holiday_multiplier(days, calib_dates, keys)
    lb = anchor_levels(days, before_dates, keys, fac, na_dates, hol)
    la = anchor_levels(days, after_dates, keys, fac, na_dates, hol)

    tgt = sorted(target_dates)
    t0, t1 = curve_bounds(tgt, before_dates, after_dates, curve_span)
    span = max((t1 - t0).days, 1)
    is_diag = _diag_mask(keys) if (shape_off is not None and shape_off != shape) else None

    out = {}
    for d in tgt:
        u = min(max((to_date(d) - t0).days / span, 0.0), 1.0)
        wa = gap_weight(shape, u, recovery_table)
        if shape_off is None or shape_off == shape:
            lvl = wa * lb + (1 - wa) * la
        else:
            wo = gap_weight(shape_off, u, recovery_table)
            w = np.where(is_diag, wa, wo)
            lvl = w * lb + (1 - w) * la
        v = lvl * fac[day_type(d)] * (hol if is_isolated_holiday(d) else 1.0)
        m = v > 0
        out[d] = DayFlows(d, keys[m], v[m])
    return out


def interp_two_anchor_eb(days, target_dates, before_dates, after_dates, keys,
                         calib_dates, na_dates=(), shrink=DAYTYPE_SHRINK,
                         eb_enabled=True, recovery_table=None, curve_span="target"):
    """Anchor blend with empirical-Bayes-shrunk day-type factors on the
    diagonal (fixed-shrink factors off it), tau-averaged recovery weight on
    the diagonal and a square-root weight off it.

    This is the one non-default combination of shrinkage rule, blend
    shape and holiday handling used by the full method (see combine.py's
    `final` construction); every other combination `daytype_factors_eb`
    supports is left for the caller to assemble from `anchors` directly.
    Setting `eb_enabled=False` (a sensitivity toggle) falls back to the
    fixed-shrink factor on the diagonal too, i.e. this degenerates to
    `interp_two_anchor` with the same two blend shapes.
    """
    fc = daytype_factors(days, calib_dates, keys, shrink=shrink)
    if eb_enabled:
        fe = daytype_factors_eb(days, calib_dates, keys)
        is_diag = _diag_mask(keys)
        fac_a = np.where(is_diag[None, :], fe, fc)
    else:
        is_diag = _diag_mask(keys)
        fac_a = fc
    fac_b = fac_a          # kept as a second name: the two are blended by
                           # weight below even though they are equal here,
                           # so the arithmetic matches the general form
                           # this is a special case of.
    hol = holiday_multiplier(days, calib_dates, keys)
    lb = anchor_levels(days, before_dates, keys, fac_b, na_dates, hol)
    la = anchor_levels(days, after_dates, keys, fac_a, na_dates, hol)

    tgt = sorted(target_dates)
    t0, t1 = curve_bounds(tgt, before_dates, after_dates, curve_span)
    span = max((t1 - t0).days, 1)
    out = {}
    for d in tgt:
        u = min(max((to_date(d) - t0).days / span, 0.0), 1.0)
        wa = np.where(is_diag, gap_weight("tau_avg", u, recovery_table), gap_weight("sqrt", u))
        k = day_type(d)
        v = (wa * lb + (1 - wa) * la) * (wa * fac_b[k] + (1 - wa) * fac_a[k]) \
            * (hol if is_isolated_holiday(d) else 1.0)
        m = v > 0
        out[d] = DayFlows(d, keys[m], v[m])
    return out
