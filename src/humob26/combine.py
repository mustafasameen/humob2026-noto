"""Merging predictions, and the named constructions of every scored arm.

A "prediction" throughout this package is a `{date: DayFlows}` dict. Three
primitive merges cover everything the named arms need:

  blend            weighted average of two predictions over the union of
                   their keys (used for the interpolation+smoother blend).
  diagonal_part    keep only a prediction's diagonal, or only its
                   off-diagonal, pairs.
  weighted_sum     weighted sum of several predictions over the union of
                   their keys (used to combine a diagonal-only foundation
                   forecast with an interpolation+smoother prediction that
                   covers both parts).

The arm builders below assemble these primitives, `interp`, `rts` and
`anchors` into the seven named model constructions this package reports:
`april_mean`, `novdec_mean` (the organisers' own baselines), `anchor`
(interpolation alone), `anchor_rts` (interpolation + smoother, evaluation
box), `anchor_rts_fullgrid` (the same, built on the whole grid and
restricted back to the box for scoring), `anchor_rts_fm` (`anchor_rts`
with a foundation-model diagonal blended in), and `final` (the
empirical-Bayes interpolation variant + smoother + foundation-model
diagonal). `anchor_rts_eb` (interpolation-with-EB-shrinkage + smoother,
without the foundation-model blend) is exposed too, since it is the
building block `final` adds the foundation-model diagonal to and is useful
on its own for isolating that one component's effect.
"""
from __future__ import annotations

import numpy as np

from .anchors import average_recovery_weights, constant_baseline, exponential_recovery_weights
from .config import ANCHOR_AFTER_START, ANCHOR_RTS_BLEND, EVAL_X, EVAL_Y, \
    FM_DIAGONAL_WEIGHT, GRID_N_CELLS, GRID_NX, N_EVAL_CELLS, RTS_RANK, Settings, TAU_LITERATURE_RANGE
from .data import top_keys
from .interp import interp_two_anchor, interp_two_anchor_eb
from .metric import DayFlows
from .rts import fit_statespace_gap

DEFAULT_SETTINGS = Settings()


def _recovery_table(settings):
    """The recovery-weight table implied by `settings`, or None to use the
    module default (the common case, and the only one that reproduces the
    submission's exact values)."""
    if settings.single_tau is not None:
        return exponential_recovery_weights(settings.single_tau)
    if settings.tau_range == TAU_LITERATURE_RANGE:
        return None
    return average_recovery_weights(*settings.tau_range)

# ---------------------------------------------------------------------------
# Primitive merges
# ---------------------------------------------------------------------------


def blend(a, b, w, target_dates):
    """(1 - w) * a + w * b, per day, over the union of the two predictions'
    keys for that day."""
    out = {}
    for d in target_dates:
        k = np.union1d(a[d].key, b[d].key)
        x = np.zeros(k.size); y = np.zeros(k.size)
        x[np.searchsorted(k, a[d].key)] = a[d].val
        y[np.searchsorted(k, b[d].key)] = b[d].val
        v = (1 - w) * x + w * y
        m = v > 0
        out[d] = DayFlows(d, k[m], v[m])
    return out


def diagonal_part(pred, diag):
    """Keep only the diagonal pairs of `pred` if `diag` else only the
    off-diagonal ones."""
    return {d: DayFlows(d, f.key[f.is_diag == diag], f.val[f.is_diag == diag])
            for d, f in pred.items()}


def weighted_sum(*preds, weights=None):
    """Weighted sum of several predictions over the union of their keys for
    each day (all `preds` must share the same date keys)."""
    weights = weights or [1.0] * len(preds)
    out = {}
    for d in preds[0]:
        k = np.unique(np.concatenate([p[d].key for p in preds]))
        v = np.zeros(k.size)
        for wi, p in zip(weights, preds):
            v[np.searchsorted(k, p[d].key)] += wi * p[d].val
        m = v > 0
        out[d] = DayFlows(d, k[m], v[m])
    return out


def restrict_to_eval_box(full_grid_pred):
    """Remap a full-grid-space prediction down to evaluation-box key space,
    dropping any pair with an end outside the box. Used to compare
    `anchor_rts_fullgrid` against evaluation-box ground truth."""
    nx = EVAL_X[1] - EVAL_X[0] + 1
    out = {}
    for d, f in full_grid_pred.items():
        o, dd = np.divmod(f.key, GRID_N_CELLS)
        oy, ox = o // GRID_NX + 1, o % GRID_NX + 1
        dy, dx = dd // GRID_NX + 1, dd % GRID_NX + 1
        inb = ((EVAL_Y[0] <= oy) & (oy <= EVAL_Y[1]) & (EVAL_X[0] <= ox) & (ox <= EVAL_X[1])
               & (EVAL_Y[0] <= dy) & (dy <= EVAL_Y[1]) & (EVAL_X[0] <= dx) & (dx <= EVAL_X[1]))
        k = (((oy - EVAL_Y[0]) * nx + (ox - EVAL_X[0])) * N_EVAL_CELLS
             + (dy - EVAL_Y[0]) * nx + (dx - EVAL_X[0]))[inb]
        v = f.val[inb]
        o2 = np.argsort(k)
        out[d] = DayFlows(d, k[o2], v[o2])
    return out


# ---------------------------------------------------------------------------
# Per-window modelling context: which keys to model, which days to train
# and calibrate the day-type factors on, given a window's hidden target.
# ---------------------------------------------------------------------------


def eval_box_context(days, window, all_dates):
    """(keys, train_dates, calib_dates) for an evaluation-box arm on `window`."""
    hidden = set(window.target)
    train = [d for d in all_dates if d not in hidden]
    keys = top_keys(days, train, n=None)
    calib = [d for d in train if d >= ANCHOR_AFTER_START]
    return keys, train, calib


def full_grid_context(full_days, window, all_full_dates):
    """(keys, train_dates, calib_dates) for the full-grid arm on `window`."""
    hidden = set(window.target)
    train = [d for d in all_full_dates if d not in hidden]
    keys = np.unique(np.concatenate([full_days[d].key for d in train]))
    calib = [d for d in train if d >= ANCHOR_AFTER_START]
    return keys, train, calib


# ---------------------------------------------------------------------------
# Named arm constructions
# ---------------------------------------------------------------------------


def build_april_mean(days, window, na):
    """The organisers' "April 2024 mean" baseline, generalised to any
    window: a flat per-pair mean over the window's own after-anchor,
    copied onto every target day."""
    return constant_baseline(days, window.target, window.after, na, by_dow=False)


def build_novdec_mean(days, window, na):
    """The organisers' "Nov-Dec 2023 mean" baseline, generalised: a flat
    per-pair mean over the window's own before-anchor."""
    return constant_baseline(days, window.target, window.before, na, by_dow=False)


def build_anchor(days, window, na, keys, calib, settings=None):
    """Two-anchor interpolation alone: tau-averaged recovery weight on the
    diagonal, square-root weight off it, fixed-shrink day-type factors."""
    s = settings or DEFAULT_SETTINGS
    return interp_two_anchor(days, window.target, window.before, window.after, keys, calib, na,
                             shape="tau_avg", shape_off="sqrt", shrink=s.daytype_shrink,
                             recovery_table=_recovery_table(s), curve_span=s.curve_span)


def build_anchor_eb(days, window, na, keys, calib, settings=None):
    """Two-anchor interpolation with empirical-Bayes-shrunk day-type
    factors on the diagonal."""
    s = settings or DEFAULT_SETTINGS
    return interp_two_anchor_eb(days, window.target, window.before, window.after, keys, calib, na,
                                shrink=s.daytype_shrink, eb_enabled=s.eb_enabled,
                                recovery_table=_recovery_table(s), curve_span=s.curve_span)


def build_rts(days, window, na, keys, train, rank=RTS_RANK):
    """The rank-reduced RTS smoother alone, fit on every non-hidden day."""
    return fit_statespace_gap(days, train, window.target, keys, train, na, rank=rank)


def build_anchor_rts(days, window, na, keys, train, calib, settings=None):
    """`anchor` blended evenly with the RTS smoother."""
    s = settings or DEFAULT_SETTINGS
    a = build_anchor(days, window, na, keys, calib, s)
    r = build_rts(days, window, na, keys, train, rank=s.rts_rank)
    return blend(a, r, s.anchor_rts_blend, window.target)


def build_anchor_rts_eb(days, window, na, keys, train, calib, settings=None):
    """`anchor_eb` blended evenly with the RTS smoother -- the building
    block `final` adds a foundation-model diagonal to."""
    s = settings or DEFAULT_SETTINGS
    a = build_anchor_eb(days, window, na, keys, calib, s)
    r = build_rts(days, window, na, keys, train, rank=s.rts_rank)
    return blend(a, r, s.anchor_rts_blend, window.target)


def build_fullgrid_prediction(full_days, window, na_full, keys, train, calib, settings=None):
    """`anchor` + RTS smoother modelled on the whole grid, in full-grid key
    space. This is also what fills in a submission's out-of-box entries
    (see submission.py), so evaluating it and shipping it always agree."""
    s = settings or DEFAULT_SETTINGS
    a = interp_two_anchor(full_days, window.target, window.before, window.after, keys, calib, na_full,
                          shape="tau_avg", shape_off="sqrt", shrink=s.daytype_shrink,
                          recovery_table=_recovery_table(s), curve_span=s.curve_span)
    r = fit_statespace_gap(full_days, train, window.target, keys, train, na_full, rank=s.rts_rank)
    return blend(a, r, s.anchor_rts_blend, window.target)


def build_anchor_rts_fullgrid(full_days, window, na_full, keys, train, calib, settings=None):
    """`anchor_rts`, but modelled on the whole grid rather than only the
    evaluation-box keys, then restricted back to the box. Returned already
    restricted, so it is directly comparable to the evaluation-box arms."""
    return restrict_to_eval_box(build_fullgrid_prediction(full_days, window, na_full, keys, train, calib, settings))


def build_anchor_rts_fm(days, window, na, keys, train, calib, forecast_diag, settings=None):
    """`anchor_rts` with its diagonal averaged 50/50 with a foundation-model
    forecast; the off-diagonal is `anchor_rts` unchanged."""
    s = settings or DEFAULT_SETTINGS
    base = build_anchor_rts(days, window, na, keys, train, calib, s)
    return weighted_sum(diagonal_part(forecast_diag, True), diagonal_part(base, True), diagonal_part(base, False),
                        weights=[s.fm_diagonal_weight, 1.0 - s.fm_diagonal_weight, 1.0])


def build_final(days, window, na, keys, train, calib, forecast_diag, settings=None):
    """The full method: `anchor_rts_eb` with its diagonal averaged 50/50
    with a foundation-model forecast; the off-diagonal is `anchor_rts_eb`
    unchanged."""
    s = settings or DEFAULT_SETTINGS
    base = build_anchor_rts_eb(days, window, na, keys, train, calib, s)
    return weighted_sum(diagonal_part(forecast_diag, True), diagonal_part(base, True), diagonal_part(base, False),
                        weights=[s.fm_diagonal_weight, 1.0 - s.fm_diagonal_weight, 1.0])


ARM_NAMES = ("april_mean", "novdec_mean", "anchor", "anchor_rts",
            "anchor_rts_fullgrid", "anchor_rts_eb", "anchor_rts_fm", "final")
# Arms that need a foundation-model diagonal forecast to be built.
ARMS_NEEDING_FORECAST = ("anchor_rts_fm", "final")
