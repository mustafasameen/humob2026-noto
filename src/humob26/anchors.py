"""Anchor-window level statistics: day-type factors and the weights used to
blend a before-anchor and an after-anchor across the gap.

Two related but distinct ideas live here:

  level       the (weekday- and holiday-adjusted) mean flow of an OD pair
              over an anchor window -- "how much flow this pair carries",
              with the weekly rhythm divided out first so the anchor is a
              level, not a mix of level and weekday composition.
  day type    the per-cell weekly (and holiday) shape multiplied back onto
              a level to get a specific day's prediction. Estimated from
              as long a window as possible and shrunk toward the pooled
              (mass-weighted, cross-cell) shape, because a handful of
              per-cell observations is too noisy to fit its own weekly
              profile outright.

Two shrinkage rules are provided for the day-type factors: a fixed pull of
`config.DAYTYPE_SHRINK` toward the pooled shape everywhere, and an
empirical-Bayes rule that estimates, cell by cell, how much of its own
sampling variance a cell's factor carries and shrinks it accordingly
(a well-observed cell keeps its own profile; a sparse one collapses
toward the pooled shape).
"""
from __future__ import annotations

import numpy as np

from .calendar import HOLIDAY_CALIB_EXCLUDE, HOLIDAYS, day_type, dow, is_isolated_holiday
from .config import DAYTYPE_SHRINK, RECOVERY_WEIGHT_TABLE
from .data import build_matrix
from .metric import DayFlows

# ---------------------------------------------------------------------------
# Plain anchor-window levels (no day-type adjustment) -- the organisers'
# style of baseline: copy a fixed window onto every target day.
# ---------------------------------------------------------------------------


def mean_level(days, dates, na_dates=()):
    """Mean flow per OD pair over `dates`, treating an absent pair as zero.

    Returns (key, val). A not-a-number day is skipped and does not count
    toward the denominator -- averaging it in as zero would bias every
    level down.
    """
    use = [d for d in dates if d in days and d not in set(na_dates)]
    if not use:
        return np.empty(0, np.int64), np.empty(0, float)
    keys = np.unique(np.concatenate([days[d].key for d in use] or [np.empty(0, np.int64)]))
    acc = np.zeros(keys.size)
    for d in use:
        f = days[d]
        acc[np.searchsorted(keys, f.key)] += f.val
    return keys, acc / len(use)


def dow_levels(days, dates, na_dates=()):
    """{weekday: (key, val)} -- a separate level per day of week."""
    out = {}
    for w in range(7):
        sub = [d for d in dates if dow(d) == w]
        out[w] = mean_level(days, sub, na_dates)
    return out


def constant_baseline(days, target_dates, anchor_dates, na_dates=(), by_dow=True):
    """Copy a fixed anchor window onto every target day. With `by_dow`,
    each target day gets its own weekday's level; otherwise every target
    day gets the same flat per-pair mean over the whole anchor window --
    the literal construction of the organisers' published baselines."""
    if by_dow:
        lv = dow_levels(days, anchor_dates, na_dates)
        return {d: DayFlows(d, *lv[dow(d)]) for d in target_dates}
    k, v = mean_level(days, anchor_dates, na_dates)
    return {d: DayFlows(d, k, v) for d in target_dates}


# ---------------------------------------------------------------------------
# Day-type (weekday + holiday) factors
# ---------------------------------------------------------------------------


def holiday_multiplier(days, calib_dates, keys, default=0.974):
    """Pooled level ratio of isolated working-weekday holidays to an
    ordinary day of the same weekday.

    Pooled across cells rather than fit per cell: an isolated holiday is
    rare enough in the record that fitting a separate factor per cell would
    let noise swamp the effect. `default` is used when fewer than two
    isolated holidays are observed in `calib_dates`, i.e. when there is not
    enough evidence to estimate the ratio at all.
    """
    use = [d for d in calib_dates if d in days and d not in HOLIDAY_CALIB_EXCLUDE]
    if not use:
        return default
    M = build_matrix(days, use, keys)
    lvl = np.nansum(M, axis=1)
    by = {d: v for d, v in zip(use, lvl)}
    base = {}
    for k in range(7):
        vals = [by[d] for d in use if dow(d) == k and d not in HOLIDAYS]
        base[k] = np.mean(vals) if vals else np.nan
    rel = [by[d] / base[dow(d)] for d in use
           if is_isolated_holiday(d) and np.isfinite(base[dow(d)])]
    return float(np.mean(rel)) if len(rel) >= 2 else default


def daytype_factors(days, calib_dates, keys, min_obs=3, shrink=DAYTYPE_SHRINK):
    """Per-cell day-type factors, shrunk a fixed amount toward the pooled
    (cross-cell) shape.

    Estimated over as long a window as possible. Small cells get pulled
    toward the pooled shape, which is what stops a handful of noisy
    observations from setting a cell's whole weekly profile.
    """
    use = [d for d in calib_dates if d in days and d not in HOLIDAY_CALIB_EXCLUDE]
    M = build_matrix(days, use, keys)
    dt_ = np.array([day_type(d) for d in use])

    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, np.nan)

    fac = np.ones((7, keys.size))
    counts = np.zeros(7)
    for k in range(7):
        m = dt_ == k
        counts[k] = m.sum()
        if m.sum() < min_obs:
            continue
        with np.errstate(invalid="ignore"):
            f = np.nanmean(M[m], axis=0) / mu
        fac[k] = np.where(np.isfinite(f) & (f > 0), f, 1.0)

    # Pooled shape: mass-weighted across cells, so high-volume cells define it.
    w = np.nan_to_num(mu)
    pooled = (fac * w).sum(axis=1) / max(w.sum(), 1e-9)
    pooled = np.where(pooled > 0, pooled, 1.0)
    fac = (1 - shrink) * fac + shrink * pooled[:, None]
    fac /= fac.mean(axis=0, keepdims=True)
    return np.where(np.isfinite(fac) & (fac > 0), fac, 1.0)


def daytype_factors_eb(days, calib_dates, keys, min_obs=3, lam_const=None, kappa=1.0):
    """Per-cell day-type factors with an empirical-Bayes shrinkage weight.

    Each cell-weekday factor is shrunk by lambda = s^2 / (s^2 + tau^2): its
    own sampling variance against the mass-weighted cross-cell signal
    variance (method of moments). A well-observed cell keeps close to its
    own profile; a sparse one collapses toward the pooled shape. Passing a
    fixed `lam_const` instead reduces this to the same rule `daytype_factors`
    uses, with a constant shrink rather than an adaptive one -- useful as a
    known-answer check.
    """
    use = [d for d in calib_dates if d in days and d not in HOLIDAY_CALIB_EXCLUDE]
    M = build_matrix(days, use, keys)
    dt_ = np.array([day_type(d) for d in use])
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, np.nan)
    fac = np.ones((7, keys.size)); s2 = np.full((7, keys.size), np.inf)
    for k in range(7):
        sel = dt_ == k
        if sel.sum() < min_obs:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.nanmean(M[sel], axis=0) / mu
            n = np.sum(np.isfinite(M[sel]), axis=0)
            v = np.nanvar(M[sel], axis=0, ddof=1) / (n * mu ** 2)
        fac[k] = np.where(np.isfinite(f) & (f > 0), f, 1.0)
        s2[k] = np.where(np.isfinite(v), v, np.inf)
    w = np.nan_to_num(mu)
    pooled = (fac * w).sum(axis=1) / max(w.sum(), 1e-9)
    pooled = np.where(pooled > 0, pooled, 1.0)
    if lam_const is not None:
        lam = np.full(fac.shape, lam_const)
    else:
        d2 = (fac - pooled[:, None]) ** 2
        tau2 = np.empty(7)
        for k in range(7):
            ok = np.isfinite(s2[k]) & (w > 0)
            tau2[k] = max(np.average(d2[k][ok] - s2[k][ok], weights=w[ok]), 1e-12) if ok.any() else 1e-12
        with np.errstate(invalid="ignore"):
            lam = kappa * s2 / (s2 + tau2[:, None])
        lam = np.clip(np.where(np.isfinite(lam), lam, 1.0), 0.0, 1.0)
    fac = (1 - lam) * fac + lam * pooled[:, None]
    fac /= fac.mean(axis=0, keepdims=True)
    return np.where(np.isfinite(fac) & (fac > 0), fac, 1.0)


def apply_factors(level, dates, fac, hol=None):
    """Multiply a per-pair level by each date's day-type (and, if given,
    isolated-holiday) factor."""
    out = level * fac[np.array([day_type(d) for d in dates])]
    if hol is not None:
        mult = np.array([hol if is_isolated_holiday(d) else 1.0 for d in dates])
        out = out * mult[:, None]
    return out


def anchor_levels(days, anchor_dates, keys, fac, na_dates=(), hol=None):
    """Day-type-adjusted mean level over an anchor window.

    Dividing each observed day by its day-type factor before averaging is
    what makes the anchor a LEVEL rather than a mix of level and weekday
    composition, which matters whenever an anchor window has an uneven
    number of weekend days.
    """
    na = set(na_dates)
    use = [d for d in anchor_dates if d in days and d not in na]
    M = build_matrix(days, use, keys)
    des = M / fac[np.array([day_type(d) for d in use])]
    if hol is not None:
        mult = np.array([hol if is_isolated_holiday(d) else 1.0 for d in use])
        des = des / mult[:, None]
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(des, axis=0))


# ---------------------------------------------------------------------------
# The recovery-weight curve used to blend the before- and after-anchor
# levels across the gap.
# ---------------------------------------------------------------------------

_RECOVERY_CURVE = (np.linspace(0.0, 1.0, len(RECOVERY_WEIGHT_TABLE)),
                    np.asarray(RECOVERY_WEIGHT_TABLE, float))


def recovery_weight(u, table=None):
    """Weight on the before-anchor at fraction `u` of the gap elapsed,
    under the time-constant-averaged recovery curve (see config.py, or
    `table`, a same-length override for a sensitivity run).

    A single exponential relaxation time constant tau is not identified
    from the data at hand, so the point prediction used here is a point
    estimate averaged over a prior on tau, rather than the curve at one
    arbitrary tau.
    """
    curve = _RECOVERY_CURVE if table is None else (np.linspace(0.0, 1.0, len(table)), np.asarray(table, float))
    return float(np.interp(u, curve[0], curve[1]))


def gap_weight(shape, u, recovery_table=None):
    """Weight on the before-anchor at fraction `u` of the gap elapsed, for
    one of the supported blend shapes: linear, sqrt (fast early recovery
    that then flattens), square, the recovery-weight curve above, or after
    (the after-anchor level throughout: no before-anchor, no curve)."""
    if shape == "after":
        return 0.0 * u
    return {"linear": 1 - u, "sqrt": 1 - np.sqrt(u), "square": (1 - u) ** 2,
            "tau_avg": recovery_weight(u, recovery_table)}[shape]


def exponential_recovery_weights(tau, n_days=60, after_anchor_offset=60.0):
    """w(t) = (exp(-t/tau) - exp(-T/tau)) / (1 - exp(-T/tau)) for t = 0 ..
    n_days - 1, T = after_anchor_offset -- the single-time-constant curve
    `RECOVERY_WEIGHT_TABLE` averages over (see config.py). Exposed for a
    sensitivity run pinned to one tau instead of the averaged prior."""
    t = np.arange(n_days, dtype=float)
    T = float(after_anchor_offset)
    return (np.exp(-t / tau) - np.exp(-T / tau)) / (1.0 - np.exp(-T / tau))


def average_recovery_weights(tau_lo, tau_hi, n_days=60, after_anchor_offset=60.0, n_grid=4000):
    """`exponential_recovery_weights` averaged over tau on a fine uniform
    grid spanning [tau_lo, tau_hi] -- how `RECOVERY_WEIGHT_TABLE` itself is
    reproduced (to about 1e-3; see config.py), and how a sensitivity run
    with a different tau range builds its own table."""
    taus = np.linspace(tau_lo, tau_hi, n_grid)
    curves = np.stack([exponential_recovery_weights(tau, n_days, after_anchor_offset) for tau in taus])
    return curves.mean(axis=0)
