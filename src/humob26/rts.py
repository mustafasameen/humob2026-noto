"""A rank-reduced state-space smoother that bridges the gap in factor space.

The missingness here is a "blackout": the entire spatial domain is dark for
58 consecutive days at once, which rules out any imputer that needs some
cell observed on the missing day (kriging, matrix completion, most deep
imputers). What remains applicable is the classical interior-gap
instrument: a state-space model with a fixed-interval (Rauch-Tung-Striebel)
smoother, where the forward Kalman pass simply skips its update step on an
unobserved day and the backward pass then carries information in from the
far side of the gap.

The pipeline here:
  1. reduce the OD pairs to `rank` spatial factors using a plain SVD of the
     day-type-deseasonalised, OBSERVED-days-only matrix;
  2. fit and RTS-smooth each factor independently as a local-linear-trend
     model, which can bend across the gap but cannot invent a bend with no
     evidence for it -- as the fitted slope variance goes to zero it
     degrades to a straight line between the two sides, the correct
     conservative limit;
  3. reconstruct the OD matrix from the smoothed factors and re-seasonalise
     with the same day-type factors used to deseasonalise it.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from .anchors import apply_factors, daytype_factors
from .calendar import day_type, to_date
from .config import DAYTYPE_SHRINK, RTS_CLIP_CAP
from .data import build_matrix
from .metric import DayFlows


def _clip_to_support(pred, observed, cap=RTS_CLIP_CAP):
    """Bound predictions by what each series was ever observed to do: a
    per-series ceiling of `cap` times its observed maximum. A structural
    model extrapolating into an unobserved window can run away, and under
    squared error a single runaway series can dominate a whole day's RMSE,
    so this is cheap insurance rather than a modelling choice."""
    with np.errstate(invalid="ignore"):
        hi = np.nanmax(observed, axis=0)
    hi = np.where(np.isfinite(hi) & (hi > 0), hi, np.nan)
    ceiling = np.where(np.isfinite(hi), cap * hi, np.inf)
    return np.clip(pred, 0.0, ceiling[None, :])


def rts_local_linear(y, observed, q_level, q_slope, r_obs):
    """Kalman filter + RTS smoother for a local-linear-trend model.

    state = [level, slope];  level_t = level_{t-1} + slope_{t-1} + w1
                             slope_t = slope_{t-1} + w2
                             y_t     = level_t + v
    Returns (smoothed_level, smoothed_variance_of_level).
    """
    T = y.size
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level, q_slope])
    H = np.array([1.0, 0.0])

    a_p = np.zeros((T, 2)); P_p = np.zeros((T, 2, 2))
    a_f = np.zeros((T, 2)); P_f = np.zeros((T, 2, 2))

    a = np.array([y[observed][0] if observed.any() else 0.0, 0.0])
    P = np.diag([1e6, 1e3])
    for t in range(T):
        a_p[t] = F @ a
        P_p[t] = F @ P @ F.T + Q
        if observed[t]:
            v = y[t] - H @ a_p[t]
            S = H @ P_p[t] @ H + r_obs
            K = (P_p[t] @ H) / S
            a = a_p[t] + K * v
            P = P_p[t] - np.outer(K, H @ P_p[t])
        else:                       # no update: predict-only through the gap
            a, P = a_p[t], P_p[t]
        a_f[t], P_f[t] = a, P

    a_s = a_f.copy(); P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        try:
            J = P_f[t] @ F.T @ np.linalg.inv(P_p[t + 1])
        except np.linalg.LinAlgError:
            continue
        a_s[t] = a_f[t] + J @ (a_s[t + 1] - a_p[t + 1])
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_p[t + 1]) @ J.T
    return a_s[:, 0], P_s[:, 0, 0]


def _loglik(y, observed, q_level, q_slope, r_obs):
    """One-step-ahead prediction error decomposition, on observed days only."""
    T = y.size
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level, q_slope]); H = np.array([1.0, 0.0])
    a = np.array([y[observed][0], 0.0]); P = np.diag([1e6, 1e3])
    ll = 0.0
    for t in range(T):
        a = F @ a; P = F @ P @ F.T + Q
        if observed[t]:
            v = y[t] - H @ a
            S = float(H @ P @ H + r_obs)
            if S <= 0:
                return -np.inf
            ll += -0.5 * (np.log(2 * np.pi * S) + v * v / S)
            K = (P @ H) / S
            a = a + K * v
            P = P - np.outer(K, H @ P)
    return ll


def fit_factor(y, observed, grid=None):
    """Pick (q_level, q_slope, r_obs) by profile likelihood on a coarse grid.

    Variances are parameterised relative to the observed variance of the
    series, so the same grid works for a dominant factor and a small one.
    """
    s2 = float(np.var(y[observed])) or 1.0
    if grid is None:
        grid = [(a, b) for a in (1e-4, 1e-3, 1e-2, 1e-1)
                for b in (1e-6, 1e-5, 1e-4, 1e-3)]
    best = (-np.inf, 1e-3, 1e-5, s2)
    for ql, qs in grid:
        for rr in (0.1, 0.5, 1.0):
            ll = _loglik(y, observed, ql * s2, qs * s2, rr * s2)
            if ll > best[0]:
                best = (ll, ql * s2, qs * s2, rr * s2)
    return best[1], best[2], best[3]


def fit_statespace_gap(days, train_dates, target_dates, keys, calib_dates,
                       na_dates=(), rank=6, clip_cap=RTS_CLIP_CAP, return_var=False):
    """Reduce spatially on observed days, RTS-smooth each factor over the gap."""
    na = set(na_dates)
    fit = sorted(d for d in train_dates if d in days and d not in na)
    tgt = sorted(target_dates)
    alld = sorted(set(fit) | set(tgt))
    t0, t1 = to_date(alld[0]), to_date(alld[-1])
    span = (t1 - t0).days + 1
    axis = [date.fromordinal(t0.toordinal() + i).strftime("%Y%m%d")
            for i in range(span)]
    pos = {d: i for i, d in enumerate(axis)}

    fac = daytype_factors(days, calib_dates, keys, shrink=DAYTYPE_SHRINK)
    M = build_matrix(days, fit, keys)
    des = M / fac[np.array([day_type(d) for d in fit])]

    # Spatial basis from the OBSERVED days only.
    mu = des.mean(axis=0)
    U, s, Vt = np.linalg.svd(des - mu, full_matrices=False)
    r = min(rank, s.size)
    B = Vt[:r]                                  # (r, n_cells) loadings
    scores_obs = (des - mu) @ B.T               # (n_fit, r)

    observed = np.zeros(span, bool)
    Z = np.zeros((span, r))
    for i, d in enumerate(fit):
        Z[pos[d]] = scores_obs[i]
        observed[pos[d]] = True

    lvl = np.zeros((span, r)); var = np.zeros((span, r))
    for j in range(r):
        ql, qs, rr = fit_factor(Z[:, j], observed)
        lvl[:, j], var[:, j] = rts_local_linear(Z[:, j], observed, ql, qs, rr)

    idx = [pos[d] for d in tgt]
    rec = mu + lvl[idx] @ B
    pred = apply_factors(rec, tgt, fac)
    pred = _clip_to_support(pred, M, clip_cap)
    out = {d: DayFlows(d, keys, pred[i]) for i, d in enumerate(tgt)}
    if not return_var:
        return out
    # Posterior sd of the reconstruction, aggregated over factors: peaks
    # mid-gap and collapses at the anchors, tracking where the smoother
    # itself is least certain.
    sd = np.sqrt((var[idx] @ (B ** 2)).clip(0))
    return out, {d: sd[i] for i, d in enumerate(tgt)}
