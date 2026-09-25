"""A synthetic, noise-free planted series every forecaster adapter is
checked against before it is trusted on real data.

The series is a weekly sine on a slow linear trend -- easy for a competent
forecaster to extrapolate almost exactly, so a large error means the
adapter (context orientation, horizon, quantile indexing, unit handling) is
wrong, not that the model is weak. It is checked forward from a short
context and, separately, backward from a longer one (time-reversed), the
same two directions every window is actually forecast from; a model that
supports infilling is also checked on an interior gap between two contexts.
"""
from __future__ import annotations

import numpy as np

HORIZON = 60
N_SERIES = 32
RMSE_FRACTION = 0.10


def planted_series(n_context, phase, reverse=False, horizon=HORIZON):
    """(context, truth, std) for one planted series: `n_context` observed
    steps followed by `horizon` held-out steps, reversed in time if
    `reverse` (context then still comes first; it is simply the tail of the
    time-reversed series)."""
    t = np.arange(n_context + horizon, dtype=float)
    y = 100 + 20 * np.sin(2 * np.pi * (t + phase) / 7) + 0.05 * t
    if reverse:
        y = y[::-1]
    return y[:n_context].astype(np.float32), y[n_context:].astype(np.float32), float(np.std(y))


def planted_block(n_before, gap, n_after, phase):
    """One continuous planted series spanning a before-context, an interior
    gap, and an after-context, for an infill check."""
    t = np.arange(n_before + gap + n_after, dtype=float)
    return (100 + 20 * np.sin(2 * np.pi * (t + phase) / 7) + 0.05 * t).astype(np.float32)


def check_directional(forecaster, horizon=HORIZON, n_series=N_SERIES, rmse_fraction=RMSE_FRACTION):
    """Forward (92-step context) and reversed (214-step context) planted
    checks. Returns a list of (n_context, reversed, ok, worst_rmse_over_sd)."""
    phases = np.linspace(0, 6, n_series)
    out = []
    for n_context, reverse in ((92, False), (214, True)):
        rows = [planted_series(n_context, ph, reverse, horizon) for ph in phases]
        ctxs = [r[0] for r in rows]
        truth = np.stack([r[1] for r in rows])
        sd = np.array([r[2] for r in rows])
        med, lo, hi = forecaster.predict(ctxs, horizon)
        rmse = np.sqrt(((med - truth) ** 2).mean(axis=1))
        ok = bool((rmse < rmse_fraction * sd).all() and np.isfinite(med).all()
                  and (lo <= med + 1e-4).all() and (med <= hi + 1e-4).all())
        out.append((n_context, reverse, ok, float(np.max(rmse / sd))))
    return out


def check_infill(forecaster, n_before=92, gap=HORIZON, n_after=214, n_series=N_SERIES,
                 rmse_fraction=RMSE_FRACTION):
    """Interior-gap infill planted check. Returns (ok, worst_rmse_over_sd)."""
    phases = np.linspace(0, 6, n_series)
    full = np.stack([planted_block(n_before, gap, n_after, ph) for ph in phases])
    sd = full.std(axis=1)
    truth = full[:, n_before:n_before + gap]
    mask = np.zeros(full.shape, bool)
    mask[:, n_before:n_before + gap] = True
    med, _, _ = forecaster.infill(full, mask)
    med_gap = med[:, n_before:n_before + gap]
    rmse = np.sqrt(((med_gap - truth) ** 2).mean(axis=1))
    ok = bool((rmse < rmse_fraction * sd).all() and np.isfinite(med_gap).all())
    return ok, float(np.max(rmse / sd))
