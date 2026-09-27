"""Completion adapters on the paper's unchanged evaluation-box context."""
import numpy as np

from humob26 import anchors, combine, data, metric
from humob26.calendar import daterange, to_date

CALENDAR = tuple(daterange('20231101', '20241031'))


def masked_matrix(days, na, window):
    dates = sorted(d for d in days if d not in na)
    keys, _, _ = combine.eval_box_context(days, window, dates)
    matrix = data.build_matrix(days, CALENDAR, keys)
    hidden = set(na) | set(window.target)
    for i, date in enumerate(CALENDAR):
        if date in hidden:
            matrix[i] = np.nan
    return keys, matrix.T.copy()


def to_prediction(keys, dates, rows):
    if not np.isfinite(rows).all():
        raise ValueError('non-finite completion')
    rows = np.maximum(rows, 0)
    return {d: metric.DayFlows(d, keys[row > 0], row[row > 0])
            for d, row in zip(dates, rows)}


def linear(days, na, window):
    keys, _, _ = combine.eval_box_context(days, window, sorted(d for d in days if d not in na))
    levels = []
    for dates in (window.before, window.after):
        k, v = anchors.mean_level(days, dates, na)
        level = np.zeros(len(keys))
        level[np.searchsorted(keys, k)] = v
        levels.append(level)
    # Match the submitted interpolation's default target-span convention.
    # Actual calendar offsets preserve NA dates inside the target interval.
    start, end = to_date(min(window.target)), to_date(max(window.target))
    span = max((end - start).days, 1)
    rows = [(1 - (to_date(d) - start).days / span) * levels[0]
            + (to_date(d) - start).days / span * levels[1] for d in window.target]
    return to_prediction(keys, window.target, np.asarray(rows))


def complete(days, na, window, method, rank, seed=1000, lags=(1, 2, 7)):
    keys, matrix = masked_matrix(days, na, window)
    np.random.seed(seed)
    n, t = matrix.shape
    lags = np.asarray(lags, dtype=int)
    if method == 'btmf':
        from ._btmf import BTMF
        init = {'W': .01 * np.random.randn(n, rank), 'X': .01 * np.random.randn(t, rank)}
        filled, *_ = BTMF(matrix, init, rank, lags, 1000, 200)
    elif method == 'trmf':
        from ._trmf import TRMF
        init = {'W': .1 * np.random.rand(n, rank), 'X': .1 * np.random.rand(t, rank),
                'theta': .1 * np.random.rand(len(lags), rank)}
        hyper = {'lambda_w': 500, 'lambda_x': 500, 'lambda_theta': 500, 'eta': .03}
        filled = TRMF(matrix, init, hyper, lags, 200)
    else:
        raise ValueError(method)
    idx = [CALENDAR.index(d) for d in window.target]
    return to_prediction(keys, window.target, filled[:, idx].T)
