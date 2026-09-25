"""Bootstrap confidence intervals for comparing two arms.

Two resampling units are used, depending on what is being compared:

  paired_bootstrap        resamples the days of a SINGLE window with
                           replacement. Appropriate when the days being
                           compared are the only source of randomness under
                           consideration (a within-window arm comparison).

  date_cluster_bootstrap  resamples CALENDAR DATES with replacement, where
                           the same calendar date may contribute rows from
                           several overlapping windows (the nine rolling
                           windows deliberately overlap). Resampling dates
                           rather than (window, date) rows keeps a
                           calendar date's total contribution as one
                           cluster, so a date that happens to appear in
                           several windows cannot be resampled as if it
                           were several independent observations.
"""
from __future__ import annotations

import numpy as np

from .config import BOOTSTRAP_REPS, BOOTSTRAP_SEED
from .metric import combined_score, nrmse_components


def paired_bootstrap(per_day_a, per_day_b, target_dates, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Bootstrap CI for the Combined-score difference (a - b) on one window.

    `per_day_a`/`per_day_b` are per-day sse tables indexed by date (as
    returned by `metric.score`, indexed via `.set_index("date")`), for the
    SAME set of `target_dates`. Returns (estimate, ci_lo, ci_hi, replicates).
    """
    target_dates = list(target_dates)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = list(rng.choice(target_dates, len(target_dates), replace=True))
        diffs[i] = combined_score(per_day_a, idx) - combined_score(per_day_b, idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    est = combined_score(per_day_a, target_dates) - combined_score(per_day_b, target_dates)
    return float(est), float(lo), float(hi), diffs


def paired_bootstrap_components(per_day_a, per_day_b, target_dates, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Like `paired_bootstrap`, but for the Combined score AND its two
    components in one pass, all three built from the SAME resampled day
    index each iteration (paired, not three independent bootstraps).
    Returns {"combined": (est, lo, hi), "diag": (...), "offdiag": (...)}.
    """
    target_dates = list(target_dates)
    rng = np.random.default_rng(seed)
    combined = np.empty(reps)
    diag = np.empty(reps)
    offdiag = np.empty(reps)
    for i in range(reps):
        idx = list(rng.choice(target_dates, len(target_dates), replace=True))
        da, oa = nrmse_components(per_day_a, idx)
        db, ob = nrmse_components(per_day_b, idx)
        diag[i] = da - db
        offdiag[i] = oa - ob
        combined[i] = (da + oa) / 2 - (db + ob) / 2
    da0, oa0 = nrmse_components(per_day_a, target_dates)
    db0, ob0 = nrmse_components(per_day_b, target_dates)
    out = {}
    for name, series, est in (("combined", combined, (da0 + oa0) / 2 - (db0 + ob0) / 2),
                              ("diag", diag, da0 - db0), ("offdiag", offdiag, oa0 - ob0)):
        lo, hi = np.percentile(series, [2.5, 97.5])
        out[name] = (float(est), float(lo), float(hi))
    return out


def date_cluster_bootstrap(dates, diffs, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Pooled bootstrap CI over calendar dates that may repeat across
    overlapping windows.

    `dates`/`diffs` are parallel sequences, one entry per (window, date)
    row, in the exact order those rows were produced (so that summing a
    date's repeated rows is reproducible bit for bit). Each bootstrap
    replicate resamples the set of UNIQUE dates with replacement and takes
    the total of the resampled dates' (possibly multi-window) diff sums
    divided by the total of their row counts. Returns
    (estimate, ci_lo, ci_hi, replicates); `estimate` is the plain mean of
    every (window, date) row, not aggregated by date.
    """
    import pandas as pd
    df = pd.DataFrame({"date": list(dates), "diff": list(diffs)})
    unique_dates = np.array(sorted(df.date.unique()))
    grouped = df.groupby("date")
    sum_by_date, n_by_date = grouped["diff"].sum(), grouped["diff"].size()
    rng = np.random.default_rng(seed)
    boot = [sum_by_date.loc[sample].sum() / n_by_date.loc[sample].sum()
            for sample in (rng.choice(unique_dates, unique_dates.size, replace=True) for _ in range(reps))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    est = df["diff"].mean()
    return float(est), float(lo), float(hi), boot
