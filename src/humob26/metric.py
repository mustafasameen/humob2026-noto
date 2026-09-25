"""The Combined NRMSE metric used to score OD-flow predictions.

For a single day:
    RMSE_diag    = sqrt(mean((pred - obs)^2))   over the N_EVAL_CELLS diagonal pairs
    RMSE_offdiag = sqrt(mean((pred - obs)^2))   over the N_EVAL_OFFDIAG_PAIRS off-diagonal pairs
For a set of days, each RMSE is averaged over days, divided by its own
normaliser (config.NORM_DIAG / config.NORM_OFFDIAG), and the two
normalised terms are averaged to give the Combined score. The convention
is dense: every pair in the evaluation box counts on every day, whether or
not either side ever observed it, so a missing pair contributes zero to
the sum of squares and one to the count.

Two other denominator conventions (a pair counts only if it is ever
non-zero in the observation-or-prediction union, or only if the
observation itself is non-zero) are computed alongside "dense" because
they cost nothing extra and are useful for sanity checks, but "dense" is
the one the leaderboard uses and the one every gate in this package
checks against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import N_EVAL_CELLS, N_EVAL_OFFDIAG_PAIRS, NORM_DIAG, NORM_OFFDIAG

CONVENTIONS = ("dense", "union", "observed")


@dataclass
class DayFlows:
    """One day of evaluation-box OD flow in sparse key/value form.

    `key` is `o_lin * N_EVAL_CELLS + d_lin`, with o_lin/d_lin the row-major
    index of the cell inside the evaluation box. Keys must be sorted and
    unique. `is_diag` is derived from `key` assuming evaluation-box
    spacing, so it is only meaningful for evaluation-box keys; full-grid
    callers (see data.load_full_grid) never call `.split()` or read
    `.is_diag` before their keys are remapped into evaluation-box space.
    """
    date: str
    key: np.ndarray          # int64
    val: np.ndarray          # float64
    is_diag: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.is_diag is None:
            o, d = np.divmod(self.key, N_EVAL_CELLS)
            self.is_diag = (o == d)

    def split(self):
        """(diagonal (key, val), off-diagonal (key, val))."""
        m = self.is_diag
        return (self.key[m], self.val[m]), (self.key[~m], self.val[~m])


def _sse_and_counts(ok, ov, pk, pv):
    """Squared-error sum plus |union| and |observed| for one sparse component."""
    keys = np.union1d(ok, pk)
    o = np.zeros(keys.size)
    p = np.zeros(keys.size)
    o[np.searchsorted(keys, ok)] = ov
    p[np.searchsorted(keys, pk)] = pv
    diff = p - o
    sse = float(diff @ diff)
    n_union = int(np.count_nonzero((o != 0) | (p != 0)))
    n_obs = int(np.count_nonzero(o != 0))
    return sse, n_union, n_obs


def _rmse(sse, n):
    return float("nan") if n == 0 else float(np.sqrt(sse / n))


def score_day(obs: DayFlows, pred: DayFlows):
    """Per-day RMSE for both components, under all three conventions."""
    (od_k, od_v), (oo_k, oo_v) = obs.split()
    (pd_k, pd_v), (po_k, po_v) = pred.split()

    sse_d, un_d, ob_d = _sse_and_counts(od_k, od_v, pd_k, pd_v)
    sse_o, un_o, ob_o = _sse_and_counts(oo_k, oo_v, po_k, po_v)

    out = {"date": obs.date, "sse_diag": sse_d, "sse_offdiag": sse_o,
           "n_pos_obs_diag": ob_d, "n_pos_obs_offdiag": ob_o}
    for conv, nd, no in (
        ("dense", N_EVAL_CELLS, N_EVAL_OFFDIAG_PAIRS),
        ("union", un_d, un_o),
        ("observed", ob_d, ob_o),
    ):
        out[f"rmse_diag_{conv}"] = _rmse(sse_d, nd)
        out[f"rmse_offdiag_{conv}"] = _rmse(sse_o, no)
    return out


def score(obs_days, pred_days, conventions=CONVENTIONS):
    """Full Combined NRMSE. `obs_days`/`pred_days` are {date: DayFlows}.

    Scored over the intersection of dates. Returns (summary_dict, per_day_df).
    """
    import pandas as pd

    dates = sorted(set(obs_days) & set(pred_days))
    if not dates:
        raise ValueError("no overlapping dates between observations and predictions")
    per_day = pd.DataFrame([score_day(obs_days[d], pred_days[d]) for d in dates])

    summary = {"n_days": len(dates)}
    for conv in conventions:
        rd = per_day[f"rmse_diag_{conv}"].mean()
        ro = per_day[f"rmse_offdiag_{conv}"].mean()
        summary[f"rmse_diag_{conv}"] = rd
        summary[f"rmse_offdiag_{conv}"] = ro
        summary[f"nrmse_diag_{conv}"] = rd / NORM_DIAG
        summary[f"nrmse_offdiag_{conv}"] = ro / NORM_OFFDIAG
        summary[f"combined_{conv}"] = (rd / NORM_DIAG + ro / NORM_OFFDIAG) / 2
    return summary, per_day


def component_share(summary, conv="dense"):
    """Share of the combined score coming from each component. Whichever
    component dominates is where modelling effort has the most leverage."""
    d = summary[f"nrmse_diag_{conv}"] / 2
    o = summary[f"nrmse_offdiag_{conv}"] / 2
    tot = d + o
    return {"diag_share": d / tot, "offdiag_share": o / tot, "combined": tot}


def empty_prediction(date):
    """All-zero prediction, the reference point every model must beat."""
    return DayFlows(date, np.empty(0, np.int64), np.empty(0, float))


# ---------------------------------------------------------------------------
# Aggregating a per-day sum-of-squares table over an arbitrary subset of
# days (used both for a plain window score and, with a resampled subset
# that may repeat days, for a bootstrap replicate).
# ---------------------------------------------------------------------------


def nrmse_components(per_day_sse, idx, dense_offdiag_pairs=N_EVAL_OFFDIAG_PAIRS):
    """(NRMSE_diag, NRMSE_offdiag) over `idx`, from a table indexed by date
    with `sse_diag`/`sse_offdiag` columns (dense convention). `idx` may
    repeat dates, which is what makes this usable for a bootstrap resample:
    pandas' `.loc` with a repeated index returns one row per repeat.
    Accepts any sequence of date labels; a tuple is converted to a list
    first, since pandas treats a bare tuple as a multi-axis indexer rather
    than a list of labels."""
    s = per_day_sse.loc[list(idx)]
    nd = np.sqrt(s.sse_diag.values / N_EVAL_CELLS).mean() / NORM_DIAG
    no = np.sqrt(s.sse_offdiag.values / dense_offdiag_pairs).mean() / NORM_OFFDIAG
    return nd, no


def combined_score(per_day_sse, idx, dense_offdiag_pairs=N_EVAL_OFFDIAG_PAIRS):
    """Combined NRMSE over `idx` -- see `nrmse_components`."""
    nd, no = nrmse_components(per_day_sse, idx, dense_offdiag_pairs)
    return (nd + no) / 2


def daily_combined(per_day):
    """Per-day Combined score (dense convention), treating each day as if
    it were the only day scored. `per_day` is either the DataFrame
    `score()` returns as its second element, or that frame already indexed
    by date. Returned as a pandas Series indexed by date."""
    df = per_day if per_day.index.name == "date" else per_day.set_index("date")
    return ((df["rmse_diag_dense"] / NORM_DIAG + df["rmse_offdiag_dense"] / NORM_OFFDIAG) / 2).rename("combined")
