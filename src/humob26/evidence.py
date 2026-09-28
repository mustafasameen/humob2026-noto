"""Per-cell recovery paths from public administrative records.

A test of external data, not part of the `final` model. Each diagonal cell's
path between the two anchors is read off the shelter-evacuee series of its
municipality, as published in the Ishikawa Prefecture disaster headquarters
bulletins (https://www.pref.ishikawa.lg.jp/saigai/202401jishin-taisakuhonbu.html).
The series is turned into the fraction of the recovery completed between the
two anchors; the anchor levels themselves still come from the mobility data,
so the records choose the route and never the destination. Everything else is
the `final` construction, and with `trust=0` the arm reproduces `final` exactly.
"""
from __future__ import annotations

import numpy as np

from .anchors import anchor_levels, daytype_factors, daytype_factors_eb, gap_weight, holiday_multiplier
from .calendar import day_type, is_isolated_holiday, to_date
from .combine import DEFAULT_SETTINGS, _recovery_table, blend, build_rts, diagonal_part, weighted_sum
from .config import EVAL_X, EVAL_Y, N_EVAL_CELLS
from .interp import _diag_mask, curve_bounds
from .metric import DayFlows

# Grid geometry published with the challenge: x = 1..100 spans 136.029 to
# 138.042 degrees east and y = 1..70 spans 36.203 to 37.646 degrees north.
LON0, LON1, NX = 136.029, 138.042, 100
LAT0, LAT1, NY = 36.203, 37.646, 70

# Centroids of the municipalities in the evaluation region that the bulletins
# report separately; each cell takes the record of the nearest one.
MUNI_CENTROID = {
    "nanao": (37.043, 136.967),
    "wajima": (37.391, 136.899),
    "suzu": (37.436, 137.261),
    "anamizu": (37.229, 136.912),
    "noto_cho": (37.312, 137.150),
    "shika": (37.099, 136.777),
    "nakanoto": (36.966, 136.895),
    "hakui": (36.895, 136.888),
}


def load_muni_curves(path, indicator="n_evacuees"):
    """{municipality: {date: value}} for the in-region municipalities of a
    bulletin table with columns date, muni, in_eval_box and `indicator`."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"date": str})
    df = df[df.in_eval_box & df[indicator].notna()]
    out = {}
    for muni, g in df.groupby("muni"):
        s = g.groupby("date")[indicator].first().sort_index()
        if len(s) >= 3:
            out[muni] = s.to_dict()
    return out


def cell_to_muni(cells):
    """Nearest municipality centroid for each evaluation-region cell index."""
    nx = EVAL_X[1] - EVAL_X[0] + 1
    names = list(MUNI_CENTROID)
    clat = np.array([MUNI_CENTROID[m][0] for m in names])
    clon = np.array([MUNI_CENTROID[m][1] for m in names])
    cells = np.asarray(cells, dtype=np.int64)
    y = cells // nx + EVAL_Y[0]
    x = cells % nx + EVAL_X[0]
    lat = LAT0 + (y - 1) * (LAT1 - LAT0) / (NY - 1)
    lon = LON0 + (x - 1) * (LON1 - LON0) / (NX - 1)
    d = (lat[:, None] - clat[None, :]) ** 2 + ((lon[:, None] - clon[None, :]) * 0.8) ** 2
    return {int(c): names[i] for c, i in zip(cells, d.argmin(axis=1))}


def recovery_weights(series, target_dates, before_dates, after_dates, anchor_days=10):
    """Weight on the before-anchor on each target day, read off one series:
    1 - (P_b - P(t)) / (P_b - P_a), clipped to [0, 1], where P_b and P_a are the
    series' mean over the anchor days nearest the gap (linear interpolation
    between bulletin dates). None, so that the recovery curve is kept, when the
    series has fewer than three points, does not cover every anchor day used
    (no extrapolation beyond the first or last bulletin), or does not move
    between the anchors."""
    obs = sorted(series.items())
    if len(obs) < 3:
        return None
    b_used, a_used = sorted(before_dates)[-anchor_days:], sorted(after_dates)[:anchor_days]
    if obs[0][0] > b_used[0] or obs[-1][0] < a_used[-1]:
        return None
    t0 = to_date(obs[0][0])
    ot = np.array([(to_date(k) - t0).days for k, _ in obs], float)
    ov = np.array([v for _, v in obs], float)

    def at(ds):
        return float(np.interp((to_date(ds) - t0).days, ot, ov))

    p_b = np.mean([at(d) for d in b_used])
    p_a = np.mean([at(d) for d in a_used])
    span = p_b - p_a
    if abs(span) < 1e-9:
        return None
    return {d: min(max(1.0 - (p_b - at(d)) / span, 0.0), 1.0) for d in target_dates}


def interp_two_anchor_eb_evidence(days, window, na, keys, calib, muni_curves, trust=1.0, settings=None):
    """`interp.interp_two_anchor_eb` with each diagonal cell's recovery-curve
    weight replaced by trust * (its municipality's record) + (1 - trust) *
    (the recovery curve). Cells without a usable record keep the curve."""
    s = settings or DEFAULT_SETTINGS
    fc = daytype_factors(days, calib, keys, shrink=s.daytype_shrink)
    is_diag = _diag_mask(keys)
    fac = np.where(is_diag[None, :], daytype_factors_eb(days, calib, keys), fc) if s.eb_enabled else fc
    hol = holiday_multiplier(days, calib, keys)
    lb = anchor_levels(days, window.before, keys, fac, na, hol)
    la = anchor_levels(days, window.after, keys, fac, na, hol)

    tgt = sorted(window.target)
    t0, t1 = curve_bounds(tgt, window.before, window.after, s.curve_span)
    span = max((t1 - t0).days, 1)
    table = _recovery_table(s)
    records = {m: recovery_weights(v, tgt, window.before, window.after) for m, v in muni_curves.items()}
    records = {m: w for m, w in records.items() if w is not None}
    cells = keys // N_EVAL_CELLS
    c2m = cell_to_muni(np.unique(cells[is_diag])) if is_diag.any() else {}
    muni_of_key = np.array([c2m.get(int(c), "") if dg else "" for c, dg in zip(cells, is_diag)])

    out = {}
    for d in tgt:
        u = min(max((to_date(d) - t0).days / span, 0.0), 1.0)
        w_curve = gap_weight("tau_avg", u, table)
        wa = np.where(is_diag, w_curve, gap_weight("sqrt", u))
        if trust:
            for m, w in records.items():
                sel = muni_of_key == m
                if sel.any():
                    wa = np.where(sel, trust * w[d] + (1 - trust) * w_curve, wa)
        k = day_type(d)
        v = (wa * lb + (1 - wa) * la) * (wa * fac[k] + (1 - wa) * fac[k]) \
            * (hol if is_isolated_holiday(d) else 1.0)
        m = v > 0
        out[d] = DayFlows(d, keys[m], v[m])
    return out, records, muni_of_key


def build_final_evidence(days, window, na, keys, train, calib, forecast_diag, muni_curves, trust=1.0,
                         settings=None):
    """The `final` construction with the evidence-conditioned interpolation."""
    s = settings or DEFAULT_SETTINGS
    a, _, _ = interp_two_anchor_eb_evidence(days, window, na, keys, calib, muni_curves, trust, s)
    r = build_rts(days, window, na, keys, train, rank=s.rts_rank)
    base = blend(a, r, s.anchor_rts_blend, window.target)
    return weighted_sum(diagonal_part(forecast_diag, True), diagonal_part(base, True), diagonal_part(base, False),
                        weights=[s.fm_diagonal_weight, 1.0 - s.fm_diagonal_weight, 1.0])
