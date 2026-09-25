"""Endpoint checks for the two-anchor interpolation blend weights."""
import numpy as np
import pytest

from humob26.anchors import gap_weight, recovery_weight
from humob26.calendar import daterange
from humob26.config import N_EVAL_CELLS, RECOVERY_WEIGHT_TABLE
from humob26.interp import interp_two_anchor
from humob26.metric import DayFlows


def test_gap_weight_endpoints():
    assert gap_weight("linear", 0.0) == 1.0
    assert gap_weight("linear", 1.0) == 0.0
    assert gap_weight("sqrt", 0.0) == 1.0
    assert gap_weight("sqrt", 1.0) == 0.0
    assert gap_weight("square", 0.0) == 1.0
    assert gap_weight("square", 1.0) == 0.0


def test_gap_weight_is_monotone_on_sqrt_and_linear():
    us = np.linspace(0, 1, 11)
    for shape in ("linear", "sqrt", "square"):
        w = [gap_weight(shape, u) for u in us]
        assert all(a >= b for a, b in zip(w, w[1:]))    # non-increasing


def test_recovery_weight_endpoints_match_table():
    assert recovery_weight(0.0) == pytest.approx(RECOVERY_WEIGHT_TABLE[0])
    assert recovery_weight(1.0) == pytest.approx(RECOVERY_WEIGHT_TABLE[-1])
    assert RECOVERY_WEIGHT_TABLE[0] == 1.0


def _flat_days(dates, key, value):
    return {d: DayFlows(d, np.array([key], dtype=np.int64), np.array([value])) for d in dates}


def test_interp_two_anchor_offdiagonal_hits_anchors_exactly_at_the_edges():
    # A flat (no weekday variation) synthetic series, so day-type factors
    # collapse to 1 and the blend at the window edges should equal the
    # anchor levels exactly under the square-root shape (weight 1 and 0
    # at u = 0 and u = 1 respectively).
    calib = daterange("20240601", "20240630")
    before = calib[-10:]
    after = daterange("20240701", "20240710")
    target = daterange("20240801", "20240805")      # no holidays in this range

    offdiag_key = 1     # (o_lin=0, d_lin=1): not on the diagonal
    days = {}
    for d in calib:
        days[d] = DayFlows(d, np.array([offdiag_key], dtype=np.int64), np.array([50.0]))
    for d in after:
        days[d] = DayFlows(d, np.array([offdiag_key], dtype=np.int64), np.array([80.0]))

    keys = np.array([offdiag_key], dtype=np.int64)
    pred = interp_two_anchor(days, target, before, after, keys, calib, na_dates=(),
                             shape="tau_avg", shape_off="sqrt")

    first, last = target[0], target[-1]
    assert pred[first].val[0] == pytest.approx(50.0, rel=1e-9)
    assert pred[last].val[0] == pytest.approx(80.0, rel=1e-9)


def test_interp_two_anchor_diagonal_starts_at_the_before_anchor():
    # The recovery-weight table is exactly 1.0 at u = 0, so a diagonal
    # pair's first target day should equal the before-anchor level exactly,
    # regardless of the table's (non-zero) tail at u = 1.
    calib = daterange("20240601", "20240630")
    before = calib[-10:]
    after = daterange("20240701", "20240710")
    target = daterange("20240801", "20240805")

    diag_key = 0 * N_EVAL_CELLS + 0
    days = {}
    for d in calib:
        days[d] = DayFlows(d, np.array([diag_key], dtype=np.int64), np.array([100.0]))
    for d in after:
        days[d] = DayFlows(d, np.array([diag_key], dtype=np.int64), np.array([200.0]))

    keys = np.array([diag_key], dtype=np.int64)
    pred = interp_two_anchor(days, target, before, after, keys, calib, na_dates=(),
                             shape="tau_avg", shape_off="sqrt")

    assert pred[target[0]].val[0] == pytest.approx(100.0, rel=1e-9)
