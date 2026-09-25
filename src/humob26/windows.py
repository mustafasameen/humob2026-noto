"""Validation windows: gap-shaped slices of the observed record used to
score every arm the same way the real prediction gap will be scored.

Since the real target period (`test_gap`) can only be scored once
predictions are submitted, every other window here carves a 58-to-60 day
span out of the OBSERVED record, hides it, and reconstructs it from a
before-anchor and an after-anchor exactly as the real gap will be
reconstructed -- so that a window's score is a genuine estimate of test
performance and not just a fit statistic.

Four windows reuse a natural adjacent-anchor geometry already present in
the record (a trending before-anchor, a discontinuity in the target, or an
anchor pair immediately adjacent to the target). Three "stable" windows and
nine "rolling" windows are carved out of the long, undisturbed Apr-Oct 2024
stretch with a fixed 10-day-before / 30-day-after anchor rule, so that
their only source of difficulty is calendar position, not a shock.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .calendar import daterange, to_date
from .config import (ANCHOR_AFTER_END, ANCHOR_AFTER_START, ANCHOR_BEFORE_DAYS,
                     ANCHOR_BEFORE_END, ANCHOR_BEFORE_START, EXCLUDED_TEST_DAYS,
                     GAP_END, GAP_START)


@dataclass(frozen=True)
class Window:
    name: str
    geometry: str
    target: tuple
    before: tuple
    after: tuple


# The four windows built on a naturally occurring anchor geometry rather
# than a fixed before/after rule.
_CANONICAL = {
    "may_jun":  (daterange("20240501", "20240630"), daterange("20240401", "20240430"),
                daterange("20240701", "20240731"), "adjacent"),
    "late_jan": (daterange("20240125", "20240131"), daterange("20240115", "20240124"),
                daterange("20240401", "20240430"), "adjacent,trend"),
    "april":    (daterange("20240401", "20240430"), daterange("20240101", "20240131"),
                daterange("20240501", "20240630"), "trend"),
    "january":  (daterange("20240101", "20240131"), daterange("20231101", "20231231"),
                daterange("20240401", "20240430"), "shock"),
}

_STABLE = (
    ("stable_spring", "20240415", "20240613"),
    ("stable_summer", "20240614", "20240812"),
    ("stable_late",   "20240813", "20241011"),
)

# Nine rolling 60-day windows across the undisturbed Apr-Oct 2024 stretch,
# each with its own fixed-offset before/after anchor (see `cal`).
ROLL_STARTS = ("20240429", "20240513", "20240527", "20240610", "20240624",
              "20240708", "20240722", "20240805", "20240819")
ROLL_LENGTH_DAYS = 59       # 60-day window: start .. start + 59


def cal(start, n, step):
    """`n` calendar dates stepping from `start` by `step` days each,
    starting at `start + step` (never including `start` itself)."""
    d0 = to_date(start)
    return [(d0 + dt.timedelta(days=step * (k + 1))).strftime("%Y%m%d") for k in range(n)]


def iter_validation_windows(days, na, anchor_before_days=ANCHOR_BEFORE_DAYS):
    """Yield every `Window` used for validation (16 total): the four
    canonical windows, the three stable windows, and the nine rolling
    windows, each with target/before/after restricted to days present in
    `days` and not flagged not-a-number in `na`. `anchor_before_days`
    overrides the default 10-day truncation, for a sensitivity run."""
    for name, (target, before, after, geom) in _CANONICAL.items():
        # The before-anchor is truncated to the days adjacent to the gap
        # unless the window's own anchor already is that (late_jan) or the
        # geometry is a discontinuity, where the whole anchor is kept.
        bw = (sorted(before) if ("shock" in geom or name == "late_jan")
              else sorted(before)[-anchor_before_days:])
        yield Window(name, geom, tuple(sorted(d for d in target if d in days and d not in na)),
                    tuple(bw), tuple(sorted(after)))

    for name, s, e in _STABLE:
        T = [d for d in daterange(s, e) if d not in na]
        bw = sorted(d for d in cal(T[0], anchor_before_days, -1) if d in days and d not in na)
        aw = sorted(d for d in cal(T[-1], 30, +1) if d in days and d not in na)
        yield Window(name, "stable", tuple(sorted(d for d in T if d in days)), tuple(bw), tuple(aw))

    for start in ROLL_STARTS:
        end = (to_date(start) + dt.timedelta(days=ROLL_LENGTH_DAYS)).strftime("%Y%m%d")
        T = [d for d in daterange(start, end) if d in days and d not in na]
        bw = sorted(d for d in cal(T[0], anchor_before_days, -1) if d in days and d not in na)
        aw = sorted(d for d in cal(T[-1], 30, +1) if d in days and d not in na)
        assert len(aw) >= anchor_before_days, start
        yield Window(f"roll_{start[4:]}", "stable", tuple(T), tuple(bw), tuple(aw))


def test_gap_window(anchor_before_days=ANCHOR_BEFORE_DAYS):
    """The real, held-out prediction gap: Feb 1 - Mar 31 2024, excluding the
    two dates the challenge itself does not score."""
    target = tuple(d for d in daterange(GAP_START, GAP_END) if d not in EXCLUDED_TEST_DAYS)
    before = tuple(daterange(ANCHOR_BEFORE_START, ANCHOR_BEFORE_END)[-anchor_before_days:])
    after = tuple(daterange(ANCHOR_AFTER_START, ANCHOR_AFTER_END))
    return Window("test_gap", "shock", target, before, after)


# Two extra early-January windows, built the same way as "late_jan" (whose
# before-anchor is kept whole rather than truncated) but shifted a week and
# two weeks earlier -- closer to the one known discontinuity in the record.
# Used only by the forecaster "shock-origin" comparison, not the main
# validation set.
_SHOCK_ORIGIN_EXTRA = {
    "jan18_24": (daterange("20240118", "20240124"), daterange("20240108", "20240117")),
    "jan11_17": (daterange("20240111", "20240117"), daterange("20240101", "20240110")),
}
SHOCK_ORIGIN_WINDOW_NAMES = ("late_jan",) + tuple(_SHOCK_ORIGIN_EXTRA)

# Nine more windows built exactly like the rolling ones but starting a week
# earlier. No modelling choice used them: they check that the rolling-window
# results do not depend on where those windows happen to start.
SHIFT_STARTS = ("20240422", "20240506", "20240520", "20240603", "20240617",
                "20240701", "20240715", "20240729", "20240812")
SHIFT_WINDOW_NAMES = tuple(f"shift_{s[4:]}" for s in SHIFT_STARTS)


def _rolling_like(name, start, days, na, anchor_before_days=ANCHOR_BEFORE_DAYS):
    end = (to_date(start) + dt.timedelta(days=ROLL_LENGTH_DAYS)).strftime("%Y%m%d")
    T = [d for d in daterange(start, end) if d in days and d not in na]
    bw = sorted(d for d in cal(T[0], anchor_before_days, -1) if d in days and d not in na)
    aw = sorted(d for d in cal(T[-1], 30, +1) if d in days and d not in na)
    assert len(aw) >= anchor_before_days, start
    return Window(name, "stable", tuple(T), tuple(bw), tuple(aw))

ROLL_WINDOW_NAMES = tuple(f"roll_{s[4:]}" for s in ROLL_STARTS)
VALIDATION_WINDOW_NAMES = tuple(_CANONICAL) + tuple(n for n, _, _ in _STABLE) + ROLL_WINDOW_NAMES
ALL_WINDOW_NAMES = VALIDATION_WINDOW_NAMES + ("test_gap",)


def get_window(name, days, na, anchor_before_days=ANCHOR_BEFORE_DAYS):
    """Look up one window by name: any of VALIDATION_WINDOW_NAMES,
    "test_gap", or a SHOCK_ORIGIN_WINDOW_NAMES-only extra."""
    if name == "test_gap":
        return test_gap_window(anchor_before_days)
    if name in SHIFT_WINDOW_NAMES:
        return _rolling_like(name, SHIFT_STARTS[SHIFT_WINDOW_NAMES.index(name)], days, na, anchor_before_days)
    if name in _SHOCK_ORIGIN_EXTRA:
        target, before = _SHOCK_ORIGIN_EXTRA[name]
        return Window(name, "shock",
                     tuple(sorted(d for d in target if d in days and d not in na)),
                     tuple(sorted(before)),
                     tuple(sorted(daterange(ANCHOR_AFTER_START, ANCHOR_AFTER_END))))
    for w in iter_validation_windows(days, na, anchor_before_days):
        if w.name == name:
            return w
    raise KeyError(f"unknown window {name!r}; choose from {ALL_WINDOW_NAMES + tuple(_SHOCK_ORIGIN_EXTRA)}")

