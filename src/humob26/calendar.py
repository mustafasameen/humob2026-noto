"""Calendar utilities: dates, weekdays, and the holiday calendar.

The prediction gap sits inside a single winter-to-spring span, so every
date here is a plain "YYYYMMDD" string and every calendar computation is
local to that span. Holidays matter because a public holiday on a working
weekday behaves like a weekend for travel purposes, which a plain
day-of-week model cannot see on its own.
"""
from __future__ import annotations

import datetime as dt

# National holidays spanning the observed record.  Only the ones that fall
# on a working weekday change anything: a holiday that already falls on a
# weekend is already captured by the weekday factor.
HOLIDAYS = {
    "20231103", "20231123",
    "20240101", "20240102", "20240103", "20240108",
    "20240211", "20240212", "20240223", "20240320",
    "20240429", "20240503", "20240504", "20240505", "20240506",
    "20240715", "20240811", "20240812", "20240813", "20240814", "20240815",
    "20240916", "20240922", "20240923", "20241014",
}

# Multi-day holiday blocks (Golden Week, Obon) are left out of the
# isolated-holiday multiplier and keep their plain weekday factor: a
# multi-day travel period is a different mechanism from a single day off, so
# pooling the two would mix two effects the model has no reason to assume
# are the same size, or even the same sign.
BLOCK_HOLIDAYS = {
    "20240429", "20240503", "20240504", "20240505", "20240506",
    "20240811", "20240812", "20240813", "20240814", "20240815",
}

# Days excluded when calibrating the holiday multiplier and day-type
# factors: the New Year window and the earthquake day the dataset begins
# its post-event period on (see the data citation in README.md), neither of
# which is informative about an ordinary weekday/holiday effect.
HOLIDAY_CALIB_EXCLUDE = {
    "20231229", "20231230", "20231231",
    "20240101", "20240102", "20240103",
    "20240104", "20240105",
}


def to_date(s: str) -> dt.date:
    """'20240213' -> date(2024, 2, 13)."""
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def dow(s: str) -> int:
    """Day of week, Monday=0 .. Sunday=6."""
    return to_date(s).weekday()


def daterange(start: str, end: str):
    """Every date from `start` to `end` inclusive, as 'YYYYMMDD' strings."""
    a, b = to_date(start), to_date(end)
    return [(a + dt.timedelta(days=i)).strftime("%Y%m%d")
            for i in range((b - a).days + 1)]


def is_holiday(d: str) -> bool:
    return d in HOLIDAYS


def is_isolated_holiday(d: str) -> bool:
    """A working-weekday holiday that is not part of a multi-day block."""
    return dow(d) < 5 and d in HOLIDAYS and d not in BLOCK_HOLIDAYS


def day_type(d: str) -> int:
    """Plain weekday 0..6. Holidays are handled by a separate multiplier,
    not folded into the day type itself."""
    return dow(d)
