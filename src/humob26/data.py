"""Reading the raw OD dataset into the two shapes the rest of the package uses.

The raw file is one line per day: `YYYYMMDD \\t {'oy_ox': {'dy_dx': count, ...}, ...}`.
The payload is a Python dict literal whose keys are only digits/underscore/
minus and whose values are ints, so swapping the single quotes for double
quotes turns it into valid JSON and lets the (much faster) C JSON decoder
read it instead of `ast.literal_eval`.

A day with no reliable measurement is written as a literal marker instead
of a dict; `iter_days` yields `None` for it, so a caller can tell "no
travel recorded" apart from "not measured".

Two downstream shapes are built from the same parse:
  eval box    only OD pairs with BOTH ends inside the scored evaluation
              box, keyed for the metric and every model in this package.
  full grid   every OD pair on the whole 100 x 70 grid, keyed the same way
              but needed only to fill in the unscored entries of a
              submission file (predictions with an end outside the box).
Both support an on-disk cache (a compressed .npz) so repeated runs skip
the JSON parse.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EVAL_X, EVAL_Y, GRID_NX, GRID_N_CELLS, N_EVAL_CELLS, OUT_OF_BOUNDS_ID
from .metric import DayFlows

NA_MARKERS = {"", "NA", "N/A", "nan", "NaN", "None", "{}", "null"}


def _parse_payload(payload: str):
    payload = payload.strip()
    if payload in NA_MARKERS:
        return None
    return json.loads(payload.replace("'", '"'))


def iter_days(path, keep_oob: bool = False):
    """Yield (date, arrays) per line of the raw dataset.

    `arrays` is None for a not-a-number day, else a dict of equal-length
    int arrays oy, ox, dy, dx and a float64 `count`, covering the WHOLE
    grid. Out-of-bounds "-1_-1" rows are dropped unless `keep_oob`, in
    which case they arrive with y=x=-1.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            date, _, payload = line.partition("\t")
            date = date.strip()
            if not date.isdigit():          # header row
                continue
            try:
                od = _parse_payload(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno} ({date}): unparseable payload") from exc
            if od is None:
                yield date, None
                continue

            oy, ox, dy, dx, cnt = [], [], [], [], []
            for o_gid, dests in od.items():
                if o_gid == OUT_OF_BOUNDS_ID and not keep_oob:
                    continue
                oys, _, oxs = o_gid.partition("_")
                oyi, oxi = int(oys), int(oxs)
                for d_gid, c in dests.items():
                    if d_gid == OUT_OF_BOUNDS_ID and not keep_oob:
                        continue
                    dys, _, dxs = d_gid.partition("_")
                    oy.append(oyi); ox.append(oxi)
                    dy.append(int(dys)); dx.append(int(dxs))
                    cnt.append(c)
            yield date, {
                "oy": np.asarray(oy, np.int16), "ox": np.asarray(ox, np.int16),
                "dy": np.asarray(dy, np.int16), "dx": np.asarray(dx, np.int16),
                "count": np.asarray(cnt, np.float64),
            }


def to_eval_box(date, arr) -> DayFlows:
    """Restrict a day to pairs with both endpoints in the evaluation box and
    pack into the sparse key/value form the metric consumes."""
    if arr is None:
        return DayFlows(date, np.empty(0, np.int64), np.empty(0, float))
    m = (
        (arr["oy"] >= EVAL_Y[0]) & (arr["oy"] <= EVAL_Y[1])
        & (arr["ox"] >= EVAL_X[0]) & (arr["ox"] <= EVAL_X[1])
        & (arr["dy"] >= EVAL_Y[0]) & (arr["dy"] <= EVAL_Y[1])
        & (arr["dx"] >= EVAL_X[0]) & (arr["dx"] <= EVAL_X[1])
    )
    nx = EVAL_X[1] - EVAL_X[0] + 1
    o_lin = (arr["oy"][m].astype(np.int64) - EVAL_Y[0]) * nx + (arr["ox"][m] - EVAL_X[0])
    d_lin = (arr["dy"][m].astype(np.int64) - EVAL_Y[0]) * nx + (arr["dx"][m] - EVAL_X[0])
    key = o_lin * N_EVAL_CELLS + d_lin
    val = arr["count"][m]

    order = np.argsort(key, kind="stable")
    key, val = key[order], val[order]
    if key.size and np.any(np.diff(key) == 0):      # defensive: collapse dupes
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.bincount(inv, weights=val)
        key = uniq
    return DayFlows(date, key, val)


def load_eval_box(path, cache: str | Path | None = None):
    """{date: DayFlows} over the evaluation box, plus the set of not-a-number
    dates. Reads `cache` if it exists, else parses `path` and, if `cache`
    is given, writes it for next time."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        dates = [str(d) for d in z["dates"]]
        na = {str(d) for d in z["na_dates"]}
        days = {}
        for i, d in enumerate(dates):
            days[d] = DayFlows(d, z[f"k_{i}"], z[f"v_{i}"])
        return days, na

    days, na = {}, set()
    for date, arr in iter_days(path):
        if arr is None:
            na.add(date)
        days[date] = to_eval_box(date, arr)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        payload = {"dates": np.array(sorted(days), dtype="U8"),
                   "na_dates": np.array(sorted(na), dtype="U8")}
        for i, d in enumerate(sorted(days)):
            payload[f"k_{i}"] = days[d].key
            payload[f"v_{i}"] = days[d].val
        np.savez_compressed(cache, **payload)
    return days, na


# ---------------------------------------------------------------------------
# Full-grid loading: needed only to fill in submission entries whose origin
# or destination falls outside the evaluation box.
# ---------------------------------------------------------------------------


def full_grid_linear(y, x):
    """Row-major linear id of a full-grid cell, 0 .. GRID_N_CELLS - 1."""
    return (np.asarray(y) - 1) * GRID_NX + (np.asarray(x) - 1)


def full_grid_unlinear(i):
    """Inverse of `full_grid_linear`: linear id -> (y, x)."""
    return i // GRID_NX + 1, i % GRID_NX + 1


def load_full_grid(path, cache: str | Path | None = None):
    """{date: DayFlows} over the WHOLE grid, key = o_lin * GRID_N_CELLS + d_lin.

    Not-a-number days are simply absent from the result (unlike
    `load_eval_box`, which keeps an empty entry for them); callers filter
    training days against the known NA list themselves.
    """
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        dates = [str(d) for d in z["dates"]]
        return {d: DayFlows(d, z[f"k_{i}"], z[f"v_{i}"]) for i, d in enumerate(dates)}

    days = {}
    for date, arr in iter_days(path):
        if arr is None:
            continue
        o = full_grid_linear(arr["oy"].astype(np.int64), arr["ox"].astype(np.int64))
        d = full_grid_linear(arr["dy"].astype(np.int64), arr["dx"].astype(np.int64))
        k = o * GRID_N_CELLS + d
        order = np.argsort(k, kind="stable")
        days[date] = DayFlows(date, k[order], arr["count"][order].astype(float))

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        payload = {"dates": np.array(sorted(days), dtype="U8")}
        for i, d in enumerate(sorted(days)):
            payload[f"k_{i}"] = days[d].key
            payload[f"v_{i}"] = days[d].val
        np.savez_compressed(cache, **payload)
    return days


# ---------------------------------------------------------------------------
# Shaping a {date: DayFlows} dict into the dense arrays the models use.
# ---------------------------------------------------------------------------


def build_matrix(days, dates, keys):
    """(len(dates) x len(keys)) matrix; NaN where the day is unobserved,
    zero for an observed day's pair that is simply absent (no flow)."""
    keys = np.asarray(keys)
    out = np.full((len(dates), keys.size), np.nan)
    for i, d in enumerate(dates):
        f = days.get(d)
        if f is None or f.key.size == 0:
            continue
        idx = np.searchsorted(keys, f.key)
        ok = (idx < keys.size) & (keys[np.clip(idx, 0, keys.size - 1)] == f.key)
        row = np.zeros(keys.size)          # observed day: absent pair == 0
        row[idx[ok]] = f.val[ok]
        out[i] = row
    return out


def top_keys(days, dates, n=None, diag_only=False):
    """OD-pair keys ranked by total squared mass, which is what RMSE
    actually weights. With `n` given, keeps only the top `n` by that
    ranking (still returned in sorted key order); otherwise returns every
    key that ever appears with a non-zero value."""
    acc = {}
    for d in dates:
        f = days.get(d)
        if f is None:
            continue
        for k, v in zip(f.key, f.val):
            if diag_only and (k // N_EVAL_CELLS) != (k % N_EVAL_CELLS):
                continue
            acc[k] = acc.get(k, 0.0) + v * v
    keys = np.array(sorted(acc), dtype=np.int64)
    if n is None:
        return keys
    mass = np.array([acc[k] for k in keys])
    return np.sort(keys[np.argsort(mass)[::-1][:n]])
