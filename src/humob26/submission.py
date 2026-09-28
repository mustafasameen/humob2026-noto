"""Building and writing a submission file.

A submission covers the whole evaluation-box-and-beyond grid, but only
pairs with both ends inside the evaluation box are actually scored. This
package therefore builds two things and combines them:

  full-grid build   `anchor` + RTS smoother modelled on the whole grid
                     (combine.build_fullgrid_prediction), which is the only
                     model that ever touches an out-of-box pair. Written
                     directly, rounded to six decimal places (matching the
                     precision the challenge itself publishes flows at).
  in-box model       whatever arm is being submitted for scored pairs (the
                     `final` arm by default), spliced into the full-grid
                     file's in-box entries at full precision.

The splice preserves every out-of-box entry byte for byte: it is read back
from the full-grid file, has its in-box entries removed, and has the
in-box model's predictions added back in, so a diff between the two files
touches only pairs with both ends in the box.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import EVAL_X, EVAL_Y, GRID_N_CELLS, N_EVAL_CELLS
from .data import full_grid_unlinear

_NX = EVAL_X[1] - EVAL_X[0] + 1


def eval_box_gid(linear_index):
    """Evaluation-box linear index -> grid id string 'y_x'."""
    return f"{linear_index // _NX + EVAL_Y[0]}_{linear_index % _NX + EVAL_X[0]}"


def in_eval_box_gid(gid):
    """Whether a 'y_x' grid id string falls inside the evaluation box."""
    if gid == "-1_-1":
        return False
    y, x = map(int, gid.split("_"))
    return EVAL_Y[0] <= y <= EVAL_Y[1] and EVAL_X[0] <= x <= EVAL_X[1]


def write_fullgrid_tsv(pred, target_dates, out_path):
    """Write a full-grid prediction, rounded to 6 decimal places (values
    that round to zero are dropped, matching the file's own precision)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for d in target_dates:
            f = pred[d]
            od = defaultdict(dict)
            o_lin, d_lin = np.divmod(f.key, GRID_N_CELLS)
            for ol, dl, v in zip(o_lin, d_lin, f.val):
                # Round first, then drop: a value below 5e-7 rounds to 0.0
                # and would otherwise be written as an explicit zero entry.
                rv = round(float(v), 6)
                if rv <= 0:
                    continue
                oy, ox = full_grid_unlinear(int(ol))
                dy, dx = full_grid_unlinear(int(dl))
                od[f"{oy}_{ox}"][f"{dy}_{dx}"] = rv
            fh.write(f"{d}\t{dict(od)}\n")


def read_tsv_lines(path):
    """A submission file's raw (date, payload_str) lines, unparsed."""
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n").split("\t", 1) for ln in fh]


def splice_inbox(model, base_lines, out_path):
    """Write `base_lines` (typically the full-grid file's own lines) with
    every in-box entry replaced by `model`'s predictions, at full
    precision. Entries with an end outside the box are copied unchanged.

    Returns the parsed {date: od_dict} that was written, so a caller can
    verify the write by re-parsing it.
    """
    raw = {d: ast.literal_eval(js) for d, js in base_lines}
    for d in raw:
        od = raw[d]
        for o in list(od):
            if in_eval_box_gid(o):
                for dest in [k for k in od[o] if in_eval_box_gid(k)]:
                    del od[o][dest]
                if not od[o]:
                    del od[o]
        for k, v in zip(model[d].key, model[d].val):
            o, dest = divmod(int(k), N_EVAL_CELLS)
            od.setdefault(eval_box_gid(o), {})[eval_box_gid(dest)] = float(v)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for d, _ in base_lines:
            fh.write(f"{d}\t{dict(raw[d])}\n")
    return raw


def verify_inbox_written(out_path, model, target_dates):
    """Re-load `out_path` (evaluation-box view) and check it returns
    exactly `model`'s keys and values for every target date."""
    from .data import load_eval_box
    back, _ = load_eval_box(out_path)
    return all(
        np.array_equal(back[d].key, model[d].key) and np.array_equal(back[d].val, model[d].val)
        for d in target_dates
    )


def verify_outbox_unchanged(out_path, base_lines, target_dates):
    """Count entries with an end outside the box that differ between the
    written file and `base_lines`. Zero means the splice touched only
    in-box pairs."""
    base_raw = {d: ast.literal_eval(js) for d, js in base_lines}
    new_raw = {d: ast.literal_eval(js) for d, js in read_tsv_lines(out_path)}
    diffs = 0
    for d in target_dates:
        for o, dd in base_raw[d].items():
            for dest, v in dd.items():
                if not (in_eval_box_gid(o) and in_eval_box_gid(dest)):
                    diffs += new_raw[d].get(o, {}).get(dest) != v
        for o, dd in new_raw[d].items():
            for dest in dd:
                if not (in_eval_box_gid(o) and in_eval_box_gid(dest)):
                    diffs += dest not in base_raw[d].get(o, {})
    return diffs
