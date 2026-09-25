"""Round-trip tests for the submission writers."""
import numpy as np

from humob26.config import EVAL_X, EVAL_Y, GRID_N_CELLS, N_EVAL_CELLS
from humob26.data import full_grid_linear, load_full_grid
from humob26.metric import DayFlows
from humob26.submission import (read_tsv_lines, splice_inbox, verify_inbox_written,
                                verify_outbox_unchanged, write_fullgrid_tsv)

NX = EVAL_X[1] - EVAL_X[0] + 1


def test_write_fullgrid_tsv_round_trips_through_the_loader(tmp_path):
    d0, d1 = "20240201", "20240202"
    o = full_grid_linear(EVAL_Y[0], EVAL_X[0])           # a cell inside the box, for realism
    dd = full_grid_linear(EVAL_Y[0], EVAL_X[0] + 1)
    key = np.array([o * GRID_N_CELLS + dd], dtype=np.int64)
    pred = {
        d0: DayFlows(d0, key, np.array([12.3456789])),
        d1: DayFlows(d1, key, np.array([0.0000001])),    # rounds to 0.0 and must be dropped
    }
    out = tmp_path / "fullgrid.tsv"
    write_fullgrid_tsv(pred, [d0, d1], out)

    back = load_full_grid(out)
    assert np.array_equal(back[d0].key, key)
    assert back[d0].val[0] == round(12.3456789, 6)
    # The near-zero value rounds to 0.0 and is dropped, leaving an empty
    # `{}` payload for d1 -- indistinguishable on reload from a
    # not-a-number day, so `load_full_grid` (correctly) omits it entirely.
    assert d1 not in back


def test_splice_inbox_replaces_only_in_box_entries(tmp_path):
    date = "20240201"
    # A base file with one out-of-box entry and one in-box entry that the
    # splice must remove.
    base_od = {
        f"{1}_{1}": {f"{1}_{2}": 3.5},
        f"{EVAL_Y[0]}_{EVAL_X[0]}": {f"{EVAL_Y[0]}_{EVAL_X[0] + 1}": 999.0},
    }
    base_path = tmp_path / "base.tsv"
    base_path.write_text(f"{date}\t{base_od}\n")
    base_lines = read_tsv_lines(base_path)

    # The model's own in-box prediction for the same pair, at full precision.
    model_key = np.array([0 * N_EVAL_CELLS + 1], dtype=np.int64)   # o_lin=0, d_lin=1 inside the box
    model = {date: DayFlows(date, model_key, np.array([42.123456789]))}

    out_path = tmp_path / "final.tsv"
    splice_inbox(model, base_lines, out_path)

    assert verify_inbox_written(out_path, model, [date])
    assert verify_outbox_unchanged(out_path, base_lines, [date]) == 0

    # The out-of-box entry must be byte-for-byte the same value; the old
    # in-box value (999.0) must be gone, replaced by the model's value.
    text = out_path.read_text()
    assert "3.5" in text
    assert "999.0" not in text
    assert "42.123456789" in text
