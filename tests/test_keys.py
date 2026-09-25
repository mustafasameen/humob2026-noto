"""Round-trip tests for the grid-id / linear-index / OD-key encodings."""
import itertools

from humob26.config import EVAL_X, EVAL_Y, GRID_NX, GRID_NY, N_EVAL_CELLS
from humob26.data import full_grid_linear, full_grid_unlinear
from humob26.submission import eval_box_gid, in_eval_box_gid

N_EVAL_X = EVAL_X[1] - EVAL_X[0] + 1


def test_full_grid_linear_round_trip_corners():
    for y, x in itertools.product((1, GRID_NY, 37), (1, GRID_NX, 52)):
        lin = full_grid_linear(y, x)
        y2, x2 = full_grid_unlinear(lin)
        assert (int(y2), int(x2)) == (y, x)


def test_full_grid_linear_is_row_major_and_zero_based():
    assert full_grid_linear(1, 1) == 0
    assert full_grid_linear(1, GRID_NX) == GRID_NX - 1
    assert full_grid_linear(2, 1) == GRID_NX


def test_eval_box_gid_round_trip_every_cell():
    for o_lin in range(N_EVAL_CELLS):
        gid = eval_box_gid(o_lin)
        assert in_eval_box_gid(gid)
        y, x = map(int, gid.split("_"))
        assert EVAL_Y[0] <= y <= EVAL_Y[1]
        assert EVAL_X[0] <= x <= EVAL_X[1]
        # Reconstruct the linear index the same way the OD key is built
        # from (y, x) and check it matches what we started from.
        recovered = (y - EVAL_Y[0]) * N_EVAL_X + (x - EVAL_X[0])
        assert recovered == o_lin


def test_in_eval_box_gid_rejects_outside_and_oob():
    assert not in_eval_box_gid("-1_-1")
    assert not in_eval_box_gid("1_1")             # outside the box
    assert not in_eval_box_gid(f"{EVAL_Y[0] - 1}_{EVAL_X[0]}")
    assert in_eval_box_gid(f"{EVAL_Y[0]}_{EVAL_X[0]}")
    assert in_eval_box_gid(f"{EVAL_Y[1]}_{EVAL_X[1]}")


def test_od_key_encoding_round_trip():
    # key = o_lin * N_EVAL_CELLS + d_lin, as every DayFlows key is built.
    for o_lin, d_lin in ((0, 0), (1, 0), (0, 1), (N_EVAL_CELLS - 1, N_EVAL_CELLS - 1)):
        key = o_lin * N_EVAL_CELLS + d_lin
        o2, d2 = divmod(key, N_EVAL_CELLS)
        assert (o2, d2) == (o_lin, d_lin)
        assert (o2 == d2) == (o_lin == d_lin)      # diagonal-ness survives
