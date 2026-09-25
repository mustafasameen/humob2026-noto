"""Known-answer tests for the Combined NRMSE metric."""
import numpy as np

from humob26.config import N_EVAL_CELLS, N_EVAL_OFFDIAG_PAIRS, NORM_DIAG, NORM_OFFDIAG
from humob26.metric import DayFlows, score


def test_score_single_diagonal_pair_dense():
    # One diagonal pair (cell 0 -> cell 0): observed 10, predicted 12.
    # Every other diagonal pair is implicitly 0 == 0 under the dense
    # convention, so the sum of squares is a single (12 - 10)^2 term.
    obs = {"20240101": DayFlows("20240101", np.array([0], dtype=np.int64), np.array([10.0]))}
    pred = {"20240101": DayFlows("20240101", np.array([0], dtype=np.int64), np.array([12.0]))}

    summary, per_day = score(obs, pred)

    expected_rmse_diag = np.sqrt((12.0 - 10.0) ** 2 / N_EVAL_CELLS)
    assert summary["rmse_diag_dense"] == expected_rmse_diag
    assert summary["rmse_offdiag_dense"] == 0.0
    assert summary["nrmse_diag_dense"] == expected_rmse_diag / NORM_DIAG
    assert summary["nrmse_offdiag_dense"] == 0.0
    assert summary["combined_dense"] == (expected_rmse_diag / NORM_DIAG) / 2
    assert len(per_day) == 1


def test_score_single_offdiagonal_pair_dense():
    # Key 1 = (o_lin=0, d_lin=1): an off-diagonal pair.
    obs = {"20240101": DayFlows("20240101", np.array([1], dtype=np.int64), np.array([4.0]))}
    pred = {"20240101": DayFlows("20240101", np.array([1], dtype=np.int64), np.array([1.0]))}

    summary, _ = score(obs, pred)

    expected_rmse_offdiag = np.sqrt((1.0 - 4.0) ** 2 / N_EVAL_OFFDIAG_PAIRS)
    assert summary["rmse_diag_dense"] == 0.0
    assert summary["rmse_offdiag_dense"] == expected_rmse_offdiag
    assert summary["combined_dense"] == (expected_rmse_offdiag / NORM_OFFDIAG) / 2


def test_score_perfect_prediction_is_zero():
    key = np.array([0, 1, N_EVAL_CELLS + 1], dtype=np.int64)   # two diag, one offdiag
    val = np.array([5.0, 3.0, 7.0])
    obs = {"20240101": DayFlows("20240101", key, val)}
    pred = {"20240101": DayFlows("20240101", key.copy(), val.copy())}

    summary, _ = score(obs, pred)
    for conv in ("dense", "union", "observed"):
        assert summary[f"combined_{conv}"] == 0.0


def test_score_averages_over_days():
    # Two days with different diagonal errors; Combined should be the mean
    # of the per-day RMSE, not the RMSE of the pooled errors.
    obs = {
        "20240101": DayFlows("20240101", np.array([0], dtype=np.int64), np.array([10.0])),
        "20240102": DayFlows("20240102", np.array([0], dtype=np.int64), np.array([10.0])),
    }
    pred = {
        "20240101": DayFlows("20240101", np.array([0], dtype=np.int64), np.array([12.0])),
        "20240102": DayFlows("20240102", np.array([0], dtype=np.int64), np.array([10.0])),
    }
    summary, per_day = score(obs, pred)
    day1 = np.sqrt(4.0 / N_EVAL_CELLS)
    day2 = 0.0
    assert summary["rmse_diag_dense"] == (day1 + day2) / 2
    assert len(per_day) == 2
