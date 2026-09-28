import numpy as np

from humob26 import metric, windows
from humob26.baselines.methods import CALENDAR, masked_matrix, linear


def fixture_days():
    k = np.array([0, 1], dtype=np.int64)
    return {d: metric.DayFlows(d, k, np.array(v, dtype=float)) for d, v in {
        '20231101': [0, 10], '20231102': [10000, 10000],
        '20231103': [20000, 20000], '20231104': [8, 2]}.items()}


def test_calendar_mask_hides_targets_and_keeps_observed_zero():
    days = fixture_days()
    w = windows.Window('test', 'test', ('20231102', '20231103'), ('20231101',), ('20231104',))
    keys, matrix = masked_matrix(days, set(), w)
    assert matrix.shape == (2, 366)
    assert matrix[0, 0] == 0
    assert np.isnan(matrix[:, 1:3]).all()
    assert np.isnan(matrix[:, CALENDAR.index('20240201')]).all()
    days['20231102'].val[:] = -99999
    _, changed = masked_matrix(days, set(), w)
    np.testing.assert_equal(matrix, changed)


def test_linear_uses_plain_means_and_calendar_offsets():
    days = fixture_days()
    w = windows.Window('test', 'test', ('20231102', '20231103'), ('20231101',), ('20231104',))
    pred = linear(days, set(), w)
    np.testing.assert_equal(pred['20231102'].key, [1])
    np.testing.assert_allclose(pred['20231102'].val, [10])
    np.testing.assert_allclose(pred['20231103'].val, [8, 2])


def test_trmf_zero_is_observed():
    from humob26.baselines._trmf import TRMF
    rng = np.random.default_rng(5)
    matrix = np.array([[0., 3., np.nan, 4., 0., 3.], [3., 2., np.nan, 1., 2., 1.]])
    init = {'W': rng.random((2, 2)), 'X': rng.random((6, 2)), 'theta': rng.random((1, 2))}
    hyper = dict(lambda_w=500, lambda_x=500, lambda_theta=500, eta=.03)
    a = TRMF(matrix, {k:v.copy() for k,v in init.items()}, hyper, np.array([1]), 2)
    missing = matrix.copy(); missing[0, 0] = np.nan
    b = TRMF(missing, {k:v.copy() for k,v in init.items()}, hyper, np.array([1]), 2)
    assert np.isfinite(a).all()
    assert not np.array_equal(a, b)
    assert matrix[0, 0] == 0 and np.isnan(matrix[0, 2])


def test_btmf_nan_mask_and_short_sampling():
    from humob26.baselines._btmf import BTMF
    np.random.seed(1000)
    matrix = np.arange(60, dtype=float).reshape(5, 12)
    matrix[:, 4:6] = np.nan
    original = matrix.copy()
    init = {'W': .01 * np.random.randn(5, 2), 'X': .01 * np.random.randn(12, 2)}
    filled, *_ = BTMF(matrix, init, 2, np.array([1, 2]), 2, 2)
    assert filled.shape == matrix.shape and np.isfinite(filled).all()
    assert (filled >= 0).all()
    np.testing.assert_equal(matrix, original)
