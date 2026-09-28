import numpy as np
import pytest

from humob26 import recovery


def test_shares_count_all_rows_in_denominator_but_only_diagonal_in_numerator():
    # 10 inside-cell + 20 other in-box cell + 5 between-cell + 50 outside
    # bucket + 7 outbound + 8 inbound + 100 elsewhere in grid = total 200.
    arr = dict(oy=np.array([35, 35, 35, -1, 35, -1, 1]),
               ox=np.array([30, 31, 30, -1, 30, -1, 1]),
               dy=np.array([35, 35, 35, -1, -1, 35, 1]),
               dx=np.array([30, 31, 31, -1, -1, 30, 1]),
               count=np.array([10., 20., 5., 50., 7., 8., 100.]))
    total, groups = recovery.aggregate_day('20240122', arr, {0: 'wajima', 1: 'suzu'})
    assert total == 200
    assert groups['wajima'] / total == .05
    assert groups['suzu'] / total == .1
    assert groups['outside_grid'] / total == .25
    assert sum(groups.values()) == 80


def test_ratio_uses_anchor_means_and_resamples_days():
    b = np.array([[1., 2.], [1., 2.]])
    a = np.array([[2., 1.], [2., 1.], [2., 1.]])
    estimate, lo, hi, bad = recovery.bootstrap_ratios(b, a, reps=4000)
    np.testing.assert_equal(estimate, [2., .5])
    np.testing.assert_equal(lo, estimate)
    np.testing.assert_equal(hi, estimate)
    assert not bad.any()


def test_bootstrap_matches_independent_day_resampling():
    b = np.array([[1., 3.], [2., 7.], [4., 9.]])
    a = np.array([[3., 5.], [4., 6.]])
    est, lo, hi, _ = recovery.bootstrap_ratios(b, a, reps=100, seed=12)
    rng = np.random.default_rng(12)
    before_indices = rng.integers(3, size=(100, 3))
    after_indices = rng.integers(2, size=(100, 2))
    manual = np.array([a[j].mean(axis=0) / b[i].mean(axis=0)
                       for i, j in zip(before_indices, after_indices)])
    np.testing.assert_allclose(est, a.mean(axis=0) / b.mean(axis=0))
    np.testing.assert_allclose(lo, np.percentile(manual, 2.5, axis=0))
    np.testing.assert_allclose(hi, np.percentile(manual, 97.5, axis=0))


def test_zero_denominator_is_explicit():
    est, lo, hi, bad = recovery.bootstrap_ratios(np.zeros((2, 1)), np.ones((2, 1)), reps=10)
    assert np.isnan(est[0]) and np.isnan(lo[0]) and np.isnan(hi[0])
    assert bad[0] == 10


@pytest.mark.parametrize('before,after,shares,expected', [
    (.1, .3, [.19, .21, .3], ('20240203', 'reached')),
    (.3, .1, [.25, .19, .1], ('20240203', 'reached')),
    (0., 1., [.5, .1, .9], ('20240201', 'reached')),
    (0., 1., [.1, .2, .3], (None, 'not_reached')),
    (.1, .1, [.1, .2, .3], (None, 'no_change')),
    (0., 1., [np.nan, np.nan, .6], ('20240204', 'reached')),
])
def test_halfway_direction_and_first_available_day(before, after, shares, expected):
    assert recovery.first_halfway(before, after, ['20240201', '20240203', '20240204'], shares) == expected


def test_halfway_orders_dates_and_does_not_require_sustained_crossing():
    assert recovery.first_halfway(0, 1, ['20240204', '20240201', '20240203'], [.7, .6, .2]) == ('20240201', 'reached')
    assert recovery.first_halfway(0, 1, [], []) == (None, 'no_reconstruction')


def test_missing_raw_day_and_nonconstant_totals(tmp_path):
    p = tmp_path / 'raw.tsv'
    p.write_text("20240122\t{'35_30': {'35_30': 10}, '-1_-1': {'-1_-1': 90}}\n20240123\tNA\n")
    obs, totals, missing, fixed = recovery.read_observed(p, {0: 'wajima'})
    assert len(obs) == 1 and missing == ['20240123'] and fixed == 100
    p.write_text(p.read_text() + "20240124\t{'35_30': {'35_30': 20}, '-1_-1': {'-1_-1': 90}}\n")
    with pytest.raises(ValueError, match='not constant'):
        recovery.read_observed(p, {0: 'wajima'})
