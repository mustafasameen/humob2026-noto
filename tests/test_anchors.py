"""Empirical-Bayes day-type shrinkage: a noisy cell should be pulled toward
the pooled (cross-cell) shape much more than a clean one."""
import numpy as np

from humob26.anchors import daytype_factors_eb
from humob26.calendar import daterange, dow
from humob26.metric import DayFlows

TRUE_SHAPE = np.array([1.2, 1.15, 1.05, 1.0, 1.1, 0.6, 0.55])   # Mon .. Sun
CLEAN_BASES = [200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0]
NOISY_BASE = 250.0
NOISY_KEY = len(CLEAN_BASES)


def _synthetic_days():
    calib = daterange("20240603", "20240728")     # exactly 8 full weeks, Mon .. Sun
    start_week = {d: (i // 7) for i, d in enumerate(calib)}
    rng = np.random.default_rng(12345)
    eps = np.clip(rng.normal(0.0, 0.8, size=(8, 7)), -0.85, 5.0)   # (week, weekday)

    keys = np.arange(NOISY_KEY + 1, dtype=np.int64)
    days = {}
    for d in calib:
        k = dow(d)
        w = start_week[d]
        vals = [base * TRUE_SHAPE[k] for base in CLEAN_BASES]
        vals.append(NOISY_BASE * TRUE_SHAPE[k] * (1.0 + eps[w, k]))
        days[d] = DayFlows(d, keys.copy(), np.array(vals))
    return days, calib, keys


def test_eb_shrinkage_pulls_a_noisy_cell_toward_the_pooled_shape():
    days, calib, keys = _synthetic_days()

    shrunk = daytype_factors_eb(days, calib, keys, min_obs=3)
    unshrunk = daytype_factors_eb(days, calib, keys, min_obs=3, lam_const=0.0)

    true_shape_normalised = TRUE_SHAPE / TRUE_SHAPE.mean()
    dist_shrunk = np.abs(shrunk[:, NOISY_KEY] - true_shape_normalised).sum()
    dist_unshrunk = np.abs(unshrunk[:, NOISY_KEY] - true_shape_normalised).sum()

    # The noisy cell's own weekday profile is a poor, high-variance estimate
    # of the shape every cell actually shares; shrinking it toward the
    # pooled shape should land much closer to the truth.
    assert dist_shrunk < 0.5 * dist_unshrunk

    # A cell with zero sampling variance should barely be shrunk at all: its
    # EB and unshrunk factors should stay close to each other.
    clean_key = 0
    assert np.abs(shrunk[:, clean_key] - unshrunk[:, clean_key]).max() < 0.05


def test_eb_shrinkage_matches_fixed_shrink_when_lambda_is_forced_constant():
    days, calib, keys = _synthetic_days()
    lam = 0.35
    a = daytype_factors_eb(days, calib, keys, min_obs=3, lam_const=lam)
    # Rebuild the same fixed-shrink construction by hand and check it agrees:
    # a constant lambda is exactly what `anchors.daytype_factors` computes.
    from humob26.anchors import daytype_factors
    b = daytype_factors(days, calib, keys, shrink=lam)
    assert np.allclose(a, b)
