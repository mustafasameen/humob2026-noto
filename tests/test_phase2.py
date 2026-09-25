"""Unit tests for the phase-2 additions that need no real dataset, forecast
files, or ML framework: settings overrides, the recovery-weight formula,
and the bake-off statistics."""
import numpy as np
import pytest

from humob26.anchors import average_recovery_weights, exponential_recovery_weights
from humob26.bakeoff import bonferroni_percentiles
from humob26.config import RECOVERY_WEIGHT_TABLE, Settings
from humob26.forecasters import available, load_all
from humob26.forecasters.chronos2 import sort_quantiles


def test_settings_parse_overrides_only_the_named_field():
    overrides = Settings.parse("rts_rank", "8")
    s = Settings(**overrides)
    assert s.rts_rank == 8
    assert s.anchor_rts_blend == Settings().anchor_rts_blend    # untouched


def test_settings_parse_bool_and_tuple():
    assert Settings.parse("eb_enabled", "false") == {"eb_enabled": False}
    assert Settings.parse("eb_enabled", "true") == {"eb_enabled": True}
    assert Settings.parse("tau_range", "4.0,30.0") == {"tau_range": (4.0, 30.0)}


def test_settings_parse_rejects_unknown_key():
    with pytest.raises(KeyError):
        Settings.parse("not_a_real_setting", "1")


def test_exponential_recovery_weight_endpoints():
    w = exponential_recovery_weights(tau=12.0, n_days=60, after_anchor_offset=60.0)
    assert w[0] == pytest.approx(1.0)
    assert 0.0 < w[-1] < 0.05          # a small positive residual, not exactly 0
    assert np.all(np.diff(w) <= 0)     # monotone non-increasing


def test_average_recovery_weights_matches_default_table_approximately():
    # See config.py: this is the formula the default table is built from,
    # confirmed to match it to about 1e-3 (not exact, since the table is
    # kept as a fixed input rather than recomputed at run time).
    w = average_recovery_weights(7.588, 26.88, n_grid=2000)
    assert len(w) == len(RECOVERY_WEIGHT_TABLE)
    assert np.abs(np.asarray(w) - np.asarray(RECOVERY_WEIGHT_TABLE)).max() < 2e-3


def test_bonferroni_percentiles_narrows_with_more_arms():
    lo1, hi1 = bonferroni_percentiles(1)
    lo5, hi5 = bonferroni_percentiles(5)
    assert lo1 == pytest.approx(2.5)
    assert hi1 == pytest.approx(97.5)
    assert lo5 < lo1 < 50 < hi1 < hi5   # more arms -> a wider (more conservative) interval


def test_sort_quantiles_fixes_a_crossed_triple_and_is_a_noop_when_already_ordered():
    median = np.array([[5.0, 5.0]])
    q10 = np.array([[7.0, 1.0]])       # crossed with the median at position 0
    q90 = np.array([[3.0, 9.0]])       # crossed with the median at position 0
    m, lo, hi = sort_quantiles(median, q10, q90)
    assert (lo <= m).all() and (m <= hi).all()
    # already-ordered input is unchanged
    m2, lo2, hi2 = sort_quantiles(np.array([[5.0]]), np.array([[1.0]]), np.array([[9.0]]))
    assert (m2, lo2, hi2) == pytest.approx((5.0, 1.0, 9.0))


def test_forecaster_registry_loads_without_any_ml_framework_installed():
    load_all()
    assert set(available()) == {"timesfm3", "chronos2", "patchtst_fm"}


def test_recovery_weights_follow_the_record_and_respect_coverage():
    from humob26.evidence import recovery_weights
    series = {"20240120": 100.0, "20240201": 80.0, "20240301": 40.0, "20240420": 0.0}
    before = [f"202401{d:02d}" for d in range(22, 32)]
    after = [f"202404{d:02d}" for d in range(1, 11)]
    target = ["20240201", "20240301"]
    w = recovery_weights(series, target, before, after)
    assert w is not None and 1.0 >= w["20240201"] > w["20240301"] >= 0.0
    # a record that stops before the after-anchor is not extrapolated
    short = {k: v for k, v in series.items() if k < "20240401"}
    assert recovery_weights(short, target, before, after) is None
