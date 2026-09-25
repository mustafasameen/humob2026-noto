"""End-to-end reproduction check on the `may_jun` validation window.

This test touches the real dataset and a real set of foundation-model
forecast files, neither of which ships with this package, so it is skipped
unless the environment points at them:

  HUMOB26_TEST_DATA        path to the raw dataset TSV
  HUMOB26_TEST_FORECASTS   directory holding ctx_may_jun.npz/fc_may_jun.npz
                           (this package reads forecast files by each
                           window's own public name; there is no mechanism
                           for pointing it at a differently named pair)
  HUMOB26_TEST_REFS        directory of reference score tables to compare
                           against (optional): with it set, this test also
                           checks the `final`, `anchor_rts` and
                           `anchor_rts_fullgrid` arms against it to
                           floating-point precision; without it, this test
                           only checks that the pipeline runs end to end
                           and produces finite, non-degenerate scores.
  HUMOB26_TEST_REF_LAYOUT  JSON object overriding where those values live
                           in HUMOB26_TEST_REFS, for a reference table
                           produced under different file, row or column
                           names than this package's own. Keys and their
                           defaults: row ("may_jun"), final_file
                           ("final_scores.csv"), final_col ("final"),
                           fullgrid_file ("fullgrid_vs_box.csv"), box_col
                           ("box"), full_col ("full"), baseline_file
                           ("organizer_baseline_calib.csv"),
                           baseline_row_col ("fold"), baseline_col
                           ("b1_combined").
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

DATA = os.environ.get("HUMOB26_TEST_DATA")
FORECASTS = os.environ.get("HUMOB26_TEST_FORECASTS")
REFS = os.environ.get("HUMOB26_TEST_REFS")
LAYOUT = {
    "row": "may_jun",
    "final_file": "final_scores.csv", "final_col": "final",
    "fullgrid_file": "fullgrid_vs_box.csv", "box_col": "box", "full_col": "full",
    "baseline_file": "organizer_baseline_calib.csv",
    "baseline_row_col": "fold", "baseline_col": "b1_combined",
}
LAYOUT.update(json.loads(os.environ.get("HUMOB26_TEST_REF_LAYOUT", "{}")))

pytestmark = pytest.mark.skipif(
    not (DATA and FORECASTS),
    reason="set HUMOB26_TEST_DATA and HUMOB26_TEST_FORECASTS to run the integration test",
)


def test_may_jun_reproduces_reference_scores(tmp_path):
    from humob26 import data, windows
    from humob26.cli import Pipeline
    from humob26.config import NA_DAYS
    from humob26.metric import score

    cache_dir = tmp_path / "cache"
    days, na = data.load_eval_box(DATA, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    full_days = data.load_full_grid(DATA, cache=cache_dir / "full_grid.npz")
    na_full = set(NA_DAYS)
    all_full_dates = sorted(d for d in full_days if d not in na_full)

    window = windows.get_window("may_jun", days, na)
    obs = {d: days[d] for d in window.target}
    pipe = Pipeline(days, na, all_dates, full_days, all_full_dates, FORECASTS)

    scores = {}
    for arm in ("april_mean", "anchor_rts", "anchor_rts_fullgrid", "final"):
        pred = pipe.build(arm, window)
        summary, _ = score(obs, pred)
        scores[arm] = summary["combined_dense"]
        assert scores[arm] == scores[arm]         # finite, not NaN
        assert 0.0 < scores[arm] < 2.0             # a sane Combined-score range

    # Every arm should improve on the organisers'-style flat baseline.
    assert scores["final"] < scores["april_mean"]
    assert scores["anchor_rts"] < scores["april_mean"]

    if not REFS:
        pytest.skip("HUMOB26_TEST_REFS not set: ran end to end but did not check exact values")

    row = LAYOUT["row"]
    final_ref = pd.read_csv(Path(REFS) / LAYOUT["final_file"]).set_index("window")
    fullgrid_ref = pd.read_csv(Path(REFS) / LAYOUT["fullgrid_file"]).set_index("window")
    baseline_ref = pd.read_csv(Path(REFS) / LAYOUT["baseline_file"]).set_index(LAYOUT["baseline_row_col"])

    assert scores["final"] == pytest.approx(final_ref.loc[row, LAYOUT["final_col"]], abs=1e-12)
    assert scores["anchor_rts"] == pytest.approx(fullgrid_ref.loc[row, LAYOUT["box_col"]], abs=1e-12)
    assert scores["anchor_rts_fullgrid"] == pytest.approx(fullgrid_ref.loc[row, LAYOUT["full_col"]], abs=1e-12)
    assert scores["april_mean"] == pytest.approx(baseline_ref.loc[row, LAYOUT["baseline_col"]], abs=1e-12)
