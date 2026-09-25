#!/usr/bin/env python3
"""Do the prefecture's recovery records improve the final model? (Section 5.4 of the paper)

Arms, each identical to `final` except for the diagonal's path between the anchors:
  E1   the path read off each cell's municipality shelter-evacuee series (full trust)
  E05  that path averaged 50/50 with the recovery curve
Known answer: trust 0 reproduces `final` bit for bit on every window, and `final` reproduces the
`evaluate` output given as REFERENCE_WINDOW_SCORES to 1e-12.

Writes OUT_DIR/external_records.csv (per window and pooled) and external_records_daily.csv.
Usage: python scripts/external_records.py CACHE_DIR FORECASTS_DIR RECORDS_CSV REFERENCE_WINDOW_SCORES OUT_DIR
  CACHE_DIR                 the `prepare` cache (eval_box.npz)
  FORECASTS_DIR             TimesFM-3 ctx_/fc_<window>.npz files under public window names
  RECORDS_CSV               external/ishikawa_per_muni.csv
  REFERENCE_WINDOW_SCORES   window_scores.csv from `evaluate`, which `final` must reproduce
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from humob26 import combine, data, foundation, metric, windows  # noqa: E402
from humob26.bootstrap import date_cluster_bootstrap, paired_bootstrap_components  # noqa: E402
from humob26.evidence import build_final_evidence, interp_two_anchor_eb_evidence, load_muni_curves  # noqa: E402

NAMES = ("may_jun", "late_jan", "april", "january") + windows.ROLL_WINDOW_NAMES + \
        ("stable_spring", "stable_summer", "stable_late")
ARMS = {"E1": 1.0, "E05": 0.5}


def main():
    cache, fc_dir, records, ref_path, out = (Path(a) for a in sys.argv[1:6])
    CACHE, FC, RECORDS, REF, OUT = cache / "eval_box.npz", fc_dir, records, ref_path, out
    days, na = data.load_eval_box(None, cache=CACHE)
    all_dates = sorted(d for d in days if d not in na)
    curves = load_muni_curves(RECORDS)
    ref = {(r["window"], r["arm"]): float(r["combined"]) for r in csv.DictReader(open(REF))}
    rows, daily, per_day = [], [], {}
    for name in NAMES:
        w = windows.get_window(name, days, na)
        keys, train, calib = combine.eval_box_context(days, w, all_dates)
        fc = foundation.read_forecast(FC / f"ctx_{name}.npz", FC / f"fc_{name}.npz")
        obs = {d: days[d] for d in w.target}
        preds = {"final": combine.build_final(days, w, na, keys, train, calib, fc)}
        # known answer: zero trust is `final`
        zero = build_final_evidence(days, w, na, keys, train, calib, fc, curves, trust=0.0)
        for d in w.target:
            a, b = preds["final"][d], zero[d]
            assert np.array_equal(a.key, b.key) and np.array_equal(a.val, b.val), (name, d)
        for arm, trust in ARMS.items():
            preds[arm] = build_final_evidence(days, w, na, keys, train, calib, fc, curves, trust=trust)
        _, recs, muni_of_key = interp_two_anchor_eb_evidence(days, w, na, keys, calib, curves, 1.0)
        n_cells = int(sum(1 for m in muni_of_key if m in recs))
        scores = {}
        for arm, p in preds.items():
            summary, pd_ = metric.score(obs, p)
            per_day[(name, arm)] = pd_.set_index("date")
            scores[arm] = summary["combined_dense"]
            dc = metric.daily_combined(per_day[(name, arm)])
            for d in w.target:
                daily.append(dict(window=name, date=d, arm=arm, combined=dc[d]))
        assert abs(scores["final"] - ref[(name, "final")]) < 1e-12, (name, scores["final"], ref[(name, "final")])
        for arm in ARMS:
            parts = paired_bootstrap_components(per_day[(name, "final")], per_day[(name, arm)], w.target)
            est, lo, hi = parts["combined"]            # final - arm: positive means the arm is better
            b = scores["final"]
            rows.append(dict(window=name, arm=arm, combined=scores[arm], final=b, gain_pct=100 * est / b,
                             ci_lo_pct=100 * lo / b, ci_hi_pct=100 * hi / b, cells_with_record=n_cells,
                             municipalities_with_record=len(recs)))
        print(f"{name:14s} final {scores['final']:.6f}  E1 {scores['E1']:.6f}  E05 {scores['E05']:.6f}  "
              f"cells with a record {n_cells}", flush=True)
    # pooled rolling windows, date clusters
    for arm in ARMS:
        dates, diffs, base = [], [], []
        for name in windows.ROLL_WINDOW_NAMES:
            w = windows.get_window(name, days, na)
            da = metric.daily_combined(per_day[(name, "final")])
            db = metric.daily_combined(per_day[(name, arm)])
            for d in w.target:
                dates.append(d)
                diffs.append(da[d] - db[d])
                base.append(da[d])
        est, lo, hi, _ = date_cluster_bootstrap(dates, diffs)
        b = float(np.mean(base))
        rows.append(dict(window="rolling", arm=arm, combined=b - est, final=b, gain_pct=100 * est / b,
                         ci_lo_pct=100 * lo / b, ci_hi_pct=100 * hi / b, cells_with_record="",
                         municipalities_with_record=""))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "external_records.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    with open(OUT / "external_records_daily.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["window", "date", "arm", "combined"])
        wr.writeheader()
        wr.writerows(daily)
    for r in rows:
        if r["window"] in ("may_jun", "late_jan", "april", "january", "rolling"):
            print(f"{r['window']:9s} {r['arm']:4s} gain {r['gain_pct']:+.3f}%  [{r['ci_lo_pct']:+.3f}, {r['ci_hi_pct']:+.3f}]")
    print("wrote", OUT / "external_records.csv")


if __name__ == "__main__":
    main()
