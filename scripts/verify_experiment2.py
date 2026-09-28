#!/usr/bin/env python3
"""Independent arithmetic audit from raw dictionaries and saved Task 2 CSVs.

Usage: python scripts/verify_experiment2.py RAW_TSV RUN_DIRECTORY
This does not import the recovery implementation being checked.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from humob26 import submission


def audit(raw_path, root):
    root = Path(root)
    daily = pd.read_csv(root / 'daily_shares.csv', dtype={'date': str})
    summary = pd.read_csv(root / 'municipality_summary.csv', dtype={'halfway_date': str})
    mapping = pd.read_csv(root / 'cell_to_municipality.csv')
    gids = {f'{r.y}_{r.x}': r.municipality for r in mapping.itertuples()}
    by_date = daily.set_index(['date', 'municipality'])
    groups = summary.municipality.tolist()
    totals, observed_dates = [], []
    count = 0
    with open(raw_path) as f:
        for line in f:
            date, payload = line.rstrip('\n').split('\t', 1)
            if not date.isdigit() or payload.strip() in {'NA', 'N/A', 'nan', 'NaN', 'None', '{}', 'null', ''}:
                continue
            od = json.loads(payload.replace("'", '"'))
            total = math.fsum(v for destinations in od.values() for v in destinations.values())
            totals.append(total)
            observed_dates.append(date)
            numerator = {m: [] for m in groups}
            for origin, destinations in od.items():
                if origin in gids:
                    numerator[gids[origin]].append(destinations.get(origin, 0.))
                elif origin == '-1_-1':
                    numerator['outside_grid'].append(destinations.get(origin, 0.))
            for group in groups:
                row = by_date.loc[(date, group)]
                assert row.source == 'observed'
                assert math.isclose(row.denominator, total, rel_tol=1e-12)
                assert math.isclose(row.share, math.fsum(numerator[group]) / total, rel_tol=1e-12, abs_tol=1e-15)
                count += 1
    fixed = math.fsum(totals) / len(totals)
    with (root / 'submission.tsv').open() as f:
        for line in f:
            date, payload = line.rstrip('\n').split('\t', 1)
            od = json.loads(payload.replace("'", '"'))
            for group in groups:
                row = by_date.loc[(date, group)]
                if group == 'outside_grid':
                    assert pd.isna(row.share) and row.status == 'outside_bucket_not_predicted'
                    continue
                numerator = math.fsum(od.get(g, {}).get(g, 0.) for g, m in gids.items() if m == group)
                assert math.isclose(row.share, numerator / fixed, rel_tol=1e-12, abs_tol=1e-15)
                count += 1
    wide = daily[daily.source == 'observed'].pivot(index='date', columns='municipality', values='share')[groups]
    b = wide.loc[(wide.index >= '20240122') & (wide.index <= '20240131')]
    a = wide.loc[(wide.index >= '20240401') & (wide.index <= '20240430')]
    rng = np.random.default_rng(0)
    ib = rng.integers(len(b), size=(4000, len(b)))
    ia = rng.integers(len(a), size=(4000, len(a)))
    for r in summary.itertuples():
        before, after = b[r.municipality].to_numpy(), a[r.municipality].to_numpy()
        bm, am = math.fsum(before) / len(before), math.fsum(after) / len(after)
        assert math.isclose(r.january_mean_share, bm, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(r.april_mean_share, am, rel_tol=1e-12, abs_tol=1e-15)
        den = before[ib].mean(axis=1)
        num = after[ia].mean(axis=1)
        if (den > 0).all():
            lo, hi = np.percentile(num / den, [2.5, 97.5])
            np.testing.assert_allclose([r.april_over_january, r.ratio_ci_lo, r.ratio_ci_hi], [am / bm, lo, hi], rtol=1e-12)
        else:
            assert pd.isna(r.ratio_ci_lo) and pd.isna(r.ratio_ci_hi)
        gap = daily[(daily.municipality == r.municipality) & (daily.source == 'reconstructed')].sort_values('date')
        if r.municipality == 'outside_grid':
            assert pd.isna(r.halfway_date)
            continue
        if math.isclose(bm, am, rel_tol=1e-12, abs_tol=1e-15):
            assert r.halfway_status == 'no_change' and pd.isna(r.halfway_date)
            continue
        threshold = (bm + am) / 2
        reached = gap[gap.share >= threshold] if am > bm else gap[gap.share <= threshold]
        if reached.empty:
            assert r.halfway_status == 'not_reached' and pd.isna(r.halfway_date)
        else:
            assert r.halfway_status == 'reached' and r.halfway_date == reached.date.iloc[0]
    recorded = json.loads((root / 'verification.json').read_text())
    rounded = submission.rounded_digest(root / 'submission.tsv')
    assert rounded == recorded['submission_rounded_md5'] == 'f96f76dfcf777b36fff319e9292e8fe1'
    assert hashlib.md5((root / 'submission.fullgrid.tsv').read_bytes()).hexdigest() == 'be03ccbf5aaebd81a275b3a52953504f'
    excluded = daily[daily.date.isin(['20240202', '20240305'])]
    assert excluded.share.isna().all() and (excluded.status == 'excluded_by_challenge').all()
    print(f'PASS: independently recomputed {count} observed/reconstructed shares from raw dictionaries;')
    print('all anchor means, ratios, 4000-resample intervals, halfway-date arithmetic and excluded dates match.')
    print('Full-grid MD5 and rounded submission MD5: PASS.')


if __name__ == '__main__':
    audit(sys.argv[1], sys.argv[2])
