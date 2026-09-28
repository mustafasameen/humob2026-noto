#!/usr/bin/env python3
"""Run Experiment 2 with immutable output folders and recorded provenance."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from humob26 import recovery

EXPECTED_SUBMISSION_MD5 = 'f19bfbd37dd859e0c0e928ffd0f83c5a'
EXPECTED_FULLGRID_MD5 = 'be03ccbf5aaebd81a275b3a52953504f'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(args, out, label):
    start = time.monotonic()
    with (out / f'{label}.log').open('w') as log:
        p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    with (out / 'commands.jsonl').open('a') as log:
        log.write(json.dumps({'argv': args, 'exit_code': p.returncode,
                              'elapsed_seconds': time.monotonic() - start,
                              'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()}) + '\n')
    if p.returncode:
        raise RuntimeError(f'{label} failed with exit code {p.returncode}; see {out / (label + ".log")}')
    print(f'{label}: complete', flush=True)


def verify_outputs(daily, summary, mapping, totals):
    """Cross-check stored shares, denominators, date coverage and first crossings."""
    checks = {}
    assert len(mapping) == 1476 and mapping.cell.nunique() == 1476
    assert len(daily) == 366 * 9
    assert not daily.duplicated(['date', 'municipality']).any()
    finite = daily[daily.share.notna()]
    assert np.allclose(finite.share, finite.within_flow / finite.denominator, rtol=1e-14, atol=0)
    assert ((finite.share >= 0) & (finite.share <= 1)).all()
    assert np.allclose(totals.all_row_total, totals.all_row_total.iloc[0], rtol=1e-12, atol=1e-8)
    for row in summary.itertuples():
        obs = daily[(daily.municipality == row.municipality) & (daily.source == 'observed')]
        b = obs[obs.date.isin(recovery.JANUARY)].share.mean()
        a = obs[obs.date.isin(recovery.APRIL)].share.mean()
        assert np.isclose(b, row.january_mean_share) and np.isclose(a, row.april_mean_share)
        gap = daily[(daily.municipality == row.municipality) & (daily.source == 'reconstructed')].sort_values('date')
        if row.municipality != recovery.OUTSIDE:
            assert len(gap) == 58
            if row.halfway_status == 'reached':
                crossing = (gap.share - b) / (a - b) >= .5
                assert gap.loc[crossing, 'date'].iloc[0] == row.halfway_date
        else:
            assert gap.empty
    checks.update(mapping_cells=len(mapping), calendar_dates=366, groups=9,
                  observed_days=len(totals), municipality_reconstructed_days=58,
                  january_days=int(summary.january_observed_days.iloc[0]),
                  april_days=int(summary.april_observed_days.iloc[0]),
                  fixed_daily_total=float(totals.all_row_total.mean()),
                  total_min=float(totals.all_row_total.min()), total_max=float(totals.all_row_total.max()),
                  date_coverage='passed', share_arithmetic='passed', halfway_first_crossings='passed')
    return checks


def write_report(summary, audit, out):
    lines = ['# Experiment 2: recovery by municipality', '',
             'The original submission was regenerated from the released MLX forecasts and its MD5 matched '
             f'`{EXPECTED_SUBMISSION_MD5}`. The preceding May–June reproduction scored 0.202910 to six decimals.', '',
             '## Anchor comparison and reconstructed halfway dates', '',
             'Shares are fractions of the complete daily flow total. Percentages below multiply those fractions by 100; '
             'the ratio is April mean share divided by January mean share.', '',
             '| Municipality | January share (%) | April share (%) | April / January | 95% CI | First halfway date |',
             '|---|---:|---:|---:|---|---|']
    for r in summary.itertuples():
        ratio = f'{r.april_over_january:.4f}' if np.isfinite(r.april_over_january) else 'undefined'
        ci = f'[{r.ratio_ci_lo:.4f}, {r.ratio_ci_hi:.4f}]' if np.isfinite(r.ratio_ci_lo) else r.ratio_status
        when = r.halfway_date or r.halfway_status.replace('_', ' ')
        lines.append(f'| {r.municipality} | {r.january_mean_share*100:.4f} | {r.april_mean_share*100:.4f} | {ratio} | {ci} | {when} |')
    lines += ['', '## Method and interpretation', '',
              '- Cells are assigned with the existing `evidence.cell_to_muni` nearest-centroid mapping. These are evaluation-region cell groupings, not exact municipal boundaries or municipality-wide population estimates.',
              '- A municipality numerator sums only its evaluation-region within-cell flows. The denominator sums every raw OD row, including within/outside-grid and cross-boundary flows.',
              '- The outside-grid numerator is the within-bucket `-1_-1 -> -1_-1` flow. Cross-boundary flows enter the denominator but not this within-bucket numerator. `daily_totals.csv` also records all flows with an outside endpoint for audit.',
              f'- The raw total is constant to floating-point precision: {audit["fixed_daily_total"]:.12f}. Each observed day uses its own all-row total; reconstruction uses their mean, never the sum of the model predictions.',
              f'- January 22–31 contributes {audit["january_days"]} observed days; April contributes {audit["april_days"]}. Missing days are excluded from anchor means and resampling.',
              '- The percentile interval resamples January and April days independently, with replacement, 4,000 times, seed 0. All municipalities share the same sampled date indices. Zero denominators produce explicit undefined statuses.',
              '- Halfway means `(reconstructed share - January mean) / (April mean - January mean) >= 0.5`, including decreases. The first qualifying available day is reported; no smoothing or sustained-crossing condition is added.',
              '- The two challenge-excluded dates, February 2 and March 5, remain blank. Dates therefore refer to the first of the 58 available reconstructed days. Missing observed days are also left blank.',
              '- The submission has no outside-grid forecasts, so its reconstructed share and halfway date are unavailable. No residual or interpolated outside-bucket series is invented.',
              '- Halfway dates describe the model path; they are not observed recovery dates. Ratio intervals cover sampling variation of observed anchor days, not reconstruction uncertainty.', '',
              '## Outputs', '',
              '- `municipality_summary.csv`: anchor shares, ratios and intervals, halfway levels/dates/statuses.',
              '- `daily_shares.csv`: full-calendar observations, reconstructed shares and explicit missing statuses.',
              '- `cell_to_municipality.csv`, `daily_totals.csv`: mapping and denominator audit.',
              '- `municipality_shares.png` / `.pdf`: November–April daily-share figure.',
              '- `commands.jsonl`, numbered logs, `packages.txt`, `run.json`, `checksums.json`, `verification.json`: provenance and validation.',
              '- `submission.tsv` and `.fullgrid.tsv`, caches and forecasts are retained locally and excluded from Git by the repository rules.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/raw/humob2026-dataset.tsv')
    p.add_argument('--forecasts', default='forecasts/timesfm3_mlx')
    p.add_argument('--out', required=True, help='new output directory; existing attempts are never overwritten')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    meta = {'command': [sys.executable, *sys.argv], 'status': 'running',
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'cwd': str(Path.cwd()), 'python': sys.version, 'platform': platform.platform(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
            'thread_environment': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']},
            'bootstrap_reps': 4000, 'bootstrap_seed': 0, 'source_sha256': {}}
    write_json(out / 'run.json', meta)
    try:
        meta['source_sha256'] = {str(f): digest(f) for f in sorted(Path('src/humob26').rglob('*.py'))}
        meta['source_sha256'][__file__] = digest(__file__)
        inputs = [Path(args.data)] + [Path(args.forecasts) / f'fc_{w}.npz' for w in ('may_jun', 'test_gap')]
        meta['inputs'] = {str(f): {'resolved_path': str(f.resolve()), 'bytes': f.stat().st_size,
                                  'sha256': digest(f)} for f in inputs}
        write_json(out / 'run.json', meta)
        command([sys.executable, '-m', 'pip', 'freeze'], out, '01-packages')
        shutil.copyfile(out / '01-packages.log', out / 'packages.txt')
        cache, forecasts = out / 'cache', out / 'forecasts'
        forecasts.mkdir()
        for name in ('may_jun', 'test_gap'):
            shutil.copyfile(Path(args.forecasts) / f'fc_{name}.npz', forecasts / f'fc_{name}.npz')
        base = [sys.executable, '-m', 'humob26']
        command(base + ['prepare', '--data', args.data, '--cache', str(cache)], out, '02-prepare')
        command(base + ['contexts', '--cache', str(cache), '--out', str(forecasts),
                        '--windows', 'may_jun', 'test_gap'], out, '03-contexts')
        command(base + ['evaluate', '--cache', str(cache), '--forecasts', str(forecasts),
                        '--windows', 'may_jun', '--out', str(out / 'check')], out, '04-reproduction')
        scores = pd.read_csv(out / 'check' / 'window_scores.csv')
        score = scores.loc[(scores.arm == 'final') & (scores.window == 'may_jun'), 'combined']
        if len(score) != 1 or f'{score.iloc[0]:.6f}' != '0.202910':
            raise ValueError('May–June reproduction failed; analysis stopped')
        command(base + ['submit', '--cache', str(cache), '--forecasts', str(forecasts),
                        '--out', str(out / 'submission.tsv')], out, '05-submit')
        for name, expected in [('submission.tsv', EXPECTED_SUBMISSION_MD5),
                               ('submission.fullgrid.tsv', EXPECTED_FULLGRID_MD5)]:
            actual = digest(out / name, 'md5')
            meta[name + '_md5'] = actual
            if actual != expected:
                raise ValueError(f'{name} MD5 {actual} differs from reference {expected}')
        daily, mapping, totals = recovery.build_daily(args.data, out / 'submission.tsv')
        summary = recovery.summarize(daily)
        audit = verify_outputs(daily, summary, mapping, totals)
        audit['reproduction_combined'] = float(score.iloc[0])
        audit['submission_md5'] = meta['submission.tsv_md5']
        audit['fullgrid_md5'] = meta['submission.fullgrid.tsv_md5']
        write_json(out / 'verification.json', audit)
        daily.to_csv(out / 'daily_shares.csv', index=False)
        summary.to_csv(out / 'municipality_summary.csv', index=False)
        mapping.to_csv(out / 'cell_to_municipality.csv', index=False)
        totals.to_csv(out / 'daily_totals.csv', index=False)
        recovery.plot_shares(daily, summary, out)
        write_report(summary, audit, out)
        meta['status'] = 'success'
        print(summary.to_string(index=False))
    except BaseException:
        meta['status'] = 'failed'
        meta['error'] = traceback.format_exc()
        (out / 'failure.log').write_text(meta['error'])
        pd.DataFrame([{'status': 'failed', 'error': meta['error']}]).to_csv(out / 'failure.csv', index=False)
        print(meta['error'], file=sys.stderr)
    meta['elapsed_seconds'] = time.monotonic() - start
    meta['exit_code'] = 0 if meta['status'] == 'success' else 1
    write_json(out / 'run.json', meta)
    # Manifest excludes itself; all other run files are hashed, including local TSV/NPZ.
    write_json(out / 'checksums.json', {str(f.relative_to(out)): {'sha256': digest(f), 'bytes': f.stat().st_size}
                                       for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'checksums.json'})
    print(f'Experiment 2 {meta["status"]}: {out}', flush=True)
    return meta['exit_code']


if __name__ == '__main__':
    sys.exit(main())
