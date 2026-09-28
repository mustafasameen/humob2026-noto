"""One immutable, fully logged Experiment 1 run, or aggregate completed runs."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from humob26 import bootstrap, config, data, metric, windows
from humob26.cli import Pipeline
from .methods import complete, linear

EXPERIMENT_WINDOWS = ('may_jun', 'late_jan', 'april') + windows.ROLL_WINDOW_NAMES + windows.SHIFT_WINDOW_NAMES
RANKS = (5, 10, 20)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def run(args):
    if args.method in ('btmf', 'trmf') and args.rank is None:
        raise ValueError('factorization requires --rank')
    if args.method in ('btmf', 'trmf') and args.window != 'may_jun':
        selection = json.loads(Path(args.selection).read_text()) if args.selection else {}
        if selection.get(args.method, {}).get('rank') != args.rank:
            raise ValueError('non-tuning runs require a matching may_jun-only selection file')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a run, including failures
    started = time.monotonic()
    meta = {'command': [sys.executable, '-m', 'humob26.baselines.run', *sys.argv[1:]],
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'method': args.method, 'window': args.window, 'rank': args.rank,
            'seed': 1000, 'lags_days': args.lags, 'bootstrap_reps': config.BOOTSTRAP_REPS,
            'bootstrap_seed': config.BOOTSTRAP_SEED, 'python': sys.version,
            'platform': platform.platform(), 'status': 'running',
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_diff': subprocess.check_output(['git', 'diff'], text=True),
            'source_sha256': {str(p): fingerprint(p) for p in sorted(Path('src/humob26').rglob('*.py'))},
            'versions': {p: importlib.metadata.version(p) for p in ('humob26', 'numpy', 'pandas', 'scipy')},
            'thread_env': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')},
            'eval_cache_sha256': fingerprint(Path(args.cache) / 'eval_box.npz')}
    if args.selection:
        meta['selection'] = json.loads(Path(args.selection).read_text())
    save_json(out / 'run.json', meta)
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(out / 'packages.txt').open('w'), check=True)
    status = 0
    with (out / 'output.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            days, na = data.load_eval_box(None, cache=Path(args.cache) / 'eval_box.npz')
            w = windows.get_window(args.window, days, na)
            save_json(out / 'window.json', {'target': w.target, 'before': w.before, 'after': w.after})
            if args.method == 'final':
                meta['forecast_sha256'] = fingerprint(Path(args.forecasts) / f'fc_{w.name}.npz')
                meta['context_sha256'] = fingerprint(Path(args.forecasts) / f'ctx_{w.name}.npz')
                pipe = Pipeline(days, na, sorted(d for d in days if d not in na), forecasts_dir=args.forecasts)
                pred = pipe.build('final', w)
            elif args.method == 'linear':
                pred = linear(days, na, w)
            else:
                pred = complete(days, na, w, args.method, args.rank, lags=args.lags)
            if set(pred) != set(w.target):
                raise ValueError('prediction dates do not match the complete window target')
            summary, daily = metric.score({d: days[d] for d in w.target}, pred)
            daily.insert(0, 'window', w.name)
            daily['arm'] = args.method
            daily['combined'] = metric.daily_combined(daily).reindex(daily.date).to_numpy()
            daily.to_csv(out / 'daily_scores.csv', index=False)
            pd.DataFrame([{'window': w.name, 'arm': args.method, 'rank': args.rank,
                           'n_days': summary['n_days'], 'combined': summary['combined_dense'],
                           'nrmse_diag': summary['nrmse_diag_dense'],
                           'nrmse_offdiag': summary['nrmse_offdiag_dense']}]).to_csv(out / 'window_scores.csv', index=False)
            if args.method != 'final':
                refdir = Path(args.reference) / w.name
                refmeta = json.loads((refdir / 'run.json').read_text())
                if refmeta['status'] != 'success' or refmeta['eval_cache_sha256'] != meta['eval_cache_sha256']:
                    raise ValueError('reference incomplete or evaluated on different data')
                ref = pd.read_csv(refdir / 'daily_scores.csv', dtype={'date': str}).set_index('date')
                if set(ref.index) != set(w.target):
                    raise ValueError('reference date mismatch')
                parts = bootstrap.paired_bootstrap_components(daily.set_index('date'), ref, w.target)
                pd.DataFrame([{'window': w.name, 'arm_a': args.method, 'arm_b': 'final',
                               'component': c, 'mean_diff': v[0], 'ci_lo': v[1], 'ci_hi': v[2],
                               'reps': config.BOOTSTRAP_REPS} for c, v in parts.items()]).to_csv(out / 'comparisons.csv', index=False)
            meta['status'] = 'success'
        except BaseException:
            status = 1
            meta['status'] = 'failed'
            meta['error'] = traceback.format_exc()
            print(meta['error'])
            pd.DataFrame([{'method': args.method, 'window': args.window, 'rank': args.rank,
                           'error': meta['error']}]).to_csv(out / 'failure.csv', index=False)
    meta['elapsed_seconds'] = time.monotonic() - started
    meta['exit_code'] = status
    save_json(out / 'run.json', meta)
    print(f'{args.method} rank={args.rank} {args.window}: {meta["status"]} ({meta["elapsed_seconds"]:.1f}s)', flush=True)
    return status


def select(args):
    root = Path(args.root)
    selection = {}
    tables = []
    for method in ('btmf', 'trmf'):
        candidates = []
        for rank in RANKS:
            directory = root / 'tuning' / f'{method}_rank{rank}' / 'may_jun'
            meta = json.loads((directory / 'run.json').read_text())
            if meta['status'] != 'success' or meta['window'] != 'may_jun':
                raise ValueError(f'all three tuning runs must succeed: {directory}')
            row = pd.read_csv(directory / 'window_scores.csv').iloc[0]
            candidates.append((float(row.combined), rank))
            tables.append({'method': method, 'rank': rank, 'combined': float(row.combined), 'path': str(directory)})
        score, rank = min(candidates)
        selection[method] = {'rank': rank, 'may_jun_combined': score, 'selection_window': 'may_jun'}
    if (root / 'selection.json').exists():
        raise FileExistsError('selection already frozen')
    save_json(root / 'selection.json', selection)
    pd.DataFrame(tables).to_csv(root / 'rank_scores.csv', index=False)
    print(json.dumps(selection, indent=2))


def report(args):
    root = Path(args.root)
    chosen = json.loads((root / 'selection.json').read_text())
    all_scores, comparisons, pooled = [], [], []
    for method in ('linear', 'btmf', 'trmf'):
        daily_parts = []
        for name in EXPERIMENT_WINDOWS:
            directory = root / method / name
            if method != 'linear' and name == 'may_jun':
                directory = root / 'tuning' / f'{method}_rank{chosen[method]["rank"]}' / name
            if json.loads((directory / 'run.json').read_text())['status'] != 'success':
                raise ValueError(f'incomplete run {directory}')
            all_scores.append(pd.read_csv(directory / 'window_scores.csv'))
            comparisons.append(pd.read_csv(directory / 'comparisons.csv'))
            candidate = pd.read_csv(directory / 'daily_scores.csv', dtype={'date': str})
            ref = pd.read_csv(root / 'final' / name / 'daily_scores.csv', dtype={'date': str})
            joined = candidate.merge(ref[['window', 'date', 'combined']], on=['window', 'date'],
                                     suffixes=('', '_final'), validate='one_to_one')
            if len(joined) != len(candidate):
                raise ValueError('missing reference rows')
            daily_parts.append(joined)
        daily = pd.concat(daily_parts, ignore_index=True)
        for label, names in [('pooled_all', EXPERIMENT_WINDOWS), ('pooled_roll', windows.ROLL_WINDOW_NAMES),
                             ('pooled_shift', windows.SHIFT_WINDOW_NAMES)]:
            subset = daily[daily.window.isin(names)]
            est, lo, hi, _ = bootstrap.date_cluster_bootstrap(subset.date, subset.combined - subset.combined_final)
            pooled.append({'window': label, 'arm': method, 'combined': subset.combined.mean(),
                           'final_combined': subset.combined_final.mean(), 'mean_diff': est,
                           'ci_lo': lo, 'ci_hi': hi, 'n_window_date_rows': len(subset), 'reps': config.BOOTSTRAP_REPS})
    pd.concat(all_scores).to_csv(root / 'window_scores.csv', index=False)
    pd.concat(comparisons).to_csv(root / 'comparisons.csv', index=False)
    pd.DataFrame(pooled).to_csv(root / 'pooled_scores.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--method', required=True, choices=['final', 'linear', 'btmf', 'trmf'])
    p.add_argument('--window', required=True, choices=EXPERIMENT_WINDOWS)
    p.add_argument('--rank', type=int, choices=RANKS)
    p.add_argument('--lags', nargs='+', type=int, default=[1, 2, 7])
    p.add_argument('--cache', default='cache')
    p.add_argument('--forecasts', default='forecasts/timesfm3_mlx')
    p.add_argument('--reference', default='results/experiment1/final')
    p.add_argument('--selection')
    p.add_argument('--out', required=True)
    for action in ('select', 'report'):
        p = sub.add_parser(action)
        p.add_argument('--root', default='results/experiment1')
    args = parser.parse_args()
    if args.command == 'run':
        return run(args)
    return select(args) if args.command == 'select' else report(args)


if __name__ == '__main__':
    sys.exit(main())
