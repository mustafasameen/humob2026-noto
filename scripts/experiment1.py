"""Run Experiment 1 in reproducible stages; retains all attempts.

Usage: .venv/bin/python scripts/experiment1.py
Already successful runs are reused; failed runs stop continuation and require an
explicit new output root or manual investigation. At most two jobs run at once.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

from humob26.baselines.run import EXPERIMENT_WINDOWS

ROOT = Path('results/experiment1')
ENV = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}


def invoke(method, name, rank=None, tuning=False):
    directory = ROOT / (f'tuning/{method}_rank{rank}' if tuning else method) / name
    if directory.exists():
        status = json.loads((directory / 'run.json').read_text())['status']
        if status == 'success':
            return
        raise RuntimeError(f'Existing {status} attempt preserved: {directory}')
    cmd = [sys.executable, '-m', 'humob26.baselines.run', 'run', '--method', method,
           '--window', name, '--out', str(directory), '--reference', str(ROOT / 'final')]
    if rank is not None:
        cmd += ['--rank', str(rank), '--lags', '1', '2', '7']
        if not tuning:
            cmd += ['--selection', str(ROOT / 'selection.json')]
    subprocess.run(cmd, env=ENV, check=True)


def batch(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, *job) for job in jobs]
        for future in futures:
            future.result()


def main():
    check = pd.read_csv('results/check/window_scores.csv')
    value = check.loc[(check.arm == 'final') & (check.window == 'may_jun'), 'combined']
    if len(value) != 1 or f'{value.iloc[0]:.6f}' != '0.202910':
        raise RuntimeError('The documented reproduction gate has not passed')
    ROOT.mkdir(parents=True, exist_ok=True)
    batch([('final', n) for n in EXPERIMENT_WINDOWS])
    batch([('linear', n) for n in EXPERIMENT_WINDOWS])
    batch([(m, 'may_jun', r, True) for m in ('btmf', 'trmf') for r in (5, 10, 20)])
    if not (ROOT / 'selection.json').exists():
        subprocess.run([sys.executable, '-m', 'humob26.baselines.run', 'select', '--root', str(ROOT)], check=True)
    selected = json.loads((ROOT / 'selection.json').read_text())
    batch([(m, n, selected[m]['rank']) for m in ('btmf', 'trmf') for n in EXPERIMENT_WINDOWS if n != 'may_jun'])
    subprocess.run([sys.executable, '-m', 'humob26.baselines.run', 'report', '--root', str(ROOT)], check=True)


if __name__ == '__main__':
    main()
