#!/usr/bin/env python3
"""Run the four documented sensitivity grids through the unchanged sweep CLI."""
import argparse
import ast
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
import pandas as pd
from humob26 import windows

GRIDS = {
    'rts_rank': {'submitted': {}, **{f'rank{v}': {'rts_rank': v} for v in (4, 8, 24)}},
    'anchor_rts_blend': {'submitted': {}, **{f'blend{v}': {'anchor_rts_blend': v} for v in (.3, .7)}},
    'fm_diagonal_weight': {'submitted': {}, **{f'weight{v}': {'fm_diagonal_weight': v} for v in (.25, 1.)}},
    'single_tau': {'submitted': {}, **{f'tau{v}': {'single_tau': v} for v in (7.588, 26.88)}},
}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--reference-existing', help='existing successful evaluate output directory')
    p.add_argument('--cache', default='cache')
    p.add_argument('--forecasts', default='forecasts/timesfm3_mlx')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    meta = {'command': [sys.executable, *sys.argv], 'python': sys.version, 'platform': platform.platform(),
            'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'environment': {k: os.environ.get(k) for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            'status': 'running', 'bootstrap_reps': 4000, 'bootstrap_seed': 0,
            'windows': list(windows.VALIDATION_WINDOW_NAMES)}
    def run(cmd, label):
        start = time.time()
        with (out / f'{label}.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        with (out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'command': cmd, 'started_unix': start, 'seconds': time.time()-start, 'exit_code': result.returncode, 'log': f'{label}.log'})+'\n')
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see retained log')
        return (out / f'{label}.log').read_text()
    try:
        inputs = [*Path(args.cache).glob('*.npz')]
        for name in windows.VALIDATION_WINDOW_NAMES:
            inputs.extend(Path(args.forecasts)/f'{prefix}_{name}.npz' for prefix in ('ctx','fc'))
        meta['input_sha256'] = {str(f): sha(f) for f in inputs}
        meta['source_sha256'] = {str(f): sha(f) for f in [Path(__file__), *Path('src/humob26').rglob('*.py')]}
        (out/'packages.txt').write_text(run([sys.executable,'-m','pip','freeze'], 'packages'))
        base = [sys.executable,'-m','humob26']
        if args.reference_existing:
            shutil.copytree(args.reference_existing, out/'reference')
            meta['reused_reference'] = {'path': args.reference_existing, 'sha256': sha(Path(args.reference_existing)/'window_scores.csv')}
        else:
            run(base+['evaluate','--cache',args.cache,'--forecasts',args.forecasts,'--arms','final','--out',str(out/'reference')], 'reference')
        ref = pd.read_csv(out/'reference/window_scores.csv')
        assert set(ref.window) == set(windows.VALIDATION_WINDOW_NAMES)
        assert set(ref.arm) >= {'final'}
        ref = ref[ref.arm == 'final'].copy()
        assert f'{ref.loc[ref.window == "may_jun", "combined"].iloc[0]:.6f}' == '0.202910'
        ref.to_csv(out/'reference_final.csv', index=False)
        reports = ['# Experiment 4 sensitivity results', '', 'Lower scores are better. Pooled differences are variant minus submitted; negative means improvement. Intervals use the unchanged 4,000-replicate calendar-date bootstrap. All 16 default validation windows are retained.', '']
        pooled_rows = []
        for setting, grid in GRIDS.items():
            gridfile = out/f'{setting}.json'
            gridfile.write_text(json.dumps(grid, indent=2)+'\n')
            log = run(base+['sweep','--cache',args.cache,'--forecasts',args.forecasts,'--grid',str(gridfile),'--assert-reference',str(out/'reference_final.csv'),'--out',str(out/setting)], setting)
            rows = []
            for name, overrides in grid.items():
                scores = pd.read_csv(out/setting/f'sweep_{name}.csv')
                assert set(scores.window) == set(windows.VALIDATION_WINDOW_NAMES)
                assert scores.combined.notna().all()
                scores['run'] = name
                rows.append(scores)
                matches = [line for line in log.splitlines() if line.split() and line.split()[0] == name and 'pooled diff vs submitted:' in line]
                assert len(matches) == 1, (setting, name, matches)
                pooled = ast.literal_eval(matches[0].split('pooled diff vs submitted:',1)[1].strip())
                pooled_rows.append({'setting':setting,'run':name,'overrides':json.dumps(overrides), **pooled})
            combined = pd.concat(rows, ignore_index=True)
            baseline = combined[combined.run == 'submitted'].merge(ref[['window','combined']], on='window',suffixes=('','_ref'))
            assert len(baseline) == len(ref) and (abs(baseline.combined-baseline.combined_ref) <= 1e-12).all()
            table = combined.pivot(index='window', columns='run', values='combined').reindex(windows.VALIDATION_WINDOW_NAMES)
            table.to_csv(out/f'table_{setting}.csv')
            reports += [f'## {setting}', '', '| Window | '+' | '.join(table.columns)+' |', '|---|'+'---:|'*len(table.columns)]
            reports += ['| '+name+' | '+' | '.join(f'{v:.6f}' for v in row)+' |' for name,row in table.iterrows()]
            reports += ['']
        pooled = pd.DataFrame(pooled_rows)
        pooled.to_csv(out/'pooled_rolling_differences.csv', index=False)
        reports += ['## Pooled rolling differences', '', '| Setting | Run | Difference | 95% CI |', '|---|---|---:|---|']
        reports += [f'| {r.setting} | {r.run} | {r.pooled_diff_vs_submitted:.6f} | [{r.ci_lo:.6f}, {r.ci_hi:.6f}] |' for r in pooled.itertuples()]
        (out/'REPORT.md').write_text('\n'.join(reports)+'\n')
        meta['status'] = 'success'
    except BaseException:
        meta['status'] = 'failed'
        meta['error'] = traceback.format_exc()
        (out/'failure.log').write_text(meta['error'])
        pd.DataFrame([{'error':meta['error']}]).to_csv(out/'failure.csv', index=False)
        print(meta['error'], file=sys.stderr)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'checksums.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='checksums.json'},indent=2)+'\n')
    return int(meta['status'] != 'success')

if __name__ == '__main__':
    sys.exit(main())
