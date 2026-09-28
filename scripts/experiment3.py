#!/usr/bin/env python3
"""Run TimesFM-3 PyTorch full-context reference and six one-sided cuts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import pandas as pd
from humob26 import windows

NAMES = ('may_jun', 'late_jan', 'april', 'january', *windows.ROLL_WINDOW_NAMES)
CUTS = (('full', None, None), ('fwd14',14,None), ('fwd28',28,None), ('fwd56',56,None),
        ('bwd30',None,30), ('bwd60',None,60), ('bwd120',None,120))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--cache',default='cache')
    p.add_argument('--contexts',default='forecasts/timesfm3_mlx')
    p.add_argument('--weights',default='models/timesfm-3.0-pytorch')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    meta={'command':[sys.executable,*sys.argv], 'status':'running', 'python':sys.version,
          'platform':platform.platform(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'windows':NAMES,'bootstrap_reps':4000,'bootstrap_seed':0,'runs':{}}
    def run(cmd,label):
        start=time.time()
        with (out/f'{label}.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as f:
            f.write(json.dumps({'command':cmd,'started_unix':start,'elapsed_seconds':time.time()-start,
                                'exit_code':result.returncode,'log':f'{label}.log'})+'\n')
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see {label}.log')
    try:
        source=[Path(__file__),*Path('src/humob26').rglob('*.py')]
        meta['source_sha256']={str(f):sha(f) for f in source}
        inputs=[Path(args.cache)/'eval_box.npz', *[Path(args.contexts)/f'ctx_{n}.npz' for n in NAMES],
                *[Path(args.contexts)/f'fc_{n}.npz' for n in NAMES],
                *[f for f in Path(args.weights).glob('*') if f.is_file()]]
        meta['input_sha256']={str(f):sha(f) for f in inputs}
        run([sys.executable,'-m','pip','freeze'],'packages')
        (out/'packages.txt').write_text((out/'packages.log').read_text())
        ctx=out/'contexts'
        ctx.mkdir()
        for n in NAMES:
            (ctx/f'ctx_{n}.npz').symlink_to((Path(args.contexts)/f'ctx_{n}.npz').resolve())
        base=[sys.executable,'-m','humob26']
        for name,fwd,bwd in CUTS:
            cmd=base+['forecast','--model','timesfm3','--backend','torch','--device',args.device,
                      '--weights',args.weights,'--contexts',str(ctx),'--out',str(out/name)]
            if fwd: cmd.extend(['--context-days-fwd',str(fwd)])
            if bwd: cmd.extend(['--context-days-bwd',str(bwd)])
            run(cmd,f'forecast_{name}')
            names={p.name for p in (out/name).glob('fc_*.npz')}
            assert names=={f'fc_{n}.npz' for n in NAMES},(name,names)
            meta['runs'][name]={'fwd_days':fwd,'bwd_days':bwd,
                                'forecast_sha256':{p.name:sha(p) for p in (out/name).glob('fc_*.npz')}}
            (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
        compare=base+['compare-slot','--cache',args.cache,'--contexts',str(ctx),
                      '--reference',str(out/'full'),'--windows',*NAMES]
        for name,_,_ in CUTS[1:]:compare.extend(['--arms',f'{name}={out/name}'])
        compare.extend(['--out',str(out/'comparison')])
        run(compare,'compare_cuts')
        run(base+['compare-slot','--cache',args.cache,'--contexts',str(ctx),
                  '--reference',args.contexts,'--arms',f'torch_full={out/"full"}',
                  '--windows',*NAMES,'--out',str(out/'mlx_comparison')],'compare_mlx')
        score=pd.read_csv(out/'comparison'/'bakeoff_windows.csv')
        summ=pd.read_csv(out/'comparison'/'bakeoff_summary.csv')
        assert set(score.arm)=={x[0] for x in CUTS[1:]}
        assert set(score.window)==set(NAMES)
        lines=['# Experiment 3: TimesFM-3 context length','',
               'PyTorch full context is the reference. Percent gain is positive when a cut improves the score. The standard `compare-slot` scorer and 4,000-resample bootstrap are unchanged.','',
               '## Pooled rolling windows','',summ.to_markdown(index=False),'',
               '## Window scores','',score.to_markdown(index=False),'',
               '## MLX versus PyTorch full context','',
               pd.read_csv(out/'mlx_comparison'/'bakeoff_windows.csv').to_markdown(index=False),'']
        (out/'REPORT.md').write_text('\n'.join(lines))
        meta['status']='success'
    except BaseException:
        meta['status']='failed';meta['error']=traceback.format_exc()
        (out/'failure.log').write_text(meta['error'])
        pd.DataFrame([{'error':meta['error']}]).to_csv(out/'failure.csv',index=False)
        print(meta['error'],file=sys.stderr)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'checksums.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in out.rglob('*')
                                                if f.is_file() and not f.is_symlink() and f.name!='checksums.json'},indent=2)+'\n')
    return int(meta['status']!='success')

if __name__=='__main__':sys.exit(main())
