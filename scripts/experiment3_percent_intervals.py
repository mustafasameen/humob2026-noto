#!/usr/bin/env python3
"""Compute date-cluster bootstrap intervals directly on pooled percentage gains."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from humob26 import bakeoff, data, windows

NAMES=('may_jun','late_jan','april','january',*windows.ROLL_WINDOW_NAMES)
ARMS=('fwd14','fwd28','fwd56','bwd30','bwd60','bwd120')

def main(root,cache):
    root=Path(root)
    days,na=data.load_eval_box(None,cache=Path(cache)/'eval_box.npz')
    all_dates=sorted(d for d in days if d not in na)
    score,daily,gates=bakeoff.run(days,na,all_dates,root/'contexts',root/'full',
                                 {n:root/n for n in ARMS},window_names=NAMES,reps=4000)
    saved=pd.read_csv(root/'comparison'/'bakeoff_windows.csv')
    merged=score.merge(saved,on=['window','arm'],suffixes=('','_saved'))
    assert len(merged)==len(score)
    np.testing.assert_allclose(merged.combined,merged.combined_saved,rtol=0,atol=1e-12)
    pct=np.array([2.5,97.5])
    corrected=bakeoff.bonferroni_percentiles(len(ARMS))
    rows=[]
    for arm in ARMS:
        part=daily[(daily.arm==arm)&daily.roll]
        groups=part.groupby('date')[['diff','reference']].sum().sort_index()
        sums=groups.to_numpy()
        n=len(groups)
        draws=np.random.default_rng(0).integers(n,size=(4000,n))
        sampled=sums[draws].sum(axis=1)
        gains=-100*sampled[:,0]/sampled[:,1]
        point=-100*part['diff'].sum()/part.reference.sum()
        lo,hi=np.percentile(gains,pct)
        blo,bhi=np.percentile(gains,corrected)
        rows.append({'arm':arm,'pooled_gain_percent':point,'ci95_lo':lo,'ci95_hi':hi,
                     'familywise95_lo':blo,'familywise95_hi':bhi,'resamples':4000,'seed':0,
                     'unique_dates':n})
    result=pd.DataFrame(rows)
    result.to_csv(root/'pooled_gain_percent.csv',index=False)
    summary=pd.read_csv(root/'comparison'/'bakeoff_summary.csv')
    check=result.merge(summary,on='arm')
    np.testing.assert_allclose(check.pooled_gain_percent,check.pooled_pct,rtol=0,atol=1e-10)
    print(result.to_string(index=False))

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
