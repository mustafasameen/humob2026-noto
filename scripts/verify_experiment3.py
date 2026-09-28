#!/usr/bin/env python3
"""Audit one-sided forecast changes and saved context comparison tables."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from humob26 import windows

NAMES=('may_jun','late_jan','april','january',*windows.ROLL_WINDOW_NAMES)
CUTS={'fwd14':('fwd',14),'fwd28':('fwd',28),'fwd56':('fwd',56),
      'bwd30':('bwd',30),'bwd60':('bwd',60),'bwd120':('bwd',120)}

def verify(root):
    root=Path(root)
    rows=[]
    for name in NAMES:
        ctx=np.load(root/'contexts'/f'ctx_{name}.npz')
        full=np.load(root/'full'/f'fc_{name}.npz')
        n=len(ctx['keys'])
        assert ctx['is_diag'].sum()>0
        for arm,(cut_leg,limit) in CUTS.items():
            fc=np.load(root/arm/f'fc_{name}.npz')
            for leg in ('fwd','bwd'):
                equal=[]
                for suffix in ('','_q10','_q90'):
                    k=leg+suffix
                    assert fc[k].shape==full[k].shape and fc[k].shape[0]==n
                    assert np.isfinite(fc[k][ctx['is_diag']]).all()
                    equal.append(np.array_equal(fc[k],full[k]))
                unchanged=(leg!=cut_leg or ctx[leg].shape[1]<=limit)
                if unchanged:
                    assert all(equal),(name,arm,leg,'unexpected difference')
                else:
                    assert not all(equal),(name,arm,leg,'cut had no effect')
                rows.append((name,arm,leg,ctx[leg].shape[1],limit if leg==cut_leg else None,all(equal)))
    w=pd.read_csv(root/'comparison'/'bakeoff_windows.csv')
    s=pd.read_csv(root/'comparison'/'bakeoff_summary.csv')
    gates=pd.read_csv(root/'comparison'/'bakeoff_gates.csv')
    assert len(w)==len(NAMES)*len(CUTS)
    assert set(w.window)==set(NAMES) and set(w.arm)==set(CUTS)
    assert set(s.arm)==set(CUTS)
    assert (gates.loc[gates.gate.isin(['1_reference_matches_final','3_files']),'ok'].astype(str).str.lower()=='true').all()
    np.testing.assert_allclose(w.pct_better,100*(1-w.combined/w.reference),rtol=1e-12,atol=1e-12)
    assert w[['combined','reference','pct_better']].notna().all().all()
    pd.DataFrame(rows,columns=['window','arm','leg','available_context_days','cut_days','exactly_equal_to_full']).to_csv(root/'leg_verification.csv',index=False)
    print(f'PASS: {len(rows)} leg comparisons, {len(w)} scored window-arm pairs, reference/file gates and percent calculations.')

if __name__=='__main__': verify(sys.argv[1])
