#!/usr/bin/env python3
"""Recalculate paired archived 2024 XGBoost/MLP metrics; no training or file modification.

Requires separately supplied original archived prediction parquets; no data redistribution.
"""
from __future__ import annotations
import argparse,json,pathlib,numpy as np,pandas as pd
KEYS=['측정소코드','issue_time_kst','lead_h']
def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok]
    if len(y)==0:return {'n':0,'r2':None,'rmse':None,'mae':None}
    e=y-p;den=((y-y.mean())**2).sum()
    return {'n':len(y),'r2':float(1-(e**2).sum()/den) if den else None,'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.mean(np.abs(e)))}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xgb',type=pathlib.Path,required=True)
    ap.add_argument('--mlp',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args()
    if args.out.resolve() in (args.xgb.resolve(),args.mlp.resolve()):ap.error('output cannot overwrite an input')
    x=pd.read_parquet(args.xgb);a=pd.read_parquet(args.mlp)
    for name,d in [('XGB',x),('MLP',a)]:
        missing=[k for k in KEYS+['PM25_target'] if k not in d.columns]
        if missing:raise ValueError(f'{name} missing {missing}')
        if d.duplicated(KEYS).any():raise ValueError(f'{name} duplicate keys')
    xx=[c for c in x if c.startswith('pred_')];aa=[c for c in a if c.startswith('pred_')]
    joined=x[KEYS+['PM25_target']+xx].merge(a[KEYS+['PM25_target']+aa],on=KEYS,how='inner',validate='one_to_one',suffixes=('_xgb','_mlp'))
    t1=pd.to_numeric(joined['PM25_target_xgb'],errors='coerce');t2=pd.to_numeric(joined['PM25_target_mlp'],errors='coerce')
    valid=np.isfinite(t1)&np.isfinite(t2)
    mismatches=int((~np.isclose(t1[valid],t2[valid],rtol=0,atol=1e-8)).sum())
    if mismatches:raise ValueError(f'joined target mismatch count={mismatches}')
    joined=joined.loc[valid].copy();joined['target']=t1.loc[valid]
    rows=[]
    for group,part in [('all',joined),('1-3h',joined[joined.lead_h.between(1,3)]),('4-12h',joined[joined.lead_h.between(4,12)]),('13-24h',joined[joined.lead_h.between(13,24)])]:
        for model,col in [('XGB '+c,c) for c in xx]+[('MLP '+c,c) for c in aa]:
            m=metrics(part.target,part[col]);rows.append({'group':group,'model':model,**m})
    for lead,part in joined.groupby('lead_h',sort=True):
        for model,col in [('XGB '+c,c) for c in xx]+[('MLP '+c,c) for c in aa]:
            rows.append({'group':f'lead_{int(lead):02d}','model':model,**metrics(part.target,part[col])})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.suffix.lower()!='.csv':ap.error('--out must end with .csv')
    pd.DataFrame(rows).to_csv(args.out,index=False,encoding='utf-8-sig')
    print(json.dumps({'status':'OK','paired_rows':len(joined),'target_mismatch':mismatches,'metrics_csv':str(args.out),'model_columns':{'xgb':xx,'mlp':aa}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
