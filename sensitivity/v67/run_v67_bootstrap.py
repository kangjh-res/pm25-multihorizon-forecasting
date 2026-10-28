#!/usr/bin/env python3
"""v67: paired issue-time and day-block bootstrap on final v66 predictions; no retraining."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PRED={'M5':'pred_M5','current':'pred_M5_current_CAMS','prior_delaunay':'pred_M5_prior_CAMS','prior_bilinear':'pred_M5_prior_bilinear'}
PAIRS=[('M5_minus_prior_bilinear','M5','prior_bilinear'),('prior_bilinear_minus_current','prior_bilinear','current'),('prior_delaunay_minus_bilinear','prior_delaunay','prior_bilinear'),('M5_minus_current','M5','current')]

def metrics(sums, counts):
    return np.sqrt(sums/counts)

def run_boot(sub, cluster_col, repetitions, seed, batch=100):
    agg=sub.groupby(cluster_col,sort=True)[['se_'+k for k in PRED]].sum()
    counts=sub.groupby(cluster_col,sort=True).size().to_numpy(dtype='float64')
    sums=agg.to_numpy(dtype='float64'); n=len(sums)
    if n<2:raise ValueError('Not enough clusters')
    rng=np.random.default_rng(seed)
    diffs={name:[] for name,_,_ in PAIRS}
    for start in range(0,repetitions,batch):
        b=min(batch,repetitions-start)
        idx=rng.integers(0,n,size=(b,n))
        rmse=np.sqrt(sums[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None])
        for name,a,c in PAIRS:
            ia=list(PRED).index(a); ic=list(PRED).index(c)
            diffs[name].extend((rmse[:,ia]-rmse[:,ic]).tolist())
    point=np.sqrt(sub[['se_'+k for k in PRED]].mean().to_numpy())
    rows=[]
    for name,a,c in PAIRS:
        arr=np.array(diffs[name]); estimate=point[list(PRED).index(a)]-point[list(PRED).index(c)]
        rows.append({'comparison':name,'point_rmse_difference':float(estimate),'ci95_low':float(np.quantile(arr,.025)), 'ci95_high':float(np.quantile(arr,.975)), 'bootstrap_probability_difference_le_zero':float(np.mean(arr<=0))})
    return n,rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v66-dir',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261028)
    a=p.parse_args(); src=Path(a.v66_dir); out=Path(a.out_dir)
    if out.exists():raise FileExistsError(f'Output directory exists: {out}')
    out.mkdir(parents=True)
    frames=[]
    for h in range(1,25):
        path=src/f'predictions_h{h:02d}.parquet'
        if not path.exists():raise FileNotFoundError(f'Missing v66 prediction file: {path}')
        d=pd.read_parquet(path)
        required=['측정소코드','issue_time_kst','lead_h','PM25_target']+list(PRED.values())
        missing=set(required)-set(d.columns)
        if missing:raise ValueError(f'{path}: missing {missing}')
        d=d[required].copy()
        if not (d.lead_h==h).all():raise ValueError(f'Lead mismatch: {path}')
        if d.duplicated(['측정소코드','issue_time_kst','lead_h']).any():raise ValueError(f'Duplicate keys: {path}')
        if not np.isfinite(d[['PM25_target']+list(PRED.values())].to_numpy(float)).all():raise ValueError(f'Nonfinite values: {path}')
        frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    # UTC-normalize only for grouping; KST day boundaries are explicitly converted.
    df['issue_utc']=pd.to_datetime(df.issue_time_kst,utc=True)
    df['issue_day_kst']=df.issue_utc.dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
    df['month_kst']=df.issue_utc.dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m')
    for k,col in PRED.items():df['se_'+k]=(df.PM25_target-df[col])**2
    observed={k:float(np.sqrt(df['se_'+k].mean())) for k in PRED}
    # Guard against accidentally selecting v64 incomplete results or a different experiment.
    expected={'M5':8.929986110133894,'current':8.072715084814448,'prior_delaunay':8.263961516805594,'prior_bilinear':8.259787329081862}
    if len(df)!=2117887 or any(abs(observed[k]-v)>0.0001 for k,v in expected.items()):
        raise ValueError(f'NOT v66 final matched evaluation: n={len(df)}, RMSE={observed}. Expected n=2117887 and {expected}. No bootstrap run.')
    groups={'all':df,'h01_12':df[df.lead_h<=12],'h13_24':df[df.lead_h>=13],'h24':df[df.lead_h==24]}
    results=[]
    for i,(label,sub) in enumerate(groups.items()):
        for mode,col in [('issue_time','issue_utc'),('calendar_day_KST','issue_day_kst')]:
            n_clusters, rows=run_boot(sub,col,a.bootstrap,a.seed+i*101+(0 if mode=='issue_time' else 1))
            for row in rows:results.append({'subset':label,'bootstrap_unit':mode,'n_rows':len(sub),'n_clusters':n_clusters,'n_bootstrap':a.bootstrap,**row})
    result=pd.DataFrame(results)
    result.to_csv(out/'v67_bootstrap_95CI.csv',index=False,encoding='utf-8-sig')
    monthly=[]
    for m,sub in df.groupby('month_kst',sort=True):
        r={'month':m,'n':len(sub)}
        for k in PRED:r['rmse_'+k]=float(np.sqrt(sub['se_'+k].mean()))
        for name,x,y in PAIRS:r[name]=r['rmse_'+x]-r['rmse_'+y]
        monthly.append(r)
    pd.DataFrame(monthly).to_csv(out/'v67_monthly_audit.csv',index=False,encoding='utf-8-sig')
    report={'status':'COMPLETED','source_dir':str(src),'n_rows':len(df),'n_issue_times':int(df.issue_utc.nunique()),'n_calendar_days_KST':int(df.issue_day_kst.nunique()),'pooled_rmse':observed,'bootstrap_repetitions':a.bootstrap,'seed':a.seed,'caveats':['Paired bootstrap intervals are percentile intervals for RMSE differences; bootstrap tail proportions are descriptive, not formal hypothesis-test p-values.','Issue-time resampling does not fully capture multi-day serial correlation. Calendar-day resampling is an additional robustness check, but also does not capture longer meteorological episodes.','Current-cycle CAMS data provenance and real-time dissemination availability have not been independently verified.','1-12h prior-cycle Delaunay and bilinear predictions share the same original station-based input; differences in this interval do not test interpolation methods independently.']}
    (out/'v67_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETED', 'n=',len(df),'issue_clusters=',report['n_issue_times'],'day_clusters=',report['n_calendar_days_KST'])
    print(result[(result.subset.isin(['all','h24'])) & (result.bootstrap_unit=='calendar_day_KST')].to_string(index=False))
    print('OUTPUT:',out)
if __name__=='__main__':main()
