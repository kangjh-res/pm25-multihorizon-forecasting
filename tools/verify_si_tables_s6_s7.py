#!/usr/bin/env python3
"""Verify SI Tables S6 and S7 against archived 2024 inputs and v51 event outputs.
S6 uses horizon-specific medians on observed high-PM cases (>=35); S7 uses
fixed 449-event strict cohort and the v51 4-regime labels. No input writes.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pandas as pd
KEY=['측정소코드','issue_time_kst','lead_h']
F=['WS10_fcst','PM25_CAMS_ugm3','upwind_PM25_weighted']
REF=Path(__file__).resolve().parents[1]/'references'

def load_reference(name):
    with (REF/name).open(encoding='utf-8',newline='') as f:return list(csv.reader(f))

def s6(pred,feat):
    for label,df,cols in [('predictions',pred,KEY+['PM25_target','pred_M5_CAMS']),('features',feat,KEY+F)]:
        missing=set(cols)-set(df.columns)
        if missing:raise ValueError(f'{label} missing {sorted(missing)}')
        if df.duplicated(KEY).any():raise ValueError(f'{label} duplicate keys')
    df=pred[KEY+['PM25_target','pred_M5_CAMS']].merge(feat[KEY+F],on=KEY,how='left',validate='one_to_one')
    df=df[(df.PM25_target>=35)&df.pred_M5_CAMS.notna()].copy()
    rows=[]
    for h,g in df.groupby('lead_h'):
        if h not in [1,3,6,12,24]:continue
        med=g[F].median()
        crit=(g[F[0]]>=med[F[0]])&(g[F[1]]<=med[F[1]])&(g[F[2]]<=med[F[2]])
        for tag,subset in [('Critical',g[crit]),('Other',g[~crit])]:
            e=(subset.pred_M5_CAMS-subset.PM25_target).to_numpy(float)
            rows.append({'horizon':int(h),'regime':tag,'n':len(e),'rmse':float(np.sqrt(np.mean(e*e))) if len(e) else float('nan'),'bias':float(np.mean(e)) if len(e) else float('nan')})
    return pd.DataFrame(rows)

def s7(events):
    required=['wind_transition_regime','PM25_rise_15_18','Predicted_rise_15_18','PM25_target_h18','pred_M5_CAMS_h18']
    missing=set(required)-set(events.columns)
    if missing:raise ValueError(f'wind events missing {sorted(missing)}')
    rows=[]
    for tag,g in events.groupby('wind_transition_regime'):
        err=(g.pred_M5_CAMS_h18-g.PM25_target_h18).to_numpy(float)
        rows.append({'regime':tag,'n':len(g),'observed_delta':float(g.PM25_rise_15_18.mean()),'predicted_delta':float(g.Predicted_rise_15_18.mean()),'rmse18':float(np.sqrt(np.mean(err*err))),'bias18':float(np.mean(err))})
    return pd.DataFrame(rows)

def compare(calc,table):
    refs=load_reference(table)
    checks=[]
    if table=='si_table_s6.csv':
        for r in refs[1:]:
            h=int(r[0].replace('h','').strip())
            for tag,n,rmse,bias in [('Critical',r[1],r[3],r[5]),('Other',r[2],r[4],r[6])]:
                found=calc[(calc.horizon==h)&(calc.regime==tag)]
                for field,reference,tol in [('n',n,0),('rmse',rmse,.0051),('bias',bias,.0051)]:
                    value=float(found.iloc[0][field]) if len(found)==1 else float('nan')
                    target=float(reference.replace(',','').replace(' ','').replace('−','-'))
                    checks.append({'table':'S6','group':f'{h}h {tag}','metric':field,'calculated':value,'reference':target,'tolerance':tol,'status':'PASS' if math.isfinite(value) and abs(value-target)<=tol else 'FAIL'})
    else:
        for r in refs[1:]:
            tag=r[0];found=calc[calc.regime==tag]
            for field,reference,tol in [('n',r[1],0),('observed_delta',r[2],.0051),('predicted_delta',r[3],.0051),('rmse18',r[4],.0051),('bias18',r[5],.0051)]:
                value=float(found.iloc[0][field]) if len(found)==1 else float('nan');target=float(reference.replace(',','').replace(' ','').replace('−','-'))
                checks.append({'table':'S7','group':tag,'metric':field,'calculated':value,'reference':target,'tolerance':tol,'status':'PASS' if math.isfinite(value) and abs(value-target)<=tol else 'FAIL'})
    return checks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path);ap.add_argument('--features',type=Path)
    ap.add_argument('--wind-events',type=Path,help='v51 wind_transition_event_level.csv')
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();out=a.out_dir.resolve();inputs=[p.resolve() for p in [a.predictions,a.features,a.wind_events] if p]
    if any(out==p or out in p.parents for p in inputs):ap.error('out-dir cannot contain input files')
    if bool(a.predictions)!=bool(a.features):ap.error('--predictions and --features must be supplied together')
    if not inputs:ap.error('provide prediction/feature parquets and/or wind events CSV')
    checks=[];out.mkdir(parents=True,exist_ok=True)
    if a.predictions:
        p=pd.read_parquet(a.predictions,columns=KEY+['PM25_target','pred_M5_CAMS'])
        f=pd.read_parquet(a.features,columns=KEY+F)
        result=s6(p,f);result.to_csv(out/'computed_s6.csv',index=False)
        checks+=compare(result,'si_table_s6.csv')
    if a.wind_events:
        events=pd.read_csv(a.wind_events)
        result=s7(events);result.to_csv(out/'computed_s7.csv',index=False)
        checks+=compare(result,'si_table_s7.csv')
    with (out/'checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['table','group','metric','calculated','reference','tolerance','status']);w.writeheader();w.writerows(checks)
    summary={'checks':len(checks),'passed':sum(r['status']=='PASS' for r in checks),'failed':sum(r['status']=='FAIL' for r in checks),'note':'Only supplied SI Tables S6-S7; reference agreement requires actual original archived inputs.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));return 0 if not summary['failed'] else 2
if __name__=='__main__':raise SystemExit(main())
