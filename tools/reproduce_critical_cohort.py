#!/usr/bin/env python3
"""Reproduce the strict 24-h rapid-onset critical cohort from archived XGBoost predictions.
Read-only inputs; writes new CSV/JSON only to --out-dir. No model training.
This reproduces the selection logic of original phase4 cell 30; numerical equivalence
requires the original 2024 prediction and feature parquet files.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
KEY=['측정소코드','issue_time_kst']; JOIN=KEY+['lead_h']
FEATURES=['WS10_fcst','PM25_CAMS_ugm3','upwind_PM25_weighted']
PRED=['PM25_target','PM25_t','pred_M5_analogue','pred_M5_CAMS']

def reproduce(pred:pd.DataFrame,feat:pd.DataFrame,high_pm=35.0):
    for label,frame,columns in [('predictions',pred,JOIN+PRED),('features',feat,JOIN+FEATURES)]:
        missing=sorted(set(columns)-set(frame.columns))
        if missing: raise ValueError(f'{label} missing columns: {missing}')
        if frame.duplicated(JOIN).any(): raise ValueError(f'{label} duplicate station/issue/lead keys')
    pred=pred.copy();feat=feat.copy()
    for f in (pred,feat):
        f['측정소코드']=f['측정소코드'].astype(str)
        f['issue_time_kst']=pd.to_datetime(f['issue_time_kst'])
        f['lead_h']=pd.to_numeric(f['lead_h'],errors='raise').astype(int)
    df=pred.merge(feat[JOIN+FEATURES],on=JOIN,how='left',validate='one_to_one',indicator=True)
    unmatched=int((df['_merge']!='both').sum());df=df.drop(columns='_merge')
    high=df.loc[df.PM25_target>=high_pm]
    if high.empty:raise ValueError('no high-PM cases; cannot estimate thresholds')
    thresholds={'wind_q67':float(high.WS10_fcst.quantile(.67)),
      'cams_q33':float(high.PM25_CAMS_ugm3.quantile(.33)),
      'upwind_q33':float(high.upwind_PM25_weighted.quantile(.33))}
    if not all(np.isfinite(v) for v in thresholds.values()):raise ValueError('invalid regime thresholds')
    h24=df.loc[df.lead_h==24].copy()
    new_onset=(h24.PM25_t<high_pm)&(h24.PM25_target>=high_pm)
    critical=(h24.WS10_fcst>=thresholds['wind_q67'])&(h24.PM25_CAMS_ugm3<=thresholds['cams_q33'])&(h24.upwind_PM25_weighted<=thresholds['upwind_q33'])
    events=h24.loc[new_onset,KEY].copy();events['regime']=np.where(critical.loc[new_onset],'Critical','Other')
    traj=df.merge(events,on=KEY,how='inner',validate='many_to_one')
    traj=traj.loc[traj.lead_h.between(1,24)].copy()
    traj['complete_row']=traj[['PM25_target','pred_M5_analogue','pred_M5_CAMS']].notna().all(axis=1)
    if traj.empty: complete=traj.copy()
    else:
        checks=traj.groupby(KEY,sort=False).agg(n=('lead_h','size'),unique=('lead_h','nunique'),minlead=('lead_h','min'),maxlead=('lead_h','max'),valid=('complete_row','sum'))
        good=checks.index[(checks.n==24)&(checks.unique==24)&(checks.minlead==1)&(checks.maxlead==24)&(checks.valid==24)]
        complete=traj.set_index(KEY).loc[lambda x:x.index.isin(good)].reset_index()
    summary={'input_prediction_rows':int(len(pred)),'unmatched_feature_rows':unmatched,'high_pm_rows':int(len(high)),
      'thresholds':thresholds,'new_onset_24h':int(len(events)),
      'critical_new_onset_24h':int((events.regime=='Critical').sum()),
      'critical_complete_events':int(complete.loc[complete.regime=='Critical',KEY].drop_duplicates().shape[0]),
      'complete_events_all_regimes':int(complete[KEY].drop_duplicates().shape[0]),
      'complete_rows':int(len(complete)),'expected_reference_critical_new_onset':550,
      'expected_reference_critical_complete':449}
    if not complete.empty:
        rows=[]
        for (regime,lead),g in complete.groupby(['regime','lead_h'],sort=True):
            for col in ['pred_M5_analogue','pred_M5_CAMS']:
                y=g.PM25_target.to_numpy(float);p=g[col].to_numpy(float);e=p-y
                rows.append({'regime':regime,'lead_h':int(lead),'model':col,'n':len(g),
                  'observed_mean':float(y.mean()),'predicted_mean':float(p.mean()),
                  'rmse':float(np.sqrt(np.mean(e*e))),'mean_bias':float(e.mean())})
        table=pd.DataFrame(rows)
    else:table=pd.DataFrame(columns=['regime','lead_h','model','n','observed_mean','predicted_mean','rmse','mean_bias'])
    return summary,table,complete

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True,type=Path);p.add_argument('--features',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path);a=p.parse_args()
    out=a.out_dir.resolve();ins=[a.predictions.resolve(),a.features.resolve()]
    if any(out==x or out in x.parents for x in ins):p.error('output directory cannot contain input files')
    pred=pd.read_parquet(a.predictions,columns=JOIN+PRED)
    feat=pd.read_parquet(a.features,columns=JOIN+FEATURES)
    summary,table,complete=reproduce(pred,feat)
    out.mkdir(parents=True,exist_ok=True)
    (out/'cohort_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    table.to_csv(out/'trajectory_metrics.csv',index=False,encoding='utf-8-sig')
    complete[KEY+['regime']].drop_duplicates().to_csv(out/'complete_event_keys.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['critical_new_onset_24h']!=550 or summary['critical_complete_events']!=449:
        print('WARNING: cohort counts differ from SI reference; inspect input provenance and missingness.')
if __name__=='__main__':main()
