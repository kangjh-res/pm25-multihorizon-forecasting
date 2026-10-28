#!/usr/bin/env python3
"""Wind-transition analysis for the fixed critical-event cohort (SI Text S4).
Reads v50 cohort keys and archived 2024 prediction/feature parquets; does not modify inputs.
Original analysis reference: 대기질AI_phase4.ipynb, cell 36 (zero based).
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
KEY=['측정소코드','issue_time_kst']; JOIN=KEY+['lead_h']
PRED=['PM25_target','pred_M5_CAMS']
FEAT=['WS10_fcst','WD10_fcst']
def circular_abs(new,old):
    return np.abs((new-old+180)%360-180)
def compute(cohort,pred,features):
    for name,frame,required in [('cohort',cohort,KEY),('predictions',pred,JOIN+PRED),('features',features,JOIN+FEAT)]:
        miss=set(required)-set(frame.columns)
        if miss:raise ValueError(f'{name} missing columns: {sorted(miss)}')
    c=cohort.copy();p=pred.copy();f=features.copy()
    for frame in [c,p,f]:
        frame['측정소코드']=frame['측정소코드'].astype(str)
        frame['issue_time_kst']=pd.to_datetime(frame['issue_time_kst'],errors='raise')
    if 'regime' in c.columns:c=c.loc[c.regime.astype(str).str.lower()=='critical']
    # Legacy v56 cohort CSV contains 24 rows per event (one for each lead).
    # Accept only exact repeated event/regime labels, then reduce to event-level keys.
    if c.duplicated(KEY).any():
        if 'regime' in c.columns and c.groupby(KEY)['regime'].nunique(dropna=False).gt(1).any():
            raise ValueError('conflicting regime labels for the same cohort event')
        c=c.drop_duplicates(KEY).copy()
    if c.duplicated(KEY).any():raise ValueError('duplicate cohort keys after normalization')
    if p.duplicated(JOIN).any() or f.duplicated(JOIN).any():raise ValueError('duplicate prediction or feature keys')
    p=p.loc[p.lead_h.isin([15,18])].merge(c[KEY],on=KEY,validate='many_to_one')
    f=f.loc[f.lead_h.isin([12,15])].merge(c[KEY],on=KEY,validate='many_to_one')
    def wide(df,col,leads):
        a=df.pivot(index=KEY,columns='lead_h',values=col).reindex(columns=leads)
        a.columns=[f'{col}_h{h}' for h in leads]
        return a
    events=c.set_index(KEY)
    for col in PRED:events=events.join(wide(p,col,[15,18]))
    for col in FEAT:events=events.join(wide(f,col,[12,15]))
    events=events.reset_index()
    essential=[f'{x}_h{h}' for x in PRED for h in [15,18]]+[f'{x}_h{h}' for x in FEAT for h in [12,15]]
    missing=int(events[essential].isna().any(axis=1).sum())
    events=events.dropna(subset=essential).copy()
    if events.empty:raise ValueError('no complete events for h12/h15/h18')
    events['WD_shift_abs_12_15']=circular_abs(events.WD10_fcst_h15,events.WD10_fcst_h12)
    events['WS_change_12_15']=events.WS10_fcst_h15-events.WS10_fcst_h12
    events['PM25_rise_15_18']=events.PM25_target_h18-events.PM25_target_h15
    events['Predicted_rise_15_18']=events.pred_M5_CAMS_h18-events.pred_M5_CAMS_h15
    events['Missed_rise_15_18']=events.PM25_rise_15_18-events.Predicted_rise_15_18
    ws=float(events.WS10_fcst_h15.median());wd=float(events.WD_shift_abs_12_15.median())
    events['WS_state']=np.where(events.WS10_fcst_h15>=ws,'High WS','Low WS')
    events['WD_transition_state']=np.where(events.WD_shift_abs_12_15>=wd,'Large WD shift','Small WD shift')
    events['wind_transition_regime']=events.WS_state+' + '+events.WD_transition_state
    joint=events.groupby('wind_transition_regime',observed=True).agg(N=('PM25_rise_15_18','size'),WS_h15_mean=('WS10_fcst_h15','mean'),WD_shift_mean=('WD_shift_abs_12_15','mean'),Observed_rise_mean=('PM25_rise_15_18','mean'),Predicted_rise_mean=('Predicted_rise_15_18','mean'),Missed_rise_mean=('Missed_rise_15_18','mean')).reset_index()
    summary={'cohort_keys':int(len(c)),'complete_wind_event_keys':int(len(events)),'missing_required_data':missing,'wind_speed_median_h15':ws,'direction_shift_median_h12_h15':wd,'reference_cohort_n':449,'reference_ws_median':3.410,'reference_wd_median':25.389,'matches_reference_n':len(events)==449}
    return summary,events,joint
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',type=Path,required=True,help='v50 complete_event_keys.csv')
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();out=a.out_dir.resolve();inputs=[a.cohort.resolve(),a.predictions.resolve(),a.features.resolve()]
    if any(out==p or out in p.parents for p in inputs):ap.error('output folder must not contain input files')
    c=pd.read_csv(a.cohort);p=pd.read_parquet(a.predictions,columns=JOIN+PRED);f=pd.read_parquet(a.features,columns=JOIN+FEAT)
    summary,events,joint=compute(c,p,f)
    out.mkdir(parents=True,exist_ok=True)
    (out/'wind_transition_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    events.to_csv(out/'wind_transition_event_level.csv',index=False,encoding='utf-8-sig')
    joint.to_csv(out/'wind_transition_joint_regimes.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if not summary['matches_reference_n']:print('WARNING: cohort count differs from SI reference 449')
if __name__=='__main__':main()
