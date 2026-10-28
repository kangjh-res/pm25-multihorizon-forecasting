#!/usr/bin/env python3
"""Compare archived XGBoost prediction RMSE with published SI Table S3.
Reads user-supplied predictions; never modifies source files. A match is not a full
end-to-end raw-data reproducibility validation.
"""
import argparse,csv,json,math
from pathlib import Path
import pandas as pd
import numpy as np
MODEL_COLUMNS={
 'M0':'pred_M0_persistence','M1':'pred_M1_local',
 'M2':'pred_M2_obsmet','M3':'pred_M3_fcstmet',
 'M4':'pred_M4_transport','M5':'pred_M5_analogue',
 'M6':'pred_M5_CAMS'}

def calculate(frame, model, lead):
 p=frame.loc[frame['lead_h']==lead,['PM25_target',MODEL_COLUMNS[model]]].apply(pd.to_numeric,errors='coerce')
 ok=np.isfinite(p.iloc[:,0].to_numpy()) & np.isfinite(p.iloc[:,1].to_numpy())
 a=p.iloc[:,0].to_numpy()[ok];b=p.iloc[:,1].to_numpy()[ok]
 return (float(np.sqrt(np.mean((a-b)**2))) if len(a) else None),int(len(a))

def verify(frame,reference,tolerance=0.00051):
 results=[]
 for r in reference:
  lead=int(r['lead_h']);model=r['model'];expected=float(r['rmse_paper'])
  actual,n=calculate(frame,model,lead)
  delta=(actual-expected) if actual is not None else None
  results.append({'lead_h':lead,'model':model,'n':n,'rmse_paper':expected,
    'rmse_recomputed':actual,'delta':delta,
    'status':'PASS' if delta is not None and abs(delta)<=tolerance else 'FAIL'})
 return results

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--predictions',required=True,type=Path)
 ap.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[1]/'references'/'si_table_s3_rmse.csv')
 ap.add_argument('--out-dir',required=True,type=Path)
 ap.add_argument('--tolerance',type=float,default=0.00051)
 a=ap.parse_args()
 if a.tolerance<0:ap.error('tolerance must be nonnegative')
 if not a.predictions.is_file():ap.error('prediction file missing')
 for source in [a.predictions,a.reference]:
  if a.out_dir.resolve()==source.resolve() or source.resolve().is_relative_to(a.out_dir.resolve()):
   ap.error('output directory must not contain source inputs')
 frame=pd.read_parquet(a.predictions)
 required={'lead_h','PM25_target',*MODEL_COLUMNS.values()}
 missing=required-set(frame.columns)
 if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
 with a.reference.open(encoding='utf-8-sig',newline='') as f:ref=list(csv.DictReader(f))
 results=verify(frame,ref,a.tolerance)
 a.out_dir.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(results).to_csv(a.out_dir/'table_s3_comparison.csv',index=False,encoding='utf-8-sig')
 summary={'status':'PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
          'checks':len(results),'passed':sum(r['status']=='PASS' for r in results),
          'failed':sum(r['status']=='FAIL' for r in results),
          'tolerance':a.tolerance,'note':'Archived prediction verification only; not raw-data reproduction.'}
 (a.out_dir/'table_s3_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
 if summary['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
