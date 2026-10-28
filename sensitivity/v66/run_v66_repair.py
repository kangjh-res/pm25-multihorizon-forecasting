#!/usr/bin/env python3
"""v64: re-extract previous-cycle h25-36 with original station bilinear mapping;
train only revised prior-CAMS XGB, compare on exact v60 test keys. Read-only inputs.
"""
import argparse,json,traceback
from pathlib import Path
import numpy as np,pandas as pd
KEY=['측정소코드','issue_time_kst','lead_h']
def norm(s):return pd.to_datetime(s,utc=True)
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=y-p
 return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'r2':float(1-(e*e).sum()/((y-y.mean())**2).sum())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--research-root',required=True);a.add_argument('--search-root',required=True);a.add_argument('--v60-dir',required=True);a.add_argument('--out-dir',required=True);a.add_argument('--stage-only',action='store_true');a.add_argument('--reuse-grib',default=None,help='Path to prior_bilinear_grib_25_36.parquet from previous failed v64');z=a.parse_args()
 root=Path(z.research_root);search=Path(z.search_root);v60=Path(z.v60_dir);out=Path(z.out_dir)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 report={'status':'STARTED','limitations':['Retrospective previous-cycle alignment does not verify operational CAMS availability.','The current-cycle station CAMS file is used as supplied; original production lineage must still be verified.']}
 try:
  import eccodes
  from xgboost import XGBRegressor
  archive=list(search.rglob('cams_h36_archive_v40/grib'))
  if len(archive)!=1:raise ValueError(f'Expected one archive: {archive}')
  archive=archive[0]
  mapping=pd.read_csv(root/'data/interim/airkorea_133_cams_bilinear_mapping.csv',encoding='utf-8-sig',dtype={'측정소코드':str})
  mapping['corner']=pd.to_numeric(mapping.corner);mapping['cams_grid_id']=pd.to_numeric(mapping.cams_grid_id);mapping['weight']=pd.to_numeric(mapping.weight)
  mapping=mapping.sort_values(['측정소코드','corner'])
  if len(mapping)!=532 or mapping.duplicated(['측정소코드','corner']).any():raise ValueError('Original mapping must have 133 stations x 4 corners')
  if not np.allclose(mapping.groupby('측정소코드').weight.sum().values,1,atol=1e-9):raise ValueError('Mapping weights do not sum to 1')
  stations=mapping['측정소코드'].drop_duplicates().to_numpy()
  idx=mapping.cams_grid_id.to_numpy(int).reshape(-1,4);weights=mapping.weight.to_numpy(float).reshape(-1,4)
  # Original mapping grid IDs were verified in v63 against eccodes flattened grid indices.
  if z.reuse_grib:
   reuse=Path(z.reuse_grib)
   if not reuse.is_file():raise FileNotFoundError(reuse)
   long=pd.read_parquet(reuse)
   if not {'측정소코드','prior_issue_utc','prior_lead_h','PM25_CAMS_prior_bilinear'}.issubset(long.columns):raise ValueError('Bad reused GRIB parquet columns')
   long['prior_issue_utc']=norm(long.prior_issue_utc)
   units='kg m**-3'
   long.to_parquet(out/'prior_bilinear_grib_25_36.parquet',index=False)
   print('REUSED GRIB:',reuse,flush=True)
  else:
   dates=pd.date_range('2022-01-01','2024-12-31',freq='D')
   records=[];units=None;reference_grid=None
   for di,date in enumerate(dates):
    f=archive/str(date.year)/f'cams_{date:%Y%m%d}_00-12_met_pm25_h025-036.grib'
    if not f.exists():raise FileNotFoundError(f)
    seen=set()
    with f.open('rb') as handle:
     while True:
      gid=eccodes.codes_grib_new_from_file(handle)
      if gid is None:break
      try:
       if str(eccodes.codes_get(gid,'shortName'))!='pm2p5':continue
       day=int(eccodes.codes_get(gid,'dataDate'));hour=int(eccodes.codes_get(gid,'dataTime'))//100;lead=int(eccodes.codes_get(gid,'endStep'))
       key=(day,hour,lead)
       if hour not in (0,12) or lead not in range(25,37) or key in seen:raise ValueError(f'Bad/duplicate GRIB message {f}: {key}')
       seen.add(key)
       unit=str(eccodes.codes_get(gid,'units'))
       if unit not in ('kg m**-3','kg m-3','kg/m3','kg m^-3'):raise ValueError(f'Unexpected units: {unit}')
       if units is None:units=unit
       if units!=unit:raise ValueError('Units changed')
       lat=np.asarray(eccodes.codes_get_array(gid,'latitudes'));lon=np.asarray(eccodes.codes_get_array(gid,'longitudes'))
       if reference_grid is None:reference_grid=(lat,lon)
       elif not(np.array_equal(lat,reference_grid[0]) and np.array_equal(lon,reference_grid[1])):raise ValueError(f'GRIB grid changed {f}')
       vals=np.asarray(eccodes.codes_get_array(gid,'values'),float)
       if idx.max()>=len(vals):raise ValueError('Mapping index out of GRIB range')
       station_vals=(vals[idx]*weights).sum(axis=1)*1e9
       if not np.isfinite(station_vals).all() or (station_vals<0).any():raise ValueError(f'Invalid interpolated values: {f}')
       issue=pd.Timestamp(str(day),tz='UTC')+pd.Timedelta(hours=hour)
       records.append(pd.DataFrame({'측정소코드':stations,'prior_issue_utc':issue,'prior_lead_h':lead,'PM25_CAMS_prior_bilinear':station_vals}))
      finally:eccodes.codes_release(gid)
    if len(seen)!=24:raise ValueError(f'Expected 24 GRIB PM messages: {f}, found {len(seen)}')
    if (di+1)%100==0:print(f'GRIB {di+1}/{len(dates)}',flush=True)
   long=pd.concat(records,ignore_index=True);del records
   long.to_parquet(out/'prior_bilinear_grib_25_36.parquet',index=False)
  report['grib']={'files':int(long['prior_issue_utc'].nunique()//2),'rows':len(long),'units':units,'mapping_rows':len(mapping),'stations':len(stations)}
  # Reuse v60's ALREADY VERIFIED previous-cycle h1-12 joins.
  # The previous v64 implementation joined raw station forecasts with inconsistent
  # year partitioning and lost nearly all 2023 short-lead rows.
  short_by_year={}
  for year in (2022,2023,2024):
   f=v60/f'prior_cams_{year}.parquet'
   if not f.is_file():raise FileNotFoundError(f)
   d=pd.read_parquet(f,columns=KEY+['PM25_CAMS_prior'])
   d['측정소코드']=d['측정소코드'].astype(str)
   d['issue_time_kst']=norm(d.issue_time_kst)
   d=d.loc[d.lead_h.between(1,12),KEY+['PM25_CAMS_prior']].rename(columns={'PM25_CAMS_prior':'PM25_CAMS_prior_bilinear'})
   if d.duplicated(KEY).any():raise ValueError(f'Duplicate v60 short keys {year}')
   short_by_year[year]=d
  report['short_source']='v60 verified prior_cams_YYYY.parquet, h01-12 (original station bilinear)'
  registry=list(search.rglob('final_feature_registry.json'))
  registry=[p for p in registry if 'config' in p.parts or 'processed' in p.parts]
  if not registry:raise FileNotFoundError('final_feature_registry.json')
  reg=json.loads(registry[0].read_text(encoding='utf-8'))['models'];base=list(reg['M5_analogue']);withcams=list(reg['M5_CAMS'])
  if set(withcams)-set(base)!={'PM25_CAMS_ugm3'}:raise ValueError('Unexpected feature registry')
  prior_cols=[('PM25_CAMS_prior_bilinear' if x=='PM25_CAMS_ugm3' else x) for x in withcams]
  prior={};coverage=[]
  for year in (2022,2023,2024):
   keys=pd.read_parquet(root/f'data/processed/model_features_{year}.parquet',columns=KEY)
   keys['측정소코드']=keys['측정소코드'].astype(str);keys['issue_time_kst']=norm(keys.issue_time_kst)
   if keys.duplicated(KEY).any():raise ValueError(f'Duplicate model keys {year}')
   keys['prior_issue_utc']=keys.issue_time_kst-pd.Timedelta(hours=12);keys['prior_lead_h']=keys.lead_h+12
   short_part=keys.loc[keys.lead_h.between(1,12),KEY].merge(short_by_year[year],on=KEY,how='left',validate='one_to_one')
   long_part=keys.loc[keys.lead_h.between(13,24)].merge(long,on=['측정소코드','prior_issue_utc','prior_lead_h'],how='left',validate='one_to_one')
   prior[year]=pd.concat([short_part[KEY+['PM25_CAMS_prior_bilinear']],long_part[KEY+['PM25_CAMS_prior_bilinear']]],ignore_index=True)
   short_valid=int(short_part.PM25_CAMS_prior_bilinear.notna().sum())
   short_expected=int(short_by_year[year].PM25_CAMS_prior_bilinear.notna().sum())
   if short_valid!=short_expected:raise ValueError(f'Short v60 join changed {year}: {short_valid} vs {short_expected}')
   prior[year].to_parquet(out/f'prior_bilinear_{year}.parquet',index=False)
   coverage.append({'year':year,'rows':len(prior[year]),'paired':int(prior[year].PM25_CAMS_prior_bilinear.notna().sum())})
  report['coverage']=coverage
  for rec in coverage:
   if rec['year'] in (2023,2024) and rec['paired'] != rec['rows']:
    raise ValueError(f'Unexpected missing rows after short repair: {rec}')
  if z.stage_only:report['status']='STAGED_ONLY';return
  params=dict(objective='reg:squarederror',n_estimators=300,learning_rate=.03,max_depth=5,min_child_weight=5,subsample=.9,colsample_bytree=.8,reg_alpha=0.,reg_lambda=20.,tree_method='hist',device='cpu',random_state=42,n_jobs=4)
  allrows=[];audit=[]
  for h in range(1,25):
   years=[]
   for year in (2022,2023,2024):
    cols=list(dict.fromkeys(KEY+['PM25_target']+withcams))
    d=pd.read_parquet(root/f'data/processed/model_features_{year}.parquet',columns=cols)
    d=d.loc[d.lead_h==h].copy();d['측정소코드']=d['측정소코드'].astype(str);d['issue_time_kst']=norm(d.issue_time_kst)
    d=d.merge(prior[year].loc[prior[year].lead_h==h],on=KEY,how='left',validate='one_to_one')
    good=np.isfinite(d[list(dict.fromkeys(['PM25_target']+withcams+['PM25_CAMS_prior_bilinear']))].to_numpy(dtype='float64')).all(axis=1)
    years.append(d.loc[good])
   train=pd.concat(years[:2],ignore_index=True);test=years[2]
   old=pd.read_parquet(v60/f'predictions_h{h:02d}.parquet')
   old['측정소코드']=old['측정소코드'].astype(str);old['issue_time_kst']=norm(old.issue_time_kst)
   if old.duplicated(KEY).any() or test.duplicated(KEY).any():raise ValueError(f'Duplicate test keys h{h}')
   # Compare on common evaluation keys, explicitly quantify lost coverage.
   candidate_n=len(test)
   test=test.merge(old,on=KEY,how='inner',validate='one_to_one',suffixes=('','_v60'))
   common_n=len(test)
   audit_coverage={'lead_h':h,'v60_rows':len(old),'v64_candidate_rows':candidate_n,'common_rows':common_n,'v60_only_rows':len(old)-common_n,'v64_only_rows':candidate_n-common_n}
   report.setdefault('comparison_coverage',[]).append(audit_coverage)
   if common_n==0 or common_n/len(old)<0.95:raise ValueError(f'Low comparison coverage h{h}: {audit_coverage}')
   print('COVERAGE',audit_coverage,flush=True)
   if not np.allclose(test.PM25_target,test.PM25_target_v60,atol=1e-5):raise ValueError('Target mismatch')
   if len(train)<100:raise ValueError(f'Insufficient train rows h{h}')
   m=XGBRegressor(**params);m.fit(train[prior_cols].to_numpy('float32'),train.PM25_target.to_numpy('float32'))
   test['pred_M5_prior_bilinear']=m.predict(test[prior_cols].to_numpy('float32'))
   keep=KEY+['PM25_target','pred_M5','pred_M5_current_CAMS','pred_M5_prior_CAMS','pred_M5_prior_bilinear']
   test[keep].to_parquet(out/f'predictions_h{h:02d}.parquet',index=False)
   for name,col in [('M5','pred_M5'),('current_CAMS','pred_M5_current_CAMS'),('prior_Delaunay','pred_M5_prior_CAMS'),('prior_bilinear','pred_M5_prior_bilinear')]:
    audit.append({'lead_h':h,'model':name,**metrics(test.PM25_target,test[col])})
   allrows.append(test[keep]);print(f'FIT h{h:02d}/24 test={len(test)} train={len(train)}',flush=True)
  pd.DataFrame(audit).to_csv(out/'v64_metrics_by_horizon.csv',index=False,encoding='utf-8-sig')
  alltest=pd.concat(allrows,ignore_index=True)
  summary=[]
  for name,col in [('M5','pred_M5'),('current_CAMS','pred_M5_current_CAMS'),('prior_Delaunay','pred_M5_prior_CAMS'),('prior_bilinear','pred_M5_prior_bilinear')]:summary.append({'model':name,**metrics(alltest.PM25_target,alltest[col])})
  pd.DataFrame(summary).to_csv(out/'v64_pooled_metrics.csv',index=False,encoding='utf-8-sig')
  report['metrics']=summary;report['status']='COMPLETED_WITH_COVERAGE_AUDIT' if any(x['v60_only_rows'] or x['v64_only_rows'] for x in report['comparison_coverage']) else 'COMPLETED'
  pd.DataFrame(report['comparison_coverage']).to_csv(out/'v64_comparison_coverage.csv',index=False,encoding='utf-8-sig')
 except Exception as e:
  report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
 finally:
  (out/'v64_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  print('REPORT:',out/'v64_report.json',report['status'],flush=True)
  if report['status']=='FAILED':print(report['traceback'],flush=True)
 if report['status']=='FAILED':raise SystemExit(1)
if __name__=='__main__':main()
