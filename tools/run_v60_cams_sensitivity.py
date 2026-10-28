#!/usr/bin/env python3
"""v60 read-only retrospective prior-cycle CAMS XGBoost sensitivity; all 24 horizons.

Requires eccodes, scipy, pyarrow, pandas, numpy, xgboost. CPU-only.
Never modifies source. Writes detailed report on error; checks time/units/grid.
This is retrospective cycle alignment, NOT verification of real-time provider latency.
"""
import argparse, json, traceback, datetime, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['측정소코드','issue_time_kst','lead_h']
def utc(s):return pd.to_datetime(s,utc=True)
def stats(y,p):
    y=np.asarray(y,dtype='float64');p=np.asarray(p,dtype='float64');e=y-p
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.mean(abs(e))),'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2))}
def write_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--research-root',required=True)
    ap.add_argument('--search-root',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--stage-only',action='store_true',help='Build and audit prior-cycle features without retraining')
    a=ap.parse_args()
    research=Path(a.research_root).resolve();search=Path(a.search_root).resolve();out=Path(a.out_dir).resolve()
    if out.exists():raise FileExistsError(f'Output folder already exists: {out}')
    if research==out or research in out.parents:raise ValueError('Output must not be within research root')
    out.mkdir(parents=True)
    report={'status':'STARTED','stages':{},'warnings':['Retrospective forecast-cycle matching is not proof of operational CAMS delivery latency.','Bilinear/triangulated spatial interpolation may differ from original station CAMS extraction; compare interpolation methodology before interpreting skill.']}
    try:
        import eccodes
        from scipy.spatial import Delaunay
        import pyarrow.parquet as pq
        import xgboost as xgb
        archive_candidates=list(search.rglob('cams_h36_archive_v40/grib'))
        if len(archive_candidates)!=1:raise ValueError(f'Expected exactly one GRIB archive, found {len(archive_candidates)}')
        archive=archive_candidates[0]
        coords=pd.read_csv(research/'data/interim/airkorea_candidate_stations_133_coordinates.csv',encoding='utf-8-sig',dtype={'측정소코드':str})
        required={'측정소코드','위도','경도'}
        if not required.issubset(coords):raise ValueError(f'Missing coordinate columns {required-set(coords)}')
        coords=coords[['측정소코드','위도','경도']].drop_duplicates('측정소코드')
        if len(coords)!=133 or coords.isna().any().any():raise ValueError('Coordinate file does not contain 133 unique valid stations')
        station_codes=coords['측정소코드'].astype(str).to_numpy()
        xy=coords[['경도','위도']].to_numpy(float)
        dates=pd.date_range('2022-01-01','2024-12-31',freq='D')
        records=[];tri=None;verts=None;weights=None;refgrid=None;units=None
        for ix,date in enumerate(dates):
            f=archive/str(date.year)/f'cams_{date:%Y%m%d}_00-12_met_pm25_h025-036.grib'
            if not f.is_file():raise FileNotFoundError(f)
            seen=set()
            with f.open('rb') as handle:
                while True:
                    gid=eccodes.codes_grib_new_from_file(handle)
                    if gid is None:break
                    try:
                        name=str(eccodes.codes_get(gid,'shortName'))
                        if name!='pm2p5':continue
                        day=int(eccodes.codes_get(gid,'dataDate'))
                        hour=int(eccodes.codes_get(gid,'dataTime'))//100
                        lead=int(eccodes.codes_get(gid,'endStep'))
                        if hour not in (0,12) or lead not in range(25,37):raise ValueError(f'Unexpected GRIB cycle/lead: {f} {hour} {lead}')
                        key=(day,hour,lead)
                        if key in seen:raise ValueError(f'Duplicate GRIB message {key} in {f}')
                        seen.add(key)
                        unit=str(eccodes.codes_get(gid,'units'))
                        if units is None:units=unit
                        if unit!=units:raise ValueError(f'Inconsistent units {unit} vs {units}')
                        if tri is None:
                            lats=np.asarray(eccodes.codes_get_array(gid,'latitudes'),float)
                            lons=np.asarray(eccodes.codes_get_array(gid,'longitudes'),float)
                            refgrid=np.column_stack([lons,lats]);tri=Delaunay(refgrid)
                            simplex=tri.find_simplex(xy)
                            if np.any(simplex<0):raise ValueError(f'{sum(simplex<0)} stations outside CAMS grid; cannot interpolate safely')
                            verts=tri.simplices[simplex]
                            transform=tri.transform[simplex]
                            bary=np.einsum('ijk,ik->ij',transform[:,:2,:],xy-transform[:,2,:])
                            weights=np.column_stack([bary,1-bary.sum(axis=1)])
                        vals=np.asarray(eccodes.codes_get_array(gid,'values'),float)
                        if len(vals)!=len(refgrid):raise ValueError(f'Grid point count changed: {f}')
                        if unit in ('kg m**-3','kg m-3','kg/m3','kg m^-3'):factor=1e9
                        elif unit in ('ug m**-3','µg m**-3','ug/m3','µg/m3'):factor=1
                        else:raise ValueError(f'Unrecognized PM2.5 GRIB units {unit!r}; manual verification required')
                        v=(vals[verts]*weights).sum(axis=1)*factor
                        if not np.isfinite(v).all() or np.any(v<0):raise ValueError(f'Nonfinite/negative interpolated CAMS: {f}')
                        issue=pd.Timestamp(str(day),tz='UTC')+pd.Timedelta(hours=hour)
                        records.append(pd.DataFrame({'측정소코드':station_codes,'prior_issue_utc':issue,'prior_lead_h':lead,'PM25_CAMS_prior':v}))
                    finally:eccodes.codes_release(gid)
            if len(seen)!=24:raise ValueError(f'Expected 24 PM2.5 messages, found {len(seen)}: {f}')
            if ix%100==0:print(f'GRIB: {ix+1}/{len(dates)}',flush=True)
        prior_long=pd.concat(records,ignore_index=True)
        del records
        prior_long.to_parquet(out/'prior_grib_station_25_36.parquet',index=False)
        report['stages']['grib']={'files':len(dates),'pm25_records':len(prior_long),'units':units,'interpolation':'Delaunay barycentric linear','stations':len(coords)}
        # For h1-12, prior-cycle lead13-24 is already available in existing station forecast files.
        # Build a year-specific table keyed by current issue and lead; include boundary previous-year issues.
        cfiles=[]
        for year in (2022,2023,2024):
            f=research/f'data/processed/cams_station_forecast_{year}.parquet'
            if not f.is_file():raise FileNotFoundError(f)
            c=pd.read_parquet(f,columns=['측정소코드','issue_time_utc','lead_h','PM25_CAMS_ugm3'])
            c['측정소코드']=c['측정소코드'].astype(str)
            c['prior_issue_utc']=utc(c['issue_time_utc'])
            c=c.loc[c.lead_h.between(13,24),['측정소코드','prior_issue_utc','lead_h','PM25_CAMS_ugm3']]
            c=c.rename(columns={'lead_h':'prior_lead_h','PM25_CAMS_ugm3':'PM25_CAMS_prior'})
            cfiles.append(c)
        prior_short=pd.concat(cfiles,ignore_index=True).drop_duplicates(['측정소코드','prior_issue_utc','prior_lead_h'])
        prior_long['prior_issue_utc']=utc(prior_long['prior_issue_utc'])
        # A current issue at T requires previous cycle T-12h, previous lead h+12.
        # Issue date/time derived from source issue_time_kst, converted to UTC.
        reg_candidates=list(search.rglob('final_feature_registry.json'))
        reg_candidates=[x for x in reg_candidates if 'config' in x.parts or 'processed' in x.parts]
        if not reg_candidates:raise FileNotFoundError('final_feature_registry.json')
        reg=json.loads(reg_candidates[0].read_text(encoding='utf-8'))['models']
        base=list(reg['M5_analogue']);with_cams=list(reg['M5_CAMS'])
        if set(with_cams)-set(base)!={'PM25_CAMS_ugm3'}:raise ValueError('Unexpected CAMS-only features in registry')
        prior_by_year={}
        for year in (2022,2023,2024):
            f=research/f'data/processed/model_features_{year}.parquet'
            keys=pd.read_parquet(f,columns=KEY)
            keys['측정소코드']=keys['측정소코드'].astype(str)
            keys['issue_time_kst']=utc(keys['issue_time_kst'])
            keys['prior_issue_utc']=keys.issue_time_kst-pd.Timedelta(hours=12)
            keys['prior_lead_h']=keys.lead_h+12
            if keys.duplicated(KEY).any():raise ValueError(f'Duplicate model keys in {year}')
            a1=keys.loc[keys.lead_h.between(1,12)].merge(prior_short,on=['측정소코드','prior_issue_utc','prior_lead_h'],how='left',validate='one_to_one')
            a2=keys.loc[keys.lead_h.between(13,24)].merge(prior_long,on=['측정소코드','prior_issue_utc','prior_lead_h'],how='left',validate='one_to_one')
            merged=pd.concat([a1,a2],ignore_index=True)[KEY+['PM25_CAMS_prior']]
            merged.to_parquet(out/f'prior_cams_{year}.parquet',index=False)
            prior_by_year[year]=merged
            report['stages'][f'year_{year}']={'total':len(merged),'matched':int(merged.PM25_CAMS_prior.notna().sum()),'missing':int(merged.PM25_CAMS_prior.isna().sum())}
            print(f'{year} prior matching: {report["stages"][f"year_{year}"]}',flush=True)
        if a.stage_only:
            report['status']='STAGED_ONLY';return 0
        params=dict(objective='reg:squarederror',n_estimators=300,learning_rate=.03,max_depth=5,min_child_weight=5,subsample=.9,colsample_bytree=.8,reg_alpha=0.,reg_lambda=20.,tree_method='hist',device='cpu',random_state=42,n_jobs=4)
        rows=[]
        for h in range(1,25):
            datasets=[]
            for year in (2022,2023,2024):
                f=research/f'data/processed/model_features_{year}.parquet'
                d=pd.read_parquet(f,columns=list(dict.fromkeys(KEY+['PM25_target']+with_cams)))
                d=d.loc[d.lead_h==h].copy()
                d['측정소코드']=d['측정소코드'].astype(str)
                d['issue_time_kst']=utc(d['issue_time_kst'])
                d=d.merge(prior_by_year[year].loc[lambda x:x.lead_h==h],on=KEY,how='left',validate='one_to_one')
                columns=list(dict.fromkeys(['PM25_target']+with_cams+['PM25_CAMS_prior']))
                good=np.isfinite(d[columns].to_numpy(dtype='float64')).all(axis=1)
                datasets.append(d.loc[good].copy())
            train=pd.concat(datasets[:2],ignore_index=True);test=datasets[2]
            if len(train)<100 or len(test)<100:raise ValueError(f'Insufficient paired rows at lead {h}: train={len(train)} test={len(test)}')
            target=train.PM25_target.to_numpy('float32');y=test.PM25_target.to_numpy('float64')
            predictions={}
            for model,cols in [('M5',base),('M5_current_CAMS',with_cams),('M5_prior_CAMS',[x if x!='PM25_CAMS_ugm3' else 'PM25_CAMS_prior' for x in with_cams])]:
                m=xgb.XGBRegressor(**params)
                m.fit(train[cols].to_numpy('float32'),target)
                pred=m.predict(test[cols].to_numpy('float32'))
                predictions[model]=pred
                rows.append({'lead_h':h,'model':model,**stats(y,pred)})
            outpred=test[KEY+['PM25_target']].copy()
            for name,pred in predictions.items():outpred['pred_'+name]=pred
            outpred.to_parquet(out/f'predictions_h{h:02d}.parquet',index=False)
            print(f'FIT lead {h}/24, paired n={len(test)}',flush=True)
        metrics=pd.DataFrame(rows);metrics.to_csv(out/'v60_metrics_by_horizon.csv',index=False,encoding='utf-8-sig')
        summary=[]
        for model in ('M5','M5_current_CAMS','M5_prior_CAMS'):
            data=[]
            for h in range(1,25):
                p=pd.read_parquet(out/f'predictions_h{h:02d}.parquet',columns=['PM25_target','pred_'+model])
                data.append(p)
            dd=pd.concat(data,ignore_index=True)
            summary.append({'model':model,**stats(dd.PM25_target,dd['pred_'+model])})
        pd.DataFrame(summary).to_csv(out/'v60_pooled_metrics.csv',index=False,encoding='utf-8-sig')
        report['stages']['model_fit']={'horizons':24,'models':3,'training':'2022+2023','test':'2024','paired_per_horizon':True,'metrics':summary}
        report['status']='COMPLETED'
        return 0
    except Exception as e:
        report['status']='FAILED';report['error']=f'{type(e).__name__}: {e}';report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
        return 1
    finally:
        write_json(out/'v60_report.json',report)
        (out/'V60_REPORT.md').write_text('# v60 CAMS prior-cycle sensitivity\n\nStatus: '+report['status']+'\n\n'+ '\n'.join('- '+x for x in report['warnings'])+'\n\n'+json.dumps(report['stages'],ensure_ascii=False,indent=2,default=str)+'\n'+report.get('error',''),encoding='utf-8')
        print('REPORT:',out/'v60_report.json',flush=True)

if __name__=='__main__':raise SystemExit(main())
