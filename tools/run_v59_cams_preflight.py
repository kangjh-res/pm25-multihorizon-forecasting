"""Read-only CAMS prior-cycle 13–24h feasibility/GRIB inspection; NOT a model retrain."""
import argparse, csv, json, traceback
from pathlib import Path
from datetime import datetime, timedelta

p=argparse.ArgumentParser()
p.add_argument('--search-root',required=True)
p.add_argument('--research-root',required=True)
p.add_argument('--out-dir',required=True)
a=p.parse_args()
out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
root=Path(a.search_root);research=Path(a.research_root)
report={'run_at':datetime.now().isoformat(),'status':'PREFLIGHT_ONLY','checks':{},'notes':[]}

def check(name, value):
    report['checks'][name]=value
    print(name, value,flush=True)

try:
    archives=list(root.rglob('cams_h36_archive_v40/grib'))
    check('archive_directory_candidates',[str(x) for x in archives])
    if len(archives)!=1: raise RuntimeError('Expected one CAMS GRIB directory; found '+str(len(archives)))
    archive=archives[0]
    dates=[datetime(2022,1,1)+timedelta(days=i) for i in range(1096)]
    files=[archive/str(d.year)/f'cams_{d:%Y%m%d}_00-12_met_pm25_h025-036.grib' for d in dates]
    missing=[str(f) for f in files if not f.is_file() or f.stat().st_size==0]
    check('grib_files_present',len(files)-len(missing))
    check('grib_files_expected',1096)
    check('missing_or_empty_examples',missing[:10])
    try:
        import eccodes
        for label,f in [('first',files[0]),('last',files[-1])]:
            if not f.is_file():continue
            records=[]; grid=[]
            with f.open('rb') as h:
                while True:
                    gid=eccodes.codes_grib_new_from_file(h)
                    if gid is None:break
                    try:
                        records.append((int(eccodes.codes_get(gid,'dataTime')),int(eccodes.codes_get(gid,'endStep')),str(eccodes.codes_get(gid,'shortName'))))
                        if not grid: grid=[{'gridType':str(eccodes.codes_get(gid,'gridType')),'Ni':str(eccodes.codes_get(gid,'Ni')),'Nj':str(eccodes.codes_get(gid,'Nj'))}]
                    finally:eccodes.codes_release(gid)
            check(label+'_grib_messages',len(records))
            check(label+'_grib_unique_messages',len(set(records)))
            check(label+'_grib_leads',sorted(set(x[1] for x in records)))
            check(label+'_grib_variables',sorted(set(x[2] for x in records)))
            check(label+'_grib_grid',grid)
    except ImportError:
        check('eccodes','UNAVAILABLE: cannot verify message contents')
    except Exception as e:
        check('eccodes_error',str(e))
    # Expected time mapping: current cycle at T, previous cycle T-12, target T+h, previous lead h+12.
    mapping=[{'current_lead_h':h,'prior_cycle_lead_h':h+12,'in_new_archive_25_36':h+12>=25} for h in range(1,25)]
    with (out/'prior_cycle_lead_mapping.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    check('lead_13_24_mapping','previous-cycle CAMS leads 25–36')
    check('lead_1_12_mapping','previous-cycle CAMS leads 13–24 (existing station forecasts)')
    try:
        import pyarrow.parquet as pq
        for year in (2022,2023,2024):
            f=research/'data'/'processed'/f'cams_station_forecast_{year}.parquet'
            check(f'cams_station_{year}',{'exists':f.is_file(),'columns':pq.read_schema(f).names if f.is_file() else []})
        for year in (2022,2023,2024):
            f=research/'data'/'processed'/f'model_features_{year}.parquet'
            check(f'model_features_{year}',{'exists':f.is_file(),'columns':pq.read_schema(f).names[:20] if f.is_file() else []})
    except ImportError:
        check('pyarrow','UNAVAILABLE: schema check skipped')
    # Locate possible coordinate metadata without assuming station locations.
    names=[]
    for pattern in ('*station*coord*','*station*meta*','*측정소*','*station*location*'):
        names.extend(str(x) for x in (research/'data').rglob(pattern) if x.is_file())
    check('coordinate_metadata_candidates',sorted(set(names))[:40])
    report['notes']=[
        'This checks archive and required source schema only. It does NOT interpolate GRIB to stations, retrain models, or establish forecast skill.',
        'A full sensitivity study requires station coordinates, exact GRIB grid/time decoding, previous-cycle matching by valid time, and chronological 2022–2023 retraining / 2024 testing.',
        'Retrospective GRIB availability does not prove real-time release latency. Check provider availability separately.',
        'Use common paired rows and identical training policy to compare M5 with prior-cycle M5+CAMS.'
    ]
except Exception as e:
    report['error']=f'{type(e).__name__}: {e}'
    report['traceback']=traceback.format_exc(limit=3)
finally:
    (out/'v59_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v59 CAMS 13–24 h preflight','',f"Status: {report['status']}",'','## Checks']
    lines += [f'- **{k}**: {v}' for k,v in report['checks'].items()]
    lines += ['','## Interpretation']+[f'- {x}' for x in report['notes']]
    if 'error' in report:lines+=['','## Error',report['error']]
    (out/'V59_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT:',out/'V59_REPORT.md',flush=True)
    if 'error' in report:raise SystemExit(1)
