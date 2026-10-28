#!/usr/bin/env python3
"""v58 follow-up: paired archived predictions and CAMS h25-36 file inventory.
Read-only inputs; no GRIB processing, training, or claims of real-time availability.
"""
import argparse, csv, datetime, json, pathlib, subprocess, sys
from datetime import date, timedelta
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root',type=pathlib.Path,required=True)
    p.add_argument('--out-dir',type=pathlib.Path,required=True)
    p.add_argument('--xgb',type=pathlib.Path)
    p.add_argument('--mlp',type=pathlib.Path)
    p.add_argument('--cams-dir',type=pathlib.Path)
    a=p.parse_args()
    if not a.search_root.is_dir():p.error('Search root does not exist')
    out=a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder must be empty')
    if out==a.search_root.resolve() or a.search_root.resolve() in out.parents and not out.name.startswith('v58_'):
        pass # separate named folder inside search root is expected
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def discover(pattern,selected,prefer=None):
        if selected:
            if not selected.is_file():p.error(f'Missing input: {selected}')
            return selected.resolve(),[str(selected.resolve())]
        candidates=sorted(x.resolve() for x in a.search_root.rglob(pattern) if x.is_file() and out not in x.parents)
        if prefer:
            ranked=[x for x in candidates if prefer in str(x)]
            if len(ranked)==1:return ranked[0],[str(x) for x in candidates]
        return (candidates[0] if len(candidates)==1 else None),[str(x) for x in candidates]
    xgb,xgb_candidates=discover('phase4D1_2024_row_predictions.parquet',a.xgb,'locked_retrain_full_v16')
    mlp,mlp_candidates=discover('phase5C_locked_2024_predictions.parquet',a.mlp)
    cams=a.cams_dir
    if cams is None:
        dirs=sorted(x for x in a.search_root.rglob('cams_h36_archive_v40') if x.is_dir() and (x/'grib').is_dir())
        cams=dirs[0]/'grib' if len(dirs)==1 else None
    elif (cams/'grib').is_dir():cams=cams/'grib'
    manifest={'xgb_candidates':xgb_candidates,'mlp_candidates':mlp_candidates,'xgb_selected':str(xgb) if xgb else None,'mlp_selected':str(mlp) if mlp else None,'cams_dir':str(cams) if cams else None}
    (out/'input_discovery.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if xgb and mlp:
        dest=out/'paired_models';dest.mkdir()
        cmd=[sys.executable,str(ROOT/'tools'/'evaluate_archived.py'),'--xgb',str(xgb),'--mlp',str(mlp),'--out',str(dest/'paired_metrics.csv')]
        result=subprocess.run(cmd,capture_output=True,text=True,check=False)
        (dest/'stdout.txt').write_text(result.stdout,encoding='utf-8')
        (dest/'stderr.txt').write_text(result.stderr,encoding='utf-8')
        rows.append({'name':'paired_models','status':'PASS' if result.returncode==0 else 'FAIL','detail':result.stderr[-500:] if result.returncode else 'common-key metrics computed'})
    else:
        rows.append({'name':'paired_models','status':'SKIPPED','detail':'MLP/XGB path missing or ambiguous; see input_discovery.json'})
    if cams and cams.is_dir():
        days=[];d=date(2022,1,1)
        while d<=date(2024,12,31):
            f=cams/str(d.year)/f'cams_{d:%Y%m%d}_00-12_met_pm25_h025-036.grib'
            days.append({'date':d.isoformat(),'exists':f.is_file(),'bytes':f.stat().st_size if f.is_file() else 0,'path':str(f)})
            d+=timedelta(days=1)
        with (out/'cams_inventory.csv').open('w',newline='',encoding='utf-8-sig') as fp:
            writer=csv.DictWriter(fp,fieldnames=['date','exists','bytes','path']);writer.writeheader();writer.writerows(days)
        present=sum(r['exists'] for r in days)
        rows.append({'name':'cams_inventory','status':'PASS' if present==len(days) else 'FAIL','detail':f'{present}/{len(days)} daily files present (existence/size only, not GRIB integrity)'})
    else:
        rows.append({'name':'cams_inventory','status':'SKIPPED','detail':'CAMS archive path missing or ambiguous; pass --cams-dir'})
    with (out/'v58_summary.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.DictWriter(fp,fieldnames=['name','status','detail']);writer.writeheader();writer.writerows(rows)
    lines=['# v58 Follow-up verification','',*['- '+r['name']+': '+r['status']+' — '+r['detail'] for r in rows],'', 'CAMS inventory checks only file presence and size. The v40 download log reported 120 messages per file; this script does not independently validate GRIB messages or forecast availability at issue time.','', 'Paired metrics use intersection of archived XGBoost and MLP prediction keys. They do not retrain models.']
    (out/'V58_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'checks':rows},ensure_ascii=False,indent=2))
    return 1 if any(r['status']=='FAIL' for r in rows) else 0
if __name__=='__main__':raise SystemExit(main())
