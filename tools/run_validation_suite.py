#!/usr/bin/env python3
"""Read-only-input orchestration of v44-v54 archived-prediction checks.

Does not download data or retrain models. Writes exclusively to a fresh output
folder outside source data. Runs independent checks even if one fails and
produces a machine-readable and human-readable report.
"""
from __future__ import annotations
import argparse, datetime, json, pathlib, subprocess, sys, csv

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xgb',type=pathlib.Path,required=True,help='Archived 2024 XGBoost prediction parquet')
    p.add_argument('--features',type=pathlib.Path,required=True,help='2024 model_features parquet')
    p.add_argument('--mlp',type=pathlib.Path,help='Optional archived MLP prediction parquet')
    p.add_argument('--out-dir',type=pathlib.Path,required=True,help='New, empty output directory')
    a=p.parse_args()
    sources=[a.xgb,a.features]+([a.mlp] if a.mlp else [])
    for source in sources:
        if not source.is_file():p.error(f'Missing input: {source}')
    out=a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty (prevents overwrites)')
    for source in sources:
        resolved=source.resolve()
        if resolved==out or out in resolved.parents or resolved.parent==out or out in resolved.parents:
            p.error('Output directory must be separate from all input files and their parent directories')
    out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    def run(name,script,args,dependencies=()):
        if any(next((x['status'] for x in tasks if x['name']==dep),'SKIPPED')!='PASS' for dep in dependencies):
            tasks.append({'name':name,'status':'SKIPPED','exit_code':None,'reason':'dependency failed or skipped'});return
        cmd=[sys.executable,str(ROOT/'tools'/script),*map(str,args)]
        result=subprocess.run(cmd,capture_output=True,text=True,check=False)
        dest=out/name;dest.mkdir(exist_ok=True)
        (dest/'stdout.txt').write_text(result.stdout,encoding='utf-8')
        (dest/'stderr.txt').write_text(result.stderr,encoding='utf-8')
        tasks.append({'name':name,'status':'PASS' if result.returncode==0 else 'FAIL','exit_code':result.returncode,'reason':result.stderr[-800:] if result.returncode else ''})
    run('s3','verify_si_table_s3.py',['--predictions',a.xgb,'--out-dir',out/'s3'])
    run('critical','reproduce_critical_cohort.py',['--predictions',a.xgb,'--features',a.features,'--out-dir',out/'critical'])
    run('wind','reproduce_wind_transition.py',['--cohort',out/'critical'/'complete_event_keys.csv','--predictions',a.xgb,'--features',a.features,'--out-dir',out/'wind'],['critical'])
    run('s6_s7','verify_si_tables_s6_s7.py',['--predictions',a.xgb,'--features',a.features,'--wind-events',out/'wind'/'wind_transition_event_level.csv','--out-dir',out/'s6_s7'],['wind'])
    run('s4_references','verify_paper_references.py',['--critical-summary',out/'critical'/'cohort_summary.json','--wind-summary',out/'wind'/'wind_transition_summary.json','--out-dir',out/'s4_references'],['wind'])
    if a.mlp:
        run('paired_models','evaluate_archived.py',['--xgb',a.xgb,'--mlp',a.mlp,'--out',out/'paired_models'/'paired_metrics.csv'])
    else:
        tasks.append({'name':'paired_models','status':'SKIPPED','exit_code':None,'reason':'--mlp not supplied'})
    summary={'scope':'archived predictions and prepared features only; not raw-data or operational-time validation',
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'inputs':{k:str(v.resolve()) if v else None for k,v in [('xgb',a.xgb),('features',a.features),('mlp',a.mlp)]},
             'tasks':tasks,'passed':sum(t['status']=='PASS' for t in tasks),'failed':sum(t['status']=='FAIL' for t in tasks),'skipped':sum(t['status']=='SKIPPED' for t in tasks)}
    (out/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'validation_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['name','status','exit_code','reason']);w.writeheader();w.writerows(tasks)
    lines=['# PM2.5 archived-results validation','',f"PASS: {summary['passed']} | FAIL: {summary['failed']} | SKIPPED: {summary['skipped']}",'', '| Check | Status |','|---|---|']
    lines.extend(f"| {t['name']} | {t['status']} |" for t in tasks)
    lines+=['','This validates archived predictions/prepared features only. It does not establish raw-to-feature reproducibility or historical forecast-product availability.']
    (out/'VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':summary['passed'],'failed':summary['failed'],'skipped':summary['skipped'],'report':str(out/'VALIDATION_REPORT.md')},indent=2))
    return 2 if summary['failed'] else 0
if __name__=='__main__':raise SystemExit(main())
