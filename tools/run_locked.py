#!/usr/bin/env python3
"""Portable v43 locked-training launcher. Default = preflight only; no model fit.

Inputs: prepared model_features_{2022,2023,2024}.parquet, common_eval_keys_{year}.parquet,
        final_feature_registry.json. No network, raw preprocessing, or CAMS downloads.
"""
from __future__ import annotations
import argparse, contextlib, datetime, hashlib, io, json, pathlib, sys, traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ('측정소코드', 'issue_time_kst', 'lead_h')


def sha256(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def config_path(p):
    q=pathlib.Path(p).expanduser()
    return q.resolve()


def preflight(data_root, registry):
    import importlib.util
    data=data_root/'data'/'processed'
    required={f'model_features_{y}':data/f'model_features_{y}.parquet' for y in (2022,2023,2024)}
    required.update({f'common_eval_keys_{y}':data/f'common_eval_keys_{y}.parquet' for y in (2022,2023,2024)})
    required['registry']=registry
    checks={k:{'exists':p.is_file(),'path':str(p),'bytes':p.stat().st_size if p.is_file() else None} for k,p in required.items()}
    packages={k:importlib.util.find_spec(k) is not None for k in ('numpy','pandas','pyarrow','sklearn','xgboost','torch','scipy')}
    if registry.is_file():
        try:
            r=json.loads(registry.read_text(encoding='utf-8'))
            expected=['M1_local','M2_obsmet','M3_fcstmet','M4_transport','M5_analogue','M5_CAMS','ANN_M4','KNN_ANN_M5']
            checks['registry_models']={'present':[x for x in expected if x in r.get('models',{})], 'missing':[x for x in expected if x not in r.get('models',{})]}
        except Exception as e: checks['registry_models']={'error':str(e)}
    ok=all(x['exists'] for x in checks.values() if isinstance(x,dict) and 'exists' in x) and not checks.get('registry_models',{}).get('missing') and not checks.get('registry_models',{}).get('error')
    return {'ready_inputs':bool(ok),'inputs':checks,'packages':packages,'source_hashes':{p.name:sha256(p) for p in sorted((ROOT/'locked_sources').glob('*.py'))}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True,help='Existing read-only source project with data/processed/')
    ap.add_argument('--registry',required=True,help='Existing final_feature_registry.json')
    ap.add_argument('--output-root',default=str(ROOT/'runs'),help='Separate output directory, never inside data-root')
    ap.add_argument('--mode',choices=['preflight','train'],default='preflight')
    ap.add_argument('--family',choices=['xgboost','mlp','both'],default='both')
    args=ap.parse_args()
    data_root=config_path(args.data_root);registry=config_path(args.registry);output_root=config_path(args.output_root)
    if output_root==data_root or data_root in output_root.parents:
        ap.error('output-root must not be inside the source data-root')
    if output_root==ROOT or ROOT in output_root.parents and output_root.name in ('locked_sources','legacy_v10'):
        ap.error('output-root must not overlap source code')
    if not data_root.is_dir(): ap.error('data-root does not exist')
    result=preflight(data_root,registry)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.mode=='preflight':
        print('PREFLIGHT ONLY: no files created and no training performed')
        return 0 if result['ready_inputs'] else 2
    if not result['ready_inputs']:
        print('BLOCKED: required prepared data/registry missing',file=sys.stderr);return 2
    if not all(result['packages'].values()):
        print('BLOCKED: required Python dependencies missing',file=sys.stderr);return 2
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run=output_root/f'locked_v43_{stamp}';run.mkdir(parents=True,exist_ok=False)
    (run/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    log=run/'run.log'
    class Tee(io.TextIOBase):
        def __init__(self,*s):self.s=s
        def write(self,x):
            for f in self.s:f.write(x);f.flush()
            return len(x)
        def flush(self):
            for f in self.s:f.flush()
    def safe_read_path(p):
        p=pathlib.Path(p).resolve()
        if not p.is_file() or not (p==data_root or data_root in p.parents or p==registry):
            raise PermissionError(f'Input missing/outside approved source: {p}')
        return p
    def safe_write_path(p):
        p=pathlib.Path(p).resolve()
        if run not in p.parents or p.exists():raise PermissionError(f'Output path not new/inside run: {p}')
        p.parent.mkdir(parents=True,exist_ok=True)
        return p
    def get_project_paths():return {'source_root':data_root,'registry_file':registry}
    families=['xgboost','mlp'] if args.family=='both' else [args.family]
    status={}
    with log.open('w',encoding='utf-8') as f,contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
        for family in families:
            src=ROOT/'locked_sources'/('xgboost_phase4D1.py' if family=='xgboost' else 'mlp_phase5C.py')
            scope={'__name__':'__main__','__file__':str(src),'PILOT_OUTPUT':run,'PILOT_REGISTRY':registry,'get_project_paths':get_project_paths,'safe_read_path':safe_read_path,'safe_write_path':safe_write_path}
            print(f'\nRUN {family}; locked source SHA256={sha256(src)}')
            try:
                exec(compile(src.read_text(encoding='utf-8'),str(src),'exec'),scope)
                status[family]='COMPLETED'
            except Exception:
                status[family]='FAILED';traceback.print_exc()
            finally:scope.clear()
    (run/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('RUN_OUTPUT:',run,'STATUS:',status)
    return 0 if all(x=='COMPLETED' for x in status.values()) else 1

if __name__=='__main__':raise SystemExit(main())
