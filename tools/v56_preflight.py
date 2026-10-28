#!/usr/bin/env python3
"""Check v56 input availability without reading large files or changing originals."""
import argparse, importlib.util, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--xgb',type=Path,required=True)
p.add_argument('--features',type=Path)
p.add_argument('--mlp',type=Path)
p.add_argument('--cams-dir',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
items={key:{'path':str(value) if value else None,'exists':bool(value and (value.is_dir() if key=='cams_dir' else value.is_file())),'bytes':value.stat().st_size if value and value.is_file() else None} for key,value in [('xgb',a.xgb),('features',a.features),('mlp',a.mlp),('cams_dir',a.cams_dir)]}
mods={name:bool(importlib.util.find_spec(name)) for name in ('pandas','numpy','pyarrow','fastparquet','eccodes')}
report={'inputs':items,'modules':mods,'s3_ready':items['xgb']['exists'] and (mods['pyarrow'] or mods['fastparquet']) and mods['pandas'] and mods['numpy'],'full_suite_ready':all(items[k]['exists'] for k in ('xgb','features')) and (mods['pyarrow'] or mods['fastparquet']) and mods['pandas'] and mods['numpy'],'note':'Existence/dependency check only; not numerical validation. CAMS content integrity not checked.'}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
