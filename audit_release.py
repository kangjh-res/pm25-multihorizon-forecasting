from pathlib import Path
import ast, json, re, hashlib, sys
ROOT=Path(__file__).resolve().parent
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('__pycache__','.git','runs'))]
py=[p for p in files if p.suffix=='.py']
errors=[]; matches=[]
patterns={'windows_absolute_path':r'[A-Za-z]:\\(?:Users|KICT|\\)', 'secret_assignment':r'(?i)(?:api[_-]?key|secret|password|token)\s*=\s*[\"\'][^\"\']{8,}'}
for p in py:
 try: ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 except Exception as e: errors.append({'file':str(p.relative_to(ROOT)),'error':str(e)})
for p in files:
 if p.suffix.lower() not in ('.py','.md','.json','.txt','.yml','.yaml','.csv'): continue
 s=p.read_text(encoding='utf-8-sig',errors='replace')
 for name,pattern in patterns.items():
  if re.search(pattern,s):matches.append({'file':str(p.relative_to(ROOT)),'type':name})
report={'python_files':len(py),'syntax_errors':errors,'sensitive_pattern_matches':matches,'has_license':(ROOT/'LICENSE').exists(),'requirements_pinned':False,'full_reproduction_verified':False,'status':'STATIC_ONLY'}
print(json.dumps(report,indent=2,ensure_ascii=False))
sys.exit(1 if errors or matches else 0)
