#!/usr/bin/env python3
"""Read-only candidate artifact inventory. Does not execute analysis or modify research files."""
import argparse, csv, pathlib, json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root',required=True,type=pathlib.Path)
    p.add_argument('--manifest',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'V48_RESULT_LINEAGE.csv')
    p.add_argument('--out',type=pathlib.Path,help='Optional output CSV outside the research root')
    a=p.parse_args()
    root=a.research_root.expanduser().resolve()
    if not root.is_dir(): p.error('Research root is not a directory')
    if a.out and (a.out.resolve()==root or root in a.out.resolve().parents): p.error('Output must be outside read-only research root')
    with a.manifest.open(encoding='utf-8-sig',newline='') as fh: entries=list(csv.DictReader(fh))
    candidates=set()
    for r in entries:
        for col in ('candidate_data_inputs','candidate_result_artifacts'):
            candidates.update(n for n in r[col].split(';') if n)
    matches={n:[] for n in candidates}
    for path in root.rglob('*'):
        if path.is_file() and path.name in matches:
            matches[path.name].append(str(path.relative_to(root)))
    result=[]
    for r in entries:
        for col,kind in [('candidate_data_inputs','candidate_input'),('candidate_result_artifacts','candidate_output')]:
            for name in filter(None,r[col].split(';')):
                result.append({'analysis':r['analysis'],'kind':kind,'filename':name,'count':len(matches[name]),'relative_matches':' | '.join(sorted(matches[name])[:10])})
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('w',encoding='utf-8-sig',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=['analysis','kind','filename','count','relative_matches']);writer.writeheader();writer.writerows(result)
        print('Wrote',a.out)
    print(json.dumps({'analyses':len(entries),'candidate_references':len(result),'found_references':sum(bool(x['count']) for x in result),'missing_references':sum(not x['count'] for x in result)},ensure_ascii=False))
if __name__=='__main__': main()
