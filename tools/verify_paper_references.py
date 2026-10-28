#!/usr/bin/env python3
"""Read-only verification of v50/v51 outputs against explicitly reported SI references.
Does not claim that matching summary numbers proves full figure/table reproduction.
"""
import argparse,csv,json,math
from pathlib import Path
REFERENCES={
 'critical_new_onset_24h':(550,0),
 'critical_complete_events':(449,0),
 'wind_speed_median_h15':(3.410,0.0005),
 'direction_shift_median_h12_h15':(25.389,0.0005),
 'complete_wind_event_keys':(449,0),
}
def verify(cohort,wind):
    results=[]
    for key,expected,tol in [('critical_new_onset_24h',550,0),('critical_complete_events',449,0)]:
        val=cohort.get(key);ok=isinstance(val,(float,int)) and math.isfinite(val) and abs(val-expected)<=tol
        results.append({'check':key,'observed':val,'reference':expected,'tolerance':tol,'status':'PASS' if ok else 'FAIL'})
    for key,expected,tol in [('complete_wind_event_keys',449,0),('wind_speed_median_h15',3.410,0.0005),('direction_shift_median_h12_h15',25.389,0.0005)]:
        val=wind.get(key);ok=isinstance(val,(float,int)) and math.isfinite(val) and abs(val-expected)<=tol
        results.append({'check':key,'observed':val,'reference':expected,'tolerance':tol,'status':'PASS' if ok else 'FAIL'})
    return results
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--critical-summary',type=Path,required=True)
    p.add_argument('--wind-summary',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();inputs=[a.critical_summary.resolve(),a.wind_summary.resolve()];out=a.out_dir.resolve()
    if any(out==x or out in x.parents for x in inputs):p.error('output directory must not contain inputs')
    cohort=json.loads(a.critical_summary.read_text(encoding='utf-8'))
    wind=json.loads(a.wind_summary.read_text(encoding='utf-8'))
    rows=verify(cohort,wind)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'reference_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['check','observed','reference','tolerance','status']);w.writeheader();w.writerows(rows)
    summary={'checks':len(rows),'passed':sum(x['status']=='PASS' for x in rows),'failed':sum(x['status']=='FAIL' for x in rows),'scope':'SI Text S4 cohort counts and wind median thresholds only','note':'No full Fig. 2-4 or SI table numerical equivalence claimed'}
    (out/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0 if summary['failed']==0 else 2
if __name__=='__main__':raise SystemExit(main())
