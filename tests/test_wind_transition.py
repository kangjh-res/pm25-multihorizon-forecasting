import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import pandas as pd
from reproduce_wind_transition import compute,circular_abs
class TestWind(unittest.TestCase):
 def test_legacy_repeated_cohort(self):
  c=pd.DataFrame({'측정소코드':['1']*24,'issue_time_kst':['2024-01-01']*24,'lead_h':list(range(1,25)),'regime':['Critical']*24})
  p=pd.DataFrame([{'측정소코드':'1','issue_time_kst':'2024-01-01','lead_h':h,'PM25_target':40+h,'pred_M5_CAMS':30+h/2} for h in [15,18]])
  f=pd.DataFrame([{'측정소코드':'1','issue_time_kst':'2024-01-01','lead_h':h,'WS10_fcst':3.0,'WD10_fcst':30.0+h} for h in [12,15]])
  summary,events,joint=compute(c,p,f)
  self.assertEqual(summary['cohort_keys'],1)
  self.assertEqual(len(events),1)
 def test_wrap(self):self.assertEqual(float(circular_abs(5,355)),10.0)
 def test_regimes(self):
  c=pd.DataFrame({'측정소코드':['1','2'],'issue_time_kst':['2024-01-01']*2,'regime':['Critical']*2})
  p=[];f=[]
  for s,wd,ws in [('1',355,2),('2',50,5)]:
   for h in [15,18]:p.append({'측정소코드':s,'issue_time_kst':'2024-01-01','lead_h':h,'PM25_target':40+h,'pred_M5_CAMS':30+h/2})
   for h in [12,15]:f.append({'측정소코드':s,'issue_time_kst':'2024-01-01','lead_h':h,'WS10_fcst':ws,'WD10_fcst':wd if h==12 else (5 if s=='1' else 130)})
  summary,events,joint=compute(c,pd.DataFrame(p),pd.DataFrame(f))
  self.assertEqual(summary['complete_wind_event_keys'],2)
  self.assertEqual(len(joint),2)
  self.assertEqual(events.loc[events['측정소코드']=='1','WD_shift_abs_12_15'].iloc[0],10)
  self.assertTrue((events.Missed_rise_15_18==1.5).all())
if __name__=='__main__':unittest.main()
