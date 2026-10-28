import unittest
import pandas as pd
from tools.reproduce_critical_cohort import reproduce
class CohortTests(unittest.TestCase):
    def make_data(self):
        pred=[];feat=[]
        for station in ['A','B','C','D']:
            for h in range(1,25):
                y=40 if h==24 else 15+h
                pred.append({'측정소코드':station,'issue_time_kst':'2024-03-01 09:00','lead_h':h,'PM25_target':y,'PM25_t':10,'pred_M5_analogue':y-2,'pred_M5_CAMS':y-1})
                feat.append({'측정소코드':station,'issue_time_kst':'2024-03-01 09:00','lead_h':h,'WS10_fcst':3 if station in ('A','B') else 1,'PM25_CAMS_ugm3':10 if station in ('A','B') else 40,'upwind_PM25_weighted':10 if station in ('A','B') else 40})
        return pd.DataFrame(pred),pd.DataFrame(feat)
    def test_complete_cohort(self):
        p,f=self.make_data();s,t,c=reproduce(p,f)
        self.assertEqual(s['new_onset_24h'],4)
        self.assertEqual(s['critical_complete_events'],2)
        self.assertEqual(s['complete_rows'],96)
        self.assertEqual(len(t),96)
    def test_missing_lead_excluded(self):
        p,f=self.make_data();p=p.loc[~((p['측정소코드']=='A')&(p['lead_h']==12))]
        s,t,c=reproduce(p,f);self.assertEqual(s['critical_complete_events'],1)
        self.assertEqual(s['complete_events_all_regimes'],3)
    def test_duplicate_rejected(self):
        p,f=self.make_data();p=pd.concat([p,p.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):reproduce(p,f)
if __name__=='__main__':unittest.main()
