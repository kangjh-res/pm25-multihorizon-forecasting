import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_paper_references import verify
class TestReferences(unittest.TestCase):
 def test_match(self):
  c={'critical_new_onset_24h':550,'critical_complete_events':449}
  w={'complete_wind_event_keys':449,'wind_speed_median_h15':3.410,'direction_shift_median_h12_h15':25.389}
  self.assertTrue(all(x['status']=='PASS' for x in verify(c,w)))
 def test_mismatch(self):
  self.assertTrue(any(x['status']=='FAIL' for x in verify({},{})))
 def test_tolerance(self):
  c={'critical_new_onset_24h':550,'critical_complete_events':449}
  w={'complete_wind_event_keys':449,'wind_speed_median_h15':3.4104,'direction_shift_median_h12_h15':25.3894}
  self.assertTrue(all(x['status']=='PASS' for x in verify(c,w)))
if __name__=='__main__':unittest.main()
