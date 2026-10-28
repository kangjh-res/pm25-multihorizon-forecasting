import unittest
import pandas as pd
from tools.verify_si_table_s3 import verify
class TableS3Tests(unittest.TestCase):
 def test_pass_and_fail(self):
  df=pd.DataFrame({'lead_h':[1,1],'PM25_target':[10.,20.],**{c:[10.,20.] for c in __import__('tools.verify_si_table_s3',fromlist=['MODEL_COLUMNS']).MODEL_COLUMNS.values()}})
  self.assertEqual(verify(df,[{'lead_h':'1','model':'M0','rmse_paper':'0'}])[0]['status'],'PASS')
  self.assertEqual(verify(df,[{'lead_h':'1','model':'M0','rmse_paper':'1'}])[0]['status'],'FAIL')
 def test_reference_shape(self):
  import csv
  from pathlib import Path
  path=Path(__file__).resolve().parents[1]/'references'/'si_table_s3_rmse.csv'
  with path.open(encoding='utf-8') as f:r=list(csv.DictReader(f))
  self.assertEqual(len(r),168)
  self.assertEqual(set(int(x['lead_h']) for x in r),set(range(1,25)))
if __name__=='__main__':unittest.main()
