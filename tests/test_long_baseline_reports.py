import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from chart_long_baseline import export,frontier,render

class ReportTests(unittest.TestCase):
    def test_frontier_includes_regressions_without_replacing_observations(self):
        values=[2.,1.5,1.7,1.3,1.4];np.testing.assert_array_equal(frontier(values),[2.,1.5,1.5,1.3,1.3]);self.assertEqual(values[2],1.7)
    def test_export_and_all_seven_charts_from_metrics_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();data=root/'data';plots=root/'plots'
            losses=[];vals=[]
            for step,bpb in [(512,1.4),(1024,1.2),(2048,1.25)]:
                losses.append(dict(step=step,tokens=step*16384,tokens_per_parameter=.1,loss=3.,gradient_norm=.4,matrix_lr=.04,lr_multiplier=1.,muon_momentum=.95,muon_weight_decay=.1,seconds=.65,tokens_per_second=25000.,wall_seconds=step*.65,peak_allocated_bytes=2400000000,peak_reserved_bytes=2550000000,data_progress={d:dict(tokens=100,passes=.5) for d in ('general','technical')}))
                vals.append(dict(step=step,tokens=step*16384,wall_seconds=step*.65,evaluation=dict(aggregate_bpb=bpb,domains={d:dict(bpb=bpb) for d in ('general','technical')},seconds=2.)))
            for name,value in [('losses',losses),('evaluations',vals),('config',{'seed':301})]:(run/(name+'.json')).write_text(json.dumps(value))
            export(run,data);u=pd.read_csv(data/'updates.csv.gz');self.assertEqual(len(u),3);self.assertEqual(u.iloc[-1].tokens,2048*16384)
            render(data,plots);self.assertEqual(len(list(plots.glob('*.png'))),7);self.assertEqual(len(list(plots.glob('*.svg'))),7)

if __name__=='__main__':unittest.main()
