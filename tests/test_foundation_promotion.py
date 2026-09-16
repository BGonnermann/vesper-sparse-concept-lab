import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_foundation_final import gate

class PromotionTests(unittest.TestCase):
    def make(self,deltas=(-.1,-.11,-.09)):
        rows={}
        for seed,delta in zip((1,2,3),deltas):
            for arm,value in [('a',1.),('b',1.+delta)]:
                metric=dict(aggregate_bpb=value,domains={d:dict(bpb=value) for d in ('general','technical')})
                rows[f'{arm}{seed}']=dict(validation=copy.deepcopy(metric),test=dict(evaluation=copy.deepcopy(metric)),training=dict(timed_tokens_per_second=100.))
        return rows
    def check(self,rows,split='validation'):return gate(rows,'a{s}','b{s}',(1,2,3),split)
    def test_strong_paired_improvement(self):self.assertTrue(self.check(self.make())['passed'])
    def test_rejects_one_seed_reversal(self):self.assertFalse(self.check(self.make((-.1,-.1,.001)))['passed'])
    def test_rejects_small_effect(self):self.assertFalse(self.check(self.make((-.001,)*3))['passed'])
    def test_rejects_large_seed_variance(self):self.assertFalse(self.check(self.make((-.001,-.002,-.2)))['passed'])
    def test_rejects_domain_regression(self):
        rows=self.make()
        for seed in (1,2,3):rows[f'b{seed}']['validation']['domains']['technical']['bpb']=1.03
        self.assertFalse(self.check(rows)['passed'])
    def test_rejects_throughput_regression(self):
        rows=self.make();rows['b1']['training']['timed_tokens_per_second']=79.
        self.assertFalse(self.check(rows)['passed'])
    def test_test_failure_cannot_be_hidden_by_validation(self):
        rows=self.make()
        for seed in (1,2,3):rows[f'b{seed}']['test']['evaluation']['aggregate_bpb']=1.1
        self.assertTrue(self.check(rows)['passed']);self.assertFalse(self.check(rows,'test')['passed'])

if __name__=='__main__':unittest.main()
