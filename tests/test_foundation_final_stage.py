import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_foundation_test import require_frozen
from foundation_eval import selected_inputs

class FinalStageTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(checkpoint_sha256='weights',seed=201,dataset_fingerprint='data')
        self.record=dict(self.identity,status='completed')
    def test_accepts_frozen_identity(self):require_frozen(self.identity,self.record,'weights')
    def test_rejects_wrong_weights(self):
        with self.assertRaises(ValueError):require_frozen(self.identity,self.record,'other')
    def test_rejects_wrong_seed(self):
        self.record['seed']=202
        with self.assertRaises(ValueError):require_frozen(self.identity,self.record,'weights')
    def test_rejects_wrong_dataset(self):
        self.record['dataset_fingerprint']='other'
        with self.assertRaises(ValueError):require_frozen(self.identity,self.record,'weights')
    def test_rejects_incomplete_run(self):
        self.record['status']='failed'
        with self.assertRaises(ValueError):require_frozen(self.identity,self.record,'weights')
    def test_explicit_test_dispatch(self):
        rows=[dict(id='test',text='held out',domain='general',split='test')]
        self.assertEqual(selected_inputs(rows)[0]['text'],'held out')
        rows.append(dict(id='val',text='wrong split',domain='general',split='validation'))
        with self.assertRaises(ValueError):selected_inputs(rows)

if __name__=='__main__':unittest.main()
