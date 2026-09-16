"""Synthetic red tests for final paired-evidence rejection, not substitutes for GPU replay."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import report_foundation as report

class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.files={}
        for seed in report.SEEDS:
            for arm in report.ARMS:
                name=f'pilot-{arm}-s{seed}'
                row=dict(status='completed',seed=seed,updates=512,training_tokens=8388608,checkpointing=False,
                    stream=dict(source_tokens={'general':8388608},stream_sha256={'general':'g','technical':'t'}),
                    parameters=dict(total_parameters=135267480,active_parameters=135267480,memory_table_bytes=0,ncp_parameters=0,expert_parameters=0),
                    validation=dict(aggregate_bpb=1.,input_fingerprint='frozen',domains={d:dict(bytes=100,tokens=40,bpb=1.) for d in ('general','technical')}),
                    training=dict(peak_allocated_bytes=3*2**30),checkpoint_sha256='sealed',candidate={'feedforward':'dense'},
                    initialization_sha256=str(seed),dataset_fingerprint='data',source_hashes={'source':'hash'},
                    tokenizer={'id':'candidate' if arm=='tokenizer' else 'current'},general_weight=.5 if arm=='mixture' else .8)
                self.files[str(report.HOME/name/'result.json')]=row
                self.files[str(report.HOME/name/'losses.json')]=[dict(step=i+1,mean_loss=1.,min_loss=1.,max_loss=1.) for i in range(512)]
                self.files[str(report.HOME/name/'schedule.json')]=['schedule']
                self.files[str(report.HOME/(name+'-replay.json'))]=dict(status='verified',run_result_sha256='sealed',checkpoint_sha256='sealed')
    def run_guard(self):
        with patch.object(report,'read',side_effect=lambda p:copy.deepcopy(self.files[str(p)])),patch.object(report,'digest',return_value='sealed'):
            return report.verify_pairs()
    def row(self,arm='mixture',seed=201):return self.files[str(report.HOME/f'pilot-{arm}-s{seed}'/'result.json')]
    def test_accepts_valid_synthetic_receipts(self):self.assertEqual(len(self.run_guard()),9)
    def test_rejects_mismatched_tokens(self):
        self.row()['training_tokens']-=1
        with self.assertRaises(AssertionError):self.run_guard()
    def test_rejects_changed_parameters(self):
        self.row()['parameters']['total_parameters']+=1
        with self.assertRaises(AssertionError):self.run_guard()
    def test_rejects_changed_evaluation(self):
        self.row()['validation']['input_fingerprint']='other'
        with self.assertRaises(AssertionError):self.run_guard()
    def test_rejects_nonfinite_validation(self):
        self.row()['validation']['aggregate_bpb']=float('nan')
        with self.assertRaises(AssertionError):self.run_guard()
    def test_rejects_incomplete_replay(self):
        self.files[str(report.HOME/'pilot-mixture-s201-replay.json')]['status']='failed'
        with self.assertRaises(AssertionError):self.run_guard()
    def test_rejects_seed_reuse(self):
        for arm in report.ARMS:self.row(arm,202)['initialization_sha256']='201'
        with self.assertRaises(AssertionError):self.run_guard()

if __name__=='__main__':unittest.main()
