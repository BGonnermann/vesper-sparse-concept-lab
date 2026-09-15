"""Completion must reject active, unaudited or changed evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ncp_final_gate import validate_inventory


class FinalGateTests(unittest.TestCase):
    def test_matching_terminal_inventory(self):
        rows=[dict(trial='good',status='completed'),dict(trial='failure',status='failed')]
        audit=dict(status='completed',all_sources_committed_and_byte_verified=True,
            verified_trials=[dict(trial='good')],preserved_noncompleted=[dict(trial='failure')],
            trial_result_hashes={'good':'a','failure':'b'})
        validate_inventory(rows,audit,{'good':'a','failure':'b'})

    def test_active_stale_missing_failure_and_uncommitted_evidence_are_rejected(self):
        rows=[dict(trial='good',status='completed'),dict(trial='failure',status='failed')]
        audit=dict(status='completed',all_sources_committed_and_byte_verified=True,
            verified_trials=[dict(trial='good')],preserved_noncompleted=[dict(trial='failure')],
            trial_result_hashes={'good':'a','failure':'b'})
        cases=[]
        active=copy.deepcopy(rows);active[0]['status']='running';cases.append((active,audit,{'good':'a','failure':'b'}))
        cases.append((rows,audit,{'good':'changed','failure':'b'}))
        for key,value in [('verified_trials',[]),('preserved_noncompleted',[]),('all_sources_committed_and_byte_verified',False)]:
            altered=copy.deepcopy(audit);altered[key]=value;cases.append((rows,altered,{'good':'a','failure':'b'}))
        for case in cases:
            with self.subTest(case=case),self.assertRaises(AssertionError):
                validate_inventory(*case)


if __name__=='__main__':
    unittest.main(verbosity=2)
