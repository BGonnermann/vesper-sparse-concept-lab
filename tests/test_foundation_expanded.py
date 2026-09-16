import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import expanded_data_campaign as campaign

class ExpandedTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(fingerprint='old',outputs={'train.jsonl':'oldtrain','validation.jsonl':'v','test.jsonl':'t'})
        self.new=dict(parent_fingerprint='old',ingestor_sha256='source',outputs={'train.jsonl':'newtrain','validation.jsonl':'v','test.jsonl':'t'})
        self.rows=[dict(id='technical',domain='technical',text='unchanged')]
    def check(self,newrows=None):
        rows=self.rows if newrows is None else newrows
        with patch.object(campaign,'verify',side_effect=[self.old,self.new]),patch.object(campaign,'digest',return_value='source'),patch.object(campaign,'read_rows',side_effect=[self.rows,rows]):
            return campaign.validate_corpus_pair('old','new')
    def test_accepts_only_general_pool_change(self):self.assertEqual(self.check(),(self.old,self.new))
    def test_rejects_validation_change(self):
        self.new['outputs']['validation.jsonl']='changed'
        with self.assertRaises(AssertionError):self.check()
    def test_rejects_test_change(self):
        self.new['outputs']['test.jsonl']='changed'
        with self.assertRaises(AssertionError):self.check()
    def test_rejects_technical_change(self):
        with self.assertRaises(AssertionError):self.check([dict(self.rows[0],text='changed')])
    def test_rejects_ingestor_change(self):
        self.new['ingestor_sha256']='changed'
        with self.assertRaises(AssertionError):self.check()
    def test_rejects_wrong_parent(self):
        self.new['parent_fingerprint']='changed'
        with self.assertRaises(AssertionError):self.check()

if __name__=='__main__':unittest.main()
