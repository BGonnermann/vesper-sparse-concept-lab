"""Synthetic split-dispatch checks; never reads held-out data or model scores."""
from types import SimpleNamespace
import unittest
from depth_evaluation import evaluate_split


class SplitChecks(unittest.TestCase):
    def test_test_dispatch_reuses_metric_and_restores_loader(self):
        calls=[]
        def loader(tokenizer,batch,length,split,**kwargs):
            calls.append(split);return split
        prepare=SimpleNamespace(make_dataloader=loader)
        def metric(model,tokenizer,batch,**kwargs):
            return prepare.make_dataloader(tokenizer,batch,512,'val',**kwargs)
        prepare.evaluate_bpb=metric
        value,receipt=evaluate_split(prepare,None,None,2,'test',device='cpu')
        self.assertEqual(value,'test');self.assertEqual(calls,['test'])
        self.assertIs(prepare.make_dataloader,loader)
        self.assertEqual(receipt['actual_splits'],['test'])
        value,_=evaluate_split(prepare,None,None,2,'val',device='cpu')
        self.assertEqual(value,'val')

    def test_exception_restores_loader_and_train_is_rejected(self):
        loader=lambda *args,**kwargs:None
        def fail(*args,**kwargs):raise RuntimeError('fixture')
        prepare=SimpleNamespace(make_dataloader=loader,evaluate_bpb=fail)
        with self.assertRaises(RuntimeError):evaluate_split(prepare,None,None,2,'test')
        self.assertIs(prepare.make_dataloader,loader)
        with self.assertRaises(ValueError):evaluate_split(prepare,None,None,2,'train')


if __name__=='__main__':unittest.main(verbosity=2)
