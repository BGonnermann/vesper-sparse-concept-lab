"""CPU-only provenance/admission tests; never touch real test data or train a model."""
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_long_baseline as e

class EvaluationGuards(unittest.TestCase):
    def fixture(self,stack,root):
        cfg=dict(profile='fixture.json',updates=1,seed=301)
        result=dict(latest=dict(path=str(root/'checkpoint.pt')),config_sha256='hash',status='completed',step=1)
        state=dict(config_sha256='hash',source_hashes={},upstream={},dataset_fingerprint='data',tokenizer_sha256='tok',step=1)
        stack.enter_context(patch.object(e,'read',side_effect=lambda p:cfg if Path(p).name=='config.json' else result))
        stack.enter_context(patch.object(e,'validate_profile',return_value=({'tokenizer':{'sha256':'tok'}},{'fingerprint':'data'})))
        stack.enter_context(patch.object(e,'load_state',return_value=state))
        stack.enter_context(patch.object(e,'digest',return_value='hash'))
        stack.enter_context(patch.object(e,'source_identity',return_value={}))
        stack.enter_context(patch.object(e.r,'verify_runtime',return_value={}))
        builder=stack.enter_context(patch.object(e,'build'));evaluator=stack.enter_context(patch.object(e,'evaluate'))
        return state,builder,evaluator

    def test_recovery_refuses_completed_reference(self):
        from supervise_long_resume import validate_recovery
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'result.json').write_text('{"status":"completed"}')
            with self.assertRaisesRegex(ValueError,'already completed'):validate_recovery(root)

    def test_recovery_refuses_opened_test(self):
        from supervise_long_resume import validate_recovery
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'test-opened.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Test already opened'):validate_recovery(root)

    def test_existing_output_preserved_before_any_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'existing.json';out.write_text('preserve')
            with patch.object(e,'load_state') as load:
                with self.assertRaises(ValueError):e.execute(Path(tmp),out,'finalize')
                load.assert_not_called();self.assertEqual(out.read_text(),'preserve')

    def test_second_test_opening_rejected_before_model_build(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);state,build,evaluate=self.fixture(stack,root);(root/'test-opened.json').write_text('existing')
            with self.assertRaisesRegex(ValueError,'already opened'):e.execute(root,root/'new-output','finalize')
            build.assert_not_called();evaluate.assert_not_called()

    def test_nonfinal_checkpoint_rejected_before_test(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);state,build,evaluate=self.fixture(stack,root);state['step']=0
            with self.assertRaisesRegex(AssertionError,'Final fixed-budget'):e.execute(root,root/'new-output','finalize')
            build.assert_not_called();evaluate.assert_not_called();self.assertFalse((root/'test-opened.json').exists())

    def test_source_drift_rejected_before_model_build(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);state,build,evaluate=self.fixture(stack,root);state['source_hashes']={'changed':'hash'}
            with self.assertRaises(AssertionError):e.execute(root,root/'new-output','generate')
            build.assert_not_called();evaluate.assert_not_called()

    def test_generation_never_calls_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);state,build,evaluate=self.fixture(stack,root)
            model=torch.nn.Linear(2,2);state['model']=model.state_dict();build.return_value=(model,object(),None,0)
            stack.enter_context(patch.object(e,'samples',return_value=[]));stack.enter_context(patch.object(e,'log'))
            out=root/'generation.json';e.execute(root,out,'generate');evaluate.assert_not_called()
            import json
            receipt=json.loads(out.read_text());self.assertEqual(receipt['status'],'verified');self.assertEqual(receipt['weights_before'],receipt['weights_after'])
            self.assertFalse((root/'test-opened.json').exists())

if __name__=='__main__':unittest.main()
