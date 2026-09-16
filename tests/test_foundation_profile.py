import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_foundation_profile as runner

class ProfileTests(unittest.TestCase):
    def test_closed_campaign_guard(self):
        with self.assertRaises(ValueError):runner.validate_output(runner.HOME/'new-training-is-forbidden')
    def test_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):runner.validate_output(tmp)
    def test_new_study_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';self.assertEqual(runner.validate_output(p),p.resolve())
    def test_launcher_plumbing_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'new';profile=root/'profile.json'
            profile.write_text(json.dumps(dict(architecture_config='experiments/mainline/dense-v1.json',total_parameters=135267480,
                tokenizer=dict(path='unused',sha256='hash'),data=dict(path='unused',fingerprint='data',general_weight=.8),
                training=dict(sequence_length=512,microbatch_size=2,tokens_per_update=16384,optimizer_updates=512,
                    activation_checkpointing=False,matrix_lr=.04,amp='bfloat16'))))
            def fake_child(command,path,timeout):
                self.assertIn('--seed',command);self.assertIn('301',command)
                output.mkdir();(output/'result.json').write_text(json.dumps(dict(status='completed',dataset_fingerprint='data',
                    tokenizer=dict(sha256='hash'),parameters=dict(total_parameters=135267480),training_tokens=8388608)))
            with patch.object(runner,'verify',return_value={'fingerprint':'data'}),patch.object(runner,'digest',return_value='hash'),patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)),patch.object(runner.supervision,'checked',side_effect=fake_child),patch.object(runner.supervision,'log'):
                runner.run(profile,output,301,1200)
            receipt=json.loads((root/'new-supervisor.json').read_text());self.assertEqual(receipt['status'],'completed')

if __name__=='__main__':unittest.main()
