import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import foundation_data as data

class DataTests(unittest.TestCase):
    def row(self,ident,split='train',text=None):
        return dict(id=ident,split=split,source='fixture',domain='general',text=text or ' '.join('word'+str(i) for i in range(80)))
    def test_normalization_preserves_code(self):
        self.assertEqual(data.normalize('def x():\r\n\treturn "cafe\u0301"'), 'def x():\n\treturn "café"')
    def test_heldout_wins_exact_leakage(self):
        rows,stats=data.clean([self.row('train'),self.row('val','validation'),self.row('test','test')])
        self.assertEqual([x['id'] for x in rows],['test']);self.assertEqual(stats['fixture:exact_rejected'],2)
    def test_near_duplicate_leakage(self):
        text=self.row('x')['text'];rows,stats=data.clean([self.row('a','test',text),self.row('b','train',text+' one change')])
        self.assertEqual(len(rows),1);self.assertEqual(stats['fixture:near_rejected'],1)
    def test_shared_paragraph_leakage(self):
        p=self.row('x')['text'];rows,stats=data.clean([self.row('a','test',p+'\n\n'+'A '*200),self.row('b','train',p+'\n\n'+'B '*200)])
        self.assertEqual(len(rows),1);self.assertEqual(stats['fixture:shared_paragraph_rejected'],1)
    def test_malformed(self):
        for rows in [[{}],[self.row('x',text=123)],[self.row('x'),self.row('x')],[self.row('x','invalid')]]:
            with self.assertRaises(ValueError):data.clean(rows)
    def test_order_independence(self):
        rows=[self.row('a','test'),self.row('b','train'),self.row('c',text='different '*100)]
        self.assertEqual(data.clean(rows),data.clean(list(reversed(rows))))
    def test_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'train.jsonl').write_text('good');data.write(p/'manifest.json',dict(pipeline_sha256=data.digest(data.__file__),outputs={'train.jsonl':data.digest(p/'train.jsonl')}))
            data.verify(p);(p/'train.jsonl').write_text('bad')
            with self.assertRaisesRegex(ValueError,'Corrupted'):data.verify(p)
    def test_article_boundaries(self):
        rows=list(data.wiki_articles([' = Article one = \n','Paragraph\n',' == Section == \n','More\n',' = Article two = \n','Last\n'],'train'))
        self.assertEqual(len(rows),2);self.assertIn('Section',rows[0]['text'])

class ByteEncoding:
    def encode_ordinary(self,text):return list(text.encode())
    def decode(self,ids):return bytes(ids).decode()
    def decode_single_token_bytes(self,i):return bytes([i])

class EvaluationTests(unittest.TestCase):
    def test_all_bytes_and_shift(self):
        from foundation_eval import token_windows
        text='café 中文\t x\n'*100;enc=ByteEncoding();pairs=list(token_windows(enc,text,256,context=13))
        xs=sum((x for x,y in pairs),[]);ys=sum((y for x,y in pairs),[])
        self.assertEqual(ys,list(text.encode()));self.assertEqual(xs,[256]+ys[:-1])
    def test_bpb_uniform_reference(self):
        import torch
        from foundation_eval import evaluate
        class Model:
            def eval(self):pass
            def __call__(self,x,y,reduction):return torch.full_like(y,math.log(256),dtype=torch.float64)
        text='Unicode 中文 café'
        rows=[dict(id='a',domain='general',text=text,sha256=data.sha(text.encode()))]
        result=evaluate(Model(),ByteEncoding(),rows,256,device='cpu')
        self.assertAlmostEqual(result['aggregate_bpb'],8,places=6)
    def test_split_rejection_and_determinism(self):
        from foundation_eval import selected_inputs
        rows=[dict(id=str(i),domain='general',split='validation',text='some text') for i in range(20)]
        self.assertEqual(selected_inputs(rows),selected_inputs(list(reversed(rows))))
        rows[0]['split']='train'
        with self.assertRaises(ValueError):selected_inputs(rows)
    def test_sampling(self):
        from foundation_train import make_tape
        import numpy as np
        rows=[dict(id=d,domain=d,text=(d+' ')*200) for d in ('general','technical')]
        a,ra=make_tape(rows,ByteEncoding(),256,201,.8,2);b,rb=make_tape(rows,ByteEncoding(),256,201,.8,2)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(ra,rb)
        self.assertEqual(sum(ra['source_tokens'].values()),32768)
        self.assertTrue(np.array_equal(a[0,0,512:],a[0,1,:1]))
        c,rc=make_tape(rows,ByteEncoding(),256,202,.8,2);self.assertNotEqual(ra['tape_sha256'],rc['tape_sha256'])

if __name__=='__main__':unittest.main()
