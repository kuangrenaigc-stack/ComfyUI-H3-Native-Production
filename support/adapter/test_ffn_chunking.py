"""CPU contracts; real mixed-weight TP2/LoRA validation uses diagnostics/ffn_bench.py."""
import types
import unittest
import torch
from ffn_memory import chunked_ff, low_memory_ff

class ChunkingTests(unittest.TestCase):
    def weights(self,dtype):
        g=torch.Generator().manual_seed(71)
        a=torch.randn(32,128,generator=g,dtype=dtype)*.1
        b=torch.randn(64,32,generator=g,dtype=dtype)*.1
        calls=[]
        def inp(x):calls.append(('in',x.shape[0]));return x@a
        def out(x):calls.append(('out',x.shape[0]));return x@b
        return types.SimpleNamespace(in_proj=types.SimpleNamespace(apply=inp),out_proj=types.SimpleNamespace(apply=out)),calls,a,b

    def test_odd_tail_noncontiguous_input_and_unchanged_weights(self):
        for dtype in (torch.float32,torch.bfloat16):
            with self.subTest(dtype=dtype),torch.inference_mode():
                weights,calls,a,b=self.weights(dtype)
                x=torch.randn(70,32,dtype=dtype)[::2]
                original=x.clone();ac=a.clone();bc=b.clone()
                expected=low_memory_ff(weights,x);calls.clear()
                actual=chunked_ff(weights,x,8)
                torch.testing.assert_close(actual,expected,rtol=.02 if dtype==torch.bfloat16 else 1e-5,atol=.002 if dtype==torch.bfloat16 else 1e-6)
                self.assertEqual(calls,[item for n in (8,8,8,8,3) for item in [('in',n),('out',n)]])
                self.assertTrue(torch.equal(x,original) and torch.equal(a,ac) and torch.equal(b,bc))
                self.assertTrue(actual.is_contiguous())

    def test_small_or_disabled_path_has_one_projection_pair(self):
        for rows,chunk in [(4,8),(0,8),(35,0)]:
            with self.subTest(rows=rows,chunk=chunk),torch.inference_mode():
                weights,calls,*_=self.weights(torch.float32)
                out=chunked_ff(weights,torch.empty(rows,32),chunk)
                self.assertEqual(tuple(out.shape),(rows,32));self.assertEqual(len(calls),2)

    def test_invalid_policy_and_batched_input_rejected(self):
        weights,*_=self.weights(torch.float32)
        for chunk in (-1,True,1.5):
            with self.subTest(chunk=chunk),self.assertRaises(ValueError):chunked_ff(weights,torch.ones(2,32),chunk)
        with self.assertRaises(ValueError):chunked_ff(weights,torch.ones(1,2,32),8)

    def test_training_input_rejected(self):
        weights,*_=self.weights(torch.float32)
        with self.assertRaises(ValueError):chunked_ff(weights,torch.ones(2,32,requires_grad=True),1)

if __name__=='__main__':unittest.main()
