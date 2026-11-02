"""Exact FFN output and call-count equivalence; GPU optional for development."""
import sys
import types
import torch
import torch.nn.functional as F
from ffn_memory import low_memory_ff

device='cuda:0' if '--cuda' in sys.argv else 'cpu'
torch.manual_seed(317)
for dtype in (torch.float32,torch.bfloat16):
    x=torch.randn(1024,128,device=device,dtype=dtype)
    a=torch.randn(128,512,device=device,dtype=dtype)
    b=torch.randn(256,128,device=device,dtype=dtype)
    calls=[]
    def inp(y):calls.append('in');return y@a
    def out(y):calls.append('out');return y@b
    weights=types.SimpleNamespace(in_proj=types.SimpleNamespace(apply=inp),out_proj=types.SimpleNamespace(apply=out))
    with torch.inference_mode():
        projected=x@a;value,gate=projected.chunk(2,dim=-1)
        expected=(value*F.silu(gate))@b
        actual=low_memory_ff(weights,x)
    assert torch.equal(expected,actual),(device,dtype,(expected-actual).abs().max().item())
    assert calls==['in','out']
    print('PASS',device,dtype,'bit-exact; one input/output projection each')
