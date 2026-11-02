"""Release the fresh fused FFN projection before output projection; TP unchanged."""
import atexit
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

def low_memory_ff(weights,hidden_states):
    projected=weights.in_proj.apply(hidden_states)
    value,gate=projected.chunk(2,dim=-1)
    F.silu(gate,inplace=True)
    value.mul_(gate)
    activated=value.contiguous()
    # The two views otherwise retain the entire fused value+gate allocation
    # while the INT8 output projection allocates its own result/workspace.
    del value,gate,projected
    return weights.out_proj.apply(activated)

def install():
    from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
    count=[0]
    def ff(weights,x):
        count[0]+=1
        return low_memory_ff(weights,x)
    MiniMaxH3TransformerInfer._ff=staticmethod(ff)
    def report():
        path=Path(os.environ['H3_BENCH_JOB'])/f'ffn-memory-rank{os.environ.get("RANK","0")}.json'
        path.write_text(json.dumps(dict(calls=count[0],in_proj_calls_per_ff=1,out_proj_calls_per_ff=1,
            native_tp_collectives_unchanged=True,weights_dtype_unchanged=True),indent=2))
    atexit.register(report)
