"""Real H3 mixed-weight TP2 FFN benchmark + optional upstream rank traces.

Run with torchrun under an exclusive runtime/control lock. Inputs are synthetic;
weights, quantized kernels, dynamic LoRA, TP sharding and reduction are native.
Does not load/modify a full model, render media, or select production defaults.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','weights-dir','lora-file','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',type=int,default=32768)
    p.add_argument('--chunks',type=int,nargs='+',default=[8192,16384,32768])
    p.add_argument('--blocks',type=int,nargs='+',default=[0,29])
    p.add_argument('--profile',action='store_true')
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    for path in [here/'upstream',here.parent/'adapter',args.runtime_root/'production',args.runtime_root/'source',args.runtime_root/'python-deps']:
        sys.path.insert(0,str(path))
    sys.path.insert(0,str(here/'upstream'))
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from safetensors import safe_open
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank=dist.get_rank();mesh=init_device_mesh('cuda',(2,),mesh_dim_names=('tensor_p',))
    group=mesh.get_group('tensor_p');device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    if dist.get_world_size()!=2:raise ValueError('This diagnostic requires the production TP2 degree')
    from feihou_adapter import install as install_mixed
    install_mixed()
    worker=Path(os.environ.get('H3_NODE_WORKER',str(args.runtime_root/'node-worker')))
    sys.path.insert(0,str(worker))
    from lora_memory import install as install_lora_memory
    install_lora_memory()
    from lightx2v.models.networks.minimax_h3.lora import MiniMaxH3LoraAdapter
    from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
    from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3FeedForwardWeights
    from ffn_memory import chunked_ff,low_memory_ff
    from tools.benchmarks.module_bench import PreparedModule,benchmark
    from tools.profile.capture import Capture
    args.output.mkdir(parents=True,exist_ok=True)
    config=dict(tensor_parallel=True,device_mesh=mesh,dit_quant_scheme='int8-convrot',
                feihou_precision_manifest=str(args.weights_dir/'alignment-manifest.json'))
    shard=types.SimpleNamespace(config=config,tp_rank=rank,tp_size=2,_tp_split_type=MiniMaxH3Model._tp_split_type)
    summary=[]
    for index in args.blocks:
        weights=MiniMaxH3FeedForwardWeights(f'transformer_blocks.{index}.ff',config)
        data={}
        with safe_open(str(args.weights_dir/f'block-{index:02}.safetensors'),framework='pt',device='cpu') as f:
            for key in f.keys():
                if '.ff.' in key:
                    value=f.get_tensor(key)
                    data[key]=MiniMaxH3Model._select_tensor_parallel_shard(shard,key,value).to(device)
        weights.load(data);del data
        lora={}
        with safe_open(str(args.lora_file),framework='pt',device='cpu') as f:
            for key in f.keys():
                if key.startswith(f'transformer_blocks.{index}.ff.'):
                    normalized=MiniMaxH3LoraAdapter._normalize_lora_key(key)
                    if normalized:lora[normalized]=f.get_tensor(key).to(device)
        # Native weights use transformer_blocks as their LoRA prefix.
        weights.register_lora(lora,1.0)
        for proj in (weights.in_proj,weights.out_proj):
            if not proj._mm.has_lora_branch:raise ValueError('Expected native LoRA branch was not bound')
            proj._mm.lora_scale=torch.tensor(8/proj._mm.lora_down.shape[0],device=device)
        del lora
        torch.manual_seed(317);x=torch.randn(args.rows,5376,dtype=torch.bfloat16,device=device)*.1
        with torch.inference_mode():reference=low_memory_ff(weights,x)
        for chunk in [0]+args.chunks:
            with torch.inference_mode():actual=chunked_ff(weights,x,chunk)
            error_sq=reference_sq=0.0;max_abs=0.0;exact=True;finite=True
            for start in range(0,args.rows,4096):
                a=actual[start:start+4096].float();b=reference[start:start+4096].float();error=a-b
                error_sq+=float(error.square().sum());reference_sq+=float(b.square().sum())
                max_abs=max(max_abs,float(error.abs().max()));exact &= bool(torch.equal(a,b));finite &= bool(torch.isfinite(a).all())
            del a,b,error
            metrics=dict(max_abs=max_abs,normalized_rmse=(error_sq/max(reference_sq,1e-24))**.5,bit_exact=exact)
            if not finite or metrics['normalized_rmse']>.005:
                raise ValueError(f'FFN numerical validation failed: {metrics}')
            del actual
            identity=dict(target=dict(model='H3 Ref2AV',block=index,operation='FFN',tp=2,chunk_rows=chunk),
                          input=dict(kind='synthetic',seed=317,shape=list(x.shape),dtype='bf16'),
                          source=dict(ffn_sha256=hashlib.sha256((here.parent/'adapter/ffn_memory.py').read_bytes()).hexdigest(),
                                      native_revision=(args.runtime_root/'UPSTREAM_REVISION.txt').read_text().strip(),lora=args.lora_file.name))
            def call():
                with torch.inference_mode():return chunked_ff(weights,x,chunk)
            def check(y):return dict(status='strict_equal' if metrics['bit_exact'] else 'pending_review',scope='full FFN vs native whole-row reference',**metrics)
            prepared=PreparedModule(call,lambda:None,torch.cuda.synchronize,identity,check,str(device),group,group,None)
            cpu_start=time.process_time();wall_start=time.perf_counter()
            report,raw=benchmark(prepared,warmup=args.warmup,iterations=args.iterations,repeats=args.repeats)
            cpu_seconds=time.process_time()-cpu_start;wall_seconds=time.perf_counter()-wall_start
            cpu_local=dict(rank=rank,cpu_seconds=cpu_seconds,wall_seconds=wall_seconds,
                           cpu_cores_equivalent=cpu_seconds/wall_seconds,
                           threads=len(list(Path('/proc/self/task').iterdir())),
                           omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                           scope='whole benchmark phase including warmup and coordination; not only NCCL')
            cpu_rows=[None]*2;dist.all_gather_object(cpu_rows,cpu_local,group=group)
            report['cpu_use']=cpu_rows
            if rank==0:
                (args.output/f'block{index}-chunk{chunk}.json').write_text(json.dumps(dict(report=report,raw=raw),indent=2))
                summary.append(dict(block=index,chunk_rows=chunk,latency_ms=report['latency_ms'],ranks=report['ranks'],cpu_use=cpu_rows))
            if args.profile and chunk==16384:
                run_id=f'block{index}-chunk{chunk}'
                meta=dict(identity=identity,rank=rank,tp_ranks=[0,1],sp_ranks=[rank],measurement_ranks=[0,1],run_id=run_id)
                with Capture(args.output/f'{run_id}/rank{rank}',enabled=True,warmup=1,count=1,workloads=False,metadata=meta) as capture:
                    with torch.inference_mode():
                        capture.call(call);capture.call(call)
                dist.barrier()
        del reference,x,weights
        gc.collect();torch.cuda.empty_cache();dist.barrier()
    if rank==0:
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print('H3_FFN_BENCH_COMPLETE',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
