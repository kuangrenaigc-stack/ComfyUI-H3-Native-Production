"""Standalone FP32 whole-clip model run; no TP process or Comfy runtime."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from safetensors.torch import load_file
from latent_contract import load, save, sha
from adapter_safety import atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    # This marker commits all output files; a missing marker permits safe recomputation.
    args.output.with_suffix('.run.json').unlink(missing_ok=True)
    base=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(base/'architecture/ComfyUI-H3-Latent-Upscaler-Mamad8-main'))
    from upscaler import read_checkpoint_info, build_upscaler
    checkpoint=base/'models/h3_clean_latent_upscaler_film_epoch200.safetensors'
    if sha(checkpoint) != '984afb58f11d01274b90d880596ce2f93bc9c512db66fdeecd4e2d99b371d3e4':
        raise ValueError('Checkpoint checksum changed')
    data,meta=load(args.source)
    model=build_upscaler(load_file(str(checkpoint),device='cpu'),read_checkpoint_info(checkpoint)).float().to(args.device)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with torch.inference_mode():
        x=data['video'].to(args.device,dtype=torch.float32)
        y=model(x)
        if tuple(y.shape) != (*x.shape[:-2],x.shape[-2]*2,x.shape[-1]*2):
            raise ValueError('Model output is not exact spatial 2x')
        b,c,t,h,w=x.shape
        bilinear=torch.nn.functional.interpolate(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),
                       size=(h*2,w*2),mode='bilinear',align_corners=False)
        bilinear=bilinear.reshape(b,t,c,h*2,w*2).permute(0,2,1,3,4)
        residual_ratio=((y-bilinear).abs().mean()/bilinear.abs().mean().clamp_min(1e-6)).item()
        if residual_ratio<.005:
            raise ValueError('Learned correction nearly zero; refusing an unnoticed bilinear fallback')
        y=y.cpu()
    elapsed=time.monotonic()-started
    result=save(args.output,y,num_frames=meta['num_frames'],pixel_height=meta['pixel_height']*2,
                pixel_width=meta['pixel_width']*2,source_sha256=sha(args.source),model_sha256=sha(checkpoint))
    result.update(forward_and_copy_seconds=elapsed,residual_bilinear_ratio=residual_ratio,chunking=False,dtype='float32',
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if args.device.startswith('cuda') else None)
    atomic_json(args.output.with_suffix('.run.json'),result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
