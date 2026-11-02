"""CPU checks for packing identity and malformed/damaged handoffs."""
import os
from pathlib import Path
import tempfile
import sys
import torch
from latent_contract import load, save
sys.path.insert(0,'/opt/h3/runtime/source')
from lightx2v.models.networks.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens

with tempfile.TemporaryDirectory() as temp:
    x=torch.randn(1,24,37,24,40)
    rows=patchify_video_latents(x)
    assert torch.equal(x,unpatchify_video_tokens(rows,37,24,40))
    path=Path(temp)/'clean.safetensors'
    receipt=save(path,x,num_frames=124,pixel_height=384,pixel_width=640)
    data,meta=load(path,receipt['sha256'])
    assert torch.equal(data['video'],x)
    try:load(path,'0'*64)
    except ValueError:pass
    else:raise AssertionError('Damaged binding accepted')
    try:save(Path(temp)/'bad.safetensors',torch.full_like(x,float('nan')))
    except ValueError:pass
    else:raise AssertionError('Nonfinite latent accepted')
print('PASS: native packing roundtrip, safetensors roundtrip, checksum rejection, nonfinite rejection')
