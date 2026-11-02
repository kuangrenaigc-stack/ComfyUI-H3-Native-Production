"""Portable clean H3 diffusion-space latent handoff, without ComfyUI imports."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCHEMA = 'h3-clean-latent-2x/1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def tensor_sha(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def stats(x):
    x = x.detach().float().cpu()
    return dict(mean=x.mean().item(), std=x.std().item(),
                channel_mean=x.mean((0, 2, 3, 4)).tolist(),
                channel_std=x.std((0, 2, 3, 4)).tolist())

def validate(video, meta):
    if meta.get('schema') != SCHEMA or meta.get('space') != 'normalized-diffusion':
        raise ValueError('Unsupported latent space/contract')
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24):
        raise ValueError('Expected [1,24,T,H,W]')
    if list(video.shape) != meta.get('shape') or not video.is_floating_point() or not torch.isfinite(video).all():
        raise ValueError('Invalid latent shape or values')

def save(path, video, *, audio=None, **info):
    path = Path(path)
    video = video.detach().float().cpu().contiguous()
    meta = dict(info, schema=SCHEMA, space='normalized-diffusion', shape=list(video.shape))
    validate(video, meta)
    tensors = dict(video=video)
    if audio is not None:
        tensors['audio'] = audio.detach().float().cpu().contiguous()
    temp = path.with_suffix('.tmp')
    save_file(tensors, str(temp), metadata={'contract': json.dumps(meta)})
    temp.replace(path)
    receipt = dict(path=str(path), sha256=sha(path), video_sha256=tensor_sha(video),
                   metadata=meta, statistics=stats(video))
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def load(path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Latent checksum mismatch')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        meta = json.loads(f.metadata()['contract'])
    tensors = load_file(str(path), device='cpu')
    validate(tensors['video'], meta)
    return tensors, meta
