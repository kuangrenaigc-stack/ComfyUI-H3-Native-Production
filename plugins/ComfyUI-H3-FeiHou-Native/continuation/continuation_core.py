"""Native H3 continuation contracts. No model loading or GPU work on import."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCHEMA = "h3_native_continuation_tail/1"
FPS = 24
OVERLAP = 22
TAIL_STEPS = 7  # 22 pixel frames on H3's 5 + 17N grid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_frames(seconds):
    import math
    if type(seconds) is not int or not 5 <= seconds <= 15:
        raise ValueError("Each native segment must request 5–15 seconds")
    return 5 + 17 * math.ceil((seconds * FPS - 5) / 17)


def plan(seconds=15):
    frames = segment_frames(seconds)
    return dict(segments=2, fps=FPS, frames_per_segment=frames, overlap_frames=OVERLAP,
                output_frames=2 * frames - OVERLAP,
                output_seconds=(2 * frames - OVERLAP) / FPS,
                steps_per_segment=[10, 5], intermediate_review=False)


def validate_tail(video, audio, metadata, height=None, width=None):
    if metadata.get("schema") != SCHEMA or metadata.get("fps") != FPS:
        raise ValueError("Unsupported native tail schema or FPS")
    if video.ndim != 5 or tuple(video.shape[:3]) != (1, 24, TAIL_STEPS):
        raise ValueError("Tail video must be [1,24,7,H,W]")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (2, 32) or audio.shape[-1] != round(OVERLAP / FPS * 40):
        raise ValueError("Tail audio must contain the matching 22-frame stereo context")
    if not all(t.is_floating_point() and bool(torch.isfinite(t).all()) for t in (video, audio)):
        raise ValueError("Tail tensors must be finite floating point values")
    if tuple(video.shape[-2:]) != (metadata.get('height', 0) // 16, metadata.get('width', 0) // 16):
        raise ValueError('Tail geometry disagrees with its metadata')
    if metadata.get('overlap_frames') != OVERLAP:
        raise ValueError('Unsupported overlap')
    if height is not None and tuple(video.shape[-2:]) != (height // 16, width // 16):
        raise ValueError("Continuation canvas changed; implicit latent resize/crop is not allowed")


def save_tail(path, video_rows, audio_rows, scheduler):
    """Export only generated target rows; never retain original reference rows."""
    state = scheduler
    h, w = state.latent_height, state.latent_width
    t, a = state.num_latent_frames, state.num_audio_latents
    vr = video_rows[state.num_condition_video_rows:].detach().cpu()
    ar = audio_rows[state.num_condition_audio_rows:].detach().cpu()
    if vr.shape != (t * (h // 2) * (w // 2), 96) or ar.shape != (2 * a, 32):
        raise ValueError("Native checkpoint geometry does not match target rows")
    video = vr.reshape(1, t, h // 2, w // 2, 24, 1, 2, 2)
    video = video.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, t, h, w)
    audio = ar.reshape(2, a, 32).permute(0, 2, 1)
    video = video[:, :, -TAIL_STEPS:].contiguous().clone()
    audio = audio[..., -round(OVERLAP / FPS * 40):].contiguous().clone()
    metadata = dict(schema=SCHEMA, fps=FPS, overlap_frames=OVERLAP,
                    height=h * 16, width=w * 16, source_frames=state.num_frames)
    validate_tail(video, audio, metadata)
    path = Path(path)
    temp = path.with_suffix(".tmp")
    save_file(dict(video=video, audio=audio), str(temp), metadata={"contract": json.dumps(metadata)})
    temp.replace(path)
    return dict(path=str(path), sha256=sha256(path), **metadata)


def read_tail(path, expected_sha, height=None, width=None):
    path = Path(path).resolve(strict=True)
    if sha256(path) != expected_sha:
        raise ValueError("Continuation tail SHA256 mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = json.loads((handle.metadata() or {}).get("contract", "{}"))
    tensors = load_file(str(path), device="cpu")
    validate_tail(tensors["video"], tensors["audio"], metadata, height, width)
    return tensors["video"], tensors["audio"], metadata


def attach_guide(scheduler, video, audio):
    """Insert fixed AV guide tokens at target time zero, alongside existing refs.

    This mirrors the H3 keyframe/guide representation: fixed-condition rows,
    target-relative RoPE positions, and native per-row condition timesteps.
    It changes no tensor-parallel layer, collective, weight or sampler update.
    """
    old = scheduler.layout_cpu
    device = scheduler.video_latents.device
    h, w = scheduler.latent_height, scheduler.latent_width
    if tuple(video.shape) != (1, 24, TAIL_STEPS, h, w):
        raise ValueError("Guide shape does not match the current phase canvas")
    gv = video.reshape(1, 24, TAIL_STEPS, 1, h // 2, 2, w // 2, 2)
    gv = gv.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, 96).float()
    if tuple(audio.shape) != (2, 32, round(OVERLAP / FPS * 40)):
        raise ValueError('Invalid guide audio shape')
    ga = audio.transpose(1, 2).reshape(-1, 32).float()
    nv, na = len(gv), len(ga)
    count = nv + na
    text_end = len(old.text_indices)
    # Keep the native target noise stream unchanged. Guide augmentation has
    # its own deterministic CPU generator, identical on both TP ranks.
    gen = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn(gv.shape, generator=gen, dtype=torch.float32)
    gv, noise = gv.to(device), noise.to(device)
    timestep = torch.tensor(0.999, dtype=gv.dtype, device=device)
    gv = timestep * gv + (1.0 - timestep) * noise
    ga = ga.to(scheduler.audio_latents.device)
    # Guide tokens precede refs, as in ComfyUI H3 PackedLayout. Their time
    # positions overlap the first target frames, not the reference timeline.
    vpos = old.position_ids[old.video_indices[old.num_condition_video_rows:][:nv]]
    target_a = old.audio_indices[old.num_condition_audio_rows:].reshape(2, scheduler.num_audio_latents)
    apos = old.position_ids[target_a[:, :audio.shape[-1]].reshape(-1)]
    positions = torch.cat((old.position_ids[:text_end], vpos, apos, old.position_ids[text_end:]))
    tags = torch.cat((old.token_tags[:text_end], torch.zeros(nv, dtype=torch.long),
                      torch.full((na,), 2, dtype=torch.long), old.token_tags[text_end:]))
    vindex = torch.cat((torch.arange(text_end, text_end + nv), old.video_indices + count))
    aindex = torch.cat((torch.arange(text_end + nv, text_end + count), old.audio_indices + count))
    layout = replace(old, sequence_length=old.sequence_length + count, position_ids=positions,
                     token_tags=tags, video_indices=vindex, audio_indices=aindex,
                     num_condition_video_rows=old.num_condition_video_rows + nv,
                     num_condition_audio_rows=old.num_condition_audio_rows + na)
    if len(torch.unique(torch.cat((layout.text_indices, vindex, aindex)))) != layout.sequence_length:
        raise ValueError("Guide insertion did not produce a complete disjoint token layout")
    scheduler.video_latents = torch.cat((gv, scheduler.video_latents))
    scheduler.audio_latents = torch.cat((ga, scheduler.audio_latents))
    scheduler.layout_cpu = layout
    scheduler.layout = replace(layout, position_ids=positions.to(device=device, dtype=torch.float32),
                               token_tags=tags.to(device), video_indices=vindex.to(device),
                               audio_indices=aindex.to(device), text_indices=layout.text_indices.to(device))
    scheduler.num_condition_video_rows = layout.num_condition_video_rows
    scheduler.num_condition_audio_rows = layout.num_condition_audio_rows
    return dict(video_guide_rows=nv, audio_guide_rows=na, guide_frames=OVERLAP,
                target_noise_unchanged=True, parallel_implementation_unchanged=True)
