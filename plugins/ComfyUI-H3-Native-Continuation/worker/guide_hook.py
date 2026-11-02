"""Process-local extension. Context comes only from segment 1 final latents."""
import functools
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuation_core import attach_guide, read_tail, save_tail, OVERLAP, TAIL_STEPS


def install():
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file, load_file
    from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler
    from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
    from lightx2v.utils.utils import resize_resampling_frames
    if getattr(MiniMaxH3Scheduler, '_native_continuation_installed', False):
        raise RuntimeError('Continuation hook already installed')
    original_source = MiniMaxH3Runner._prepare_resampling_source
    original_prepare = MiniMaxH3Scheduler.prepare
    original_decode = MiniMaxH3Runner.run_vae_decoder
    pending = {}

    @functools.wraps(original_source)
    def source(self, *args, **kwargs):
        result = original_source(self, *args, **kwargs)
        job = Path(os.environ['H3_BENCH_JOB']).resolve()
        spec = json.loads(Path(os.environ['H3_CONTINUATION_MAP']).read_text()).get(str(job))
        if not spec:
            return result
        video, audio, metadata = read_tail(spec['path'], spec['sha256'])
        target = (self.request_height, self.request_width)
        converted = tuple(video.shape[-2:]) != (target[0] // 16, target[1] // 16)
        if converted:
            # Before the existing phase runner releases VAEs and loads DiT.
            # Both ranks participate in the existing VAE tile-parallel calls.
            self._ensure_vae_loaded()
            self.set_vae_decode_tile_shape()
            pixels = self.video_vae.decode(video)
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = job / 'guide-pixels.safetensors'
            if rank == 0:
                if pixels is None or pixels.shape[2] < OVERLAP:
                    raise ValueError('Tail decoder returned insufficient video context')
                frames = self._video_to_uint8_frames(pixels[:, :, -OVERLAP:]).numpy()
                resized = resize_resampling_frames(frames, *target,
                    workers=self.config.get('resample_resize_workers', 4),
                    parallel=False, device='cpu')
                rgb = torch.from_numpy(resized).permute(3, 0, 1, 2)[None].float().div_(255)
                tmp = path.with_suffix('.tmp')
                save_file({'pixels': rgb.contiguous()}, str(tmp))
                tmp.replace(path)
            del pixels
            if dist.is_initialized():
                dist.barrier()
            rgb = load_file(str(path), device='cpu')['pixels']
            video = self.video_vae.encode_condition(rgb, video=True).float()
            del rgb
            if tuple(video.shape) != (1, 24, TAIL_STEPS, target[0] // 16, target[1] // 16):
                raise ValueError(f'Converted tail geometry is incorrect: {video.shape}')
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                path.unlink()
        pending.update(video=video, audio=audio, source=spec,
                       conversion='RGB-Lanczos-VAE' if converted else 'direct-final-latent')
        return result

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        if pending:
            receipt = attach_guide(self, pending['video'], pending['audio'])
            job = Path(os.environ['H3_BENCH_JOB'])
            rank = int(os.environ.get('RANK', 0))
            (job / f'continuation-guide-rank{rank}.json').write_text(json.dumps(
                dict(source=pending['source'], conversion=pending['conversion'], **receipt), indent=2))
            pending.clear()
        return result

    @functools.wraps(original_decode)
    def decode(self, video_rows, audio_rows):
        if self.config.get('h3_sampling_stage') == 'turbo-final' and int(os.environ.get('RANK', 0)) == 0:
            job = Path(os.environ['H3_BENCH_JOB'])
            receipt = save_tail(job / 'continuation-tail.safetensors', video_rows, audio_rows, self.scheduler)
            (job / 'continuation-tail.json').write_text(json.dumps(receipt, indent=2))
        return original_decode(self, video_rows, audio_rows)

    MiniMaxH3Runner._prepare_resampling_source = source
    MiniMaxH3Scheduler.prepare = prepare
    MiniMaxH3Scheduler._native_continuation_installed = True
    MiniMaxH3Runner.run_vae_decoder = decode
