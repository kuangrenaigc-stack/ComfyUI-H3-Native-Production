"""One-job VAE memory fix; retain sampling and the original RGB arithmetic."""
import functools
import json
import os
from pathlib import Path

import torch


def normalize_pixels(video, pixel_mean, pixel_std, dtype):
    mean = pixel_mean.to(device=video.device).view(1, -1, 1, 1, 1)
    std = pixel_std.to(device=video.device).view(1, -1, 1, 1, 1)
    return video.to(dtype).mul_(std).add_(mean).clamp_(0, 1)


def install():
    from loguru import logger
    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE
    from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner

    job = Path(os.environ['H3_BENCH_JOB'])
    rank = int(os.environ.get('RANK', 0))
    original_decoder = MiniMaxH3Runner.run_vae_decoder

    def postprocess(self, video):
        result = normalize_pixels(video, self.pixel_mean, self.pixel_std, self.sensitive_layer_dtype)
        data = dict(rank=rank, shape=list(result.shape), dtype=str(result.dtype),
                    formula='clamp(video * std + mean, 0, 1)', implementation='in-place multiply/add')
        (job/f'decode-memory-rank{rank}.json').write_text(json.dumps(data,indent=2))
        logger.info('H3_LOW_MEMORY_RGB_POSTPROCESS {}', data)
        return result

    @functools.wraps(original_decoder)
    def checkpoint_and_decode(self, video_rows, audio_rows):
        state = {key: getattr(self.scheduler,key) for key in (
            'num_condition_video_rows','num_condition_audio_rows','num_latent_frames',
            'latent_height','latent_width','num_audio_latents')}
        checkpoint = job/f'sampled-latents-rank{rank}.pt'
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(dict(video_rows=video_rows.detach().cpu(),audio_rows=audio_rows.detach().cpu(),
                        scheduler=state,config=dict(self.config)),temporary)
        temporary.replace(checkpoint)
        logger.info('H3_DECODE_CHECKPOINT {}', checkpoint)
        return original_decoder(self,video_rows,audio_rows)

    MiniMaxH3VideoVAE.postprocess = postprocess
    MiniMaxH3Runner.run_vae_decoder = checkpoint_and_decode
