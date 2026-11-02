"""Process-local hooks; never replace installed production source or TP layers."""
import functools
import json
import os
from pathlib import Path
from latent_contract import load, save, tensor_sha, sha

def install():
    import torch
    from lightx2v.models.runners.minimax_h3 import minimax_h3_runner as native
    from lightx2v.models.networks.minimax_h3.packing import (
        unpatchify_video_tokens, unpack_audio_tokens, video_latent_num_frames)
    cls = native.MiniMaxH3Runner
    decode, source = cls.run_vae_decoder, cls._prepare_resampling_source

    @functools.wraps(decode)
    def export(self, video_rows, audio_rows):
        job = Path(os.environ['H3_BENCH_JOB'])
        rank = int(os.environ.get('RANK', 0))
        if self.config.get('h3_sampling_stage') == 'native-draft':
            s = self.scheduler
            if s.draft_steps is not None or s.step_index != s.infer_steps - 1 or float(s.video_sigmas[-1]) != 0:
                raise ValueError('Export requires complete native schedule to sigma zero')
            v = unpatchify_video_tokens(video_rows[s.num_condition_video_rows:], s.num_latent_frames,
                    s.latent_height, s.latent_width, 24, tuple(self.config.get('patch_size', (1,2,2))))
            a = unpack_audio_tokens(audio_rows[s.num_condition_audio_rows:], s.num_audio_latents)
            receipt = dict(rank=rank, shape=list(v.shape), video_sha256=tensor_sha(v),
                           audio_sha256=tensor_sha(a), steps=s.infer_steps, sigma_end=0.)
            (job/f'clean-export-rank{rank}.json').write_text(json.dumps(receipt, indent=2))
            if rank == 0:
                save(job/'clean.safetensors', v, audio=a, num_frames=s.num_frames,
                     pixel_height=s.height, pixel_width=s.width, native_steps=s.infer_steps)
        return decode(self, video_rows, audio_rows)

    @functools.wraps(source)
    def import_source(self):
        spec_path = os.environ.get('H3_LEARNED_SOURCE')
        if not spec_path:
            result = source(self)
            if self.source_audio_latents is not None:
                job=Path(os.environ['H3_BENCH_JOB'])
                rank=int(os.environ.get('RANK',0))
                (job/f'rgb-source-rank{rank}.json').write_text(json.dumps(dict(
                    rank=rank, audio_sha256=tensor_sha(self.source_audio_latents),
                    video_shape=list(self.source_video_latents.shape)),indent=2))
            return result
        spec = json.loads(Path(spec_path).read_text())
        tensors, meta = load(spec['path'], spec['sha256'])
        v = tensors['video']
        expected = (1,24,video_latent_num_frames(self.request_num_frames),self.request_height//16,self.request_width//16)
        if tuple(v.shape) != expected or meta['num_frames'] != self.request_num_frames:
            raise ValueError('Learned source geometry/duration does not match final request')
        # Match baseline audio encoding byte-for-byte. Do not run video encoding.
        video_path = self._resample_option('resample_source_video')
        frames, fps, soundtrack = native.decode_reference_video(video_path)
        if abs(fps-24) > 1e-4 or len(frames) != self.request_num_frames or soundtrack is None:
            raise ValueError('Invalid baseline audio source')
        del frames
        self._ensure_vae_loaded()
        n = native.audio_latent_num_frames(self.request_num_frames) * self.audio_vae.hop_length
        waveform, sample_rate = soundtrack
        waveform = native.prepare_reference_waveform(waveform, sample_rate, self.audio_vae.sampling_rate,
                                                    self.request_num_frames/24. + .1)
        missing = n-waveform.shape[-1]
        if missing > self.audio_vae.sampling_rate/24:
            raise ValueError('Baseline audio is too short')
        waveform = torch.nn.functional.pad(waveform,(0,max(0,missing)))[:,:n].contiguous()
        self.source_video_latents = v
        self.source_audio_latents = self.audio_vae.encode(waveform).float()
        job = Path(os.environ['H3_BENCH_JOB'])
        rank = int(os.environ.get('RANK',0))
        (job/f'learned-source-rank{rank}.json').write_text(json.dumps(dict(
            rank=rank, path=spec['path'], sha256=spec['sha256'], shape=list(v.shape),
            video_sha256=tensor_sha(v), audio_sha256=tensor_sha(self.source_audio_latents),
            noise='native flow_renoise', tp_core_modified=False), indent=2))

    cls.run_vae_decoder = export
    cls._prepare_resampling_source = import_source
