"""Ref2AV lifecycle adapter for the H3 TP2 deployment."""
import ctypes
import gc
import json
import os
from pathlib import Path

import torch
from loguru import logger
from acceleration import validate_sampling_config

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner


def release_allocator_caches():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


class Ref2VAPhaseRunner(MiniMaxH3Runner):
    def __init__(self, config):
        if config.get('use_adaln_cache'):
            from lightx2v.models.networks.minimax_h3.adaln_cache import _cache_path
            from lightx2v.models.networks.minimax_h3.adaln_binding import verify_cache_binding
            binding = verify_cache_binding(config, _cache_path(config))
            logger.info('H3_ADALN_MODEL_BINDING_PREFLIGHT {}', binding)
        if config.get("model_variant") != "ref2av":
            raise ValueError("This deployment accepts Ref2AV only")
        if config.get("text_encoder_cpu_offload"):
            raise ValueError("This adapter releases components completely between phases")
        config["text_encoder_load_on_init"] = False
        config.setdefault("warmup", False)
        validate_sampling_config(config)
        if config.get('fps') != 24:
            raise ValueError('Deployment requires 24fps')
        if not config.get('cpu_offload') or config.get('offload_granularity') != 'block':
            raise ValueError('This deployment requires official DiT block-level CPU offload')
        if config.get('use_compile') or config.get('feature_caching', 'NoCaching') != 'NoCaching':
            raise ValueError('User disabled torch.compile and DPCache')
        job = Path(os.environ['H3_BENCH_JOB'])
        rank = int(os.environ.get('RANK', 0))
        self.trace_job, self.trace_rank = job, rank
        fields = ('infer_steps', 'fps', 'parallel', 'tensor_parallel', 'cpu_offload',
                  'offload_granularity', 'cpu_offload_layout', 'feature_caching',
                  'model_variant', 'h3_step_update', 'lora_configs', 'warmup',
                  'use_compile', 'attn_type', 'rms_type', 'rope_type', 'use_adaln_cache',
                  'h3_cache_start_step', 'enable_cfg', 'adaln_cache_dir', 'h3_sampling_stage')
        effective = {key: config.get(key) for key in fields}
        (job / f'effective-config-rank{rank}.json').write_text(json.dumps(effective, indent=2))
        logger.info('H3_DEPLOYMENT_CONFIG rank={} {}', rank, json.dumps(effective))
        super().__init__(config)

    def load_model(self):
        self.model = None
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()

    def init_modules(self):
        DefaultRunner.init_modules(self)
        self.run_input_encoder = self._run_input_encoder_local_h3

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        try:
            return super().run_text_encoder(input_info, keyframes, references)
        finally:
            self.text_encoders[0].unload()
            release_allocator_caches()
            assert self.text_encoders[0].text_encoder is None
            assert self.text_encoders[0].vision_encoder is None
            logger.info(
                "ENCODER_RELEASED text=None vision=None cuda_allocated_GiB={:.3f}",
                torch.cuda.memory_allocated() / 2**30,
            )

    def init_run(self):
        if self.model is None:
            self.video_vae = None
            self.audio_vae = None
            release_allocator_caches()
            logger.info("VAE_RELEASED before transformer loading; reload from SSD for decode")
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        result = super().init_run()
        scheduler = self.scheduler
        end = scheduler.draft_steps or scheduler.infer_steps
        trace = dict(rank=self.trace_rank, start_step=scheduler.start_step_index,
                     schedule_steps=scheduler.infer_steps,
                     evaluations=end - scheduler.start_step_index,
                     draft_steps=scheduler.draft_steps,
                     source=bool(getattr(self.input_info, 'resample_source_video', None)),
                     compile=self.config.get('use_compile'),
                     feature_caching=self.config.get('feature_caching'))
        (self.trace_job / f'sampling-rank{self.trace_rank}.json').write_text(json.dumps(trace, indent=2))
        return result

    def release_dit(self):
        self.model = None
        release_allocator_caches()

    def run_vae_decoder(self, video_rows, audio_rows):
        self.release_dit()
        if self.video_vae is None:
            self.video_vae, self.audio_vae = self.load_vae()
        return super().run_vae_decoder(video_rows, audio_rows)

    def run_main(self):
        try:
            return super().run_main()
        finally:
            self.release_dit()
