"""Fixed TP2 deployment policy; optional cache/compile adapters removed."""
MODES = ('standard',)
DEFAULT_MODE = 'standard'


def validate_sampling_config(config):
    if config.get('parallel') != {'tensor_p_size': 2}:
        raise ValueError('This deployment requires TP2')
    steps = config.get('infer_steps')
    stage = config.get('h3_sampling_stage', 'turbo')
    if stage == 'native-draft':
        if type(steps) is not int or not 1 <= steps <= 29:
            raise ValueError('Native first-pass steps must be an integer from 1 to 29')
        if config.get('lora_configs') or config.get('lora_dynamic_apply'):
            raise ValueError('Native first pass must not load a LoRA')
        if config.get('h3_step_update') != 'reference_blend':
            raise ValueError('Native first pass uses the official reference_blend update')
    elif stage in ('turbo', 'turbo-final'):
        if type(steps) is not int or steps not in (4, 8):
            raise ValueError('Turbo pass requires a supported 4-step or 8-step LoRA schedule')
    else:
        raise ValueError('Unknown H3 sampling stage')
    if config.get('enable_cfg', False):
        raise ValueError('MiniMax-H3 is guidance-distilled; official CFG is disabled')


def configure(config, mode, root):
    if mode not in MODES:
        raise ValueError('DPCache is disabled by user request; use standard')
    validate_sampling_config(config)
    config.update(h3_acceleration_mode='standard', feature_caching='NoCaching',
                  use_compile=False, vae_use_compile=False, cpu_offload=True,
                  offload_granularity='block', cpu_offload_layout='per_tensor',
                  h3_resident_blocks=0, dit_prepost_resident=False, warmup=False)
    for key in list(config):
        if key.startswith('dpcache_') or key in ('h3_cache_device', 'offload_resident_blocks'):
            config.pop(key)
    return config


