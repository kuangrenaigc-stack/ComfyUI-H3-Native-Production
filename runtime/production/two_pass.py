"""H3 Roll orchestration. Native full-schedule first pass followed by a Turbo suffix."""
import json
import math
from pathlib import Path
import subprocess


def turbo_schedule_steps(config):
    steps = config.get('infer_steps', 8)
    if type(steps) is not int or steps not in (4, 8):
        raise ValueError('Turbo schedule must have 4 or 8 steps')
    return steps


def resample_start(config):
    steps = turbo_schedule_steps(config)
    start = config.get('h3_resample_start', 1 if steps == 4 else 3)
    if type(start) is not int or not 0 <= start < steps:
        raise ValueError(f'resample_start_step must be an integer from 0 to {steps - 1}')
    return start


def geometry(width, height):
    scale = min(1.0, 480 / min(width, height))
    return [max(32, int(height * scale) // 32 * 32),
            max(32, int(width * scale) // 32 * 32)]


def apply_plan(payload, config):
    enabled = payload.get('two_pass', False)
    if type(enabled) is not bool:
        raise ValueError('two_pass must be boolean')
    if not enabled:
        return
    draft = payload.get('draft_steps', config.get('h3_native_draft_steps', 10))
    steps = turbo_schedule_steps(config)
    start = payload.get('resample_start_step', resample_start(config))
    if type(draft) is not int or not 1 <= draft <= 29:
        raise ValueError('draft_steps must be an integer from 1 to 29')
    if type(start) is not int or not 0 <= start < steps:
        raise ValueError(f'resample_start_step must be an integer from 0 to {steps - 1}')
    size = geometry(config['size'][1], config['size'][0])
    if 'draft_width' in payload or 'draft_height' in payload:
        size = [payload.get('draft_height'), payload.get('draft_width')]
        if any(type(x) is not int or x < 32 or x % 32 for x in size):
            raise ValueError('draft dimensions must be positive multiples of 32')
        if any(x > y for x, y in zip(size, config['size'])):
            raise ValueError('draft dimensions must not exceed the final canvas')
    config.update(h3_two_pass=True, h3_draft_steps=draft,
                  h3_draft_size=size, h3_resample_start=start)


def sampling_plan(config):
    steps = turbo_schedule_steps(config)
    if not config.get('h3_two_pass'):
        return {'two_pass': False, 'steps': steps,
                'acceleration': config['h3_acceleration_mode']}
    k, start = config['h3_draft_steps'], resample_start(config)
    return {'two_pass': True, 'draft_size': config['h3_draft_size'],
            'final_size': config['size'], 'draft_steps': k,
            'resample_start_step': start, 'final_steps': steps - start,
            'draft_acceleration': 'standard',
            'draft_schedule_steps': k, 'draft_lora': False, 'draft_cfg': False,
            'draft_sampling': 'full-native-schedule',
            'final_schedule_steps': steps, 'final_lora': bool(config.get('lora_configs')),
            'final_cfg': False,
            'final_acceleration': phase_config(config, 'final')['h3_acceleration_mode'],
            'dit_offload_granularity': config.get('offload_granularity'),
            'compile': config.get('use_compile', False),
            'warmup': False, 'handoff': 'RGB-Lanczos-VAE',
            'adaln_cache_scope': 'phase-and-schedule-specific'}


def phase_config(config, phase):
    result = {k: v for k, v in config.items() if not k.startswith('h3_') or k in
              ('h3_step_update', 'h3_acceleration_mode', 'h3_resident_blocks', 'h3_rms_backend')}
    result['warmup'] = False
    if phase == 'draft':
        result.update(size=config['h3_draft_size'], infer_steps=config['h3_draft_steps'],
                      lora_configs=[], lora_dynamic_apply=False, enable_cfg=False,
                      video_flow_shift=12.0, audio_flow_shift=3.0,
                      h3_step_update='reference_blend', h3_sampling_stage='native-draft',
                      h3_cache_start_step=0, h3_acceleration_mode='standard')
    elif phase == 'final':
        result.update(infer_steps=turbo_schedule_steps(config), enable_cfg=False, h3_sampling_stage='turbo-final',
                      h3_cache_start_step=resample_start(config))
    else:
        raise ValueError('Unknown sampling phase')
    return result


def verify_draft(path, frames, size, ffprobe):
    probe = json.loads(subprocess.check_output(
        [ffprobe, '-v', 'error', '-show_streams', '-of', 'json', str(path)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    num, den = map(int, video['r_frame_rate'].split('/'))
    if int(video['nb_frames']) != frames or num / den != 24:
        raise ValueError('Draft frame count / FPS differs from the final request')
    if [video['height'], video['width']] != size:
        raise ValueError('Draft dimensions differ from the sampling plan')
    duration = float(audio.get('duration', 0))
    if not math.isfinite(duration) or duration < frames / 24 - 1 / 24:
        raise ValueError('Draft audio is missing or too short')
    if not Path(path).is_file():
        raise ValueError('Draft output is missing')
