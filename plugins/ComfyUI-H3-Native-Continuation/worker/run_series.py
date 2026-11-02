"""Single ComfyUI job: exactly two complete native segments, one publication."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

PLUGIN = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('H3_ROOT', '/opt/h3/runtime'))
sys.path.insert(0, str(PLUGIN))
sys.path.insert(0, str(ROOT / 'production'))
from continuation_core import plan, read_tail
from media import merge, executable
from receipt_contract import validate_guide_receipts


def execute(job, runtime, phase_config, verify_draft, publish):
    config = json.loads((job/'config.json').read_text())
    requests = json.loads((job/'requests.json').read_text())
    planned = plan(config['segment_seconds'])
    if len(requests) != 2 or not config.get('h3_two_pass') or config.get('h3_draft_steps') != 10 or config.get('h3_resample_start') != 3:
        raise ValueError('Continuation requires exactly two complete 10+5 segments')
    if any(r['num_frames'] != planned['frames_per_segment'] for r in requests):
        raise ValueError('Unexpected native frame count')
    summary = dict(returncode=1, plan=planned, passes=[], guide_receipts=[], final_latent_source_only=True)
    started = time.monotonic()
    map_path = job/'continuation-map.json'
    map_path.write_text('{}')
    os.environ['H3_CONTINUATION_MAP'] = str(map_path)
    runtime.BENCH_ROOT = PLUGIN/'worker'
    continuation_source = None
    try:
        for number, overrides in enumerate(requests, 1):
            folder = job/f'segment{number}'
            folder.mkdir()
            request = json.loads(runtime.BASE_REQUEST.read_text())
            request.update(overrides)
            for stage in ('draft', 'final'):
                target = folder/'draft' if stage == 'draft' else folder
                cfg = phase_config(config, stage)
                effective = dict(request)
                effective.pop('draft_steps', None)
                if stage == 'final':
                    effective.update(resample_source_video=str(folder/'draft/generated.mp4'),
                                     resample_start_step=config['h3_resample_start'])
                runtime.run_roll_phase(target, cfg, effective, f'segment{number}-{stage}', summary['passes'])
                if number == 2:
                    checked = validate_guide_receipts(target, continuation_source, cfg['size'])
                    summary['guide_receipts'].append(dict(stage=stage, ranks=checked))
                verify_draft(target/'generated.mp4', request['num_frames'], cfg['size'], executable('ffprobe'))
            tail = json.loads((folder/'continuation-tail.json').read_text())
            read_tail(tail['path'], tail['sha256'], config['size'][0], config['size'][1])
            if number == 1:
                continuation_source = tail
                mapping = {str((job/'segment2'/s).resolve()): tail for s in ('draft', '.')}
                map_path.write_text(json.dumps(mapping, indent=2))
        summary['media'] = merge(job/'segment1/generated.mp4', job/'segment2/generated.mp4', job/'generated.mp4')
        summary['published_video'] = publish(job/'generated.mp4', 'ComfyUI native continuation 2 segments')
        summary['returncode'] = 0
        return summary
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary['wall_seconds'] = time.monotonic()-started
        (job/'series-result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    import job_runner
    from two_pass import phase_config, verify_draft
    from native_delivery import publish
    job = Path(sys.argv[1]).resolve(strict=True)
    inherited = os.environ.get('H3_RUN_LOCK_FD')
    if inherited is not None:
        try:
            fd = int(inherited)
            if fd < 0:
                raise ValueError('negative descriptor')
            held, expected = os.fstat(fd), job_runner.LOCK.stat()
            if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError('descriptor does not refer to the native runtime lock')
        except (OSError, ValueError) as exc:
            raise RuntimeError('Invalid inherited H3 runtime lock descriptor') from exc
        # The supervisor retains this same open lock until every child is gone.
        # Do not close or unlock the inherited descriptor here.
        execute(job, job_runner, phase_config, verify_draft, publish)
        return
    with job_runner.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(job, job_runner, phase_config, verify_draft, publish)


if __name__ == '__main__':
    main()
