import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from acceleration import configure, DEFAULT_MODE
from two_pass import phase_config, sampling_plan, verify_draft

ROOT = Path(os.environ.get('H3_ROOT', '/opt/h3/runtime'))
BENCH_ROOT = ROOT / 'production'
SOURCE = ROOT / 'source'
VENV = ROOT / '.venv'
MODEL = ROOT / 'model-bf16'
BASE_REQUEST = BENCH_ROOT / 'request-template.json'
LOCK = ROOT / 'run.lock'


def run_pass(pass_job, config, request, label):
    pass_job.mkdir(exist_ok=True)
    # Preserve the immutable workflow plan for later user-approved resumption.
    config_path = pass_job / 'phase-config.json'
    config_path.write_text(json.dumps(config, indent=2))
    effective = dict(request, save_result_path=str(pass_job / 'generated.mp4'))
    (pass_job / 'effective-request.json').write_text(json.dumps(effective, ensure_ascii=False, indent=2))
    env = dict(os.environ, H3_ROOT=str(ROOT), PLATFORM='cuda', DTYPE='BF16',
               CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True',
               PYTHONPATH=os.pathsep.join([str(ROOT/'python-deps'), str(BENCH_ROOT), str(SOURCE)]),
               H3_RMS_BACKEND=config.get('h3_rms_backend', ''), H3_BENCH_JOB=str(pass_job))
    command = [str(VENV / 'bin/torchrun'), '--standalone', '--nproc_per_node=2',
               str(BENCH_ROOT / 'entry.py'), '--model_cls', 'minimax_h3',
               '--model-variant', 'ref2av', '--task', 'ref2av',
               '--model_path', str(MODEL), '--config_json', str(config_path)]
    for key, value in effective.items():
        command.extend(['--'+key, str(value)])
    print('H3_PASS_START '+json.dumps(dict(label=label, mode=config['h3_acceleration_mode'],
          feature_caching=config['feature_caching'], layout=config['offload_granularity'] if config.get('cpu_offload') else 'disabled',
          cpu_offload_layout=config.get('cpu_offload_layout'))), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=SOURCE, env=env)
    data = dict(label=label, returncode=result.returncode, wall_seconds=time.monotonic()-started,
                output_exists=(pass_job/'generated.mp4').is_file())
    print('H3_PASS_COMPLETE '+json.dumps(data), flush=True)
    return data


def run_roll_phase(folder, config, request, label, passes):
    config = configure(config, config['h3_acceleration_mode'], ROOT)
    result = run_pass(folder, config, request, label)
    result.update(acceleration=config['h3_acceleration_mode'],
                  compile=config.get('use_compile'),
                  cache_start_step=config.get('h3_cache_start_step', 0))
    passes.append(result)
    if result['returncode'] or not result['output_exists']:
        raise RuntimeError(f'{label} failed; subsequent phases were not started')
    return result


def run_two_pass(job, config, request, passes, stage, approved_hash, summary):
    import draft_review
    draft = phase_config(config, 'draft')
    first = dict(request)
    # The native first pass runs its complete N-step grid to sigma zero.
    first.pop('draft_steps', None)
    source = job / 'draft' / 'generated.mp4'
    if stage in ('draft', 'complete'):
        if (job/'draft-review.json').exists() or source.exists():
            raise ValueError('Draft already exists; approve it or use a new job for another draft')
        run_roll_phase(job / 'draft', draft, first, 'draft', passes)
    else:
        summary['review'] = draft_review.require_approval(job, config, request, approved_hash)
    verify_draft(source, request['num_frames'], draft['size'],
                 shutil.which('ffprobe') or str(Path.home() / '.local/bin/ffprobe'))
    if stage == 'draft':
        receipt = draft_review.ready(job, config, request)
        summary.update(status='awaiting_user_approval', published_draft=receipt['published_draft'])
        return 0
    final = phase_config(config, 'final')
    second = dict(request, resample_source_video=str(source),
                  resample_start_step=config['h3_resample_start'])
    return run_roll_phase(job, final, second, 'final-suffix', passes)['returncode']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('--stage', choices=('complete', 'draft', 'final'), default='complete')
    parser.add_argument('--approve-draft-sha256')
    args = parser.parse_args()
    if args.stage != 'final' and args.approve_draft_sha256:
        parser.error('Approval is only valid with --stage final')
    job = args.job.resolve()
    config = json.loads((job / 'config.json').read_text())
    mode = config.get('h3_acceleration_mode', DEFAULT_MODE)
    config = configure(config, mode, ROOT)
    if not config.get('h3_two_pass'):
        raise ValueError('Staged production requires the original two-pass workflow config, not a phase config')
    request = json.loads(BASE_REQUEST.read_text())
    overrides = job / 'request-overrides.json'
    if overrides.is_file():
        request.update(json.loads(overrides.read_text()))
    started = time.monotonic()
    passes = []
    summary = dict(returncode=1, source_revision=(ROOT/'UPSTREAM_REVISION.txt').read_text().strip(),
                   acceleration_mode=mode, cpu_offload_layout=config['offload_granularity'] if config.get('cpu_offload') else 'disabled',
                   passes=passes, stage=args.stage)
    with LOCK.open('w') as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if config.get('h3_two_pass'):
                summary['sampling'] = sampling_plan(config)
                summary['returncode'] = run_two_pass(job, config, request, passes,
                    args.stage, args.approve_draft_sha256, summary)
                return
            data = run_pass(job, config, request, mode)
            passes.append(data)
            summary['returncode'] = data['returncode']
        except Exception as exc:
            summary['error'] = str(exc)
            raise
        finally:
            output = job/'draft/generated.mp4' if config.get('h3_two_pass') and args.stage == 'draft' else job/'generated.mp4'
            summary.update(wall_seconds=time.monotonic()-started,
                           output_exists=output.is_file(), output_path=str(output))
            if summary['returncode'] == 0 and summary['output_exists'] and not (config.get('h3_two_pass') and args.stage == 'draft'):
                from native_delivery import publish
                try:
                    summary['published_video'] = publish(job/'generated.mp4', job.parent.name)
                    if config.get('h3_two_pass') and args.stage == 'final':
                        import draft_review
                        receipt = summary['review']
                        receipt.update(status='finalized', published_video=summary['published_video'])
                        draft_review.save(job, receipt)
                except Exception as exc:
                    summary['delivery_error'] = str(exc)
                    (job/'stage-result.json').write_text(json.dumps(summary, indent=2))
                    raise
            (job/'stage-result.json').write_text(json.dumps(summary, indent=2))
    raise SystemExit(summary['returncode'])


if __name__ == '__main__':
    main()
