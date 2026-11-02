"""Package-owned learned handoff for the single-segment entry.

Patches only the worker's imported job_runner object; native files stay intact.
"""
import importlib
import os
from pathlib import Path
import subprocess
import sys
from learned_bridge import ADAPTER, draft_geometry
from package_binding import bind

def dependencies():
    here=ADAPTER.resolve(strict=True)
    sys.path.insert(0,str(here))
    module=importlib.import_module('adapter_safety')
    if Path(module.__file__).resolve()!=here/'adapter_safety.py':
        raise RuntimeError('Unexpected adapter_safety module origin')
    import latent_contract
    if Path(latent_contract.__file__).resolve()!=here/'latent_contract.py':
        raise RuntimeError('Unexpected latent_contract module origin')
    return module


def run_two_pass(job, config, request, passes, stage, approved_hash, summary):
    import job_runner
    import draft_review
    from two_pass import phase_config
    if (config.get('h3_handoff')!='learned-clean-latent-2x' or
        config.get('h3_draft_steps')!=10 or (config.get('infer_steps',8),config.get('h3_resample_start')) not in ((8,3),(8,4),(4,1)) or not config.get('h3_two_pass')):
        raise ValueError('Requires learned handoff, native10 and a supported Turbo suffix (8/3, 8/4 or 4/1)')
    summary['node_package_binding']=bind(job)
    safe = dependencies()
    from latent_contract import sha
    job_runner.BENCH_ROOT = ADAPTER
    if config['h3_draft_size'] != draft_geometry(config['size'][1],config['size'][0]):
        raise ValueError('Exact learned 2x geometry required')
    request = safe.clean_request(request)
    safe.bind_job(job, config, [request], job_runner, ADAPTER)
    os.environ.pop('H3_LEARNED_SOURCE', None)
    # Ordinary complete jobs have no continuation guide. Dedicated continuation
    # workers supply their own source map and validate both ranks.
    os.environ.pop('H3_CONTINUATION_MAP', None)
    draft = job/'draft'
    if stage in ('draft', 'complete'):
        safe.run_phase(job_runner, draft, phase_config(config, 'draft'), request, 'draft', passes)
    else:
        summary['review'] = draft_review.require_approval(job, config, request, approved_hash)
    safe.verify_clean(draft)
    if stage == 'draft':
        receipt = draft_review.ready(job, config, request)
        summary.update(status='awaiting_user_approval', published_draft=receipt['published_draft'])
        return 0
    enlarged = job/'learned.safetensors'
    if not safe.upscale_ready(enlarged, draft/'clean.safetensors'):
        subprocess.run([str(job_runner.VENV/'bin/python'), str(ADAPTER/'upscale.py'),
                        str(draft/'clean.safetensors'), str(enlarged)], check=True,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTORCH_ALLOC_CONF='expandable_segments:True'))
    if not safe.upscale_ready(enlarged, draft/'clean.safetensors'):
        raise ValueError('Learned latent handoff did not commit')
    summary['upscale'] = safe.read(enlarged.with_suffix('.run.json'))
    spec = job/'learned-source.json'
    safe.atomic_json(spec, dict(path=str(enlarged), sha256=sha(enlarged)))
    os.environ['H3_LEARNED_SOURCE'] = str(spec)
    final = dict(request, resample_source_video=str(draft/'generated.mp4'), resample_start_step=config['h3_resample_start'])
    safe.run_phase(job_runner, job, phase_config(config, 'final'), final, 'final-suffix', passes,
                           dict(draft_sha256=sha(draft/'generated.mp4'), learned_sha256=sha(enlarged)))
    summary['source_receipts'] = safe.check_sources(job, enlarged)
    summary['handoff'] = 'learned-clean-latent-2x'
    return 0
