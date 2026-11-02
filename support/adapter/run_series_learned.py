"""Two complete TP2 segments, committed learned handoffs and native tail guides."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from latent_contract import sha
from adapter_safety import (ROOT,PLUGIN,runtime,read,atomic_json,clean_request,bind_job,
                            run_phase,verify_clean,upscale_ready,check_sources,run_locked)
HERE=Path(__file__).resolve().parent
job_runner=runtime()
sys.path.insert(0,str(PLUGIN))
from two_pass import phase_config
from continuation_core import plan,read_tail
from receipt_contract import validate_guide_receipts
from duration_plan import validate_plan


def execute(job):
    config=read(job/'config.json');overrides=read(job/'requests.json')
    requests=[]
    for override in overrides:
        request=read(job_runner.BASE_REQUEST);request.update(override)
        requests.append(clean_request(request))
    low,high=config['h3_draft_size'],config['size']
    if config.get('h3_duration_plan') is not None:
        planned=validate_plan(config['h3_duration_plan'],segments=2)
        frame_counts=planned['segment_frames']
    else:
        planned=plan(config['segment_seconds'])
        frame_counts=[planned['frames_per_segment']]*2
    if len(requests)!=2 or high!=[n*2 for n in low] or any(n%32 for n in low):
        raise ValueError('Two segments and native exact 2x geometry required')
    if config['h3_draft_steps']!=10 or (config.get('infer_steps',8),config['h3_resample_start']) not in ((8,3),(8,4),(4,1)):
        raise ValueError('Requires native10 and a supported Turbo suffix (8/3, 8/4 or 4/1)')
    if any(r['num_frames']!=frames for r,frames in zip(requests,frame_counts)):
        raise ValueError('Native duration mismatch')
    binding=bind_job(job,config,requests,job_runner,HERE)
    job_runner.BENCH_ROOT=HERE
    mapping=job/'continuation-map.json';atomic_json(mapping,{})
    os.environ['H3_CONTINUATION_MAP']=str(mapping)
    summary=dict(returncode=1,root=str(ROOT),input_binding_sha256=binding,plan=planned,passes=[],
                 upscales=[],guide_receipts=[],handoff='learned-clean-latent-2x',size=high,
                 exact_30_seconds=bool(config.get('deliver_exact_30_seconds')))
    started=time.monotonic();tail=None
    try:
        for number,request in enumerate(requests,1):
            segment_config=dict(config,num_frames=request['num_frames'])
            if config.get('h3_duration_plan') is not None:
                segment_config['segment_seconds']=planned['segment_seconds'][number-1]
            folder=job/f'segment{number}';folder.mkdir(exist_ok=True);draft=folder/'draft'
            guide_dependency=dict(continuation_sha256=tail['sha256']) if tail else {}
            os.environ.pop('H3_LEARNED_SOURCE',None)
            run_phase(job_runner,draft,phase_config(segment_config,'draft'),request,f'segment{number}-draft',
                      summary['passes'],guide_dependency)
            verify_clean(draft)
            if number==2:
                summary['guide_receipts'].append(dict(segment=2,stage='draft',
                    ranks=validate_guide_receipts(draft,tail,low)))
            enlarged=folder/'learned.safetensors'
            if not upscale_ready(enlarged,draft/'clean.safetensors'):
                subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'upscale.py'),
                                str(draft/'clean.safetensors'),str(enlarged)],check=True)
            if not upscale_ready(enlarged,draft/'clean.safetensors'):raise ValueError('Upscale did not commit')
            summary['upscales'].append(read(enlarged.with_suffix('.run.json')))
            spec=folder/'learned-source.json';atomic_json(spec,dict(path=str(enlarged),sha256=sha(enlarged)))
            os.environ['H3_LEARNED_SOURCE']=str(spec)
            final=dict(request,resample_source_video=str(draft/'generated.mp4'),resample_start_step=config['h3_resample_start'])
            dependencies=dict(guide_dependency,draft_sha256=sha(draft/'generated.mp4'),learned_sha256=sha(enlarged))
            run_phase(job_runner,folder,phase_config(segment_config,'final'),final,f'segment{number}-final',
                      summary['passes'],dependencies)
            check_sources(folder,enlarged)
            if number==2:
                summary['guide_receipts'].append(dict(segment=2,stage='final',
                    ranks=validate_guide_receipts(folder,tail,high)))
            exported=read(folder/'continuation-tail.json')
            read_tail(exported['path'],exported['sha256'],*high)
            if number==1:
                tail=exported
                atomic_json(mapping,{str((job/'segment2'/p).resolve()):tail for p in ('draft','.')})
        if config.get('deliver_exact_30_seconds'):
            from merge_30 import merge
        else:
            from media import merge
        summary['media']=merge(job/'segment1/generated.mp4',job/'segment2/generated.mp4',job/'generated.mp4')
        from native_delivery import publish
        summary['published_video']=publish(job/'generated.mp4','H3 learned latent 2x continuation, two native segments')
        summary['returncode']=0
    except BaseException as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary['wall_seconds']=time.monotonic()-started
        atomic_json(job/'series-result.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    return run_locked(args.job.resolve(),job_runner,execute,Path(__file__).resolve(),args.worker)

if __name__=='__main__':raise SystemExit(main())
