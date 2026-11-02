"""Isolated complete generation or matched RGB comparison with safe resumption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from latent_contract import sha
from adapter_safety import (ROOT,runtime,read,atomic_json,clean_request,bind_job,
                            run_phase,verify_clean,upscale_ready,check_sources,run_locked)
HERE=Path(__file__).resolve().parent
job_runner=runtime()
from two_pass import phase_config


def execute(job,compare_rgb=False,publish=False):
    config=read(job/'config.json')
    request=read(job_runner.BASE_REQUEST);request.update(read(job/'request-overrides.json'))
    request=clean_request(request)
    low,high=config['h3_draft_size'],config['size']
    if high!=[n*2 for n in low] or any(n%32 for n in low):
        raise ValueError('Exact 2x, patch-aligned draft/final geometry required')
    if config['h3_draft_steps']!=10 or config['h3_resample_start']!=3:
        raise ValueError('Preserve native 10+5')
    binding=bind_job(job,config,[request],job_runner,HERE)
    job_runner.BENCH_ROOT=HERE
    summary=dict(returncode=1,passes=[],root=str(ROOT),input_binding_sha256=binding,
                 experiment='matched-rgb-vs-learned' if compare_rgb else 'learned-complete',
                 final_size=high,draft_size=low,tp2_core_modified=False)
    start=time.monotonic()
    try:
        os.environ.pop('H3_CONTINUATION_MAP',None)
        os.environ.pop('H3_LEARNED_SOURCE',None)
        draft=job/'draft'
        run_phase(job_runner,draft,phase_config(config,'draft'),request,'draft-10',summary['passes'])
        summary['draft_export']=verify_clean(draft)
        output=job/'learned.safetensors'
        if not upscale_ready(output,draft/'clean.safetensors'):
            subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'upscale.py'),
                            str(draft/'clean.safetensors'),str(output)],check=True)
        if not upscale_ready(output,draft/'clean.safetensors'):raise ValueError('Upscale did not commit')
        summary['upscale']=read(output.with_suffix('.run.json'))
        spec=job/'learned-source.json';atomic_json(spec,dict(path=str(output),sha256=sha(output)))
        second=dict(request,resample_source_video=str(draft/'generated.mp4'),resample_start_step=3)
        modes=('rgb','learned') if compare_rgb else ('learned',)
        receipts=[]
        for mode in modes:
            folder=job/mode
            if mode=='learned':os.environ['H3_LEARNED_SOURCE']=str(spec)
            else:os.environ.pop('H3_LEARNED_SOURCE',None)
            dependencies=dict(draft_sha256=sha(draft/'generated.mp4'))
            if mode=='learned':dependencies['learned_sha256']=sha(output)
            run_phase(job_runner,folder,phase_config(config,'final'),second,mode+'-final-5',
                      summary['passes'],dependencies)
            if mode=='learned':receipts.extend(check_sources(folder,output))
            else:receipts.extend(read(folder/f'rgb-source-rank{i}.json') for i in (0,1))
        if len({x['audio_sha256'] for x in receipts})!=1:raise ValueError('Audio sources differ')
        summary.update(source_receipts=receipts,rgb_video=str(job/'rgb/generated.mp4') if compare_rgb else None,
                       learned_video=str(job/'learned/generated.mp4'))
        if publish:
            from native_delivery import publish as publish_video
            summary['published_video']=publish_video(job/'learned/generated.mp4','H3 learned latent 2x 10+5')
        summary['returncode']=0
    except BaseException as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary['wall_seconds']=time.monotonic()-start
        atomic_json(job/'experiment-result.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path)
    p.add_argument('--compare-rgb',action='store_true');p.add_argument('--publish',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();flags=[]
    if args.compare_rgb:flags.append('--compare-rgb')
    if args.publish:flags.append('--publish')
    return run_locked(args.job.resolve(),job_runner,
        lambda job:execute(job,args.compare_rgb,args.publish),Path(__file__).resolve(),args.worker,flags)

if __name__=='__main__':raise SystemExit(main())
