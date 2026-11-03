"""Resample an existing learned handoff in a NEW research job, without publishing.

Use run_locked.py. Keeps the source job immutable and its original sampler/media.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','adapter-dir','source-job','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Smoke output must be a new directory')
    config=json.loads((a.source_job/'phase-config.json').read_text())
    request=json.loads((a.source_job/'effective-request.json').read_text())
    if (config.get('infer_steps'),config.get('h3_cache_start_step'),config.get('h3_sampling_stage'))!=(8,3,'turbo-final'):
        raise ValueError('Smoke requires the production 8-step schedule / five-evaluation suffix')
    source=a.source_job/'learned-source.json';spec=json.loads(source.read_text())
    if sha(spec['path'])!=spec['sha256']:raise ValueError('Learned source checksum mismatch')
    sys.path.insert(0,str(a.runtime_root/'production'))
    import job_runner
    if job_runner.ROOT.resolve()!=a.runtime_root.resolve():raise ValueError('Runtime binding differs')
    job_runner.BENCH_ROOT=a.adapter_dir
    os.environ['H3_LEARNED_SOURCE']=str(source)
    a.output.mkdir(parents=True)
    result=job_runner.run_pass(a.output,config,request,'ffn-chunk-resample-smoke')
    if result['returncode'] or not result['output_exists']:raise RuntimeError('FFN full resample failed')
    output=a.output/'generated.mp4'
    probe=json.loads(subprocess.check_output([shutil.which('ffprobe') or '/usr/bin/ffprobe','-v','error','-show_streams','-of','json',str(output)],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    if [video['height'],video['width']]!=config['size'] or int(video['nb_frames'])!=request['num_frames'] or video['r_frame_rate']!='24/1':
        raise ValueError('Native geometry or frame count changed')
    if not any(s['codec_type']=='audio' for s in probe['streams']):raise ValueError('Audio missing')
    result.update(sha256=sha(output),video=video,ffn=[json.loads((a.output/f'ffn-memory-rank{r}.json').read_text()) for r in (0,1)],published=False)
    (a.output/'smoke-result.json').write_text(json.dumps(result,indent=2))
    print('H3_FFN_SMOKE_COMPLETE',flush=True)

if __name__=='__main__':main()
