"""Runtime identity, committed artifacts and supervised execution for this adapter."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from latent_contract import sha, load, tensor_sha

ROOT=Path('/opt/h3/runtime')
PLUGIN=Path('/opt/h3/comfyui/custom_nodes/ComfyUI-H3-Native-Continuation')

def runtime():
    configured=os.environ.get('H3_ROOT')
    if configured and Path(configured).resolve()!=ROOT.resolve():
        raise RuntimeError('H3_ROOT points outside this adapter runtime')
    os.environ['H3_ROOT']=str(ROOT)
    sys.path.insert(0,str(ROOT/'production'))
    module=importlib.import_module('job_runner')
    if module.ROOT.resolve()!=ROOT.resolve() or module.LOCK.resolve()!=(ROOT/'run.lock').resolve():
        raise RuntimeError('Imported job_runner uses another runtime or lock')
    return module

def atomic_json(path,data):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,indent=2)
            f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)

def read(path):return json.loads(Path(path).read_text())

def clean_request(request):
    request=dict(request);request.pop('draft_steps',None)
    return request

def bind_job(job,config,requests,runner,here):
    # A binding is immutable; interrupted old jobs without one require a new directory.
    assets={}
    for request in requests:
        for key in ('image_path','video_path','audio_path'):
            for name in str(request.get(key,'')).split(','):
                if name:assets[str(Path(name).resolve(strict=True))]=sha(Path(name))
    files=[p for p in Path(here).glob('*.py') if not p.name.startswith('test')]
    files += [ROOT/'UPSTREAM_REVISION.txt',ROOT/'production/entry.py',
              ROOT/'production/job_runner.py',ROOT/'production/two_pass.py',
              ROOT/'production/acceleration.py']
    files += [PLUGIN/'worker/guide_hook.py',PLUGIN/'continuation_core.py',
              PLUGIN/'receipt_contract.py',PLUGIN/'process_control.py',PLUGIN/'media.py',
              Path(here).parent/'architecture/ComfyUI-H3-Latent-Upscaler-Mamad8-main/upscaler.py']
    binding=dict(root=str(ROOT),config=config,requests=requests,assets=assets,
                 code={str(p):sha(p) for p in files})
    path=job/'input-binding.json'
    if path.exists():
        if read(path)!=binding:raise ValueError('Job inputs/code changed; use a new job directory')
    else:
        if any(job.rglob('generated.mp4')) or any(job.rglob('clean.safetensors')) or any(job.rglob('learned.safetensors')):
            raise ValueError('Existing job has no input binding; use a new job directory')
        atomic_json(path,binding)
    return sha(path)

def run_phase(runner,folder,config,request,label,passes,dependencies=None):
    from acceleration import configure
    from two_pass import verify_draft
    config=dict(config)
    if config.get('h3_sampling_stage')=='native-draft':
        config.pop('ffn_inplace_activation',None)  # Memory change is final-stage only.
    config=configure(config,config['h3_acceleration_mode'],ROOT)
    request=clean_request(request)
    expected=dict(config=config,request=dict(request,save_result_path=str(folder/'generated.mp4')),
                  dependencies=dependencies or {})
    folder.mkdir(exist_ok=True)
    marker=folder/'phase-complete.json'
    if marker.exists():
        receipt=read(marker)
        if receipt.get('binding')!=expected:
            raise ValueError('Completed phase inputs or dependencies changed: '+label)
        if read(folder/'phase-config.json')!=config or read(folder/'effective-request.json')!=expected['request']:
            raise ValueError('Stored phase configuration/request differs: '+label)
        if not (folder/'generated.mp4').is_file() or sha(folder/'generated.mp4')!=receipt['output_sha256']:
            raise ValueError('Completed phase output was changed: '+label)
        passes.append(dict(label=label,reused=True,returncode=0,wall_seconds=0,output_exists=True))
        return True
    # Incomplete output is never a cache hit. Preserve it before regenerating this phase.
    output=folder/'generated.mp4'
    if output.exists():
        index=1
        while (folder/f'incomplete-{index}.mp4').exists():index+=1
        output.rename(folder/f'incomplete-{index}.mp4')
    runner.run_roll_phase(folder,config,request,label,passes)
    verify_draft(output,request['num_frames'],config['size'],
                 '/usr/bin/ffprobe')
    atomic_json(marker,dict(binding=expected,output_sha256=sha(output)))
    return False

def verify_clean(folder):
    exports=[read(folder/f'clean-export-rank{i}.json') for i in (0,1)]
    for key in ('shape','video_sha256','audio_sha256','steps','sigma_end'):
        if exports[0][key]!=exports[1][key]:raise ValueError('TP2 clean export mismatch: '+key)
    receipt=read(folder/'clean.json')
    data,meta=load(folder/'clean.safetensors',receipt['sha256'])
    if (receipt['video_sha256']!=exports[0]['video_sha256'] or
        tensor_sha(data['video'])!=exports[0]['video_sha256'] or
        tensor_sha(data['audio'])!=exports[0]['audio_sha256'] or
        list(data['video'].shape)!=exports[0]['shape']):
        raise ValueError('Clean artifact does not match the TP2 exports')
    return exports[0]

def upscale_ready(output,source):
    marker=output.with_suffix('.run.json')
    if not output.exists() or not marker.exists() or not output.with_suffix('.json').exists():
        return False
    report=read(marker)
    if report['sha256']!=sha(output) or report['metadata']['source_sha256']!=sha(source):
        raise ValueError('Learned artifact output/source binding changed')
    data,meta=load(output,report['sha256'])
    if meta!=report['metadata'] or tensor_sha(data['video'])!=report['video_sha256']:
        raise ValueError('Learned artifact contract differs from its commit marker')
    return True

def check_sources(folder,source):
    expected=sha(source)
    receipts=[read(folder/f'learned-source-rank{i}.json') for i in (0,1)]
    for r in receipts:
        if r['sha256']!=expected or Path(r['path']).resolve()!=source.resolve():
            raise ValueError('Final phase consumed a different learned source')
    for key in ('video_sha256','audio_sha256','sha256'):
        if receipts[0][key]!=receipts[1][key]:raise ValueError('TP2 learned source mismatch')
    return receipts

def run_locked(job,runner,execute,script,worker=False,flags=(),control_path=None,grace_seconds=45):
    control=control_path or Path(f'/run/user/{os.getuid()}/h3-native/control.lock')
    if worker:
        for name,path in [('H3_ADAPTER_CONTROL_FD',control),('H3_ADAPTER_RUNTIME_FD',runner.LOCK)]:
            try:
                fd=int(os.environ[name]);actual=os.fstat(fd);expected=path.stat()
                if fd<0 or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise ValueError()
            except (KeyError,OSError,ValueError) as exc:
                raise RuntimeError('Missing/invalid inherited adapter lock') from exc
        execute(job)
        return 0
    control.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(PLUGIN))
    from process_control import supervise
    with control.open('a') as guard,runner.LOCK.open('a') as lock:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if busy:raise RuntimeError('GPU busy; unrelated tasks left untouched')
        env=dict(os.environ,H3_ROOT=str(ROOT),H3_ADAPTER_CONTROL_FD=str(guard.fileno()),
                 H3_ADAPTER_RUNTIME_FD=str(lock.fileno()))
        return supervise([sys.executable,str(script),str(job),'--worker',*flags],job,
                         env=env,pass_fds=(guard.fileno(),lock.fileno()),grace_seconds=grace_seconds)
