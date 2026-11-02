"""Stable native two-pass generation with server-only numbered delivery."""
from pathlib import Path
import sys
ROOT=Path('/opt/h3/runtime')
sys.path.insert(0,str(ROOT/'production'))
import job_runner
import os
job_runner.BENCH_ROOT=Path(__file__).resolve().parent
# Package-local dispatch; no production source or runtime default is changed.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'continuation'))
from learned_pipeline import run_two_pass
from inherited_lock import InheritedRuntimeLock
if 'H3_RUN_LOCK_FD' in os.environ:
    job_runner.LOCK=InheritedRuntimeLock(ROOT/'run.lock',os.environ['H3_RUN_LOCK_FD'])
os.environ.update(CUDA_VISIBLE_DEVICES='0,1',PYTORCH_ALLOC_CONF='expandable_segments:True')
job_runner.run_two_pass=run_two_pass
original_plan=job_runner.sampling_plan
def learned_plan(config):
    result=original_plan(config)
    result['handoff']='learned-clean-latent-2x'
    return result
job_runner.sampling_plan=learned_plan
job_runner.main()
