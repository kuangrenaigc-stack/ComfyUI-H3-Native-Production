"""Retain both locks while the native complete worker/ranks are cleaned up."""
import fcntl
import os
from pathlib import Path
import sys

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'continuation'))
from process_control import supervise

def main():
    root=Path(os.environ['H3_ROOT'])
    job=Path(sys.argv[1]).resolve(strict=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,H3_RUN_LOCK_FD=str(lock.fileno()))
        env.pop('H3_CONTROL_LOCK_FD',None)
        return supervise([sys.executable,'-u',str(PACKAGE/'worker/run_job.py'),str(job),'--stage','complete'],
                         job,env=env,pass_fds=(lock.fileno(),))

if __name__=='__main__':sys.exit(main())
