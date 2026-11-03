"""Run one bounded diagnostic under the same production locks; never stop other work."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--control-dir',type=Path,required=True,help='Use the control directory of the installed production node')
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.timeout<1:p.error('command and positive timeout required')
    # Do not create a substitute production lock at a misspelled path.
    if not (args.runtime_root/'run.lock').is_file() or not (args.control_dir/'control.lock').is_file():
        raise ValueError('Production runtime and control locks must already exist')
    with (args.control_dir/'control.lock').open('a') as control,(args.runtime_root/'run.lock').open('a') as runtime:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(runtime,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        if busy:raise RuntimeError('GPU is busy; no existing process was stopped')
        args.log.parent.mkdir(parents=True,exist_ok=True)
        with args.log.open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:return process.wait(timeout=args.timeout)
            finally:
                # Include descendants even if the launcher has already exited.
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass

if __name__=='__main__':raise SystemExit(main())
