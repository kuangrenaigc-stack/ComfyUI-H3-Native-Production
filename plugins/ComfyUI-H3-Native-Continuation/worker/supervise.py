"""Keep the runtime lock and own every rank until normal exit or cancellation."""
import fcntl
import os
from pathlib import Path
import sys

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))
from process_control import supervise


def main():
    root = Path(os.environ.get('H3_ROOT', '/opt/h3/runtime'))
    job = Path(sys.argv[1]).resolve(strict=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, H3_RUN_LOCK_FD=str(lock.fileno()))
        env.pop('H3_CONTROL_LOCK_FD', None)
        code = supervise([sys.executable, '-u', str(PLUGIN/'worker/run_series.py'), str(job)],
                         job, env=env, pass_fds=(lock.fileno(),))
    return code


if __name__ == '__main__':
    sys.exit(main())
