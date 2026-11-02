"""Linux child-subreaper supervision for this plugin's own processes only."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_table():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail = path.read_text().rpartition(') ')[2].split()
            result[int(path.parent.name)] = dict(ppid=int(tail[1]), start=tail[19], state=tail[0])
        except (OSError, ValueError, IndexError):
            continue  # Process exited while taking this cleanup snapshot.
    return result


def descendants(root):
    table = process_table()
    owned = {root}
    while True:
        expanded = owned | {pid for pid, row in table.items() if row['ppid'] in owned}
        if expanded == owned:
            return {pid: table[pid] for pid in owned if pid != root and pid in table}
        owned = expanded


def _signal_owned(records, sig):
    current = process_table()
    for pid, identity in records.items():
        if current.get(pid, {}).get('start') == identity['start']:
            fd = None
            try:
                if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                    fd = os.pidfd_open(pid)
                    # Recheck after opening: the fd pins this exact process,
                    # so exit/PID reuse cannot redirect the signal to a new one.
                    tail = Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
                    if tail[19] == identity['start']:
                        signal.pidfd_send_signal(fd, sig)
                else:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)


def _reap_adopted(worker):
    # Never reap Popen's direct child behind its back; preserve its return code.
    worker.poll()
    for pid, row in process_table().items():
        if pid != worker.pid and row['ppid'] == os.getpid():
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def _enable_subreaper():
    if sys.platform != 'linux' or not Path('/proc/self/stat').exists():
        raise RuntimeError('H3 continuation process supervision requires Linux /proc')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _children_remain(worker):
    if worker.poll() is None:
        return True
    # Once Popen has reaped its child, use the kernel rather than one /proc
    # snapshot to prove no adopted rank is still alive or becoming reparented.
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        if pid == 0:
            return True


def supervise(command, job, *, env=None, pass_fds=(), grace_seconds=45):
    """Run one worker, adopt orphan ranks and retain locks until cleanup ends.

    TERM/INT requests cooperative shutdown; this supervisor must not itself be
    killed on the UI's 15s timer. Child ranks may use separate sessions. We
    discover/reap only our own descendants, and check PID start times on signals.
    """
    _enable_subreaper()
    job = Path(job)
    cancelled = [False]
    def cancel(signum, frame):
        cancelled[0] = True
    previous = {s: signal.signal(s, cancel) for s in (signal.SIGTERM, signal.SIGINT)}
    worker = None
    result = dict(cleanup_complete=False, cancelled=False, returncode=1)
    try:
        worker = subprocess.Popen(command, env=env, pass_fds=pass_fds, start_new_session=True)
        while worker.poll() is None and not cancelled[0]:
            try:
                worker.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
        code = worker.poll()
        owned = descendants(os.getpid())
        live = {p: r for p, r in owned.items() if r['state'] != 'Z'}
        remaining = _children_remain(worker)
        leaked = not cancelled[0] and (bool(live) or remaining)
        if cancelled[0] or live or remaining:
            # Subreaper keeps ranks rooted here after their original leader
            # exits. Re-discover during shutdown to catch late/reparented forks.
            deadline = time.monotonic()+grace_seconds
            seen, signalled = {}, set()
            while True:
                _reap_adopted(worker)
                owned = descendants(os.getpid())
                live = {p: r for p, r in owned.items() if r['state'] != 'Z'}
                if not live and not _children_remain(worker):
                    break
                seen.update(live)
                if time.monotonic() >= deadline:
                    _signal_owned(live, signal.SIGKILL)
                else:
                    new = {p: r for p, r in live.items() if (p, r['start']) not in signalled}
                    _signal_owned(new, signal.SIGTERM)
                    signalled.update((p, r['start']) for p, r in new.items())
                time.sleep(.05)
            result['terminated_descendants'] = len(seen)
        worker.wait()
        _reap_adopted(worker)
        result.update(cleanup_complete=True, cancelled=cancelled[0],
                      returncode=130 if cancelled[0] else (1 if leaked else worker.returncode))
        return result['returncode']
    finally:
        # If the OS leaves a child in uninterruptible sleep, the cleanup loop
        # keeps this supervisor and both inherited locks alive until it exits.
        # Do not claim success or release exclusion locks while a rank survives.
        if worker is not None:
            result['worker_returncode'] = worker.poll()
        (job/'supervisor-result.json').write_text(json.dumps(result, indent=2))
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_supervisor(process, timeout_seconds=70):
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        # The still-running supervisor retains runtime/control locks. Killing
        # it here would recreate the orphan-rank bug we are preventing.
        raise RuntimeError('接續任務仍在清理子程序；生成鎖保持佔用，請查看本次紀錄。') from exc
