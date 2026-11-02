"""Real Linux process regressions. No model, GPU, ComfyUI or service changes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))


def alive(pid):
    try:
        tail=Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return tail[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_file(path, process):
    deadline=time.monotonic()+5
    while not path.exists():
        if process.poll() is not None:
            raise AssertionError(f'Supervisor stopped early: {process.returncode}')
        if time.monotonic()>deadline:
            raise AssertionError('Test worker did not become ready')
        time.sleep(.02)


@unittest.skipUnless(sys.platform=='linux', 'Linux subreaper regression')
class ProcessControl(unittest.TestCase):
    def launch(self, directory, child_code, grace=.3):
        wrapper='''
import fcntl,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from process_control import supervise
job=Path(sys.argv[2])
with (job/'run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.exit(supervise([sys.executable,'-c',sys.argv[3],str(job)],job,grace_seconds=float(sys.argv[4])))
'''
        return subprocess.Popen([sys.executable,'-c',wrapper,str(PLUGIN),str(directory),child_code,str(grace)],start_new_session=True)

    def test_cancel_cleans_detached_stubborn_rank_and_retains_lock(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            rank_code='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'
            worker='''
import subprocess,sys,time
from pathlib import Path
rank=subprocess.Popen([sys.executable,'-c',%r],start_new_session=True,stdout=subprocess.PIPE,text=True)
rank.stdout.readline()
(Path(sys.argv[1])/'rank.pid').write_text(str(rank.pid))
time.sleep(60)
''' % rank_code
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            proc=self.launch(job,worker,grace=1)
            rank=None
            try:
                wait_file(job/'rank.pid',proc)
                rank=int((job/'rank.pid').read_text())
                proc.send_signal(signal.SIGTERM)
                time.sleep(.1)
                with (job/'run.lock').open('a') as lock:
                    with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(proc.wait(timeout=6),130)
                self.assertFalse(alive(rank))
                self.assertIsNone(unrelated.poll())
                with (job/'run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                receipt=json.loads((job/'supervisor-result.json').read_text())
                self.assertTrue(receipt['cleanup_complete'])
                self.assertTrue(receipt['cancelled'])
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=5)
                if rank is not None and alive(rank):os.kill(rank,signal.SIGKILL)
                unrelated.kill();unrelated.wait(timeout=5)

    def test_normal_worker_exit_does_not_abandon_orphan(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            worker='''
import subprocess,sys
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
(Path(sys.argv[1])/'rank.pid').write_text(str(p.pid))
'''
            proc=self.launch(job,worker)
            try:
                self.assertEqual(proc.wait(timeout=6),1)
                rank=int((job/'rank.pid').read_text())
                self.assertFalse(alive(rank))
                self.assertTrue(json.loads((job/'supervisor-result.json').read_text())['cleanup_complete'])
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=5)

    def test_worker_failure_exit_code_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            proc=self.launch(Path(d),'import sys;sys.exit(7)')
            self.assertEqual(proc.wait(timeout=6),7)
            receipt=json.loads((Path(d)/'supervisor-result.json').read_text())
            self.assertTrue(receipt['cleanup_complete'])
            self.assertEqual(receipt['worker_returncode'],7)

    def test_healthy_worker_success(self):
        with tempfile.TemporaryDirectory() as d:
            proc=self.launch(Path(d),'pass')
            self.assertEqual(proc.wait(timeout=6),0)
            receipt=json.loads((Path(d)/'supervisor-result.json').read_text())
            self.assertTrue(receipt['cleanup_complete'])
            self.assertFalse(receipt['cancelled'])


if __name__=='__main__':unittest.main(verbosity=2)
