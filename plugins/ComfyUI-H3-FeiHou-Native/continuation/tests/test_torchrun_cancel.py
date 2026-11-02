"""Cancel actual two-rank torchrun workers, using CPU only and no H3 weights."""
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

PLUGIN=Path(__file__).resolve().parents[1]
RUNTIME=Path('/opt/h3/runtime')


@unittest.skipUnless(sys.platform=='linux','Linux torchrun integration')
class TorchrunCancel(unittest.TestCase):
    def test_actual_detached_ranks_terminated_before_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            rank_script=job/'rank.py'
            rank_script.write_text('''import os,signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(os.environ['AUDIT_JOB'], 'rank'+os.environ['RANK']+'.pid').write_text(str(os.getpid()))
time.sleep(60)
''')
            wrapper='''import fcntl,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from process_control import supervise
job=Path(sys.argv[2])
with (job/'run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sys.exit(supervise([sys.argv[3],'--standalone','--nproc_per_node=2',str(job/'rank.py')],job,env=dict(os.environ,AUDIT_JOB=str(job)),grace_seconds=1))
'''
            proc=subprocess.Popen([sys.executable,'-c',wrapper,str(PLUGIN),str(job),str(RUNTIME/'.venv/bin/torchrun')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            ranks=[]
            try:
                deadline=time.monotonic()+15
                while not all((job/f'rank{r}.pid').exists() for r in (0,1)):
                    self.assertIsNone(proc.poll())
                    if time.monotonic()>deadline:self.fail('torchrun ranks never became ready')
                    time.sleep(.05)
                ranks=[int((job/f'rank{r}.pid').read_text()) for r in (0,1)]
                self.assertTrue(all(os.getsid(p)!=proc.pid for p in ranks))
                proc.send_signal(signal.SIGTERM)
                time.sleep(.1)
                with (job/'run.lock').open('a') as lock:
                    with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(proc.wait(timeout=8),130)
                for pid in ranks:
                    self.assertFalse(Path(f'/proc/{pid}').exists())
                with (job/'run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(json.loads((job/'supervisor-result.json').read_text())['cleanup_complete'])
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=5)
                for pid in ranks:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass

if __name__=='__main__':unittest.main(verbosity=2)
