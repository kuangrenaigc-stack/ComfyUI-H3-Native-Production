"""Real complete-entry supervisor, inherited native lock, detached CPU ranks."""
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
sys.path.insert(0,str(PLUGIN))
from process_control import stop_supervisor

@unittest.skipUnless(sys.platform=='linux','Linux child-subreaper test')
class CompleteCancel(unittest.TestCase):
    def test_complete_holds_both_locks_until_detached_ranks_exit(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);(job/'worker').mkdir()
            (job/'rank.py').write_text("import os,signal,time\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath(os.environ['AUDIT_JOB'],'rank'+os.environ['RANK']+'.pid').write_text(str(os.getpid()))\ntime.sleep(60)\n")
            (job/'worker/run_job.py').write_text('''import fcntl,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,os.environ['PLUGIN_PATH'])
from inherited_lock import InheritedRuntimeLock
job=Path(sys.argv[1]);lock=InheritedRuntimeLock(job/'run.lock',os.environ['H3_RUN_LOCK_FD'])
with lock.open('w') as native:
 fcntl.flock(native,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (job/'native-lock-ok').write_text('yes')
 subprocess.run([os.environ['TORCHRUN'],'--standalone','--nproc_per_node=2',str(job/'rank.py')],check=True)
''')
            driver='''import fcntl,functools,importlib.util,os,sys
from pathlib import Path
job=Path(sys.argv[1]);sys.argv=[sys.argv[0],str(job)]
spec=importlib.util.spec_from_file_location('complete_supervisor',sys.argv[0] if False else os.environ['COMPLETE_SUPERVISOR'])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.PACKAGE=job;module.supervise=functools.partial(module.supervise,grace_seconds=1)
control=int(os.environ['H3_CONTROL_LOCK_FD'])
assert os.fstat(control).st_ino==(job/'control.lock').stat().st_ino
sys.exit(module.main())
'''
            env=dict(os.environ,H3_ROOT=str(job),AUDIT_JOB=str(job),PLUGIN_PATH=str(PLUGIN),
                TORCHRUN=str(RUNTIME/'.venv/bin/torchrun'),
                COMPLETE_SUPERVISOR=str(PLUGIN.parent/'worker/supervise_complete.py'))
            control=(job/'control.lock').open('a')
            fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env['H3_CONTROL_LOCK_FD']=str(control.fileno())
            proc=subprocess.Popen([sys.executable,'-c',driver,str(job)],env=env,pass_fds=(control.fileno(),),
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            control.close()  # Only the inherited supervisor copy retains control.
            ranks=[]
            try:
                deadline=time.monotonic()+20
                while not all((job/f'rank{i}.pid').exists() for i in (0,1)):
                    self.assertIsNone(proc.poll())
                    if time.monotonic()>deadline:self.fail('CPU ranks not ready')
                    time.sleep(.05)
                ranks=[int((job/f'rank{i}.pid').read_text()) for i in (0,1)]
                self.assertTrue((job/'native-lock-ok').exists())
                with self.assertRaisesRegex(RuntimeError,'仍在清理'):
                    stop_supervisor(proc,timeout_seconds=.1)
                self.assertIsNone(proc.poll())
                for name in ('control.lock','run.lock'):
                    with (job/name).open('a') as lock:
                        with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(proc.wait(timeout=10),130)
                self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in ranks))
                for name in ('control.lock','run.lock'):
                    with (job/name).open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(json.loads((job/'supervisor-result.json').read_text())['cleanup_complete'])
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=5)
                for pid in ranks:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass

if __name__=='__main__':unittest.main()
