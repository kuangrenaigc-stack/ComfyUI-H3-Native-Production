import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

PLUGIN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PLUGIN))
from inherited_lock import InheritedRuntimeLock
from package_binding import bind
from delivery_status import published_after_cleanup

class ReviewRepairs(unittest.TestCase):
    def test_real_adapter_dependency_origin_accepts_symlink(self):
        code=r'''
import sys,tempfile
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import learned_pipeline
actual=learned_pipeline.ADAPTER.resolve(strict=True)
with tempfile.TemporaryDirectory() as d:
 alias=Path(d)/'adapter-link';alias.symlink_to(actual,target_is_directory=True)
 learned_pipeline.ADAPTER=alias
 module=learned_pipeline.dependencies()
 assert Path(module.__file__).resolve()==actual/'adapter_safety.py'
'''
        subprocess.run([sys.executable,'-c',code,str(PLUGIN)],check=True,
            env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=subprocess.DEVNULL)

    def test_real_job_runner_main_uses_inherited_lock_without_unlocking_supervisor(self):
        native=Path('/opt/h3/runtime/production/job_runner.py')
        self.assertNotIn('LOCK_UN',native.read_text())
        code=r'''
import fcntl,json,os,sys
from pathlib import Path
root=Path('/opt/h3/runtime')
os.environ['H3_ROOT']=str(root);sys.path.insert(0,str(root/'production'))
import job_runner
sys.path.insert(0,sys.argv[1]);from inherited_lock import InheritedRuntimeLock
job=Path(sys.argv[2]);fd=int(os.environ['H3_RUN_LOCK_FD'])
job_runner.LOCK=InheritedRuntimeLock(job/'lock',fd)
config=json.loads((root/'configs/feihou-ref2va-native10plus5.json').read_text())
(job/'config.json').write_text(json.dumps(config))
calls=[]
def fake(*args):
 with (job/'lock').open('a') as outside:
  try:fcntl.flock(outside,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Native main lost exclusion')
 calls.append('real-main');return 0
job_runner.run_two_pass=fake
sys.argv=[sys.argv[0],str(job),'--stage','complete'];job_runner.main()
assert calls==['real-main']
assert json.loads((job/'stage-result.json').read_text())['returncode']==0
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with (job/'lock').open('a') as parent:
                fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
                subprocess.run([sys.executable,'-c',code,str(PLUGIN),str(job)],check=True,
                    env=dict(os.environ,H3_RUN_LOCK_FD=str(parent.fileno())),
                    pass_fds=(parent.fileno(),),stdout=subprocess.DEVNULL)
                with (job/'lock').open('a') as outsider:
                    with self.assertRaises(BlockingIOError):fcntl.flock(outsider,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_inherited_lock_native_reacquisition_and_parent_retention(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock'
            with p.open('a') as parent:
                fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
                proxy=InheritedRuntimeLock(p,parent.fileno())
                with proxy.open('w') as worker:
                    fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with p.open('a') as outsider:
                    with self.assertRaises(BlockingIOError):fcntl.flock(outsider,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(proxy.resolve(),p.resolve())
            with p.open('a') as outsider:fcntl.flock(outsider,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_package_binding_refuses_changed_code_and_unbound_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code=root/'code';code.mkdir();f=code/'a.py';f.write_text('a=1')
            job=root/'job';job.mkdir();digest=bind(job,code)
            self.assertEqual(bind(job,code),digest)
            f.write_text('a=2')
            with self.assertRaises(ValueError):bind(job,code)
            other=root/'unbound';other.mkdir();(other/'generated.mp4').write_bytes(b'output')
            with self.assertRaises(ValueError):bind(other,code)

    def test_recover_delivery_only_after_cleanup_and_original_worker_success(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);videos=job/'videos';videos.mkdir();output=videos/'test.mp4';output.write_bytes(b'video')
            receipt=dict(returncode=0,published_video=dict(filename=output.name,sha256=hashlib.sha256(b'video').hexdigest()))
            (job/'series-result.json').write_text(json.dumps(receipt))
            state=dict(cleanup_complete=False,cancelled=False,worker_returncode=0)
            def check():
                (job/'supervisor-result.json').write_text(json.dumps(state))
                return published_after_cleanup(job,videos)
            self.assertFalse(check());state['cleanup_complete']=True;self.assertTrue(check())
            state['cancelled']=True;self.assertFalse(check())
            state['cancelled']=False;state['worker_returncode']=1;self.assertFalse(check())
            state['worker_returncode']=0;output.write_bytes(b'changed');self.assertFalse(check())

    def test_single_worker_installs_learned_dispatch_before_native_main(self):
        runner=ModuleType('job_runner');runner.sampling_plan=lambda c:{'handoff':'RGB'}
        calls=[]
        def main():
            self.assertEqual(runner.run_two_pass.__module__,'learned_pipeline')
            self.assertEqual(runner.sampling_plan({})['handoff'],'learned-clean-latent-2x')
            calls.append('main')
        runner.main=main
        with patch.dict(sys.modules,{'job_runner':runner}),patch.dict(os.environ):
            os.environ.pop('H3_RUN_LOCK_FD',None)
            runpy.run_path(str(PLUGIN.parent/'worker/run_job.py'),run_name='worker_test')
        self.assertEqual(calls,['main'])

    def test_real_learned_module_resolves_declared_standalone_contracts(self):
        # Fresh process avoids the deliberate test-suite fake module cache.
        code=r'''
import importlib.util,json,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,sys.argv[1]);os.environ['H3_ROOT']='/opt/h3/runtime'
import learned_bridge as bridge
original=importlib.util.spec_from_file_location
class Loader:
 def __init__(self,real):self.real=real
 def create_module(self,spec):return self.real.create_module(spec)
 def exec_module(self,module):
  self.real.exec_module(module)
  def execute(job):
   expected=Path('/opt/h3/comfyui/custom_nodes/ComfyUI-H3-Native-Continuation')
   import media
   for name in ('continuation_core','receipt_contract','media'):
    assert Path(sys.modules[name].__file__).resolve()==expected/(name+'.py')
   (job/'series-result.json').write_text(json.dumps({'returncode':0}));return 'real-import-ok'
  module.execute=execute
def spec(name,path,*a,**kw):
 s=original(name,path,*a,**kw)
 if name=='h3_segment_learned_series':s.loader=Loader(s.loader)
 return s
importlib.util.spec_from_file_location=spec
with tempfile.TemporaryDirectory() as d:
 job=Path(d);assert bridge.execute(job)=='real-import-ok'
 assert json.loads((job/'series-result.json').read_text())['node_package_binding']
'''
        env=dict(os.environ,OMP_NUM_THREADS='2')
        subprocess.run([sys.executable,'-c',code,str(PLUGIN)],check=True,env=env,stdout=subprocess.DEVNULL)

if __name__=='__main__':unittest.main()
