"""CPU-only regression cases; no model generation, GPU work or publication."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import torch
import adapter_safety as safety
from latent_contract import save,sha,tensor_sha

HERE=Path(__file__).resolve().parent

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.job=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_missing_root_uses_current_runtime(self):
        env=dict(os.environ);env.pop('H3_ROOT',None)
        code="import run_experiment as a; import run_series_learned as b; assert a.job_runner.ROOT.resolve()==a.ROOT.resolve(); assert b.job_runner.LOCK.resolve()==(b.ROOT/'run.lock').resolve()"
        subprocess.run([sys.executable,'-c',code],cwd=HERE,env=env,check=True)

    def test_wrong_root_is_rejected(self):
        env=dict(os.environ,H3_ROOT=str(self.job/'wrong'))
        result=subprocess.run([sys.executable,'-c','import run_experiment'],cwd=HERE,env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('outside this adapter runtime',result.stderr)

    def test_job_request_change_fails_before_any_phase(self):
        import run_experiment as entry
        cfg=dict(size=[128,128],h3_draft_size=[64,64],h3_draft_steps=10,h3_resample_start=3)
        request=dict(prompt='original',seed=1,num_frames=22,image_path='',video_path='',audio_path='')
        safety.atomic_json(self.job/'config.json',cfg)
        safety.atomic_json(self.job/'request-overrides.json',request)
        merged=safety.read(entry.job_runner.BASE_REQUEST);merged.update(request)
        safety.bind_job(self.job,cfg,[safety.clean_request(merged)],entry.job_runner,HERE)
        request['seed']=2;safety.atomic_json(self.job/'request-overrides.json',request)
        with patch.object(entry.job_runner,'run_roll_phase') as phase:
            with self.assertRaisesRegex(ValueError,'inputs/code changed'):
                entry.execute(self.job,publish=True)
            phase.assert_not_called()

    def test_changed_reference_file_rejected(self):
        asset=self.job/'image.png';asset.write_bytes(b'original')
        request=dict(image_path=str(asset),seed=1)
        safety.bind_job(self.job,{},[request],None,HERE)
        asset.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'inputs/code changed'):
            safety.bind_job(self.job,{},[request],None,HERE)

    def test_legacy_job_without_binding_fails_closed(self):
        (self.job/'generated.mp4').write_bytes(b'old')
        with self.assertRaisesRegex(ValueError,'no input binding'):
            safety.bind_job(self.job,{},[],None,HERE)

    def test_phase_cleans_draft_steps_and_rejects_stale_cache(self):
        cfg=dict(parallel={'tensor_p_size':2},infer_steps=10,h3_sampling_stage='native-draft',
                 h3_step_update='reference_blend',h3_acceleration_mode='standard',lora_configs=[],
                 lora_dynamic_apply=False,enable_cfg=False,size=[64,64])
        request=dict(prompt='p',seed=1,num_frames=22,draft_steps=3)
        calls=[]
        def fake(folder,config,request,label,passes):
            calls.append(request)
            safety.atomic_json(folder/'phase-config.json',config)
            safety.atomic_json(folder/'effective-request.json',dict(request,save_result_path=str(folder/'generated.mp4')))
            (folder/'generated.mp4').write_bytes(b'completed')
        runner=types.SimpleNamespace(run_roll_phase=fake)
        phase=self.job/'phase'
        with patch('two_pass.verify_draft'):
            safety.run_phase(runner,phase,cfg,request,'draft',[],{'source':'a'})
        self.assertNotIn('draft_steps',calls[0])
        self.assertTrue(safety.run_phase(runner,phase,cfg,request,'draft',[],{'source':'a'}))
        with self.assertRaisesRegex(ValueError,'inputs or dependencies changed'):
            safety.run_phase(runner,phase,cfg,dict(request,seed=2),'draft',[],{'source':'a'})
        with self.assertRaisesRegex(ValueError,'inputs or dependencies changed'):
            safety.run_phase(runner,phase,cfg,request,'draft',[],{'source':'b'})
        self.assertEqual(len(calls),1)

    def test_incomplete_upscale_is_not_a_cache_hit(self):
        src=self.job/'source.safetensors';out=self.job/'learned.safetensors'
        save(src,torch.zeros(1,24,2,4,4),num_frames=22,pixel_height=64,pixel_width=64)
        report=save(out,torch.zeros(1,24,2,8,8),num_frames=22,pixel_height=128,pixel_width=128,source_sha256=sha(src))
        self.assertFalse(safety.upscale_ready(out,src))
        safety.atomic_json(out.with_suffix('.run.json'),report)
        self.assertTrue(safety.upscale_ready(out,src))
        save(src,torch.ones(1,24,2,4,4),num_frames=22,pixel_height=64,pixel_width=64)
        with self.assertRaisesRegex(ValueError,'output/source binding changed'):
            safety.upscale_ready(out,src)

    def test_clean_artifact_cross_checked_with_rank_exports(self):
        video=torch.zeros(1,24,2,4,4);audio=torch.zeros(2,32,37)
        save(self.job/'clean.safetensors',video,audio=audio,num_frames=22,pixel_height=64,pixel_width=64)
        receipt=dict(shape=list(video.shape),video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio),steps=10,sigma_end=0.)
        for rank in (0,1):safety.atomic_json(self.job/f'clean-export-rank{rank}.json',dict(receipt,rank=rank))
        safety.verify_clean(self.job)
        # Both rank receipts agreeing is insufficient when the saved tensor differs.
        save(self.job/'clean.safetensors',torch.ones_like(video),audio=audio,num_frames=22,pixel_height=64,pixel_width=64)
        with self.assertRaisesRegex(ValueError,'does not match the TP2 exports'):
            safety.verify_clean(self.job)

    def test_final_receipts_must_point_to_current_source(self):
        source=self.job/'learned.safetensors';source.write_bytes(b'current')
        receipt=dict(path=str(source),sha256='0'*64,video_sha256='v',audio_sha256='a')
        for rank in (0,1):safety.atomic_json(self.job/f'learned-source-rank{rank}.json',dict(receipt,rank=rank))
        with self.assertRaisesRegex(ValueError,'different learned source'):
            safety.check_sources(self.job,source)

    def test_cancel_cleans_detached_child_before_releasing_locks(self):
        fixture=self.job/'fixture.py'
        fixture.write_text('''import os,sys,time,subprocess,types
from pathlib import Path
from unittest.mock import patch
import adapter_safety as s
j=Path(sys.argv[1]);r=types.SimpleNamespace(LOCK=j/'runtime.lock')
def work(job):
 p=subprocess.Popen([sys.executable,'-c',"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"],start_new_session=True)
 (j/'ready').write_text(str(p.pid))
 while True:time.sleep(.05)
with patch('subprocess.check_output',return_value=''):
 code=s.run_locked(j,r,work,Path(__file__),worker='--worker' in sys.argv,control_path=j/'control.lock',grace_seconds=.3)
raise SystemExit(code)
''')
        env=dict(os.environ,PYTHONPATH=str(HERE))
        process=subprocess.Popen([sys.executable,str(fixture),str(self.job)],env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+12
            while not (self.job/'ready').exists():
                if process.poll() is not None:self.fail(process.communicate()[1])
                if time.monotonic()>deadline:self.fail('CPU cancellation fixture did not start')
                time.sleep(.03)
            for name in ('control.lock','runtime.lock'):
                with (self.job/name).open('a') as lock:
                    with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pid=int((self.job/'ready').read_text())
            # Send TERM only to the supervisor parent, not to its process group.
            process.send_signal(signal.SIGTERM)
            stdout,stderr=process.communicate(timeout=12)
            self.assertEqual(process.returncode,130,stderr)
            receipt=safety.read(self.job/'supervisor-result.json')
            self.assertTrue(receipt['cleanup_complete']);self.assertTrue(receipt['cancelled'])
            self.assertFalse(Path(f'/proc/{pid}').exists())
            for name in ('control.lock','runtime.lock'):
                with (self.job/name).open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM);process.communicate(timeout=12)

if __name__=='__main__':unittest.main(verbosity=2)
