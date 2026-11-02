"""Real worker -> native main -> learned pipeline, replacing GPU work only.

Receipt checks, native configure/phase rules, dispatch and import checks are
real. Synthetic AV/latent fixtures never publish to the production directory.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

PLUGIN=Path(__file__).resolve().parents[1]


class RealSingleChain(unittest.TestCase):
    def test_native_configuration_reaches_real_learned_pipeline(self):
        code=r'''
import json,os,runpy,subprocess,sys,tempfile
from pathlib import Path
import torch
from PIL import Image
root=Path('/opt/h3/runtime')
package=Path(sys.argv[1]);schedule=int(sys.argv[2]);start=int(sys.argv[3]);adapter=root.parent/'research/learned-latent-upscale-20261027/adapter'
os.environ['H3_ROOT']=str(root);os.environ.pop('H3_RUN_LOCK_FD',None)
sys.path.insert(0,str(root/'production'));sys.path.insert(0,str(adapter))
import job_runner,native_delivery
from latent_contract import save,load,sha,tensor_sha
from adapter_safety import atomic_json
real_run=subprocess.run
calls=[];original_publish=native_delivery.publish
with tempfile.TemporaryDirectory() as d:
 job=Path(d);job_runner.LOCK=job/'isolated-test.lock'
 image=job/'reference.png';Image.new('RGB',(64,64),(120,120,120)).save(image)
 config=json.loads((root/'configs/feihou-ref2va-native10plus5.json').read_text())
 config.update(size=[640,1088],h3_draft_size=[320,544],num_frames=39,
  h3_two_pass=True,h3_draft_steps=10,infer_steps=schedule,h3_resample_start=start,
  h3_handoff='learned-clean-latent-2x',ffn_inplace_activation=True)
 (job/'config.json').write_text(json.dumps(config))
 request=dict(prompt='complete CPU test prompt',image_path=str(image),video_path='',audio_path='',num_frames=39,seed=1)
 (job/'request-overrides.json').write_text(json.dumps(request))
 def phase(folder,cfg,req,label,passes):
  calls.append(label);folder.mkdir(exist_ok=True)
  assert json.loads((job/'config.json').read_text())['h3_handoff']=='learned-clean-latent-2x'
  h,w=cfg['size'];effective=dict(req,save_result_path=str(folder/'generated.mp4'))
  atomic_json(folder/'phase-config.json',cfg);atomic_json(folder/'effective-request.json',effective)
  real_run(['/usr/bin/ffmpeg','-nostdin','-y','-v','error',
   '-f','lavfi','-i',f'color=c=black:s={w}x{h}:r=24',
   '-f','lavfi','-i','anullsrc=r=32000:cl=stereo','-frames:v','39','-t','1.625',
   '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac',str(folder/'generated.mp4')],check=True)
  if label=='draft':
   assert cfg['infer_steps']==10 and cfg['lora_configs']==[]
   video=torch.randn(1,24,3,h//16,w//16);audio=torch.randn(2,32,37)
   receipt=save(folder/'clean.safetensors',video,audio=audio,num_frames=39,pixel_height=h,pixel_width=w,native_steps=10)
   for rank in (0,1):
    atomic_json(folder/f'clean-export-rank{rank}.json',dict(rank=rank,shape=list(video.shape),
     video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio),steps=10,sigma_end=0.))
  elif label=='final-suffix':
   assert cfg['infer_steps']==schedule and cfg['h3_cache_start_step']==start and cfg['lora_configs']
   assert req['resample_start_step']==start
   source=json.loads(Path(os.environ['H3_LEARNED_SOURCE']).read_text())
   data,meta=load(source['path'],source['sha256'])
   assert data['video'].shape[-2:]==(h//16,w//16)
   for rank in (0,1):atomic_json(folder/f'learned-source-rank{rank}.json',dict(rank=rank,
     path=source['path'],sha256=source['sha256'],video_sha256=tensor_sha(data['video']),audio_sha256='fixture-audio'))
  else:raise AssertionError(label)
  passes.append(dict(label=label,returncode=0,output_exists=True,wall_seconds=0))
 def gpu_only(command,*args,**kwargs):
  if isinstance(command,(list,tuple)) and len(command)>1 and str(command[1])==str(adapter/'upscale.py'):
   assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='0,1'
   source,output=Path(command[2]),Path(command[3]);data,meta=load(source)
   high=data['video'].repeat_interleave(2,-2).repeat_interleave(2,-1)+.01
   receipt=save(output,high,num_frames=39,pixel_height=meta['pixel_height']*2,
    pixel_width=meta['pixel_width']*2,source_sha256=sha(source),model_sha256='CPU fixture, no neural execution')
   atomic_json(output.with_suffix('.run.json'),receipt)
   return subprocess.CompletedProcess(command,0)
  return real_run(command,*args,**kwargs)
 job_runner.run_roll_phase=phase
 subprocess.run=gpu_only
 native_delivery.publish=lambda path,label:dict(filename='cpu-fixture.mp4',sha256=sha(path),test_only=True)
 try:
  sys.argv=[str(package/'worker/run_job.py'),str(job),'--stage','complete']
  runpy.run_path(str(package/'worker/run_job.py'),run_name='__main__')
 finally:subprocess.run=real_run;native_delivery.publish=original_publish
 result=json.loads((job/'stage-result.json').read_text())
 assert result['returncode']==0 and result['output_exists']
 assert calls==['draft','final-suffix']
 assert result['handoff']=='learned-clean-latent-2x'
 assert result['sampling']['handoff']=='learned-clean-latent-2x'
 assert result['sampling']['final_steps']==schedule-start
 assert len(result['source_receipts'])==2
 assert result['published_video']['test_only']
 assert (job/'node-package-binding.json').is_file()
 print('REAL_SINGLE_CHAIN_CPU_OK')
'''
        for schedule,start in ((4,1),(8,3),(8,4)):
            with self.subTest(schedule=schedule,start=start):
                result=subprocess.run([sys.executable,'-c',code,str(PLUGIN.parent),str(schedule),str(start)],
                    env=dict(os.environ,OMP_NUM_THREADS='2'),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertIn('REAL_SINGLE_CHAIN_CPU_OK',result.stdout)

if __name__=='__main__':unittest.main()
