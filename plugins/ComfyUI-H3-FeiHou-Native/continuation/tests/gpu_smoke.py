"""Explicit isolated native integration test; no ComfyUI restart or publication."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

PLUGIN=Path(__file__).resolve().parents[1]
ROOT=Path('/opt/h3/runtime')
sys.path.insert(0,str(PLUGIN))
sys.path.insert(0,str(PLUGIN/'worker'))
sys.path.insert(0,str(ROOT/'production'))
os.environ['H3_ROOT']=str(ROOT)


def main():
    control=Path(f'/run/user/{os.getuid()}/h3-native/control.lock')
    control.parent.mkdir(parents=True,exist_ok=True)
    with control.open('a') as guard, (ROOT/'run.lock').open('a') as runlock:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        if busy:raise RuntimeError('GPU busy; integration test not started')
        from PIL import Image, ImageDraw
        import job_runner
        from two_pass import phase_config,verify_draft
        from run_series import execute
        job=PLUGIN/'smoke-jobs'/uuid.uuid4().hex
        job.mkdir(parents=True)
        print('SMOKE_JOB '+str(job),flush=True)
        image=Image.new('RGB',(512,512),(125,125,125))
        draw=ImageDraw.Draw(image);draw.rectangle((0,350,511,511),fill=(80,80,80))
        draw.ellipse((180,220,280,320),fill=(220,40,30))
        source=job/'reference.png';image.save(source)
        config=json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
        config.update(size=[512,512],num_frames=124,segment_seconds=5,h3_two_pass=True,
            h3_draft_steps=10,h3_resample_start=3,h3_draft_size=[480,480])
        (job/'config.json').write_text(json.dumps(config,indent=2))
        prompt=('short_description: A red ball rests on a grey floor. '
            'detailed_description: [Shot 1] A single static shot of the red ball from <Picture 1> on the floor, '
            'soft light from the upper left, a faint contact shadow, no camera motion. '
            'dialogue: N/A. overall_soundscape: Quiet room tone. '
            'non_diegetic_music: N/A. voiceover: N/A.')
        requests=[dict(prompt=prompt,image_path=str(source),video_path='',audio_path='',num_frames=124,seed=2026102601+i) for i in range(2)]
        (job/'requests.json').write_text(json.dumps(requests,indent=2))
        def unpublished(path,label):return dict(test_only=True,path=str(path),published=False)
        result=execute(job,job_runner,phase_config,verify_draft,unpublished)
        for phase,conversion in [('draft','RGB-Lanczos-VAE'),('.','direct-final-latent')]:
            receipts=[json.loads((job/'segment2'/phase/f'continuation-guide-rank{rank}.json').read_text()) for rank in range(2)]
            assert all(r['conversion']==conversion for r in receipts)
            assert all(r['source']['path']==str(job/'segment1/continuation-tail.safetensors') for r in receipts)
        assert result['media']['frames']==226
        print('SMOKE_COMPLETE '+json.dumps(dict(job=str(job),frames=result['media']['frames'],seconds=result['media']['seconds'],published=False)),flush=True)


if __name__=='__main__':main()
