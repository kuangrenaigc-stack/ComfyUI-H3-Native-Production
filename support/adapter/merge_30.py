"""User-requested 30s delivery: mild synchronized retime, no content cut."""
from pathlib import Path
import subprocess
from media import probe,executable

def merge(first,second,destination,overlap=22):
    a,b=probe(first),probe(second)
    for key in ('width','height','fps','sample_rate','channels'):
        if a[key]!=b[key]:raise ValueError('Segment format differs: '+key)
    count=a['frames']+b['frames']-overlap
    target=720
    rate=a['sample_rate']
    end_a,end_b,cut=[round(n*rate/24) for n in (a['frames'],b['frames'],overlap)]
    filters=(f'[0:v]setpts=PTS-STARTPTS[v0];'
        f'[1:v]trim=start_frame={overlap},setpts=PTS-STARTPTS[v1];'
        '[v0][v1]concat=n=2:v=1:a=0[vc];'
        f'[vc]setpts=(PTS-STARTPTS)*{target}/{count},fps=24:round=near,'
        f'tpad=stop_mode=clone:stop_duration=0.1,trim=end_frame={target},setpts=PTS-STARTPTS[v];'
        f'[0:a]apad,atrim=end_sample={end_a},asetpts=PTS-STARTPTS[a0];'
        f'[1:a]apad,atrim=start_sample={cut}:end_sample={end_b},asetpts=PTS-STARTPTS[a1];'
        '[a0][a1]concat=n=2:v=0:a=1[ac];'
        f'[ac]atempo={count/target:.12f},apad,atrim=end_sample={30*rate},asetpts=PTS-STARTPTS[a]')
    destination=Path(destination);temp=destination.with_name(destination.stem+'.partial.mp4')
    subprocess.run([executable('ffmpeg'),'-nostdin','-y','-v','error','-i',str(first),'-i',str(second),
        '-filter_complex',filters,'-map','[v]','-map','[a]','-c:v','libx264','-preset','fast','-crf','19',
        '-pix_fmt','yuv420p','-fps_mode','passthrough','-c:a','aac','-b:a','192k','-movflags','+faststart',str(temp)],check=True)
    result=probe(temp)
    if result['frames']!=target or (result['width'],result['height'])!=(a['width'],a['height']):
        raise ValueError('30-second merge contract failed')
    temp.replace(destination)
    return dict(**result,seconds=30.,native_merged_seconds=count/24,overlap_frames=overlap,
                segments=2,retimed=True,speed_factor=count/target,
                frame_adjustment='uniform duplicate/drop at 24fps; no motion interpolation',
                audio_adjustment='atempo, pitch preserved',spatial_crop=False)
