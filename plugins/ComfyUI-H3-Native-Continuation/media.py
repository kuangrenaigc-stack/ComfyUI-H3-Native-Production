"""Frame-exact overlap removal. Never resize, retime or synthesize video frames."""
import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess


def executable(name):
    return shutil.which(name) or str(Path.home() / '.local/bin' / name)


def probe(path):
    data = json.loads(subprocess.check_output([executable('ffprobe'), '-v', 'error',
        '-count_frames', '-show_streams', '-of', 'json', str(path)], text=True))
    video = next(x for x in data['streams'] if x['codec_type'] == 'video')
    audio = next(x for x in data['streams'] if x['codec_type'] == 'audio')
    result = dict(width=int(video['width']), height=int(video['height']),
        frames=int(video['nb_read_frames']), fps=str(Fraction(video['r_frame_rate'])),
        sample_rate=int(audio['sample_rate']), channels=int(audio['channels']),
        audio_seconds=float(audio['duration']))
    if result['fps'] != '24' or result['audio_seconds'] < result['frames'] / 24 - 1/24:
        raise ValueError('Missing/short audio or non-native FPS')
    return result


def merge(first, second, destination, overlap=22):
    a, b = probe(first), probe(second)
    for key in ('width', 'height', 'fps', 'sample_rate', 'channels'):
        if a[key] != b[key]:
            raise ValueError(f'Segment {key} differs; implicit conversion refused')
    if not 0 < overlap < min(a['frames'], b['frames']):
        raise ValueError('Invalid overlap')
    sr = a['sample_rate']
    end_a, end_b, cut = [round(n * sr / 24) for n in (a['frames'], b['frames'], overlap)]
    filters = (f'[0:v]setpts=PTS-STARTPTS[v0];'
        f'[1:v]trim=start_frame={overlap},setpts=PTS-STARTPTS[v1];'
        f'[0:a]apad,atrim=end_sample={end_a},asetpts=PTS-STARTPTS[a0];'
        f'[1:a]apad,atrim=start_sample={cut}:end_sample={end_b},asetpts=PTS-STARTPTS[a1];'
        '[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]')
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    subprocess.run([executable('ffmpeg'), '-nostdin', '-y', '-v', 'error',
        '-i', str(first), '-i', str(second), '-filter_complex', filters,
        '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-c:a', 'aac',
        '-b:a', '192k', '-movflags', '+faststart', str(temporary)], check=True)
    result = probe(temporary)
    expected = a['frames'] + b['frames'] - overlap
    if result['frames'] != expected or (result['width'], result['height']) != (a['width'], a['height']):
        raise ValueError('Merged video geometry/frame count mismatch')
    temporary.replace(destination)
    return dict(**result, seconds=expected/24, overlap_frames=overlap,
                segments=2, merge_encode='libx264-crf19 / AAC', retimed=False)
