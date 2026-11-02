"""Visual-only references: remove soundtrack with a lossless video stream copy.

Native H3 numbers video soundtracks before standalone audio. Excluding tracks
keeps explicit Audio N bindings deterministic and avoids importing clip music.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


class VisualVideo:
    def __init__(self, original):
        self.original = original

    def save_to(self, destination):
        destination = Path(destination)
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('ffmpeg is required for lossless removal of reference-video audio')
        with tempfile.TemporaryDirectory(prefix='.h3-v2-video-', dir=destination.parent) as directory:
            source, output = Path(directory)/'source.mp4', Path(directory)/'visual.mp4'
            self.original.save_to(str(source))
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(source),
                            '-map', '0:v:0', '-c:v', 'copy', '-an', '-y', str(output)],
                           check=True, timeout=180)
            os.replace(output, destination)


def visual_only(materials):
    return [dict(segment, videos=[VisualVideo(v) for v in segment['videos']]) for segment in materials]
