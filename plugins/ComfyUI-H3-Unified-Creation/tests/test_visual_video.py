import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('visual_video_fixture',Path(__file__).resolve().parents[1]/'visual_video.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class VisualVideoTests(unittest.TestCase):
    def test_fail_closed_without_ffmpeg(self):
        with patch.object(MODULE.shutil,'which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'ffmpeg'):
                MODULE.VisualVideo(object()).save_to('/nonexistent/unused.mp4')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'requires existing FFmpeg CLI')
    def test_real_video_packets_preserved_and_audio_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);source=directory/'input.mp4';output=directory/'output.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=5:d=1',
                            '-f','lavfi','-i','sine=frequency=440:sample_rate=8000:duration=1',
                            '-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(source)],check=True)
            class Source:
                def save_to(self,path):shutil.copyfile(source,path)
            MODULE.VisualVideo(Source()).save_to(output)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))
            self.assertEqual([s['codec_type'] for s in info['streams']],['video'])
            self.assertEqual(info['streams'][0]['width'],64)
            def hashes(path):
                value=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets',
                    '-show_data_hash','sha256','-show_entries','packet=data_hash','-of','json',str(path)]))
                return [p['data_hash'] for p in value['packets']]
            self.assertEqual(hashes(source),hashes(output))
            self.assertFalse(list(directory.glob('.h3-v2-video-*')))


if __name__=='__main__':unittest.main()
