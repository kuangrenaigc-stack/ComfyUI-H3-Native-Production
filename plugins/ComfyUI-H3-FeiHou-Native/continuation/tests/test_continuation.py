import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))
ROOT = Path(os.environ.get('H3_ROOT', '/opt/h3/runtime'))
NATIVE = ROOT/'source/lightx2v/models/networks/minimax_h3'
package = ModuleType('native_contract')
package.__path__ = [str(NATIVE)]
sys.modules[package.__name__] = package

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

packing = load('native_contract.packing', NATIVE/'packing.py')
refpack = load('native_contract.packing_ref2av', NATIVE/'packing_ref2av.py')
core = load('continuation_core', PLUGIN/'continuation_core.py')
media = load('media', PLUGIN/'media.py')
series = load('series_test', PLUGIN/'worker/run_series.py')
phases = load('two_pass_test', ROOT/'production/two_pass.py')


def state(frames=39, height=64, width=64, references=True):
    t, a = packing.video_latent_num_frames(frames), packing.audio_latent_num_frames(frames)
    refs = [refpack.MiniMaxH3PreparedReference(kind='image', latent_height=4, latent_width=4),
        refpack.MiniMaxH3PreparedReference(kind='video', latent_height=4, latent_width=4,
            num_latent_frames=7, num_audio_latents=37),
        refpack.MiniMaxH3PreparedReference(kind='audio', num_audio_latents=20)] if references else []
    layout = refpack.build_ref2av_packed_sequence(torch.ones(8, dtype=torch.long), refs,
        t, height//16, width//16, a)
    return SimpleNamespace(layout_cpu=layout, layout=layout, num_frames=frames,
        height=height, width=width, latent_height=height//16, latent_width=width//16,
        num_latent_frames=t, num_audio_latents=a,
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows,
        video_latents=torch.randn(len(layout.video_indices), 96),
        audio_latents=torch.randn(len(layout.audio_indices), 32))


class Contracts(unittest.TestCase):
    def test_two_segments_native_duration(self):
        p = core.plan(15)
        self.assertEqual((p['segments'], p['frames_per_segment'], p['output_frames'], p['output_seconds']), (2, 362, 702, 29.25))
        for value in (4, 16, 15.0, True):
            with self.assertRaises(ValueError): core.plan(value)

    def test_tail_roundtrip_excludes_reference_and_guide_rows(self):
        s = state()
        core.attach_guide(s, torch.randn(1,24,7,4,4), torch.randn(2,32,37))
        with tempfile.TemporaryDirectory() as d:
            record = core.save_tail(Path(d)/'tail.safetensors', s.video_latents, s.audio_latents, s)
            v, a, _ = core.read_tail(record['path'], record['sha256'], 64, 64)
            target_v = s.video_latents[s.num_condition_video_rows:]
            restored = packing.unpatchify_video_tokens(target_v, s.num_latent_frames, 4, 4)
            target_a = packing.unpack_audio_tokens(s.audio_latents[s.num_condition_audio_rows:], s.num_audio_latents)
            self.assertTrue(torch.equal(v, restored[:, :, -7:]))
            self.assertTrue(torch.equal(a, target_a[:, :, -37:]))
            with self.assertRaises(ValueError): core.read_tail(record['path'], '0'*64)
            with self.assertRaises(ValueError): core.read_tail(record['path'], record['sha256'], 32, 64)

    def test_fixed_guides_leave_original_references_and_noise_unchanged(self):
        s = state()
        old = copy.deepcopy(s)
        core.attach_guide(s, torch.randn(1,24,7,4,4), torch.randn(2,32,37))
        nv, na = 28, 74
        self.assertTrue(torch.equal(s.video_latents[nv:], old.video_latents))
        self.assertTrue(torch.equal(s.audio_latents[na:], old.audio_latents))
        layout = s.layout_cpu
        self.assertTrue(torch.equal(layout.position_ids[layout.video_indices[nv:]], old.layout_cpu.position_ids[old.layout_cpu.video_indices]))
        self.assertTrue(torch.equal(layout.position_ids[layout.audio_indices[na:]], old.layout_cpu.position_ids[old.layout_cpu.audio_indices]))
        vtarget = old.layout_cpu.video_indices[old.num_condition_video_rows:][:nv]
        self.assertTrue(torch.equal(layout.position_ids[layout.video_indices[:nv]], old.layout_cpu.position_ids[vtarget]))
        atarget = old.layout_cpu.audio_indices[old.num_condition_audio_rows:].reshape(2, -1)[:, :37].flatten()
        self.assertTrue(torch.equal(layout.position_ids[layout.audio_indices[:na]], old.layout_cpu.position_ids[atarget]))
        unique, inverse = packing.build_row_timesteps(layout, .2, .3)
        times = unique[inverse]
        self.assertTrue(torch.allclose(times[layout.video_indices[:nv]], torch.full((nv,), .999, dtype=times.dtype)))
        self.assertTrue(torch.all(times[layout.audio_indices[:na]] == 1))
        self.assertTrue(torch.all(times[layout.video_indices[s.num_condition_video_rows:]] == .2))

    def test_guide_augmentation_is_rank_deterministic(self):
        s = state()
        second = copy.deepcopy(s)
        v, a = torch.randn(1,24,7,4,4), torch.randn(2,32,37)
        before = torch.get_rng_state().clone()
        core.attach_guide(s, v, a)
        core.attach_guide(second, v, a)
        self.assertTrue(torch.equal(s.video_latents, second.video_latents))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_bad_guide_does_not_mutate_scheduler(self):
        s = state()
        before = s.video_latents.clone()
        with self.assertRaises(ValueError): core.attach_guide(s, torch.randn(1,24,7,4,4), torch.randn(2,32,36))
        self.assertTrue(torch.equal(s.video_latents, before))

    def test_nonfinite_tail_refused(self):
        s = state()
        s.video_latents[-1] = float('nan')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): core.save_tail(Path(d)/'tail.safetensors', s.video_latents, s.audio_latents, s)


class MediaAndOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temp.name)/'fixture.mp4'
        subprocess.run([media.executable('ffmpeg'), '-v','error','-y','-f','lavfi','-i','testsrc2=s=64x64:r=24',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=32000', '-frames:v','124', '-t',str(124/24),
            '-c:v','libx264','-threads','2','-c:a','aac','-ac','2',str(cls.fixture)], check=True)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_frame_exact_merge_and_audio_alignment(self):
        target = Path(self.temp.name)/'merged.mp4'
        result = media.merge(self.fixture, self.fixture, target)
        self.assertEqual(result['frames'], 226)
        self.assertEqual((result['width'],result['height']), (64,64))
        self.assertLess(abs(result['audio_seconds']-226/24), 1/24)

    def test_two_complete_jobs_use_same_final_source_no_intermediate_publish(self):
        with tempfile.TemporaryDirectory() as d:
            job = Path(d)
            config = dict(segment_seconds=5, h3_two_pass=True, h3_draft_steps=10,
                h3_resample_start=3, size=[64,64], h3_draft_size=[64,64], lora_configs=[{'path':'existing'}])
            (job/'config.json').write_text(json.dumps(config))
            (job/'requests.json').write_text(json.dumps([dict(prompt='first',num_frames=124,seed=1),dict(prompt='second',num_frames=124,seed=2)]))
            template = job/'base-request.json'; template.write_text('{}')
            calls, sources, publications = [], [], []
            def run(folder, cfg, request, label, passes):
                folder.mkdir(exist_ok=True)
                calls.append((label,cfg,request))
                mapping = json.loads(Path(os.environ['H3_CONTINUATION_MAP']).read_text())
                if label.startswith('segment2'):
                    source = mapping[str(folder.resolve())]
                    sources.append(source['path'])
                    for rank in (0, 1):
                        receipt = dict(source=source, conversion='direct-final-latent',
                            guide_frames=22, video_guide_rows=28, audio_guide_rows=74,
                            target_noise_unchanged=True, parallel_implementation_unchanged=True)
                        (folder/f'continuation-guide-rank{rank}.json').write_text(json.dumps(receipt))
                shutil.copyfile(self.fixture, folder/'generated.mp4')
                if label.endswith('final'):
                    s = state(frames=124, references=False)
                    tail = core.save_tail(folder/'continuation-tail.safetensors', s.video_latents, s.audio_latents, s)
                    (folder/'continuation-tail.json').write_text(json.dumps(tail))
                passes.append({'label':label,'returncode':0})
            def publish(path,label):
                publications.append(str(path)); return {'filename':'test.mp4','sha256':core.sha256(path)}
            runtime = SimpleNamespace(BASE_REQUEST=template, run_roll_phase=run)
            result = series.execute(job,runtime,phases.phase_config,phases.verify_draft,publish)
            self.assertEqual(result['returncode'],0)
            self.assertEqual([x[0] for x in calls], ['segment1-draft','segment1-final','segment2-draft','segment2-final'])
            self.assertEqual([x[1]['infer_steps'] for x in calls], [10,8,10,8])
            self.assertEqual([x[1]['h3_cache_start_step'] for x in calls], [0,3,0,3])
            self.assertEqual(sources,[str(job/'segment1/continuation-tail.safetensors')]*2)
            self.assertEqual(publications,[str(job/'generated.mp4')])

    def test_failed_phase_never_starts_next_or_publishes(self):
        with tempfile.TemporaryDirectory() as d:
            job = Path(d)
            (job/'config.json').write_text(json.dumps(dict(segment_seconds=5,h3_two_pass=True,h3_draft_steps=10,h3_resample_start=3)))
            (job/'requests.json').write_text(json.dumps([dict(num_frames=124)]*2))
            template = job/'base.json';template.write_text('{}')
            calls=[]
            def fail(*args): calls.append(args[3]); raise RuntimeError('injected failure')
            runtime=SimpleNamespace(BASE_REQUEST=template,run_roll_phase=fail)
            with self.assertRaises(RuntimeError):
                series.execute(job,runtime,lambda c,p:{},lambda *a:None,lambda *a:self.fail('published failed job'))
            self.assertEqual(calls,['segment1-draft'])
            self.assertEqual(json.loads((job/'series-result.json').read_text())['returncode'],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
