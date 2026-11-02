"""Receipt failures must stop the production path before its next consumer."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))
from receipt_contract import validate_guide_receipts


def load_series():
    spec = importlib.util.spec_from_file_location('receipt_series_test', PLUGIN/'worker/run_series.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_receipts(folder, source, phase_size, *, mutate=None, omit_rank=None):
    for rank in (0, 1):
        if rank == omit_rank:
            continue
        data = dict(source=copy.deepcopy(source),
            conversion='direct-final-latent' if phase_size == [source['height'], source['width']] else 'RGB-Lanczos-VAE',
            guide_frames=22, video_guide_rows=7*(phase_size[0]//32)*(phase_size[1]//32),
            audio_guide_rows=74, target_noise_unchanged=True,
            parallel_implementation_unchanged=True)
        if mutate:
            mutate(data, rank)
        (folder/f'continuation-guide-rank{rank}.json').write_text(json.dumps(data))


class ReceiptContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = dict(path=str(self.folder/'segment1-final.safetensors'), sha256='a'*64,
                           height=512, width=512)

    def tearDown(self): self.temp.cleanup()

    def test_two_ranks_require_matching_source_conversion_and_geometry(self):
        for size, conversion in [([480,480], 'RGB-Lanczos-VAE'), ([512,512], 'direct-final-latent')]:
            with self.subTest(size=size):
                write_receipts(self.folder, self.source, size)
                result = validate_guide_receipts(self.folder, self.source, size)
                self.assertEqual([r['rank'] for r in result], [0,1])
                self.assertTrue(all(r['conversion'] == conversion for r in result))

    def test_missing_or_malformed_rank_receipt_fails(self):
        write_receipts(self.folder, self.source, [512,512], omit_rank=1)
        with self.assertRaisesRegex(ValueError, 'rank 1'):
            validate_guide_receipts(self.folder, self.source, [512,512])
        (self.folder/'continuation-guide-rank1.json').write_text('{broken')
        with self.assertRaisesRegex(ValueError, 'rank 1'):
            validate_guide_receipts(self.folder, self.source, [512,512])

    def test_bad_source_or_guide_contract_fails_closed(self):
        mutations = {
            'source_path': lambda d: d['source'].update(path=str(self.folder/'wrong-tail.safetensors')),
            'source_sha': lambda d: d['source'].update(sha256='b'*64),
            'conversion': lambda d: d.update(conversion='RGB-Lanczos-VAE'),
            'video_rows': lambda d: d.update(video_guide_rows=7),
            'audio_rows': lambda d: d.update(audio_guide_rows=0),
            'frames': lambda d: d.update(guide_frames=39),
            'noise_flag': lambda d: d.update(target_noise_unchanged=False),
            'parallel_flag': lambda d: d.update(parallel_implementation_unchanged=False),
            'bool_is_not_integer': lambda d: d.update(guide_frames=True),
            'one_is_not_true_flag': lambda d: d.update(target_noise_unchanged=1),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                write_receipts(self.folder, self.source, [512,512],
                    mutate=lambda d,r: mutation(d) if r == 1 else None)
                with self.assertRaisesRegex(ValueError, 'rank 1'):
                    validate_guide_receipts(self.folder, self.source, [512,512])


class PhaseGateAndLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = load_series()

    def run_missing_receipt_case(self, broken_stage):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            config = dict(segment_seconds=5, h3_two_pass=True, h3_draft_steps=10,
                h3_resample_start=3, size=[512,512], h3_draft_size=[480,480])
            (job/'config.json').write_text(json.dumps(config))
            (job/'requests.json').write_text(json.dumps([dict(num_frames=124)]*2))
            template = job/'template.json'; template.write_text('{}')
            calls, publications, merges = [], [], []
            source = dict(path=str(job/'segment1/continuation-tail.safetensors'),
                          sha256='a'*64, height=512, width=512)
            def run(folder, cfg, request, label, passes):
                folder.mkdir(exist_ok=True)
                calls.append(label)
                if label == 'segment1-final':
                    (folder/'continuation-tail.json').write_text(json.dumps(source))
                if label.startswith('segment2'):
                    write_receipts(folder, source, cfg['size'],
                        omit_rank=1 if label == broken_stage else None)
            def phase(c, stage): return dict(size=c['h3_draft_size'] if stage=='draft' else c['size'])
            runtime = SimpleNamespace(BASE_REQUEST=template, run_roll_phase=run)
            import continuation_core, media
            with patch.object(continuation_core,'read_tail'), patch.object(media,'merge',side_effect=lambda *a: merges.append(a)):
                with self.assertRaisesRegex(ValueError,'rank 1'):
                    self.series.execute(job,runtime,phase,lambda *a:None,lambda *a:publications.append(a))
            expected=['segment1-draft','segment1-final','segment2-draft']
            if broken_stage == 'segment2-final': expected.append('segment2-final')
            self.assertEqual(calls, expected)
            self.assertEqual(publications, [])
            self.assertEqual(merges, [])
            result=json.loads((job/'series-result.json').read_text())
            self.assertEqual(result['returncode'],1)
            self.assertNotIn('published_video',result)

    def test_missing_draft_receipt_blocks_final_merge_and_publish(self):
        self.run_missing_receipt_case('segment2-draft')

    def test_missing_final_receipt_blocks_merge_and_publish(self):
        self.run_missing_receipt_case('segment2-final')

    def test_inherited_runtime_lock_is_used_without_reopening(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock=root/'run.lock';lock.touch()
            runtime=ModuleType('job_runner');runtime.LOCK=lock
            phases=ModuleType('two_pass');phases.phase_config=object();phases.verify_draft=object()
            delivery=ModuleType('native_delivery');delivery.publish=object()
            with lock.open('a') as held, patch.dict(sys.modules,{'job_runner':runtime,'two_pass':phases,'native_delivery':delivery}):
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.dict(os.environ,{'H3_RUN_LOCK_FD':str(held.fileno())}), patch.object(sys,'argv',['run_series.py',str(root)]), patch.object(self.series,'execute') as execute:
                    self.series.main()
                    execute.assert_called_once()
                    os.fstat(held.fileno())  # The worker must not close the supervisor's lock.

    def test_wrong_inherited_runtime_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock=root/'run.lock';lock.touch();other=root/'other.lock'
            runtime=ModuleType('job_runner');runtime.LOCK=lock
            phases=ModuleType('two_pass');phases.phase_config=object();phases.verify_draft=object()
            delivery=ModuleType('native_delivery');delivery.publish=object()
            with other.open('a') as held, patch.dict(sys.modules,{'job_runner':runtime,'two_pass':phases,'native_delivery':delivery}):
                with patch.dict(os.environ,{'H3_RUN_LOCK_FD':str(held.fileno())}), patch.object(sys,'argv',['run_series.py',str(root)]), patch.object(self.series,'execute') as execute:
                    with self.assertRaisesRegex(RuntimeError,'inherited H3 runtime lock'):
                        self.series.main()
                    execute.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
