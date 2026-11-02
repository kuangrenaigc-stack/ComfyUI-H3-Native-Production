"""CPU contracts for schedule selection and both canonical segment requests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path('/opt/h3/runtime')
ADAPTER=ROOT.parent/'research/learned-latent-upscale-20261027/adapter'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
phases=load('four_step_policy_phases',ROOT/'production/two_pass.py')
acceleration=load('four_step_policy_acceleration',ROOT/'production/acceleration.py')


class TurboPolicyTests(unittest.TestCase):
    def test_live_defaults_are_native10_then_exact_five_of_eight(self):
        c=json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
        self.assertEqual(c['infer_steps'],8)
        self.assertEqual(c['h3_resample_start'],3)
        self.assertTrue(c['lora_configs'][0]['path'].endswith('minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors'))
        self.assertEqual(c['lora_configs'][0]['alpha'],8)
        draft=phases.phase_config(c,'draft');final=phases.phase_config(c,'final')
        for cfg in (draft,final):acceleration.validate_sampling_config(cfg)
        self.assertEqual((draft['infer_steps'],draft['lora_configs'],draft['lora_dynamic_apply']),(10,[],False))
        self.assertEqual((final['infer_steps'],final['h3_cache_start_step']),(8,3))
        self.assertEqual((final['video_flow_shift'],final['audio_flow_shift']),(12.,3.))
        plan=phases.sampling_plan(c)
        self.assertEqual((plan['draft_steps'],plan['final_schedule_steps'],plan['final_steps']),(10,8,5))
        self.assertFalse(plan['final_cfg'])

    def test_old_eight_step_plan_remains_valid_but_three_step_schedule_is_rejected(self):
        c=json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
        c.update(infer_steps=8,h3_resample_start=3)
        self.assertEqual(phases.sampling_plan(c)['final_steps'],5)
        acceleration.validate_sampling_config(phases.phase_config(c,'final'))
        c['infer_steps']=3
        with self.assertRaises(ValueError):phases.phase_config(c,'final')
        c.update(infer_steps=4,h3_resample_start=4)
        with self.assertRaises(ValueError):phases.phase_config(c,'final')

    def test_both_canonical_segments_receive_start_three_and_eight_step_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);base=job/'base.json';base.write_text('{}')
            config=json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
            config.update(size=[128,128],h3_draft_size=[64,64],segment_seconds=5)
            (job/'config.json').write_text(json.dumps(config))
            (job/'requests.json').write_text(json.dumps([dict(num_frames=124,prompt='first'),dict(num_frames=124,prompt='second')]))
            calls=[]
            safe=types.ModuleType('adapter_safety')
            safe.ROOT=ROOT;safe.PLUGIN=ROOT
            safe.runtime=lambda:types.SimpleNamespace(BASE_REQUEST=base)
            safe.read=lambda p:json.loads(Path(p).read_text())
            safe.atomic_json=lambda p,d:Path(p).write_text(json.dumps(d))
            safe.clean_request=lambda p:p
            safe.bind_job=lambda *a:'CPU-dispatch-fixture'
            safe.verify_clean=lambda *a:None
            safe.upscale_ready=lambda *a:True
            safe.check_sources=lambda *a:[]
            safe.run_locked=lambda *a:None
            def phase(runner,folder,cfg,req,label,passes,*deps):
                folder.mkdir(parents=True,exist_ok=True)
                calls.append((label,cfg,req));passes.append({'label':label,'test_only':True})
                (folder/'generated.mp4').write_bytes(b'CPU dispatch fixture, never published')
                (folder.parent/'learned.run.json' if label.endswith('-draft') else folder/'learned.run.json').write_text('{}')
                (folder/'continuation-tail.json').write_text(json.dumps(dict(path=str(folder/'fixture.latent'),sha256='fixture')))
            safe.run_phase=phase
            latent=types.ModuleType('latent_contract');latent.sha=lambda p:'fixture'
            core=types.ModuleType('continuation_core')
            core.plan=lambda seconds:dict(frames_per_segment=124)
            core.read_tail=lambda *args:None
            receipt=types.ModuleType('receipt_contract');receipt.validate_guide_receipts=lambda *a:[]
            duration=types.ModuleType('duration_plan');duration.validate_plan=lambda *a:None
            media=types.ModuleType('media')
            def merge(first,second,output):
                Path(output).write_bytes(b'CPU merge fixture');return {'test_only':True}
            media.merge=merge
            delivery=types.ModuleType('native_delivery');delivery.publish=lambda *a:dict(test_only=True)
            modules={'adapter_safety':safe,'latent_contract':latent,'two_pass':phases,
                'continuation_core':core,'receipt_contract':receipt,'duration_plan':duration,
                'media':media,'native_delivery':delivery}
            with patch.dict(sys.modules,modules):
                run=load('four_step_canonical_series',ADAPTER/'run_series_learned.py')
                run.execute(job)
            self.assertEqual([x[0] for x in calls],['segment1-draft','segment1-final','segment2-draft','segment2-final'])
            for label,cfg,req in calls:
                if label.endswith('-final'):
                    self.assertEqual((cfg['infer_steps'],cfg['h3_cache_start_step'],req['resample_start_step']),(8,3,3))
                else:self.assertEqual((cfg['infer_steps'],cfg['lora_configs']),(10,[]))
            result=json.loads((job/'series-result.json').read_text())
            self.assertEqual(result['returncode'],0)
            self.assertTrue(result['published_video']['test_only'])

if __name__=='__main__':unittest.main()
