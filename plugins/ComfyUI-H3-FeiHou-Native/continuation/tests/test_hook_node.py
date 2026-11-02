"""Exercise plugin hooks and node contracts without loading a model or ComfyUI."""
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

import numpy as np
import torch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))
from continuation_core import SCHEMA, sha256, read_tail
from safetensors.torch import save_file


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def module(name,**attrs):
    m=ModuleType(name)
    m.__dict__.update(attrs)
    return m


class HookContracts(unittest.TestCase):
    def test_final_latent_conversion_and_direct_path_before_scheduler(self):
        calls=[]
        class Scheduler:
            def prepare(self): calls.append('prepare')
        class VAE:
            def decode(self,v):
                calls.append(('decode',tuple(v.shape)))
                return torch.ones(1,3,22,64,64)*.5
            def encode_condition(self,pixels,video):
                calls.append(('encode',tuple(pixels.shape)))
                return torch.ones(1,24,7,pixels.shape[-2]//16,pixels.shape[-1]//16)
        class Runner:
            def _prepare_resampling_source(self): calls.append('existing-resample')
            def _ensure_vae_loaded(self): calls.append('vae-ready')
            def set_vae_decode_tile_shape(self): pass
            def run_vae_decoder(self,*a): calls.append('output-decode')
            @staticmethod
            def _video_to_uint8_frames(v): return (v[0].permute(1,2,3,0)*255).round().to(torch.uint8)
        def resize(frames,h,w,**kwargs):
            self.assertFalse(kwargs['parallel'], 'Only rank zero resizes; never enter a collective here')
            return np.zeros((22,h,w,3),dtype=np.uint8)
        stubs={
            'lightx2v.models.schedulers.minimax_h3.scheduler':module('scheduler',MiniMaxH3Scheduler=Scheduler),
            'lightx2v.models.runners.minimax_h3.minimax_h3_runner':module('runner',MiniMaxH3Runner=Runner),
            'lightx2v.utils.utils':module('utils',resize_resampling_frames=resize)}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,stubs):
            job=Path(d);tail=job/'tail.safetensors'
            metadata=dict(schema=SCHEMA,fps=24,overlap_frames=22,height=64,width=64,source_frames=124)
            save_file(dict(video=torch.ones(1,24,7,4,4),audio=torch.ones(2,32,37)),str(tail),metadata={'contract':json.dumps(metadata)})
            source=dict(path=str(tail),sha256=sha256(tail))
            mapping=job/'mapping.json';mapping.write_text(json.dumps({str(job):source}))
            with patch.dict(os.environ,{'H3_BENCH_JOB':str(job),'H3_CONTINUATION_MAP':str(mapping),'RANK':'0'}):
                hook=load('isolated_hook',PLUGIN/'worker/guide_hook.py')
                attached=[]
                hook.attach_guide=lambda s,v,a: attached.append((v.clone(),a.clone())) or {'guide_frames':22}
                hook.install()
                r=Runner();r.video_vae=VAE();r.request_height=r.request_width=32;r.config={}
                r._prepare_resampling_source();Scheduler().prepare()
                self.assertEqual([c for c in calls if isinstance(c,tuple)], [('decode',(1,24,7,4,4)),('encode',(1,3,22,32,32))])
                self.assertEqual(tuple(attached[-1][0].shape),(1,24,7,2,2))
                self.assertFalse((job/'guide-pixels.safetensors').exists())
                receipt=json.loads((job/'continuation-guide-rank0.json').read_text())
                self.assertEqual(receipt['source'],source)
                self.assertEqual(receipt['conversion'],'RGB-Lanczos-VAE')
                calls.clear();r.request_height=r.request_width=64
                r._prepare_resampling_source();Scheduler().prepare()
                self.assertEqual(calls,['existing-resample','prepare'])
                self.assertEqual(tuple(attached[-1][0].shape),(1,24,7,4,4))
                self.assertEqual(json.loads((job/'continuation-guide-rank0.json').read_text())['conversion'],'direct-final-latent')
                calls.clear();mapping.write_text('{}')
                r._prepare_resampling_source();Scheduler().prepare()
                self.assertEqual(calls,['existing-resample','prepare'])

    def test_node_accepts_existing_reference_socket_and_preserves_original_bytes(self):
        management=module('comfy.model_management')
        stubs={'comfy':module('comfy',model_management=management), 'comfy.model_management':management,
            'comfy_api':module('comfy_api'),
            'comfy_api.input_impl':module('comfy_api.input_impl',VideoFromFile=object),
            'comfy_api.latest':module('comfy_api.latest',ui=SimpleNamespace(),io=SimpleNamespace())}
        with patch.dict(sys.modules,stubs):
            node=load('isolated_node',PLUGIN/'__init__.py')
            cls=node.H3NativeContinuation
            self.assertEqual(cls.INPUT_TYPES()['required']['references'][0],'H3_REFERENCES')
            self.assertIs(cls.VALIDATE_INPUTS(first_prompt='first',continuation_prompt='next',width=736,height=1280,segment_seconds=15,seed=1,timeout_minutes=240),True)
            self.assertIsNot(cls.VALIDATE_INPUTS(first_prompt='first',continuation_prompt='',width=736,height=1280,segment_seconds=15,seed=1,timeout_minutes=240),True)
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);source=root/'original.png';source.write_bytes(b'original PNG bytes: no recompression')
                batch=SimpleNamespace(_h3_original_path=str(source),_h3_original_sha256=sha256(source))
                job=root/'job';job.mkdir()
                refs,receipts=node.archive_references({'images':[batch]},job)
                self.assertEqual(Path(refs['image_path']).read_bytes(),source.read_bytes())
                self.assertEqual(receipts[0]['sha256'],sha256(source))
                source.write_bytes(b'changed')
                with self.assertRaises(ValueError):node.archive_references({'images':[batch]},job)


if __name__=='__main__':unittest.main(verbosity=2)
