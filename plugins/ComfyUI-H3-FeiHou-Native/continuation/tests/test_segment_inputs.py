"""Each segment gets only its explicit uploads, including its own storyboard."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_hook_node import module, load

PLUGIN=Path(__file__).resolve().parents[1]


class SegmentInputTests(unittest.TestCase):
    def node(self):
        management=module('comfy.model_management')
        stubs={'comfy':module('comfy',model_management=management),'comfy.model_management':management,
            'comfy_api':module('comfy_api'),
            'comfy_api.input_impl':module('comfy_api.input_impl',VideoFromFile=object),
            'comfy_api.latest':module('comfy_api.latest',ui=SimpleNamespace(),io=SimpleNamespace())}
        with patch.dict(sys.modules,stubs):
            return load('isolated_segment_node',PLUGIN/'__init__.py')

    def test_independent_references_and_storyboards_reach_requests(self):
        node=self.node()
        class Submitted(Exception):pass
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir()
            (root/'configs/feihou-ref2va-native10plus5.json').write_text('{}')
            def picture(name):
                path=root/name;path.write_bytes(('original '+name).encode())
                return SimpleNamespace(_h3_original_path=str(path),_h3_original_sha256=node.sha(path))
            a,b,c=picture('first.png'),picture('second.png'),picture('first-board.png')
            e=picture('second-board.png')
            first={'images':[a,c],'videos':[],'audios':[]}
            second={'images':[b,e],'videos':[],'audios':[]}
            calls=[]
            def stop(job,timeout):calls.append(job);raise Submitted()
            with patch.object(node,'ROOT',root),patch.object(node.H3NativeContinuation,'_run',side_effect=stop):
                with self.assertRaises(Submitted):
                    node.H3NativeSegmentInputs().generate(first,'first exact dialogue',second,'second exact dialogue')
            requests=json.loads((calls[0]/'requests.json').read_text())
            for request,expected in zip(requests,((a,c),(b,e))):
                paths=request['image_path'].split(',')
                self.assertEqual(len(paths),2)
                self.assertEqual([Path(p).read_bytes() for p in paths],
                    [Path(p._h3_original_path).read_bytes() for p in expected])
            self.assertEqual(requests[0]['prompt'],'first exact dialogue')
            self.assertEqual(requests[1]['prompt'],'second exact dialogue')
            self.assertNotEqual(requests[0]['image_path'],requests[1]['image_path'])
            self.assertEqual(len(first['images']),2,'Must not mutate upstream reference lists')
            receipt=json.loads((calls[0]/'input-receipt.json').read_text())
            self.assertEqual([len(r['original_images']) for r in receipt['segment_inputs']],[2,2])
            config=json.loads((calls[0]/'config.json').read_text())
            self.assertEqual(config['size'],[1280,768])
            self.assertEqual(config['h3_draft_size'],[640,384])
            self.assertEqual(config['h3_handoff'],'learned-clean-latent-2x')
            self.assertIs(config['deliver_exact_30_seconds'],False)

    def test_second_material_is_required_without_implicit_fallback(self):
        node=self.node()
        cls=node.H3NativeSegmentInputs
        schema=cls.INPUT_TYPES()
        self.assertIn('first_references',schema['required'])
        self.assertIn('continuation_references',schema['required'])
        self.assertNotIn('references',schema['required'])
        self.assertNotIn('optional',schema)
        with self.assertRaises(ValueError):cls().generate({},'first',None,'second')

    def test_no_storyboard_cross_segment_fallback(self):
        node=self.node()
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def archive(refs,folder):calls.append(refs);return {},[]
            _,receipt=node.archive_segments(Path(d),({'images':[1,3]},{'images':[2]}),archive)
            self.assertEqual(calls[0]['images'],[1,3])
            self.assertEqual(calls[1]['images'],[2])
            self.assertEqual([r['segment'] for r in receipt],[1,2])

    def test_exact_half_geometry_and_reject_silent_rounding(self):
        node=self.node()
        self.assertEqual(node.draft_geometry(1280,768),[384,640])
        self.assertEqual(node.draft_geometry(704,1280),[640,352])
        self.assertIs(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=768,height=1280),True)
        self.assertIsNot(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=736,height=1280),True)
        with self.assertRaises(ValueError):node.draft_geometry(736,1280)
        with self.assertRaises(ValueError):node.draft_geometry(1024,576)
        self.assertIs(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=None,height=832),True)
        self.assertIsNot(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=1024,height=576),True)
        self.assertIs(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=1920,height=1088),True)
        self.assertIs(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=1088,height=1920),True)
        self.assertIsNot(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=2048,height=768),True)
        self.assertIsNot(node.H3NativeSegmentInputs.VALIDATE_INPUTS(width=768,height=2048),True)

if __name__=='__main__':unittest.main()
