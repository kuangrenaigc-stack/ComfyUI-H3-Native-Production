"""Verify the actual package-owned single-entry handoff, without GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

PLUGIN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PLUGIN))
spec=importlib.util.spec_from_file_location('package_learned_pipeline',PLUGIN/'learned_pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)

class LearnedPipelineTests(unittest.TestCase):
    def test_complete_dispatches_native_draft_upscale_and_final_suffix(self):
        calls=[]
        runner=ModuleType('job_runner');runner.VENV=Path('/native-venv')
        review=ModuleType('draft_review')
        phases=ModuleType('two_pass')
        def phase(c,stage):
            return dict(size=c['h3_draft_size'] if stage=='draft' else c['size'],
                        infer_steps=10 if stage=='draft' else 8,
                        h3_cache_start_step=0 if stage=='draft' else 3,
                        lora_configs=[] if stage=='draft' else ['current-lora'])
        phases.phase_config=phase
        contract=ModuleType('latent_contract');contract.sha=lambda p:'bound-sha'
        safe=SimpleNamespace(clean_request=lambda q:{k:v for k,v in q.items() if k!='draft_steps'},
            bind_job=lambda *a:calls.append('bind'),verify_clean=lambda p:calls.append('verify-clean'),
            read=lambda p:{},check_sources=lambda *a:[{'rank':0},{'rank':1}],
            atomic_json=lambda p,data:Path(p).write_text(json.dumps(data)))
        def run_phase(r,folder,config,request,label,passes,*extra):
            calls.append((label,config,request));folder.mkdir(exist_ok=True)
        safe.run_phase=run_phase
        stubs={'job_runner':runner,'draft_review':review,'two_pass':phases,'latent_contract':contract}
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,stubs),patch.dict(os.environ,{'H3_CONTINUATION_MAP':'must-clear'}):
            job=Path(d);summary={}
            config=dict(size=[1088,1920],h3_draft_size=[544,960],h3_resample_start=3,
                        h3_handoff='learned-clean-latent-2x',h3_draft_steps=10,h3_two_pass=True)
            with patch.object(pipeline,'dependencies',return_value=safe),patch.object(safe,'upscale_ready',side_effect=[False,True],create=True),patch.object(pipeline.subprocess,'run') as upscale:
                code=pipeline.run_two_pass(job,config,{'num_frames':362,'draft_steps':2},[],'complete',None,summary)
            self.assertEqual(code,0)
            stages=[c for c in calls if isinstance(c,tuple)]
            self.assertEqual([c[0] for c in stages],['draft','final-suffix'])
            self.assertEqual(stages[0][1]['size'],[544,960])
            self.assertEqual(stages[1][1]['size'],[1088,1920])
            self.assertEqual(stages[0][1]['infer_steps'],10)
            self.assertEqual(stages[0][1]['lora_configs'],[])
            self.assertEqual(stages[1][1]['h3_cache_start_step'],3)
            self.assertNotIn('draft_steps',stages[0][2])
            upscale.assert_called_once()
            self.assertEqual(summary['handoff'],'learned-clean-latent-2x')
            self.assertNotIn('H3_CONTINUATION_MAP',os.environ)
            self.assertTrue(Path(os.environ['H3_LEARNED_SOURCE']).is_file())

if __name__=='__main__':unittest.main()
