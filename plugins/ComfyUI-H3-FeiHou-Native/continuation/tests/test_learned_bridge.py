import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

PLUGIN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PLUGIN))
spec=importlib.util.spec_from_file_location('series_learned_routing',PLUGIN/'worker/run_series.py')
series=importlib.util.module_from_spec(spec);spec.loader.exec_module(series)

class LearnedRoutingTests(unittest.TestCase):
    def test_learned_jobs_delegate_without_starting_legacy_rgb_phases(self):
        calls=[]
        bridge=ModuleType('learned_bridge')
        bridge.execute=lambda job:calls.append(job) or 'learned-result'
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'learned_bridge':bridge}):
            job=Path(d)
            (job/'config.json').write_text(json.dumps({'h3_handoff':'learned-clean-latent-2x'}))
            def forbidden(*a,**k):raise AssertionError('Legacy path must not run')
            self.assertEqual(series.execute(job,None,forbidden,forbidden,forbidden),'learned-result')
            self.assertEqual(calls,[job])

if __name__=='__main__':unittest.main()
