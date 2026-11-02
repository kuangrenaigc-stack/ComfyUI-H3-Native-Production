import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('h3_installer',REPO/'tools/install.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        base=Path(self.tmp.name)
        self.data={k:str(base/k) for k in installer.PATH_KEYS}
        self.p=installer.settings(self.data)

    def test_plan_renders_and_validates_all_sources(self):
        files=installer.plan(self.p)
        self.assertGreater(len(files),1000)
        config=json.loads(files[self.p['runtime_root']/'configs/feihou-ref2va-native10plus5.json'])
        self.assertEqual(config['infer_steps']-config['h3_resample_start'],5)
        self.assertEqual(config['lora_configs'][0]['path'],str(self.p['lora_file']))
        self.assertEqual(config['feihou_precision_manifest'],str(self.p['precision_manifest']))
        self.assertFalse(any('/opt/h3/runtime' in text for text in files.values()))

    def test_overlapping_and_checkout_destinations_rejected(self):
        for k,value in [('support_root',self.data['runtime_root']),('runtime_root',str(REPO))]:
            with self.subTest(k=k),self.assertRaises(ValueError):installer.settings(dict(self.data,**{k:value}))

    def test_install_into_empty_directory_and_refuse_overwrite(self):
        installer.install(self.p,installer.plan(self.p))
        self.assertTrue((self.p['runtime_root']/'.venv').is_symlink())
        self.assertTrue((self.p['runtime_root']/'source/lightx2v/__init__.py').is_file())
        self.assertTrue((self.p['state_dir']/'installation.json').is_file())
        with self.assertRaises(ValueError):installer.install(self.p,{})

    def test_reject_unsafe_source_substitution(self):
        with self.assertRaises(ValueError):installer.settings(dict(self.data,runtime_root="/tmp/a'b"))

if __name__=='__main__':unittest.main()
