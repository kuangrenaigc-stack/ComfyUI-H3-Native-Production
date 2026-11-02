import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from input_validation import validate_values

BASE=dict(first_prompt='first',continuation_prompt='next',width=736,height=1280,segment_seconds=15,seed=1,timeout_minutes=240)

class Inputs(unittest.TestCase):
    def test_unknown_links_defer_until_execution(self):
        self.assertIs(validate_values(dict(BASE,first_prompt=None,width=None)),True)
        self.assertIsNot(validate_values(dict(BASE,first_prompt=None,width=None),complete=True),True)

    def test_bad_constants_not_hidden_by_other_links(self):
        self.assertIsNot(validate_values(dict(BASE,first_prompt=None,height=1279)),True)
        self.assertIsNot(validate_values(dict(BASE,first_prompt=None,segment_seconds=16)),True)

    def test_resolved_values_get_complete_validation(self):
        self.assertIs(validate_values(BASE,complete=True),True)
        for field,value in [('first_prompt',' '),('first_prompt',['node',0]),('width',None),('seed',True),('height',2000)]:
            with self.subTest(field=field,value=value):
                self.assertIsNot(validate_values(dict(BASE,**{field:value}),complete=True),True)

if __name__=='__main__':unittest.main(verbosity=2)
