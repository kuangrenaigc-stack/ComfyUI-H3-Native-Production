import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_compiler import PACKAGE, compiler, ROOT

duration = __import__(PACKAGE.__name__+'.duration_plan', fromlist=['plan_total'])


def script_for(total):
    count = 1 if total <= 15 else 2
    return dict(version=1, subjects=[dict(id='lead', kind='actor', description='the protagonist')],
                references=[dict(id='face',media='image',slot=1,role='appearance',subject='lead')],
                segments=[dict(id=f'part{i}',shots=[dict(id=f's{i}',at=0,
                    subjects=['lead'],description='Follow {{lead}} in a continuous take.')])
                          for i in range(count)])


class DurationTests(unittest.TestCase):
    def test_every_total_5_through_30_reaches_correct_backend_once(self):
        ref=object()
        for total in range(5,31):
            with self.subTest(total=total):
                seen=[]
                class Backend:
                    def generate(self,**kw):
                        seen.append(kw)
                        return {'result':('video','path')}
                source=json.dumps(script_for(total))
                plan=compiler.compile_script(source,total)
                with patch.object(PACKAGE,'backend_class',return_value=Backend) as dispatch:
                    result=PACKAGE.H3UnifiedCreation().generate(
                        dict(images=[ref]),source,total_seconds=total)
                self.assertEqual(len(seen),1)
                self.assertEqual(dispatch.call_args.args[0],1 if total<=15 else 2)
                self.assertEqual(len(result['result']),3)
                self.assertEqual(plan['nominal_seconds'],total)
                self.assertEqual(sum(s['duration'] for s in plan['segments']),total)
                self.assertFalse(plan['retimed'])
                for n in plan['duration_plan']['segment_frames']:
                    self.assertEqual((n-5)%17,0)
                    self.assertTrue(124<=n<=362)
                if total<=15:
                    self.assertEqual(seen[0]['duration_seconds'],total)
                else:
                    self.assertEqual(seen[0]['duration_plan'],plan['duration_plan'])

    def test_overlap_budget_is_part_of_sampling_not_post_processing(self):
        p=duration.plan_total(20)
        self.assertEqual(p['segment_seconds'],[15,5])
        self.assertEqual(p['segment_frames'],[362,141])
        self.assertEqual(p['output_frames'],481)
        self.assertAlmostEqual(p['output_seconds'],20.0416666667)
        self.assertFalse(p['retimed'])
        self.assertEqual(duration.plan_total(30)['output_frames'],702)

    def test_unequal_script_durations_and_cumulative_markers(self):
        data=script_for(20)
        data['segments'][0]['duration']=12
        data['segments'][1]['duration']=8
        data['segments'][1]['shots'].append(dict(id='new_scene',at=5,
            subjects=['lead'],description='A new view of {{lead}}.'))
        p=compiler.compile_script(json.dumps(data))
        self.assertEqual(p['segments'][1]['nominal_global_start'],12)
        self.assertEqual(p['segments'][1]['timeline'][1]['nominal_global_start'],17)

    def test_shortened_segment_rejects_overrunning_cut_instead_of_moving_it(self):
        data=script_for(20)
        data['segments'][1]['shots'].append(dict(id='late',at=6,
            subjects=['lead'],description='Hold the scene.'))
        with self.assertRaisesRegex(compiler.ScriptError,'within 5 seconds'):
            compiler.compile_script(json.dumps(data),20)

    def test_no_automatic_merging_or_splitting_of_user_shots(self):
        for total,data in [(20,script_for(10)),(10,script_for(20))]:
            with self.subTest(total=total), self.assertRaisesRegex(compiler.ScriptError,'requires'):
                compiler.compile_script(json.dumps(data),total)

    def test_invalid_totals_are_rejected(self):
        for value in (True,False,4,31,20.5,'20'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                duration.plan_total(value)

    def test_old_fifteen_plus_fifteen_scripts_keep_native_output(self):
        default=PACKAGE.duration_input()['total_seconds'][1]['default']
        self.assertEqual(default,0)
        p=compiler.compile_script(json.dumps(script_for(30)),default)
        self.assertEqual(p['duration_plan']['segment_frames'],[362,362])
        self.assertEqual(p['duration_plan']['output_seconds'],29.25)

    def test_short_script_total_cannot_concatenate(self):
        data=script_for(20)
        for segment in data['segments']:segment['duration']=5
        with self.assertRaisesRegex(compiler.ScriptError,'one segment'):
            compiler.compile_script(json.dumps(data))

    def test_malformed_segment_reports_object_error(self):
        with self.assertRaisesRegex(compiler.ScriptError,'expected an object'):
            compiler.compile_script(json.dumps(dict(version=1,segments=['bad'])))

    def test_old_backend_rejected_before_generate(self):
        class OldBackend:
            def generate(self, first_prompt):raise AssertionError('Must fail before calling the old backend')
        with self.assertRaisesRegex(compiler.ScriptError,'duration_plan support'):
            PACKAGE.delegate(OldBackend(),dict(first_prompt='test',duration_plan=duration.plan_total(20)))

    def test_forged_schedule_is_rejected_at_worker_boundary(self):
        for key,value in [('segment_frames',[362,124]),('output_frames',480),
                          ('fps',True),('retimed',True)]:
            p=duration.plan_total(20);p[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                duration.validate_plan(p,2)

    def test_three_boundary_copies_are_identical(self):
        stage=ROOT.parent
        adapter=stage/'adapter/duration_plan.py'
        if not adapter.is_file():
            adapter=ROOT.parents[1]/'support/adapter/duration_plan.py'
        if not adapter.is_file():
            adapter=Path('/opt/h3/support/adapter/duration_plan.py')
        paths=[ROOT/'duration_plan.py',adapter,
               stage/'ComfyUI-H3-FeiHou-Native/continuation/duration_plan.py']
        self.assertEqual(len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),1)


if __name__=='__main__':unittest.main()
