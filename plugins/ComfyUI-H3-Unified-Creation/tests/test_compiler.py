import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('h3_unified_test_package', ROOT/'__init__.py',
                                            submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
compiler = sys.modules[SPEC.name+'.compiler']


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'examples/two-scenes.json').read_text())

    def compile(self):
        return compiler.compile_script(json.dumps(self.data, ensure_ascii=False))

    def bad(self):
        with self.assertRaises(compiler.ScriptError):
            self.compile()

    def test_actor_123_then_134_and_stable_subjects(self):
        p = self.compile()
        self.assertEqual(p['segments'][0]['slots']['images'], [1,2,3,5,7])
        self.assertEqual(p['segments'][1]['slots']['images'], [1,3,4,5,6,8,9])
        self.assertEqual(p['segments'][1]['subject_labels']['actor3'], '<Subject 3>')
        self.assertEqual(p['segments'][1]['reference_labels']['face3'], '<Picture 2>')
        self.assertNotIn('actor2', p['segments'][1]['subject_labels'])

    def test_change_scene_at_global_20_without_sixth_shot(self):
        p = self.compile()['segments'][1]
        self.assertEqual(p['timeline'][1]['nominal_global_start'], 20)
        self.assertIn('sceneB', p['timeline'][1]['subjects'])
        self.assertNotIn('sceneB', p['timeline'][0]['subjects'])
        self.assertEqual(p['prompt'].count('the shot cuts to'), 2)
        self.assertIn('At 00:05.000', p['prompt'])

    def test_prop_only_second_segment(self):
        p = self.compile()['segments']
        self.assertNotIn('phone_image', p[0]['reference_labels'])
        self.assertIn('phone_image', p[1]['reference_labels'])
        self.assertNotIn('phone', p[1]['timeline'][0]['subjects'])

    def test_voices_keep_global_speakers_and_local_audio(self):
        p = self.compile()
        self.assertEqual(p['speakers'], {'actor3':'S1','actor1':'S2','actor4':'S3'})
        self.assertIn('<Audio 1> is the voice-timbre/delivery reference for <Subject 4> (S3)', p['segments'][1]['prompt'])
        self.assertEqual(p['segments'][0]['slots']['audios'], [1])
        self.assertEqual(p['segments'][1]['slots']['audios'], [2])

    def test_motion_video_routed_only_where_used(self):
        p = self.compile()['segments']
        self.assertEqual(p[0]['slots']['videos'], [1])
        self.assertEqual(p[1]['slots']['videos'], [])

    def test_two_separate_storyboards(self):
        p = self.compile()['segments']
        self.assertIn('board1', p[0]['reference_labels'])
        self.assertNotIn('board2', p[0]['reference_labels'])
        self.assertNotIn('board1', p[1]['reference_labels'])
        self.assertIn('panel 03 to [Shot 3]', p[1]['prompt'])

    def test_original_objects_not_reencoded_or_mutated(self):
        p = self.compile()
        objects = {'images':[object() for _ in range(9)], 'videos':[object()], 'audios':[object(),object()]}
        before = {k:tuple(v) for k,v in objects.items()}
        selected = compiler.allocate(p, objects)
        self.assertIs(selected[1]['images'][1], objects['images'][2])
        self.assertIs(selected[1]['audios'][0], objects['audios'][1])
        self.assertEqual(before, {k:tuple(v) for k,v in objects.items()})

    def test_all_reference_categories_optional_for_compile(self):
        self.data = {'version':1,'segments':[{'id':'part1','shots':[{'id':'s1','at':0,'description':'An empty landscape.'}]}]}
        p = self.compile()
        self.assertEqual(p['segments'][0]['slots'], {'images':[],'videos':[],'audios':[]})
        self.assertEqual(compiler.allocate(p, None), [{'images':[],'videos':[],'audios':[]}])

    def test_no_storyboard_no_audio_no_video_is_valid(self):
        self.data = json.loads((ROOT/'examples/minimal.json').read_text())
        p = self.compile()['segments'][0]
        self.assertIsNone(p['storyboard'])
        self.assertEqual(p['slots']['audios'], [])
        self.assertEqual(p['slots']['videos'], [])

    def test_exact_dialogue_and_no_translation(self):
        value = '  对比起 （稍微停顿）是我辜负了你！ {{not_a_placeholder}}  '
        self.data['segments'][1]['shots'][0]['dialogue'][0]['text'] = value
        prompt = self.compile()['segments'][1]['prompt']
        self.assertIn('<d>[Cantonese] '+value+'</d>', prompt)

    def test_exact_six_fields_no_bgm(self):
        p = self.compile()['segments'][0]['prompt']
        fields = ['subject_definitions:', 'summary:', 'retention_analysis:', 'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:']
        self.assertEqual(sorted(fields, key=p.index), fields)
        self.assertTrue(p.endswith('non_diegetic_music:\nN/A\n'))
        self.assertNotIn('[Shot 1] At', p)

    def test_first_dialogue_order_not_cast_order(self):
        self.assertEqual(self.compile()['speakers']['actor3'], 'S1')

    def test_wrong_voice_actor_rejected(self):
        self.data['segments'][1]['shots'][1]['dialogue'][0]['voice'] = 'voice3'
        self.bad()

    def test_unused_reference_does_not_force_missing_file(self):
        self.data['references'].append({'id':'unused','media':'image','slot':100,'role':'style'})
        p = self.compile()
        compiler.allocate(p, {'images':list(range(9)), 'videos':[1], 'audios':[1,2]})

    def test_missing_actual_slot_rejected(self):
        with self.assertRaises(compiler.ScriptError):
            compiler.allocate(self.compile(), {'images':[1], 'audios':[], 'videos':[]})

    def test_voice_missing_binding(self):
        del self.data['references'][9]['subject']
        self.bad()

    def test_reference_to_absent_actor_rejected(self):
        self.data['segments'][0]['shots'][0]['uses'] = ['walk3']
        self.bad()

    def test_undeclared_placeholder_rejected(self):
        self.data['segments'][0]['shots'][0]['description'] = '{{actor4}} appears early.'
        self.bad()

    def test_storyboard_order_mismatch_rejected(self):
        self.data['segments'][0]['storyboard']['shots'] = ['a2','a1','a3']
        self.bad()

    def test_no_extra_storyboard_panel(self):
        self.data['segments'][0]['storyboard']['shots'].append('a4')
        self.bad()

    def test_new_audio_category_not_required(self):
        self.data['segments'][0]['shots'][1].pop('dialogue')
        self.data['segments'][1]['shots'][1]['dialogue'][0].pop('voice')
        self.assertEqual(self.compile()['segments'][0]['slots']['audios'], [])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(compiler.ScriptError):
            compiler.compile_script('{"version":1,"version":1,"segments":[]}')

    def test_bad_time_values(self):
        for value in [True, float('nan'), float('inf'), -1, 15, '5', 0]:
            with self.subTest(value=value):
                self.setUp();self.data['segments'][0]['shots'][1]['at'] = value;self.bad()

    def test_first_time_must_be_zero(self):
        self.data['segments'][0]['shots'][0]['at'] = 1
        self.bad()

    def test_duplicate_shot_id_across_segments(self):
        self.data['segments'][1]['shots'][0]['id'] = 'a1'
        self.bad()

    def test_unsupported_segment_count(self):
        self.data['segments'].append(copy.deepcopy(self.data['segments'][0]))
        self.bad()

    def test_unsupported_duration(self):
        self.data['segments'][0]['duration'] = 20
        self.bad()

    def test_does_not_accept_handwritten_picture_numbers(self):
        self.data['segments'][0]['shots'][0]['description'] = 'Use <Picture 8>.'
        self.bad()

    def test_duplicate_slot_reuses_single_model_picture(self):
        self.data['references'].append({'id':'style_alias','media':'image','slot':1,'role':'style'})
        self.data['segments'][0]['shots'][0]['uses'] = ['style_alias']
        p = self.compile()['segments'][0]
        self.assertEqual(p['reference_labels']['style_alias'], p['reference_labels']['face1'])
        self.assertEqual(p['slots']['images'], [1,2,3,5,7])

    def test_native_limits_are_per_segment_not_whole_project(self):
        for i in range(10,15):
            self.data['references'].append({'id':'style'+str(i),'media':'image','slot':i,'role':'style'})
            self.data['segments'][0]['shots'][0].setdefault('uses', []).append('style'+str(i))
        self.bad()

    def test_malformed_types_report_script_error(self):
        for key,value in [('media',[]),('subject',[]),('role',[]),('slot',True)]:
            with self.subTest(key=key):
                self.setUp();self.data['references'][0][key]=value;self.bad()

    def test_model_tag_in_dialogue_rejected(self):
        self.data['segments'][0]['shots'][1]['dialogue'][0]['text']='Hello</d>'
        self.bad()

    def test_no_embedded_shot_or_section_markers(self):
        for value in ['A smile.\n[Shot 9] Run.', 'non_diegetic_music: piano', 'hello\rworld', '[Shot 2] jump']:
            with self.subTest(value=value):
                self.setUp();self.data['segments'][0]['shots'][0]['description']=value;self.bad()
                if not value.startswith('non_diegetic_music:'):
                    self.setUp();self.data['segments'][0]['shots'][1]['dialogue'][0]['text']=value;self.bad()

    def test_literal_field_word_in_dialogue_is_preserved(self):
        value='Summary: we lost.'
        self.data['segments'][0]['shots'][1]['dialogue'][0]['text']=value
        self.assertIn('<d>[English] '+value+'</d>',self.compile()['segments'][0]['prompt'])

    def test_all_line_controls_and_nul_rejected(self):
        for separator in ['\x00','\x0b','\x0c','\x1c','\x1d','\x1e','\x85','\u2028','\u2029']:
            with self.subTest(separator=repr(separator)):
                self.setUp();self.data['segments'][0]['shots'][1]['dialogue'][0]['text']='hello'+separator+'there';self.bad()

    def test_description_has_valid_style_preamble(self):
        p=self.compile()['segments'][0]['prompt'].split('detailed_description:\n')[1]
        self.assertTrue(p.startswith(self.data['style']))
        self.assertIn('\n[Shot 1] ',p)

    def test_offscreen_voice_does_not_import_appearance(self):
        self.data['segments'][0]['shots'][0]['dialogue']=[dict(speaker='actor4',language='English',text='Hello.',offscreen=True,voice='voice4')]
        p=self.compile()['segments'][0]
        self.assertNotIn('face4',p['reference_labels'])
        self.assertIn('voice4',p['reference_labels'])
        self.assertIn('actor4',p['timeline'][0]['offscreen_speakers'])
        self.assertNotIn('actor4',p['timeline'][0]['visible_subjects'])
        self.assertIn('off screen',p['prompt'])

    def test_same_actor_different_costume_by_segment(self):
        self.data['references'][0]['segments']=['part1']
        self.data['references'].append(dict(id='costumeB',media='image',slot=10,role='appearance',subject='actor1',segments=['part2']))
        p=self.compile()['segments']
        self.assertIn('face1',p[0]['reference_labels']);self.assertNotIn('costumeB',p[0]['reference_labels'])
        self.assertIn('costumeB',p[1]['reference_labels']);self.assertNotIn('face1',p[1]['reference_labels'])

    def test_inactive_scope_cannot_be_used_explicitly(self):
        self.data['references'][11]['segments']=['part2']
        self.bad()

    def test_unknown_scope_rejected(self):
        self.data['references'][0]['segments']=['part3'];self.bad()

    def test_ambiguous_audio_slot_binding_rejected(self):
        self.data['references'][10]['slot']=1;self.bad()

    def test_board_cannot_share_appearance_slot(self):
        self.data['references'][6]['slot']=1;self.bad()

    def test_large_integer_errors_stay_structured(self):
        self.data['segments'][0]['shots'][1]['at']=10**350;self.bad()
        with self.assertRaises(compiler.ScriptError):compiler.compile_script('{"version":'+'9'*5000+'}')

    def test_overlap_cut_rejected(self):
        self.data['segments'][1]['shots'][1]['at']=0.5;self.bad()

    def test_duplicate_millisecond_cut_rejected(self):
        self.data['segments'][0]['shots'][2]['at']=5.00001;self.bad()

    def test_end_rounding_rejected(self):
        self.data['segments'][0]['shots'][2]['at']=14.9996;self.bad()

    def test_visual_video_policy_is_explicit(self):
        self.assertIn('soundtrack is excluded',self.compile()['segments'][0]['prompt'])

    def test_collector_keeps_gaps_and_metadata_objects(self):
        image=object()
        material=compiler.collect_materials(images={'图片1':image,'图片3':image})
        self.assertIs(material['images'][0],image);self.assertIsNone(material['images'][1])
        self.assertIs(material['images'][2],image)
        self.data=json.loads((ROOT/'examples/minimal.json').read_text());self.data['references'][0]['slot']=2
        with self.assertRaises(compiler.ScriptError):compiler.allocate(self.compile(),material)

    def test_collector_supports_more_than_nine_project_images(self):
        material=compiler.collect_materials(images={f'图片{i}':object() for i in range(1,19)})
        self.assertEqual(len(material['images']),18)
        with self.assertRaises(compiler.ScriptError):compiler.collect_materials(images={'图片19':object()})

    def test_collector_total_capacity(self):
        with self.assertRaises(compiler.ScriptError):
            compiler.collect_materials(images={f'图片{i}':object() for i in range(1,19)},videos={f'视频{i}':object() for i in range(1,7)},audios={'音频1':object()})


class NodeTests(unittest.TestCase):
    setUp = CompilerTests.setUp
    def test_two_segment_dispatch_one_backend_call_and_exact_objects(self):
        seen = []
        class Backend:
            def generate(self, **kwargs):
                seen.append(kwargs)
                return {'ui':{'test':[]}, 'result':('video','server-path')}
        refs = {'images':[object() for _ in range(9)],'videos':[object()],'audios':[object(),object()]}
        with patch.object(PACKAGE, 'backend_class', return_value=Backend):
            result = PACKAGE.H3UnifiedCreation().generate(refs, json.dumps(self.data))
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0]['segment_seconds'],15)
        self.assertIs(seen[0]['continuation_references']['images'][1], refs['images'][2])
        self.assertNotIn('actor2', json.loads(result['result'][2])['segments'][1]['subject_labels'])
        self.assertEqual(result['result'][:2], ('video','server-path'))

    def test_single_dispatch(self):
        self.data=json.loads((ROOT/'examples/minimal.json').read_text());seen=[]
        class Backend:
            def generate(self,**kwargs):seen.append(kwargs);return {'result':('video','path')}
        with patch.object(PACKAGE,'backend_class',return_value=Backend):
            PACKAGE.H3UnifiedCreation().generate({'images':[object()]},json.dumps(self.data))
        self.assertEqual(seen[0]['duration_seconds'],15)
        self.assertNotIn('continuation_references',seen[0])

    def test_missing_visual_input_does_not_invent_reference(self):
        self.data={'version':1,'segments':[{'id':'one','shots':[{'id':'s1','at':0,'description':'Clouds.'}]}]}
        with patch.object(PACKAGE,'backend_class') as backend:
            with self.assertRaises(compiler.ScriptError):PACKAGE.H3UnifiedCreation().generate({},json.dumps(self.data))
        backend.assert_not_called()

    def test_missing_dependency_clear_message(self):
        with patch.dict(sys.modules,{'nodes':types.SimpleNamespace(NODE_CLASS_MAPPINGS={})}):
            with self.assertRaisesRegex(RuntimeError,'not loaded'):PACKAGE.backend_class(2)

    def test_private_segment_backend_needs_no_retired_node(self):
        class Single:
            pass
        Single.__module__ = 'private_h3_backend_test'
        segment = object()
        fake = types.ModuleType('private_h3_backend_test.continuation')
        fake.H3NativeSegmentInputs = segment
        registry = {'H3FeiHouNativeComplete': Single}
        with patch.dict(sys.modules, {
                'nodes': types.SimpleNamespace(NODE_CLASS_MAPPINGS=registry),
                fake.__name__: fake}):
            self.assertIs(PACKAGE.backend_class(1), Single)
            self.assertIs(PACKAGE.backend_class(2), segment)
        self.assertEqual(set(registry), {'H3FeiHouNativeComplete'})

    def test_geometry_and_linked_validation(self):
        self.assertIs(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(),True)
        self.assertIs(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(width=704,height=1280),True)
        self.assertIsNot(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(width=736),True)
        self.assertIsNot(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(width=2048,height=1088),True)
        self.assertIsNot(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(width=640,height=640),True)
        self.assertIsNot(PACKAGE.H3UnifiedCreation.VALIDATE_INPUTS(script='bad',width=None),True)

    def test_delegate_respects_backend_validation(self):
        class Backend:
            @classmethod
            def VALIDATE_INPUTS(cls,prompt):return 'rejected by fixture'
            def generate(self,**kwargs):raise AssertionError('must not generate')
        with self.assertRaisesRegex(compiler.ScriptError,'rejected by fixture'):
            PACKAGE.delegate(Backend(),{'prompt':'hello','references':{}})

    def test_preview_is_output_node_and_returns_visible_text(self):
        self.assertTrue(PACKAGE.H3UnifiedPreview.OUTPUT_NODE)
        result=PACKAGE.H3UnifiedPreview().preview(json.dumps(self.data))
        self.assertEqual(len(result['ui']['text']),2)
        self.assertEqual(len(result['result']),2)


if __name__=='__main__':unittest.main()
