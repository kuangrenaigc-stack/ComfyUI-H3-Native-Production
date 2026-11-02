import importlib
import json
import unittest
from test_compiler import PACKAGE, compiler

Materials=importlib.import_module(PACKAGE.__name__+'.materials').H3UnifiedMaterials


class MaterialSliderTests(unittest.TestCase):
    def test_eighteen_images_split_into_two_independent_nine_image_segments(self):
        data=dict(version=1,
            subjects=[dict(id=f'actor{i}',kind='actor',description=f'adult actor {i}') for i in range(1,19)],
            references=[dict(id=f'face{i}',media='image',slot=i,role='appearance',subject=f'actor{i}') for i in range(1,19)],
            segments=[dict(id=f'part{part}',duration=15,shots=[dict(id=f'shot{part}',at=0,
                subjects=[f'actor{i}' for i in indices],description='A steady group portrait.')])
                for part,indices in [(1,range(1,10)),(2,range(10,19))]])
        plan=compiler.compile_script(json.dumps(data))
        objects=[object() for _ in range(18)]
        refs=Materials.execute(image_count=18,images={f'图片{i}':item for i,item in enumerate(objects,1)}).result[0]
        assigned=compiler.allocate(plan,refs)
        self.assertEqual(plan['segments'][0]['slots']['images'],list(range(1,10)))
        self.assertEqual(plan['segments'][1]['slots']['images'],list(range(10,19)))
        self.assertEqual(assigned[0]['images'],objects[:9])
        self.assertEqual(assigned[1]['images'],objects[9:])

    def test_schema_keeps_old_inputs_optional_and_doubles_each_medium(self):
        schema=Materials.define_schema()
        inputs={i.id:i for i in schema.inputs}
        for name,cap in [('image_count',18),('video_count',6),('audio_count',6)]:
            self.assertEqual(inputs[name].max,cap)
            self.assertTrue(inputs[name].optional)
            self.assertEqual(inputs[name].display_mode.value,'slider')

    def test_visibility_counts_never_truncate_or_compact_objects(self):
        one,high,video,audio=object(),object(),object(),object()
        refs=Materials.execute(images={'图片1':one,'图片18':high},videos={'视频6':video},
            audios={'音频6':audio},image_count=0,video_count=0,audio_count=0).result[0]
        self.assertEqual(len(refs['images']),18)
        self.assertIs(refs['images'][0],one)
        self.assertTrue(all(v is None for v in refs['images'][1:17]))
        self.assertIs(refs['images'][17],high)
        self.assertIs(refs['videos'][5],video)
        self.assertIs(refs['audios'][5],audio)
        plan={'segments':[{'id':'part2','slots':{'images':[18],'videos':[6],'audios':[6]}}]}
        selected=compiler.allocate(plan,refs)[0]
        self.assertIs(selected['images'][0],high)
        self.assertIs(selected['videos'][0],video)
        self.assertIs(selected['audios'][0],audio)
        with self.assertRaises(compiler.ScriptError):
            compiler.allocate({'segments':[{'id':'empty','slots':{'images':[2]}}]},refs)

    def test_invalid_counts_and_combined_capacity_still_fail(self):
        for kwargs in [{'image_count':19},{'video_count':7},{'audio_count':7},
                       {'image_count':-1},{'image_count':True}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(compiler.ScriptError):
                Materials.execute(**kwargs)
        with self.assertRaises(compiler.ScriptError):
            Materials.execute(image_count=18,video_count=6,audio_count=6,
                images={f'图片{i}':object() for i in range(1,19)},
                videos={f'视频{i}':object() for i in range(1,7)},audios={'音频1':object()})
