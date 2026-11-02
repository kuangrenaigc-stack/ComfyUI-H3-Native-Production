"""Real ComfyUI loader and graph validation in a separate CPU-only process."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys

COMFY=Path('/opt/h3/comfyui')
PLUGIN=Path(os.environ.get('H3_TEST_PLUGIN_PATH',Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(COMFY))
sys.argv=[sys.argv[0],'--cpu']
from comfy.cli_args import args
args.cpu=True
import nodes
import execution


class ReferencesForValidation:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{}}
    RETURN_TYPES=('H3_REFERENCES',)
    FUNCTION='output'
    def output(self):raise RuntimeError('Validation only; must never execute')


class StringForValidation:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{}}
    RETURN_TYPES=('STRING',)
    FUNCTION='output'
    def output(self):raise RuntimeError('Validation only')

class IntForValidation(StringForValidation):
    RETURN_TYPES=('INT',)


async def main():
    loaded=await nodes.load_custom_node(str(PLUGIN),module_parent='continuation_validation')
    assert loaded
    assert 'H3FeiHouNativeComplete' in nodes.NODE_CLASS_MAPPINGS
    assert 'H3OriginalImage' in nodes.NODE_CLASS_MAPPINGS
    assert 'H3FeiHouNativeSegments' not in nodes.NODE_CLASS_MAPPINGS
    unified=PLUGIN.parent/'ComfyUI-H3-Unified-Creation'
    assert await nodes.load_custom_node(str(unified),module_parent='unified_validation')
    v2=nodes.NODE_CLASS_MAPPINGS['H3UnifiedCreationV2']
    backend=sys.modules[v2.__module__].backend_class(2)
    assert backend is sys.modules[backend.__module__].H3NativeSegmentInputs
    nodes.NODE_CLASS_MAPPINGS['H3InternalBackendValidationOnly']=backend
    assert 'H3FeiHouNativeContinuation' not in nodes.NODE_CLASS_MAPPINGS
    nodes.NODE_CLASS_MAPPINGS['ContinuationTestReferences']=ReferencesForValidation
    graph={'1':{'class_type':'ContinuationTestReferences','inputs':{}},
        '2':{'class_type':'H3InternalBackendValidationOnly','inputs':dict(first_references=['1',0],continuation_references=['5',0],
            first_prompt='first complete prompt',continuation_prompt='next complete prompt',
            width=768,height=1280,segment_seconds=15,seed=1,timeout_minutes=240)}}
    graph['5']={'class_type':'ContinuationTestReferences','inputs':{}}
    nodes.NODE_CLASS_MAPPINGS['ContinuationTestString']=StringForValidation
    nodes.NODE_CLASS_MAPPINGS['ContinuationTestInt']=IntForValidation
    graph['3']={'class_type':'ContinuationTestString','inputs':{}}
    graph['4']={'class_type':'ContinuationTestInt','inputs':{}}
    results={}
    cases=[('constants',{},True),
        ('linked_first',{'first_prompt':['3',0]},True),
        ('linked_both_prompts',{'first_prompt':['3',0],'continuation_prompt':['3',0]},True),
        ('linked_geometry',{'width':['4',0],'height':['4',0]},True),
        ('linked_seed',{'seed':['4',0]},True),
        ('bad_constant_with_link',{'first_prompt':['3',0],'height':1279},False),
        ('blank_prompt_with_link',{'first_prompt':['3',0],'continuation_prompt':' '},False),
        ('wrong_socket_type',{'width':['3',0]},False),
        ('wrong_second_material',{'continuation_references':['3',0]},False),
        ('reject_736',{'width':736},False),
        ('exact_half_704',{'width':704},True),
        ('segment_below_1080',{'width':1024,'height':576},False),
        ('segment_max_1920',{'width':1920,'height':1088},True),
        ('segment_max_1920_portrait',{'width':1088,'height':1920},True)]
    for label,updates,expected in cases:
        current=copy.deepcopy(graph)
        current['2']['inputs'].update(updates)
        result=await execution.validate_prompt('continuation-validation-'+label,current,None)
        assert result[0] is expected,(label,result)
        results[label]=result[0]
    single_cases=[('single_1920',1920,1088,True),('single_portrait_1920',1088,1920,True),
                  ('single_over_1920',2048,768,False),('single_nonhalf_736',1280,736,False),('single_below_1080',1024,576,False),('single_area_over',1920,1280,False),
                  ('single_short_over',1344,1344,False)]
    for label,width,height,expected in single_cases:
        current={'1':graph['1'], '2':{'class_type':'H3FeiHouNativeComplete',
            'inputs':dict(references=['1',0],prompt='complete prompt',width=width,height=height,
                          duration_seconds=15,seed=1,timeout_minutes=120)}}
        result=await execution.validate_prompt('single-validation-'+label,current,None)
        assert result[0] is expected,(label,result)
        results[label]=result[0]
    for label,updates in [('single_linked_prompt',{'prompt':['3',0]}),
                          ('single_linked_geometry',{'width':['4',0],'height':['4',0]})]:
        current={'1':graph['1'],'3':graph['3'],'4':graph['4'],
                 '2':{'class_type':'H3FeiHouNativeComplete',
                      'inputs':dict(references=['1',0],prompt='complete prompt',width=1472,height=832,
                                    duration_seconds=15,seed=1,timeout_minutes=120)}}
        current['2']['inputs'].update(updates)
        result=await execution.validate_prompt(label,current,None)
        assert result[0] is True,(label,result)
        results[label]=result[0]
    print('COMFY_VALIDATION '+json.dumps(dict(loaded=loaded,cases=results,passed=len(results))))



if __name__=='__main__':asyncio.run(main())
