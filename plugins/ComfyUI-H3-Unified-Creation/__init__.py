"""Independent unified-input frontend; delegates to installed stable H3 nodes."""
import json
import inspect
from .compiler import ScriptError, compile_script, allocate
from .visual_video import visual_only
WEB_DIRECTORY = './web'
__version__ = '0.3.0'


def controls(width=None, height=None, seed=None, timeout_minutes=None, total_seconds=None):
    for key, value in (('width', width), ('height', height)):
        if value is not None and (type(value) is not int or value < 64 or value % 64):
            raise ScriptError(key+': final dimensions must be multiples of 64; no automatic resizing')
    if width is not None and height is not None:
        if not 1080 <= max(width, height) <= 1920 or min(width, height) > 1280 or width*height > 1920*1088:
            raise ScriptError('Final geometry exceeds the installed H3 size policy')
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2147483647):
        raise ScriptError('Invalid seed')
    if timeout_minutes is not None and (type(timeout_minutes) is not int or not 5 <= timeout_minutes <= 360):
        raise ScriptError('Timeout must be 5..360 minutes')
    if total_seconds is not None and (type(total_seconds) is not int or
            not (total_seconds == 0 or 5 <= total_seconds <= 30)):
        raise ScriptError('Total duration must be 5..30 whole seconds, or 0 to use the script')


def duration_input():
    # Optional preserves older saved graphs with no total-duration widget.
    return {'total_seconds': ('INT', {'default': 0, 'min': 0, 'max': 30, 'step': 1,
        'tooltip': '0=沿用腳本；選5～30整秒指定總時長（1～4不支援）。≤15秒一段，>15秒兩段。指定時控件優先，鏡頭起點不會改寫。原生幀數可能略有出入。'})}


def backend_class(segments):
    import importlib
    import nodes
    if segments not in (1, 2):
        raise ValueError('H3 supports one or two native segments')
    single = nodes.NODE_CLASS_MAPPINGS.get('H3FeiHouNativeComplete')
    if single is None:
        raise RuntimeError('Required stable node H3FeiHouNativeComplete is not loaded. Keep ComfyUI-H3-FeiHou-Native installed and restart ComfyUI.')
    if segments == 1:
        return single
    # The segmented backend is private: no retired editor node is required.
    backend = importlib.import_module('.continuation', package=single.__module__)
    return backend.H3NativeSegmentInputs


def delegate(implementation, kwargs):
    if 'duration_plan' in kwargs:
        signature = inspect.signature(implementation.generate)
        if 'duration_plan' not in signature.parameters and not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            raise ScriptError('Stable H3 backend needs duration_plan support; install the matching duration update')
    validator = getattr(type(implementation), 'VALIDATE_INPUTS', None)
    if validator is not None:
        signature = inspect.signature(validator)
        accepts_all = any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        values = kwargs if accepts_all else {k:v for k,v in kwargs.items() if k in signature.parameters}
        valid = validator(**values)
        if valid is not True:
            raise ScriptError('Stable H3 backend rejected this plan: '+str(valid))
    return implementation.generate(**kwargs)


class H3UnifiedPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'script': ('STRING', {'multiline': True, 'default': '',
                    'tooltip': 'One JSON creation script. Each segment has its own duration. Reference types are optional.'})},
                'optional': duration_input()}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('分段完整提示词', '素材分配与编号映射')
    FUNCTION = 'preview'
    CATEGORY = 'H3 / 统一创作 V2'
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, script=None, total_seconds=None):
        try:
            controls(total_seconds=total_seconds)
        except ScriptError as exc:
            return str(exc)
        if script is None:
            return True
        try:
            compile_script(script, total_seconds)
            return True
        except ScriptError as exc:
            return str(exc)

    def preview(self, script, total_seconds=None):
        plan = compile_script(script, total_seconds)
        prompts = '\n\n'.join('=== '+s['id']+' ===\n'+s['prompt'] for s in plan['segments'])
        mapping = json.dumps(plan, ensure_ascii=False, indent=2)
        return {'ui': {'text': [prompts, mapping]}, 'result': (prompts, mapping)}


class H3UnifiedCreation:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'references': ('H3_REFERENCES', {'tooltip': 'One shared collector; only declared per-segment slots are selected.'}),
            'script': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'The complete structured creation script, not two separate prompt fields.'}),
            'width': ('INT', {'default': 768, 'min': 64, 'max': 1920, 'step': 64}),
            'height': ('INT', {'default': 1280, 'min': 64, 'max': 1920, 'step': 64}),
            'seed': ('INT', {'default': 2026102701, 'min': 0, 'max': 2147483647, 'control_after_generate': True}),
            'timeout_minutes': ('INT', {'default': 240, 'min': 5, 'max': 360})},
            'optional': duration_input()}

    RETURN_TYPES = ('VIDEO', 'STRING', 'STRING')
    RETURN_NAMES = ('编号成片', '服务器成片路径', '素材分配与编号映射')
    FUNCTION = 'generate'
    CATEGORY = 'H3 / 统一创作 V2'
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, script=None, width=None, height=None, seed=None, timeout_minutes=None, total_seconds=None):
        try:
            controls(width, height, seed, timeout_minutes, total_seconds)
            if script is not None:
                compile_script(script, total_seconds)
            return True
        except ScriptError as exc:
            return str(exc)

    def generate(self, references, script, width=768, height=1280, seed=2026102701, timeout_minutes=240, total_seconds=None):
        controls(width, height, seed, timeout_minutes, total_seconds)
        plan = compile_script(script, total_seconds)
        materials = allocate(plan, references)
        # All reference categories are optional. The current full-reference
        # backend nevertheless requires a visual input for each rendered segment.
        for segment, selected in zip(plan['segments'], materials):
            if not selected['images'] and not selected['videos']:
                raise ScriptError(segment['id']+': the current full-reference backend requires an image or video. Script-only preview remains available; no absent optional category needs to be filled.')
        materials = visual_only(materials)
        impl = backend_class(len(materials))()
        if len(materials) == 1:
            result = delegate(impl,dict(references=materials[0], prompt=plan['segments'][0]['prompt'],
                                   width=width, height=height, duration_seconds=plan['segments'][0]['duration'],
                                   seed=seed, timeout_minutes=timeout_minutes))
        else:
            result = delegate(impl,dict(first_references=materials[0], first_prompt=plan['segments'][0]['prompt'],
                                   continuation_references=materials[1], continuation_prompt=plan['segments'][1]['prompt'],
                                   width=width, height=height, segment_seconds=plan['segments'][0]['duration'],
                                   duration_plan=plan['duration_plan'],
                                   seed=seed, timeout_minutes=timeout_minutes))
        if not isinstance(result, dict) or len(result.get('result', ())) != 2:
            raise RuntimeError('Stable H3 backend returned an unsupported output contract')
        return dict(result, result=tuple(result['result'])+(json.dumps(plan, ensure_ascii=False, indent=2),))


NODE_CLASS_MAPPINGS = {'H3UnifiedCreationV2': H3UnifiedCreation, 'H3UnifiedPreviewV2': H3UnifiedPreview}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3UnifiedCreationV2': 'H3 V2 · 统一素材与创作脚本',
    'H3UnifiedPreviewV2': 'H3 V2 · 解析预览（不生成）'}

# Pure compiler tests can import without the ComfyUI SDK. A real ComfyUI
# process must provide this module and will register the native autogrow node.
try:
    from .materials import H3UnifiedMaterials
except ModuleNotFoundError as exc:
    if exc.name != 'comfy_api' and not str(exc.name).startswith('comfy_api.'):
        raise
else:
    NODE_CLASS_MAPPINGS['H3UnifiedMaterialsV2'] = H3UnifiedMaterials
    NODE_DISPLAY_NAME_MAPPINGS['H3UnifiedMaterialsV2'] = 'H3 V2 · 统一参考素材'
