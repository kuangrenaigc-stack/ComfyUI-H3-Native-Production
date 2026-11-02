"""Use the current canonical learned handoff, inside our existing supervisor."""
import importlib.util
from pathlib import Path
import sys
import json
import os
if __package__:
    from .package_binding import bind
else:
    from package_binding import bind

ADAPTER=Path('/opt/h3/support/adapter')


def draft_geometry(width, height):
    if any(type(n) is not int or n < 64 or n % 64 for n in (width, height)):
        raise ValueError('2倍潛空間放大要求最終寬高為64的倍數；一採寬高各為一半，不會自動改尺寸。')
    if not 1080 <= max(width,height) <= 1920:
        raise ValueError('成片長邊須介於1080至1920；以64對齊尺寸為準。')
    if min(width,height)>1280 or width*height>1920*1088:
        raise ValueError('超出現有短邊或總像素安全範圍。')
    return [height//2, width//2]


def execute(job):
    # Do not invoke its CLI/lock wrapper: our supervisor already holds the
    # control/runtime locks until all TP2 descendants exit.
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    binding=bind(job)
    here=ADAPTER.resolve(strict=True)
    sys.path.insert(0,str(here))
    path=here/'run_series_learned.py'
    spec=importlib.util.spec_from_file_location('h3_segment_learned_series',path)
    module=importlib.util.module_from_spec(spec)
    # A clean worker must resolve these from the adapter's declared PLUGIN,
    # not a previously imported bundled copy.
    for name in ('continuation_core','receipt_contract','media'):
        if name in sys.modules:
            raise RuntimeError('Continuation module already imported before learned dispatch: '+name)
    spec.loader.exec_module(module)
    import adapter_safety
    for name in ('adapter_safety','latent_contract'):
        if Path(sys.modules[name].__file__).resolve()!=here/(name+'.py'):
            raise RuntimeError('Unexpected learned module origin: '+name)
    for name in ('continuation_core','receipt_contract','media'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=adapter_safety.PLUGIN/(name+'.py'):
            raise RuntimeError('Unexpected continuation module origin: '+name)
    try:
        return module.execute(job)
    finally:
        result=Path(job)/'series-result.json'
        if result.exists():
            data=json.loads(result.read_text());data['node_package_binding']=binding
            from adapter_safety import atomic_json
            atomic_json(result,data)
