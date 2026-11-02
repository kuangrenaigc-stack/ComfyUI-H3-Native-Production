from pathlib import Path
import runpy
import sys
import os
import json
ROOT=Path('/opt/h3/runtime')
BASE=Path('/opt/h3/comfyui/custom_nodes/ComfyUI-H3-FeiHou-Native/worker')
sys.path.insert(0,str(BASE))
from lora_memory import install as install_memory
install_memory()
phase=json.loads((Path(os.environ['H3_BENCH_JOB'])/'phase-config.json').read_text())
if phase.get('ffn_inplace_activation') and phase.get('h3_sampling_stage')=='turbo-final':
    from ffn_memory import install as install_ffn_memory
    install_ffn_memory()
from handoff_hook import install
install()
if os.environ.get('H3_CONTINUATION_MAP'):
    plugin=Path('/opt/h3/comfyui/custom_nodes/ComfyUI-H3-Native-Continuation')
    sys.path.insert(0,str(plugin/'worker'))
    from guide_hook import install as install_continuation
    install_continuation()
sys.path.insert(0,str(ROOT/'production'))
runpy.run_path(str(ROOT/'production/entry.py'),run_name='__main__')
