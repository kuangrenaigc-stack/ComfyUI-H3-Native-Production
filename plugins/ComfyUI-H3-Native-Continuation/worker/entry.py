"""Stable TP2 production entry plus only continuation conditioning/export."""
from pathlib import Path
import runpy
import sys

ROOT = Path('/opt/h3/runtime')
BASE_WORKER = Path('/opt/h3/comfyui/custom_nodes/ComfyUI-H3-FeiHou-Native/worker')
sys.path.insert(0, str(BASE_WORKER))
from lora_memory import install as install_lora_memory
install_lora_memory()
from guide_hook import install
install()
sys.path.insert(0, str(ROOT / 'production'))
runpy.run_path(str(ROOT / 'production/entry.py'), run_name='__main__')
