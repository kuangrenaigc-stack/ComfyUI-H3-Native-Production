"""Use the stable FeiHou entry and LoRA memory fix without extra profiling."""
from lora_memory import install
install()
from pathlib import Path
import runpy,sys
ROOT=Path('/opt/h3/runtime')
sys.path.insert(0,str(ROOT/'production'))
runpy.run_path(str(ROOT/'production/entry.py'),run_name='__main__')
