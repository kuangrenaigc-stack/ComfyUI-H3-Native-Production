"""Bind worker package code before any generation/cache reuse."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

def bind(job,root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    code={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*.py')) if 'tests' not in p.relative_to(root).parts}
    path=Path(job)/'node-package-binding.json'
    if path.exists():
        if json.loads(path.read_text())!=code:raise ValueError('Node package changed; use a new job')
    else:
        if any(Path(job).rglob('generated.mp4')) or any(Path(job).rglob('clean.safetensors')):
            raise ValueError('Existing job has no node package binding; use a new job')
        fd,tmp=tempfile.mkstemp(prefix='.node-binding-',dir=job)
        try:
            with os.fdopen(fd,'w') as out:json.dump(code,out,indent=2)
            os.replace(tmp,path)
        finally:Path(tmp).unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()
