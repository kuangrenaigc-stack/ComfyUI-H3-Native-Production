"""Dependency-free release checks; no GPU imports or inference."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,text=True)
    files=[ROOT/name for name in tracked.stdout.split('\0') if name] if tracked.returncode==0 and tracked.stdout else [p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('.git','__pycache__'))]
    problems=[];python_count=json_count=0
    private_patterns=[r'/Users/kuang' + r'ren',r'/home/kuang' + r'ren',r'/data1' + r't/',r'192\.168\.\d+\.\d+',r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}']
    for p in files:
        rel=p.relative_to(ROOT)
        if p.suffix.lower() in {'.safetensors','.pt','.pth','.mp4','.mov','.wav','.zip','.gz','.jpg','.png'} or p.name.startswith('.env') or p.name=='settings.local.json':
            problems.append(f'Unexpected artifact: {rel}');continue
        if p.stat().st_size>5*1024**2:problems.append(f'Oversized source: {rel}')
        text=p.read_text(encoding='utf-8')
        if any(re.search(pattern,text) for pattern in private_patterns):problems.append(f'Private identifier or credential pattern: {rel}')
        try:
            if p.suffix=='.py':ast.parse(text,filename=str(rel));python_count+=1
            if p.suffix=='.json':json.loads(text);json_count+=1
        except (SyntaxError,ValueError) as exc:problems.append(f'{rel}: {exc}')
    for p in (ROOT/'workflows').glob('*.json'):
        data=json.loads(p.read_text());ids={n['id'] for n in data['nodes']}
        for link in data['links']:
            if link[1] not in ids or link[3] not in ids:problems.append(f'Broken link in {p.name}: {link[0]}')
    if problems:raise SystemExit('\n'.join(problems))
    print(json.dumps(dict(status='ok',files=len(files),python=python_count,json=json_count)))

if __name__=='__main__':main()
