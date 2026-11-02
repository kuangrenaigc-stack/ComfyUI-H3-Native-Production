"""Install into new paths only. Never downloads weights or starts ComfyUI."""
import argparse, ast, json, os, shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PATH_KEYS=('comfyui_root','runtime_root','support_root','output_dir','draft_dir','state_dir','cache_dir','model_dir','weights_dir','precision_manifest','runtime_venv','lora_file','upscaler_file')

def settings(data):
    missing=[k for k in PATH_KEYS if not isinstance(data.get(k),str) or not data[k]]
    if missing:raise ValueError('Missing paths: '+', '.join(missing))
    paths={k:Path(data[k]).expanduser().resolve() for k in PATH_KEYS}
    if any(any(c in str(p) for c in "'\"\n\r\0") for p in paths.values()):
        raise ValueError('Paths cannot contain quotes, newlines or NUL')
    targets = [paths['runtime_root'], paths['support_root'], paths['comfyui_root'] / 'custom_nodes']
    for i, target in enumerate(targets):
        if target == REPO or target in REPO.parents or REPO in target.parents:
            raise ValueError('Deployment paths must be outside this source checkout')
        if any(target == other or target in other.parents or other in target.parents for other in targets[i+1:]):
            raise ValueError('Runtime, support and custom_nodes paths must not overlap')
    return paths

def replacements(p):
    return {
        '/opt/h3/runtime':str(p['runtime_root']),
        '/opt/h3/support':str(p['support_root']),
        '/opt/h3/comfyui-parent':str(p['comfyui_root'].parent),
        '/opt/h3/comfyui':str(p['comfyui_root']),
        '/opt/h3/weights':str(p['weights_dir']),
        '/opt/h3/adaln-cache':str(p['cache_dir']),
        '/opt/h3/videos':str(p['output_dir']),
        '/opt/h3/drafts':str(p['draft_dir']),
        '/opt/h3/state':str(p['state_dir']),
        '/opt/h3/loras':str(p['runtime_root']/'lora'),
    }

def render(text,mapping):
    for old,new in sorted(mapping.items(),key=lambda x:-len(x[0])):text=text.replace(old,new)
    return text

def plan(p):
    mapping=replacements(p);files={}
    for section in ('runtime','support','plugins'):
        for src in sorted((REPO/section).rglob('*')):
            if not src.is_file() or '__pycache__' in src.parts:continue
            rel=src.relative_to(REPO/section)
            base=p['comfyui_root']/'custom_nodes' if section=='plugins' else p[section+'_root']
            dest=base/rel;text=render(src.read_text(),mapping)
            if src.suffix=='.py':ast.parse(text)
            if src.suffix=='.json':json.loads(text)
            files[dest]=text
    config_path=p['runtime_root']/'configs/feihou-ref2va-native10plus5.json'
    c=json.loads(files[config_path])
    c['feihou_precision_manifest']=str(p['precision_manifest'])
    c['lora_configs'][0]['path']=str(p['lora_file'])
    files[config_path]=json.dumps(c,indent=2)+'\n'
    return files

def validate_assets(p):
    for k in ('comfyui_root','model_dir','weights_dir','precision_manifest','runtime_venv','lora_file','upscaler_file'):
        if not p[k].exists():raise ValueError(f'{k} does not exist: {p[k]}')
    if not (p['runtime_venv']/'bin/torchrun').is_file():raise ValueError('runtime_venv needs bin/torchrun')
    import hashlib
    expected={'lora_file':'46c64fc94509489f18dd1f545172e30c8df7d4cc50a30e96744409b4288c937d',
              'upscaler_file':'984afb58f11d01274b90d880596ce2f93bc9c512db66fdeecd4e2d99b371d3e4'}
    for k,value in expected.items():
        with p[k].open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=value:
            raise ValueError(k+' is not the tested checkpoint')
    for tool in ('ffmpeg','ffprobe'):
        if not Path('/usr/bin',tool).is_file():
            raise ValueError('/usr/bin/'+tool+' is required by this deployment profile')

def install(p,files):
    targets=[p['runtime_root'],p['support_root']]
    targets += [p['comfyui_root']/'custom_nodes'/d.name for d in (REPO/'plugins').iterdir() if d.is_dir()]
    for target in targets:
        if target.exists() or target.is_symlink():raise ValueError('Refusing to overwrite an existing deployment: '+str(target))
    for d in (p['output_dir'],p['draft_dir'],p['state_dir'],p['cache_dir']):d.mkdir(parents=True,exist_ok=True)
    created=[]
    try:
        for t in targets:t.mkdir(parents=True);created.append(t)
        for dest,text in files.items():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        (p['runtime_root']/'.venv').symlink_to(p['runtime_venv'],target_is_directory=True)
        (p['runtime_root']/'model-bf16').symlink_to(p['model_dir'],target_is_directory=True)
        models=p['support_root']/'models';models.mkdir(exist_ok=True)
        (models/'h3_clean_latent_upscaler_film_epoch200.safetensors').symlink_to(p['upscaler_file'])
        (p['runtime_root']/'jobs').mkdir()
        (p['runtime_root']/'run.lock').touch()
        (p['state_dir']/'installation.json').write_text(json.dumps({k:str(v) for k,v in p.items()},indent=2))
    except BaseException:
        for t in reversed(created):shutil.rmtree(t)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();p=settings(json.loads(args.config.read_text()));files=plan(p)
    if not args.dry_run:validate_assets(p);install(p,files)
    print(json.dumps(dict(status='dry-run' if args.dry_run else 'installed',files=len(files),
        next='Prepare native dependencies and bound AdaLN caches; upload reference assets, then import workflows. Restart ComfyUI manually.'),indent=2))

if __name__=='__main__':main()
