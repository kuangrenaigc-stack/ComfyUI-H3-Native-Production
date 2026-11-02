"""Recover only verified publication after successful descendant cleanup."""
import hashlib
import json
from pathlib import Path

def published_after_cleanup(job,videos):
    try:
        supervisor=json.loads((job/'supervisor-result.json').read_text())
        if supervisor.get('cleanup_complete') is not True or supervisor.get('cancelled'):
            return False
        if supervisor.get('worker_returncode')!=0:return False
        result_path=job/('series-result.json' if (job/'series-result.json').exists() else 'stage-result.json')
        result=json.loads(result_path.read_text())
        if result.get('returncode')!=0:return False
        item=result['published_video'];output=Path(videos)/item['filename']
        if output.resolve().parent!=Path(videos).resolve():return False
        digest=hashlib.sha256()
        with output.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
        if digest.hexdigest()!=item['sha256']:return False
        (job/'delivery-warning.json').write_text(json.dumps(dict(
            reason='Publication succeeded; supervisor cleaned residual descendants',
            published_video=item,cleanup_complete=True),indent=2))
        return True
    except (OSError,ValueError,KeyError,TypeError):return False
