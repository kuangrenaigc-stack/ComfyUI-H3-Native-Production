"""Explicit human-review boundary between native H3 draft and final sampling."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from native_delivery import sha, publish_draft


def binding(config, request):
    assets = {}
    for key in ('image_path', 'video_path', 'audio_path'):
        for value in str(request.get(key, '')).split(','):
            if value:
                assets[value] = sha(Path(value))
    return hashlib.sha256(json.dumps(dict(config=config, request=request, assets=assets),
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(job, receipt):
    path = Path(job) / 'draft-review.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    temporary.replace(path)


def ready(job, config, request):
    job = Path(job)
    source = job / 'draft/generated.mp4'
    row = publish_draft(source, job / 'generated.mp4', job.parent.name)
    receipt = dict(status='awaiting_user_approval', draft_sha256=row['sha256'],
        request_binding=binding(config, request), published_draft=row)
    save(job, receipt)
    return receipt


def require_approval(job, config, request, approved_hash):
    path = Path(job) / 'draft-review.json'
    if not approved_hash:
        raise ValueError('Final sampling requires explicit user approval: --approve-draft-sha256 HASH')
    if not path.is_file():
        raise ValueError('No reviewed draft receipt exists; generate the draft first')
    receipt = json.loads(path.read_text())
    if receipt['request_binding'] != binding(config, request):
        raise ValueError('Draft settings or request changed; generate a new draft before approval')
    if receipt['draft_sha256'] != approved_hash:
        raise ValueError('Approval does not identify this draft')
    if sha(Path(job) / 'draft/generated.mp4') != approved_hash:
        raise ValueError('Draft contents changed after review')
    if receipt['status'] == 'finalized':
        raise ValueError('This draft already has a finished video; use a new job for another variant')
    receipt.update(status='approved', approved_draft_sha256=approved_hash,
                   approved_at=datetime.now(timezone.utc).isoformat())
    save(job, receipt)
    return receipt
