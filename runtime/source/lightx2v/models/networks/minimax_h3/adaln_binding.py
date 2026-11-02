"""Bind AdaLN cache values to verified checkpoint bytes; no inference arithmetic.

Bindings are made only after a new build or an explicit independently verified
adoption. Inference never creates/blesses a missing binding. Unchanged files use
inode/size/mtime/ctime checks; changed files must pass the recorded SHA256.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

SCHEMA = 1
BINDING_FILE = 'model-binding.json'
_REHASHED = set()


class CacheBindingError(RuntimeError):
    pass


def _fail(message):
    raise CacheBindingError('H3_ADALN_MODEL_BINDING: ' + message + '. Rebuild the cache for this checkpoint; do not reuse or auto-bind an old cache.')


def _stat(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while data := handle.read(16 * 1024**2):
            digest.update(data)
    return digest.hexdigest()


def _record(path):
    before = _stat(path)
    value = _sha256(path)
    if _stat(path) != before:
        _fail('file changed while hashing: ' + str(path))
    return dict(sha256=value, stat=before)


def _root(config):
    root = Path(config['dit_original_ckpt']).expanduser().resolve(strict=True)
    if config.get('dit_quantized'):
        quant = Path(config['dit_quantized_ckpt']).expanduser().resolve(strict=True)
        if quant != root:
            _fail('AdaLN and inference checkpoints resolve to different paths')
    return root


def _files(root):
    if root.is_file():
        return {root.name: root}
    result = {str(p.relative_to(root)): p for p in root.rglob('*.safetensors') if p.is_file()}
    for pattern in ('*.safetensors.index.json', 'config.json', 'alignment-manifest.json'):
        result.update({p.name: p for p in root.glob(pattern) if p.is_file()})
    if not any(name.endswith('.safetensors') for name in result):
        _fail('checkpoint has no safetensors shards')
    return dict(sorted(result.items()))


def _fingerprint(records):
    contents = {name: dict(sha256=r['sha256'], size=r['stat']['size']) for name, r in sorted(records.items())}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint_snapshot(config):
    root = _root(config)
    before = _files(root)
    records = {name: _record(path) for name, path in before.items()}
    if set(_files(root)) != set(before):
        _fail('checkpoint shard set changed while hashing')
    precision = config.get('feihou_precision_manifest')
    if precision and Path(precision).expanduser().resolve(strict=True) != root / 'alignment-manifest.json':
        _fail('precision manifest is outside the checkpoint package')
    return dict(checkpoint_root=str(root), checkpoint_sha256=_fingerprint(records), files=records)


def _check_file(path, record):
    try:
        current = _stat(path)
        if current == record['stat']:
            return 'stat'
        identity = (str(path), tuple(sorted(current.items())), record['sha256'])
        if identity not in _REHASHED:
            if _sha256(path) != record['sha256'] or _stat(path) != current:
                _fail('content hash mismatch: ' + str(path))
            _REHASHED.add(identity)
        return 'sha256'
    except OSError as exc:
        _fail('missing or unreadable bound file: ' + str(path) + ' (' + str(exc) + ')')


def _verify_checkpoint(config, snapshot):
    root = _root(config)
    if str(root) != snapshot['checkpoint_root']:
        _fail('checkpoint path differs from the cache binding')
    if _fingerprint(snapshot['files']) != snapshot['checkpoint_sha256']:
        _fail('invalid checkpoint fingerprint in binding')
    paths = _files(root)
    if set(paths) != set(snapshot['files']):
        _fail('checkpoint file set differs from the cache binding')
    precision = config.get('feihou_precision_manifest')
    if precision and Path(precision).expanduser().resolve(strict=True) != root / 'alignment-manifest.json':
        _fail('precision manifest differs from the bound checkpoint package')
    counts = {'stat': 0, 'sha256': 0}
    for name, path in paths.items():
        counts[_check_file(path, snapshot['files'][name])] += 1
    return counts


def write_cache_binding(config, cache_path, snapshot, *, provenance):
    """Seal a just-built or independently numerically verified cache, never at load."""
    cache_path = Path(cache_path).resolve(strict=True)
    target = cache_path / BINDING_FILE
    if target.exists():
        _fail('refusing to overwrite an existing cache binding')
    _verify_checkpoint(config, snapshot)
    record = dict(schema_version=SCHEMA, **snapshot, cache_root=str(cache_path),
                  cache_files={name: _record(cache_path / name) for name in ('manifest.json', 'adaln_cache.safetensors')},
                  provenance=provenance, created_unix_seconds=time.time())
    fd, temporary = tempfile.mkstemp(prefix='.model-binding-', dir=cache_path)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic create, without replacing a concurrent binding.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    verify_cache_binding(config, cache_path)
    return target


def verify_cache_binding(config, cache_path):
    cache_path = Path(cache_path).resolve(strict=True)
    try:
        record = json.loads((cache_path / BINDING_FILE).read_text())
        if record['schema_version'] != SCHEMA or record['cache_root'] != str(cache_path):
            _fail('unsupported or relocated cache binding')
        if set(record['cache_files']) != {'manifest.json', 'adaln_cache.safetensors'}:
            _fail('cache binding has an incomplete file set')
        checks = _verify_checkpoint(config, record)
        for name, expected in record['cache_files'].items():
            mode = _check_file(cache_path / name, expected)
            checks[mode] += 1
        return dict(passed=True, checkpoint_sha256=record['checkpoint_sha256'],
                    checked_files=sum(checks.values()), stat_checks=checks['stat'], sha256_checks=checks['sha256'])
    except CacheBindingError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail('missing or invalid model-binding.json (' + str(exc) + ')')
