"""Fail-closed verification of the two TP2 continuation conditioning receipts."""
import json
from pathlib import Path


def validate_guide_receipts(folder, source, phase_size):
    """Require matching source, conversion and guide geometry from both ranks.

    This is a publication/phase gate, not an instruction to rerun sampling.
    A missing or inconsistent receipt aborts before the next phase or merge.
    """
    if not isinstance(phase_size, (list, tuple)) or len(phase_size) != 2 or any(
            type(n) is not int or n < 32 or n % 32 for n in phase_size):
        raise ValueError('Invalid continuation phase canvas')
    if not isinstance(source, dict) or not isinstance(source.get('path'), str) or not isinstance(source.get('sha256'), str):
        raise ValueError('Invalid expected continuation source')
    source_size = (source.get('height'), source.get('width'))
    if any(type(n) is not int or n < 32 or n % 32 for n in source_size):
        raise ValueError('Expected continuation source has invalid geometry')
    expected_path = Path(source['path']).resolve()
    expected_conversion = ('direct-final-latent' if tuple(phase_size) == source_size
                           else 'RGB-Lanczos-VAE')
    expected = dict(conversion=expected_conversion, guide_frames=22,
                    video_guide_rows=7 * (phase_size[0] // 32) * (phase_size[1] // 32),
                    audio_guide_rows=74, target_noise_unchanged=True,
                    parallel_implementation_unchanged=True)
    validated = []
    for rank in (0, 1):
        path = Path(folder) / f'continuation-guide-rank{rank}.json'
        try:
            receipt = json.loads(path.read_text())
            if not isinstance(receipt, dict):
                raise ValueError('receipt must be an object')
            actual_source = receipt.get('source')
            if not isinstance(actual_source, dict) or not isinstance(actual_source.get('path'), str):
                raise ValueError('source is missing')
            if Path(actual_source['path']).resolve() != expected_path or actual_source.get('sha256') != source['sha256']:
                raise ValueError('source path or SHA256 differs from segment 1 final tail')
            for key, value in expected.items():
                actual = receipt.get(key)
                if type(actual) is not type(value) or actual != value:
                    raise ValueError(f'{key} differs from the expected guide contract')
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f'Invalid continuation guide receipt for rank {rank}: {path}: {exc}') from exc
        validated.append(dict(rank=rank, receipt_path=str(path), source_path=str(expected_path),
                              source_sha256=source['sha256'], **expected))
    return validated
