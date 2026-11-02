"""Pure native-duration contract, vendored unchanged at the three job boundaries."""
import math

FPS = 24
OVERLAP = 22
MIN_SECONDS = 5
MAX_SECONDS = 15


def native_frames(seconds):
    if type(seconds) is not int or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError('Each native segment must request 5..15 whole seconds')
    return 5 + 17 * math.ceil((seconds * FPS - 5) / 17)


def plan_durations(seconds):
    if not isinstance(seconds, (list, tuple)) or not 1 <= len(seconds) <= 2:
        raise ValueError('One or two native segment durations are required')
    durations = list(seconds)
    frames = [native_frames(n) for n in durations]
    overlap = OVERLAP if len(frames) == 2 else 0
    target = sum(durations) * FPS
    if len(frames) == 2:
        # Retain the first segment's native scheduling. Reserve context in the
        # second segment by choosing the nearest legal H3 frame count. This
        # changes the sampling request, never the speed or length of a render.
        candidates = range(native_frames(MIN_SECONDS), native_frames(MAX_SECONDS) + 1, 17)
        frames[1] = min(candidates, key=lambda n: (
            abs(frames[0] + n - overlap - target),
            frames[0] + n - overlap < target, n))
    output = sum(frames) - overlap
    return dict(schema='h3-native-duration-plan/1', segment_seconds=durations,
                segment_frames=frames, fps=FPS, overlap_frames=overlap,
                nominal_seconds=sum(durations), output_frames=output,
                output_seconds=output / FPS, retimed=False)


def plan_total(total_seconds):
    if type(total_seconds) is not int or not 5 <= total_seconds <= 30:
        raise ValueError('Total duration must be 5..30 whole seconds')
    if total_seconds <= MAX_SECONDS:
        return plan_durations([total_seconds])
    first = min(MAX_SECONDS, total_seconds - MIN_SECONDS)
    return plan_durations([first, total_seconds - first])


def validate_plan(value, segments=None):
    if not isinstance(value, dict):
        raise ValueError('Invalid native duration plan')
    expected = plan_durations(value.get('segment_seconds'))
    if segments is not None and len(expected['segment_seconds']) != segments:
        raise ValueError('Duration plan has the wrong segment count')
    # Reject booleans masquerading as integers and other weak JSON equality.
    if set(value) != set(expected) or any(
            type(value[k]) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError('Native duration plan differs from its sampling contract')
    for key in ('segment_seconds', 'segment_frames'):
        if any(type(n) is not int for n in value[key]):
            raise ValueError('Duration/frame values must be integers')
    return expected
