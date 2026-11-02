"""Validate known widget constants, then all resolved inputs at execution."""
FIELDS = ('first_prompt', 'continuation_prompt', 'width', 'height', 'segment_seconds', 'seed', 'timeout_minutes')


def validate_values(values, *, complete=False):
    if complete and any(name not in values or values[name] is None for name in FIELDS):
        return '生成輸入尚未解析完整。'
    known = {k: v for k, v in values.items() if k in FIELDS and v is not None}
    for name in ('first_prompt', 'continuation_prompt'):
        if name in known and (not isinstance(known[name], str) or not known[name].strip()):
            return '請填寫兩段完整提示詞。'
    for name in ('width', 'height'):
        if name in known:
            n = known[name]
            if type(n) is not int or n < 32 or n % 32:
                return '寬高必須是32的倍數。'
            if n > 1920:
                return '超出現有生成入口支援的尺寸。'
    if 'width' in known and 'height' in known:
        w, h = known['width'], known['height']
        if min(w, h) > 1280 or w*h > 1920*1088:
            return '超出現有生成入口支援的尺寸。'
    for name, low, high, message in (
        ('segment_seconds', 5, 15, '每段須為5至15秒。'),
        ('seed', 0, 2147483647, '種子超出範圍。'),
        ('timeout_minutes', 5, 360, '等待上限須為5至360分鐘。')):
        if name in known and (type(known[name]) is not int or not low <= known[name] <= high):
            return message
    return True
