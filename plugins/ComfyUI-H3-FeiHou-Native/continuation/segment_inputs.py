"""Explicit segment inputs: no material inference, shared fallback or rewriting."""


def archive_segments(job, references, archive):
    if len(references) != 2:
        raise ValueError('需要兩段獨立素材。')
    outputs, receipts = [], []
    for number, refs in enumerate(references, 1):
        if not isinstance(refs, dict):
            raise ValueError(f'第{number}段未連接自己的素材入口。')
        material = {}
        for kind in ('images', 'videos', 'audios'):
            values = refs.get(kind, [])
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'第{number}段素材格式錯誤：{kind}')
            material[kind] = list(values)
        folder = job/'inputs'/f'segment{number}'
        folder.mkdir(parents=True)
        request, images = archive(material, folder)
        outputs.append(request)
        receipts.append(dict(segment=number, original_images=images))
    return outputs, receipts
