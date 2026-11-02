"""Deterministic, media-free compiler for an explicit H3 creation script.

No model, file access, inference of missing assets, or natural-language splitting.
"""
import json
import math
import re
from .duration_plan import plan_durations, plan_total

ID = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')
TOKEN = re.compile(r'\{\{([A-Za-z][A-Za-z0-9_-]{0,63})\}\}')
MODEL_LABEL = re.compile(r'<(?:Subject|Picture|Video|Audio)\s+\d+>|</?d>')
STRUCTURE = re.compile(r'\[Shot\s+\d+\]|(?:subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*:', re.IGNORECASE)
OVERLAP_SECONDS = 22/24
MEDIA = {'image': ('images', 'Picture', 9), 'video': ('videos', 'Video', 3),
         'audio': ('audios', 'Audio', 3)}
ROLES = {'appearance': {'image', 'video'}, 'voice': {'audio'},
         'motion': {'video'}, 'camera': {'video'}, 'style': {'image', 'video'},
         'ambience': {'audio'}, 'effect': {'audio'}, 'storyboard': {'image'}}


class ScriptError(ValueError):
    pass


def fail(where, message):
    raise ScriptError(f'{where}: {message}')


def obj(value, where, allowed, required=()):
    if not isinstance(value, dict):
        fail(where, 'expected an object')
    unknown, missing = set(value)-set(allowed), set(required)-set(value)
    if unknown:
        fail(where, 'unknown fields: '+', '.join(sorted(unknown)))
    if missing:
        fail(where, 'missing fields: '+', '.join(sorted(missing)))
    return value


def seq(value, where):
    if not isinstance(value, list):
        fail(where, 'expected a list')
    return value


def text(value, where, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        fail(where, 'expected nonempty text' if nonempty else 'expected text')
    return value


def line_content(value, where, nonempty=True, section_markers=True):
    value = text(value, where, nonempty)
    line_break = ''.join(value.splitlines()) != value or any(ord(c) < 32 for c in value)
    marker = STRUCTURE.search(value) if section_markers else re.search(r'\[Shot\s+\d+\]', value, re.IGNORECASE)
    if line_break or marker:
        fail(where, 'use single-line content without embedded shot/section markers; content is rejected, never rewritten')
    return value


def ident(value, where):
    if not isinstance(value, str) or not ID.fullmatch(value):
        fail(where, 'use a stable ASCII identifier, e.g. actor1 or scene_B')
    return value


def ids(values, where):
    result = [ident(v, where) for v in seq(values, where)]
    if len(set(result)) != len(result):
        fail(where, 'duplicate identifiers')
    return result


def prose(value, where, nonempty=True):
    value = line_content(value, where, nonempty)
    if MODEL_LABEL.search(value):
        fail(where, 'use {{stable_id}} rather than handwritten model labels or dialogue tags')
    # A malformed placeholder must not survive into a model prompt.
    if '{{' in TOKEN.sub('', value) or '}}' in TOKEN.sub('', value):
        fail(where, 'malformed {{stable_id}} placeholder')
    return value


def plain(value, where, nonempty=True):
    value = prose(value, where, nonempty)
    if TOKEN.search(value):
        fail(where, 'placeholders belong in shot descriptions/delivery, not global definitions')
    return value


def timestamp(seconds):
    milliseconds = round(seconds*1000)
    return f'{milliseconds//60000:02d}:{milliseconds//1000%60:02d}.{milliseconds%1000:03d}'


def parse(source):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail('script', 'duplicate JSON key: '+key)
            result[key] = value
        return result
    if not isinstance(source, str) or len(source) > 1_000_000:
        fail('script', 'expected JSON text of at most 1 MB')
    try:
        data = json.loads(source, object_pairs_hook=pairs,
                          parse_constant=lambda x: fail('script', 'nonfinite number: '+x))
    except (ValueError, RecursionError) as exc:
        raise ScriptError('script: invalid JSON: '+str(exc)) from exc
    obj(data, 'script', ('version', 'subjects', 'references', 'segments', 'summary', 'style', 'soundscape'),
        ('version', 'segments'))
    if type(data['version']) is not int or data['version'] != 1:
        fail('version', 'only script version 1 is supported')
    return data


def compile_script(source, total_seconds=None):
    data = parse(source)
    if type(total_seconds) is int and total_seconds == 0:
        total_seconds = None
    for key in ('summary', 'style', 'soundscape'):
        plain(data.get(key, ''), key, False)
    subjects, references, subject_numbers = {}, {}, {}
    for index, s in enumerate(seq(data.get('subjects', []), 'subjects'), 1):
        obj(s, 'subject', ('id', 'kind', 'description'), ('id', 'kind', 'description'))
        sid = ident(s['id'], 'subject.id')
        if sid in subjects:
            fail(sid, 'duplicate subject')
        if s['kind'] not in ('actor', 'creature', 'scene', 'prop', 'other'):
            fail(sid, 'unknown subject kind')
        plain(s['description'], sid+'.description')
        subjects[sid], subject_numbers[sid] = s, index
    for r in seq(data.get('references', []), 'references'):
        obj(r, 'reference', ('id', 'media', 'slot', 'role', 'subject', 'description', 'segments'),
            ('id', 'media', 'slot', 'role'))
        rid = ident(r['id'], 'reference.id')
        if rid in references or rid in subjects:
            fail(rid, 'duplicate or ambiguous reference/subject identifier')
        if (not isinstance(r['media'], str) or not isinstance(r['role'], str) or
                r['media'] not in MEDIA or r['role'] not in ROLES or r['media'] not in ROLES[r['role']]):
            fail(rid, 'unsupported media/role combination')
        if type(r['slot']) is not int or r['slot'] < 1:
            fail(rid, 'slot is a positive, one-based integer within its media category')
        target = r.get('subject')
        if target is not None:
            ident(target, rid+'.subject')
        if target is not None and target not in subjects:
            fail(rid, 'unknown subject binding')
        if r['role'] in ('appearance', 'voice', 'motion') and target is None:
            fail(rid, 'this reference role requires an explicit subject binding')
        if r['role'] == 'voice' and subjects[target]['kind'] not in ('actor', 'creature'):
            fail(rid, 'voice must bind to an actor or creature')
        plain(r.get('description', ''), rid+'.description', False)
        if 'segments' in r and not ids(r['segments'], rid+'.segments'):
            fail(rid, 'segments scope must be nonempty when specified')
        references[rid] = r
    segments = seq(data['segments'], 'segments')
    if not 1 <= len(segments) <= 2:
        fail('segments', 'this engine supports one or two native segments')
    for s in segments:
        obj(s, 'segment', ('id','duration','summary','storyboard','shots'), ('id','shots'))
    try:
        duration_plan = (plan_total(total_seconds) if total_seconds is not None else
                         plan_durations([s.get('duration', 15) for s in segments]))
    except ValueError as exc:
        fail('duration', str(exc))
    durations = duration_plan['segment_seconds']
    if len(durations) != len(segments):
        fail('segments', f'{duration_plan["nominal_seconds"]} seconds requires {len(durations)} segment(s); supply that many script segments, without automatic shot rewriting')
    if len(segments) == 2 and sum(durations) <= 15:
        fail('segments', 'Totals up to 15 seconds use one segment; do not concatenate two short segments')
    starts = [0] + durations[:-1]
    segment_ids = set()
    for s in segments:
        obj(s, 'segment', ('id','duration','summary','storyboard','shots'), ('id','shots'))
        segment_ids.add(ident(s['id'], 'segment.id'))
    for rid,r in references.items():
        if set(r.get('segments', []))-segment_ids:
            fail(rid, 'unknown segment scope')
    # Same physical voice cannot silently bind different speakers. An image
    # may intentionally define multiple subjects, as permitted by Ref2VA.
    by_slot = {}
    for rid,r in references.items():
        by_slot.setdefault((r['media'],r['slot']), []).append(r)
    for group in by_slot.values():
        roles = {r['role'] for r in group}
        if 'storyboard' in roles and roles != {'storyboard'}:
            fail('references', 'a storyboard slot cannot simultaneously serve another role')
        if len({r['subject'] for r in group if r['role']=='voice'}) > 1:
            fail('references', 'one audio slot cannot bind multiple actors as an isolated voice reference')
    used_ids, speaker_numbers, prepared = set(), {}, []
    for si, segment in enumerate(segments):
        where = f'segments[{si}]'
        obj(segment, where, ('id', 'duration', 'summary', 'storyboard', 'shots'), ('id', 'shots'))
        seg_id = ident(segment['id'], where+'.id')
        if seg_id in used_ids:
            fail(seg_id, 'segment/shot identifiers must be globally unique')
        used_ids.add(seg_id)
        # An explicit node total overrides nominal durations, not shot times.
        duration = durations[si]
        plain(segment.get('summary', ''), seg_id+'.summary', False)
        shots = seq(segment['shots'], seg_id+'.shots')
        if not shots:
            fail(seg_id, 'at least one continuous shot is required')
        selected_subjects, selected_refs, shot_plans = set(), set(), []
        previous = -1
        for ni, shot in enumerate(shots, 1):
            obj(shot, seg_id+'.shot', ('id', 'at', 'subjects', 'uses', 'description', 'dialogue'),
                ('id', 'at', 'description'))
            shot_id = ident(shot['id'], 'shot.id')
            if shot_id in used_ids:
                fail(shot_id, 'segment/shot identifiers must be globally unique')
            used_ids.add(shot_id)
            at = shot['at']
            if (type(at) not in (int, float) or not 0 <= at < duration or
                    not math.isfinite(at) or at <= previous or (ni == 1 and at != 0)):
                fail(shot_id, f'shot starts must begin at zero and strictly increase within {duration} seconds')
            if round(at*1000) >= duration*1000 or (ni > 1 and round(at*1000) <= round(previous*1000)):
                fail(shot_id, f'shot starts must remain distinct inside {duration} seconds at millisecond precision')
            previous = at
            if si and ni > 1 and at < OVERLAP_SECONDS:
                fail(shot_id, 'new cuts must occur after the 22-frame continuation overlap')
            current_subjects = set(ids(shot.get('subjects', []), shot_id+'.subjects'))
            if current_subjects-set(subjects):
                fail(shot_id, 'unknown subject: '+', '.join(sorted(current_subjects-set(subjects))))
            current_refs = set(ids(shot.get('uses', []), shot_id+'.uses'))
            if current_refs-set(references):
                fail(shot_id, 'unknown reference: '+', '.join(sorted(current_refs-set(references))))
            current_refs.update(rid for rid, r in references.items()
                                if r['role'] == 'appearance' and r.get('subject') in current_subjects
                                and seg_id in r.get('segments', [seg_id]))
            prose(shot['description'], shot_id+'.description')
            dialogue = seq(shot.get('dialogue', []), shot_id+'.dialogue')
            speakers, offscreen = set(), set()
            for event in dialogue:
                obj(event, shot_id+'.dialogue', ('speaker', 'language', 'text', 'delivery', 'voice', 'offscreen'),
                    ('speaker', 'language', 'text'))
                speaker = ident(event['speaker'], shot_id+'.speaker')
                if speaker not in subjects or subjects[speaker]['kind'] not in ('actor', 'creature'):
                    fail(shot_id, 'dialogue speaker must be a participating actor/creature')
                if type(event.get('offscreen',False)) is not bool:
                    fail(shot_id, 'offscreen must be boolean')
                if event.get('offscreen',False):
                    if speaker in current_subjects:
                        fail(shot_id, 'offscreen speaker must not also be listed among visible subjects')
                    offscreen.add(speaker)
                elif speaker not in current_subjects:
                    fail(shot_id, 'visible speaker must be in subjects; use offscreen:true for voice-only speech')
                speakers.add(speaker)
                if speaker not in speaker_numbers:
                    speaker_numbers[speaker] = len(speaker_numbers)+1
                language = text(event['language'], shot_id+'.language')
                if not re.fullmatch(r'[\w -]{1,64}', language):
                    fail(shot_id, 'invalid language/dialect label')
                utterance = line_content(event['text'], shot_id+'.text', section_markers=False)
                if '<d>' in utterance or '</d>' in utterance:
                    fail(shot_id, 'literal dialogue boundary tags are not allowed inside an utterance')
                prose(event.get('delivery', ''), shot_id+'.delivery', False)
                if 'voice' in event:
                    ident(event['voice'], shot_id+'.voice')
                    voice = references.get(event['voice'])
                    if not voice or voice['role'] != 'voice' or voice.get('subject') != speaker:
                        fail(shot_id, 'voice reference must bind to this exact speaker')
                    current_refs.add(event['voice'])
            voices = {}
            for rid in current_refs:
                r = references[rid]
                if seg_id not in r.get('segments', [seg_id]):
                    fail(shot_id, 'reference is outside its declared segment scope: '+rid)
                if r['role'] == 'storyboard':
                    fail(shot_id, 'storyboards belong in the segment storyboard field')
                participants = current_subjects | (offscreen if r['role']=='voice' else set())
                if r.get('subject') is not None and r['subject'] not in participants:
                    fail(shot_id, 'reference is bound to a subject absent from this shot: '+rid)
                if r['role'] == 'voice':
                    if r['subject'] not in speakers:
                        fail(shot_id, 'voice reference requires dialogue by its bound speaker')
                    voices.setdefault(r['subject'], set()).add((r['media'], r['slot']))
            if any(len(v) > 1 for v in voices.values()):
                fail(shot_id, 'ambiguous multiple voice references for one speaker in the same shot')
            allowed = current_subjects | offscreen | current_refs
            for content in [shot['description']]+[e.get('delivery', '') for e in dialogue]:
                unknown = set(TOKEN.findall(content))-allowed
                if unknown:
                    fail(shot_id, 'placeholder not active in this shot: '+', '.join(sorted(unknown)))
            selected_subjects.update(current_subjects | offscreen)
            selected_refs.update(current_refs)
            shot_plans.append(dict(source=shot, subjects=current_subjects | offscreen,
                                   visible_subjects=current_subjects, offscreen=offscreen,
                                   references=current_refs, local_number=ni, nominal_global_start=starts[si]+at))
        board = segment.get('storyboard')
        if board is not None:
            obj(board, seg_id+'.storyboard', ('reference', 'shots'), ('reference', 'shots'))
            rid = ident(board['reference'], seg_id+'.storyboard.reference')
            if rid not in references or references[rid]['role'] != 'storyboard':
                fail(seg_id, 'storyboard must name a declared image/storyboard reference')
            if seg_id not in references[rid].get('segments', [seg_id]):
                fail(seg_id, 'storyboard is outside its declared segment scope')
            if board['shots'] != [s['id'] for s in shots]:
                fail(seg_id, 'one board panel per shot, in exactly this segment shot order')
            selected_refs.add(rid)
        prepared.append(dict(source=segment, subjects=selected_subjects, references=selected_refs,
                             shots=shot_plans, storyboard=board))

    compiled = []
    for si, plan in enumerate(prepared):
        seg, refs = plan['source'], plan['references']
        slots, labels = {}, {}
        for medium, (key, label, maximum) in MEDIA.items():
            slots[key] = sorted({references[rid]['slot'] for rid in refs if references[rid]['media'] == medium})
            if len(slots[key]) > maximum:
                fail(seg['id'], f'native limit exceeded: {maximum} {key} per segment')
            lookup = {slot: f'<{label} {i}>' for i, slot in enumerate(slots[key], 1)}
            labels.update({rid: lookup[references[rid]['slot']] for rid in refs
                           if references[rid]['media'] == medium})
        if sum(map(len, slots.values())) > 12:
            fail(seg['id'], 'native total reference limit is 12 per segment')
        subjects_in_order = sorted(plan['subjects'], key=subject_numbers.__getitem__)
        subject_labels = {sid: f'<Subject {subject_numbers[sid]}>' for sid in subjects_in_order}
        definitions, retention = [], []
        for sid in subjects_in_order:
            appearance = sorted({labels[rid] for rid in refs if references[rid]['role'] == 'appearance'
                                 and references[rid].get('subject') == sid})
            origin = ' Appearance/identity reference: '+', '.join(appearance)+'.' if appearance else ''
            definitions.append(f'{subject_labels[sid]} is {subjects[sid]["description"]}.{origin}')
            numbers = [str(s['local_number']) for s in plan['shots'] if sid in s['subjects']]
            visible = [str(s['local_number']) for s in plan['shots'] if sid in s['visible_subjects']]
            hidden = [str(s['local_number']) for s in plan['shots'] if sid in s['offscreen']]
            if visible:
                retention.append(f'{subject_labels[sid]} (Shots {", ".join(visible)}): fully_preserved - retain the defined identity and relevant design in those shots.')
            if hidden:
                retention.append(f'{subject_labels[sid]} (Shots {", ".join(hidden)}): partially_preserved - voice only, off screen; do not introduce a visible appearance in these shots.')
        ref_order = sorted(refs, key=lambda rid: (list(MEDIA).index(references[rid]['media']),
                                                references[rid]['slot'], rid))
        for rid in ref_order:
            r = references[rid]
            if r['role'] == 'appearance':
                continue
            where = [str(s['local_number']) for s in plan['shots'] if rid in s['references']]
            target = subject_labels.get(r.get('subject'), '')
            extra = ' '+r['description'] if r.get('description') else ''
            if r['role'] == 'storyboard':
                mapping = ', '.join(f'panel {i:02d} to [Shot {i}]' for i in range(1, len(plan['shots'])+1))
                definitions.append(f'{labels[rid]} is this segment\'s storyboard: {mapping}; guide opening compositions only.')
                retention.append(f'{labels[rid]}: partially_preserved - use viewpoint, placement and shot order; original subject references govern identity.')
            elif r['role'] == 'voice':
                definitions.append(f'{labels[rid]} is the voice-timbre/delivery reference for {target} (S{speaker_numbers[r["subject"]]}).{extra}')
                retention.append(f'{labels[rid]} (Shots {", ".join(where)}): reference - follow the bound speaker\'s voice, without copying source dialogue.')
            else:
                definitions.append(f'{labels[rid]} provides {r["role"]} reference'+(f' for {target}' if target else '')+f' in Shots {", ".join(where)}.{extra}')
                if r['media']=='video':
                    definitions[-1] += ' Visual reference only; its soundtrack is excluded from the model input.'
                marker = 'reference' if r['media'] == 'audio' else 'partially_preserved'
                retention.append(f'{labels[rid]} (Shots {", ".join(where)}): {marker} - use only its declared {r["role"]} role.')
        render_labels = dict(subject_labels, **labels)
        def render(value):
            return TOKEN.sub(lambda m: render_labels[m.group(1)], value)
        description = [data.get('style', '')]
        if plan['storyboard']:
            description.append('Omit storyboard borders, gutters and numbers. Retain the target video style; the storyboard is soft composition guidance.')
        if si:
            description.append('Continue the prior final state during the initial approximately 0.92-second overlap context; arrange new dialogue/action after that context. Preserve only continuity appropriate to the described shot.')
        timeline = []
        for s in plan['shots']:
            shot, n = s['source'], s['local_number']
            prefix = '[Shot 1] ' if n == 1 else f'[Shot {n}] At {timestamp(shot["at"])}, the shot cuts to '
            paragraph = prefix+render(shot['description'])
            if si and n==1:
                paragraph += ' During the first 22 frames, retain the overlapping prior state without new dialogue; perform the new scripted action and speech only after that context.'
            for event in shot.get('dialogue', []):
                delivery = render(event.get('delivery', ''))
                speaker = event['speaker']
                paragraph += f' {subject_labels[speaker]} (S{speaker_numbers[speaker]})'
                if event.get('offscreen'):
                    paragraph += ' off screen'
                paragraph += f' {delivery}' if delivery else ' says'
                paragraph += f': <d>[{event["language"]}] {event["text"]}</d>'
            description.append(paragraph)
            timeline.append(dict(id=shot['id'], shot=n, local_start=shot['at'], nominal_global_start=s['nominal_global_start'],
                                 subjects=sorted(s['subjects'], key=subject_numbers.__getitem__),
                                 visible_subjects=sorted(s['visible_subjects'],key=subject_numbers.__getitem__),
                                 offscreen_speakers=sorted(s['offscreen'],key=subject_numbers.__getitem__),
                                 references=sorted(s['references'])))
        summary = seg.get('summary') or data.get('summary') or 'Follow the planned continuous shots in their declared order.'
        sections = [('subject_definitions', '\n'.join(definitions) or 'No external subject references are assigned.'),
                    ('summary', '[reference generation] '+summary),
                    ('retention_analysis', '\n'.join(retention) or 'N/A'),
                    ('detailed_description', '\n'.join(v for v in description if v)),
                    ('overall_soundscape', data.get('soundscape') or 'Natural ambience and physical/nonverbal sounds only.'),
                    ('non_diegetic_music', 'N/A')]
        compiled.append(dict(id=seg['id'], duration=durations[si], nominal_global_start=starts[si], slots=slots,
                             native_frames=duration_plan['segment_frames'][si],
                             subject_labels=subject_labels, reference_labels=labels, timeline=timeline,
                             storyboard=plan['storyboard'], prompt='\n\n'.join(k+':\n'+v for k, v in sections)+'\n'))
    return dict(schema='h3-unified-creation-plan/1', segments=compiled,
                subjects={sid: f'<Subject {n}>' for sid, n in subject_numbers.items()},
                speakers={sid: f'S{n}' for sid, n in speaker_numbers.items()},
                nominal_seconds=duration_plan['nominal_seconds'], duration_plan=duration_plan, retimed=False,
                limitations=['Shot-level reference timing is prompt guidance, not hard per-frame masking.',
                             f'Native planned output is about {duration_plan["output_seconds"]:.3f} seconds; no retiming or end trimming.'])


def allocate(plan, references):
    """Filter the original objects without reading, copying or compressing media."""
    if references is None:
        references = {}
    obj(references, 'references input', ('images', 'videos', 'audios'))
    for kind in ('images', 'videos', 'audios'):
        if not isinstance(references.get(kind, []), (list, tuple)):
            fail(kind, 'material collector must supply a list or tuple')
    outputs = []
    for segment in plan['segments']:
        selected = {}
        for kind, indices in segment['slots'].items():
            original = references.get(kind, [])
            if any(i > len(original) for i in indices):
                fail(segment['id'], f'missing actual {kind} input for declared slots {indices}')
            if any(original[i-1] is None for i in indices):
                fail(segment['id'], f'declared {kind} slot is unconnected; slot numbers are not compacted')
            selected[kind] = [original[i-1] for i in indices]
        outputs.append(selected)
    return outputs


def collect_materials(images=None, videos=None, audios=None):
    """Two-segment project input capacity; preserve physical socket positions."""
    result = {}
    for kind, values, limit in (('images', images, 18), ('videos', videos, 6), ('audios', audios, 6)):
        if values is None:
            values = {}
        if not isinstance(values, dict):
            fail(kind, 'expected named autogrow inputs')
        numbered = {}
        for name, value in values.items():
            match = re.search(r'(\d+)$', name) if isinstance(name, str) else None
            if not match or not 1 <= int(match[1]) <= limit:
                fail(kind, 'invalid input socket number')
            index = int(match[1])
            if index in numbered:
                fail(kind, 'duplicate socket number')
            numbered[index] = value
        length = max(numbered, default=0)
        result[kind] = [numbered.get(i) for i in range(1, length+1)]
    if sum(v is not None for values in result.values() for v in values) > 24:
        fail('materials', 'a two-segment project supports at most 24 supplied references')
    return result
