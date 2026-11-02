"""Independent continuation nodes; existing node packages/workflows stay untouched."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
import wave

import comfy.model_management
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ui as UI, io as IO
from .input_validation import validate_values
from .process_control import stop_supervisor
from .segment_inputs import archive_segments
from .learned_bridge import draft_geometry
from .duration_plan import validate_plan

ROOT = Path('/opt/h3/runtime')
WORKER = Path(__file__).resolve().parent/'worker'
VIDEOS = Path('/opt/h3/videos')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def archive_references(references, job):
    images, videos, audios, receipts = [], [], [], []
    for batch in references.get('images', []):
        source, expected = getattr(batch, '_h3_original_path', None), getattr(batch, '_h3_original_sha256', None)
        if not source or not expected:
            raise ValueError('請使用原有 H3 原圖載入節點，保留圖片原始檔案。')
        source = Path(source).resolve(strict=True)
        if sha(source) != expected:
            raise ValueError('原圖已變更，請重新載入。')
        dest = job/f'picture{len(images)+1}{source.suffix.lower()}'
        shutil.copyfile(source, dest)
        if sha(dest) != expected:
            raise ValueError('原圖複製校驗失敗。')
        images.append(str(dest))
        receipts.append(dict(source=str(source), path=str(dest), sha256=expected, bytes=dest.stat().st_size))
    for index, video in enumerate(references.get('videos', []), 1):
        dest = job/f'video{index}.mp4'
        video.save_to(str(dest))
        videos.append(str(dest))
    for index, audio in enumerate(references.get('audios', []), 1):
        samples = audio['waveform'].detach().cpu()
        if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[1] not in (1, 2):
            raise ValueError('音頻接口須為單段單聲道或立體聲。')
        dest = job/f'audio{index}.wav'
        pcm = (samples[0].clamp(-1, 1).T.numpy()*32767).round().astype('<i2')
        with wave.open(str(dest), 'wb') as wav:
            wav.setnchannels(pcm.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(int(audio['sample_rate']))
            wav.writeframes(pcm.tobytes())
        audios.append(str(dest))
    if not images and not videos:
        raise ValueError('至少需要一張原圖或一段參考視頻。')
    if len(images) > 9 or len(videos) > 3 or len(audios) > 3 or len(images)+len(videos)+len(audios) > 12:
        raise ValueError('參考素材超出原生限制。')
    return dict(image_path=','.join(images), video_path=','.join(videos), audio_path=','.join(audios)), receipts


class H3NativeContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'references': ('H3_REFERENCES',),
            'first_prompt': ('STRING', {'multiline': True, 'default': '', 'tooltip': '第一段完整提示詞；沿用原有參考編號。'}),
            'continuation_prompt': ('STRING', {'multiline': True, 'default': '', 'tooltip': '第二段完整提示詞。開頭約0.917秒是上一段尾部重疊上下文，之後再安排新動作及台詞。'}),
            'width': ('INT', {'default': 736, 'min': 32, 'max': 1920, 'step': 32}),
            'height': ('INT', {'default': 1280, 'min': 32, 'max': 1920, 'step': 32}),
            'segment_seconds': ('INT', {'default': 15, 'min': 5, 'max': 15, 'tooltip': '固定兩段；每段15秒的原生輸出合併後預計29.25秒，以實際幀數為準。'}),
            'seed': ('INT', {'default': 2026102601, 'min': 0, 'max': 2147483647, 'control_after_generate': True}),
            'timeout_minutes': ('INT', {'default': 240, 'min': 5, 'max': 360})}}

    RETURN_TYPES = ('VIDEO', 'STRING')
    RETURN_NAMES = ('接續成片', '伺服器成片路徑')
    FUNCTION = 'generate'
    CATEGORY = 'H3 / 獨立接續插件'
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Unresolved links arrive as None during ComfyUI graph validation.
        # ComfyUI still validates socket types; generate validates all values
        # again once upstream nodes have produced the actual inputs.
        return validate_values(kwargs)

    def _run(self, job, timeout_minutes):
        control = Path(f'/run/user/{os.getuid()}/h3-native')
        control.mkdir(parents=True, exist_ok=True)
        with (control/'control.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('其他服務正在切換，請稍後再試。') from exc
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
            if busy:
                raise RuntimeError('顯卡有其他任務；不會停止其他工作。')
            command = [str(ROOT/'.venv/bin/python'), '-u', str(getattr(self,'SUPERVISOR',WORKER/'supervise.py')), str(job)]
            with (job/'run.log').open('w') as log:
                process = subprocess.Popen(command,
                    env=dict(os.environ, H3_ROOT=str(ROOT), H3_CONTROL_LOCK_FD=str(lock.fileno())),
                    pass_fds=(lock.fileno(),),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic()+timeout_minutes*60
                    while True:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        try:
                            code = process.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() > deadline:
                                raise TimeoutError('超過等待上限，已停止本次生成任務。')
                    if code:
                        from .delivery_status import published_after_cleanup
                        if published_after_cleanup(job,VIDEOS):return
                        raise RuntimeError(f'生成失敗，紀錄：{job}/run.log')
                finally:
                    stop_supervisor(process)

    def generate(self, references, first_prompt, continuation_prompt, width=736, height=1280,
                 segment_seconds=15, seed=2026102601, timeout_minutes=240,
                 continuation_references=None, learned_2x=False, duration_plan=None):
        valid = validate_values(dict(first_prompt=first_prompt, continuation_prompt=continuation_prompt,
            width=width, height=height, segment_seconds=segment_seconds, seed=seed,
            timeout_minutes=timeout_minutes), complete=True)
        if valid is not True:
            raise ValueError(valid)
        if duration_plan is not None:
            duration_plan = validate_plan(duration_plan, segments=2)
            if segment_seconds != duration_plan['segment_seconds'][0]:
                raise ValueError('segment_seconds differs from the first duration-plan segment')
            if not learned_2x:
                raise ValueError('Variable segment duration requires the learned-latent workflow')
        # Imports stay inside execution; no native model or hooks in ComfyUI's process.
        import math
        job = ROOT/'jobs/comfyui-continuation'/uuid.uuid4().hex
        job.mkdir(parents=True)
        # Legacy node alone retains its explicit shared-reference behavior.
        # The new node requires both connections and never takes this fallback.
        second = references if continuation_references is None else continuation_references
        refs, receipt = archive_segments(job, (references, second), archive_references)
        config = json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
        frames = 5+17*math.ceil((segment_seconds*24-5)/17)
        frame_counts = duration_plan['segment_frames'] if duration_plan is not None else [frames, frames]
        scale = min(1., 480/min(width, height))
        config.update(size=[height, width], num_frames=frames, segment_seconds=segment_seconds,
                      h3_two_pass=True, h3_draft_steps=10,
                      h3_draft_size=[max(32, int(height*scale)//32*32), max(32, int(width*scale)//32*32)])
        if learned_2x:
            config.update(h3_draft_size=draft_geometry(width,height),
                          h3_handoff='learned-clean-latent-2x',ffn_inplace_activation=True,
                          deliver_exact_30_seconds=False)
        if duration_plan is not None:
            config.update(h3_duration_plan=duration_plan, num_frames=frame_counts[0])
        requests = [dict(refs[i], prompt=p, num_frames=frame_counts[i], seed=(seed+i) % 2147483648)
                    for i, p in enumerate((first_prompt, continuation_prompt))]
        (job/'config.json').write_text(json.dumps(config, indent=2))
        (job/'requests.json').write_text(json.dumps(requests, ensure_ascii=False, indent=2))
        (job/'input-receipt.json').write_text(json.dumps(dict(segment_inputs=receipt,
            final_latent_source_only=True, segments=2, final_size=config['size'],
            draft_size=config['h3_draft_size'],handoff=config.get('h3_handoff','RGB-Lanczos-VAE')),
            ensure_ascii=False, indent=2))
        self._run(job, timeout_minutes)
        result = json.loads((job/'series-result.json').read_text())
        published = result.get('published_video')
        if result.get('returncode') != 0 or not published:
            raise RuntimeError(f'未生成完整接續成片，請查看 {job}/series-result.json')
        output = VIDEOS/published['filename']
        if not output.is_file() or sha(output) != published['sha256']:
            raise RuntimeError('成片校驗失敗。')
        preview = UI.PreviewVideo([UI.SavedResult(output.name, 'H3-成片', IO.FolderType.output)]).as_dict()
        return {'ui': preview, 'result': (VideoFromFile(str(output)), str(output))}


class H3NativeSegmentInputs(H3NativeContinuation):
    CATEGORY = 'LightX2V / 目前完整工作流'

    @classmethod
    def INPUT_TYPES(cls):
        original = H3NativeContinuation.INPUT_TYPES()['required']
        original['width']=('INT',{'default':768,'min':64,'max':1920,'step':64,
            'tooltip':'最終成片寬度；一採寬度為一半，經2倍潛空間放大。'})
        original['height']=('INT',{'default':1280,'min':64,'max':1920,'step':64,
            'tooltip':'最終成片高度；一採高度為一半，經2倍潛空間放大。'})
        return {'required': {
            'first_references': ('H3_REFERENCES', {'tooltip': '第一段專用素材；不會傳給第二段。圖片從Picture 1編號。'}),
            'first_prompt': original['first_prompt'],
            'continuation_references': ('H3_REFERENCES', {'tooltip': '第二段專用素材；不會傳給第一段。圖片重新從Picture 1編號。'}),
            'continuation_prompt': original['continuation_prompt'],
            **{key: value for key, value in original.items()
               if key not in ('references','first_prompt','continuation_prompt')}
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        result=validate_values(kwargs)
        if result is not True:return result
        if kwargs.get('duration_plan') is not None:
            try:
                p=validate_plan(kwargs['duration_plan'], segments=2)
                if kwargs.get('segment_seconds') is not None and kwargs['segment_seconds'] != p['segment_seconds'][0]:
                    return 'segment_seconds differs from the first duration-plan segment'
            except ValueError as exc:return str(exc)
        for key in ('width','height'):
            if kwargs.get(key) is not None:
                n=kwargs[key]
                if type(n) is not int or n<64 or n%64:
                    return '2倍潛空間放大要求最終寬高為64的倍數。'
        if kwargs.get('width') is not None and kwargs.get('height') is not None:
            try:draft_geometry(kwargs['width'],kwargs['height'])
            except ValueError as exc:return str(exc)
        return True

    def generate(self, first_references, first_prompt, continuation_references, continuation_prompt,
                 width=768, height=1280, segment_seconds=15, seed=2026102601, timeout_minutes=240,
                 duration_plan=None):
        draft_geometry(width,height)
        if not isinstance(first_references, dict) or not isinstance(continuation_references, dict):
            raise ValueError('請分別連接第一段與第二段素材，不會自動共用。')
        return super().generate(first_references, first_prompt, continuation_prompt,
            width, height, segment_seconds, seed, timeout_minutes,
            continuation_references=continuation_references,learned_2x=True,
            duration_plan=duration_plan)


# Internal execution backend for the single-pass wrapper and unified V2.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
