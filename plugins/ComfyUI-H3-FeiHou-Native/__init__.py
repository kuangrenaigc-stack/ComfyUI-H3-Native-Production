"""ComfyUI interface to the existing native FeiHou TP2 complete workflow."""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import wave

import numpy as np
from PIL import Image, ImageOps
import torch
import folder_paths
import comfy.model_management
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ui as UI
from comfy_api.latest import io as IO

ROOT = Path('/opt/h3/runtime')
WORKER = Path(__file__).resolve().parent / 'worker'
JOBS = ROOT / 'jobs/comfyui-complete'
VIDEOS = Path('/opt/h3/videos')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 * 1024**2):
            h.update(data)
    return h.hexdigest()


class H3OriginalImage:
    @classmethod
    def INPUT_TYPES(cls):
        root = Path(folder_paths.get_input_directory())
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        return {'required': {'image': (files, {'image_upload': True})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('原圖',)
    FUNCTION = 'load_image'
    CATEGORY = 'LightX2V / 目前完整工作流'

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        path = Path(folder_paths.get_annotated_filepath(image))
        return True if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS else '請選擇有效的原圖。'

    @classmethod
    def IS_CHANGED(cls, image):
        return sha(folder_paths.get_annotated_filepath(image))

    def load_image(self, image):
        path = Path(folder_paths.get_annotated_filepath(image)).resolve(strict=True)
        # The tensor is for ComfyUI connections only. Inference reads original bytes.
        with Image.open(path) as source:
            preview = np.array(ImageOps.exif_transpose(source).convert('RGB'), copy=True)
        tensor = torch.from_numpy(preview).float().div_(255).unsqueeze(0)
        tensor._h3_original_path = str(path)
        tensor._h3_original_sha256 = sha(path)
        return (tensor,)


class H3NativeComplete:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'references': ('H3_REFERENCES',),
            'prompt': ('STRING', {'multiline': True, 'default': '', 'tooltip': '完整六段提示詞；圖片編號依參考素材節點順序。'}),
            'width': ('INT', {'default': 1280, 'min': 64, 'max': 1920, 'step': 64}),
            'height': ('INT', {'default': 768, 'min': 64, 'max': 1920, 'step': 64}),
            'duration_seconds': ('INT', {'default': 15, 'min': 5, 'max': 15, 'tooltip': '保留模型原生時長，不裁剪或重定時。'}),
            'seed': ('INT', {'default': 2026102508, 'min': 0, 'max': 2147483647, 'control_after_generate': True}),
            'timeout_minutes': ('INT', {'default': 120, 'min': 5, 'max': 360})}}

    RETURN_TYPES = ('VIDEO', 'STRING')
    RETURN_NAMES = ('原生成片', '伺服器成片路徑')
    FUNCTION = 'generate'
    CATEGORY = 'LightX2V / 目前完整工作流'
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, prompt=None, width=None, height=None, duration_seconds=None, seed=None, timeout_minutes=None):
        from .continuation.input_validation import validate_values
        values=dict(first_prompt=prompt,width=width,height=height,
                    segment_seconds=duration_seconds,seed=seed,timeout_minutes=timeout_minutes)
        result=validate_values(values)
        if result is not True:return result
        for n in (width,height):
            if n is not None and (type(n) is not int or n<64 or n%64):
                return '2倍潛空間放大要求最終寬高為64的倍數。'
        if width is not None and height is not None:
            from .continuation.learned_bridge import draft_geometry
            try:draft_geometry(width,height)
            except ValueError as exc:return str(exc)
        return True

    SUPERVISOR=WORKER/'supervise_complete.py'

    def _run(self, job, timeout_minutes):
        from .continuation import H3NativeContinuation
        return H3NativeContinuation._run(self,job,timeout_minutes)

    def generate(self, references, prompt, width=1280, height=768,
                 duration_seconds=15, seed=2026102508, timeout_minutes=120):
        if any(v is None for v in (references,prompt,width,height,duration_seconds,seed,timeout_minutes)):
            raise ValueError('生成輸入尚未解析完整。')
        valid = self.VALIDATE_INPUTS(prompt, width, height, duration_seconds, seed, timeout_minutes)
        if valid is not True:
            raise ValueError(valid)
        JOBS.mkdir(parents=True, exist_ok=True)
        job = JOBS/uuid.uuid4().hex
        job.mkdir()
        images, videos, audios, receipts = [], [], [], []
        for batch in references.get('images', []):
            path = getattr(batch, '_h3_original_path', None)
            expected = getattr(batch, '_h3_original_sha256', None)
            if path is None or expected is None:
                raise ValueError('請使用「H3 原圖載入」連接原來的參考素材節點，以保留原始圖片檔案。')
            source = Path(path).resolve(strict=True)
            if sha(source) != expected:
                raise ValueError('原圖已變更，請重新載入圖片。')
            dest = job/f'picture{len(images)+1}{source.suffix.lower()}'
            shutil.copyfile(source, dest)
            if sha(dest) != expected:
                raise ValueError('原圖複製校驗失敗。')
            images.append(str(dest))
            receipts.append(dict(source=str(source), file=str(dest), sha256=expected,
                                 bytes=dest.stat().st_size))
        for index, video in enumerate(references.get('videos', []), 1):
            dest = job/f'video{index}.mp4'
            video.save_to(str(dest))
            videos.append(str(dest))
        for index, audio in enumerate(references.get('audios', []), 1):
            samples = audio['waveform'].detach().cpu()
            if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[1] not in (1, 2):
                raise ValueError('每個音頻接口須提供單段單聲道或立體聲音頻。')
            dest = job/f'audio{index}.wav'
            pcm = (samples[0].clamp(-1, 1).T.numpy()*32767).round().astype('<i2')
            with wave.open(str(dest), 'wb') as wav:
                wav.setnchannels(pcm.shape[1]); wav.setsampwidth(2)
                wav.setframerate(int(audio['sample_rate'])); wav.writeframes(pcm.tobytes())
            audios.append(str(dest))
        if not images and not videos:
            raise ValueError('請提供至少一張原圖或一段參考視頻。')
        if len(images) > 9 or len(videos) > 3 or len(audios) > 3 or len(images)+len(videos)+len(audios) > 12:
            raise ValueError('參考素材超出數量限制。')
        config = json.loads((ROOT/'configs/feihou-ref2va-native10plus5.json').read_text())
        draft_size = [height//2, width//2]
        frames = math.ceil((duration_seconds*24-5)/17)*17+5
        config.update(size=[height, width], num_frames=frames, h3_two_pass=True,
                      h3_draft_steps=10, h3_draft_size=draft_size,
                      h3_handoff='learned-clean-latent-2x',ffn_inplace_activation=True)
        request = dict(prompt=prompt, image_path=','.join(images), video_path=','.join(videos),
                       audio_path=','.join(audios), num_frames=frames, seed=seed)
        (job/'config.json').write_text(json.dumps(config, indent=2))
        (job/'request-overrides.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
        (job/'input-receipt.json').write_text(json.dumps(dict(original_images=receipts,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), stage='complete'),
            ensure_ascii=False, indent=2))
        self._run(job, timeout_minutes)
        result = json.loads((job/'stage-result.json').read_text())
        if result.get('returncode') != 0 or result.get('stage') != 'complete' or not result.get('published_video'):
            raise RuntimeError(f'沒有完整成片，請查看：{job}/stage-result.json')
        published = result['published_video']
        output = VIDEOS/published['filename']
        if not output.is_file() or sha(output) != published['sha256']:
            raise RuntimeError('成片校驗失敗。')
        preview = UI.PreviewVideo([UI.SavedResult(output.name, 'H3-成片', IO.FolderType.output)]).as_dict()
        return {'ui': preview, 'result': (VideoFromFile(str(output)), str(output))}


NODE_CLASS_MAPPINGS = {'H3OriginalImage': H3OriginalImage, 'H3FeiHouNativeComplete': H3NativeComplete}
NODE_DISPLAY_NAME_MAPPINGS = {'H3OriginalImage': 'H3 原圖載入（保留原檔）',
                            'H3FeiHouNativeComplete': 'H3 全能參考 · FeiHou 雙卡10＋5'}

# Validation only: never enqueue or execute a generation from this endpoint.
try:
    from server import PromptServer
    if getattr(PromptServer, 'instance', None) is not None:
        from aiohttp import web
        @PromptServer.instance.routes.post('/h3_native/validate')
        async def validate_native_workflow(request):
            import execution
            graph = await request.json()
            valid, error, outputs, errors = await execution.validate_prompt(uuid.uuid4().hex, graph, None)
            return web.json_response(dict(valid=valid, error=error, outputs=outputs, node_errors=errors))
except ImportError:
    pass

