"""Original-object collection with explicit slider-controlled socket visibility."""
from comfy_api.latest import io as IO
from .compiler import collect_materials, ScriptError


class H3UnifiedMaterials(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='H3UnifiedMaterialsV2', display_name='H3 V2 · 统一参考素材', category='H3 / 统一创作 V2',
            inputs=[
                IO.Int.Input('image_count', display_name='图片输入数量', optional=True,
                    default=3, min=0, max=18, step=1, display_mode=IO.NumberDisplay.slider,
                    tooltip='显示0–18个图片接口。包含故事版；已连接的接口不会被隐藏或删除。'),
                IO.Int.Input('video_count', display_name='视频输入数量', optional=True,
                    default=0, min=0, max=6, step=1, display_mode=IO.NumberDisplay.slider,
                    tooltip='显示0–6个视频接口；每段最多选3段。视频音轨不作为独立音频输入。'),
                IO.Int.Input('audio_count', display_name='音频输入数量', optional=True,
                    default=0, min=0, max=6, step=1, display_mode=IO.NumberDisplay.slider,
                    tooltip='显示0–6个音频接口；每段最多选3段。全项目实际素材总数最多24份。'),
                IO.Autogrow.Input('images', display_name='参考图片（含按段故事版）', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Image.Input('image'), [f'图片{i}' for i in range(1,19)], min=0)),
                IO.Autogrow.Input('videos', display_name='参考视频', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Video.Input('video'), [f'视频{i}' for i in range(1,7)], min=0)),
                IO.Autogrow.Input('audios', display_name='参考音频', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Audio.Input('audio'), [f'音频{i}' for i in range(1,7)], min=0)),
            ], outputs=[IO.Custom('H3_REFERENCES').Output(display_name='统一参考素材')])

    @classmethod
    def execute(cls, images=None, videos=None, audios=None,
                image_count=3, video_count=0, audio_count=0):
        for name,value,maximum in [('image_count',image_count,18),
                                   ('video_count',video_count,6),('audio_count',audio_count,6)]:
            if type(value) is not int or not 0 <= value <= maximum:
                raise ScriptError(f'{name}: expected an integer from 0 to {maximum}')
        # Counts control the editor only. Never truncate or compact material objects:
        # old canvases and connected high-numbered sockets must keep their identities.
        return IO.NodeOutput(collect_materials(images, videos, audios))
