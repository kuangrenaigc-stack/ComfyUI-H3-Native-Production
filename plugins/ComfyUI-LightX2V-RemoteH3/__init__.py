"""Reference collector only; no HTTP client or credentials."""
import re
from comfy_api.latest import IO
WEB_DIRECTORY="./web"

class H3References(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='LightX2VH3References', display_name='H3 多模态参考素材',
            category='LightX2V / 服务器 H3',
            inputs=[
                IO.Int.Input('image_count', display_name='图片输入数量', default=0, min=0, max=9, step=1, display_mode=IO.NumberDisplay.slider, tooltip='拖动滑块显示0–9个图片接口；设为0隐藏。已连接接口不会被自动删除。'),
                IO.Int.Input('video_count', display_name='视频输入数量', default=0, min=0, max=3, step=1, display_mode=IO.NumberDisplay.slider, tooltip='最多3段视频，合计不超过15秒；包含原音轨。'),
                IO.Int.Input('audio_count', display_name='音频输入数量', default=0, min=0, max=3, step=1, display_mode=IO.NumberDisplay.slider, tooltip='最多3段音频，视频音轨也占用音频名额。'),
                IO.Autogrow.Input('images', display_name='参考图片', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Image.Input('image'), [f'图片{i}' for i in range(1,10)], min=0)),
                IO.Autogrow.Input('videos', display_name='参考视频（含原音轨）', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Video.Input('video'), [f'视频{i}' for i in range(1,4)], min=0)),
                IO.Autogrow.Input('audios', display_name='参考音频', optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Audio.Input('audio'), [f'音频{i}' for i in range(1,4)], min=0)),
            ], outputs=[IO.Custom('H3_REFERENCES').Output(display_name='全能参考素材')])
    @classmethod
    def execute(cls, image_count=0, video_count=0, audio_count=0, images=None, videos=None, audios=None):
        def ordered(items):
            return [v for k,v in sorted((items or {}).items(), key=lambda pair:int(re.search(r'(\d+)$',pair[0]).group(1))) if v is not None]
        groups = {'images':ordered(images),'videos':ordered(videos),'audios':ordered(audios)}
        for items,count,limit in ((images,image_count,9),(videos,video_count,3),(audios,audio_count,3)):
            if type(count) is not int or not 0<=count<=limit: raise ValueError('参考接口数量超出范围。')
            if any(v is not None and int(re.search(r'(\d+)$',k).group(1))>count for k,v in (items or {}).items()):
                raise ValueError('已连接素材的编号超过输入数量，请增大对应滑块。')
        if sum(len(batch) for batch in groups['images'])>9:
            raise ValueError('参考图片最多9张，图片批次中的每张图片也计入总数。')
        if sum(len(batch) for batch in groups['images'])+len(groups['videos'])+len(groups['audios'])>12:
            raise ValueError('参考文件合计不能超过12个。')
        return IO.NodeOutput(groups)

NODE_CLASS_MAPPINGS={"LightX2VH3References":H3References}
