// Pure editor helpers, also exercised by Node tests without loading ComfyUI.
export const groups = [
  {key:'images', widget:'image_count', label:'图片', type:'IMAGE', max:18},
  {key:'videos', widget:'video_count', label:'视频', type:'VIDEO', max:6},
  {key:'audios', widget:'audio_count', label:'音频', type:'AUDIO', max:6}
];

export function syncReferenceInputs(node) {
  if (node.h3V2Syncing) return;
  node.h3V2Syncing=true;
  try {
    // The sliders, rather than independent trailing-socket callbacks, own visibility.
    if (node.comfyDynamic) node.comfyDynamic.autogrow={};
    for (const group of groups) {
      const widget=node.widgets?.find(w=>w.name===group.widget);
      if (!widget) continue;
      const prefix=group.key+'.';
      const connected=(node.inputs || []).filter(s=>s.name.startsWith(prefix)&&s.link!=null);
      const highest=connected.reduce((n,s)=>Math.max(n,Number(s.name.match(/\d+$/)?.[0]||0)),0);
      const requested=Number(widget.value);
      const count=Math.max(highest,Math.min(group.max,Math.max(0,Math.round(Number.isFinite(requested)?requested:0))));
      widget.value=count;
      for (let i=node.inputs.length-1;i>=0;i--) {
        const slot=node.inputs[i];
        if (slot.name.startsWith(prefix)&&Number(slot.name.match(/\d+$/)?.[0])>count&&slot.link==null) {
          node.removeInput(i);
        }
      }
      for (let i=1;i<=count;i++) {
        const name=prefix+group.label+i;
        if (!node.inputs.some(s=>s.name===name)) node.addInput(name,group.type,{label:group.label+i,shape:7});
      }
    }
    const rank=slot=>{
      const group=groups.findIndex(g=>slot.name.startsWith(g.key+'.'));
      // 32 separates the three media groups even when there are 18 images.
      return group<0?-1:group*32+Number(slot.name.match(/\d+$/)?.[0]||0);
    };
    node.inputs.sort((a,b)=>rank(a)-rank(b));
    node.inputs.forEach((slot,index)=>{
      if(slot.link==null) return;
      const links=node.graph?.links;
      const link=typeof links?.get==='function'?links.get(slot.link):links?.[slot.link];
      if(link) link.target_slot=index;
    });
    const size=node.computeSize();
    node.setSize([Math.max(390,size[0]),Math.max(220,size[1])]);
    node.setDirtyCanvas(true,true);
  } finally {node.h3V2Syncing=false;}
}

export function configureReferenceInputs(node) {
  for (const group of groups) {
    const widget=node.widgets?.find(w=>w.name===group.widget);
    if (!widget||widget.h3V2CountHook) continue;
    widget.h3V2CountHook=true;
    widget.label=group.label+'输入数量';
    const original=widget.callback;
    widget.callback=function(...args) {
      original?.apply(this,args);
      syncReferenceInputs(node);
    };
  }
  syncReferenceInputs(node);
}
