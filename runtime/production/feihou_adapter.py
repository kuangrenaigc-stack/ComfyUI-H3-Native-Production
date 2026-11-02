"""Isolated FeiHou mixed-weight adapter; uses native MM operators and TP2.

No weight quantization/dequantization or sampling/collective implementation here.
The only heterogeneous offload projection in this checkpoint is fc2.
"""
from functools import lru_cache
import atexit
from collections import Counter
import json
import os
from pathlib import Path

import torch
from loguru import logger

_INSTALLED = False
_AUDIT = Counter()


@lru_cache(maxsize=4)
def precision_map(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get('passed') or not manifest.get('no_weight_dtype_changes'):
        raise ValueError('A verified lossless alignment manifest is required')
    mapping = {}
    for name, record in manifest['tensors'].items():
        if name.startswith('transformer_blocks.') and name.endswith('.weight'):
            dtype = record['spec']['dtype']
            if dtype not in ('I8', 'BF16'):
                raise ValueError(f'Unsupported mixed checkpoint dtype: {name} {dtype}')
            mapping[name] = 'int8-convrot' if dtype == 'I8' else 'Default'
    if len(mapping) != 550:
        raise ValueError(f'Expected 550 main-block weight tensors; found {len(mapping)}')
    # Restrict this adapter to the exact verified precision layout.
    bf16_fc2 = [i for i in range(50) if mapping[f'transformer_blocks.{i}.ff.net.2.weight'] == 'Default']
    if bf16_fc2 != [29, 39, 44, 45, 49]:
        raise ValueError(f'Unexpected heterogeneous fc2 layout: {bf16_fc2}')
    return mapping


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from lightx2v.models.networks.minimax_h3.weights import transformer_weights as weights
    from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
    from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
    from lightx2v.common.ops.mm.mm_weight import MMWeightWint8ConvRot
    original_linear = weights._linear
    original_block = weights.MiniMaxH3TransformerBlockWeights
    original_init = MiniMaxH3Model.__init__
    original_apply = MMWeightWint8ConvRot.apply

    def convrot_apply(self, value):
        if self.weight.dtype != torch.int8 or self.weight_scale.dtype != torch.float32:
            raise TypeError('ConvRot must retain original INT8 weight and FP32 scales')
        _AUDIT['int8_convrot_calls'] += 1
        return original_apply(self, value)

    def save_audit():
        folder = os.environ.get('H3_BENCH_JOB')
        if folder:
            rank = int(os.environ.get('RANK', 0))
            (Path(folder)/f'feihou-mixed-rank{rank}.json').write_text(json.dumps(dict(
                rank=rank, native_tp_unchanged=True, weight_requantization=False,
                **_AUDIT), indent=2))

    def linear(config, name, **kwargs):
        manifest = config.get('feihou_precision_manifest')
        if not manifest:
            return original_linear(config, name, **kwargs)
        scheme = precision_map(manifest)[name + '.weight']
        return original_linear(dict(config, dit_quant_scheme=scheme), name, **kwargs)

    class MixedBlock(original_block):
        def __init__(self, index, config, create_cuda_buffer=False):
            super().__init__(index, config, create_cuda_buffer)
            self._fc2_variants = None
            if config.get('feihou_precision_manifest') and create_cuda_buffer:
                active = self.ff.out_proj
                active_scheme = precision_map(config['feihou_precision_manifest'])[
                    f'transformer_blocks.{index}.ff.net.2.weight']
                self._fc2_variants = {active_scheme: active}
                # Read only the alternate projection into a second, correctly
                # typed buffer. Both variants survive slot swaps; neither is cast.
                for scheme, sample in (('Default', 29), ('int8-convrot', 0)):
                    if scheme not in self._fc2_variants:
                        self._fc2_variants[scheme] = linear(
                            config, f'transformer_blocks.{sample}.ff.net.2',
                            create_cuda_buffer=True, tp_split='row')

        def _inactive(self):
            return [] if self._fc2_variants is None else [
                value for value in self._fc2_variants.values() if value is not self.ff.out_proj]

        def load(self, source):
            for value in self._inactive():
                value.load(source)
            return super().load(source)

        def register_lora(self, source, strength):
            for value in self._inactive():
                value.register_lora(source, strength)
            return super().register_lora(source, strength)

        def update_lora(self, source, strength):
            for value in self._inactive():
                value.update_lora(source, strength)
            return super().update_lora(source, strength)

        def remove_lora(self):
            for value in self._inactive():
                value.remove_lora()
            return super().remove_lora()

        def load_state_dict(self, source, block_index, adapter_block_index=None):
            if self._fc2_variants is not None:
                key = f'transformer_blocks.{block_index}.ff.net.2.weight'
                dtype = source[key].dtype
                if dtype not in (torch.int8, torch.bfloat16):
                    raise TypeError(f'Unexpected fc2 storage dtype for {key}: {dtype}')
                scheme = 'int8-convrot' if dtype == torch.int8 else 'Default'
                self.ff.add_module('out_proj', self._fc2_variants[scheme])
                op = unwrap_tp_weight(self.ff.out_proj)
                if op.weight_cuda_buffer.dtype != dtype:
                    raise TypeError('Refusing an implicit offload-buffer dtype conversion')
                _AUDIT['fc2_buffer_' + scheme] += 1
            return super().load_state_dict(source, block_index, adapter_block_index)

    def model_init(self, model_path, config, *args, **kwargs):
        if config.get('feihou_precision_manifest'):
            precision_map(config['feihou_precision_manifest'])
            if not config.get('dit_quantized') or config.get('dit_quant_scheme') != 'int8-convrot':
                raise ValueError('FeiHou mixed weights require the native INT8 ConvRot path')
            if config.get('lora_configs') and not config.get('lora_dynamic_apply'):
                raise ValueError('Never merge a LoRA into preserved quantized weight bytes')
            if config.get('cpu_offload_layout') != 'per_tensor' or config.get('offload_granularity') != 'block':
                raise ValueError('Mixed weights currently require the existing per-tensor block offload')
            logger.info('FEIHOU_MIXED: original BF16/INT8 precision; native TP and kernels; typed fc2 buffers')
        return original_init(self, model_path, config, *args, **kwargs)

    weights._linear = linear
    weights.MiniMaxH3TransformerBlockWeights = MixedBlock
    MiniMaxH3Model.__init__ = model_init
    MMWeightWint8ConvRot.apply = convrot_apply
    atexit.register(save_audit)
    _INSTALLED = True
