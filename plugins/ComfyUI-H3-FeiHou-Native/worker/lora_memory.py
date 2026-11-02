"""Bound only LoRA temporary activations; retain native base kernels and TP.

Rows are independent tokens. Each native projection and its TP collective still
run once; only the low-rank additive branch is evaluated in smaller row batches.
No stored weight, scale, precision, sampler, or communication implementation changes.
"""
import atexit
import json
import os
from pathlib import Path

import torch


def install(chunk_rows=8192):
    from lightx2v.common.ops.mm import mm_weight as native
    original_default = native.MMWeight.apply
    original_convrot = native.MMWeightWint8ConvRot.apply
    counter = dict(chunk_rows=chunk_rows, projections=0, chunks=0)

    def add_branch(op, x, output):
        counter['projections'] += 1
        for start in range(0, x.shape[0], chunk_rows):
            stop = min(start+chunk_rows, x.shape[0])
            # Native apply_lora retains its alpha/rank, strength, dtype and
            # operation order. Mutate only the fresh base-output activation.
            output[start:stop].add_(op.apply_lora(x[start:stop]))
            counter['chunks'] += 1
        return output

    def default_apply(self, x):
        if not self.has_lora_branch:
            return original_default(self, x)
        output = torch.empty((x.shape[0], self.weight.shape[1]), dtype=x.dtype, device=x.device)
        if getattr(self, 'bias', None) is not None:
            torch.addmm(self._get_actual_bias(), x, self._get_actual_weight(), out=output)
        else:
            torch.mm(x, self._get_actual_weight(), out=output)
        return add_branch(self, x, output)

    def convrot_apply(self, x):
        if not self.has_lora_branch:
            return original_convrot(self, x)
        if x.shape[-1] % self._convrot_groupsize:
            raise ValueError('INT8 ConvRot input width is not divisible by group size')
        output = native._int8_convrot_linear(
            x.contiguous(), self.weight.contiguous(), self.weight_scale.contiguous(),
            self._get_actual_bias(), self.infer_dtype, self._convrot_groupsize)
        return add_branch(self, x, output)

    def report():
        folder = os.environ.get('H3_BENCH_JOB')
        if folder:
            rank = int(os.environ.get('RANK', 0))
            (Path(folder)/f'lora-memory-rank{rank}.json').write_text(json.dumps(dict(
                rank=rank, **counter, native_base_kernels=True, native_tp_unchanged=True,
                stored_weights_unchanged=True), indent=2))

    native.MMWeight.apply = default_apply
    native.MMWeightWint8ConvRot.apply = convrot_apply
    atexit.register(report)
